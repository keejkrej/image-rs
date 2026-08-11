//! Sandboxed WebAssembly Component execution behind the operation registry seam.

mod dataset;

#[allow(dead_code, unused_variables)]
mod bindings {
    wasmtime::component::bindgen!({
        path: "wit",
        world: "image-operation-plugin",
        imports: { default: trappable },
    });
}

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Map, Number, Value};
use thiserror::Error;
use wasmparser::{Encoding, Parser, Payload, Validator};
use wasmtime::component::{Component, HasSelf, Linker};
use wasmtime::{Config, Engine, Store, StoreLimits, StoreLimitsBuilder};

use self::bindings::exports::image_rs::plugin::image_operation as wit_operation;
use self::bindings::image_rs::plugin::{host as wit_host, types as wit_types};
use self::dataset::{
    DatasetAdapter, DatasetAdapterError, plugin_position_to_wit, wit_pixel_type_to_plugin,
    wit_position_to_plugin,
};
use super::PluginCatalog;
use crate::commands::{
    MeasurementTable, OpOutput, OpSchema, Operation, OperationRegistry, OpsError,
};
use crate::model::DatasetF32;
use crate::plugins::contract::{
    MAX_PLUGIN_MEASUREMENT_ROWS, MAX_PLUGIN_MEASUREMENTS_PER_ROW, PluginContractError,
    PluginOperationCapabilities, PluginPayloadBudget, PluginPixelType, PluginPlaneLayout,
    PluginPlanePosition, PluginPlaneScope, PluginProgress, validate_name,
};

// One memory can simultaneously hold a maximum-size input and replacement plane plus canonical
// ABI/runtime overhead. Keeping a single bounded memory also limits host allocations during lift.
const STORE_MEMORY_BYTES: usize = 160 * 1024 * 1024;
const STORE_TABLE_ELEMENTS: usize = 100_000;
const STORE_INSTANCES: usize = 64;
const STORE_TABLES: usize = 16;
const STORE_MEMORIES: usize = 1;
const INVOCATION_FUEL: u64 = 100_000_000;
const INVOCATION_TIMEOUT: Duration = Duration::from_secs(5);
const CAPABILITY_FUEL: u64 = 10_000_000;
const CAPABILITY_TIMEOUT: Duration = Duration::from_millis(500);
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_REGISTRATION_PACKAGES: usize = 32;
const MAX_REGISTRATION_OPERATIONS: usize = 256;
const MAX_REGISTRATION_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;
// Discovery accepts a larger artifact so future non-operation worlds can share the package
// format. Native compilation has a deliberately narrower admission policy because
// `Component::new` is synchronous and cannot be interrupted with fuel or epochs.
const MAX_COMPILABLE_COMPONENT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_COMPONENT_EMBEDDING_DEPTH: u64 = 8;
const MAX_EMBEDDED_CORE_MODULES: u64 = 32;
const MAX_EMBEDDED_COMPONENTS: u64 = 16;
const MAX_EMBEDDED_BINARY_BYTES: u64 = 32 * 1024 * 1024;
const MAX_WASM_SECTION_ITEMS: u64 = 100_000;
const MAX_DEFINED_CORE_FUNCTIONS: u64 = 10_000;
const MAX_CORE_CODE_BYTES: u64 = 12 * 1024 * 1024;
const MAX_CORE_FUNCTION_BODY_BYTES: u64 = 512 * 1024;
const MAX_CORE_OPERATORS: u64 = 1_000_000;
const MAX_CORE_FUNCTION_OPERATORS: u64 = 50_000;
const MAX_CORE_LOCALS: u64 = 1_000_000;
const MAX_CORE_FUNCTION_LOCALS: u64 = 100_000;
const MAX_CUSTOM_SECTIONS: u64 = 64;
const MAX_CUSTOM_SECTION_BYTES: u64 = 1024 * 1024;
const MAX_CUSTOM_SECTION_BYTES_TOTAL: u64 = 2 * 1024 * 1024;
const EPOCH_TICK: Duration = Duration::from_millis(10);
const MAX_ERROR_MESSAGE_BYTES: usize = 2048;

/// Failure while compiling and registering discovered plugin operations.
#[derive(Debug, Error)]
pub enum PluginRuntimeError {
    #[error("failed to initialize the plugin runtime: {0}")]
    RuntimeSetup(String),

    #[error("failed to read component for plugin `{plugin_id}` at `{path}`: {message}")]
    ArtifactRead {
        plugin_id: String,
        path: PathBuf,
        message: String,
    },

    #[error(
        "component for plugin `{plugin_id}` at `{path}` contains {actual} bytes, above the {limit}-byte limit"
    )]
    ArtifactLimit {
        plugin_id: String,
        path: PathBuf,
        actual: u64,
        limit: u64,
    },

    #[error(
        "component for plugin `{plugin_id}` was rejected by compile-admission policy: {message}"
    )]
    Admission { plugin_id: String, message: String },

    #[error("failed to compile component for plugin `{plugin_id}`: {message}")]
    Compile { plugin_id: String, message: String },

    #[error(
        "component for plugin `{plugin_id}` does not implement the image-operation world: {message}"
    )]
    WorldMismatch { plugin_id: String, message: String },

    #[error("plugin operation `{operation_id}` has unusable capabilities: {message}")]
    Capabilities {
        operation_id: String,
        message: String,
    },

    #[error("could not register plugin operation `{operation_id}`: {message}")]
    Registry {
        operation_id: String,
        message: String,
    },

    #[error("plugin registration exceeds the host policy: {0}")]
    RegistrationLimit(String),
}

struct RuntimeCore {
    engine: Arc<Engine>,
}

impl RuntimeCore {
    fn new() -> Result<Arc<Self>, PluginRuntimeError> {
        let mut config = Config::new();
        config
            .wasm_component_model(true)
            .consume_fuel(true)
            .epoch_interruption(true)
            .max_wasm_stack(512 * 1024);
        let engine = Arc::new(
            Engine::new(&config)
                .map_err(|error| PluginRuntimeError::RuntimeSetup(error.to_string()))?,
        );
        start_epoch_ticker(&engine);
        Ok(Arc::new(Self { engine }))
    }

    fn compile(
        &self,
        plugin_id: &str,
        path: &Path,
    ) -> Result<bindings::ImageOperationPluginPre<HostState>, PluginRuntimeError> {
        let bytes = read_component(plugin_id, path)?;
        admit_component(&bytes).map_err(|message| PluginRuntimeError::Admission {
            plugin_id: plugin_id.to_string(),
            message,
        })?;
        let component =
            Component::new(&self.engine, bytes).map_err(|error| PluginRuntimeError::Compile {
                plugin_id: plugin_id.to_string(),
                message: concise_error(&error.to_string()),
            })?;
        let mut linker = Linker::<HostState>::new(&self.engine);
        bindings::ImageOperationPlugin::add_to_linker::<_, HasSelf<_>>(&mut linker, |state| state)
            .map_err(|error| PluginRuntimeError::RuntimeSetup(concise_error(&error.to_string())))?;
        let pre = linker.instantiate_pre(&component).map_err(|error| {
            PluginRuntimeError::WorldMismatch {
                plugin_id: plugin_id.to_string(),
                message: concise_error(&error.to_string()),
            }
        })?;
        bindings::ImageOperationPluginPre::new(pre).map_err(|error| {
            PluginRuntimeError::WorldMismatch {
                plugin_id: plugin_id.to_string(),
                message: concise_error(&error.to_string()),
            }
        })
    }

    fn store(&self, timeout: Duration, fuel: u64) -> Result<Store<HostState>, String> {
        let mut store = Store::new(&self.engine, HostState::new(timeout));
        store.limiter(|state| &mut state.limits);
        store
            .set_fuel(fuel)
            .map_err(|error| concise_error(&error.to_string()))?;
        let ticks = timeout
            .as_nanos()
            .div_ceil(EPOCH_TICK.as_nanos())
            .max(1)
            .min(u128::from(u64::MAX)) as u64;
        store.set_epoch_deadline(ticks);
        store.epoch_deadline_trap();
        Ok(store)
    }
}

fn start_epoch_ticker(engine: &Arc<Engine>) {
    let engine = Arc::downgrade(engine);
    thread::spawn(move || {
        loop {
            thread::sleep(EPOCH_TICK);
            let Some(engine) = engine.upgrade() else {
                break;
            };
            engine.increment_epoch();
        }
    });
}

struct HostState {
    limits: StoreLimits,
    progress: PluginProgress,
    output_budget: PluginPayloadBudget,
    deadline: Instant,
    contract_violation: Option<String>,
}

impl HostState {
    fn new(timeout: Duration) -> Self {
        Self {
            limits: StoreLimitsBuilder::new()
                .memory_size(STORE_MEMORY_BYTES)
                .table_elements(STORE_TABLE_ELEMENTS)
                .instances(STORE_INSTANCES)
                .tables(STORE_TABLES)
                .memories(STORE_MEMORIES)
                .trap_on_grow_failure(true)
                .build(),
            progress: PluginProgress::default(),
            output_budget: PluginPayloadBudget::new(),
            deadline: Instant::now() + timeout,
            contract_violation: None,
        }
    }

    fn cancelled(&self) -> bool {
        Instant::now() >= self.deadline
    }

    fn reject(&mut self, message: String) -> wasmtime::Error {
        if self.contract_violation.is_none() {
            self.contract_violation = Some(message.clone());
        }
        wasmtime::Error::msg(message)
    }
}

impl wit_types::Host for HostState {}

impl wit_host::Host for HostState {
    fn report_progress(&mut self, update: wit_types::ProgressUpdate) -> wasmtime::Result<()> {
        if self.cancelled() {
            return Err(self.reject("plugin invocation deadline exceeded".to_string()));
        }
        if let Err(error) = self.progress.update(
            update.completed,
            update.total,
            update.message.as_deref(),
            &mut self.output_budget,
        ) {
            return Err(self.reject(error.to_string()));
        }
        Ok(())
    }

    fn is_cancelled(&mut self) -> wasmtime::Result<bool> {
        Ok(self.cancelled())
    }
}

/// Register every discovered image operation as an ordinary application operation.
///
/// Registration is atomic: artifacts are compiled, linked, capability-checked, and added to a
/// cloned registry before the supplied registry is changed. The current operation interface has
/// no active-plane or ROI context, so this adapter accepts only operations that support the
/// all-planes scope without requiring an ROI.
impl PluginCatalog {
    pub fn register_operations(
        &self,
        registry: &mut OperationRegistry,
    ) -> Result<usize, PluginRuntimeError> {
        if self.operations.is_empty() {
            return Ok(0);
        }

        // Reject known collisions before compiling or entering any untrusted component code.
        let existing = registry
            .list()
            .into_iter()
            .map(|schema| schema.name)
            .collect::<BTreeSet<_>>();
        if let Some(operation) = self
            .operations
            .values()
            .find(|operation| existing.contains(&operation.id))
        {
            return Err(PluginRuntimeError::Registry {
                operation_id: operation.id.clone(),
                message: OpsError::DuplicateOperation(operation.id.clone()).to_string(),
            });
        }

        preflight_registration(self)?;

        let runtime = RuntimeCore::new()?;
        let registration_deadline = Instant::now() + REGISTRATION_TIMEOUT;
        let mut staged_registry = registry.clone();
        let mut registered = 0;

        for (plugin_id, package) in &self.packages {
            let operations = self
                .operations
                .values()
                .filter(|operation| operation.plugin_id == *plugin_id)
                .collect::<Vec<_>>();
            if operations.is_empty() {
                continue;
            }

            let pre = runtime.compile(plugin_id, &package.component_path)?;
            if Instant::now() >= registration_deadline {
                return Err(PluginRuntimeError::RegistrationLimit(
                    "the 15-second aggregate deadline expired during component compilation"
                        .to_string(),
                ));
            }
            for operation in operations {
                let remaining = registration_deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(PluginRuntimeError::RegistrationLimit(
                        "the 15-second aggregate deadline expired during capability checks"
                            .to_string(),
                    ));
                }
                let capabilities = query_capabilities(
                    &runtime,
                    &pre,
                    &operation.id,
                    &operation.export,
                    remaining.min(CAPABILITY_TIMEOUT),
                )?;
                ensure_registry_compatible(&operation.id, &capabilities)?;
                let adapter = WasmOperation {
                    id: operation.id.clone(),
                    entrypoint: operation.export.clone(),
                    schema: operation.schema.clone(),
                    capabilities,
                    runtime: runtime.clone(),
                    pre: pre.clone(),
                };
                staged_registry
                    .register(Arc::new(adapter))
                    .map_err(|error| PluginRuntimeError::Registry {
                        operation_id: operation.id.clone(),
                        message: error.to_string(),
                    })?;
                registered += 1;
            }
        }

        *registry = staged_registry;
        Ok(registered)
    }
}

struct WasmOperation {
    id: String,
    entrypoint: String,
    schema: OpSchema,
    capabilities: PluginOperationCapabilities,
    runtime: Arc<RuntimeCore>,
    pre: bindings::ImageOperationPluginPre<HostState>,
}

impl Operation for WasmOperation {
    fn name(&self) -> &str {
        &self.id
    }

    fn schema(&self) -> OpSchema {
        self.schema.clone()
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> crate::commands::Result<OpOutput> {
        self.execute_inner(dataset, params)
    }
}

impl WasmOperation {
    fn execute_inner(
        &self,
        dataset: &DatasetF32,
        params: &Value,
    ) -> crate::commands::Result<OpOutput> {
        let parameters_json = validate_parameters(&self.schema, params)?;
        let mut input_budget = PluginPayloadBudget::new();
        input_budget
            .validate_text("operation id", &self.id)
            .map_err(|error| invalid_parameters(error.to_string()))?;
        input_budget
            .validate_text("operation entrypoint", &self.entrypoint)
            .map_err(|error| invalid_parameters(error.to_string()))?;
        input_budget
            .validate_json_object("parameters", &parameters_json)
            .map_err(|error| invalid_parameters(error.to_string()))?;
        let adapter = DatasetAdapter::with_input_budget(dataset, &mut input_budget)
            .map_err(dataset_input_error)?;

        self.capabilities
            .validate_invocation(
                adapter.layout().pixel_type(),
                PluginPlaneScope::AllPlanes,
                false,
            )
            .map_err(|error| OpsError::UnsupportedLayout(error.to_string()))?;

        let mut store = self
            .runtime
            .store(INVOCATION_TIMEOUT, INVOCATION_FUEL)
            .map_err(|message| self.execution_error(message))?;
        let instance = self
            .pre
            .instantiate(&mut store)
            .map_err(|error| self.trap_error(&store, error))?;
        let guest = instance.image_rs_plugin_image_operation();

        let invocation_capabilities = guest
            .call_capabilities(&mut store, &self.entrypoint)
            .map_err(|error| self.trap_error(&store, error))?
            .map_err(|error| self.guest_error(&mut store, error))?;
        let invocation_capabilities = convert_capabilities(invocation_capabilities)
            .map_err(|error| self.contract_error(error.to_string()))?;
        if invocation_capabilities != self.capabilities {
            return Err(self.contract_error(
                "capabilities changed between registration and invocation".to_string(),
            ));
        }

        let active = PluginPlanePosition {
            channel: 0,
            z: 0,
            time: 0,
        };
        let begin = wit_operation::BeginRequest {
            operation_id: self.id.clone(),
            command_id: None,
            command_label: None,
            argument: String::new(),
            parameters_json,
            image: adapter.image_metadata(),
            selected_scope: wit_operation::PlaneScope::AllPlanes,
            active_position: plugin_position_to_wit(active),
            plane_count: adapter.layout().plane_count(),
        };
        let invocation = guest
            .call_begin(&mut store, &self.entrypoint, &begin)
            .map_err(|error| self.trap_error(&store, error))?
            .map_err(|error| self.guest_error(&mut store, error))?;

        let staging = (|| -> crate::commands::Result<(DatasetF32, Vec<Value>)> {
            let mut staged = adapter.staged_dataset();
            let mut schedule = adapter
                .layout()
                .all_plane_schedule()
                .map_err(|error| self.contract_error(error.to_string()))?;
            let mut result_rows = Vec::new();

            for &position in adapter.layout().all_plane_positions() {
                check_invocation_state(&store).map_err(|error| self.execution_error(error))?;
                let plane = adapter
                    .encode_plane(position)
                    .map_err(dataset_input_error)?;
                let request = wit_operation::PlaneRequest {
                    plane,
                    area_roi: None,
                };
                let output = guest
                    .operation_invocation()
                    .call_process_plane(&mut store, invocation, &request)
                    .map_err(|error| self.trap_error(&store, error))?
                    .map_err(|error| self.guest_error(&mut store, error))?;

                if let Some(replacement) = output.replacement.as_ref() {
                    let input_layout = adapter
                        .layout()
                        .plane_layout(position)
                        .map_err(|error| self.contract_error(error.to_string()))?;
                    let replacement_position = wit_position_to_plugin(replacement.position);
                    let replacement_layout = PluginPlaneLayout::new(
                        replacement.width,
                        replacement.height,
                        adapter.layout().bounds(),
                        replacement_position,
                        wit_pixel_type_to_plugin(replacement.sample_type),
                        replacement.pixels.len(),
                    )
                    .map_err(|error| self.contract_error(error.to_string()))?;
                    input_layout
                        .validate_replacement(Some(&replacement_layout), &self.capabilities)
                        .map_err(|error| self.contract_error(error.to_string()))?;
                    adapter
                        .scatter_replacement(&mut staged, replacement)
                        .map_err(|error| self.contract_error(error.to_string()))?;
                }
                schedule
                    .record(position)
                    .map_err(|error| self.contract_error(error.to_string()))?;
                append_measurements(
                    output.measurements,
                    &mut store.data_mut().output_budget,
                    &mut result_rows,
                )
                .map_err(|error| self.contract_error(error.to_string()))?;
            }

            schedule
                .finish()
                .map_err(|error| self.contract_error(error.to_string()))?;
            check_invocation_state(&store).map_err(|error| self.execution_error(error))?;
            Ok((staged, result_rows))
        })();
        let (mut staged, mut result_rows) = match staging {
            Ok(staged) => staged,
            Err(error) => {
                // A borrowed `process-plane` call leaves ownership with the host. Explicitly
                // release it on every pre-finish failure instead of relying on Store teardown.
                let _ = invocation.resource_drop(&mut store);
                return Err(error);
            }
        };
        if let Err(error) = check_invocation_state(&store) {
            let _ = invocation.resource_drop(&mut store);
            return Err(self.execution_error(error));
        }

        // `finish` consumes the resource even when the guest returns a structured error.
        let finish = guest
            .call_finish(&mut store, invocation)
            .map_err(|error| self.trap_error(&store, error))?
            .map_err(|error| self.guest_error(&mut store, error))?;

        if let Some(metadata) = finish.metadata.as_ref() {
            adapter
                .apply_finish_metadata(&mut staged, metadata, &mut store.data_mut().output_budget)
                .map_err(|error| self.contract_error(error.to_string()))?;
        }
        append_measurements(
            finish.measurements,
            &mut store.data_mut().output_budget,
            &mut result_rows,
        )
        .map_err(|error| self.contract_error(error.to_string()))?;
        let status = match finish.status {
            Some(status) => {
                store
                    .data_mut()
                    .output_budget
                    .validate_text("operation status", &status)
                    .map_err(|error| self.contract_error(error.to_string()))?;
                Some(status)
            }
            None => None,
        };

        check_invocation_state(&store).map_err(|error| self.execution_error(error))?;
        staged.validate()?;
        let measurements = (!result_rows.is_empty()).then(|| MeasurementTable {
            values: BTreeMap::from([("rows".to_string(), Value::Array(result_rows))]),
        });
        Ok(OpOutput {
            dataset: staged,
            measurements,
            status,
        })
    }

    fn execution_error(&self, message: String) -> OpsError {
        OpsError::PluginExecution {
            operation: self.id.clone(),
            message: concise_error(&message),
        }
    }

    fn contract_error(&self, message: String) -> OpsError {
        self.execution_error(format!("contract violation: {message}"))
    }

    fn trap_error(&self, store: &Store<HostState>, error: wasmtime::Error) -> OpsError {
        if let Some(message) = &store.data().contract_violation {
            return self.contract_error(message.clone());
        }
        if store.data().cancelled() {
            return self.execution_error("deadline exceeded".to_string());
        }
        if store.get_fuel().ok() == Some(0) {
            return self.execution_error("fuel exhausted".to_string());
        }
        let message = root_error_message(&error);
        let normalized = message.to_ascii_lowercase();
        if normalized.contains("memory")
            && (normalized.contains("limit") || normalized.contains("grow"))
        {
            return self.execution_error(format!("memory limit exceeded: {message}"));
        }
        self.execution_error(format!("sandbox trap: {message}"))
    }

    fn guest_error(&self, store: &mut Store<HostState>, error: wit_types::PluginError) -> OpsError {
        if let Err(validation) = validate_guest_error(&error, &mut store.data_mut().output_budget) {
            return self.contract_error(validation.to_string());
        }
        let mut message = error.message;
        if let Some(details) = error.details_json {
            message.push_str("; details: ");
            message.push_str(&details);
        }
        match error.kind {
            wit_types::ErrorKind::InvalidParameters => invalid_parameters(message),
            wit_types::ErrorKind::UnsupportedImage => OpsError::UnsupportedLayout(message),
            wit_types::ErrorKind::Cancelled => {
                self.execution_error(format!("cancelled: {message}"))
            }
            wit_types::ErrorKind::InvalidBuffer => {
                self.contract_error(format!("guest rejected a buffer: {message}"))
            }
            wit_types::ErrorKind::ResourceLimit => {
                self.execution_error(format!("guest resource limit: {message}"))
            }
            wit_types::ErrorKind::Internal => {
                self.execution_error(format!("guest internal error: {message}"))
            }
        }
    }
}

fn preflight_registration(catalog: &PluginCatalog) -> Result<(), PluginRuntimeError> {
    if catalog.operations.len() > MAX_REGISTRATION_OPERATIONS {
        return Err(PluginRuntimeError::RegistrationLimit(format!(
            "{} operations exceed the per-call limit of {MAX_REGISTRATION_OPERATIONS}",
            catalog.operations.len()
        )));
    }
    let package_ids = catalog
        .operations
        .values()
        .map(|operation| operation.plugin_id.as_str())
        .collect::<BTreeSet<_>>();
    if package_ids.len() > MAX_REGISTRATION_PACKAGES {
        return Err(PluginRuntimeError::RegistrationLimit(format!(
            "{} executable packages exceed the per-call limit of {MAX_REGISTRATION_PACKAGES}",
            package_ids.len()
        )));
    }

    let mut artifact_bytes = 0_u64;
    for plugin_id in package_ids {
        let package = catalog.packages.get(plugin_id).ok_or_else(|| {
            PluginRuntimeError::RuntimeSetup(format!(
                "catalog operation references missing package `{plugin_id}`"
            ))
        })?;
        let metadata = package.component_path.symlink_metadata().map_err(|error| {
            PluginRuntimeError::ArtifactRead {
                plugin_id: plugin_id.to_string(),
                path: package.component_path.clone(),
                message: error.to_string(),
            }
        })?;
        artifact_bytes = artifact_bytes.checked_add(metadata.len()).ok_or_else(|| {
            PluginRuntimeError::RegistrationLimit(
                "aggregate component size overflowed the host address space".to_string(),
            )
        })?;
        if artifact_bytes > MAX_REGISTRATION_ARTIFACT_BYTES {
            return Err(PluginRuntimeError::RegistrationLimit(format!(
                "component artifacts contain {artifact_bytes} bytes in aggregate, above the {MAX_REGISTRATION_ARTIFACT_BYTES}-byte limit"
            )));
        }
    }
    Ok(())
}

fn query_capabilities(
    runtime: &RuntimeCore,
    pre: &bindings::ImageOperationPluginPre<HostState>,
    operation_id: &str,
    entrypoint: &str,
    timeout: Duration,
) -> Result<PluginOperationCapabilities, PluginRuntimeError> {
    let mut store = runtime
        .store(timeout, CAPABILITY_FUEL)
        .map_err(PluginRuntimeError::RuntimeSetup)?;
    let instance =
        pre.instantiate(&mut store)
            .map_err(|error| PluginRuntimeError::Capabilities {
                operation_id: operation_id.to_string(),
                message: registration_trap_message(&store, error),
            })?;
    let guest = instance.image_rs_plugin_image_operation();
    let capabilities = guest
        .call_capabilities(&mut store, entrypoint)
        .map_err(|error| PluginRuntimeError::Capabilities {
            operation_id: operation_id.to_string(),
            message: registration_trap_message(&store, error),
        })?
        .map_err(|error| PluginRuntimeError::Capabilities {
            operation_id: operation_id.to_string(),
            message: validated_registration_guest_error(&mut store, error),
        })?;
    convert_capabilities(capabilities).map_err(|error| PluginRuntimeError::Capabilities {
        operation_id: operation_id.to_string(),
        message: error.to_string(),
    })
}

fn ensure_registry_compatible(
    operation_id: &str,
    capabilities: &PluginOperationCapabilities,
) -> Result<(), PluginRuntimeError> {
    let supports_a_pixel_type = [
        PluginPixelType::U8,
        PluginPixelType::U16,
        PluginPixelType::F32,
    ]
    .into_iter()
    .any(|pixel_type| {
        capabilities
            .validate_invocation(pixel_type, PluginPlaneScope::AllPlanes, false)
            .is_ok()
    });
    if !supports_a_pixel_type {
        return Err(PluginRuntimeError::Capabilities {
            operation_id: operation_id.to_string(),
            message: "the operation-registry adapter requires all-planes scope and no required ROI"
                .to_string(),
        });
    }
    Ok(())
}

fn convert_capabilities(
    capabilities: wit_operation::OperationCapabilities,
) -> Result<PluginOperationCapabilities, PluginContractError> {
    PluginOperationCapabilities::new(
        capabilities
            .supported_pixel_types
            .into_iter()
            .map(wit_pixel_type_to_plugin),
        capabilities
            .supported_scopes
            .into_iter()
            .map(|scope| match scope {
                wit_operation::PlaneScope::ActivePlane => PluginPlaneScope::ActivePlane,
                wit_operation::PlaneScope::ZStack => PluginPlaneScope::ZStack,
                wit_operation::PlaneScope::AllPlanes => PluginPlaneScope::AllPlanes,
            }),
        capabilities.requires_area_roi,
        capabilities.accepts_area_mask,
        capabilities.modifies_pixels,
    )
}

fn validate_parameters(schema: &OpSchema, params: &Value) -> crate::commands::Result<String> {
    let params = match params {
        Value::Null => Map::new(),
        Value::Object(params) => params.clone(),
        _ => {
            return Err(invalid_parameters(
                "plugin operation parameters must be a JSON object".to_string(),
            ));
        }
    };
    let specs = schema
        .params
        .iter()
        .map(|spec| (spec.name.as_str(), spec))
        .collect::<BTreeMap<_, _>>();
    for spec in &schema.params {
        if spec.required && params.get(&spec.name).is_none_or(Value::is_null) {
            return Err(invalid_parameters(format!(
                "missing required parameter `{}`",
                spec.name
            )));
        }
    }
    let mut normalized = Map::new();
    for (name, value) in params {
        let Some(spec) = specs.get(name.as_str()) else {
            return Err(invalid_parameters(format!(
                "unknown parameter `{name}` for operation `{}`",
                schema.name
            )));
        };
        if value.is_null() {
            continue;
        }
        if !parameter_kind_accepts(&spec.kind, &value) {
            return Err(invalid_parameters(format!(
                "parameter `{name}` must have kind `{}`",
                spec.kind
            )));
        }
        normalized.insert(name, value);
    }
    serde_json::to_string(&normalized)
        .map_err(|error| invalid_parameters(format!("parameters could not be serialized: {error}")))
}

fn parameter_kind_accepts(kind: &str, value: &Value) -> bool {
    match kind {
        "bool" => value.is_boolean(),
        "float" => value.is_number(),
        "int" | "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "string" => value.is_string(),
        "array" => value.is_array(),
        "array<int>" => value.as_array().is_some_and(|values| {
            values
                .iter()
                .all(|value| value.as_i64().is_some() || value.as_u64().is_some())
        }),
        "object" => value.is_object(),
        _ => false,
    }
}

fn append_measurements(
    rows: Vec<wit_types::MeasurementRow>,
    budget: &mut PluginPayloadBudget,
    output: &mut Vec<Value>,
) -> Result<(), PluginContractError> {
    let cumulative_rows = output
        .len()
        .checked_add(rows.len())
        .ok_or(PluginContractError::PayloadOverflow)?;
    if cumulative_rows > MAX_PLUGIN_MEASUREMENT_ROWS {
        return Err(PluginContractError::CollectionLimit {
            field: "measurement rows",
            actual: cumulative_rows,
            limit: MAX_PLUGIN_MEASUREMENT_ROWS,
        });
    }

    // Canonical ABI lifting has already materialized `rows`. Validate every remaining
    // collection cardinality before allocating the final JSON maps, then convert directly into
    // the output representation instead of constructing a second measurement object graph.
    for row in &rows {
        if row.values.len() > MAX_PLUGIN_MEASUREMENTS_PER_ROW {
            return Err(PluginContractError::CollectionLimit {
                field: "measurements per row",
                actual: row.values.len(),
                limit: MAX_PLUGIN_MEASUREMENTS_PER_ROW,
            });
        }
    }
    budget.charge(rows.len(), 0)?;

    for row in rows {
        let mut values = Map::new();
        if let Some(label) = row.label {
            budget.validate_text("measurement label", &label)?;
            values.insert("Label".to_string(), Value::String(label));
        }
        for measurement in row.values {
            validate_name("measurement column", &measurement.column)?;
            budget.validate_text("measurement column", &measurement.column)?;
            budget.charge(1, 0)?;
            let value = match measurement.value {
                wit_types::MeasurementValue::Number(value) => Value::Number(
                    Number::from_f64(value).ok_or(PluginContractError::NonFiniteMeasurement)?,
                ),
                wit_types::MeasurementValue::Integer(value) => Value::Number(value.into()),
                wit_types::MeasurementValue::Boolean(value) => Value::Bool(value),
                wit_types::MeasurementValue::Text(value) => {
                    budget.validate_text("measurement text", &value)?;
                    Value::String(value)
                }
                wit_types::MeasurementValue::Missing => Value::Null,
            };
            if values.insert(measurement.column, value).is_some() {
                return Err(PluginContractError::DuplicateName {
                    field: "measurement column",
                });
            }
        }
        output.push(Value::Object(values));
    }
    Ok(())
}

fn validate_guest_error(
    error: &wit_types::PluginError,
    budget: &mut PluginPayloadBudget,
) -> Result<(), PluginContractError> {
    budget.validate_text("plugin error message", &error.message)?;
    if let Some(details) = &error.details_json {
        budget.validate_json_value("plugin error details", details)?;
    }
    Ok(())
}

fn check_invocation_state(store: &Store<HostState>) -> Result<(), String> {
    if let Some(message) = &store.data().contract_violation {
        return Err(format!("contract violation: {message}"));
    }
    if store.data().cancelled() {
        return Err("deadline exceeded".to_string());
    }
    Ok(())
}

fn dataset_input_error(error: DatasetAdapterError) -> OpsError {
    match error {
        DatasetAdapterError::Core(error) => OpsError::Core(error),
        other => OpsError::UnsupportedLayout(other.to_string()),
    }
}

fn invalid_parameters(message: String) -> OpsError {
    OpsError::InvalidParams(message)
}

#[derive(Debug, Default, Eq, PartialEq)]
struct ComponentAdmissionStats {
    artifact_bytes: u64,
    max_embedding_depth: u64,
    core_modules: u64,
    nested_components: u64,
    embedded_binary_bytes: u64,
    section_items: u64,
    defined_core_functions: u64,
    code_bodies: u64,
    code_bytes: u64,
    operators: u64,
    locals: u64,
    custom_sections: u64,
    custom_bytes: u64,
}

/// Apply deterministic, engine-independent limits before synchronous native compilation.
///
/// This does not pretend to impose a wall-clock deadline on `Component::new`. Instead it rejects
/// the compact encodings that can expand into disproportionate validation or compiler work and
/// puts an explicit byte ceiling on everything that remains.
fn admit_component(bytes: &[u8]) -> Result<ComponentAdmissionStats, String> {
    let artifact_bytes = u64::try_from(bytes.len())
        .map_err(|_| "component size does not fit the admission counter".to_string())?;
    if artifact_bytes > MAX_COMPILABLE_COMPONENT_BYTES {
        return Err(format!(
            "artifact has {artifact_bytes} bytes, above the {MAX_COMPILABLE_COMPONENT_BYTES}-byte compilation limit"
        ));
    }
    if !Parser::is_component(bytes) {
        return Err("artifact is not a WebAssembly Component".to_string());
    }

    let mut stats = ComponentAdmissionStats {
        artifact_bytes,
        ..ComponentAdmissionStats::default()
    };
    let mut saw_root = false;
    let mut embedding_depth = 0_u64;

    for payload in Parser::new(0).parse_all(bytes) {
        let payload =
            payload.map_err(|error| format!("could not inspect WebAssembly structure: {error}"))?;
        match payload {
            Payload::Version { encoding, .. } => {
                if !saw_root {
                    saw_root = true;
                    if encoding != Encoding::Component {
                        return Err("artifact root is not a WebAssembly Component".to_string());
                    }
                } else {
                    embedding_depth = checked_sum(embedding_depth, 1, "component embedding depth")?;
                    stats.max_embedding_depth = stats.max_embedding_depth.max(embedding_depth);
                    enforce_limit(
                        "component embedding depth",
                        embedding_depth,
                        MAX_COMPONENT_EMBEDDING_DEPTH,
                    )?;
                }
            }
            Payload::TypeSection(section) => {
                charge_section_items(&mut stats, section.count(), "core type section")?;
            }
            Payload::ImportSection(section) => {
                charge_section_items(&mut stats, section.count(), "core import section")?;
            }
            Payload::FunctionSection(section) => {
                let count = u64::from(section.count());
                stats.defined_core_functions = checked_sum(
                    stats.defined_core_functions,
                    count,
                    "defined core functions",
                )?;
                enforce_limit(
                    "defined core functions",
                    stats.defined_core_functions,
                    MAX_DEFINED_CORE_FUNCTIONS,
                )?;
                charge_section_items(&mut stats, section.count(), "core function section")?;
            }
            Payload::TableSection(section) => {
                charge_section_items(&mut stats, section.count(), "core table section")?;
            }
            Payload::MemorySection(section) => {
                charge_section_items(&mut stats, section.count(), "core memory section")?;
            }
            Payload::TagSection(section) => {
                charge_section_items(&mut stats, section.count(), "core tag section")?;
            }
            Payload::GlobalSection(section) => {
                charge_section_items(&mut stats, section.count(), "core global section")?;
            }
            Payload::ExportSection(section) => {
                charge_section_items(&mut stats, section.count(), "core export section")?;
            }
            Payload::StartSection { .. } => {
                charge_items(&mut stats.section_items, 1, "core start section")?;
            }
            Payload::ElementSection(section) => {
                charge_section_items(&mut stats, section.count(), "core element section")?;
            }
            Payload::DataCountSection { count, .. } => {
                charge_section_items(&mut stats, count, "core data-count section")?;
            }
            Payload::DataSection(section) => {
                charge_section_items(&mut stats, section.count(), "core data section")?;
            }
            Payload::CodeSectionStart { count, .. } => {
                charge_section_items(&mut stats, count, "core code section")?;
            }
            Payload::CodeSectionEntry(body) => inspect_function_body(&body, &mut stats)?,
            Payload::ModuleSection {
                unchecked_range, ..
            } => {
                charge_limited(
                    &mut stats.core_modules,
                    1,
                    "embedded core modules",
                    MAX_EMBEDDED_CORE_MODULES,
                )?;
                charge_embedded_range(&mut stats, unchecked_range)?;
                charge_items(&mut stats.section_items, 1, "embedded core module")?;
            }
            Payload::InstanceSection(section) => {
                charge_section_items(&mut stats, section.count(), "core instance section")?;
            }
            Payload::CoreTypeSection(section) => {
                charge_section_items(&mut stats, section.count(), "component core-type section")?;
            }
            Payload::ComponentSection {
                unchecked_range, ..
            } => {
                charge_limited(
                    &mut stats.nested_components,
                    1,
                    "nested components",
                    MAX_EMBEDDED_COMPONENTS,
                )?;
                charge_embedded_range(&mut stats, unchecked_range)?;
                charge_items(&mut stats.section_items, 1, "nested component")?;
            }
            Payload::ComponentInstanceSection(section) => {
                charge_section_items(&mut stats, section.count(), "component instance section")?;
            }
            Payload::ComponentAliasSection(section) => {
                charge_section_items(&mut stats, section.count(), "component alias section")?;
            }
            Payload::ComponentTypeSection(section) => {
                charge_section_items(&mut stats, section.count(), "component type section")?;
            }
            Payload::ComponentCanonicalSection(section) => {
                charge_section_items(&mut stats, section.count(), "component canonical section")?;
            }
            Payload::ComponentStartSection { .. } => {
                charge_items(&mut stats.section_items, 1, "component start section")?;
            }
            Payload::ComponentImportSection(section) => {
                charge_section_items(&mut stats, section.count(), "component import section")?;
            }
            Payload::ComponentExportSection(section) => {
                charge_section_items(&mut stats, section.count(), "component export section")?;
            }
            Payload::CustomSection(section) => {
                charge_limited(
                    &mut stats.custom_sections,
                    1,
                    "custom sections",
                    MAX_CUSTOM_SECTIONS,
                )?;
                let section_bytes = u64::try_from(section.data().len()).map_err(|_| {
                    "custom-section size does not fit the admission counter".to_string()
                })?;
                enforce_limit(
                    "one custom section",
                    section_bytes,
                    MAX_CUSTOM_SECTION_BYTES,
                )?;
                charge_limited(
                    &mut stats.custom_bytes,
                    section_bytes,
                    "custom-section bytes",
                    MAX_CUSTOM_SECTION_BYTES_TOTAL,
                )?;
            }
            Payload::UnknownSection { id, .. } => {
                return Err(format!("unknown WebAssembly section id {id}"));
            }
            Payload::End(_) => {
                embedding_depth = embedding_depth.saturating_sub(1);
            }
            _ => {
                return Err("unsupported WebAssembly structure encountered".to_string());
            }
        }
    }

    Validator::new()
        .validate_all(bytes)
        .map_err(|error| format!("WebAssembly validation failed: {error}"))?;
    Ok(stats)
}

fn inspect_function_body(
    body: &wasmparser::FunctionBody<'_>,
    stats: &mut ComponentAdmissionStats,
) -> Result<(), String> {
    charge_limited(
        &mut stats.code_bodies,
        1,
        "core function bodies",
        MAX_DEFINED_CORE_FUNCTIONS,
    )?;
    let body_bytes = u64::try_from(body.as_bytes().len())
        .map_err(|_| "function-body size does not fit the admission counter".to_string())?;
    enforce_limit(
        "one core function body",
        body_bytes,
        MAX_CORE_FUNCTION_BODY_BYTES,
    )?;
    charge_limited(
        &mut stats.code_bytes,
        body_bytes,
        "core function-body bytes",
        MAX_CORE_CODE_BYTES,
    )?;

    let mut function_locals = 0_u64;
    for local in body
        .get_locals_reader()
        .map_err(|error| format!("could not inspect core function locals: {error}"))?
    {
        let (count, _) =
            local.map_err(|error| format!("could not inspect core function locals: {error}"))?;
        function_locals = checked_sum(function_locals, u64::from(count), "core function locals")?;
        enforce_limit(
            "locals in one core function",
            function_locals,
            MAX_CORE_FUNCTION_LOCALS,
        )?;
    }
    charge_limited(
        &mut stats.locals,
        function_locals,
        "core function locals",
        MAX_CORE_LOCALS,
    )?;

    let mut function_operators = 0_u64;
    for operator in body
        .get_operators_reader()
        .map_err(|error| format!("could not inspect core function operators: {error}"))?
    {
        operator.map_err(|error| format!("could not inspect core function operators: {error}"))?;
        function_operators = checked_sum(function_operators, 1, "operators in one core function")?;
        enforce_limit(
            "operators in one core function",
            function_operators,
            MAX_CORE_FUNCTION_OPERATORS,
        )?;
    }
    charge_limited(
        &mut stats.operators,
        function_operators,
        "core function operators",
        MAX_CORE_OPERATORS,
    )?;
    Ok(())
}

fn charge_section_items(
    stats: &mut ComponentAdmissionStats,
    count: u32,
    label: &str,
) -> Result<(), String> {
    charge_items(&mut stats.section_items, u64::from(count), label)
}

fn charge_items(total: &mut u64, count: u64, label: &str) -> Result<(), String> {
    charge_limited(total, count, label, MAX_WASM_SECTION_ITEMS)
}

fn charge_embedded_range(
    stats: &mut ComponentAdmissionStats,
    range: std::ops::Range<usize>,
) -> Result<(), String> {
    let bytes = range
        .end
        .checked_sub(range.start)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| "embedded binary range does not fit the admission counter".to_string())?;
    charge_limited(
        &mut stats.embedded_binary_bytes,
        bytes,
        "aggregate embedded-binary bytes",
        MAX_EMBEDDED_BINARY_BYTES,
    )
}

fn charge_limited(total: &mut u64, amount: u64, label: &str, limit: u64) -> Result<(), String> {
    *total = checked_sum(*total, amount, label)?;
    enforce_limit(label, *total, limit)
}

fn checked_sum(current: u64, amount: u64, label: &str) -> Result<u64, String> {
    current
        .checked_add(amount)
        .ok_or_else(|| format!("{label} overflowed the admission counter"))
}

fn enforce_limit(label: &str, actual: u64, limit: u64) -> Result<(), String> {
    if actual > limit {
        return Err(format!(
            "{label} total {actual} exceeds the limit of {limit}"
        ));
    }
    Ok(())
}

fn read_component(plugin_id: &str, path: &Path) -> Result<Vec<u8>, PluginRuntimeError> {
    let metadata = path
        .symlink_metadata()
        .map_err(|error| PluginRuntimeError::ArtifactRead {
            plugin_id: plugin_id.to_string(),
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(PluginRuntimeError::ArtifactRead {
            plugin_id: plugin_id.to_string(),
            path: path.to_path_buf(),
            message: "the component must remain a regular file rather than a symlink".to_string(),
        });
    }
    if metadata.len() > MAX_COMPILABLE_COMPONENT_BYTES {
        return Err(PluginRuntimeError::ArtifactLimit {
            plugin_id: plugin_id.to_string(),
            path: path.to_path_buf(),
            actual: metadata.len(),
            limit: MAX_COMPILABLE_COMPONENT_BYTES,
        });
    }
    let file = File::open(path).map_err(|error| PluginRuntimeError::ArtifactRead {
        plugin_id: plugin_id.to_string(),
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_COMPILABLE_COMPONENT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| PluginRuntimeError::ArtifactRead {
            plugin_id: plugin_id.to_string(),
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    if bytes.len() as u64 > MAX_COMPILABLE_COMPONENT_BYTES {
        return Err(PluginRuntimeError::ArtifactLimit {
            plugin_id: plugin_id.to_string(),
            path: path.to_path_buf(),
            actual: bytes.len() as u64,
            limit: MAX_COMPILABLE_COMPONENT_BYTES,
        });
    }
    Ok(bytes)
}

fn registration_trap_message(store: &Store<HostState>, error: wasmtime::Error) -> String {
    if let Some(message) = &store.data().contract_violation {
        return format!("contract violation: {message}");
    }
    if store.data().cancelled() {
        return "deadline exceeded".to_string();
    }
    if store.get_fuel().ok() == Some(0) {
        return "fuel exhausted".to_string();
    }
    format!("sandbox trap: {}", root_error_message(&error))
}

fn validated_registration_guest_error(
    store: &mut Store<HostState>,
    error: wit_types::PluginError,
) -> String {
    if let Err(validation) = validate_guest_error(&error, &mut store.data_mut().output_budget) {
        return format!("invalid structured error: {validation}");
    }
    format!("guest {:?}: {}", error.kind, concise_error(&error.message))
}

fn concise_error(message: &str) -> String {
    let mut output = message.replace('\n', " ");
    if output.len() > MAX_ERROR_MESSAGE_BYTES {
        let mut end = MAX_ERROR_MESSAGE_BYTES;
        while !output.is_char_boundary(end) {
            end -= 1;
        }
        output.truncate(end);
        output.push('…');
    }
    output
}

fn root_error_message(error: &wasmtime::Error) -> String {
    let message = error
        .chain()
        .last()
        .map(ToString::to_string)
        .unwrap_or_else(|| error.to_string());
    concise_error(&message)
}

#[cfg(test)]
mod tests;
