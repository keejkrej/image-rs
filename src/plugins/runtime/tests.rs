use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ndarray::{ArrayD, IxDyn};
use serde_json::{Value, json};

use super::*;
use crate::commands::{
    AreaMask, AreaMaskSupport, CancellationToken, ExecutionControl, InvocationRequest,
    OperationScope, ParamSpec, PlanePosition, ProgressEvent, ProgressSink, default_registry,
};
use crate::model::{AxisKind, Dataset, Dim, Metadata, PixelType};
use crate::runtime::OpsService;
use crate::workflow::{OpInvocation, PipelineSpec, run_pipeline};

const ADD_ONE: &str = "org.image-rs.fixture.add-one.add-one";
const FAIL_FINISH: &str = "org.image-rs.fixture.add-one.fail-finish";
const SPIN: &str = "org.image-rs.fixture.add-one.spin";
const GROW_MEMORY: &str = "org.image-rs.fixture.add-one.grow-memory";
const BAD_PROGRESS: &str = "org.image-rs.fixture.add-one.bad-progress";
const BAD_REPLACEMENT: &str = "org.image-rs.fixture.add-one.bad-replacement";
const NEEDS_ROI: &str = "org.image-rs.fixture.add-one.needs-roi";

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/plugins")
}

fn test_dataset(pixel_type: PixelType) -> DatasetF32 {
    // Z, Y, X deliberately differs from ndarray's default metadata order.
    let shape = [2, 2, 3];
    let data = ArrayD::from_shape_vec(
        IxDyn(&shape),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    )
    .unwrap();
    Dataset::new(
        data,
        Metadata {
            dims: vec![
                Dim::new(AxisKind::Z, 2),
                Dim::new(AxisKind::Y, 2),
                Dim::new(AxisKind::X, 3),
            ],
            pixel_type,
            source: Some(PathBuf::from("source-preserved.tif")),
            extras: [("fixture".to_string(), json!(true))].into(),
            ..Metadata::default()
        },
    )
    .unwrap()
}

fn scoped_dataset() -> DatasetF32 {
    // T, Z, C, Y, X keeps all scheduling axes non-trivial and non-canonical.
    let shape = [2, 2, 2, 2, 3];
    let mut data = ArrayD::zeros(IxDyn(&shape));
    for time in 0..shape[0] {
        for z in 0..shape[1] {
            for channel in 0..shape[2] {
                for y in 0..shape[3] {
                    for x in 0..shape[4] {
                        data[IxDyn(&[time, z, channel, y, x])] =
                            (time * 1_000 + z * 100 + channel * 10 + y * 3 + x) as f32;
                    }
                }
            }
        }
    }
    Dataset::new(
        data,
        Metadata {
            dims: vec![
                Dim::new(AxisKind::Time, shape[0]),
                Dim::new(AxisKind::Z, shape[1]),
                Dim::new(AxisKind::Channel, shape[2]),
                Dim::new(AxisKind::Y, shape[3]),
                Dim::new(AxisKind::X, shape[4]),
            ],
            pixel_type: PixelType::F32,
            ..Metadata::default()
        },
    )
    .unwrap()
}

fn result_dataset<'a>(
    result: &'a crate::commands::InvocationResult,
    input: &'a Arc<DatasetF32>,
) -> &'a DatasetF32 {
    result.dataset_effect.dataset(input).as_ref()
}

#[test]
fn no_op_detection_uses_pixel_bits_including_nan_payloads() {
    let mut left = test_dataset(PixelType::F32);
    left.data[IxDyn(&[0, 0, 0])] = f32::from_bits(0x7fc0_0011);
    let mut right = left.clone();
    assert!(datasets_bit_identical(&left, &right));

    right.data[IxDyn(&[0, 0, 0])] = f32::from_bits(0x7fc0_0012);
    assert!(!datasets_bit_identical(&left, &right));
}

#[derive(Default)]
struct RecordingProgress(Mutex<Vec<ProgressEvent>>);

impl ProgressSink for RecordingProgress {
    fn report(&self, event: ProgressEvent) {
        self.0.lock().unwrap().push(event);
    }
}

struct CancelOnGuestProgress {
    cancellation: CancellationToken,
}

impl ProgressSink for CancelOnGuestProgress {
    fn report(&self, event: ProgressEvent) {
        if event.detail.is_some() {
            self.cancellation.cancel();
        }
    }
}

fn fixture_catalog() -> PluginCatalog {
    let discovery = PluginCatalog::discover(fixture_root()).unwrap();
    assert!(discovery.rejected.is_empty(), "{:?}", discovery.rejected);
    discovery.catalog
}

fn fixture_registry() -> OperationRegistry {
    let catalog = fixture_catalog();
    let mut registry = OperationRegistry::default();
    assert_eq!(catalog.register_operations(&mut registry).unwrap(), 7);
    registry
}

#[test]
fn required_roi_capability_registers_as_a_scoped_descriptor() {
    let catalog = fixture_catalog();
    let mut registry = OperationRegistry::default();
    assert_eq!(catalog.register_operations(&mut registry).unwrap(), 7);
    let descriptor = registry.describe(NEEDS_ROI).unwrap();
    assert_eq!(descriptor.area_mask, AreaMaskSupport::Required);
    assert_eq!(
        descriptor.scopes,
        vec![
            OperationScope::ActivePlane,
            OperationScope::ZStack,
            OperationScope::AllPlanes,
        ]
    );
}

#[test]
fn fixture_runs_through_catalog_registry_and_ops_service_for_every_pixel_type() {
    let service = OpsService::from_registry(fixture_registry());

    for pixel_type in [PixelType::U8, PixelType::U16, PixelType::F32] {
        let input = test_dataset(pixel_type);
        let original = input.data.clone();
        let output = service.execute(ADD_ONE, &input, &json!({})).unwrap();

        assert_eq!(
            input.data, original,
            "the input dataset must remain immutable"
        );
        assert_eq!(output.dataset.data, original.mapv(|value| value + 1.0));
        assert_eq!(output.dataset.metadata.source, input.metadata.source);
        assert_eq!(
            output.dataset.metadata.pixel_type,
            input.metadata.pixel_type
        );
        assert_eq!(output.dataset.metadata.extras, input.metadata.extras);
        assert_eq!(
            output.dataset.metadata.channel_names,
            input.metadata.channel_names
        );
        let x_dimension = output
            .dataset
            .metadata
            .dims
            .iter()
            .find(|dimension| dimension.axis == AxisKind::X)
            .unwrap();
        assert_eq!(x_dimension.spacing, Some(1.25));
        assert_eq!(output.status.as_deref(), Some("add-one complete"));

        let rows = output
            .measurements
            .as_ref()
            .and_then(|table| table.values.get("rows"))
            .and_then(Value::as_array)
            .unwrap();
        assert_eq!(
            rows,
            &vec![
                json!({"plane": 1}),
                json!({"plane": 2}),
                json!({
                    "Label": "summary",
                    "score": 1.5,
                    "ok": true,
                    "note": "done",
                    "missing": null
                }),
            ]
        );
    }
}

#[test]
fn ops_service_invokes_active_plane_and_z_stack_with_exact_progress_context() {
    let service = OpsService::from_registry(fixture_registry());
    let input = Arc::new(scoped_dataset());
    let original = input.data.clone();
    let active = PlanePosition {
        channel: 1,
        z: 1,
        time: 1,
    };
    let progress = Arc::new(RecordingProgress::default());
    let control = ExecutionControl::new(CancellationToken::default(), progress.clone());
    let active_result = service
        .invoke(
            InvocationRequest {
                operation: ADD_ONE.to_string(),
                input: input.clone(),
                parameters: json!({}),
                scope: OperationScope::ActivePlane,
                active,
                area_mask: None,
            },
            &control,
        )
        .unwrap();
    let active_output = result_dataset(&active_result, &input);
    let mut expected = original.clone();
    for y in 0..2 {
        for x in 0..3 {
            expected[IxDyn(&[1, 1, 1, y, x])] += 1.0;
        }
    }
    assert_eq!(active_output.data, expected);
    assert_eq!(input.data, original);

    let events = progress.0.lock().unwrap();
    assert!(events.iter().any(|event| {
        event.completed_planes == 0
            && event.total_planes == 1
            && event.current_plane == Some(active)
            && event.detail.as_ref().is_some_and(|detail| {
                detail.completed == 1
                    && detail.total == Some(1)
                    && detail.message.as_deref() == Some("add-one")
            })
    }));
    assert!(events.iter().any(|event| {
        event.completed_planes == 1
            && event.total_planes == 1
            && event.current_plane == Some(active)
            && event.detail.is_none()
    }));
    assert!(events.iter().any(|event| {
        event.completed_planes == 1
            && event.total_planes == 1
            && event.current_plane.is_none()
            && event.detail.as_ref().is_some_and(|detail| {
                detail.completed == 1
                    && detail.total == Some(1)
                    && detail.message.as_deref() == Some("finish")
            })
    }));
    drop(events);

    let z_result = service
        .invoke(
            InvocationRequest {
                operation: ADD_ONE.to_string(),
                input: input.clone(),
                parameters: json!({}),
                scope: OperationScope::ZStack,
                active,
                area_mask: None,
            },
            &ExecutionControl::default(),
        )
        .unwrap();
    let z_output = result_dataset(&z_result, &input);
    let mut expected = original.clone();
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..3 {
                expected[IxDyn(&[1, z, 1, y, x])] += 1.0;
            }
        }
    }
    assert_eq!(z_output.data, expected);
    assert_eq!(input.data, original);
}

#[test]
fn ops_service_enforces_required_irregular_roi_on_every_z_plane() {
    let service = OpsService::from_registry(fixture_registry());
    let input = Arc::new(scoped_dataset());
    let original = input.data.clone();
    let active = PlanePosition {
        channel: 1,
        z: 1,
        time: 1,
    };

    let error = service
        .invoke(
            InvocationRequest {
                operation: NEEDS_ROI.to_string(),
                input: input.clone(),
                parameters: json!({}),
                scope: OperationScope::ZStack,
                active,
                area_mask: None,
            },
            &ExecutionControl::default(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("requires an area mask"));

    // Irregular checkerboard membership inside the full 3x2 plane. The fixture deliberately
    // changes every replacement pixel, so equality here proves host-side restoration.
    let area_mask = AreaMask::new(0, 0, 3, 2, vec![1, 0, 1, 0, 1, 0]).unwrap();
    let result = service
        .invoke(
            InvocationRequest {
                operation: NEEDS_ROI.to_string(),
                input: input.clone(),
                parameters: json!({}),
                scope: OperationScope::ZStack,
                active,
                area_mask: Some(area_mask),
            },
            &ExecutionControl::default(),
        )
        .unwrap();
    let output = result_dataset(&result, &input);
    let mut expected = original.clone();
    for z in 0..2 {
        for (y, x) in [(0, 0), (0, 2), (1, 1)] {
            expected[IxDyn(&[1, z, 1, y, x])] += 1.0;
        }
    }
    assert_eq!(output.data, expected);
    assert_eq!(input.data, original);
}

#[test]
fn caller_cancellation_and_traps_discard_every_staged_replacement() {
    let service = OpsService::from_registry(fixture_registry());
    let input = Arc::new(test_dataset(PixelType::F32));
    let original = input.as_ref().clone();
    let cancellation = CancellationToken::default();
    let control = ExecutionControl::new(
        cancellation.clone(),
        Arc::new(CancelOnGuestProgress { cancellation }),
    );

    let error = service
        .invoke(
            InvocationRequest {
                operation: ADD_ONE.to_string(),
                input: input.clone(),
                parameters: json!({}),
                scope: OperationScope::AllPlanes,
                active: PlanePosition::default(),
                area_mask: None,
            },
            &control,
        )
        .unwrap_err();
    assert!(error.to_string().contains("cancelled"), "{error}");
    assert_eq!(input.data, original.data);
    assert_eq!(input.metadata, original.metadata);

    let trap = service
        .invoke(
            InvocationRequest {
                operation: BAD_REPLACEMENT.to_string(),
                input: input.clone(),
                parameters: json!({}),
                scope: OperationScope::ActivePlane,
                active: PlanePosition::default(),
                area_mask: None,
            },
            &ExecutionControl::default(),
        )
        .unwrap_err();
    assert!(trap.to_string().contains("plane position"), "{trap}");
    assert_eq!(input.data, original.data);
    assert_eq!(input.metadata, original.metadata);
}

#[test]
fn fixture_operation_survives_the_workflow_boundary_losslessly() {
    let registry = fixture_registry();
    let input = test_dataset(PixelType::F32);
    let spec = PipelineSpec {
        name: Some("two plugin steps".to_string()),
        operations: vec![
            OpInvocation {
                op: ADD_ONE.to_string(),
                params: json!({}),
            },
            OpInvocation {
                op: ADD_ONE.to_string(),
                params: json!({}),
            },
        ],
    };

    let (output, report) = run_pipeline(&spec, &input, &registry).unwrap();
    assert_eq!(output.data, input.data.mapv(|value| value + 2.0));
    assert_eq!(report.steps.len(), 2);
    assert!(
        report
            .steps
            .iter()
            .all(|step| step.status.as_deref() == Some("add-one complete"))
    );
    let expected_rows = vec![
        json!({"plane": 1}),
        json!({"plane": 2}),
        json!({
            "Label": "summary",
            "score": 1.5,
            "ok": true,
            "note": "done",
            "missing": null
        }),
    ];
    for step in &report.steps {
        let rows = step
            .measurements
            .as_ref()
            .and_then(|table| table.values.get("rows"))
            .and_then(Value::as_array)
            .unwrap();
        assert_eq!(rows, &expected_rows);
    }
}

#[test]
fn integer_typed_builtin_output_is_quantized_at_the_plugin_boundary() {
    let catalog = fixture_catalog();
    let mut registry = default_registry();
    catalog.register_operations(&mut registry).unwrap();
    let input = test_dataset(PixelType::U8);

    let blurred = registry
        .execute("gaussian.blur", &input, &json!({"sigma": 0.75}))
        .unwrap()
        .dataset;
    assert!(blurred.data.iter().any(|value| value.fract() != 0.0));
    let expected = blurred
        .data
        .mapv(|value| value.clamp(0.0, 255.0).round() + 1.0);

    let spec = PipelineSpec {
        name: Some("built-in then plugin".to_string()),
        operations: vec![
            OpInvocation {
                op: "gaussian.blur".to_string(),
                params: json!({"sigma": 0.75}),
            },
            OpInvocation {
                op: ADD_ONE.to_string(),
                params: json!({}),
            },
        ],
    };
    let (output, _) = run_pipeline(&spec, &input, &registry).unwrap();
    assert_eq!(output.data, expected);
    assert_eq!(output.metadata.pixel_type, PixelType::U8);
}

#[test]
fn finish_error_discards_every_staged_replacement() {
    let registry = fixture_registry();
    let input = test_dataset(PixelType::F32);
    let original = input.clone();

    let error = registry
        .execute(FAIL_FINISH, &input, &json!({}))
        .unwrap_err();
    assert!(error.to_string().contains("intentional finish failure"));
    assert_eq!(input.data, original.data);
    assert_eq!(input.metadata, original.metadata);
}

#[test]
fn sandbox_and_host_contract_fail_closed_for_adversarial_guests() {
    let registry = fixture_registry();
    let input = test_dataset(PixelType::F32);
    let original = input.clone();

    for (operation, expected) in [
        (SPIN, "fuel exhausted"),
        (GROW_MEMORY, "memory limit"),
        (BAD_PROGRESS, "progress regressed"),
        (BAD_REPLACEMENT, "plane position"),
    ] {
        let error = registry.execute(operation, &input, &json!({})).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "{operation} returned unexpected error: {error}"
        );
        assert_eq!(input.data, original.data);
        assert_eq!(input.metadata, original.metadata);
    }
}

#[test]
fn epoch_deadline_interrupts_guest_even_when_fuel_is_effectively_unbounded() {
    let discovery = PluginCatalog::discover(fixture_root()).unwrap();
    let package = &discovery.catalog.packages["org.image-rs.fixture.add-one"];
    let runtime = RuntimeCore::new().unwrap();
    let pre = runtime
        .compile("org.image-rs.fixture.add-one", &package.component_path)
        .unwrap();
    let timeout = Duration::from_millis(30);
    let mut store = runtime.store(timeout, u64::MAX).unwrap();
    let instance = pre.instantiate(&mut store).unwrap();
    let guest = instance.image_rs_plugin_image_operation();
    let input = test_dataset(PixelType::F32);
    let adapter = DatasetAdapter::new(&input).unwrap();
    let active = PluginPlanePosition {
        channel: 0,
        z: 0,
        time: 0,
    };
    let begin = wit_operation::BeginRequest {
        operation_id: SPIN.to_string(),
        command_id: None,
        command_label: None,
        argument: String::new(),
        parameters_json: "{}".to_string(),
        image: adapter.image_metadata(),
        selected_scope: wit_operation::PlaneScope::AllPlanes,
        active_position: plugin_position_to_wit(active),
        plane_count: u32::try_from(
            adapter
                .layout()
                .plane_positions(active, PluginPlaneScope::AllPlanes)
                .unwrap()
                .len(),
        )
        .unwrap(),
    };
    let invocation = guest
        .call_begin(&mut store, "spin", &begin)
        .unwrap()
        .unwrap();
    let request = wit_operation::PlaneRequest {
        plane: adapter.encode_plane(active).unwrap(),
        area_roi: None,
    };

    let started = Instant::now();
    let error = guest
        .operation_invocation()
        .call_process_plane(&mut store, invocation, &request)
        .unwrap_err();
    assert!(started.elapsed() < Duration::from_secs(1));
    assert!(
        root_error_message(&error)
            .to_ascii_lowercase()
            .contains("epoch")
            || root_error_message(&error)
                .to_ascii_lowercase()
                .contains("interrupt")
    );
    let _ = invocation.resource_drop(&mut store);
}

#[test]
fn linker_rejects_an_ambient_wasi_import() {
    let bytes = wat::parse_str(
        r#"
            (component
                (type $run (func))
                (type $wasi (instance
                    (export "run" (func (type $run)))
                ))
                (import "wasi:cli/run@0.2.0" (instance (type $wasi)))
            )
        "#,
    )
    .unwrap();
    let runtime = RuntimeCore::new().unwrap();
    let component = Component::new(&runtime.engine, bytes).unwrap();
    let mut linker = Linker::<HostState>::new(&runtime.engine);
    bindings::ImageOperationPlugin::add_to_linker::<_, HasSelf<_>>(&mut linker, |state| state)
        .unwrap();

    let error = match linker.instantiate_pre(&component) {
        Ok(_) => panic!("an ambient WASI import must not link"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("wasi:cli/run"));
}

#[test]
fn registration_collision_is_atomic_and_parameters_are_checked_before_guest_entry() {
    let catalog = fixture_catalog();
    let mut registry = OperationRegistry::default();
    catalog.register_operations(&mut registry).unwrap();
    let before = registry.len();

    let error = catalog.register_operations(&mut registry).unwrap_err();
    assert!(matches!(error, PluginRuntimeError::Registry { .. }));
    assert_eq!(registry.len(), before);

    let error = registry
        .execute(ADD_ONE, &test_dataset(PixelType::F32), &json!({"extra": 1}))
        .unwrap_err();
    assert!(matches!(error, OpsError::InvalidParams(_)));
}

#[test]
fn optional_null_parameters_are_omitted_and_required_nulls_are_rejected() {
    let schema = OpSchema {
        name: "fixture.params".to_string(),
        description: String::new(),
        params: vec![
            ParamSpec {
                name: "optional".to_string(),
                description: String::new(),
                required: false,
                kind: "float".to_string(),
            },
            ParamSpec {
                name: "required".to_string(),
                description: String::new(),
                required: true,
                kind: "int".to_string(),
            },
        ],
    };

    assert_eq!(
        validate_parameters(&schema, &json!({"optional": null, "required": 2})).unwrap(),
        "{\"required\":2}"
    );
    assert!(matches!(
        validate_parameters(&schema, &json!({"required": null})),
        Err(OpsError::InvalidParams(_))
    ));
}

#[test]
fn compile_admission_accepts_the_checked_in_component_fixture() {
    let bytes = std::fs::read(
        fixture_root()
            .join("add-one")
            .join("add-one.component.wasm"),
    )
    .unwrap();
    let stats = admit_component(&bytes).unwrap();

    assert_eq!(stats.artifact_bytes, bytes.len() as u64);
    assert!(stats.core_modules > 0);
    assert!(stats.defined_core_functions > 0);
    assert!(stats.code_bytes > 0);
    assert!(stats.operators > 0);
}

#[test]
fn compile_admission_rejects_a_compact_local_expansion_bomb() {
    let mut body = vec![1]; // one local declaration group
    encode_u32_leb((MAX_CORE_FUNCTION_LOCALS + 1) as u32, &mut body);
    body.extend_from_slice(&[0x7f, 0x0b]); // i32 locals, then `end`
    let bytes = component_with_one_core_function(body);
    assert!(bytes.len() < 64, "the hostile encoding should stay compact");

    let error = admit_component(&bytes).unwrap_err();
    assert!(
        error.contains("locals in one core function")
            && error.contains(&MAX_CORE_FUNCTION_LOCALS.to_string()),
        "unexpected admission error: {error}"
    );
}

#[test]
fn compile_admission_rejects_an_operator_heavy_function() {
    let mut body = Vec::with_capacity(MAX_CORE_FUNCTION_OPERATORS as usize + 2);
    body.push(0); // no local declaration groups
    body.extend(std::iter::repeat_n(
        0x01, // `nop`
        MAX_CORE_FUNCTION_OPERATORS as usize,
    ));
    body.push(0x0b); // `end` is also an operator and crosses the limit
    let bytes = component_with_one_core_function(body);

    let error = admit_component(&bytes).unwrap_err();
    assert!(
        error.contains("operators in one core function")
            && error.contains(&MAX_CORE_FUNCTION_OPERATORS.to_string()),
        "unexpected admission error: {error}"
    );
}

#[test]
fn compile_admission_rejects_deeply_nested_components() {
    let mut bytes = component_header();
    for _ in 0..=MAX_COMPONENT_EMBEDDING_DEPTH {
        let mut outer = component_header();
        push_wasm_section(4, &bytes, &mut outer); // nested component section
        bytes = outer;
    }

    let error = admit_component(&bytes).unwrap_err();
    assert!(
        error.contains("component embedding depth")
            && error.contains(&MAX_COMPONENT_EMBEDDING_DEPTH.to_string()),
        "unexpected admission error: {error}"
    );
}

#[test]
fn compile_admission_rejects_oversized_custom_debug_payloads() {
    let mut bytes = component_header();
    let mut custom = Vec::with_capacity(MAX_CUSTOM_SECTION_BYTES as usize + 2);
    custom.push(0); // empty custom-section name
    custom.resize(MAX_CUSTOM_SECTION_BYTES as usize + 2, 0);
    push_wasm_section(0, &custom, &mut bytes);

    let error = admit_component(&bytes).unwrap_err();
    assert!(
        error.contains("one custom section")
            && error.contains(&MAX_CUSTOM_SECTION_BYTES.to_string()),
        "unexpected admission error: {error}"
    );
}

fn component_with_one_core_function(body: Vec<u8>) -> Vec<u8> {
    let mut module = b"\0asm\x01\0\0\0".to_vec();
    push_wasm_section(1, &[1, 0x60, 0, 0], &mut module); // () -> () type
    push_wasm_section(3, &[1, 0], &mut module); // one function using type 0
    let mut code = vec![1]; // one body
    encode_u32_leb(body.len() as u32, &mut code);
    code.extend_from_slice(&body);
    push_wasm_section(10, &code, &mut module);

    let mut component = component_header();
    push_wasm_section(1, &module, &mut component); // embedded core module
    component
}

fn component_header() -> Vec<u8> {
    b"\0asm\x0d\0\x01\0".to_vec()
}

fn push_wasm_section(id: u8, payload: &[u8], output: &mut Vec<u8>) {
    output.push(id);
    encode_u32_leb(payload.len() as u32, output);
    output.extend_from_slice(payload);
}

fn encode_u32_leb(mut value: u32, output: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        output.push(byte | (u8::from(value != 0) * 0x80));
        if value == 0 {
            break;
        }
    }
}

#[test]
fn measurement_row_limit_is_cumulative_across_plugin_results() {
    let mut budget = PluginPayloadBudget::new();
    let mut output = Vec::new();
    let empty_rows = |count| {
        (0..count)
            .map(|_| wit_types::MeasurementRow {
                label: None,
                values: Vec::new(),
            })
            .collect()
    };

    append_measurements(empty_rows(6_000), &mut budget, &mut output).unwrap();
    let error = append_measurements(empty_rows(4_001), &mut budget, &mut output).unwrap_err();

    assert_eq!(output.len(), 6_000);
    assert_eq!(
        error,
        PluginContractError::CollectionLimit {
            field: "measurement rows",
            actual: 10_001,
            limit: MAX_PLUGIN_MEASUREMENT_ROWS,
        }
    );
}

#[test]
fn measurement_value_cardinalities_are_preflighted_before_conversion() {
    let mut budget = PluginPayloadBudget::new();
    let initial_budget = budget;
    let mut output = Vec::new();
    let rows = vec![
        wit_types::MeasurementRow {
            label: Some("valid".to_string()),
            values: vec![wit_types::Measurement {
                column: "value".to_string(),
                value: wit_types::MeasurementValue::Integer(1),
            }],
        },
        wit_types::MeasurementRow {
            label: None,
            values: (0..=MAX_PLUGIN_MEASUREMENTS_PER_ROW)
                .map(|index| wit_types::Measurement {
                    column: format!("value-{index}"),
                    value: wit_types::MeasurementValue::Missing,
                })
                .collect(),
        },
    ];

    let error = append_measurements(rows, &mut budget, &mut output).unwrap_err();

    assert!(output.is_empty());
    assert_eq!(budget, initial_budget);
    assert_eq!(
        error,
        PluginContractError::CollectionLimit {
            field: "measurements per row",
            actual: MAX_PLUGIN_MEASUREMENTS_PER_ROW + 1,
            limit: MAX_PLUGIN_MEASUREMENTS_PER_ROW,
        }
    );
}
