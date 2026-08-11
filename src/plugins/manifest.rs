use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use semver::Version;
use serde::Deserialize;
use serde_json::{Map, Number, Value};
use wasmparser::{Parser, Validator};

use crate::commands::{OpSchema, ParamSpec};

use super::{
    PLUGIN_API_VERSION, PLUGIN_MANIFEST_FILE, PLUGIN_SCHEMA_VERSION, PluginCommandContribution,
    PluginCommandTarget, PluginDescriptor, PluginOperationContribution,
};

const MAX_MANIFEST_BYTES: u64 = 256 * 1024;
const MAX_COMPONENT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_AUTHORS: usize = 32;
const MAX_OPERATIONS: usize = 128;
const MAX_HANDLERS: usize = 128;
const MAX_COMMANDS: usize = 256;
const MAX_CONTRIBUTIONS: usize = 256;
const MAX_PARAMS_PER_OPERATION: usize = 64;
const MAX_TOTAL_PARAMS: usize = 1024;
const MAX_DEFAULT_NESTING: usize = 8;
const MAX_DEFAULT_NODES: usize = 1024;
const MAX_DEFAULT_BYTES: usize = 64 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PluginManifest {
    schema_version: u32,
    id: String,
    name: String,
    version: Version,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    authors: Vec<String>,
    runtime: RuntimeManifest,
    #[serde(default)]
    operations: Vec<OperationManifest>,
    #[serde(default)]
    handlers: Vec<HandlerManifest>,
    #[serde(default)]
    commands: Vec<CommandManifest>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeManifest {
    kind: RuntimeKind,
    api_version: Version,
    path: PathBuf,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum RuntimeKind {
    WasmComponent,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OperationManifest {
    id: String,
    description: String,
    export: String,
    #[serde(default)]
    params: Vec<OperationParamManifest>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OperationParamManifest {
    name: String,
    description: String,
    required: bool,
    kind: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HandlerManifest {
    id: String,
    description: String,
    export: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CommandManifest {
    id: String,
    label: String,
    menu_path: Vec<String>,
    target: CommandTargetManifest,
    #[serde(default)]
    default_params: toml::Table,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum CommandTargetManifest {
    Operation { id: String },
    Handler { id: String },
}

#[derive(Debug, Clone, Copy)]
enum ParamKind {
    Bool,
    Float,
    Int,
    String,
    Array,
    IntArray,
    Object,
}

impl ParamKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "bool" => Ok(Self::Bool),
            "float" => Ok(Self::Float),
            "int" | "integer" => Ok(Self::Int),
            "string" => Ok(Self::String),
            "array" => Ok(Self::Array),
            "array<int>" => Ok(Self::IntArray),
            "object" => Ok(Self::Object),
            _ => Err(format!(
                "unsupported parameter kind `{value}`; expected bool, float, int, integer, string, array, array<int>, or object"
            )),
        }
    }

    fn accepts(self, value: &Value) -> bool {
        match self {
            Self::Bool => value.is_boolean(),
            Self::Float => value.is_number(),
            Self::Int => value.as_i64().is_some() || value.as_u64().is_some(),
            Self::String => value.is_string(),
            Self::Array => value.is_array(),
            Self::IntArray => value.as_array().is_some_and(|values| {
                values
                    .iter()
                    .all(|value| value.as_i64().is_some() || value.as_u64().is_some())
            }),
            Self::Object => value.is_object(),
        }
    }
}

pub(super) struct ValidatedPackage {
    pub(super) descriptor: PluginDescriptor,
    pub(super) component_path: PathBuf,
    pub(super) handler_exports: BTreeMap<String, String>,
    pub(super) operations: BTreeMap<String, PluginOperationContribution>,
    pub(super) commands: BTreeMap<String, PluginCommandContribution>,
}

pub(super) fn load_package(package_root: &Path) -> Result<ValidatedPackage, String> {
    let manifest_path = package_root.join(PLUGIN_MANIFEST_FILE);
    let metadata = fs::symlink_metadata(&manifest_path)
        .map_err(|error| format!("cannot inspect `{}`: {error}", manifest_path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "plugin manifest `{}` must be a regular file rather than a symlink",
            manifest_path.display()
        ));
    }
    if metadata.len() > MAX_MANIFEST_BYTES {
        return Err(format!(
            "plugin manifest `{}` exceeds the {MAX_MANIFEST_BYTES}-byte limit",
            manifest_path.display()
        ));
    }
    let source = String::from_utf8(read_file_at_most(
        &manifest_path,
        MAX_MANIFEST_BYTES,
        "plugin manifest",
    )?)
    .map_err(|error| {
        format!(
            "plugin manifest `{}` is not UTF-8: {error}",
            manifest_path.display()
        )
    })?;
    let manifest = toml::from_str::<PluginManifest>(&source)
        .map_err(|error| format!("invalid `{}`: {error}", manifest_path.display()))?;
    validate_manifest(package_root, manifest)
}

fn validate_manifest(
    package_root: &Path,
    manifest: PluginManifest,
) -> Result<ValidatedPackage, String> {
    if manifest.schema_version != PLUGIN_SCHEMA_VERSION {
        return Err(format!(
            "unsupported schema_version {}; this host supports exactly {}",
            manifest.schema_version, PLUGIN_SCHEMA_VERSION
        ));
    }
    validate_plugin_id(&manifest.id)?;
    validate_text("plugin name", &manifest.name, 128)?;
    if let Some(description) = &manifest.description {
        validate_text("plugin description", description, 4096)?;
    }
    validate_count("authors", manifest.authors.len(), MAX_AUTHORS)?;
    validate_count("operations", manifest.operations.len(), MAX_OPERATIONS)?;
    validate_count("handlers", manifest.handlers.len(), MAX_HANDLERS)?;
    validate_count("commands", manifest.commands.len(), MAX_COMMANDS)?;
    let contribution_count = manifest
        .operations
        .len()
        .saturating_add(manifest.handlers.len())
        .saturating_add(manifest.commands.len());
    validate_count("total contributions", contribution_count, MAX_CONTRIBUTIONS)?;
    for author in &manifest.authors {
        validate_text("plugin author", author, 256)?;
    }
    validate_api_version(&manifest.runtime.api_version)?;

    match manifest.runtime.kind {
        RuntimeKind::WasmComponent => {}
    }
    let component_path = validate_component_path(package_root, &manifest.runtime.path)?;
    if manifest.operations.is_empty() && manifest.handlers.is_empty() {
        return Err("plugin must declare at least one operation or command handler".to_string());
    }

    let mut operations = BTreeMap::new();
    let mut local_operations = BTreeSet::new();
    let mut operation_params = BTreeMap::<String, BTreeMap<String, ParamKind>>::new();
    let mut total_params = 0usize;
    for operation in manifest.operations {
        validate_local_id("operation", &operation.id)?;
        if !local_operations.insert(operation.id.clone()) {
            return Err(format!("duplicate local operation id `{}`", operation.id));
        }
        validate_text(
            &format!("description for operation `{}`", operation.id),
            &operation.description,
            4096,
        )?;
        validate_export(&operation.export)?;
        validate_count(
            &format!("parameters for operation `{}`", operation.id),
            operation.params.len(),
            MAX_PARAMS_PER_OPERATION,
        )?;
        total_params = total_params.saturating_add(operation.params.len());
        validate_count("total operation parameters", total_params, MAX_TOTAL_PARAMS)?;

        let mut params = Vec::with_capacity(operation.params.len());
        let mut param_kinds = BTreeMap::new();
        for param in operation.params {
            validate_local_id("parameter", &param.name)?;
            if param_kinds.contains_key(&param.name) {
                return Err(format!(
                    "duplicate parameter `{}` in operation `{}`",
                    param.name, operation.id
                ));
            }
            validate_text(
                &format!("description for parameter `{}`", param.name),
                &param.description,
                1024,
            )?;
            validate_text(
                &format!("kind for parameter `{}`", param.name),
                &param.kind,
                64,
            )?;
            let kind = ParamKind::parse(&param.kind)?;
            param_kinds.insert(param.name.clone(), kind);
            params.push(ParamSpec {
                name: param.name,
                description: param.description,
                required: param.required,
                kind: param.kind,
            });
        }

        let full_id = qualify(&manifest.id, &operation.id);
        operation_params.insert(operation.id, param_kinds);
        operations.insert(
            full_id.clone(),
            PluginOperationContribution {
                id: full_id.clone(),
                plugin_id: manifest.id.clone(),
                schema: OpSchema {
                    name: full_id,
                    description: operation.description,
                    params,
                },
                export: operation.export,
            },
        );
    }

    let mut handler_exports = BTreeMap::new();
    let mut local_handlers = BTreeSet::new();
    for handler in manifest.handlers {
        validate_local_id("handler", &handler.id)?;
        if !local_handlers.insert(handler.id.clone()) {
            return Err(format!("duplicate local handler id `{}`", handler.id));
        }
        validate_text(
            &format!("description for handler `{}`", handler.id),
            &handler.description,
            4096,
        )?;
        validate_export(&handler.export)?;
        handler_exports.insert(qualify(&manifest.id, &handler.id), handler.export);
    }

    let mut commands = BTreeMap::new();
    let mut local_commands = BTreeSet::new();
    let mut default_budget = DefaultBudget::new();
    for command in manifest.commands {
        let CommandManifest {
            id,
            label,
            menu_path,
            target,
            default_params: raw_default_params,
        } = command;
        validate_local_id("command", &id)?;
        if !local_commands.insert(id.clone()) {
            return Err(format!("duplicate local command id `{id}`"));
        }
        validate_text(&format!("label for command `{id}`"), &label, 128)?;
        validate_menu_path(&menu_path)?;

        let (target, declared_params) = match target {
            CommandTargetManifest::Operation { id: operation_id } => {
                validate_local_id("command operation reference", &operation_id)?;
                let Some(declared_params) = operation_params.get(&operation_id) else {
                    return Err(format!(
                        "command `{id}` references undeclared local operation `{operation_id}`"
                    ));
                };
                (
                    PluginCommandTarget::Operation {
                        operation_id: qualify(&manifest.id, &operation_id),
                    },
                    Some(declared_params),
                )
            }
            CommandTargetManifest::Handler { id: handler_id } => {
                validate_local_id("command handler reference", &handler_id)?;
                let qualified = qualify(&manifest.id, &handler_id);
                if !handler_exports.contains_key(&qualified) {
                    return Err(format!(
                        "command `{id}` references undeclared local handler `{handler_id}`"
                    ));
                }
                (
                    PluginCommandTarget::Handler {
                        handler_id: qualified,
                    },
                    None,
                )
            }
        };

        let mut default_params = Map::new();
        for (name, value) in raw_default_params {
            default_budget.charge_key(&format!("default parameter `{name}`"), name.len())?;
            let value = toml_to_json(
                value,
                &format!("default parameter `{name}`"),
                1,
                &mut default_budget,
            )?;
            if let Some(declared_params) = declared_params {
                let Some(kind) = declared_params.get(&name) else {
                    return Err(format!(
                        "command `{id}` supplies undeclared parameter `{name}` for its operation"
                    ));
                };
                if !kind.accepts(&value) {
                    return Err(format!(
                        "command `{id}` default for parameter `{name}` does not match declared kind"
                    ));
                }
            }
            default_params.insert(name, value);
        }

        let full_id = qualify(&manifest.id, &id);
        commands.insert(
            full_id.clone(),
            PluginCommandContribution {
                id: full_id,
                plugin_id: manifest.id.clone(),
                label,
                menu_path,
                target,
                default_params,
            },
        );
    }

    Ok(ValidatedPackage {
        descriptor: PluginDescriptor {
            id: manifest.id,
            name: manifest.name,
            version: manifest.version,
            description: manifest.description,
            authors: manifest.authors,
            api_version: manifest.runtime.api_version,
        },
        component_path,
        handler_exports,
        operations,
        commands,
    })
}

fn validate_api_version(plugin: &Version) -> Result<(), String> {
    let host = Version::parse(PLUGIN_API_VERSION).expect("host plugin API version is valid semver");
    let same_compatibility_line = if host.major == 0 {
        plugin.major == 0 && plugin.minor == host.minor
    } else {
        plugin.major == host.major
    };
    if !same_compatibility_line || plugin > &host {
        return Err(format!(
            "incompatible plugin api_version `{plugin}`; this host supports versions compatible with and not newer than `{host}`"
        ));
    }
    Ok(())
}

fn validate_component_path(package_root: &Path, relative: &Path) -> Result<PathBuf, String> {
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "runtime path `{}` must be a normalized relative path without traversal",
            relative.display()
        ));
    }
    if relative.extension().and_then(|value| value.to_str()) != Some("wasm") {
        return Err(format!(
            "runtime path `{}` must identify a `.wasm` component",
            relative.display()
        ));
    }

    let canonical_root = package_root.canonicalize().map_err(|error| {
        format!(
            "cannot resolve plugin package `{}`: {error}",
            package_root.display()
        )
    })?;
    let candidate = package_root.join(relative);
    let canonical_candidate = candidate.canonicalize().map_err(|error| {
        format!(
            "cannot resolve runtime component `{}`: {error}",
            candidate.display()
        )
    })?;
    if !canonical_candidate.starts_with(&canonical_root) {
        return Err(format!(
            "runtime component `{}` resolves outside its plugin package",
            relative.display()
        ));
    }
    let component_metadata = canonical_candidate.metadata().map_err(|error| {
        format!(
            "cannot inspect runtime component `{}`: {error}",
            relative.display()
        )
    })?;
    if !component_metadata.is_file() {
        return Err(format!(
            "runtime component `{}` is not a regular file",
            relative.display()
        ));
    }
    if component_metadata.len() > MAX_COMPONENT_BYTES {
        return Err(format!(
            "runtime component `{}` exceeds the {MAX_COMPONENT_BYTES}-byte limit",
            relative.display()
        ));
    }

    let bytes = read_file_at_most(
        &canonical_candidate,
        MAX_COMPONENT_BYTES,
        "runtime component",
    )?;
    if !Parser::is_component(&bytes) {
        return Err(format!(
            "runtime component `{}` must use WebAssembly Component encoding; core modules are not accepted",
            relative.display()
        ));
    }
    Validator::new().validate_all(&bytes).map_err(|error| {
        format!(
            "runtime component `{}` is not a valid WebAssembly Component: {error}",
            relative.display()
        )
    })?;
    Ok(canonical_candidate)
}

fn validate_plugin_id(id: &str) -> Result<(), String> {
    validate_text("plugin id", id, 128)?;
    if !id.contains('.') || !id.split('.').all(valid_id_segment) {
        return Err(format!(
            "plugin id `{id}` must be a lowercase reverse-domain-style identifier"
        ));
    }
    Ok(())
}

fn validate_local_id(kind: &str, id: &str) -> Result<(), String> {
    validate_text(&format!("{kind} id"), id, 128)?;
    if id.contains('.') || !valid_id_segment(id) {
        return Err(format!(
            "{kind} id `{id}` must be one lowercase identifier without dots"
        ));
    }
    Ok(())
}

fn valid_id_segment(segment: &str) -> bool {
    let mut chars = segment.chars();
    matches!(chars.next(), Some(first) if first.is_ascii_lowercase())
        && chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-')
}

fn validate_export(export: &str) -> Result<(), String> {
    validate_text("runtime export", export, 128)?;
    let mut chars = export.chars();
    if !matches!(chars.next(), Some(first) if first.is_ascii_alphabetic())
        || !chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        return Err(format!("invalid WebAssembly runtime export `{export}`"));
    }
    Ok(())
}

fn validate_menu_path(path: &[String]) -> Result<(), String> {
    if path.is_empty() || path.len() > 8 || path.first().map(String::as_str) != Some("Plugins") {
        return Err(
            "command menu_path must start with `Plugins` and contain at most 8 segments"
                .to_string(),
        );
    }
    for segment in path {
        validate_text("command menu path segment", segment, 64)?;
        if segment.contains('>') {
            return Err("command menu path segments cannot contain `>`".to_string());
        }
    }
    Ok(())
}

fn validate_text(label: &str, value: &str, maximum: usize) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{label} cannot be empty"));
    }
    if value.len() > maximum {
        return Err(format!("{label} cannot exceed {maximum} bytes"));
    }
    if value.chars().any(char::is_control) {
        return Err(format!("{label} cannot contain control characters"));
    }
    Ok(())
}

fn read_file_at_most(path: &Path, maximum: u64, kind: &str) -> Result<Vec<u8>, String> {
    let file = File::open(path)
        .map_err(|error| format!("cannot open {kind} `{}`: {error}", path.display()))?;
    let length = file
        .metadata()
        .map_err(|error| format!("cannot inspect {kind} `{}`: {error}", path.display()))?
        .len();
    if length > maximum {
        return Err(format!(
            "{kind} `{}` exceeds the {maximum}-byte limit",
            path.display()
        ));
    }
    let mut bytes = Vec::with_capacity(length.min(maximum) as usize);
    file.take(maximum.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read {kind} `{}`: {error}", path.display()))?;
    if bytes.len() as u64 > maximum {
        return Err(format!(
            "{kind} `{}` exceeds the {maximum}-byte limit",
            path.display()
        ));
    }
    Ok(bytes)
}

fn validate_count(label: &str, actual: usize, maximum: usize) -> Result<(), String> {
    if actual > maximum {
        return Err(format!(
            "plugin declares {actual} {label}, exceeding the limit of {maximum}"
        ));
    }
    Ok(())
}

fn qualify(plugin_id: &str, local_id: &str) -> String {
    format!("{plugin_id}.{local_id}")
}

struct DefaultBudget {
    nodes: usize,
    bytes: usize,
}

impl DefaultBudget {
    fn new() -> Self {
        Self { nodes: 0, bytes: 0 }
    }

    fn charge(&mut self, context: &str, bytes: usize) -> Result<(), String> {
        self.nodes = self.nodes.saturating_add(1);
        self.bytes = self.bytes.saturating_add(bytes);
        if self.nodes > MAX_DEFAULT_NODES {
            return Err(format!(
                "{context} exceeds the aggregate limit of {MAX_DEFAULT_NODES} default-value nodes"
            ));
        }
        if self.bytes > MAX_DEFAULT_BYTES {
            return Err(format!(
                "{context} exceeds the aggregate {MAX_DEFAULT_BYTES}-byte default-value limit"
            ));
        }
        Ok(())
    }

    fn charge_key(&mut self, context: &str, bytes: usize) -> Result<(), String> {
        self.bytes = self.bytes.saturating_add(bytes);
        if self.bytes > MAX_DEFAULT_BYTES {
            return Err(format!(
                "{context} exceeds the aggregate {MAX_DEFAULT_BYTES}-byte default-value limit"
            ));
        }
        Ok(())
    }
}

fn toml_to_json(
    value: toml::Value,
    context: &str,
    depth: usize,
    budget: &mut DefaultBudget,
) -> Result<Value, String> {
    if depth > MAX_DEFAULT_NESTING {
        return Err(format!(
            "{context} exceeds the maximum default-value nesting depth of {MAX_DEFAULT_NESTING}"
        ));
    }
    match value {
        toml::Value::String(value) => {
            budget.charge(context, value.len())?;
            Ok(Value::String(value))
        }
        toml::Value::Integer(value) => {
            budget.charge(context, std::mem::size_of::<i64>())?;
            Ok(Value::Number(value.into()))
        }
        toml::Value::Float(value) => {
            budget.charge(context, std::mem::size_of::<f64>())?;
            Number::from_f64(value)
                .map(Value::Number)
                .ok_or_else(|| format!("{context} must contain a finite number"))
        }
        toml::Value::Boolean(value) => {
            budget.charge(context, std::mem::size_of::<bool>())?;
            Ok(Value::Bool(value))
        }
        toml::Value::Array(values) => {
            budget.charge(context, 0)?;
            values
                .into_iter()
                .enumerate()
                .map(|(index, value)| {
                    toml_to_json(value, &format!("{context}[{index}]"), depth + 1, budget)
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Value::Array)
        }
        toml::Value::Table(values) => {
            budget.charge(context, 0)?;
            values
                .into_iter()
                .map(|(key, value)| {
                    budget.charge_key(context, key.len())?;
                    let value =
                        toml_to_json(value, &format!("{context}.{key}"), depth + 1, budget)?;
                    Ok((key, value))
                })
                .collect::<Result<Map<_, _>, _>>()
                .map(Value::Object)
        }
        toml::Value::Datetime(_) => Err(format!("{context} cannot contain a TOML datetime")),
    }
}
