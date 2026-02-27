use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, from_str, json};
use std::sync::OnceLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CommandScope {
    Launcher,
    Viewer,
    Both,
}

impl CommandScope {
    pub fn contains(&self, window_label: &str) -> bool {
        match self {
            Self::Both => true,
            Self::Launcher => window_label == "main",
            Self::Viewer => window_label.starts_with("viewer-"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CommandMetadata {
    pub scope: CommandScope,
    pub implemented: bool,
    pub frontend_only: bool,
    pub requires_image: bool,
    pub default_params: Option<Value>,
    pub notes: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandCatalogEntry {
    pub id: String,
    pub label: String,
    pub shortcut: Option<String>,
    pub scope: CommandScope,
    pub implemented: bool,
    pub frontend_only: bool,
    pub requires_image: bool,
    pub enabled: bool,
    #[serde(default)]
    pub default_params: Option<Value>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandCatalog {
    pub version: String,
    pub entries: Vec<CommandCatalogEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CommandExecuteRequest {
    pub command_id: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandExecuteResult {
    pub status: CommandExecuteStatus,
    pub message: String,
    #[serde(default)]
    pub payload: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CommandExecuteStatus {
    Ok,
    Unimplemented,
    Blocked,
}

impl CommandExecuteResult {
    pub fn ok(message: impl Into<String>) -> Self {
        Self {
            status: CommandExecuteStatus::Ok,
            message: message.into(),
            payload: None,
        }
    }

    pub fn with_payload(message: impl Into<String>, payload: Value) -> Self {
        Self {
            status: CommandExecuteStatus::Ok,
            message: message.into(),
            payload: Some(payload),
        }
    }

    pub fn unimplemented(message: impl Into<String>) -> Self {
        Self {
            status: CommandExecuteStatus::Unimplemented,
            message: message.into(),
            payload: None,
        }
    }

    pub fn blocked(message: impl Into<String>) -> Self {
        Self {
            status: CommandExecuteStatus::Blocked,
            message: message.into(),
            payload: None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct MenuManifestTopLevel {
    #[serde(rename = "id")]
    _id: String,
    items: Vec<MenuManifestItem>,
}

#[derive(Debug, Deserialize)]
struct MenuManifestItem {
    #[serde(rename = "type")]
    kind: String,
    id: Option<String>,
    label: Option<String>,
    command: Option<String>,
    shortcut: Option<String>,
    enabled: Option<bool>,
    items: Option<Vec<MenuManifestItem>>,
}

#[derive(Debug, Clone)]
struct MenuManifestCommand {
    id: String,
    label: String,
    shortcut: Option<String>,
    enabled: bool,
}

impl CommandMetadata {
    const fn unsupported() -> Self {
        Self {
            scope: CommandScope::Both,
            implemented: false,
            frontend_only: false,
            requires_image: false,
            default_params: None,
            notes: None,
        }
    }

    fn with(
        scope: CommandScope,
        implemented: bool,
        frontend_only: bool,
        requires_image: bool,
        default_params: Option<Value>,
        notes: Option<&'static str>,
    ) -> Self {
        Self {
            scope,
            implemented,
            frontend_only,
            requires_image,
            default_params,
            notes,
        }
    }
}

pub fn metadata(command_id: &str) -> CommandMetadata {
    match command_id {
        "file.open" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Open files using the native platform file picker."),
        ),
        "file.close" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Close the active ImageJ-like window."),
        ),
        "file.quit" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Exit the image application."),
        ),
        "window.next" => CommandMetadata::with(
            CommandScope::Both,
            true,
            false,
            false,
            None,
            Some("Cycles active image windows."),
        ),
        "window.previous" => CommandMetadata::with(
            CommandScope::Both,
            true,
            false,
            false,
            None,
            Some("Cycles active image windows in reverse order."),
        ),
        "image.zoom.in" | "image.zoom.out" | "image.zoom.reset" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("Canvas navigation controls are implemented in the shell."),
        ),
        "launcher.tool.rect"
        | "launcher.tool.oval"
        | "launcher.tool.poly"
        | "launcher.tool.free"
        | "launcher.tool.line"
        | "launcher.tool.point"
        | "launcher.tool.wand"
        | "launcher.tool.text"
        | "launcher.tool.zoom"
        | "launcher.tool.hand"
        | "launcher.tool.dropper"
        | "launcher.tool.custom1"
        | "launcher.tool.custom2"
        | "launcher.tool.custom3"
        | "launcher.tool.more" => {
            let notes = match command_id {
                "launcher.tool.hand" => {
                    "Selects the Hand tool; drag in viewer to pan the active image."
                }
                "launcher.tool.zoom" => {
                    "Selects the Zoom tool; click to zoom in and Shift-click/right-click to zoom out."
                }
                "launcher.tool.point" => {
                    "Selects the Point tool; click in viewer to pin coordinate/value telemetry."
                }
                _ => {
                    "Tool selection is implemented; tool-specific drawing/edit behavior is not implemented yet."
                }
            };
            CommandMetadata::with(CommandScope::Both, true, true, false, None, Some(notes))
        }
        "process.smooth" | "process.gaussian" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"sigma": 1.0})),
            Some("Processes the active image using gaussian blur/ smooth path."),
        ),
        "analyze.measure" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Computes measurement summary for the active image."),
        ),
        _ => CommandMetadata::unsupported(),
    }
}

pub fn command_catalog() -> CommandCatalog {
    CommandCatalog {
        version: "1.0.0".to_string(),
        entries: manifest_commands()
            .iter()
            .map(|entry| {
                let metadata = metadata(&entry.id);
                CommandCatalogEntry {
                    id: entry.id.clone(),
                    label: entry.label.clone(),
                    shortcut: entry.shortcut.clone(),
                    scope: metadata.scope,
                    implemented: metadata.implemented,
                    frontend_only: metadata.frontend_only,
                    requires_image: metadata.requires_image,
                    enabled: entry.enabled,
                    default_params: metadata.default_params,
                    notes: metadata.notes.map(std::string::ToString::to_string),
                }
            })
            .collect(),
    }
}

fn manifest_commands() -> &'static Vec<MenuManifestCommand> {
    static COMMANDS: OnceLock<Vec<MenuManifestCommand>> = OnceLock::new();
    COMMANDS.get_or_init(|| {
        let raw: Vec<MenuManifestTopLevel> =
            from_str(include_str!("menu/imagej-menu-manifest.json"))
                .expect("failed to parse menu manifest");

        let mut entries = Vec::new();
        for top in raw {
            append_items(&top.items, &mut entries);
        }
        entries
    })
}

fn append_items(items: &[MenuManifestItem], output: &mut Vec<MenuManifestCommand>) {
    for item in items {
        match item.kind.as_str() {
            "item" => {
                let id = item
                    .command
                    .clone()
                    .or_else(|| item.id.clone())
                    .unwrap_or_else(|| String::from(""));
                let label = item.label.clone().unwrap_or_else(|| id.clone());
                if id.is_empty() {
                    continue;
                }

                output.push(MenuManifestCommand {
                    id,
                    label,
                    shortcut: item.shortcut.clone(),
                    enabled: item.enabled.unwrap_or(true),
                });
            }
            "submenu" => {
                if let Some(children) = item.items.as_deref() {
                    append_items(children, output);
                }
            }
            "separator" => {}
            _ => {
                if let Some(children) = item.items.as_deref() {
                    append_items(children, output);
                }
            }
        }
    }
}

pub fn merge_params(command_id: &str, params: Option<Value>) -> Value {
    let metadata = metadata(command_id);
    let default = metadata
        .default_params
        .unwrap_or_else(|| Value::Object(Map::new()));
    match params {
        Some(Value::Object(map)) if !map.is_empty() => Value::Object(map),
        Some(Value::Null) | None => default,
        Some(other) => other,
    }
}

#[cfg(test)]
mod tests {
    use super::{CommandScope, metadata};

    #[test]
    fn command_scope_contains_expected_window_labels() {
        assert!(CommandScope::Launcher.contains("main"));
        assert!(!CommandScope::Launcher.contains("viewer-1"));

        assert!(CommandScope::Viewer.contains("viewer-1"));
        assert!(!CommandScope::Viewer.contains("main"));

        assert!(CommandScope::Both.contains("main"));
        assert!(CommandScope::Both.contains("viewer-42"));
    }

    #[test]
    fn launcher_tool_commands_are_marked_implemented_and_frontend_only() {
        for command in [
            "launcher.tool.rect",
            "launcher.tool.oval",
            "launcher.tool.poly",
            "launcher.tool.free",
            "launcher.tool.line",
            "launcher.tool.point",
            "launcher.tool.wand",
            "launcher.tool.text",
            "launcher.tool.zoom",
            "launcher.tool.hand",
            "launcher.tool.dropper",
        ] {
            let metadata = metadata(command);
            assert!(metadata.implemented, "{command} should be implemented");
            assert!(metadata.frontend_only, "{command} should be frontend-only");
            assert!(metadata.scope.contains("main"));
            assert!(metadata.scope.contains("viewer-1"));
        }
    }
}
