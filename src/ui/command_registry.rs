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
        "file.new" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            Some(
                json!({"width": 512, "height": 512, "slices": 1, "channels": 1, "fill": 0.0, "pixelType": "f32"}),
            ),
            Some("Create a new blank image via a native utility dialog."),
        ),
        "file.open" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Open files using the native platform file picker."),
        ),
        "file.save_as" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("Save the active image to a new path using the native file picker."),
        ),
        "file.close" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Close the active ImageJ-like window."),
        ),
        "file.save" | "file.revert" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Persist or restore the active image."),
        ),
        "file.import.image" | "file.import.raw" | "file.import.url" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Import images using sequence, raw, or URL workflows."),
        ),
        "file.export.image" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("Export the active image using the save-as flow."),
        ),
        "file.export.results" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Export the shared results table as CSV or JSON."),
        ),
        "file.quit" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Exit the image application."),
        ),
        "file.open_recent.none" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Open recently used files from persisted desktop state."),
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
        "edit.undo" | "edit.redo" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Undo or redo committed image edits in the active viewer."),
        ),
        "edit.cut" | "edit.copy" | "edit.paste" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Use the internal image clipboard for ROI or frame data."),
        ),
        "edit.options.appearance" | "edit.options.memory" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Open informational utility dialogs for appearance and memory."),
        ),
        "image.zoom.in"
        | "image.zoom.out"
        | "image.zoom.reset"
        | "image.zoom.original"
        | "image.zoom.view100"
        | "image.zoom.to_selection"
        | "image.zoom.scale_to_fit"
        | "image.zoom.set"
        | "image.zoom.maximize" => CommandMetadata::with(
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
        | "launcher.tool.angle"
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
        "launcher.tool.rect.mode.rectangle"
        | "launcher.tool.rect.mode.rounded"
        | "launcher.tool.rect.mode.rotated"
        | "launcher.tool.oval.mode.oval"
        | "launcher.tool.oval.mode.ellipse"
        | "launcher.tool.oval.mode.brush"
        | "launcher.tool.line.mode.straight"
        | "launcher.tool.line.mode.segmented"
        | "launcher.tool.line.mode.freehand"
        | "launcher.tool.line.mode.arrow"
        | "launcher.tool.point.mode.point"
        | "launcher.tool.point.mode.multipoint"
        | "tool.dropper.palette.white_black"
        | "tool.dropper.palette.black_white"
        | "tool.dropper.palette.red"
        | "tool.dropper.palette.green"
        | "tool.dropper.palette.blue"
        | "tool.dropper.palette.yellow"
        | "tool.dropper.palette.cyan"
        | "tool.dropper.palette.magenta"
        | "tool.dropper.palette.foreground"
        | "tool.dropper.palette.background"
        | "tool.dropper.palette.colors"
        | "tool.dropper.palette.color_picker"
        | "viewer.roi.clear"
        | "viewer.roi.abort"
        | "viewer.roi.select_next" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("ImageJ-style interaction command routed to the viewer shell."),
        ),
        "process.smooth" | "process.gaussian" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"sigma": 1.0})),
            Some("Processes the active image using gaussian blur/ smooth path."),
        ),
        "image.type.8bit" | "image.type.16bit" | "image.type.32bit" | "image.type.rgb" => {
            CommandMetadata::with(
                CommandScope::Viewer,
                true,
                false,
                true,
                None,
                Some("Convert the active image pixel type or channel layout."),
            )
        }
        "image.adjust.brightness" | "image.adjust.threshold" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Applies automatic normalization or thresholding to the active image."),
        ),
        "image.adjust.size" | "image.adjust.canvas" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("Open utility dialogs for X/Y resize and canvas size changes."),
        ),
        "image.stacks.next" | "image.stacks.previous" | "image.stacks.set" => {
            CommandMetadata::with(
                CommandScope::Viewer,
                true,
                true,
                true,
                None,
                Some("Image stack navigation is implemented in the viewer shell."),
            )
        }
        "process.binary.make" | "process.binary.erode" | "process.binary.dilate" => {
            CommandMetadata::with(
                CommandScope::Viewer,
                true,
                false,
                true,
                None,
                Some("Runs binary threshold or binary morphology on the active image."),
            )
        }
        "process.sharpen" | "process.find_edges" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Run sharpen or edge-detection filters on the active image."),
        ),
        "analyze.measure"
        | "analyze.histogram"
        | "analyze.set_measurements"
        | "analyze.analyze_particles"
        | "analyze.plot_profile"
        | "analyze.tools.roi_manager"
        | "analyze.tools.results" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Analyze the active image and surface results in shared utility windows."),
        ),
        "plugins.commands.find"
        | "help.about"
        | "help.docs"
        | "help.shortcuts"
        | "window.tile"
        | "window.cascade" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Desktop utility commands are implemented using native utility windows."),
        ),
        "plugins.macros.run"
        | "plugins.macros.record"
        | "plugins.macros.install"
        | "plugins.utilities.startup" => CommandMetadata::with(
            CommandScope::Both,
            true,
            false,
            false,
            None,
            Some("Macro/plugin compatibility is intentionally deferred in this pass."),
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
            "launcher.tool.angle",
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

    #[test]
    fn viewer_edit_and_file_commands_require_an_image() {
        for command in [
            "file.save",
            "file.save_as",
            "file.revert",
            "edit.undo",
            "edit.redo",
        ] {
            let metadata = metadata(command);
            assert!(metadata.implemented, "{command} should be implemented");
            assert!(metadata.requires_image, "{command} should require an image");
            assert!(metadata.scope.contains("viewer-1"));
            assert!(!metadata.scope.contains("main"));
        }
    }

    #[test]
    fn manifest_core_commands_are_classified() {
        for command in [
            "file.new",
            "file.import.image",
            "file.export.results",
            "image.type.rgb",
            "image.adjust.size",
            "process.sharpen",
            "analyze.analyze_particles",
            "analyze.tools.results",
            "plugins.commands.find",
            "help.shortcuts",
        ] {
            assert!(
                metadata(command).implemented,
                "{command} should be implemented or explicitly handled"
            );
        }
    }
}
