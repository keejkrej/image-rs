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
                json!({"width": 512, "height": 512, "slices": 1, "channels": 1, "frames": 1, "fill": 0.0, "pixelType": "f32"}),
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
        "window.next" | "window.put_behind" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Cycles active image windows."),
        ),
        "window.main" | "window.show_all" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("ImageJ-style window management command handled by the native shell."),
        ),
        "window.previous" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
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
        "edit.invert" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Invert active image intensities using ImageJ-style ranges."),
        ),
        "edit.options.appearance" | "edit.options.memory" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Open informational utility dialogs for appearance and memory."),
        ),
        "edit.options.line_width" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            Some(json!({"width": 1.0})),
            Some("Set the default ImageJ-style line selection width."),
        ),
        "image.zoom.in"
        | "image.zoom.out"
        | "image.zoom.reset"
        | "image.zoom.original"
        | "image.zoom.view100"
        | "image.zoom.to_selection"
        | "image.zoom.scale_to_fit"
        | "image.zoom.maximize" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("Canvas navigation controls are implemented in the shell."),
        ),
        "image.zoom.set" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            Some(json!({"zoom_percent": null, "x": null, "y": null})),
            Some("Set ImageJ-style exact canvas zoom and optional image center."),
        ),
        "image.lookup.apply_lut"
        | "image.lookup.invert_lut"
        | "image.lookup.fire"
        | "image.lookup.grays"
        | "image.lookup.ice"
        | "image.lookup.spectrum"
        | "image.lookup.rgb332"
        | "image.lookup.red"
        | "image.lookup.green"
        | "image.lookup.blue"
        | "image.lookup.cyan"
        | "image.lookup.magenta"
        | "image.lookup.yellow"
        | "image.lookup.red_green" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("Apply an ImageJ-style display lookup table to the active viewer."),
        ),
        "image.overlay.add_selection"
        | "image.overlay.flatten"
        | "image.overlay.from_roi_manager"
        | "image.overlay.hide"
        | "image.overlay.labels"
        | "image.overlay.show"
        | "image.overlay.remove"
        | "image.overlay.options"
        | "image.overlay.to_roi_manager"
        | "image.overlay.toggle"
        | "image.overlay.list"
        | "image.overlay.measure" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("Manage ImageJ-style overlay ROI elements for the active image."),
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
        "process.smooth" | "process.gaussian" | "process.filters.gaussian" => {
            CommandMetadata::with(
                CommandScope::Viewer,
                true,
                false,
                true,
                Some(json!({"sigma": 1.0})),
                Some("Processes the active image using gaussian blur/ smooth path."),
            )
        }
        "process.enhance_contrast" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"saturated_percent": 0.35, "normalize": true})),
            Some("Run ImageJ Process/Enhance Contrast saturated-tail contrast stretching."),
        ),
        "process.subtract_background" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({
                "radius": 50.0,
                "light_background": true,
                "create_background": false
            })),
            Some("Run ImageJ Process/Subtract Background using a grayscale background estimate."),
        ),
        "process.filters.gaussian_3d" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"sigma": 2.0})),
            Some("Run ImageJ Process/Filters 3D Gaussian blur on the active image."),
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
        "image.adjust.window_level" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"low": 0.0, "high": 1.0})),
            Some("Apply an ImageJ-style window/level clamp to the active image."),
        ),
        "image.adjust.size" | "image.adjust.canvas" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("Open utility dialogs for X/Y resize and canvas size changes."),
        ),
        "image.adjust.line_width" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            Some(json!({"width": 1.0})),
            Some("Set the default ImageJ-style line selection width."),
        ),
        "image.adjust.coordinates" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"x_unit": "pixel", "y_unit": "<same as x unit>"})),
            Some("Update ImageJ-style coordinate calibration metadata."),
        ),
        "image.show_info" | "image.properties" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("Return ImageJ-style active image information and metadata."),
        ),
        "image.crop" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Crop the active image to the selected ROI bounds."),
        ),
        "image.duplicate" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("Duplicate the active image into a new viewer."),
        ),
        "image.rename" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            Some(json!({"title": ""})),
            Some("Rename the active image viewer title."),
        ),
        "image.scale" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"x_scale": 0.5, "y_scale": 0.5})),
            Some("Scale the active image by X/Y factors."),
        ),
        "image.stacks.next" | "image.stacks.previous" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("Image stack navigation is implemented in the viewer shell."),
        ),
        "image.stacks.set" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            Some(json!({"slice": null, "channel": null, "frame": null})),
            Some("Set the active ImageJ-style stack or hyperstack position."),
        ),
        "image.stacks.images_to_stack" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Combine open compatible 2D image windows into an ImageJ-style Z stack."),
        ),
        "image.stacks.add_slice" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"index": null, "fill": 0.0})),
            Some("Add a blank slice to the active image stack."),
        ),
        "image.stacks.delete_slice" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"index": null})),
            Some("Delete the active slice from the image stack."),
        ),
        "image.stacks.make_substack" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"slices": null, "indices": null})),
            Some("Create an ImageJ-style substack from selected Z slices."),
        ),
        "image.stacks.reslice" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"start": "top"})),
            Some("Create an ImageJ-style orthogonal reslice stack."),
        ),
        "image.stacks.stack_to_images" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("Split the active ImageJ-style Z stack into separate image windows."),
        ),
        "image.stacks.z_project" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"method": "average", "start": 0, "stop": null})),
            Some("Project the active Z stack using ImageJ-style Z Project."),
        ),
        "image.stacks.plot_z_profile" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({
                "left": null,
                "top": null,
                "width": null,
                "height": null,
                "min_threshold": null,
                "max_threshold": null
            })),
            Some("Plot the ImageJ-style mean gray-value profile along Z."),
        ),
        "image.stacks.plot_xy_profile" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            Some(json!({
                "left": null,
                "top": null,
                "width": null,
                "height": null,
                "x0": null,
                "y0": null,
                "x1": null,
                "y1": null,
                "vertical": false
            })),
            Some("Plot ImageJ-style XY profiles for every Z slice into the Results table."),
        ),
        "image.stacks.measure_stack" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("Measure every Z slice in the active ImageJ-style stack into the Results table."),
        ),
        "image.stacks.statistics" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({
                "left": null,
                "top": null,
                "width": null,
                "height": null,
                "min_threshold": null,
                "max_threshold": null
            })),
            Some("Compute ImageJ-style statistics for the active Z stack."),
        ),
        "image.stacks.make_montage" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({
                "columns": null,
                "rows": null,
                "scale": 1.0,
                "first": 0,
                "last": null,
                "increment": 1,
                "border_width": 0,
                "fill": 0.0
            })),
            Some("Create an ImageJ-style montage from the active Z stack."),
        ),
        "image.stacks.montage_to_stack" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"columns": null, "rows": null, "border_width": 0})),
            Some("Convert an ImageJ-style montage image back to a Z stack."),
        ),
        "image.stacks.grouped_z_project" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"method": "average", "group_size": 2})),
            Some("Project adjacent fixed-size Z groups into a shorter ImageJ-style stack."),
        ),
        "image.stacks.combine" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            Some(json!({"vertical": false, "fill": 0.0})),
            Some("Combine two open ImageJ-style image stacks horizontally or vertically."),
        ),
        "image.stacks.concatenate" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            Some(json!({"fill": 0.0})),
            Some("Concatenate open ImageJ-style image stacks along Z."),
        ),
        "image.stacks.insert" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            Some(json!({"x": 0, "y": 0, "source": null, "destination": null})),
            Some("Insert one ImageJ-style image or stack into another at an X/Y offset."),
        ),
        "image.stacks.reduce" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"factor": 2})),
            Some("Reduce the active Z stack by keeping every Nth slice."),
        ),
        "image.stacks.reverse" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"axis": "z"})),
            Some("Reverse the active image stack along Z."),
        ),
        "image.stacks.set_label" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            Some(json!({"label": "", "slice": null})),
            Some("Set the ImageJ-style label for the active Z slice."),
        ),
        "image.stacks.remove_slice_labels" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("Remove ImageJ-style labels from all slices in the active stack."),
        ),
        "image.hyperstacks.stack_to_hyperstack" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({
                "channels": 1,
                "slices": null,
                "frames": 1,
                "order": "czt"
            })),
            Some("Convert the active ImageJ-style stack into a C/Z/T hyperstack."),
        ),
        "image.hyperstacks.new" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            Some(json!({
                "width": 400,
                "height": 300,
                "slices": 4,
                "channels": 3,
                "frames": 5,
                "fill": 0.0,
                "pixelType": "f32"
            })),
            Some("Create a new blank ImageJ-style C/Z/T hyperstack."),
        ),
        "image.hyperstacks.hyperstack_to_stack" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Flatten the active ImageJ-style hyperstack into a linear Z stack."),
        ),
        "image.hyperstacks.reduce_dimensionality" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({
                "keep_channels": true,
                "keep_slices": true,
                "keep_frames": true,
                "channel": null,
                "z": null,
                "time": null
            })),
            Some("Reduce an ImageJ-style hyperstack by keeping or collapsing C/Z/T axes."),
        ),
        "image.hyperstacks.make_subset" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({
                "channels": null,
                "slices": null,
                "frames": null
            })),
            Some("Create an ImageJ-style hyperstack subset from C/Z/T range strings."),
        ),
        "image.transform.flip_horizontal"
        | "image.transform.flip_vertical"
        | "image.transform.flip_z"
        | "image.transform.rotate_right"
        | "image.transform.rotate_left" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Transform the active image with ImageJ-style flip or 90-degree rotation."),
        ),
        "image.transform.rotate" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(
                json!({"angle": 15.0, "fill": 0.0, "interpolation": "bilinear", "enlarge": false}),
            ),
            Some("Rotate the active image by an arbitrary angle."),
        ),
        "image.transform.translate" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"x": 15.0, "y": 15.0, "fill": 0.0, "interpolation": "nearest"})),
            Some("Translate the active image by X/Y pixel offsets."),
        ),
        "image.transform.bin" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"x": 2, "y": 2, "z": 1, "method": "average"})),
            Some("Reduce the active image dimensions using ImageJ-style binning."),
        ),
        "image.transform.image_to_results" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("Convert the active image slice or selection to an ImageJ-style results table."),
        ),
        "image.transform.results_to_image" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            false,
            None,
            Some("Convert numeric results table columns to a float image."),
        ),
        "process.binary.make" | "process.binary.convert_mask" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"method": "default", "background": "default"})),
            Some("Convert the active image to an ImageJ-style binary mask."),
        ),
        "process.binary.erode"
        | "process.binary.dilate"
        | "process.binary.open"
        | "process.binary.close"
        | "process.binary.median"
        | "process.binary.outline"
        | "process.binary.fill_holes"
        | "process.binary.skeletonize"
        | "process.binary.distance_map"
        | "process.binary.ultimate_points"
        | "process.binary.watershed"
        | "process.binary.voronoi" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Runs binary threshold or binary morphology on the active image."),
        ),
        "process.binary.options" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            Some(json!({"iterations": 1, "count": 1})),
            Some("Open ImageJ-style Process/Binary options for morphology commands."),
        ),
        "process.noise.add" | "process.noise.specified" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"sigma": 25.0, "seed": 0})),
            Some("Add ImageJ-style Gaussian noise to the active image."),
        ),
        "process.noise.salt_pepper" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"percent": 0.05, "seed": 0})),
            Some("Apply ImageJ-style salt-and-pepper noise to an integer-backed active image."),
        ),
        "process.noise.despeckle" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"radius": 1.0})),
            Some("Apply ImageJ-style despeckle median filtering to the active image."),
        ),
        "process.noise.remove_nans" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"radius": 2.0})),
            Some("Replace NaN pixels using ImageJ-style neighborhood medians."),
        ),
        "process.noise.remove_outliers" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"radius": 2.0, "threshold": 50.0, "which": "bright"})),
            Some("Replace bright or dark outliers using ImageJ-style neighborhood medians."),
        ),
        "process.math.add" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"operation": "add", "value": 25.0 / 255.0})),
            Some("Run ImageJ-style per-pixel math on the active image."),
        ),
        "process.math.subtract" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"operation": "subtract", "value": 25.0 / 255.0})),
            Some("Run ImageJ-style per-pixel math on the active image."),
        ),
        "process.math.multiply" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"operation": "multiply", "value": 1.25})),
            Some("Run ImageJ-style per-pixel math on the active image."),
        ),
        "process.math.divide" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"operation": "divide", "value": 1.25})),
            Some("Run ImageJ-style per-pixel math on the active image."),
        ),
        "process.math.and" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"operation": "and", "value": "11110000"})),
            Some("Run ImageJ-style bitwise math on integer-backed active images."),
        ),
        "process.math.or" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"operation": "or", "value": "11110000"})),
            Some("Run ImageJ-style bitwise math on integer-backed active images."),
        ),
        "process.math.xor" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"operation": "xor", "value": "11110000"})),
            Some("Run ImageJ-style bitwise math on integer-backed active images."),
        ),
        "process.math.min" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"operation": "min", "value": 0.0})),
            Some("Run ImageJ-style per-pixel math on the active image."),
        ),
        "process.math.max" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"operation": "max", "value": 1.0})),
            Some("Run ImageJ-style per-pixel math on the active image."),
        ),
        "process.math.gamma" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"operation": "gamma", "value": 0.5})),
            Some("Run ImageJ-style per-pixel math on the active image."),
        ),
        "process.math.set" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"operation": "set", "value": 25.0 / 255.0})),
            Some("Run ImageJ-style per-pixel math on the active image."),
        ),
        "process.math.nan_background" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"lower": 0.0, "upper": 1.0})),
            Some("Set pixels outside the threshold range to NaN for 32-bit float images."),
        ),
        "process.math.log"
        | "process.math.exp"
        | "process.math.square"
        | "process.math.sqrt"
        | "process.math.reciprocal"
        | "process.math.abs" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"operation": command_id.strip_prefix("process.math.").unwrap_or("")})),
            Some("Run ImageJ-style per-pixel math on the active image."),
        ),
        "process.shadows.north"
        | "process.shadows.northeast"
        | "process.shadows.east"
        | "process.shadows.southeast"
        | "process.shadows.south"
        | "process.shadows.southwest"
        | "process.shadows.west"
        | "process.shadows.northwest" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"direction": command_id.strip_prefix("process.shadows.").unwrap_or("")})),
            Some("Run ImageJ Process/Shadows directional 3x3 convolution."),
        ),
        "process.shadows.demo" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"iterations": 20})),
            Some("Create an ImageJ Process/Shadows eight-direction demo stack."),
        ),
        "process.filters.mean"
        | "process.filters.minimum"
        | "process.filters.maximum"
        | "process.filters.variance" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(
                json!({"filter": command_id.strip_prefix("process.filters.").unwrap_or(""), "radius": 2.0}),
            ),
            Some("Run ImageJ Process/Filters rank filters on the active image."),
        ),
        "process.filters.top_hat" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(
                json!({"filter": "top_hat", "radius": 2.0, "light_background": false, "dont_subtract": false}),
            ),
            Some("Run ImageJ Process/Filters top-hat background subtraction on the active image."),
        ),
        "process.filters.median_3d"
        | "process.filters.mean_3d"
        | "process.filters.minimum_3d"
        | "process.filters.maximum_3d"
        | "process.filters.variance_3d" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({
                "x_radius": 2.0,
                "y_radius": 2.0,
                "z_radius": 2.0
            })),
            Some("Run ImageJ Process/Filters 3D rank filters on the active image."),
        ),
        "process.filters.median" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"radius": 2.0})),
            Some("Run ImageJ Process/Filters median filter on the active image."),
        ),
        "process.filters.convolve" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({
                "width": 3,
                "height": 3,
                "kernel": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "normalize": false
            })),
            Some("Run ImageJ Process/Filters custom convolution on the active image."),
        ),
        "process.filters.unsharp_mask" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"sigma": 1.0, "weight": 0.6})),
            Some("Run ImageJ Process/Filters unsharp mask on the active image."),
        ),
        "process.filters.show_circular_masks" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Create the ImageJ rank-filter circular mask demonstration stack."),
        ),
        "process.repeat_command" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Repeat the last non-frontend command, matching ImageJ Process/Repeat Command."),
        ),
        "process.sharpen" | "process.find_edges" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Run sharpen or edge-detection filters on the active image."),
        ),
        "process.find_maxima" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({
                "prominence": 0.0,
                "strict": true,
                "exclude_edges": false,
                "light_background": false
            })),
            Some("Find local maxima and return an ImageJ-style maxima mask."),
        ),
        "process.fft.swap_quadrants" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Run ImageJ Process/FFT quadrant swapping on the active image."),
        ),
        "process.fft.fft" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Run ImageJ Process/FFT power spectrum on the active image."),
        ),
        "process.fft.bandpass" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({
                "filter_large": 40.0,
                "filter_small": 3.0,
                "suppress_stripes": "none",
                "tolerance": 5.0,
                "autoscale": true
            })),
            Some("Run ImageJ Process/FFT bandpass filtering on the active image."),
        ),
        "process.fft.make_circular_selection" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            Some(json!({"radius": null})),
            Some("Create an ImageJ Process/FFT centered circular selection on the active image."),
        ),
        "analyze.measure"
        | "analyze.analyze_particles"
        | "analyze.tools.roi_manager"
        | "analyze.tools.results" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Analyze the active image and surface results in shared utility windows."),
        ),
        "analyze.tools.save_xy_coordinates" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            Some(json!({"background": null, "invert_y": false})),
            Some(
                "Write ImageJ-style XY coordinate/value rows for the active image or selection into the results table.",
            ),
        ),
        "analyze.label" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            None,
            Some("Add an ImageJ-style numeric label overlay to the selected ROI."),
        ),
        "analyze.plot_profile" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({
                "left": null,
                "top": null,
                "width": null,
                "height": null,
                "x0": null,
                "y0": null,
                "x1": null,
                "y1": null,
                "vertical": false
            })),
            Some("Compute an ImageJ-style profile plot from a line or rectangular selection."),
        ),
        "analyze.histogram" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"bins": 256, "min": null, "max": null, "stack": false})),
            Some("Compute an ImageJ-style histogram for the active image."),
        ),
        "analyze.surface_plot" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            true,
            true,
            Some(json!({
                "plot_width": 350,
                "polygon_multiplier": 100,
                "source_background_lighter": false,
                "black_fill": false
            })),
            Some("Create an ImageJ-style grayscale surface plot image from the active plane."),
        ),
        "analyze.set_scale" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({
                "distance_pixels": 0.0,
                "known_distance": 0.0,
                "pixel_aspect_ratio": 1.0,
                "unit": "pixel",
                "global": false
            })),
            Some("Apply ImageJ Analyze/Set Scale calibration metadata."),
        ),
        "analyze.calibrate" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            Some(json!({"function": "none", "unit": "Gray Value", "global": false})),
            Some("Apply ImageJ Analyze/Calibrate value-unit metadata."),
        ),
        "analyze.set_measurements" => CommandMetadata::with(
            CommandScope::Viewer,
            true,
            false,
            true,
            None,
            Some("Open ImageJ-style measurement settings."),
        ),
        "analyze.summarize" | "analyze.clear_results" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            None,
            Some("Operate on the shared ImageJ-style results table."),
        ),
        "analyze.distribution" => CommandMetadata::with(
            CommandScope::Both,
            true,
            true,
            false,
            Some(json!({"column": null, "bins": 10})),
            Some("Compute a distribution from a numeric results-table column."),
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
        Some(Value::Object(map)) if map.is_empty() => default,
        Some(Value::Object(map)) if !map.is_empty() => {
            let mut merged = match default {
                Value::Object(map) => map,
                other => return other,
            };

            for (key, value) in map {
                merged.insert(key, value);
            }

            Value::Object(merged)
        }
        Some(Value::Null) | None => default,
        Some(other) => other,
    }
}

#[cfg(test)]
mod tests {
    use super::{CommandScope, merge_params, metadata};
    use serde_json::json;

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
            "edit.invert",
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
            "edit.options.line_width",
            "image.type.rgb",
            "image.adjust.window_level",
            "image.adjust.size",
            "image.adjust.line_width",
            "image.adjust.coordinates",
            "image.show_info",
            "image.properties",
            "image.crop",
            "image.duplicate",
            "image.rename",
            "image.scale",
            "image.stacks.set",
            "image.stacks.add_slice",
            "image.stacks.delete_slice",
            "image.stacks.images_to_stack",
            "image.stacks.make_substack",
            "image.stacks.reslice",
            "image.stacks.stack_to_images",
            "image.stacks.make_montage",
            "image.stacks.montage_to_stack",
            "image.stacks.plot_z_profile",
            "image.stacks.plot_xy_profile",
            "image.stacks.measure_stack",
            "image.stacks.statistics",
            "image.stacks.z_project",
            "image.stacks.combine",
            "image.stacks.concatenate",
            "image.stacks.grouped_z_project",
            "image.stacks.insert",
            "image.stacks.reduce",
            "image.stacks.reverse",
            "image.stacks.set_label",
            "image.stacks.remove_slice_labels",
            "image.hyperstacks.new",
            "image.hyperstacks.stack_to_hyperstack",
            "image.hyperstacks.hyperstack_to_stack",
            "image.hyperstacks.reduce_dimensionality",
            "image.hyperstacks.make_subset",
            "image.transform.flip_horizontal",
            "image.transform.flip_z",
            "image.transform.rotate",
            "image.transform.translate",
            "image.transform.bin",
            "image.transform.image_to_results",
            "image.transform.results_to_image",
            "image.lookup.apply_lut",
            "image.lookup.invert_lut",
            "image.lookup.fire",
            "image.lookup.grays",
            "image.lookup.ice",
            "image.lookup.spectrum",
            "image.lookup.rgb332",
            "image.lookup.red",
            "image.lookup.green",
            "image.lookup.blue",
            "image.lookup.cyan",
            "image.lookup.magenta",
            "image.lookup.yellow",
            "image.lookup.red_green",
            "image.overlay.add_selection",
            "image.overlay.flatten",
            "image.overlay.from_roi_manager",
            "image.overlay.hide",
            "image.overlay.labels",
            "image.overlay.show",
            "image.overlay.remove",
            "image.overlay.options",
            "image.overlay.to_roi_manager",
            "image.overlay.toggle",
            "image.overlay.list",
            "image.overlay.measure",
            "process.noise.add",
            "process.noise.specified",
            "process.noise.salt_pepper",
            "process.noise.despeckle",
            "process.noise.remove_outliers",
            "process.noise.remove_nans",
            "process.binary.convert_mask",
            "process.binary.open",
            "process.binary.close",
            "process.binary.median",
            "process.binary.outline",
            "process.binary.fill_holes",
            "process.binary.skeletonize",
            "process.binary.distance_map",
            "process.binary.ultimate_points",
            "process.binary.watershed",
            "process.binary.voronoi",
            "process.binary.options",
            "process.shadows.north",
            "process.shadows.southwest",
            "process.shadows.demo",
            "process.filters.convolve",
            "process.filters.gaussian",
            "process.filters.mean",
            "process.filters.unsharp_mask",
            "process.filters.variance",
            "process.filters.top_hat",
            "process.filters.gaussian_3d",
            "process.filters.median_3d",
            "process.filters.mean_3d",
            "process.filters.minimum_3d",
            "process.filters.maximum_3d",
            "process.filters.variance_3d",
            "process.filters.show_circular_masks",
            "process.repeat_command",
            "process.math.sqrt",
            "process.math.nan_background",
            "process.fft.fft",
            "process.fft.bandpass",
            "process.fft.swap_quadrants",
            "process.fft.make_circular_selection",
            "process.enhance_contrast",
            "process.find_maxima",
            "process.subtract_background",
            "process.sharpen",
            "analyze.set_scale",
            "analyze.calibrate",
            "analyze.analyze_particles",
            "analyze.summarize",
            "analyze.distribution",
            "analyze.label",
            "analyze.clear_results",
            "analyze.surface_plot",
            "analyze.tools.save_xy_coordinates",
            "analyze.tools.results",
            "plugins.commands.find",
            "window.show_all",
            "window.main",
            "window.put_behind",
            "help.shortcuts",
        ] {
            assert!(
                metadata(command).implemented,
                "{command} should be implemented or explicitly handled"
            );
        }
    }

    #[test]
    fn process_math_commands_are_operation_backed() {
        for command in [
            "process.math.add",
            "process.math.and",
            "process.math.gamma",
            "process.math.sqrt",
            "process.math.nan_background",
            "process.math.abs",
            "image.stacks.reverse",
        ] {
            let metadata = metadata(command);
            assert!(metadata.implemented, "{command} should be implemented");
            assert!(metadata.requires_image, "{command} should require an image");
            assert!(!metadata.frontend_only, "{command} should run an operation");
            assert!(
                metadata.default_params.is_some(),
                "{command} should provide default operation params"
            );
        }
    }

    #[test]
    fn merge_params_uses_filter_defaults() {
        assert_eq!(
            merge_params("process.filters.median", None),
            json!({"radius": 2.0})
        );
        assert_eq!(
            merge_params("process.filters.median", Some(json!({ "radius": 4.0 }))),
            json!({ "radius": 4.0 })
        );
        assert_eq!(
            merge_params("process.filters.median", Some(json!({}))),
            json!({ "radius": 2.0 })
        );
        assert_eq!(
            merge_params(
                "process.filters.top_hat",
                Some(json!({ "dont_subtract": true }))
            ),
            json!({
                "filter": "top_hat",
                "radius": 2.0,
                "light_background": false,
                "dont_subtract": true
            })
        );
        assert_eq!(
            merge_params("process.filters.convolve", None,),
            json!({
                "width": 3,
                "height": 3,
                "kernel": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "normalize": false
            })
        );
        assert_eq!(
            merge_params("process.filters.convolve", Some(json!({ "width": 5 })),),
            json!({
                "width": 5,
                "height": 3,
                "kernel": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "normalize": false
            })
        );
        assert_eq!(
            merge_params("process.filters.top_hat", None),
            json!({
                "filter": "top_hat",
                "radius": 2.0,
                "light_background": false,
                "dont_subtract": false
            })
        );
        assert_eq!(
            merge_params("process.math.sqrt", Some(json!({ "operation": "log" }))),
            json!({"operation": "log"})
        );
        assert_eq!(
            merge_params("process.math.sqrt", Some(json!({ "value": 0.125 }))),
            json!({
                "operation": "sqrt",
                "value": 0.125
            })
        );
    }
}
