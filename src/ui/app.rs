use super::dialogs::*;
use super::macros::*;
use super::repaint::*;
use super::state::{
    BinaryOptions, DesktopState, MeasurementSettings, OverlaySettings, ResultsTableState,
    installed_macros_dir, load_desktop_state, load_startup_macro, push_recent_file,
    save_desktop_state, save_startup_macro, startup_macro_path,
};
use super::toolbar::*;
use super::{command_registry, interaction, menu};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    mpsc::{self, Receiver, Sender},
};
use std::time::Duration;

use super::image_plus::*;
use super::interaction::roi::{RoiKind, RoiModel, RoiStore};
use super::interaction::tooling::{
    LineMode, OvalMode, PointMode, RectMode, ToolId, ToolOptionsState, ToolState,
};
use super::interaction::transform::{ViewerTransformState, zoom_level_down, zoom_level_up};
use super::lut::*;
use crate::commands::MeasurementTable;
use crate::formats::supported_formats;
use crate::model::{AxisKind, Dataset, DatasetF32, Dim, Metadata, PixelType};
use crate::runtime::AppContext;
use eframe::egui;
use image::load_from_memory;
use ndarray::{ArrayD, Axis, Dimension, IxDyn, stack};
use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

const LAUNCHER_LABEL: &str = "main";
const VIEWER_PREFIX: &str = "viewer-";
const SOURCE_COMMITTED: &str = "committed";
const SOURCE_PREVIEW_PREFIX: &str = "preview:";
#[cfg(test)]
const VIEWER_MIN_WINDOW_SIZE: [f32; 2] = [220.0, 160.0];
#[cfg(test)]
const VIEWER_WINDOW_EXTRA_SIZE: [f32; 2] = [24.0, 120.0];
const LAUNCHER_MIN_WINDOW_SIZE: [f32; 2] = [600.0, 720.0];
const BINARY_MAX_ITERATIONS: usize = 100;
const BINARY_MAX_COUNT: usize = 8;
const SLICE_LABELS_KEY: &str = "slice_labels";
const CURRENT_SLICE_LABEL_KEY: &str = "Slice_Label";

#[derive(Debug)]
struct UiState {
    app: AppContext,
    startup_input: Option<PathBuf>,
    path_to_label: HashMap<PathBuf, String>,
    label_to_path: HashMap<String, PathBuf>,
    label_to_session: HashMap<String, ViewerSession>,
    next_window_id: u64,
    next_job_id: u64,
}

impl UiState {
    fn new(startup_input: Option<PathBuf>) -> Self {
        Self {
            app: AppContext::new(),
            startup_input,
            path_to_label: HashMap::new(),
            label_to_path: HashMap::new(),
            label_to_session: HashMap::new(),
            next_window_id: 0,
            next_job_id: 0,
        }
    }

    fn next_window_id(&mut self) -> u64 {
        self.next_window_id = self.next_window_id.saturating_add(1);
        self.next_window_id
    }

    fn next_job_id(&mut self) -> u64 {
        self.next_job_id = self.next_job_id.saturating_add(1);
        self.next_job_id
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ThresholdOverlay {
    low: f32,
    high: f32,
    mode: ThresholdOverlayMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThresholdOverlayMode {
    Red,
    BlackAndWhite,
    OverUnder,
}

impl ThresholdOverlayMode {
    fn from_label(label: &str) -> Self {
        match label {
            "B&W" | "Black & White" | "black_and_white" => Self::BlackAndWhite,
            "Over/Under" | "over_under" => Self::OverUnder,
            _ => Self::Red,
        }
    }
}

#[derive(Debug, Clone)]
struct ViewerSession {
    path: PathBuf,
    base_source: ViewerImageSource,
    committed_source: ViewerImageSource,
    committed_summary: ImageSummary,
    display_range: Option<(f32, f32)>,
    channel_display_ranges: HashMap<usize, (f32, f32)>,
    threshold_overlay: Option<ThresholdOverlay>,
    undo_stack: Vec<Arc<DatasetF32>>,
    redo_stack: Vec<Arc<DatasetF32>>,
    active_preview: Option<String>,
    preview_cache: HashMap<String, Arc<DatasetF32>>,
    frame_cache: HashMap<FrameKey, Arc<ViewerFrameBuffer>>,
    generation: u64,
    active_job: Option<ActiveJob>,
}

impl ViewerSession {
    fn new(path: PathBuf, source: ViewerImageSource) -> Self {
        let committed_summary = summarize_source(&source, &path);
        Self {
            path,
            base_source: source.clone(),
            committed_source: source,
            committed_summary,
            display_range: None,
            channel_display_ranges: HashMap::new(),
            threshold_overlay: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            active_preview: None,
            preview_cache: HashMap::new(),
            frame_cache: HashMap::new(),
            generation: 0,
            active_job: None,
        }
    }

    fn current_source_kind(&self) -> String {
        if let Some(key) = &self.active_preview {
            format!("{SOURCE_PREVIEW_PREFIX}{key}")
        } else {
            SOURCE_COMMITTED.to_string()
        }
    }

    fn source_for_kind(&self, source_kind: &str) -> Option<ViewerImageSource> {
        if source_kind == SOURCE_COMMITTED {
            return Some(self.committed_source.clone());
        }

        if let Some(key) = source_kind.strip_prefix(SOURCE_PREVIEW_PREFIX) {
            return self
                .preview_cache
                .get(key)
                .cloned()
                .map(ViewerImageSource::Dataset);
        }

        None
    }

    fn set_active_preview(&mut self, key: Option<String>) {
        self.active_preview = key;
        self.frame_cache.clear();
    }

    fn set_display_range(&mut self, range: Option<(f32, f32)>) {
        self.display_range = range;
        self.channel_display_ranges.clear();
        self.generation = self.generation.saturating_add(1);
        self.frame_cache.clear();
    }

    fn set_channel_display_range(&mut self, channel: usize, range: Option<(f32, f32)>) {
        if let Some(range) = range {
            self.channel_display_ranges.insert(channel, range);
        } else {
            self.channel_display_ranges.remove(&channel);
        }
        self.generation = self.generation.saturating_add(1);
        self.frame_cache.clear();
    }

    fn set_threshold_overlay(&mut self, overlay: Option<ThresholdOverlay>) {
        self.threshold_overlay = overlay;
        self.generation = self.generation.saturating_add(1);
    }

    fn committed_dataset(&self) -> Option<Arc<DatasetF32>> {
        match &self.committed_source {
            ViewerImageSource::Native(_) => None,
            ViewerImageSource::Dataset(dataset) => Some(dataset.clone()),
        }
    }

    fn committed_source(&self) -> ViewerImageSource {
        self.committed_source.clone()
    }

    fn ensure_committed_dataset(&mut self) -> Result<Arc<DatasetF32>, String> {
        let dataset = self.committed_source.to_dataset()?;
        self.committed_source = ViewerImageSource::Dataset(dataset.clone());
        self.committed_summary = summarize_source(&self.committed_source, &self.path);
        self.frame_cache.clear();
        Ok(dataset)
    }

    fn commit_dataset(&mut self, dataset: Arc<DatasetF32>) {
        if let Some(current) = self.committed_dataset() {
            self.undo_stack.push(current);
        }
        self.redo_stack.clear();
        self.replace_committed_dataset(dataset);
    }

    fn replace_committed_dataset(&mut self, dataset: Arc<DatasetF32>) {
        self.committed_source = ViewerImageSource::Dataset(dataset);
        self.committed_summary = summarize_source(&self.committed_source, &self.path);
        self.active_preview = None;
        self.preview_cache.clear();
        self.frame_cache.clear();
        self.channel_display_ranges.clear();
        self.threshold_overlay = None;
    }

    fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    fn undo(&mut self) -> bool {
        let Some(previous) = self.undo_stack.pop() else {
            return false;
        };
        if let Some(current) = self.committed_dataset() {
            self.redo_stack.push(current);
        }
        self.replace_committed_dataset(previous);
        true
    }

    fn redo(&mut self) -> bool {
        let Some(next) = self.redo_stack.pop() else {
            return false;
        };
        if let Some(current) = self.committed_dataset() {
            self.undo_stack.push(current);
        }
        self.replace_committed_dataset(next);
        true
    }

    fn revert_to_base(&mut self) -> bool {
        if source_ptr_eq(&self.committed_source, &self.base_source) {
            self.committed_source = self.base_source.clone();
            self.committed_summary = summarize_source(&self.committed_source, &self.path);
            self.active_preview = None;
            self.preview_cache.clear();
            self.frame_cache.clear();
            self.channel_display_ranges.clear();
            self.threshold_overlay = None;
            self.undo_stack.clear();
            self.redo_stack.clear();
            return false;
        }

        self.committed_source = self.base_source.clone();
        self.committed_summary = summarize_source(&self.committed_source, &self.path);
        self.active_preview = None;
        self.preview_cache.clear();
        self.frame_cache.clear();
        self.channel_display_ranges.clear();
        self.threshold_overlay = None;
        self.redo_stack.clear();
        self.undo_stack.clear();
        true
    }

    fn mark_saved(&mut self, path: PathBuf) {
        self.path = path;
        self.base_source = self.committed_source.clone();
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.committed_summary = summarize_source(&self.committed_source, &self.path);
    }

    fn is_active_job(&self, job_id: u64, generation: u64) -> bool {
        self.active_job
            .as_ref()
            .is_some_and(|job| job.job_id == job_id && job.generation == generation)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ActiveJob {
    job_id: u64,
    generation: u64,
    mode: OpRunMode,
    op: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FrameKey {
    source_kind: String,
    z: usize,
    t: usize,
    channel: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct PreviewRequest {
    op: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ViewerOpRequest {
    op: String,
    #[serde(default)]
    params: Value,
    mode: OpRunMode,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum OpRunMode {
    Preview,
    Apply,
    NewWindow,
}

impl OpRunMode {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Preview => "preview",
            Self::Apply => "apply",
            Self::NewWindow => "new window",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct JobTicket {
    job_id: u64,
}

#[derive(Debug, Clone, Serialize, Default)]
struct OpenResult {
    opened: Vec<String>,
    focused: Vec<String>,
    skipped: Vec<String>,
    errors: Vec<String>,
}

#[derive(Debug, Clone)]
enum OpenOutcome {
    Opened { label: String },
    Focused { label: String },
}

#[derive(Debug, Clone)]
struct OpRunOutput {
    dataset: Arc<DatasetF32>,
    measurements: Option<MeasurementTable>,
}

#[derive(Debug, Clone)]
enum WorkerEvent {
    OpFinished {
        window_label: String,
        job_id: u64,
        generation: u64,
        mode: OpRunMode,
        op: String,
        new_window_title: Option<String>,
        preview_key: Option<String>,
        result: Result<OpRunOutput, String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HoverInfo {
    x: usize,
    y: usize,
    value: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
enum ProgressState {
    #[default]
    Idle,
    Running(String),
}

#[derive(Debug, Clone, Default)]
struct ViewerTelemetry {
    hover: Option<HoverInfo>,
    pinned: Option<HoverInfo>,
    zoom: f32,
    z: usize,
    t: usize,
    c: usize,
    active_job: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct LauncherStatusModel {
    active_viewer: Option<String>,
    text: String,
    progress: ProgressState,
}

#[derive(Debug, Default)]
struct LauncherUiState {
    status: LauncherStatusModel,
    fallback_text: String,
}

#[derive(Debug, Clone, Default)]
struct ClipboardState {
    dataset: Option<Arc<DatasetF32>>,
    paste_mode: PasteMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum PasteMode {
    #[default]
    Copy,
    Add,
}

#[derive(Debug, Clone, Default)]
struct PlotProfileState {
    title: String,
    samples: Vec<f32>,
}

struct ViewerUiState {
    title: String,
    z: usize,
    t: usize,
    channel: usize,
    zoom: f32,
    pan: egui::Vec2,
    transform: ViewerTransformState,
    rois: interaction::roi::RoiStore,
    status_message: String,
    tool_message: Option<String>,
    hover: Option<HoverInfo>,
    pinned: Option<HoverInfo>,
    frame: Option<Arc<ViewerFrameBuffer>>,
    texture: Option<egui::TextureHandle>,
    lookup_table: LookupTable,
    last_request: Option<ViewerFrameRequest>,
    last_generation: u64,
    initial_magnification: f32,
    fit_requested: bool,
    pending_zoom: Option<ZoomCommand>,
    active_drag_started: Option<egui::Pos2>,
    active_polygon_points: Vec<egui::Pos2>,
    last_status_pointer: Option<egui::Pos2>,
}

impl ViewerUiState {
    fn new(_label: &str, title: String) -> Self {
        Self {
            title,
            z: 0,
            t: 0,
            channel: 0,
            zoom: 1.0,
            pan: egui::vec2(0.0, 0.0),
            transform: ViewerTransformState::default(),
            rois: interaction::roi::RoiStore::default(),
            status_message: "Ready.".to_string(),
            tool_message: None,
            hover: None,
            pinned: None,
            frame: None,
            texture: None,
            lookup_table: LookupTable::Grays,
            last_request: None,
            last_generation: 0,
            initial_magnification: 1.0,
            fit_requested: true,
            pending_zoom: None,
            active_drag_started: None,
            active_polygon_points: Vec::new(),
            last_status_pointer: None,
        }
    }

    fn telemetry(&self, active_job: Option<String>) -> ViewerTelemetry {
        ViewerTelemetry {
            hover: self.hover,
            pinned: self.pinned,
            zoom: self.transform.magnification,
            z: self.z,
            t: self.t,
            c: self.channel,
            active_job,
        }
    }

    fn status_text(&self, tool: ToolId) -> String {
        let sample = self.pinned.or(self.hover);
        let sample_text = if let Some(sample) = sample {
            format!("X:{} Y:{} Value:{:.4}", sample.x, sample.y, sample.value)
        } else {
            "X:- Y:- Value:-".to_string()
        };
        let roi_message = self.rois.active_status_text();
        let message = self
            .tool_message
            .as_deref()
            .or(roi_message.as_deref())
            .unwrap_or(&self.status_message);

        format!(
            "Tool:{}  {}  Z:{} T:{} C:{}  Zoom:{:.0}%  {}",
            tool.label(),
            sample_text,
            self.z,
            self.t,
            self.channel,
            self.transform.magnification * 100.0,
            message
        )
    }
}

#[derive(Debug)]
enum UiAction {
    Command {
        window_label: String,
        command_id: String,
        params: Option<Value>,
    },
    RunInstalledMacro {
        window_label: String,
        path: PathBuf,
        macro_name: String,
    },
    OpenPaths {
        paths: Vec<PathBuf>,
    },
    CloseViewer {
        label: String,
    },
}

#[derive(Debug, Clone)]
struct StoredCommand {
    command_id: String,
    params: Option<Value>,
}

#[derive(Debug, Default)]
struct MacroRecorderState {
    open: bool,
    recording: bool,
    text: String,
    last_run_log: String,
}

#[derive(Debug, Default)]
struct StartupMacroState {
    open: bool,
    text: String,
    last_run_log: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ZoomCommand {
    In,
    Out,
    Original,
    View100,
    ToSelection,
    ScaleToFit,
    Set {
        magnification: f32,
        x_center: Option<f32>,
        y_center: Option<f32>,
    },
    Maximize,
}

#[derive(Debug, Clone, Copy)]
enum LayoutMode {
    Tile,
    Cascade,
}

#[derive(Debug, Clone, Copy)]
enum OverlayVisibility {
    Show,
    Hide,
    Toggle,
}

fn binary_morphology_uses_options(command_id: &str) -> bool {
    matches!(
        command_id,
        "process.binary.erode"
            | "process.binary.dilate"
            | "process.binary.open"
            | "process.binary.close"
    )
}

fn clamp_binary_options(options: &mut BinaryOptions) {
    options.iterations = options.iterations.clamp(1, BINARY_MAX_ITERATIONS);
    options.count = options.count.clamp(1, BINARY_MAX_COUNT);
}

fn update_binary_options_from_params(options: &mut BinaryOptions, params: &Value) {
    if let Some(iterations) = params.get("iterations").and_then(Value::as_u64) {
        options.iterations = iterations as usize;
    }
    if let Some(count) = params.get("count").and_then(Value::as_u64) {
        options.count = count as usize;
    }
    clamp_binary_options(options);
}

fn binary_morphology_params(
    command_id: &str,
    request_params: Option<Value>,
    options: &BinaryOptions,
) -> Value {
    let mut params = command_registry::merge_params(command_id, request_params);
    if binary_morphology_uses_options(command_id) {
        if let Value::Object(map) = &mut params {
            map.entry("iterations".to_string())
                .or_insert_with(|| json!(options.iterations.clamp(1, BINARY_MAX_ITERATIONS)));
            map.entry("count".to_string())
                .or_insert_with(|| json!(options.count.clamp(1, BINARY_MAX_COUNT)));
        }
    }
    params
}

struct ImageUiApp {
    state: UiState,
    desktop_state: DesktopState,
    menus: Vec<menu::MenuManifestTopLevel>,
    command_catalog: command_registry::CommandCatalog,
    launcher_ui: LauncherUiState,
    tool_state: ToolState,
    tool_options: ToolOptionsState,
    toolbar_icons: HashMap<ToolbarIcon, egui::TextureHandle>,
    viewers_ui: HashMap<String, ViewerUiState>,
    viewer_telemetry: HashMap<String, ViewerTelemetry>,
    results_table: ResultsTableState,
    clipboard: ClipboardState,
    profile_plot: PlotProfileState,
    new_image_dialog: NewImageDialogState,
    adjust_dialog: AdjustDialogState,
    threshold_apply_dialog: ThresholdApplyDialogState,
    threshold_set_dialog: ThresholdSetDialogState,
    apply_lut_dialog: ApplyLutDialogState,
    set_display_range_dialog: SetDisplayRangeDialogState,
    set_window_level_dialog: SetWindowLevelDialogState,
    resize_dialog: ResizeDialogState,
    canvas_dialog: ResizeDialogState,
    stack_position_dialog: StackPositionDialogState,
    stack_label_dialog: StackLabelDialogState,
    zoom_set_dialog: ZoomSetDialogState,
    color_dialog: ColorDialogState,
    raw_import_dialog: RawImportDialogState,
    url_import_dialog: UrlImportDialogState,
    command_finder: CommandFinderState,
    macro_recorder: MacroRecorderState,
    startup_macro: StartupMacroState,
    macro_compatibility_open: bool,
    macro_compatibility_command: Option<String>,
    macro_compatibility_path: Option<PathBuf>,
    macro_compatibility_preview: String,
    macro_compatibility_run_log: String,
    roi_manager: RoiManagerState,
    last_repeatable_command: Option<StoredCommand>,
    active_viewer_label: Option<String>,
    worker_tx: Sender<WorkerEvent>,
    worker_rx: Receiver<WorkerEvent>,
    focus_launcher: bool,
    focus_viewer_label: Option<String>,
    should_quit: bool,
}

impl ImageUiApp {
    #[cfg(test)]
    fn new_for_test() -> Self {
        let menus = menu::manifest().clone();
        let command_catalog = command_registry::command_catalog();
        let (worker_tx, worker_rx) = mpsc::channel();

        Self {
            state: UiState::new(None),
            desktop_state: DesktopState::default(),
            menus,
            command_catalog,
            launcher_ui: LauncherUiState {
                status: LauncherStatusModel::default(),
                fallback_text: String::new(),
            },
            tool_state: ToolState::default(),
            tool_options: ToolOptionsState::default(),
            toolbar_icons: HashMap::new(),
            viewers_ui: HashMap::new(),
            viewer_telemetry: HashMap::new(),
            results_table: ResultsTableState::default(),
            clipboard: ClipboardState::default(),
            profile_plot: PlotProfileState::default(),
            new_image_dialog: NewImageDialogState::default(),
            adjust_dialog: AdjustDialogState::default(),
            threshold_apply_dialog: ThresholdApplyDialogState::default(),
            threshold_set_dialog: ThresholdSetDialogState::default(),
            apply_lut_dialog: ApplyLutDialogState::default(),
            set_display_range_dialog: SetDisplayRangeDialogState::default(),
            set_window_level_dialog: SetWindowLevelDialogState::default(),
            resize_dialog: ResizeDialogState::default(),
            canvas_dialog: ResizeDialogState::default(),
            stack_position_dialog: StackPositionDialogState::default(),
            stack_label_dialog: StackLabelDialogState::default(),
            zoom_set_dialog: ZoomSetDialogState::default(),
            color_dialog: ColorDialogState::default(),
            raw_import_dialog: RawImportDialogState::default(),
            url_import_dialog: UrlImportDialogState::default(),
            command_finder: CommandFinderState::default(),
            macro_recorder: MacroRecorderState::default(),
            startup_macro: StartupMacroState::default(),
            macro_compatibility_open: false,
            macro_compatibility_command: None,
            macro_compatibility_path: None,
            macro_compatibility_preview: String::new(),
            macro_compatibility_run_log: String::new(),
            roi_manager: RoiManagerState::default(),
            last_repeatable_command: None,
            active_viewer_label: None,
            worker_tx,
            worker_rx,
            focus_launcher: false,
            focus_viewer_label: None,
            should_quit: false,
        }
    }

    fn new(_cc: &eframe::CreationContext<'_>, startup_input: Option<PathBuf>) -> Self {
        let menus = menu::manifest().clone();
        let command_catalog = command_registry::command_catalog();
        let (worker_tx, worker_rx) = mpsc::channel();
        let desktop_state = load_desktop_state();

        let mut app = Self {
            state: UiState::new(startup_input),
            desktop_state,
            menus,
            command_catalog,
            launcher_ui: LauncherUiState {
                status: LauncherStatusModel::default(),
                fallback_text: "Ready. Use File > Open or drop image files onto this window."
                    .to_string(),
            },
            tool_state: ToolState::default(),
            tool_options: ToolOptionsState::default(),
            toolbar_icons: HashMap::new(),
            viewers_ui: HashMap::new(),
            viewer_telemetry: HashMap::new(),
            results_table: ResultsTableState::default(),
            clipboard: ClipboardState::default(),
            profile_plot: PlotProfileState::default(),
            new_image_dialog: NewImageDialogState::default(),
            adjust_dialog: AdjustDialogState::default(),
            threshold_apply_dialog: ThresholdApplyDialogState::default(),
            threshold_set_dialog: ThresholdSetDialogState::default(),
            apply_lut_dialog: ApplyLutDialogState::default(),
            set_display_range_dialog: SetDisplayRangeDialogState::default(),
            set_window_level_dialog: SetWindowLevelDialogState::default(),
            resize_dialog: ResizeDialogState::default(),
            canvas_dialog: ResizeDialogState::default(),
            stack_position_dialog: StackPositionDialogState::default(),
            stack_label_dialog: StackLabelDialogState::default(),
            zoom_set_dialog: ZoomSetDialogState::default(),
            color_dialog: ColorDialogState::default(),
            raw_import_dialog: RawImportDialogState::default(),
            url_import_dialog: UrlImportDialogState::default(),
            command_finder: CommandFinderState::default(),
            macro_recorder: MacroRecorderState::default(),
            startup_macro: StartupMacroState::default(),
            macro_compatibility_open: false,
            macro_compatibility_command: None,
            macro_compatibility_path: None,
            macro_compatibility_preview: String::new(),
            macro_compatibility_run_log: String::new(),
            roi_manager: RoiManagerState::default(),
            last_repeatable_command: None,
            active_viewer_label: None,
            worker_tx,
            worker_rx,
            focus_launcher: false,
            focus_viewer_label: None,
            should_quit: false,
        };

        if let Some(path) = app.state.startup_input.clone() {
            let result = app.open_paths(vec![path]);
            app.apply_open_result(&result);
        }

        app.run_startup_macro_if_present();
        app
    }

    fn apply_open_result(&mut self, result: &OpenResult) {
        self.launcher_ui.fallback_text = format!(
            "Opened {}, focused {}, skipped {}, errors {}",
            result.opened.len(),
            result.focused.len(),
            result.skipped.len(),
            result.errors.len()
        );
        self.refresh_launcher_status();
    }

    fn set_fallback_status(&mut self, status: impl Into<String>) {
        self.launcher_ui.fallback_text = status.into();
        self.refresh_launcher_status();
    }

    fn persist_desktop_state(&self) {
        let _ = save_desktop_state(&self.desktop_state);
    }

    fn run_startup_macro_if_present(&mut self) {
        let run_at_startup = match load_startup_macro() {
            Ok(contents) => contents,
            Err(error) => {
                self.set_fallback_status(format!("startup macro load failed: {error}"));
                return;
            }
        };
        let active_label = self
            .active_viewer_label
            .clone()
            .unwrap_or_else(|| LAUNCHER_LABEL.to_string());
        let mut reports = Vec::new();

        if !run_at_startup.trim().is_empty() {
            let report =
                self.run_simple_macro_source("RunAtStartup.ijm", &run_at_startup, &active_label);
            reports.push(format!("RunAtStartup: {}", first_report_line(&report)));
        }

        if let Some(path) = startup_macro_set_path() {
            match fs::read_to_string(&path) {
                Ok(contents) => {
                    if let Some(block) = startup_auto_run_macro_block(&contents) {
                        let source_name = format!("{}:{}", path.display(), block.name);
                        let named_blocks = macro_named_block_statement_map(&contents);
                        let report = self.run_simple_macro_lines(
                            &source_name,
                            block.statements,
                            &active_label,
                            &named_blocks,
                        );
                        reports.push(format!("{}: {}", block.name, first_report_line(&report)));
                    }
                }
                Err(error) => reports.push(format!(
                    "StartupMacros load failed: {}: {error}",
                    path.display()
                )),
            }
        }

        if !reports.is_empty() {
            self.set_fallback_status(format!("startup macro: {}", reports.join("; ")));
        }
    }

    fn set_active_viewer(&mut self, label: Option<String>) {
        self.active_viewer_label = label;
        self.refresh_launcher_status();
    }

    fn update_viewer_telemetry(&mut self, label: &str, telemetry: ViewerTelemetry) {
        self.viewer_telemetry.insert(label.to_string(), telemetry);
        self.refresh_launcher_status();
    }

    fn open_color_dialog(&mut self, mode: ColorDialogMode) {
        self.color_dialog.open = true;
        self.color_dialog.mode = mode;
        self.color_dialog.foreground = self.tool_options.foreground_color;
        self.color_dialog.background = self.tool_options.background_color;
    }

    fn refresh_launcher_status(&mut self) {
        let active_label = self
            .active_viewer_label
            .clone()
            .filter(|label| self.state.label_to_session.contains_key(label));
        self.launcher_ui.status.active_viewer = active_label.clone();

        let active_telemetry = active_label
            .as_ref()
            .and_then(|label| self.viewer_telemetry.get(label));
        self.launcher_ui.status.progress = active_telemetry
            .and_then(|telemetry| telemetry.active_job.clone())
            .map(ProgressState::Running)
            .unwrap_or(ProgressState::Idle);
        self.launcher_ui.status.text = format_launcher_status(
            self.tool_state.selected,
            &self.launcher_ui.status,
            active_telemetry,
            &self.state.label_to_path,
            &self.launcher_ui.fallback_text,
        );
    }

    fn has_active_jobs(&self) -> bool {
        self.state
            .label_to_session
            .values()
            .any(|session| session.active_job.is_some())
    }

    fn poll_worker_events(&mut self) -> bool {
        let mut state_changed = false;
        while let Ok(event) = self.worker_rx.try_recv() {
            match event {
                WorkerEvent::OpFinished {
                    window_label,
                    job_id,
                    generation,
                    mode,
                    op,
                    new_window_title,
                    preview_key,
                    result,
                } => {
                    let mut status = format!("{op} failed");
                    let mut measurements = None;
                    let mut new_window_dataset = None;
                    if let Some(session) = self.state.label_to_session.get_mut(&window_label) {
                        if !session.is_active_job(job_id, generation) {
                            continue;
                        }
                        state_changed = true;

                        match result {
                            Ok(output) => {
                                match mode {
                                    OpRunMode::Preview => {
                                        if let Some(key) = preview_key {
                                            session
                                                .preview_cache
                                                .insert(key.clone(), output.dataset);
                                            session.set_active_preview(Some(key));
                                        }
                                    }
                                    OpRunMode::Apply => {
                                        session.commit_dataset(output.dataset);
                                        measurements = output.measurements;
                                    }
                                    OpRunMode::NewWindow => {
                                        new_window_dataset =
                                            Some((output.dataset.clone(), new_window_title));
                                        measurements = output.measurements;
                                    }
                                }
                                session.active_job = None;
                                status = format!("Applied {op} (job {job_id})");
                            }
                            Err(error) => {
                                session.active_job = None;
                                status = error;
                            }
                        }
                    }

                    if let Some((dataset, title)) = new_window_dataset {
                        let title = title
                            .as_deref()
                            .filter(|title| !title.trim().is_empty())
                            .map(sanitize_image_title)
                            .unwrap_or_else(|| {
                                format!("Untitled-{}", self.state.next_window_id + 1)
                            });
                        let path = normalize_path(&PathBuf::from(format!(
                            "{}-{}.tif",
                            title,
                            self.state.next_window_id + 1
                        )));
                        let label = self.create_viewer(path, ViewerImageSource::Dataset(dataset));
                        status = format!("Created {label} from {op} (job {job_id})");
                    }

                    if let Some(measurements) = measurements {
                        if op == "measurements.profile" {
                            if let Some(samples) = profile_samples_from_table(&measurements) {
                                self.profile_plot.title = format!("Profile ({window_label})");
                                self.profile_plot.samples = samples;
                                self.desktop_state.utility_windows.profile_plot_open = true;
                            }
                        } else {
                            for row in measurement_rows_from_table(&measurements) {
                                self.results_table.add_row(row);
                            }
                            self.desktop_state.utility_windows.results_open = true;
                        }
                    }

                    if let Some(viewer) = self.viewers_ui.get_mut(&window_label) {
                        viewer.status_message = status;
                        viewer.last_request = None;
                        viewer.last_generation = 0;
                        viewer.hover = None;
                    }
                }
            }
        }
        if state_changed {
            self.refresh_launcher_status();
        }
        state_changed
    }

    fn open_paths(&mut self, paths: Vec<PathBuf>) -> OpenResult {
        let mut result = OpenResult::default();

        for path in paths {
            if !is_supported_image_path(&path) {
                result.skipped.push(path.display().to_string());
                continue;
            }

            let normalized = normalize_path(&path);
            if !normalized.exists() {
                result
                    .errors
                    .push(format!("{} (file not found)", normalized.display()));
                continue;
            }

            match self.open_or_focus_viewer(&normalized) {
                Ok(OpenOutcome::Opened { label }) => {
                    result
                        .opened
                        .push(format!("{} -> {label}", normalized.display()));
                }
                Ok(OpenOutcome::Focused { label }) => {
                    result
                        .focused
                        .push(format!("{} -> {label}", normalized.display()));
                }
                Err(error) => {
                    result
                        .errors
                        .push(format!("{} ({error})", normalized.display()));
                }
            }
        }

        result
    }

    fn open_or_focus_viewer(&mut self, path: &Path) -> Result<OpenOutcome, String> {
        if !is_supported_image_path(path) {
            return Err("unsupported image type".to_string());
        }

        let normalized_path = normalize_path(path);
        push_recent_file(&mut self.desktop_state, &normalized_path);
        self.persist_desktop_state();
        if let Some(label) = self.state.path_to_label.get(&normalized_path).cloned() {
            if self.state.label_to_session.contains_key(&label) {
                self.focus_viewer_label = Some(label.clone());
                self.set_active_viewer(Some(label.clone()));
                return Ok(OpenOutcome::Focused { label });
            }
            self.remove_stale_mapping(&normalized_path, &label);
        }

        let source = if let Some(native) = self
            .state
            .app
            .io_service()
            .read_native(&normalized_path)
            .map_err(|error: crate::runtime::AppError| error.to_string())?
        {
            ViewerImageSource::Native(Arc::new(native))
        } else {
            let dataset = self
                .state
                .app
                .io_service()
                .read(&normalized_path)
                .map_err(|error| error.to_string())?;
            ViewerImageSource::Dataset(Arc::new(dataset))
        };
        let label = self.create_viewer(normalized_path.clone(), source);
        Ok(OpenOutcome::Opened { label })
    }

    fn create_viewer(&mut self, path: PathBuf, source: ViewerImageSource) -> String {
        let session = ViewerSession::new(path.clone(), source);
        let id = self.state.next_window_id();
        let label = format!("{VIEWER_PREFIX}{id}");
        let title = format!(
            "{} - image-rs",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Image")
        );

        self.state.path_to_label.insert(path.clone(), label.clone());
        self.state.label_to_path.insert(label.clone(), path);
        self.state.label_to_session.insert(label.clone(), session);
        self.viewers_ui
            .insert(label.clone(), ViewerUiState::new(&label, title));
        self.focus_viewer_label = Some(label.clone());
        self.set_active_viewer(Some(label.clone()));
        label
    }

    fn remove_stale_mapping(&mut self, path: &Path, label: &str) {
        if self
            .state
            .path_to_label
            .get(path)
            .is_some_and(|known| known == label)
        {
            self.state.path_to_label.remove(path);
        }
        if self
            .state
            .label_to_path
            .get(label)
            .is_some_and(|known| known == path)
        {
            self.state.label_to_path.remove(label);
        }
        self.state.label_to_session.remove(label);
        self.viewers_ui.remove(label);
        self.viewer_telemetry.remove(label);
    }

    fn remove_viewer_by_label(&mut self, label: &str) {
        if let Some(path) = self.state.label_to_path.remove(label)
            && self
                .state
                .path_to_label
                .get(&path)
                .is_some_and(|known| known == label)
        {
            self.state.path_to_label.remove(&path);
        }

        self.state.label_to_session.remove(label);
        self.viewers_ui.remove(label);
        self.viewer_telemetry.remove(label);
        if self.active_viewer_label.as_deref() == Some(label) {
            let mut labels = self.state.label_to_path.keys().cloned().collect::<Vec<_>>();
            labels.sort_by_key(|candidate| viewer_sort_key(candidate));
            self.set_active_viewer(labels.first().cloned());
        }
    }

    fn active_dataset_slice(
        &self,
        window_label: &str,
        request: &ViewerFrameRequest,
    ) -> Result<SliceImage, String> {
        let session = self
            .state
            .label_to_session
            .get(window_label)
            .ok_or_else(|| format!("no viewer session for `{window_label}`"))?;
        let source = session
            .source_for_kind(&session.current_source_kind())
            .ok_or_else(|| format!("no dataset found for `{window_label}`"))?;
        extract_slice_from_source(&source, request.z, request.t, request.channel)
    }

    fn adjust_histogram_for_viewer(
        &self,
        viewer_label: &str,
        bins: usize,
        stack_histogram: bool,
    ) -> Result<AdjustHistogram, String> {
        let viewer = self
            .viewers_ui
            .get(viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
        let bbox = selected_roi_bbox(viewer);
        if stack_histogram {
            let values = self.viewer_stack_values(viewer_label, viewer.channel, viewer.t, bbox)?;
            return adjust_histogram(&values, bins);
        }
        let request = ViewerFrameRequest {
            z: viewer.z,
            t: viewer.t,
            channel: viewer.channel,
        };
        let slice = self.active_dataset_slice(viewer_label, &request)?;
        let values = slice_values_in_bbox(&slice, bbox)?;
        adjust_histogram(&values, bins)
    }

    fn viewer_stack_values(
        &self,
        viewer_label: &str,
        channel: usize,
        time: usize,
        bbox: Option<(usize, usize, usize, usize)>,
    ) -> Result<Vec<f32>, String> {
        let session = self
            .state
            .label_to_session
            .get(viewer_label)
            .ok_or_else(|| format!("no viewer session for `{viewer_label}`"))?;
        let z_slices = session.committed_summary.z_slices.max(1);
        let mut values = Vec::new();
        for z in 0..z_slices {
            let request = ViewerFrameRequest {
                z,
                t: time,
                channel,
            };
            let slice = self.active_dataset_slice(viewer_label, &request)?;
            values.extend(slice_values_in_bbox(&slice, bbox)?);
        }
        Ok(values)
    }

    fn save_viewer(&mut self, window_label: &str, path: Option<PathBuf>) -> Result<String, String> {
        let current_path = self
            .state
            .label_to_path
            .get(window_label)
            .cloned()
            .ok_or_else(|| format!("no viewer session for `{window_label}`"))?;
        let target_path = path.unwrap_or_else(|| current_path.clone());
        let normalized_target = normalize_path(&target_path);
        if self
            .state
            .path_to_label
            .get(&normalized_target)
            .is_some_and(|existing| existing != window_label)
        {
            return Err(format!(
                "{} is already open in another viewer",
                normalized_target.display()
            ));
        }
        let source = self
            .state
            .label_to_session
            .get(window_label)
            .map(ViewerSession::committed_source)
            .ok_or_else(|| format!("no viewer session for `{window_label}`"))?;

        match &source {
            ViewerImageSource::Native(image) => {
                if self
                    .state
                    .app
                    .io_service()
                    .write_native(&normalized_target, image.as_ref())
                    .is_err()
                {
                    let dataset = image
                        .to_dataset()
                        .map_err(|error: crate::formats::IoError| error.to_string())?;
                    self.state
                        .app
                        .io_service()
                        .write(&normalized_target, &dataset)
                        .map_err(|error| error.to_string())?;
                }
            }
            ViewerImageSource::Dataset(dataset) => {
                self.state
                    .app
                    .io_service()
                    .write(&normalized_target, dataset.as_ref())
                    .map_err(|error| error.to_string())?;
            }
        }

        if normalized_target != current_path {
            self.state.path_to_label.remove(&current_path);
            self.state
                .path_to_label
                .insert(normalized_target.clone(), window_label.to_string());
            self.state
                .label_to_path
                .insert(window_label.to_string(), normalized_target.clone());
        }

        if let Some(session) = self.state.label_to_session.get_mut(window_label) {
            session.mark_saved(normalized_target.clone());
        }
        push_recent_file(&mut self.desktop_state, &normalized_target);
        self.persist_desktop_state();
        if let Some(viewer) = self.viewers_ui.get_mut(window_label) {
            viewer.title = format!(
                "{} - image-rs",
                normalized_target
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("Image")
            );
        }
        self.refresh_launcher_status();

        Ok(format!("saved {}", normalized_target.display()))
    }

    fn current_viewer_label<'a>(&'a self, window_label: &'a str) -> Option<&'a str> {
        if window_label == LAUNCHER_LABEL {
            self.active_viewer_label.as_deref()
        } else {
            Some(window_label)
        }
    }

    fn create_new_image(&mut self, params: &Value) -> Result<String, String> {
        let dataset = new_image_dataset(params)?;
        let title = params
            .get("title")
            .and_then(Value::as_str)
            .filter(|title| !title.trim().is_empty())
            .map(sanitize_image_title)
            .unwrap_or_else(|| format!("Untitled-{}", self.state.next_window_id + 1));
        let path = normalize_path(&PathBuf::from(format!("{title}.tif")));
        let label = self.create_viewer(path, ViewerImageSource::Dataset(Arc::new(dataset)));
        Ok(format!("created new image in {label}"))
    }

    fn select_macro_window(&mut self, params: &Value) -> Result<String, String> {
        let title = params
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();
        if title.is_empty() {
            return Err("selectWindow requires a title".to_string());
        }

        let Some(label) = self.macro_viewer_label_by_title(title) else {
            return Err(format!("window `{title}` not found"));
        };

        self.focus_viewer_label = Some(label.clone());
        self.set_active_viewer(Some(label.clone()));
        Ok(format!("selected window {title}"))
    }

    fn close_macro_window(&mut self, params: &Value) -> Result<String, String> {
        let title = params
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();
        if title.is_empty() {
            return Err("close requires a window title".to_string());
        }

        let Some(label) = self.macro_viewer_label_by_title(title) else {
            return Err(format!("window `{title}` not found"));
        };

        self.remove_viewer_by_label(&label);
        Ok(format!("closed window {title}"))
    }

    fn select_macro_image(&mut self, params: &Value) -> Result<String, String> {
        let id = params
            .get("id")
            .and_then(Value::as_u64)
            .ok_or_else(|| "selectImage requires a numeric id".to_string())?;
        let label = format!("{VIEWER_PREFIX}{id}");
        if !self.state.label_to_session.contains_key(&label) {
            return Err(format!("image id {id} not found"));
        }

        self.focus_viewer_label = Some(label.clone());
        self.set_active_viewer(Some(label.clone()));
        Ok(format!("selected image {id}"))
    }

    fn macro_viewer_label_by_title(&self, title: &str) -> Option<String> {
        let normalized_title = normalize_macro_command_label(title);
        self.state.label_to_path.iter().find_map(|(label, path)| {
            let label_matches = normalize_macro_command_label(label) == normalized_title;
            let file_name_matches = path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| normalize_macro_command_label(name) == normalized_title)
                .unwrap_or(false);
            let stem_matches = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(|stem| normalize_macro_command_label(stem) == normalized_title)
                .unwrap_or(false);
            (label_matches || file_name_matches || stem_matches).then(|| label.clone())
        })
    }

    fn open_recent_path(&mut self, path: &Path) -> Result<String, String> {
        let normalized = normalize_path(path);
        if !normalized.exists() {
            self.desktop_state
                .recent_files
                .retain(|candidate| candidate != &normalized);
            self.persist_desktop_state();
            return Err(format!("{} no longer exists", normalized.display()));
        }
        match self.open_or_focus_viewer(&normalized)? {
            OpenOutcome::Opened { label } => Ok(format!("opened recent file in {label}")),
            OpenOutcome::Focused { label } => Ok(format!("focused recent file in {label}")),
        }
    }

    fn import_image_sequence(&mut self) -> Result<String, String> {
        let extensions = supported_formats();
        let mut paths = FileDialog::new()
            .add_filter("Supported images", extensions)
            .set_title("Import Image Sequence")
            .pick_files()
            .unwrap_or_default();
        if paths.is_empty() {
            return Ok("image sequence import canceled".to_string());
        }
        paths.sort();

        let mut datasets = Vec::with_capacity(paths.len());
        for path in &paths {
            datasets.push(
                self.state
                    .app
                    .io_service()
                    .read(path)
                    .map_err(|error| error.to_string())?,
            );
        }

        let first = datasets
            .first()
            .ok_or_else(|| "no images selected".to_string())?;
        let base_shape = first.shape().to_vec();
        for dataset in &datasets[1..] {
            if dataset.shape() != base_shape {
                return Err("all sequence images must have identical shape".to_string());
            }
        }

        let mut output_shape = base_shape.clone();
        output_shape.push(datasets.len());
        let mut values = Vec::new();
        for dataset in &datasets {
            values.extend(dataset.data.iter().copied());
        }
        let data = ArrayD::from_shape_vec(IxDyn(&output_shape), values)
            .map_err(|error| format!("sequence shape error: {error}"))?;
        let mut metadata = first.metadata.clone();
        metadata.dims.push(Dim::new(AxisKind::Z, datasets.len()));
        metadata
            .extras
            .insert("sequence_sources".to_string(), json!(paths));
        let dataset = Dataset::new(data, metadata).map_err(|error| error.to_string())?;
        let first_name = paths
            .first()
            .and_then(|path| path.file_stem())
            .and_then(|name| name.to_str())
            .unwrap_or("sequence");
        let virtual_path = normalize_path(&PathBuf::from(format!("{first_name}-sequence.tif")));
        self.create_viewer(virtual_path, ViewerImageSource::Dataset(Arc::new(dataset)));
        Ok("image sequence imported".to_string())
    }

    fn import_from_url(&mut self, params: &Value) -> Result<String, String> {
        let url = params
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| "`url` is required".to_string())?;
        let response = ureq::get(url)
            .call()
            .map_err(|error| format!("url import failed: {error}"))?;
        let mut reader = response.into_reader();
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .map_err(|error| format!("url read failed: {error}"))?;
        let hint = url.rsplit('.').next().unwrap_or("png");
        let mut dataset = self
            .state
            .app
            .io_service()
            .read_bytes(&bytes, hint)
            .map_err(|error| error.to_string())?;
        dataset.metadata.source = Some(PathBuf::from(url));
        self.create_viewer(
            normalize_path(&PathBuf::from(
                url.rsplit('/').next().unwrap_or("downloaded-image.tif"),
            )),
            ViewerImageSource::Dataset(Arc::new(dataset)),
        );
        Ok("image imported from URL".to_string())
    }

    fn finish_raw_import(&mut self) -> String {
        let Some(path) = self.raw_import_dialog.path.clone() else {
            return "raw import requires a file".to_string();
        };
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) => return format!("raw import failed: {error}"),
        };
        let dataset = match self.state.app.io_service().read_raw(
            &bytes,
            self.raw_import_dialog.width,
            self.raw_import_dialog.height,
            self.raw_import_dialog.slices,
            self.raw_import_dialog.channels,
            self.raw_import_dialog.pixel_type,
            self.raw_import_dialog.little_endian,
            self.raw_import_dialog.byte_offset,
        ) {
            Ok(dataset) => dataset,
            Err(error) => return error.to_string(),
        };
        self.create_viewer(
            normalize_path(&path),
            ViewerImageSource::Dataset(Arc::new(dataset)),
        );
        self.raw_import_dialog.open = false;
        "raw image imported".to_string()
    }

    fn export_results(&mut self) -> Result<String, String> {
        if self.results_table.is_empty() {
            return Err("results table is empty".to_string());
        }
        let Some(path) = FileDialog::new()
            .add_filter("CSV", &["csv"])
            .add_filter("JSON", &["json"])
            .set_title("Export Results")
            .save_file()
        else {
            return Ok("results export canceled".to_string());
        };
        let columns = self.results_table.columns();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("csv")
            .to_ascii_lowercase();
        if extension == "json" {
            let payload = serde_json::to_vec_pretty(&self.results_table.rows)
                .map_err(|error| format!("json export failed: {error}"))?;
            fs::write(&path, payload).map_err(|error| format!("write failed: {error}"))?;
        } else {
            let mut file =
                fs::File::create(&path).map_err(|error| format!("create failed: {error}"))?;
            writeln!(file, "{}", columns.join(","))
                .map_err(|error| format!("write failed: {error}"))?;
            for row in &self.results_table.rows {
                let line = columns
                    .iter()
                    .map(|column| row.get(column).map(value_to_csv).unwrap_or_default())
                    .collect::<Vec<_>>()
                    .join(",");
                writeln!(file, "{line}").map_err(|error| format!("write failed: {error}"))?;
            }
        }
        Ok(format!("exported results to {}", path.display()))
    }

    fn measure_active_viewer(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for measure".to_string())?
            .to_string();
        let (slice, roi_bbox, z, t, channel) = self.measurement_context(&viewer_label)?;
        let row = measurement_row_from_slice(
            &slice.values,
            slice.width,
            slice.height,
            roi_bbox,
            &self.desktop_state.measurement_settings,
            z,
            t,
            channel,
        );
        self.results_table.add_row(row);
        self.desktop_state.utility_windows.results_open = true;
        self.persist_desktop_state();
        Ok("measurement added to results".to_string())
    }

    fn measure_stack(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for Measure Stack".to_string())?
            .to_string();
        let (channel, time, roi_bbox) = {
            let viewer = self
                .viewers_ui
                .get(&viewer_label)
                .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
            (viewer.channel, viewer.t, selected_roi_bbox(viewer))
        };
        let dataset = self
            .state
            .label_to_session
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer session for `{viewer_label}`"))?
            .ensure_committed_dataset()?;
        let rows = stack_measurement_rows(
            dataset.as_ref(),
            &self.desktop_state.measurement_settings,
            roi_bbox,
            channel,
            time,
        )?;
        let count = rows.len();
        for row in rows {
            self.results_table.add_row(row);
        }
        self.desktop_state.utility_windows.results_open = true;
        self.persist_desktop_state();
        Ok(format!("measured {count} stack slices"))
    }

    fn plot_stack_xy_profile(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for Plot XY Profile".to_string())?
            .to_string();
        let dataset = self
            .state
            .label_to_session
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer session for `{viewer_label}`"))?
            .ensure_committed_dataset()?;
        let rows = stack_xy_profile_rows(dataset.as_ref(), params)?;
        let count = rows.len();
        self.results_table.clear();
        for row in rows {
            self.results_table.add_row(row);
        }
        self.desktop_state.utility_windows.results_open = true;
        self.persist_desktop_state();
        Ok(format!("plotted XY profiles for {count} stack slices"))
    }

    fn open_stack_label_dialog(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for Set Label".to_string())?
            .to_string();
        let (slice, label) = {
            let viewer = self
                .viewers_ui
                .get(&viewer_label)
                .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
            let dataset = self
                .state
                .label_to_session
                .get(&viewer_label)
                .and_then(ViewerSession::committed_dataset)
                .ok_or_else(|| "Set Label requires a dataset-backed image".to_string())?;
            let slice = viewer.z;
            let label = stack_slice_label(dataset.as_ref(), slice).unwrap_or_default();
            (slice, label)
        };
        self.stack_label_dialog.open = true;
        self.stack_label_dialog.window_label = viewer_label;
        self.stack_label_dialog.slice = slice + 1;
        self.stack_label_dialog.label = label;
        Ok("slice label dialog opened".to_string())
    }

    fn set_stack_slice_label(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for Set Label".to_string())?
            .to_string();
        let viewer_z = self
            .viewers_ui
            .get(&viewer_label)
            .map(|viewer| viewer.z)
            .unwrap_or(0);
        let slice = params
            .get("slice")
            .and_then(Value::as_u64)
            .map(|slice| slice.saturating_sub(1) as usize)
            .unwrap_or(viewer_z);
        let label = params
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let dataset = self
            .state
            .label_to_session
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer session for `{viewer_label}`"))?
            .ensure_committed_dataset()?;
        let labeled = set_stack_slice_label_dataset(dataset.as_ref(), slice, &label)?;
        let session = self
            .state
            .label_to_session
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer session for `{viewer_label}`"))?;
        session.commit_dataset(Arc::new(labeled));
        Ok(format!("set label for slice {}", slice + 1))
    }

    fn remove_stack_slice_labels(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for Remove Slice Labels".to_string())?
            .to_string();
        let dataset = self
            .state
            .label_to_session
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer session for `{viewer_label}`"))?
            .ensure_committed_dataset()?;
        let unlabeled = remove_stack_slice_labels_dataset(dataset.as_ref())?;
        let session = self
            .state
            .label_to_session
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer session for `{viewer_label}`"))?;
        session.commit_dataset(Arc::new(unlabeled));
        Ok("removed slice labels".to_string())
    }

    fn image_to_results(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for image to results".to_string())?
            .to_string();
        let (slice, roi_bbox, _, _, _) = self.measurement_context(&viewer_label)?;
        let rows = image_slice_to_results_rows(&slice, roi_bbox)?;
        let row_count = rows.len();
        self.results_table.clear();
        for row in rows {
            self.results_table.add_row(row);
        }
        self.desktop_state.utility_windows.results_open = true;
        self.persist_desktop_state();
        Ok(format!("image slice copied to results ({row_count} rows)"))
    }

    fn save_xy_coordinates(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for XY coordinates".to_string())?
            .to_string();
        let (slice, roi_bbox, _, _, _) = self.measurement_context(&viewer_label)?;
        let rows = xy_coordinate_rows(&slice, roi_bbox, params)?;
        let row_count = rows.len();
        self.results_table.clear();
        for row in rows {
            self.results_table.add_row(row);
        }
        self.desktop_state.utility_windows.results_open = true;
        self.persist_desktop_state();
        Ok(format!(
            "XY coordinates copied to results ({row_count} rows)"
        ))
    }

    fn results_to_image(&mut self) -> Result<String, String> {
        let dataset = results_rows_to_dataset(&self.results_table.rows)?;
        let path = normalize_path(&PathBuf::from(format!(
            "Results Table-{}.tif",
            self.state.next_window_id + 1
        )));
        let label = self.create_viewer(path, ViewerImageSource::Dataset(Arc::new(dataset)));
        Ok(format!("created results image in {label}"))
    }

    fn surface_plot_viewer(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for surface plot".to_string())?
            .to_string();
        let dataset = self
            .state
            .label_to_session
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer session for `{viewer_label}`"))?
            .ensure_committed_dataset()?;
        let output = self
            .state
            .app
            .ops_service()
            .execute("image.surface_plot", dataset.as_ref(), params)
            .map_err(|error| error.to_string())?;
        let path = normalize_path(&PathBuf::from(format!(
            "Surface Plot-{}.tif",
            self.state.next_window_id + 1
        )));
        let label = self.create_viewer(path, ViewerImageSource::Dataset(Arc::new(output.dataset)));
        Ok(format!("created surface plot in {label}"))
    }

    fn stack_to_images(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for Stack to Images".to_string())?
            .to_string();
        let (source_path, dataset) = {
            let session = self
                .state
                .label_to_session
                .get_mut(&viewer_label)
                .ok_or_else(|| format!("no viewer session for `{viewer_label}`"))?;
            let source_path = session.path.clone();
            let dataset = session.ensure_committed_dataset()?;
            (source_path, dataset)
        };
        let images = stack_to_image_datasets(dataset.as_ref())?;
        let count = images.len();
        for (index, image) in images.into_iter().enumerate() {
            let path = stack_slice_path(&source_path, index + 1);
            self.create_viewer(path, ViewerImageSource::Dataset(Arc::new(image)));
        }
        Ok(format!("created {count} image windows from stack"))
    }

    fn images_to_stack(&mut self) -> Result<String, String> {
        let mut labels = self
            .state
            .label_to_session
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        labels.sort_by_key(|label| viewer_sort_key(label));

        let mut images = Vec::new();
        for label in labels {
            let Some(session) = self.state.label_to_session.get_mut(&label) else {
                continue;
            };
            let dataset = session.ensure_committed_dataset()?;
            if dataset.axis_index(AxisKind::Z).is_some() {
                continue;
            }
            let title = session
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Image")
                .to_string();
            images.push((title, dataset));
        }

        let image_refs = images
            .iter()
            .map(|(title, dataset)| (title.as_str(), dataset.as_ref()))
            .collect::<Vec<_>>();
        let stack = images_to_stack_dataset(&image_refs)?;
        let count = image_refs.len();
        let path = normalize_path(&PathBuf::from(format!(
            "Stack-{}.tif",
            self.state.next_window_id + 1
        )));
        let label = self.create_viewer(path, ViewerImageSource::Dataset(Arc::new(stack)));
        Ok(format!("created stack {label} from {count} images"))
    }

    fn combine_stacks(&mut self, window_label: &str, params: &Value) -> Result<String, String> {
        let first_label = params
            .get("first")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| self.current_viewer_label(window_label).map(str::to_string))
            .ok_or_else(|| "two open images are required for Combine".to_string())?;
        let second_label = params
            .get("second")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| self.next_dataset_viewer_label(&first_label))
            .ok_or_else(|| "a second open image is required for Combine".to_string())?;
        if first_label == second_label {
            return Err("Combine requires two different images".to_string());
        }
        let vertical = params
            .get("vertical")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let fill = params.get("fill").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        let first = self.ensure_viewer_dataset(&first_label)?;
        let second = self.ensure_viewer_dataset(&second_label)?;
        let combined = combine_stack_datasets(first.as_ref(), second.as_ref(), vertical, fill)?;
        let path = normalize_path(&PathBuf::from(format!(
            "Combined Stacks-{}.tif",
            self.state.next_window_id + 1
        )));
        let label = self.create_viewer(path, ViewerImageSource::Dataset(Arc::new(combined)));
        Ok(format!("created combined stack in {label}"))
    }

    fn concatenate_stacks(&mut self, params: &Value) -> Result<String, String> {
        let fill = params.get("fill").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        let mut labels = self
            .state
            .label_to_session
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        labels.sort_by_key(|label| viewer_sort_key(label));

        let mut datasets = Vec::new();
        for label in labels {
            let Some(session) = self.state.label_to_session.get_mut(&label) else {
                continue;
            };
            let title = session
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Image")
                .to_string();
            let dataset = session.ensure_committed_dataset()?;
            datasets.push((title, dataset));
        }
        let refs = datasets
            .iter()
            .map(|(title, dataset)| (title.as_str(), dataset.as_ref()))
            .collect::<Vec<_>>();
        let concatenated = concatenate_stack_datasets(&refs, fill)?;
        let count = refs.len();
        let path = normalize_path(&PathBuf::from(format!(
            "Concatenated Stacks-{}.tif",
            self.state.next_window_id + 1
        )));
        let label = self.create_viewer(path, ViewerImageSource::Dataset(Arc::new(concatenated)));
        Ok(format!(
            "created concatenated stack {label} from {count} images"
        ))
    }

    fn insert_stack(&mut self, window_label: &str, params: &Value) -> Result<String, String> {
        let destination_label = params
            .get("destination")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| self.current_viewer_label(window_label).map(str::to_string))
            .ok_or_else(|| "a destination image is required for Insert".to_string())?;
        let source_label = params
            .get("source")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| self.next_dataset_viewer_label(&destination_label))
            .ok_or_else(|| "a source image is required for Insert".to_string())?;
        if source_label == destination_label {
            return Err("Insert requires different source and destination images".to_string());
        }
        let x = params.get("x").and_then(Value::as_i64).unwrap_or(0) as isize;
        let y = params.get("y").and_then(Value::as_i64).unwrap_or(0) as isize;
        let source = self.ensure_viewer_dataset(&source_label)?;
        let destination = self.ensure_viewer_dataset(&destination_label)?;
        let inserted = insert_stack_dataset(source.as_ref(), destination.as_ref(), x, y)?;
        let session = self
            .state
            .label_to_session
            .get_mut(&destination_label)
            .ok_or_else(|| format!("no viewer session for `{destination_label}`"))?;
        session.commit_dataset(Arc::new(inserted));
        if let Some(viewer) = self.viewers_ui.get_mut(&destination_label) {
            viewer.last_request = None;
            viewer.last_generation = 0;
            viewer.status_message = "stack insert complete".to_string();
        }
        self.set_active_viewer(Some(destination_label.clone()));
        Ok(format!("inserted {source_label} into {destination_label}"))
    }

    fn ensure_viewer_dataset(&mut self, label: &str) -> Result<Arc<DatasetF32>, String> {
        self.state
            .label_to_session
            .get_mut(label)
            .ok_or_else(|| format!("no viewer session for `{label}`"))?
            .ensure_committed_dataset()
    }

    fn next_dataset_viewer_label(&self, first_label: &str) -> Option<String> {
        let mut labels = self
            .state
            .label_to_session
            .keys()
            .filter(|label| label.as_str() != first_label)
            .cloned()
            .collect::<Vec<_>>();
        labels.sort_by_key(|label| viewer_sort_key(label));
        labels.into_iter().next()
    }

    fn summarize_results(&mut self) -> Result<String, String> {
        let summary_rows = results_summary_rows(&self.results_table.rows)?;
        let count = summary_rows.len();
        for row in summary_rows {
            self.results_table.add_row(row);
        }
        self.desktop_state.utility_windows.results_open = true;
        self.persist_desktop_state();
        Ok(format!("added {count} summary rows"))
    }

    fn clear_results(&mut self) -> String {
        self.results_table.clear();
        self.desktop_state.utility_windows.results_open = true;
        self.persist_desktop_state();
        "results table cleared".to_string()
    }

    fn results_distribution_payload(&mut self, params: &Value) -> Result<Value, String> {
        let bins = params.get("bins").and_then(Value::as_u64).unwrap_or(10) as usize;
        let column = params.get("column").and_then(Value::as_str);
        let payload = results_distribution(&self.results_table.rows, column, bins)?;
        self.desktop_state.utility_windows.results_open = true;
        self.persist_desktop_state();
        Ok(payload)
    }

    fn show_circular_masks(&mut self) -> Result<String, String> {
        let dataset = create_circular_masks_dataset()?;
        let path = normalize_path(&PathBuf::from(format!(
            "Circular Masks-{}.tif",
            self.state.next_window_id + 1
        )));
        let label = self.create_viewer(path, ViewerImageSource::Dataset(Arc::new(dataset)));
        Ok(format!("created circular masks stack in {label}"))
    }

    fn duplicate_viewer(&mut self, window_label: &str, params: &Value) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for duplicate".to_string())?
            .to_string();
        let (source, stem) = {
            let session = self
                .state
                .label_to_session
                .get(&viewer_label)
                .ok_or_else(|| format!("no viewer session for `{viewer_label}`"))?;
            let source = session
                .source_for_kind(&session.current_source_kind())
                .ok_or_else(|| format!("no image source for `{viewer_label}`"))?;
            let stem = session
                .path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("Image")
                .to_string();
            (source, stem)
        };
        let dataset = source.to_dataset()?;
        let title = params
            .get("title")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|title| !title.is_empty())
            .map(sanitize_image_title)
            .unwrap_or_else(|| format!("{stem}-copy-{}", self.state.next_window_id + 1));
        let path = normalize_path(&PathBuf::from(format!("{title}.tif")));
        let label = self.create_viewer(
            path,
            ViewerImageSource::Dataset(Arc::new(dataset.as_ref().clone())),
        );
        Ok(format!("duplicated image as {title} in {label}"))
    }

    fn rename_viewer(&mut self, window_label: &str, params: &Value) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for rename".to_string())?
            .to_string();
        let current_path = self
            .state
            .label_to_path
            .get(&viewer_label)
            .cloned()
            .ok_or_else(|| format!("no viewer path for `{viewer_label}`"))?;
        let title = params
            .get("title")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|title| !title.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("Untitled-{}", self.state.next_window_id + 1));
        let renamed_path = renamed_image_path(&current_path, &title);
        if self
            .state
            .path_to_label
            .get(&renamed_path)
            .is_some_and(|label| label != &viewer_label)
        {
            return Err(format!("{} is already open", renamed_path.display()));
        }

        self.state.path_to_label.remove(&current_path);
        self.state
            .path_to_label
            .insert(renamed_path.clone(), viewer_label.clone());
        self.state
            .label_to_path
            .insert(viewer_label.clone(), renamed_path.clone());
        if let Some(session) = self.state.label_to_session.get_mut(&viewer_label) {
            session.path = renamed_path.clone();
            session.committed_summary = summarize_source(&session.committed_source, &renamed_path);
        }
        if let Some(viewer) = self.viewers_ui.get_mut(&viewer_label) {
            viewer.title = format!(
                "{} - image-rs",
                renamed_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("Image")
            );
            viewer.status_message = "image renamed".to_string();
        }
        self.refresh_launcher_status();
        Ok(format!("renamed image to {}", renamed_path.display()))
    }

    fn image_info_payload(&self, window_label: &str) -> Result<Value, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for image info".to_string())?;
        let session = self
            .state
            .label_to_session
            .get(viewer_label)
            .ok_or_else(|| format!("no viewer session for `{viewer_label}`"))?;
        let source = session
            .source_for_kind(&session.current_source_kind())
            .ok_or_else(|| format!("no image source for `{viewer_label}`"))?;
        let summary = summarize_source(&source, &session.path);
        let dataset = source.to_dataset()?;
        let axes = dataset
            .metadata
            .dims
            .iter()
            .map(|dim| {
                json!({
                    "axis": format!("{:?}", dim.axis),
                    "size": dim.size,
                    "spacing": dim.spacing,
                    "unit": dim.unit
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "shape": summary.shape,
            "axes": summary.axes,
            "channels": summary.channels,
            "zSlices": summary.z_slices,
            "times": summary.times,
            "min": summary.min,
            "max": summary.max,
            "source": summary.source,
            "pixelType": format!("{:?}", dataset.metadata.pixel_type),
            "calibration": axes,
            "channelNames": &dataset.metadata.channel_names,
            "metadata": &dataset.metadata,
        }))
    }

    fn measure_all_rois(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for measure".to_string())?
            .to_string();
        let roi_ids = self
            .viewers_ui
            .get(&viewer_label)
            .map(|viewer| {
                viewer
                    .rois
                    .overlay_rois
                    .iter()
                    .map(|roi| roi.id)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if roi_ids.is_empty() {
            return Err("no ROIs available".to_string());
        }
        let original = self
            .viewers_ui
            .get(&viewer_label)
            .and_then(|viewer| viewer.rois.selected_roi_id);
        for roi_id in roi_ids {
            if let Some(viewer) = self.viewers_ui.get_mut(&viewer_label) {
                viewer.rois.selected_roi_id = Some(roi_id);
            }
            self.measure_active_viewer(&viewer_label)?;
        }
        if let Some(viewer) = self.viewers_ui.get_mut(&viewer_label) {
            viewer.rois.selected_roi_id = original;
        }
        Ok("measured all ROIs".to_string())
    }

    fn label_active_selection(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for label".to_string())?
            .to_string();
        let label_number = self.results_table.rows.len();
        if label_number == 0 {
            return Err("measurement counter is zero".to_string());
        }

        let viewer = self
            .viewers_ui
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
        if viewer.rois.selected_roi_id.is_none() && viewer.rois.active_roi.is_some() {
            viewer.rois.commit_active(true);
        }

        let (anchor, position) = {
            let roi = viewer
                .rois
                .selected_roi_id
                .and_then(|id| viewer.rois.overlay_rois.iter().find(|roi| roi.id == id))
                .or(viewer.rois.active_roi.as_ref())
                .ok_or_else(|| "a selection is required for label".to_string())?;
            let anchor = roi_label_anchor(&roi.kind)
                .ok_or_else(|| "a valid selection is required for label".to_string())?;
            (anchor, roi.position)
        };

        viewer.rois.begin_active(
            RoiKind::Text {
                at: anchor,
                text: label_number.to_string(),
            },
            position,
        );
        if let Some(active) = viewer.rois.active_roi.as_mut() {
            active.name = format!("Label {label_number}");
        }
        viewer.rois.commit_active(true);
        viewer.tool_message = Some(format!("labeled selection {label_number}"));

        Ok(format!("selection labeled {label_number}"))
    }

    fn add_selection_to_overlay(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for overlay add".to_string())?
            .to_string();
        let viewer = self
            .viewers_ui
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
        add_selection_to_overlay(&mut viewer.rois).map(str::to_string)
    }

    fn remove_overlay_rois_by_name(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<String, String> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if name.is_empty() {
            return Err("overlay ROI name cannot be empty".to_string());
        }
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for overlay remove".to_string())?
            .to_string();
        let viewer = self
            .viewers_ui
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
        let before = viewer.rois.overlay_rois.len();
        viewer.rois.overlay_rois.retain(|roi| roi.name != name);
        let removed = before.saturating_sub(viewer.rois.overlay_rois.len());
        if removed == 0 {
            return Err(format!("no overlay ROIs named `{name}`"));
        }
        if viewer
            .rois
            .selected_roi_id
            .is_none_or(|id| viewer.rois.overlay_rois.iter().all(|roi| roi.id != id))
        {
            viewer.rois.selected_roi_id = viewer.rois.overlay_rois.first().map(|roi| roi.id);
        }
        Ok(format!("removed {removed} overlay ROI(s) named {name}"))
    }

    fn remove_overlay_selection(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<String, String> {
        let index = params
            .get("index")
            .and_then(Value::as_u64)
            .ok_or_else(|| "overlay selection index is required".to_string())?
            as usize;
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for overlay remove".to_string())?
            .to_string();
        let viewer = self
            .viewers_ui
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
        if index >= viewer.rois.overlay_rois.len() {
            return Err(format!("overlay selection index {index} out of range"));
        }
        let removed = viewer.rois.overlay_rois.remove(index);
        if viewer.rois.selected_roi_id == Some(removed.id) {
            viewer.rois.selected_roi_id = viewer.rois.overlay_rois.first().map(|roi| roi.id);
        }
        Ok(format!("removed overlay selection {index}"))
    }

    fn activate_overlay_selection(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<String, String> {
        let index = params
            .get("index")
            .and_then(Value::as_u64)
            .ok_or_else(|| "overlay selection index is required".to_string())?
            as usize;
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for overlay activation".to_string())?
            .to_string();
        let viewer = self
            .viewers_ui
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
        let Some(roi) = viewer.rois.overlay_rois.get_mut(index) else {
            return Err(format!("overlay selection index {index} out of range"));
        };
        roi.visible = true;
        viewer.rois.selected_roi_id = Some(roi.id);
        Ok(format!("activated overlay selection {index}"))
    }

    fn overlay_from_roi_manager(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for overlay import".to_string())?
            .to_string();
        let viewer = self
            .viewers_ui
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
        let count = overlay_from_roi_manager(&mut viewer.rois)?;
        self.desktop_state.utility_windows.roi_manager_open = true;
        self.persist_desktop_state();
        Ok(format!("loaded {count} ROI Manager elements into overlay"))
    }

    fn overlay_to_roi_manager(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for overlay export".to_string())?
            .to_string();
        let viewer = self
            .viewers_ui
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
        let count = overlay_to_roi_manager(&mut viewer.rois)?;
        self.desktop_state.utility_windows.roi_manager_open = true;
        self.persist_desktop_state();
        Ok(format!("{count} overlay elements available in ROI Manager"))
    }

    fn flatten_overlay(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for overlay flatten".to_string())?
            .to_string();
        let (request, rois) = {
            let viewer = self
                .viewers_ui
                .get(&viewer_label)
                .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
            let position = interaction::roi::RoiPosition {
                channel: viewer.channel,
                z: viewer.z,
                t: viewer.t,
            };
            let rois = viewer
                .rois
                .visible_rois(position)
                .cloned()
                .collect::<Vec<_>>();
            (
                ViewerFrameRequest {
                    z: viewer.z,
                    t: viewer.t,
                    channel: viewer.channel,
                },
                rois,
            )
        };
        let slice = self.active_dataset_slice(&viewer_label, &request)?;
        let dataset = flatten_overlay_slice(&slice, &rois)?;
        let path = normalize_path(&PathBuf::from(format!(
            "Flattened-{}.tif",
            self.state.next_window_id + 1
        )));
        let label = self.create_viewer(path, ViewerImageSource::Dataset(Arc::new(dataset)));
        Ok(format!("created flattened overlay image in {label}"))
    }

    fn apply_lookup_table(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for apply LUT".to_string())?
            .to_string();
        let (request, lut) = {
            let viewer = self
                .viewers_ui
                .get(&viewer_label)
                .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
            (
                ViewerFrameRequest {
                    z: viewer.z,
                    t: viewer.t,
                    channel: viewer.channel,
                },
                viewer.lookup_table,
            )
        };
        let slice = self.active_dataset_slice(&viewer_label, &request)?;
        let dataset = lookup_table_slice_to_rgb(&slice, lut)?;
        let path = normalize_path(&PathBuf::from(format!(
            "Applied LUT-{}.tif",
            self.state.next_window_id + 1
        )));
        let label = self.create_viewer(path, ViewerImageSource::Dataset(Arc::new(dataset)));
        Ok(format!("created LUT-applied RGB image in {label}"))
    }

    fn remove_overlay(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for overlay removal".to_string())?
            .to_string();
        let viewer = self
            .viewers_ui
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
        let count = viewer.rois.overlay_rois.len();
        if count == 0 {
            return Err("no overlay elements".to_string());
        }
        viewer.rois.overlay_rois.clear();
        viewer.rois.selected_roi_id = None;
        Ok(format!("removed {count} overlay elements"))
    }

    fn set_overlay_visibility(
        &mut self,
        window_label: &str,
        mode: OverlayVisibility,
    ) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for overlay visibility".to_string())?
            .to_string();
        let viewer = self
            .viewers_ui
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
        let count = apply_overlay_visibility(&mut viewer.rois.overlay_rois, mode)?;
        Ok(match mode {
            OverlayVisibility::Show => format!("showing {count} overlay elements"),
            OverlayVisibility::Hide => format!("hid {count} overlay elements"),
            OverlayVisibility::Toggle => format!("toggled {count} overlay elements"),
        })
    }

    fn list_overlay_elements(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for overlay list".to_string())?
            .to_string();
        let viewer = self
            .viewers_ui
            .get(&viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
        let rows = overlay_element_rows(&viewer.rois.overlay_rois)?;
        let count = rows.len();
        self.results_table.clear();
        for row in rows {
            self.results_table.add_row(row);
        }
        self.desktop_state.utility_windows.results_open = true;
        self.persist_desktop_state();
        Ok(format!("listed {count} overlay elements"))
    }

    fn analyze_particles(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for particle analysis".to_string())?
            .to_string();
        let (slice, _, z, t, channel) = self.measurement_context(&viewer_label)?;
        let binary = threshold_slice_otsu(&slice.values);
        let particles = connected_components_2d(slice.width, slice.height, &binary);
        if particles.is_empty() {
            return Ok("no particles found".to_string());
        }
        if let Some(viewer) = self.viewers_ui.get_mut(&viewer_label) {
            for particle in &particles {
                let rect = RoiKind::Rect {
                    start: egui::pos2(particle.min_x as f32, particle.min_y as f32),
                    end: egui::pos2((particle.max_x + 1) as f32, (particle.max_y + 1) as f32),
                    rounded: false,
                    rotated: false,
                };
                viewer
                    .rois
                    .begin_active(rect, interaction::roi::RoiPosition { channel, z, t });
                if let Some(active) = viewer.rois.active_roi.as_mut() {
                    active.name = format!("Particle {}", particle.label);
                }
                viewer.rois.commit_active(true);
            }
        }
        for particle in particles {
            let row = measurement_row_from_slice(
                &particle.values,
                particle.width(),
                particle.height(),
                Some((
                    particle.min_x,
                    particle.min_y,
                    particle.max_x,
                    particle.max_y,
                )),
                &self.desktop_state.measurement_settings,
                z,
                t,
                channel,
            );
            self.results_table.add_row(row);
        }
        self.desktop_state.utility_windows.results_open = true;
        self.desktop_state.utility_windows.roi_manager_open = true;
        self.persist_desktop_state();
        Ok("particle analysis added to results".to_string())
    }

    fn copy_selection(&mut self, window_label: &str, cut: bool) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for copy/cut".to_string())?
            .to_string();
        let (slice, roi_bbox, z, t, channel) = self.measurement_context(&viewer_label)?;
        let (min_x, min_y, max_x, max_y) = roi_bbox.unwrap_or((
            0,
            0,
            slice.width.saturating_sub(1),
            slice.height.saturating_sub(1),
        ));
        let width = max_x.saturating_sub(min_x) + 1;
        let height = max_y.saturating_sub(min_y) + 1;
        let mut values = Vec::with_capacity(width * height);
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                values.push(slice.values[x + y * slice.width]);
            }
        }
        let data = ArrayD::from_shape_vec(IxDyn(&[height, width]), values.clone())
            .map_err(|error| format!("clipboard shape error: {error}"))?;
        let metadata = Metadata {
            dims: vec![Dim::new(AxisKind::Y, height), Dim::new(AxisKind::X, width)],
            pixel_type: PixelType::F32,
            ..Metadata::default()
        };
        self.clipboard.dataset = Some(Arc::new(
            Dataset::new(data, metadata).map_err(|error| error.to_string())?,
        ));

        if cut {
            let background = f32::from(self.tool_options.background_color.r()) / 255.0;
            self.apply_slice_patch(
                &viewer_label,
                z,
                t,
                channel,
                min_x,
                min_y,
                width,
                height,
                None,
                background,
                PasteMode::Copy,
            )?;
            return Ok("selection cut to clipboard".to_string());
        }

        Ok("selection copied to clipboard".to_string())
    }

    fn paste_selection(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for paste".to_string())?
            .to_string();
        let clipboard = self
            .clipboard
            .dataset
            .clone()
            .ok_or_else(|| "clipboard is empty".to_string())?;
        let (slice, roi_bbox, z, t, channel) = self.measurement_context(&viewer_label)?;
        let (x, y) = roi_bbox
            .map(|(min_x, min_y, _, _)| (min_x, min_y))
            .or_else(|| {
                self.viewers_ui.get(&viewer_label).and_then(|viewer| {
                    viewer
                        .hover
                        .map(|hover| (hover.x.min(slice.width - 1), hover.y.min(slice.height - 1)))
                })
            })
            .unwrap_or((0, 0));
        self.apply_slice_patch(
            &viewer_label,
            z,
            t,
            channel,
            x,
            y,
            clipboard.shape()[1],
            clipboard.shape()[0],
            Some(clipboard.as_ref()),
            0.0,
            self.clipboard.paste_mode,
        )?;
        Ok("clipboard pasted".to_string())
    }

    fn show_internal_clipboard(&mut self) -> Result<String, String> {
        let dataset = self
            .clipboard
            .dataset
            .clone()
            .ok_or_else(|| "clipboard is empty".to_string())?;
        let next = self.state.next_window_id.saturating_add(1);
        let path = PathBuf::from(format!("Internal Clipboard {next}"));
        let label = self.create_viewer(path, ViewerImageSource::Dataset(dataset));
        Ok(format!("internal clipboard opened as {label}"))
    }

    fn fill_selection_or_slice(
        &mut self,
        window_label: &str,
        fill: f32,
        action: &str,
    ) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| format!("a loaded image is required for {action}"))?
            .to_string();
        let (slice, roi_bbox, z, t, channel) = self.measurement_context(&viewer_label)?;
        let (min_x, min_y, max_x, max_y) = roi_bbox.unwrap_or((
            0,
            0,
            slice.width.saturating_sub(1),
            slice.height.saturating_sub(1),
        ));
        let width = max_x.saturating_sub(min_x) + 1;
        let height = max_y.saturating_sub(min_y) + 1;
        self.apply_slice_patch(
            &viewer_label,
            z,
            t,
            channel,
            min_x,
            min_y,
            width,
            height,
            None,
            fill,
            PasteMode::Copy,
        )?;
        Ok(format!("{action} applied"))
    }

    fn interpolate_selection(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for Interpolate".to_string())?
            .to_string();
        let interval = params
            .get("interval")
            .and_then(Value::as_f64)
            .unwrap_or(1.0) as f32;
        let smooth = params
            .get("smooth")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let adjust = params
            .get("adjust")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let viewer = self
            .viewers_ui
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;

        let selected_id = viewer.rois.selected_roi_id;
        let roi = selected_id
            .and_then(|id| viewer.rois.overlay_rois.iter_mut().find(|roi| roi.id == id))
            .or(viewer.rois.active_roi.as_mut())
            .ok_or_else(|| "Interpolate requires a selection".to_string())?;
        roi.kind = interpolate_roi_kind(&roi.kind, interval, smooth, adjust)?;
        Ok("selection interpolated".to_string())
    }

    fn selection_properties(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for selection properties".to_string())?
            .to_string();
        let viewer = self
            .viewers_ui
            .get(&viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
        let roi = viewer
            .rois
            .selected_roi_id
            .and_then(|id| viewer.rois.overlay_rois.iter().find(|roi| roi.id == id))
            .or(viewer.rois.active_roi.as_ref())
            .ok_or_else(|| "selection properties require a selection".to_string())?;
        let row = selection_properties_row(roi)?;
        self.results_table.add_row(row);
        self.desktop_state.utility_windows.results_open = true;
        self.persist_desktop_state();
        Ok("selection properties added to results".to_string())
    }

    fn apply_macro_set_option(&mut self, window_label: &str, params: &Value) -> String {
        let option = params
            .get("option")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let state = params.get("state").and_then(Value::as_bool).unwrap_or(true);
        let normalized = option.trim().to_ascii_lowercase();

        if normalized == "stack position" {
            self.desktop_state.measurement_settings.slice = state;
            self.desktop_state.measurement_settings.channel = state;
            self.desktop_state.measurement_settings.time = state;
            self.persist_desktop_state();
            return format!("setOption Stack position={state}");
        }

        if normalized.starts_with("show all") {
            if let Some(viewer_label) = self.current_viewer_label(window_label).map(str::to_string)
                && let Some(viewer) = self.viewers_ui.get_mut(&viewer_label)
                && !viewer.rois.overlay_rois.is_empty()
            {
                for roi in &mut viewer.rois.overlay_rois {
                    roi.visible = state;
                }
                return format!("setOption Show All={state}");
            }
            return format!("setOption Show All={state} acknowledged");
        }

        if matches!(
            normalized.as_str(),
            "debugmode" | "interpolatelines" | "monospacedtext"
        ) {
            return format!("setOption {option}={state} acknowledged");
        }

        format!("setOption {option}={state} acknowledged")
    }

    fn apply_macro_call(&mut self, params: &Value) -> String {
        let target = params
            .get("target")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match target {
            "ij.plugin.MacroInstaller.installFromJar"
            | "ij.plugin.frame.Recorder.recordString"
            | "ij.plugin.frame.Recorder.scriptMode"
            | "ij.Prefs.get"
            | "ij.Prefs.set"
            | "ij.gui.Line.getWidth" => format!("call {target} acknowledged"),
            _ if target.is_empty() => "call acknowledged".to_string(),
            _ => format!("call {target} acknowledged"),
        }
    }

    fn apply_macro_builtin_call(&mut self, params: &Value) -> String {
        let target = params
            .get("target")
            .and_then(Value::as_str)
            .unwrap_or_default();
        format!("{target} acknowledged")
    }

    fn apply_macro_set_color(&mut self, params: &Value) -> Result<String, String> {
        let target = params
            .get("target")
            .and_then(Value::as_str)
            .unwrap_or("foreground")
            .trim()
            .to_ascii_lowercase();
        let red = macro_color_component(params, "red")?;
        let green = macro_color_component(params, "green")?;
        let blue = macro_color_component(params, "blue")?;
        let color = egui::Color32::from_rgb(red, green, blue);

        match target.as_str() {
            "foreground" => {
                self.tool_options.foreground_color = color;
                Ok(format!("set foreground color to {red},{green},{blue}"))
            }
            "background" => {
                self.tool_options.background_color = color;
                Ok(format!("set background color to {red},{green},{blue}"))
            }
            _ => Err(format!("unknown macro color target `{target}`")),
        }
    }

    fn apply_macro_set_roi_name(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<String, String> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();
        if name.is_empty() {
            return Err("ROI name cannot be empty".to_string());
        }

        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for ROI naming".to_string())?
            .to_string();
        let viewer = self
            .viewers_ui
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;

        if let Some(id) = viewer.rois.selected_roi_id
            && let Some(roi) = viewer.rois.overlay_rois.iter_mut().find(|roi| roi.id == id)
        {
            roi.name = name.to_string();
            return Ok(format!("ROI named {name}"));
        }

        let Some(roi) = viewer.rois.active_roi.as_mut() else {
            return Err("ROI naming requires an active selection".to_string());
        };
        roi.name = name.to_string();
        Ok(format!("ROI named {name}"))
    }

    fn apply_macro_set_metadata(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<String, String> {
        let key = params
            .get("key")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();
        if key.is_empty() {
            return Err("metadata key cannot be empty".to_string());
        }
        let value = params
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for metadata".to_string())?
            .to_string();
        let session = self
            .state
            .label_to_session
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer session for `{viewer_label}`"))?;
        let mut dataset = session.ensure_committed_dataset()?.as_ref().clone();
        dataset
            .metadata
            .extras
            .insert(key.to_string(), json!(value));
        session.commit_dataset(Arc::new(dataset));
        Ok(format!("metadata {key} set"))
    }

    fn apply_macro_set_tool(&mut self, params: &Value) -> Result<String, String> {
        let tool = params
            .get("tool")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if tool_from_command_id(tool).is_none() {
            return Err("setTool requires a known tool".to_string());
        }

        if let Some(mode) = params
            .get("mode")
            .and_then(Value::as_str)
            .filter(|mode| !mode.is_empty())
        {
            let result = self.dispatch_command(LAUNCHER_LABEL, mode, None);
            if !matches!(result.status, command_registry::CommandExecuteStatus::Ok) {
                return Err(result.message);
            }
        }

        let result = self.dispatch_command(LAUNCHER_LABEL, tool, None);
        if matches!(result.status, command_registry::CommandExecuteStatus::Ok) {
            Ok(result.message)
        } else {
            Err(result.message)
        }
    }

    fn apply_macro_set_paste_mode(&mut self, params: &Value) -> Result<String, String> {
        let mode = params
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        self.clipboard.paste_mode = if mode.contains("add") {
            PasteMode::Add
        } else {
            PasteMode::Copy
        };
        Ok(format!("paste mode set to {mode}"))
    }

    fn make_macro_roi(
        &mut self,
        window_label: &str,
        params: &Value,
        oval: bool,
    ) -> Result<String, String> {
        let x = params.get("x").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        let y = params.get("y").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        let width = params.get("width").and_then(Value::as_f64).unwrap_or(1.0) as f32;
        let height = params.get("height").and_then(Value::as_f64).unwrap_or(1.0) as f32;
        if !x.is_finite() || !y.is_finite() || !width.is_finite() || !height.is_finite() {
            return Err("macro selection coordinates must be finite".to_string());
        }
        if width <= 0.0 || height <= 0.0 {
            return Err("macro selection width and height must be positive".to_string());
        }

        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for macro selection".to_string())?
            .to_string();
        let viewer = self
            .viewers_ui
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
        let position = interaction::roi::RoiPosition {
            channel: viewer.channel,
            z: viewer.z,
            t: viewer.t,
        };
        let start = egui::pos2(x, y);
        let end = egui::pos2(x + width, y + height);
        let kind = if oval {
            RoiKind::Oval {
                start,
                end,
                ellipse: false,
                brush: false,
            }
        } else {
            RoiKind::Rect {
                start,
                end,
                rounded: false,
                rotated: false,
            }
        };
        viewer.rois.begin_active(kind, position);
        Ok(if oval {
            "oval selection created".to_string()
        } else {
            "rectangular selection created".to_string()
        })
    }

    fn make_macro_line(&mut self, window_label: &str, params: &Value) -> Result<String, String> {
        let x1 = params.get("x1").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        let y1 = params.get("y1").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        let x2 = params.get("x2").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        let y2 = params.get("y2").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        if !x1.is_finite() || !y1.is_finite() || !x2.is_finite() || !y2.is_finite() {
            return Err("macro line coordinates must be finite".to_string());
        }

        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for macro line".to_string())?
            .to_string();
        let viewer = self
            .viewers_ui
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
        let position = interaction::roi::RoiPosition {
            channel: viewer.channel,
            z: viewer.z,
            t: viewer.t,
        };
        viewer.rois.begin_active(
            RoiKind::Line {
                start: egui::pos2(x1, y1),
                end: egui::pos2(x2, y2),
                arrow: false,
            },
            position,
        );
        Ok("line selection created".to_string())
    }

    fn make_macro_selection(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<String, String> {
        let selection_type = params
            .get("selection_type")
            .and_then(Value::as_str)
            .unwrap_or("polygon")
            .to_ascii_lowercase();
        let points = params
            .get("points")
            .and_then(Value::as_array)
            .ok_or_else(|| "macro selection requires points".to_string())?
            .iter()
            .map(|point| {
                let x = point
                    .get("x")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| "macro selection point requires x".to_string())?
                    as f32;
                let y = point
                    .get("y")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| "macro selection point requires y".to_string())?
                    as f32;
                if !x.is_finite() || !y.is_finite() {
                    return Err("macro selection point coordinates must be finite".to_string());
                }
                Ok(egui::pos2(x, y))
            })
            .collect::<Result<Vec<_>, String>>()?;
        if selection_type.contains("point") {
            if points.is_empty() {
                return Err("macro point selection requires at least one point".to_string());
            }
        } else if points.len() < 2 {
            return Err("macro selection requires at least two points".to_string());
        }

        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for macro selection".to_string())?
            .to_string();
        let viewer = self
            .viewers_ui
            .get_mut(&viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
        let position = interaction::roi::RoiPosition {
            channel: viewer.channel,
            z: viewer.z,
            t: viewer.t,
        };
        let kind = if selection_type.contains("point") {
            RoiKind::Point {
                points,
                multi: true,
            }
        } else if selection_type.contains("free") {
            RoiKind::Freehand { points }
        } else {
            RoiKind::Polygon {
                points,
                closed: !selection_type.contains("line"),
                spline_fit: false,
            }
        };
        viewer.rois.begin_active(kind, position);
        Ok("macro selection created".to_string())
    }

    fn layout_viewers(&mut self, mode: LayoutMode) -> Result<String, String> {
        let mut labels = self.state.label_to_path.keys().cloned().collect::<Vec<_>>();
        labels.sort_by_key(|label| viewer_sort_key(label));
        if labels.is_empty() {
            return Err("no viewers are open".to_string());
        }
        self.focus_viewer_label = self
            .active_viewer_label
            .clone()
            .or_else(|| labels.first().cloned());
        self.launcher_ui.fallback_text = match mode {
            LayoutMode::Tile => "Tabbed viewer layout is active".to_string(),
            LayoutMode::Cascade => "Tabbed viewer layout is active".to_string(),
        };
        Ok(match mode {
            LayoutMode::Tile => "image tabs remain in the main window".to_string(),
            LayoutMode::Cascade => "image tabs remain in the main window".to_string(),
        })
    }

    fn measurement_context(
        &self,
        viewer_label: &str,
    ) -> Result<
        (
            SliceImage,
            Option<(usize, usize, usize, usize)>,
            usize,
            usize,
            usize,
        ),
        String,
    > {
        let viewer = self
            .viewers_ui
            .get(viewer_label)
            .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))?;
        let request = ViewerFrameRequest {
            z: viewer.z,
            t: viewer.t,
            channel: viewer.channel,
        };
        let slice = self.active_dataset_slice(viewer_label, &request)?;
        let roi_bbox = selected_roi_bbox(viewer);
        Ok((slice, roi_bbox, viewer.z, viewer.t, viewer.channel))
    }

    fn apply_slice_patch(
        &mut self,
        viewer_label: &str,
        z: usize,
        t: usize,
        channel: usize,
        dst_x: usize,
        dst_y: usize,
        patch_width: usize,
        patch_height: usize,
        patch: Option<&DatasetF32>,
        fill: f32,
        paste_mode: PasteMode,
    ) -> Result<String, String> {
        let session = self
            .state
            .label_to_session
            .get_mut(viewer_label)
            .ok_or_else(|| format!("no viewer session for `{viewer_label}`"))?;
        let mut dataset = (*session.ensure_committed_dataset()?).clone();
        let x_axis = dataset.axis_index(AxisKind::X).unwrap_or(1);
        let y_axis = dataset.axis_index(AxisKind::Y).unwrap_or(0);
        let z_axis = dataset.axis_index(AxisKind::Z);
        let t_axis = dataset.axis_index(AxisKind::Time);
        let c_axis = dataset.axis_index(AxisKind::Channel);

        for py in 0..patch_height {
            for px in 0..patch_width {
                let x = dst_x + px;
                let y = dst_y + py;
                if x >= dataset.shape()[x_axis] || y >= dataset.shape()[y_axis] {
                    continue;
                }
                let mut index = vec![0usize; dataset.ndim()];
                index[x_axis] = x;
                index[y_axis] = y;
                if let Some(axis) = z_axis {
                    index[axis] = z.min(dataset.shape()[axis].saturating_sub(1));
                }
                if let Some(axis) = t_axis {
                    index[axis] = t.min(dataset.shape()[axis].saturating_sub(1));
                }
                if let Some(axis) = c_axis {
                    index[axis] = channel.min(dataset.shape()[axis].saturating_sub(1));
                }
                dataset.data[IxDyn(&index)] = if let Some(patch) = patch {
                    let value = patch.data[IxDyn(&[py, px])];
                    match paste_mode {
                        PasteMode::Copy => value,
                        PasteMode::Add => dataset.data[IxDyn(&index)] + value,
                    }
                } else {
                    fill
                };
            }
        }
        let arc = Arc::new(dataset);
        if let Some(session) = self.state.label_to_session.get_mut(viewer_label) {
            session.commit_dataset(arc);
        }
        if let Some(viewer) = self.viewers_ui.get_mut(viewer_label) {
            viewer.last_request = None;
            viewer.last_generation = 0;
        }
        Ok("image updated".to_string())
    }

    fn cycle_window(&mut self, current_label: &str, direction: i32) -> String {
        let mut labels = self.state.label_to_path.keys().cloned().collect::<Vec<_>>();
        labels.sort_by_key(|label| viewer_sort_key(label));
        if labels.is_empty() {
            self.focus_launcher = true;
            return LAUNCHER_LABEL.to_string();
        }

        let offset = if direction < 0 { -1 } else { 1 };
        let target_index = labels
            .iter()
            .position(|label| label == current_label)
            .map(|index| {
                let len = labels.len() as isize;
                let raw = index as isize + offset;
                raw.rem_euclid(len) as usize
            })
            .unwrap_or_else(|| if direction < 0 { labels.len() - 1 } else { 0 });

        let target = labels[target_index].clone();
        self.focus_viewer_label = Some(target.clone());
        self.set_active_viewer(Some(target.clone()));
        target
    }

    fn show_all_windows(&mut self) -> Result<String, String> {
        if self.viewers_ui.is_empty() {
            self.focus_launcher = true;
            return Err("no viewers are open".to_string());
        }

        self.focus_launcher = true;
        Ok("all image tabs are shown in the main window".to_string())
    }

    fn dispatch_command(
        &mut self,
        window_label: &str,
        command_id: &str,
        params: Option<Value>,
    ) -> command_registry::CommandExecuteResult {
        let metadata = command_registry::metadata(command_id);
        let target_label = if window_label == LAUNCHER_LABEL
            && matches!(metadata.scope, command_registry::CommandScope::Viewer)
        {
            self.active_viewer_label
                .clone()
                .unwrap_or_else(|| window_label.to_string())
        } else {
            window_label.to_string()
        };

        if let Some(result) = self.handle_local_command(&target_label, command_id, params.as_ref())
        {
            return result;
        }

        let request = command_registry::CommandExecuteRequest {
            command_id: command_id.to_string(),
            params,
        };
        self.execute_command(&target_label, request)
    }

    fn open_adjust_dialog(
        &mut self,
        window_label: &str,
        kind: AdjustDialogKind,
    ) -> command_registry::CommandExecuteResult {
        let target = self
            .current_viewer_label(window_label)
            .unwrap_or(window_label)
            .to_string();
        let histogram = self.adjust_histogram_for_viewer(&target, 256, false).ok();
        self.adjust_dialog.kind = kind;
        self.adjust_dialog.window_label = target.clone();
        self.adjust_dialog.open = true;
        self.adjust_dialog.histogram = histogram;
        self.adjust_dialog.line_width = self.tool_options.line_width_px;
        self.adjust_dialog.spline_fit = self
            .viewers_ui
            .get(&target)
            .is_some_and(selected_roi_spline_fit);
        self.adjust_dialog.color_balance_lut_color = false;
        self.adjust_dialog.contrast_auto_threshold = 0;

        if let Some(session) = self.state.label_to_session.get(&target) {
            self.adjust_dialog.default_min = self
                .adjust_dialog
                .histogram
                .as_ref()
                .map(|histogram| histogram.min)
                .unwrap_or(session.committed_summary.min);
            self.adjust_dialog.default_max = self
                .adjust_dialog
                .histogram
                .as_ref()
                .map(|histogram| histogram.max)
                .unwrap_or(session.committed_summary.max);
            let (display_min, display_max) = session
                .display_range
                .unwrap_or((session.committed_summary.min, session.committed_summary.max));
            self.adjust_dialog.min = display_min;
            self.adjust_dialog.max = display_max;
            if display_min.is_finite() && display_max.is_finite() {
                self.adjust_dialog.threshold = (display_min + display_max) * 0.5;
            }
            if kind == AdjustDialogKind::ColorBalance {
                self.adjust_dialog.color_balance_lut_color = color_balance_uses_lut_color(session);
                self.adjust_dialog.color_balance_channel_labels =
                    color_balance_channel_labels_for_session(session);
                self.adjust_dialog.color_balance_channel = self
                    .adjust_dialog
                    .color_balance_channel_labels
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "Red".to_string());
            }
            sync_brightness_contrast_from_min_max(&mut self.adjust_dialog);
            let width = session.committed_summary.shape.get(1).copied().unwrap_or(1) as f32;
            let height = session
                .committed_summary
                .shape
                .first()
                .copied()
                .unwrap_or(1) as f32;
            self.adjust_dialog.right = width;
            self.adjust_dialog.bottom = height;
            self.adjust_dialog.back = session.committed_summary.z_slices.max(1) as f32;
            if kind == AdjustDialogKind::Coordinates {
                if let Ok(dataset) = session.committed_source.to_dataset() {
                    init_coordinates_dialog(
                        &mut self.adjust_dialog,
                        &dataset.metadata,
                        self.viewers_ui.get(&target),
                        session.committed_summary.z_slices.max(1),
                        width,
                        height,
                    );
                }
            }
        }

        command_registry::CommandExecuteResult::ok(format!("{} dialog opened", kind.title()))
    }

    fn threshold_adjuster_supports_viewer(&self, window_label: &str) -> Result<(), String> {
        let target = self
            .current_viewer_label(window_label)
            .unwrap_or(window_label)
            .to_string();
        let session = self
            .state
            .label_to_session
            .get(&target)
            .ok_or_else(|| format!("no viewer session for `{target}`"))?;
        if session.committed_summary.channels >= 3 {
            return Err(
                "Image>Adjust>Threshold only works with grayscale images; use Image>Adjust>Color Threshold for RGB images"
                    .to_string(),
            );
        }
        Ok(())
    }

    fn open_set_display_range_dialog_from_adjust(
        &mut self,
        command_id: &str,
        low_key: &str,
        high_key: &str,
    ) {
        let channel_count = self
            .state
            .label_to_session
            .get(&self.adjust_dialog.window_label)
            .map(|session| session.committed_summary.channels.max(1))
            .unwrap_or(1);
        let channel = if self.adjust_dialog.kind == AdjustDialogKind::ColorBalance {
            self.adjust_dialog.color_balance_channel.clone()
        } else {
            String::new()
        };
        self.set_display_range_dialog = SetDisplayRangeDialogState {
            open: true,
            window_label: self.adjust_dialog.window_label.clone(),
            command_id: command_id.to_string(),
            low_key: low_key.to_string(),
            high_key: high_key.to_string(),
            minimum: self.adjust_dialog.min,
            maximum: self.adjust_dialog.max,
            unsigned_16bit_range: "Automatic".to_string(),
            propagate: false,
            all_channels: false,
            show_all_channels: self.adjust_dialog.kind == AdjustDialogKind::ColorBalance
                && channel_count > 1
                && channel != "All",
            channel,
            channel_count,
        };
    }

    fn open_threshold_set_dialog_from_adjust(&mut self) {
        self.threshold_set_dialog = ThresholdSetDialogState {
            open: true,
            window_label: self.adjust_dialog.window_label.clone(),
            min: self.adjust_dialog.min,
            max: self.adjust_dialog.max,
            mode: self.adjust_dialog.threshold_mode.clone(),
            dark_background: self.adjust_dialog.dark_background,
        };
    }

    fn open_set_window_level_dialog_from_adjust(&mut self) {
        let window = self.adjust_dialog.max - self.adjust_dialog.min;
        let level = self.adjust_dialog.min + window * 0.5;
        self.set_window_level_dialog = SetWindowLevelDialogState {
            open: true,
            window_label: self.adjust_dialog.window_label.clone(),
            level,
            window,
            propagate: false,
        };
    }

    fn apply_display_range(
        &mut self,
        window_label: &str,
        params: &Value,
        low_key: &str,
        high_key: &str,
    ) -> Result<String, String> {
        let target = self
            .current_viewer_label(window_label)
            .unwrap_or(window_label)
            .to_string();
        let low = params
            .get(low_key)
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or_else(|| {
                self.state
                    .label_to_session
                    .get(&target)
                    .map(|session| session.committed_summary.min)
                    .unwrap_or(0.0)
            });
        let high = params
            .get(high_key)
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or_else(|| {
                self.state
                    .label_to_session
                    .get(&target)
                    .map(|session| session.committed_summary.max)
                    .unwrap_or(1.0)
            });

        if !low.is_finite() || !high.is_finite() || high <= low {
            return Err(format!(
                "`{high_key}` must be a finite value greater than `{low_key}`"
            ));
        }

        let channel = params
            .get("channel")
            .and_then(Value::as_str)
            .and_then(color_balance_channel_index);
        let all_channels = params
            .get("all_channels")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let propagate = params
            .get("propagate")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let channel_count = {
            let session = self
                .state
                .label_to_session
                .get_mut(&target)
                .ok_or_else(|| format!("no viewer session for `{target}`"))?;
            if all_channels {
                let channel_count = session.committed_summary.channels.max(1);
                for channel in 0..channel_count {
                    session.set_channel_display_range(channel, Some((low, high)));
                }
                channel_count
            } else if let Some(channel) = channel {
                session.set_channel_display_range(channel, Some((low, high)));
                session.committed_summary.channels.max(1)
            } else {
                session.set_display_range(Some((low, high)));
                session.committed_summary.channels.max(1)
            }
        };

        if propagate {
            let labels = self
                .state
                .label_to_session
                .keys()
                .filter(|label| *label != &target)
                .cloned()
                .collect::<Vec<_>>();
            for label in labels {
                if let Some(session) = self.state.label_to_session.get_mut(&label) {
                    session.set_display_range(Some((low, high)));
                }
                if let Some(viewer) = self.viewers_ui.get_mut(&label) {
                    viewer.last_generation = 0;
                    viewer.last_request = None;
                    viewer.status_message = format!("Display range {low:.4}..{high:.4}");
                }
            }
        }
        if let Some(viewer) = self.viewers_ui.get_mut(&target) {
            viewer.last_generation = 0;
            viewer.last_request = None;
            viewer.status_message = if all_channels {
                format!("All channel display ranges {low:.4}..{high:.4}")
            } else if let Some(channel) = channel {
                format!("Channel {} display range {low:.4}..{high:.4}", channel + 1)
            } else {
                format!("Display range {low:.4}..{high:.4}")
            };
        }

        Ok(if all_channels {
            format!("{channel_count} channel display ranges set to {low:.4}..{high:.4}")
        } else if let Some(channel) = channel {
            format!(
                "channel {} display range set to {low:.4}..{high:.4}",
                channel + 1
            )
        } else {
            format!("display range set to {low:.4}..{high:.4}")
        })
    }

    fn reset_display_range(&mut self, window_label: &str) -> Result<String, String> {
        let target = self
            .current_viewer_label(window_label)
            .unwrap_or(window_label)
            .to_string();
        let session = self
            .state
            .label_to_session
            .get_mut(&target)
            .ok_or_else(|| format!("no viewer session for `{target}`"))?;
        session.set_display_range(None);
        if let Some(viewer) = self.viewers_ui.get_mut(&target) {
            viewer.last_generation = 0;
            viewer.last_request = None;
            viewer.status_message = "Display range reset".to_string();
        }
        Ok("display range reset".to_string())
    }

    fn ensure_apply_lut_supported(&self, window_label: &str) -> Result<(), String> {
        let target = self
            .current_viewer_label(window_label)
            .unwrap_or(window_label)
            .to_string();
        let session = self
            .state
            .label_to_session
            .get(&target)
            .ok_or_else(|| format!("no viewer session for `{target}`"))?;
        let dataset = session.committed_source.to_dataset()?;
        if dataset.metadata.pixel_type == PixelType::F32 {
            return Err("\"Apply\" does not work with 32-bit images".to_string());
        }
        Ok(())
    }

    fn set_threshold_overlay(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<String, String> {
        let target = self
            .current_viewer_label(window_label)
            .unwrap_or(window_label)
            .to_string();

        if params
            .get("reset")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            let no_reset = params
                .get("no_reset")
                .or_else(|| params.get("dont_reset_range"))
                .and_then(Value::as_bool)
                .unwrap_or(true);
            return self.reset_threshold_overlay(&target, no_reset);
        }

        let (low, high) = if let (Some(low), Some(high)) = (
            params.get("min").and_then(Value::as_f64),
            params.get("max").and_then(Value::as_f64),
        ) {
            (low as f32, high as f32)
        } else {
            let stack_histogram = params
                .get("stack")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let viewer = self
                .viewers_ui
                .get(&target)
                .ok_or_else(|| format!("no viewer UI state for `{target}`"))?;
            let bbox = selected_roi_bbox(viewer);
            let request = ViewerFrameRequest {
                z: viewer.z,
                t: viewer.t,
                channel: viewer.channel,
            };
            let slice = self.active_dataset_slice(&target, &request)?;
            let pixel_type = slice.pixel_type;
            let values = if stack_histogram {
                self.viewer_stack_values(&target, viewer.channel, viewer.t, bbox)?
            } else {
                slice_values_in_bbox(&slice, bbox)?
            };
            let dataset = threshold_values_dataset(values, pixel_type)?;
            let output = self
                .state
                .app
                .ops_service()
                .execute("threshold.make_binary", &dataset, params)
                .map_err(|error| error.to_string())?;
            let measurements = output
                .measurements
                .ok_or_else(|| "threshold did not report threshold range".to_string())?;
            let low = measurements
                .values
                .get("threshold_min")
                .and_then(Value::as_f64)
                .ok_or_else(|| "threshold did not report threshold_min".to_string())?
                as f32;
            let high = measurements
                .values
                .get("threshold_max")
                .and_then(Value::as_f64)
                .ok_or_else(|| "threshold did not report threshold_max".to_string())?
                as f32;
            (low, high)
        };

        if !low.is_finite() || !high.is_finite() || high < low {
            return Err("threshold min/max must be finite and ordered".to_string());
        }

        let mode = params
            .get("mode")
            .and_then(Value::as_str)
            .map(ThresholdOverlayMode::from_label)
            .unwrap_or(ThresholdOverlayMode::Red);
        let session = self
            .state
            .label_to_session
            .get_mut(&target)
            .ok_or_else(|| format!("no viewer session for `{target}`"))?;
        session.set_threshold_overlay(Some(ThresholdOverlay { low, high, mode }));
        if let Some(viewer) = self.viewers_ui.get_mut(&target) {
            viewer.last_generation = 0;
            viewer.status_message = format!("Threshold {low:.4}..{high:.4}");
        }
        if self.adjust_dialog.open && self.adjust_dialog.window_label == target {
            self.adjust_dialog.min = low;
            self.adjust_dialog.max = high;
        }
        Ok(format!("threshold set to {low:.4}..{high:.4}"))
    }

    fn reset_threshold_overlay(
        &mut self,
        window_label: &str,
        no_reset_range: bool,
    ) -> Result<String, String> {
        let target = self
            .current_viewer_label(window_label)
            .unwrap_or(window_label)
            .to_string();
        let session = self
            .state
            .label_to_session
            .get_mut(&target)
            .ok_or_else(|| format!("no viewer session for `{target}`"))?;
        session.set_threshold_overlay(None);
        if !no_reset_range {
            session.set_display_range(None);
        }
        if let Some(viewer) = self.viewers_ui.get_mut(&target) {
            viewer.last_generation = 0;
            viewer.status_message = if no_reset_range {
                "Threshold reset".to_string()
            } else {
                "Threshold and display range reset".to_string()
            };
        }
        Ok(if no_reset_range {
            "threshold reset".to_string()
        } else {
            "threshold and display range reset".to_string()
        })
    }

    fn threshold_apply_bounds(
        &self,
        window_label: &str,
        params: &Value,
    ) -> Result<(f32, f32), String> {
        if let (Some(lower), Some(upper)) = (
            params.get("min").and_then(Value::as_f64),
            params.get("max").and_then(Value::as_f64),
        ) {
            let lower = lower as f32;
            let upper = upper as f32;
            if lower.is_finite() && upper.is_finite() && upper >= lower {
                return Ok((lower, upper));
            }
            return Err("threshold min/max must be finite and ordered".to_string());
        }

        let target = self
            .current_viewer_label(window_label)
            .unwrap_or(window_label)
            .to_string();
        let session = self
            .state
            .label_to_session
            .get(&target)
            .ok_or_else(|| format!("no viewer session for `{target}`"))?;
        let threshold = session
            .threshold_overlay
            .ok_or_else(|| "Threshold is not set".to_string())?;
        Ok((threshold.low, threshold.high))
    }

    fn threshold_apply_needs_float_prompt(&self, window_label: &str) -> bool {
        let target = self
            .current_viewer_label(window_label)
            .unwrap_or(window_label)
            .to_string();
        self.state
            .label_to_session
            .get(&target)
            .and_then(|session| session.committed_source.to_dataset().ok())
            .is_some_and(|dataset| dataset.metadata.pixel_type == PixelType::F32)
    }

    fn apply_lut_needs_confirmation(&self, window_label: &str) -> bool {
        let target = self
            .current_viewer_label(window_label)
            .unwrap_or(window_label)
            .to_string();
        self.state
            .label_to_session
            .get(&target)
            .and_then(|session| session.committed_source.to_dataset().ok())
            .is_some_and(|dataset| {
                matches!(dataset.metadata.pixel_type, PixelType::U8 | PixelType::U16)
            })
    }

    fn apply_lut_dialog_state(
        &self,
        window_label: String,
        command_id: String,
        params: Value,
    ) -> ApplyLutDialogState {
        let target = self
            .current_viewer_label(&window_label)
            .unwrap_or(&window_label)
            .to_string();
        let stack_slices = self
            .state
            .label_to_session
            .get(&target)
            .map(|session| session.committed_summary.z_slices.max(1))
            .unwrap_or(1);
        let mut slice_params = params.clone();
        if stack_slices > 1 {
            if let Some(viewer) = self.viewers_ui.get(&target)
                && let Some(map) = slice_params.as_object_mut()
            {
                map.insert("slice".to_string(), json!(viewer.z));
            }
        }
        ApplyLutDialogState {
            open: true,
            stack_prompt: false,
            window_label,
            command_id,
            params,
            slice_params,
            stack_slices,
        }
    }

    fn resize_current_slice_new_window_if_requested(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Option<command_registry::CommandExecuteResult> {
        if params
            .get("process_stack")
            .and_then(Value::as_bool)
            .unwrap_or(true)
        {
            return None;
        }
        if !params
            .get("create_new_window")
            .and_then(Value::as_bool)
            .unwrap_or(true)
        {
            return Some(
                match self.resize_current_slice_in_place(window_label, params) {
                    Ok(payload) => command_registry::CommandExecuteResult::with_payload(
                        "resize current slice applied",
                        payload,
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                },
            );
        }

        Some(
            match self.resize_current_slice_new_window(window_label, params) {
                Ok(payload) => command_registry::CommandExecuteResult::with_payload(
                    "resize current slice created",
                    payload,
                ),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
        )
    }

    fn resize_current_slice_new_window(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<Value, String> {
        let target = self
            .current_viewer_label(window_label)
            .unwrap_or(window_label)
            .to_string();
        let viewer = self
            .viewers_ui
            .get(&target)
            .ok_or_else(|| format!("no viewer UI state for `{target}`"))?;
        let (z, t) = (viewer.z, viewer.t);
        let dataset = self
            .state
            .label_to_session
            .get(&target)
            .ok_or_else(|| format!("no viewer session for `{target}`"))?
            .committed_source
            .to_dataset()?;
        let plane = current_zt_plane_dataset(dataset.as_ref(), z, t)?;
        let mut resize_params = params.clone();
        if let Some(map) = resize_params.as_object_mut() {
            map.remove("depth");
            map.remove("frames");
            map.remove("process_stack");
            map.remove("create_new_window");
            map.remove("title");
        }
        let output = self
            .state
            .app
            .ops_service()
            .execute("image.resize", &plane, &resize_params)
            .map_err(|error| error.to_string())?;
        let title = params
            .get("title")
            .and_then(Value::as_str)
            .filter(|title| !title.trim().is_empty())
            .map(sanitize_image_title)
            .unwrap_or_else(|| format!("Untitled-{}", self.state.next_window_id + 1));
        let path = normalize_path(&PathBuf::from(format!(
            "{}-{}.tif",
            title,
            self.state.next_window_id + 1
        )));
        let label = self.create_viewer(path, ViewerImageSource::Dataset(Arc::new(output.dataset)));
        Ok(json!({ "window": label, "op": "image.resize", "slice": z }))
    }

    fn resize_current_slice_in_place(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<Value, String> {
        let target = self
            .current_viewer_label(window_label)
            .unwrap_or(window_label)
            .to_string();
        let viewer = self
            .viewers_ui
            .get(&target)
            .ok_or_else(|| format!("no viewer UI state for `{target}`"))?;
        let (z, t) = (viewer.z, viewer.t);
        let dataset = self
            .state
            .label_to_session
            .get(&target)
            .ok_or_else(|| format!("no viewer session for `{target}`"))?
            .committed_source
            .to_dataset()?;
        let scaled = scale_current_zt_slice_in_place(dataset.as_ref(), z, t, params)?;
        let session = self
            .state
            .label_to_session
            .get_mut(&target)
            .ok_or_else(|| format!("no viewer session for `{target}`"))?;
        session.commit_dataset(Arc::new(scaled));
        if let Some(viewer) = self.viewers_ui.get_mut(&target) {
            viewer.last_generation = 0;
            viewer.last_request = None;
            viewer.status_message = format!("Current slice {} scaled", z + 1);
        }
        Ok(json!({ "window": target, "op": "image.resize", "slice": z }))
    }

    fn handle_color_threshold_command(
        &mut self,
        window_label: &str,
        params: Value,
    ) -> command_registry::CommandExecuteResult {
        let action = params
            .get("action")
            .and_then(Value::as_str)
            .unwrap_or("threshold");

        match action {
            "original" => {
                let Some(target) = self.current_viewer_label(window_label).map(str::to_string)
                else {
                    return command_registry::CommandExecuteResult::blocked(
                        "a loaded image is required for Color Threshold",
                    );
                };
                let Some(session) = self.state.label_to_session.get_mut(&target) else {
                    return command_registry::CommandExecuteResult::blocked(format!(
                        "no viewer session for `{target}`"
                    ));
                };
                session.set_active_preview(None);
                if let Some(viewer) = self.viewers_ui.get_mut(&target) {
                    viewer.last_generation = 0;
                    viewer.last_request = None;
                    viewer.status_message = "Color threshold original".to_string();
                }
                command_registry::CommandExecuteResult::ok("color threshold original restored")
            }
            "filtered" => {
                let mut params = params;
                if let Some(map) = params.as_object_mut() {
                    map.insert("output".to_string(), json!("filtered"));
                }
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.color_threshold".to_string(),
                        params,
                        mode: OpRunMode::Preview,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "color threshold filtered preview started",
                        json!({ "job_id": ticket.job_id, "op": "image.color_threshold" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "select" => match self.create_color_threshold_selection(window_label, &params) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "sample" => match self.sample_color_threshold_selection(window_label, &params) {
                Ok(_) => {
                    let mut preview_params = color_threshold_params(&self.adjust_dialog);
                    if let Some(map) = preview_params.as_object_mut() {
                        map.insert("output".to_string(), json!("filtered"));
                    }
                    match self.viewer_start_op(
                        window_label,
                        ViewerOpRequest {
                            op: "image.color_threshold".to_string(),
                            params: preview_params,
                            mode: OpRunMode::Preview,
                        },
                    ) {
                        Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                            "color threshold ranges sampled and preview started",
                            json!({ "job_id": ticket.job_id, "op": "image.color_threshold" }),
                        ),
                        Err(error) => command_registry::CommandExecuteResult::blocked(error),
                    }
                }
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "auto" => match self.auto_color_threshold_ranges(window_label, &params) {
                Ok(_) => {
                    let mut preview_params = color_threshold_params(&self.adjust_dialog);
                    if let Some(map) = preview_params.as_object_mut() {
                        map.insert("output".to_string(), json!("filtered"));
                    }
                    match self.viewer_start_op(
                        window_label,
                        ViewerOpRequest {
                            op: "image.color_threshold".to_string(),
                            params: preview_params,
                            mode: OpRunMode::Preview,
                        },
                    ) {
                        Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                            "color threshold auto ranges preview started",
                            json!({ "job_id": ticket.job_id, "op": "image.color_threshold" }),
                        ),
                        Err(error) => command_registry::CommandExecuteResult::blocked(error),
                    }
                }
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "stack" => {
                let mut params = params;
                if let Some(map) = params.as_object_mut() {
                    map.insert("output".to_string(), json!("filtered"));
                }
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.color_threshold".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "color threshold stack started",
                        json!({ "job_id": ticket.job_id, "op": "image.color_threshold" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "macro" => match self.record_color_threshold_macro(&params) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "help" => match self.show_color_threshold_help(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            _ => match self.viewer_start_op(
                window_label,
                ViewerOpRequest {
                    op: "image.color_threshold".to_string(),
                    params,
                    mode: OpRunMode::Apply,
                },
            ) {
                Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                    "color threshold started",
                    json!({ "job_id": ticket.job_id, "op": "image.color_threshold" }),
                ),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
        }
    }

    fn show_color_threshold_help(&mut self, window_label: &str) -> Result<String, String> {
        let target = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for Color Threshold Help".to_string())?
            .to_string();
        let message = "Color Thresholder: choose HSB/RGB/CIE Lab/YUV component ranges; Pass keeps pixels inside a band, otherwise outside; Original restores the source, Filtered previews, Stack applies, Select traces, Sample reads the selection.";
        let viewer = self
            .viewers_ui
            .get_mut(&target)
            .ok_or_else(|| format!("no viewer UI state for `{target}`"))?;
        viewer.status_message = "Color Threshold help shown".to_string();
        viewer.tool_message = Some(message.to_string());
        Ok("color threshold help shown".to_string())
    }

    fn create_color_threshold_selection(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<String, String> {
        let target = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for Color Threshold Select".to_string())?
            .to_string();
        let (request, dataset) = {
            let viewer = self
                .viewers_ui
                .get(&target)
                .ok_or_else(|| format!("no viewer UI state for `{target}`"))?;
            let session = self
                .state
                .label_to_session
                .get(&target)
                .ok_or_else(|| format!("no viewer session for `{target}`"))?;
            (
                ViewerFrameRequest {
                    z: viewer.z,
                    t: viewer.t,
                    channel: 0,
                },
                session.committed_source.to_dataset()?,
            )
        };
        let mask = self
            .state
            .app
            .ops_service()
            .execute("image.color_threshold", dataset.as_ref(), params)
            .map_err(|error| error.to_string())?
            .dataset;
        let slice = extract_slice(&mask, request.z, request.t, 0)?;
        let roi = mask_foreground_roi(&slice)
            .ok_or_else(|| "Color Threshold Select found no thresholded pixels".to_string())?;
        let viewer = self
            .viewers_ui
            .get_mut(&target)
            .ok_or_else(|| format!("no viewer UI state for `{target}`"))?;
        let position = interaction::roi::RoiPosition {
            channel: viewer.channel,
            z: viewer.z,
            t: viewer.t,
        };
        viewer.rois.begin_active(roi, position);
        viewer.rois.commit_active(false);
        viewer.status_message = "Color threshold selection created".to_string();
        Ok("color threshold selection created".to_string())
    }

    fn sample_color_threshold_selection(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<String, String> {
        let target = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for Color Threshold Sample".to_string())?
            .to_string();
        let color_space = params
            .get("color_space")
            .or_else(|| params.get("space"))
            .and_then(Value::as_str)
            .unwrap_or("HSB");
        let color_space = color_threshold_space_key(color_space);
        let (bbox, z, t, dataset) = {
            let viewer = self
                .viewers_ui
                .get(&target)
                .ok_or_else(|| format!("no viewer UI state for `{target}`"))?;
            let roi = viewer
                .rois
                .active_roi
                .as_ref()
                .or_else(|| {
                    viewer
                        .rois
                        .selected_roi_id
                        .and_then(|id| viewer.rois.overlay_rois.iter().find(|roi| roi.id == id))
                })
                .ok_or_else(|| "Color Threshold Sample requires a selection".to_string())?;
            let bbox = roi_kind_bbox(&roi.kind)
                .ok_or_else(|| "Color Threshold Sample requires a bounded selection".to_string())?;
            let session = self
                .state
                .label_to_session
                .get(&target)
                .ok_or_else(|| format!("no viewer session for `{target}`"))?;
            (
                bbox,
                viewer.z,
                viewer.t,
                session.committed_source.to_dataset()?,
            )
        };
        let sample =
            sample_color_threshold_ranges_with_passes(dataset.as_ref(), bbox, z, t, &color_space)?;
        let ranges = sample.ranges;
        self.adjust_dialog.hue_min = ranges[0].0;
        self.adjust_dialog.hue_max = ranges[0].1;
        self.adjust_dialog.saturation_min = ranges[1].0;
        self.adjust_dialog.saturation_max = ranges[1].1;
        self.adjust_dialog.brightness_min = ranges[2].0;
        self.adjust_dialog.brightness_max = ranges[2].1;
        self.adjust_dialog.hue_pass = sample.passes[0];
        self.adjust_dialog.saturation_pass = sample.passes[1];
        self.adjust_dialog.brightness_pass = sample.passes[2];
        if let Some(viewer) = self.viewers_ui.get_mut(&target) {
            viewer.status_message = "Color threshold ranges sampled".to_string();
        }
        Ok("color threshold ranges sampled from selection".to_string())
    }

    fn auto_color_threshold_ranges(
        &mut self,
        window_label: &str,
        params: &Value,
    ) -> Result<String, String> {
        let target = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for Color Threshold auto".to_string())?
            .to_string();
        let color_space = params
            .get("color_space")
            .or_else(|| params.get("space"))
            .and_then(Value::as_str)
            .unwrap_or("HSB");
        let color_space = color_threshold_space_key(color_space);
        let method = params
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or("Default");
        let background = if params
            .get("dark_background")
            .and_then(Value::as_bool)
            .unwrap_or(true)
        {
            "dark"
        } else {
            "light"
        };
        let (z, t, dataset) = {
            let viewer = self
                .viewers_ui
                .get(&target)
                .ok_or_else(|| format!("no viewer UI state for `{target}`"))?;
            let session = self
                .state
                .label_to_session
                .get(&target)
                .ok_or_else(|| format!("no viewer session for `{target}`"))?;
            (viewer.z, viewer.t, session.committed_source.to_dataset()?)
        };
        let ranges = color_threshold_auto_ranges(
            dataset.as_ref(),
            z,
            t,
            &color_space,
            method,
            background,
            &self.state.app,
        )?;
        match color_space.as_str() {
            "rgb" => {
                self.adjust_dialog.hue_min = ranges[0].0;
                self.adjust_dialog.hue_max = ranges[0].1;
                self.adjust_dialog.saturation_min = ranges[1].0;
                self.adjust_dialog.saturation_max = ranges[1].1;
                self.adjust_dialog.brightness_min = ranges[2].0;
                self.adjust_dialog.brightness_max = ranges[2].1;
            }
            "lab" | "yuv" => {
                self.adjust_dialog.hue_min = ranges[0].0;
                self.adjust_dialog.hue_max = ranges[0].1;
            }
            _ => {
                self.adjust_dialog.brightness_min = ranges[2].0;
                self.adjust_dialog.brightness_max = ranges[2].1;
            }
        }
        self.adjust_dialog.hue_pass = true;
        self.adjust_dialog.saturation_pass = true;
        self.adjust_dialog.brightness_pass = true;
        if let Some(viewer) = self.viewers_ui.get_mut(&target) {
            viewer.status_message = "Color threshold auto ranges set".to_string();
        }
        Ok("color threshold auto ranges set".to_string())
    }

    fn record_color_threshold_macro(&mut self, params: &Value) -> Result<String, String> {
        if !self.macro_recorder.open || !self.macro_recorder.recording {
            return Err(
                "Color Threshold Macro requires the macro recorder to be running".to_string(),
            );
        }
        let text = color_threshold_macro_text(params);
        if !self.macro_recorder.text.is_empty() && !self.macro_recorder.text.ends_with('\n') {
            self.macro_recorder.text.push('\n');
        }
        self.macro_recorder.text.push_str(&text);
        if !self.macro_recorder.text.ends_with('\n') {
            self.macro_recorder.text.push('\n');
        }
        Ok("color threshold macro recorded".to_string())
    }

    fn remember_repeatable_command(
        &mut self,
        command_id: &str,
        params: Option<&Value>,
        result: &command_registry::CommandExecuteResult,
    ) {
        if command_id == "process.repeat_command"
            || !matches!(result.status, command_registry::CommandExecuteStatus::Ok)
        {
            return;
        }

        let metadata = command_registry::metadata(command_id);
        if !metadata.implemented || metadata.frontend_only {
            return;
        }

        self.last_repeatable_command = Some(StoredCommand {
            command_id: command_id.to_string(),
            params: params.cloned(),
        });
    }

    fn record_macro_command(
        &mut self,
        command_id: &str,
        params: Option<&Value>,
        result: &command_registry::CommandExecuteResult,
    ) {
        if !self.macro_recorder.open
            || !self.macro_recorder.recording
            || command_id.starts_with("plugins.macros.")
            || !matches!(result.status, command_registry::CommandExecuteStatus::Ok)
        {
            return;
        }

        let Some(line) = macro_record_line_for_command(command_id, params, &self.command_catalog)
        else {
            return;
        };
        if !self.macro_recorder.text.is_empty() && !self.macro_recorder.text.ends_with('\n') {
            self.macro_recorder.text.push('\n');
        }
        self.macro_recorder.text.push_str(&line);
        self.macro_recorder.text.push('\n');
    }

    fn repeat_last_command(
        &mut self,
        window_label: &str,
    ) -> command_registry::CommandExecuteResult {
        let Some(command) = self.last_repeatable_command.clone() else {
            return command_registry::CommandExecuteResult::blocked("no command to repeat");
        };

        let metadata = command_registry::metadata(&command.command_id);
        let target_label = if metadata.requires_image && window_label == LAUNCHER_LABEL {
            match self.active_viewer_label.clone() {
                Some(label) => label,
                None => {
                    return command_registry::CommandExecuteResult::blocked(
                        "a loaded image is required to repeat this command",
                    );
                }
            }
        } else {
            window_label.to_string()
        };

        self.dispatch_command(&target_label, &command.command_id, command.params)
    }

    fn handle_local_command(
        &mut self,
        window_label: &str,
        command_id: &str,
        params: Option<&Value>,
    ) -> Option<command_registry::CommandExecuteResult> {
        if command_id == "file.open" {
            if let Some(path) = params
                .and_then(|params| params.get("path"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|path| !path.is_empty())
            {
                let result = self.open_paths(vec![PathBuf::from(path)]);
                self.apply_open_result(&result);
                if result.errors.is_empty() {
                    return Some(command_registry::CommandExecuteResult::ok(format!(
                        "opened {path}"
                    )));
                }
                return Some(command_registry::CommandExecuteResult::blocked(
                    result.errors.join("; "),
                ));
            }

            let extensions = supported_formats();
            let picked_paths = FileDialog::new()
                .add_filter("Supported images", extensions)
                .set_title("Open Image")
                .pick_files()
                .unwrap_or_default();

            if picked_paths.is_empty() {
                self.set_fallback_status("Open canceled");
                return Some(command_registry::CommandExecuteResult::ok("open canceled"));
            }

            let result = self.open_paths(picked_paths);
            self.apply_open_result(&result);
            return Some(command_registry::CommandExecuteResult::ok(
                "opened files from native picker",
            ));
        }

        if command_id == "file.new" || command_id == "image.hyperstacks.new" {
            if let Some(params) = params {
                return Some(match self.create_new_image(params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                });
            }
            if command_id == "image.hyperstacks.new" {
                self.new_image_dialog.width = 400;
                self.new_image_dialog.height = 300;
                self.new_image_dialog.slices = 4;
                self.new_image_dialog.channels = 3;
                self.new_image_dialog.frames = 5;
            }
            self.new_image_dialog.open = true;
            return Some(command_registry::CommandExecuteResult::ok(
                "new image dialog opened",
            ));
        }

        if command_id == "file.import.image" {
            return Some(match self.import_image_sequence() {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            });
        }

        if command_id == "file.import.raw" {
            self.raw_import_dialog.open = true;
            return Some(command_registry::CommandExecuteResult::ok(
                "raw import dialog opened",
            ));
        }

        if command_id == "file.import.url" {
            if let Some(params) = params {
                return Some(match self.import_from_url(params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                });
            }
            self.url_import_dialog.open = true;
            return Some(command_registry::CommandExecuteResult::ok(
                "URL import dialog opened",
            ));
        }

        if let Some(path) = command_id.strip_prefix("file.open_recent:") {
            return Some(match self.open_recent_path(Path::new(path)) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            });
        }

        if command_id == "file.export.image" {
            return Some(self.dispatch_command(window_label, "file.save_as", None));
        }

        if command_id == "file.export.results" {
            return Some(match self.export_results() {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            });
        }

        if command_id == "file.save_as" {
            if window_label == LAUNCHER_LABEL {
                return Some(command_registry::CommandExecuteResult::blocked(
                    "a loaded image is required for save as",
                ));
            }

            if let Some(path) = params
                .and_then(|params| params.get("path"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|path| !path.is_empty())
            {
                let format = params
                    .and_then(|params| params.get("format"))
                    .and_then(Value::as_str);
                let target_path = macro_save_path(path, format);
                return Some(match self.save_viewer(window_label, Some(target_path)) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                });
            }

            let current_path = self.state.label_to_path.get(window_label).cloned();
            let mut dialog = FileDialog::new()
                .add_filter("Supported images", supported_formats())
                .set_title("Save Image As");
            if let Some(path) = &current_path {
                if let Some(parent) = path.parent() {
                    dialog = dialog.set_directory(parent);
                }
                if let Some(file_name) = path.file_name().and_then(|name| name.to_str()) {
                    dialog = dialog.set_file_name(file_name);
                }
            }

            let Some(target_path) = dialog.save_file() else {
                self.set_fallback_status("Save As canceled");
                return Some(command_registry::CommandExecuteResult::ok(
                    "save as canceled",
                ));
            };

            return Some(match self.save_viewer(window_label, Some(target_path)) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            });
        }

        if command_id == "edit.options.appearance" {
            self.desktop_state.utility_windows.help_shortcuts_open = true;
            self.persist_desktop_state();
            return Some(command_registry::CommandExecuteResult::ok(
                "appearance/help window opened",
            ));
        }

        if command_id == "edit.options.memory" {
            return Some(command_registry::CommandExecuteResult::ok(
                "memory dialog is informational in image-rs",
            ));
        }

        if command_id == "edit.options.line_width" || command_id == "image.adjust.line_width" {
            if params.is_none() {
                return Some(self.open_adjust_dialog(window_label, AdjustDialogKind::LineWidth));
            }
            let params = command_registry::merge_params(command_id, params.cloned());
            return Some(match line_width_from_params(&params) {
                Ok(width) => {
                    self.tool_options.line_width_px = width;
                    if let Some(spline_fit) = params.get("spline_fit").and_then(Value::as_bool)
                        && let Some(viewer_label) =
                            self.current_viewer_label(window_label).map(str::to_string)
                        && let Some(viewer) = self.viewers_ui.get_mut(&viewer_label)
                    {
                        set_selected_roi_spline_fit(viewer, spline_fit);
                    }
                    command_registry::CommandExecuteResult::with_payload(
                        "line width updated",
                        json!({ "width": width }),
                    )
                }
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            });
        }

        if command_id == "__dialog.resize" || command_id == "__dialog.canvas_resize" {
            let Some(viewer_label) = self.current_viewer_label(window_label).map(str::to_string)
            else {
                return Some(command_registry::CommandExecuteResult::blocked(
                    "a loaded image is required",
                ));
            };
            let params = params.cloned().unwrap_or_else(|| json!({}));
            let op = if command_id == "__dialog.resize" {
                "image.resize"
            } else {
                "image.canvas_resize"
            };
            let mode = if command_id == "__dialog.resize" {
                resize_op_mode_from_params(&params)
            } else {
                OpRunMode::Apply
            };
            if command_id == "__dialog.resize"
                && let Some(result) =
                    self.resize_current_slice_new_window_if_requested(&viewer_label, &params)
            {
                return Some(result);
            }
            return Some(
                match self.viewer_start_op(
                    &viewer_label,
                    ViewerOpRequest {
                        op: op.to_string(),
                        params,
                        mode,
                    },
                ) {
                    Ok(_) => command_registry::CommandExecuteResult::ok("image transform started"),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                },
            );
        }

        if command_id == "__roi.measure_all" {
            return Some(match self.measure_all_rois(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            });
        }

        if let Some(tool) = tool_from_command_id(command_id) {
            self.tool_state.selected = tool;
            let message = if tool.has_behavior() {
                format!("{} tool selected", tool.label())
            } else {
                format!(
                    "{} tool selected (reserved for future extension)",
                    tool.label()
                )
            };
            self.set_fallback_status(message.clone());
            return Some(command_registry::CommandExecuteResult::ok(message));
        }

        match command_id {
            "launcher.tool.rect.mode.rectangle" => {
                self.tool_options.rect_mode = RectMode::Rectangle;
                return Some(command_registry::CommandExecuteResult::ok(
                    "rectangle mode selected",
                ));
            }
            "launcher.tool.rect.mode.rounded" => {
                self.tool_options.rect_mode = RectMode::Rounded;
                return Some(command_registry::CommandExecuteResult::ok(
                    "rounded rectangle mode selected",
                ));
            }
            "launcher.tool.rect.mode.rotated" => {
                self.tool_options.rect_mode = RectMode::Rotated;
                return Some(command_registry::CommandExecuteResult::ok(
                    "rotated rectangle mode selected",
                ));
            }
            "launcher.tool.oval.mode.oval" => {
                self.tool_options.oval_mode = OvalMode::Oval;
                return Some(command_registry::CommandExecuteResult::ok(
                    "oval mode selected",
                ));
            }
            "launcher.tool.oval.mode.ellipse" => {
                self.tool_options.oval_mode = OvalMode::Ellipse;
                return Some(command_registry::CommandExecuteResult::ok(
                    "ellipse mode selected",
                ));
            }
            "launcher.tool.oval.mode.brush" => {
                self.tool_options.oval_mode = OvalMode::Brush;
                return Some(command_registry::CommandExecuteResult::ok(
                    "brush mode selected",
                ));
            }
            "launcher.tool.line.mode.straight" => {
                self.tool_options.line_mode = LineMode::Straight;
                return Some(command_registry::CommandExecuteResult::ok(
                    "straight line mode selected",
                ));
            }
            "launcher.tool.line.mode.segmented" => {
                self.tool_options.line_mode = LineMode::Segmented;
                return Some(command_registry::CommandExecuteResult::ok(
                    "segmented line mode selected",
                ));
            }
            "launcher.tool.line.mode.freehand" => {
                self.tool_options.line_mode = LineMode::Freehand;
                return Some(command_registry::CommandExecuteResult::ok(
                    "freehand line mode selected",
                ));
            }
            "launcher.tool.line.mode.arrow" => {
                self.tool_options.line_mode = LineMode::Arrow;
                return Some(command_registry::CommandExecuteResult::ok(
                    "arrow mode selected",
                ));
            }
            "launcher.tool.point.mode.point" => {
                self.tool_options.point_mode = PointMode::Point;
                return Some(command_registry::CommandExecuteResult::ok(
                    "point mode selected",
                ));
            }
            "launcher.tool.point.mode.multipoint" => {
                self.tool_options.point_mode = PointMode::MultiPoint;
                return Some(command_registry::CommandExecuteResult::ok(
                    "multipoint mode selected",
                ));
            }
            "tool.dropper.palette.white_black" => {
                self.tool_options.foreground_color = egui::Color32::WHITE;
                self.tool_options.background_color = egui::Color32::BLACK;
                return Some(command_registry::CommandExecuteResult::ok(
                    "palette set to white/black",
                ));
            }
            "tool.dropper.palette.black_white" => {
                self.tool_options.foreground_color = egui::Color32::BLACK;
                self.tool_options.background_color = egui::Color32::WHITE;
                return Some(command_registry::CommandExecuteResult::ok(
                    "palette set to black/white",
                ));
            }
            "tool.dropper.palette.red" => {
                self.tool_options.foreground_color = egui::Color32::RED;
                return Some(command_registry::CommandExecuteResult::ok(
                    "foreground set to red",
                ));
            }
            "tool.dropper.palette.green" => {
                self.tool_options.foreground_color = egui::Color32::GREEN;
                return Some(command_registry::CommandExecuteResult::ok(
                    "foreground set to green",
                ));
            }
            "tool.dropper.palette.blue" => {
                self.tool_options.foreground_color = egui::Color32::BLUE;
                return Some(command_registry::CommandExecuteResult::ok(
                    "foreground set to blue",
                ));
            }
            "tool.dropper.palette.yellow" => {
                self.tool_options.foreground_color = egui::Color32::YELLOW;
                return Some(command_registry::CommandExecuteResult::ok(
                    "foreground set to yellow",
                ));
            }
            "tool.dropper.palette.cyan" => {
                self.tool_options.foreground_color = egui::Color32::from_rgb(0, 255, 255);
                return Some(command_registry::CommandExecuteResult::ok(
                    "foreground set to cyan",
                ));
            }
            "tool.dropper.palette.magenta" => {
                self.tool_options.foreground_color = egui::Color32::from_rgb(255, 0, 255);
                return Some(command_registry::CommandExecuteResult::ok(
                    "foreground set to magenta",
                ));
            }
            "tool.dropper.palette.foreground" => {
                self.open_color_dialog(ColorDialogMode::Foreground);
                return Some(command_registry::CommandExecuteResult::ok(
                    "foreground color dialog opened",
                ));
            }
            "tool.dropper.palette.background" => {
                self.open_color_dialog(ColorDialogMode::Background);
                return Some(command_registry::CommandExecuteResult::ok(
                    "background color dialog opened",
                ));
            }
            "tool.dropper.palette.colors" => {
                self.open_color_dialog(ColorDialogMode::Colors);
                return Some(command_registry::CommandExecuteResult::ok(
                    "color dialog opened",
                ));
            }
            "tool.dropper.palette.color_picker" => {
                self.open_color_dialog(ColorDialogMode::Picker);
                return Some(command_registry::CommandExecuteResult::ok(
                    "color picker opened",
                ));
            }
            "image.color.color_picker" => {
                self.open_color_dialog(ColorDialogMode::Picker);
                return Some(command_registry::CommandExecuteResult::ok(
                    "color picker opened",
                ));
            }
            _ => {}
        }

        let viewer = self.viewers_ui.get_mut(window_label)?;

        match command_id {
            "viewer.slice.next" | "image.stacks.next" => {
                if let Some(session) = self.state.label_to_session.get(window_label) {
                    let max_z = session.committed_summary.z_slices.saturating_sub(1);
                    viewer.z = (viewer.z + 1).min(max_z);
                    viewer.last_request = None;
                }
                Some(command_registry::CommandExecuteResult::ok(
                    "moved to next slice",
                ))
            }
            "viewer.slice.previous" | "image.stacks.previous" => {
                viewer.z = viewer.z.saturating_sub(1);
                viewer.last_request = None;
                Some(command_registry::CommandExecuteResult::ok(
                    "moved to previous slice",
                ))
            }
            "image.stacks.set" => {
                let params = command_registry::merge_params(command_id, params.cloned());
                let Some(session) = self.state.label_to_session.get(window_label) else {
                    return Some(command_registry::CommandExecuteResult::blocked(
                        "a loaded image is required for set slice",
                    ));
                };
                if !has_stack_position_param(&params) {
                    self.stack_position_dialog.open = true;
                    self.stack_position_dialog.window_label = window_label.to_string();
                    self.stack_position_dialog.channel = viewer.channel + 1;
                    self.stack_position_dialog.slice = viewer.z + 1;
                    self.stack_position_dialog.frame = viewer.t + 1;
                    return Some(command_registry::CommandExecuteResult::ok(
                        "set slice dialog opened",
                    ));
                }
                match stack_position_from_params(
                    &params,
                    viewer.channel,
                    viewer.z,
                    viewer.t,
                    session.committed_summary.channels,
                    session.committed_summary.z_slices,
                    session.committed_summary.times,
                ) {
                    Ok((channel, z, t)) => {
                        viewer.channel = channel;
                        viewer.z = z;
                        viewer.t = t;
                        viewer.last_request = None;
                        Some(command_registry::CommandExecuteResult::ok(format!(
                            "set position to C{} Z{} T{}",
                            channel + 1,
                            z + 1,
                            t + 1
                        )))
                    }
                    Err(error) => Some(command_registry::CommandExecuteResult::blocked(error)),
                }
            }
            "viewer.roi.delete" => {
                if let Some(selected) = viewer.rois.selected_roi_id {
                    viewer.rois.overlay_rois.retain(|roi| roi.id != selected);
                    viewer.rois.selected_roi_id = None;
                } else {
                    viewer.rois.clear_all();
                }
                Some(command_registry::CommandExecuteResult::ok(
                    "deleted selection",
                ))
            }
            "image.zoom.in" => {
                viewer.pending_zoom = Some(ZoomCommand::In);
                Some(command_registry::CommandExecuteResult::ok("zoomed in"))
            }
            "image.zoom.out" => {
                viewer.pending_zoom = Some(ZoomCommand::Out);
                Some(command_registry::CommandExecuteResult::ok("zoomed out"))
            }
            "image.zoom.reset" | "image.zoom.original" => {
                viewer.pending_zoom = Some(ZoomCommand::Original);
                Some(command_registry::CommandExecuteResult::ok("zoom reset"))
            }
            "image.zoom.view100" => {
                viewer.pending_zoom = Some(ZoomCommand::View100);
                Some(command_registry::CommandExecuteResult::ok("zoomed to 100%"))
            }
            "image.zoom.to_selection" => {
                viewer.pending_zoom = Some(ZoomCommand::ToSelection);
                Some(command_registry::CommandExecuteResult::ok(
                    "zoomed to selection",
                ))
            }
            "image.zoom.scale_to_fit" => {
                viewer.pending_zoom = Some(ZoomCommand::ScaleToFit);
                Some(command_registry::CommandExecuteResult::ok(
                    "scale-to-fit enabled",
                ))
            }
            "image.zoom.set" => {
                let params = command_registry::merge_params(command_id, params.cloned());
                if !has_zoom_set_param(&params) {
                    let center = egui::pos2(
                        viewer.transform.src_rect.x + viewer.transform.src_rect.width * 0.5,
                        viewer.transform.src_rect.y + viewer.transform.src_rect.height * 0.5,
                    );
                    self.zoom_set_dialog.open = true;
                    self.zoom_set_dialog.window_label = window_label.to_string();
                    self.zoom_set_dialog.zoom_percent = viewer.transform.magnification * 100.0;
                    self.zoom_set_dialog.x_center = center.x;
                    self.zoom_set_dialog.y_center = center.y;
                    return Some(command_registry::CommandExecuteResult::ok(
                        "set zoom dialog opened",
                    ));
                }
                match zoom_set_params(&params) {
                    Ok((magnification, x_center, y_center)) => {
                        viewer.pending_zoom = Some(ZoomCommand::Set {
                            magnification,
                            x_center,
                            y_center,
                        });
                        Some(command_registry::CommandExecuteResult::ok(format!(
                            "zoom set to {:.0}%",
                            magnification * 100.0
                        )))
                    }
                    Err(error) => Some(command_registry::CommandExecuteResult::blocked(error)),
                }
            }
            "process.fft.make_circular_selection" => {
                let Some(session) = self.state.label_to_session.get(window_label) else {
                    return Some(command_registry::CommandExecuteResult::blocked(
                        "a loaded image is required for circular selection",
                    ));
                };
                let radius = params
                    .and_then(|params| params.get("radius"))
                    .and_then(Value::as_f64)
                    .map(|value| value as f32);
                let roi = match centered_circular_roi(&session.committed_summary.shape, radius) {
                    Ok(roi) => roi,
                    Err(error) => {
                        return Some(command_registry::CommandExecuteResult::blocked(error));
                    }
                };
                let position = interaction::roi::RoiPosition {
                    channel: viewer.channel,
                    z: viewer.z,
                    t: viewer.t,
                };
                viewer.rois.begin_active(roi, position);
                viewer.rois.commit_active(false);
                Some(command_registry::CommandExecuteResult::ok(
                    "circular selection created",
                ))
            }
            "edit.selection.all" => {
                let Some(session) = self.state.label_to_session.get(window_label) else {
                    return Some(command_registry::CommandExecuteResult::blocked(
                        "a loaded image is required for select all",
                    ));
                };
                let roi = match full_image_rect_roi(&session.committed_summary.shape) {
                    Ok(roi) => roi,
                    Err(error) => {
                        return Some(command_registry::CommandExecuteResult::blocked(error));
                    }
                };
                let position = interaction::roi::RoiPosition {
                    channel: viewer.channel,
                    z: viewer.z,
                    t: viewer.t,
                };
                viewer.rois.begin_active(roi, position);
                viewer.rois.commit_active(false);
                Some(command_registry::CommandExecuteResult::ok("selected all"))
            }
            "image.zoom.maximize" => {
                viewer.pending_zoom = Some(ZoomCommand::Maximize);
                Some(command_registry::CommandExecuteResult::ok("zoom maximized"))
            }
            "image.lookup.invert_lut"
            | "image.color.invert_luts"
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
            | "image.lookup.red_green" => {
                let lut = lookup_table_from_command(command_id).expect("lookup command");
                viewer.lookup_table = lut;
                viewer.texture = None;
                viewer.last_request = None;
                Some(command_registry::CommandExecuteResult::ok(format!(
                    "{} LUT applied",
                    lut.label()
                )))
            }
            "viewer.roi.clear" | "edit.selection.none" => {
                viewer.rois.clear_all();
                Some(command_registry::CommandExecuteResult::ok(
                    "selection cleared",
                ))
            }
            "viewer.roi.abort" => {
                viewer.rois.abort_active();
                viewer.active_drag_started = None;
                viewer.active_polygon_points.clear();
                Some(command_registry::CommandExecuteResult::ok(
                    "selection aborted",
                ))
            }
            "viewer.roi.select_next" => {
                viewer.rois.select_next();
                Some(command_registry::CommandExecuteResult::ok(
                    "selected next ROI",
                ))
            }
            _ => None,
        }
    }

    fn execute_command(
        &mut self,
        window_label: &str,
        request: command_registry::CommandExecuteRequest,
    ) -> command_registry::CommandExecuteResult {
        let metadata = command_registry::metadata(&request.command_id);
        if !metadata.scope.contains(window_label) {
            return command_registry::CommandExecuteResult::blocked(format!(
                "command `{}` is not available in this window",
                request.command_id
            ));
        }

        if metadata.requires_image && !self.state.label_to_session.contains_key(window_label) {
            return command_registry::CommandExecuteResult::blocked(
                "a loaded image is required for this command",
            );
        }

        match request.command_id.as_str() {
            "process.repeat_command" => self.repeat_last_command(window_label),
            "macro.set_option" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                command_registry::CommandExecuteResult::ok(
                    self.apply_macro_set_option(window_label, &params),
                )
            }
            "macro.call" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                command_registry::CommandExecuteResult::ok(self.apply_macro_call(&params))
            }
            "macro.builtin_call" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                command_registry::CommandExecuteResult::ok(self.apply_macro_builtin_call(&params))
            }
            "macro.select_window" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.select_macro_window(&params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "macro.select_image" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.select_macro_image(&params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "macro.close_window" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.close_macro_window(&params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "macro.set_color" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.apply_macro_set_color(&params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "macro.set_roi_name" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.apply_macro_set_roi_name(window_label, &params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "macro.set_metadata" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.apply_macro_set_metadata(window_label, &params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "macro.set_tool" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.apply_macro_set_tool(&params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "macro.set_paste_mode" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.apply_macro_set_paste_mode(&params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "macro.remove_overlay_rois" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.remove_overlay_rois_by_name(window_label, &params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "macro.remove_overlay_selection" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.remove_overlay_selection(window_label, &params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "macro.activate_overlay_selection" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.activate_overlay_selection(window_label, &params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "macro.make_rectangle" | "macro.make_oval" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.make_macro_roi(
                    window_label,
                    &params,
                    request.command_id == "macro.make_oval",
                ) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "macro.make_line" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.make_macro_line(window_label, &params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "macro.make_selection" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.make_macro_selection(window_label, &params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "file.close" => {
                if window_label == LAUNCHER_LABEL {
                    if let Some(label) = self.active_viewer_label.clone() {
                        self.remove_viewer_by_label(&label);
                        command_registry::CommandExecuteResult::ok("tab closed")
                    } else {
                        command_registry::CommandExecuteResult::blocked(
                            "the launcher window is always visible",
                        )
                    }
                } else {
                    self.remove_viewer_by_label(window_label);
                    command_registry::CommandExecuteResult::ok("tab closed")
                }
            }
            "file.quit" => {
                self.should_quit = true;
                command_registry::CommandExecuteResult::ok("application exiting")
            }
            "file.save" => match self.save_viewer(window_label, None) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "file.revert" => {
                let Some(session) = self.state.label_to_session.get_mut(window_label) else {
                    return command_registry::CommandExecuteResult::blocked(
                        "a loaded image is required for revert",
                    );
                };
                let changed = session.revert_to_base();
                if let Some(viewer) = self.viewers_ui.get_mut(window_label) {
                    viewer.last_request = None;
                    viewer.last_generation = 0;
                    viewer.status_message = if changed {
                        "reverted to last saved image".to_string()
                    } else {
                        "image already matches the last saved state".to_string()
                    };
                }
                self.refresh_launcher_status();
                command_registry::CommandExecuteResult::ok(if changed {
                    "reverted to last saved image"
                } else {
                    "image already matches the last saved state"
                })
            }
            "edit.undo" => {
                let Some(session) = self.state.label_to_session.get_mut(window_label) else {
                    return command_registry::CommandExecuteResult::blocked(
                        "a loaded image is required for undo",
                    );
                };
                if !session.undo() {
                    return command_registry::CommandExecuteResult::blocked("nothing to undo");
                }
                if let Some(viewer) = self.viewers_ui.get_mut(window_label) {
                    viewer.last_request = None;
                    viewer.last_generation = 0;
                    viewer.status_message = "undo complete".to_string();
                }
                self.refresh_launcher_status();
                command_registry::CommandExecuteResult::ok("undo complete")
            }
            "edit.redo" => {
                let Some(session) = self.state.label_to_session.get_mut(window_label) else {
                    return command_registry::CommandExecuteResult::blocked(
                        "a loaded image is required for redo",
                    );
                };
                if !session.redo() {
                    return command_registry::CommandExecuteResult::blocked("nothing to redo");
                }
                if let Some(viewer) = self.viewers_ui.get_mut(window_label) {
                    viewer.last_request = None;
                    viewer.last_generation = 0;
                    viewer.status_message = "redo complete".to_string();
                }
                self.refresh_launcher_status();
                command_registry::CommandExecuteResult::ok("redo complete")
            }
            "window.next" => {
                let target = self.cycle_window(window_label, 1);
                command_registry::CommandExecuteResult::with_payload(
                    "cycled to next window",
                    json!({ "target": target }),
                )
            }
            "window.previous" => {
                let target = self.cycle_window(window_label, -1);
                command_registry::CommandExecuteResult::with_payload(
                    "cycled to previous window",
                    json!({ "target": target }),
                )
            }
            "window.main" => {
                self.focus_launcher = true;
                command_registry::CommandExecuteResult::ok("main window focused")
            }
            "window.put_behind" => {
                let target = self.cycle_window(window_label, 1);
                command_registry::CommandExecuteResult::with_payload(
                    "window put behind",
                    json!({ "target": target }),
                )
            }
            "edit.invert" => match self.viewer_start_op(
                window_label,
                ViewerOpRequest {
                    op: "intensity.invert".to_string(),
                    params: json!({}),
                    mode: OpRunMode::Apply,
                },
            ) {
                Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                    "invert started",
                    json!({ "job_id": ticket.job_id, "op": "intensity.invert" }),
                ),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "edit.clear" => {
                let background = f32::from(self.tool_options.background_color.r()) / 255.0;
                match self.fill_selection_or_slice(window_label, background, "clear") {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "edit.fill" => {
                let foreground = f32::from(self.tool_options.foreground_color.r()) / 255.0;
                match self.fill_selection_or_slice(window_label, foreground, "fill") {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "edit.internal_clipboard" => match self.show_internal_clipboard() {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "edit.selection.interpolate" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.interpolate_selection(window_label, &params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "edit.selection.properties" => match self.selection_properties(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "image.type.8bit" | "image.type.16bit" | "image.type.32bit" | "image.type.rgb" => {
                let target = match request.command_id.as_str() {
                    "image.type.8bit" => "u8",
                    "image.type.16bit" => "u16",
                    "image.type.32bit" => "f32",
                    "image.type.rgb" => "rgb",
                    _ => unreachable!(),
                };
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.convert".to_string(),
                        params: json!({ "target": target }),
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "image conversion started",
                        json!({ "job_id": ticket.job_id, "target": target }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.type.make_composite" => {
                if let Some(viewer) = self.viewers_ui.get_mut(window_label) {
                    viewer.status_message = "composite display mode acknowledged".to_string();
                    viewer.tool_message = Some(viewer.status_message.clone());
                }
                command_registry::CommandExecuteResult::ok("composite display mode acknowledged")
            }
            "image.type.8bit_color"
            | "image.type.rgb_stack"
            | "image.type.hsb_stack"
            | "image.type.hsb_32bit"
            | "image.type.lab_stack" => {
                if let Some(viewer) = self.viewers_ui.get_mut(window_label) {
                    viewer.status_message = format!("{} acknowledged", request.command_id);
                    viewer.tool_message = Some(viewer.status_message.clone());
                }
                command_registry::CommandExecuteResult::ok(format!(
                    "{} acknowledged",
                    request.command_id
                ))
            }
            "image.color.stack_to_rgb" => {
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.convert".to_string(),
                        params: json!({ "target": "rgb" }),
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "stack to RGB started",
                        json!({ "job_id": ticket.job_id, "target": "rgb" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.color.split_channels"
            | "image.color.merge_channels"
            | "image.color.arrange_channels"
            | "image.color.channels_tool"
            | "image.color.show_lut"
            | "image.color.display_luts"
            | "image.color.edit_lut"
            | "image.overlay.add_image"
            | "image.stacks.orthogonal_views"
            | "image.stacks.project_3d"
            | "image.stacks.animation.start"
            | "image.stacks.animation.stop"
            | "image.stacks.animation.options"
            | "image.stacks.tools.magic_montage_tools" => {
                if let Some(viewer) = self.viewers_ui.get_mut(window_label) {
                    viewer.status_message =
                        format!("{} compatibility command acknowledged", request.command_id);
                    viewer.tool_message = Some(viewer.status_message.clone());
                }
                command_registry::CommandExecuteResult::ok(format!(
                    "{} compatibility command acknowledged",
                    request.command_id
                ))
            }
            "image.adjust.color_balance" => {
                if request.params.is_none() {
                    return self.open_adjust_dialog(window_label, AdjustDialogKind::ColorBalance);
                }
                let params = command_registry::merge_params(&request.command_id, request.params);
                if params
                    .get("apply")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    if let Err(error) = self.ensure_apply_lut_supported(window_label) {
                        return command_registry::CommandExecuteResult::blocked(error);
                    }
                    let low = params
                        .get("min")
                        .and_then(Value::as_f64)
                        .map(|value| value as f32)
                        .unwrap_or(0.0);
                    let high = params
                        .get("max")
                        .and_then(Value::as_f64)
                        .map(|value| value as f32)
                        .unwrap_or(1.0);
                    let mut op_params = json!({ "low": low, "high": high });
                    if let Some(channel) = params
                        .get("channel")
                        .and_then(Value::as_str)
                        .and_then(color_balance_channel_index)
                        && let Some(map) = op_params.as_object_mut()
                    {
                        map.insert("channel".to_string(), json!(channel));
                    }
                    return match self.viewer_start_op(
                        window_label,
                        ViewerOpRequest {
                            op: "intensity.window".to_string(),
                            params: op_params,
                            mode: OpRunMode::Apply,
                        },
                    ) {
                        Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                            "apply LUT started",
                            json!({ "job_id": ticket.job_id, "op": "intensity.window" }),
                        ),
                        Err(error) => command_registry::CommandExecuteResult::blocked(error),
                    };
                }
                if params.get("min").is_some() || params.get("max").is_some() {
                    return match self.apply_display_range(window_label, &params, "min", "max") {
                        Ok(message) => command_registry::CommandExecuteResult::ok(message),
                        Err(error) => command_registry::CommandExecuteResult::blocked(error),
                    };
                }
                command_registry::CommandExecuteResult::blocked(
                    "color balance requires min/max display range parameters",
                )
            }
            "image.adjust.color_threshold" => {
                if request.params.is_none() {
                    return self.open_adjust_dialog(window_label, AdjustDialogKind::ColorThreshold);
                }
                let params = command_registry::merge_params(&request.command_id, request.params);
                self.handle_color_threshold_command(window_label, params)
            }
            "image.stacks.add_slice" | "image.stacks.delete_slice" => {
                let viewer_z = self
                    .viewers_ui
                    .get(window_label)
                    .map(|viewer| viewer.z)
                    .unwrap_or(0);
                let mut params =
                    command_registry::merge_params(&request.command_id, request.params);
                if let Some(map) = params.as_object_mut() {
                    let needs_index = map.get("index").map_or(true, Value::is_null);
                    if needs_index {
                        let index = if request.command_id == "image.stacks.add_slice" {
                            viewer_z.saturating_add(1)
                        } else {
                            viewer_z
                        };
                        map.insert("index".to_string(), json!(index));
                    }
                }
                let op = if request.command_id == "image.stacks.add_slice" {
                    "image.stack.add_slice"
                } else {
                    "image.stack.delete_slice"
                };
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: op.to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "stack edit started",
                        json!({ "job_id": ticket.job_id, "op": op }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.stacks.images_to_stack" => match self.images_to_stack() {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "image.stacks.plot_z_profile" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.stack.z_profile".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "Z-axis profile started",
                        json!({ "job_id": ticket.job_id, "op": "image.stack.z_profile" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.stacks.measure_stack" => match self.measure_stack(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "image.stacks.plot_xy_profile" => {
                let viewer_label = match self.current_viewer_label(window_label) {
                    Some(label) => label.to_string(),
                    None => {
                        return command_registry::CommandExecuteResult::blocked(
                            "a loaded image is required for Plot XY Profile",
                        );
                    }
                };
                let params = match self
                    .viewers_ui
                    .get(&viewer_label)
                    .ok_or_else(|| format!("no viewer UI state for `{viewer_label}`"))
                    .and_then(|viewer| selected_roi_profile_params(viewer, request.params.as_ref()))
                {
                    Ok(params) => params,
                    Err(error) => return command_registry::CommandExecuteResult::blocked(error),
                };
                match self.plot_stack_xy_profile(&viewer_label, &params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.stacks.label" | "image.stacks.set_label" => {
                if request.params.is_none() {
                    match self.open_stack_label_dialog(window_label) {
                        Ok(message) => command_registry::CommandExecuteResult::ok(message),
                        Err(error) => command_registry::CommandExecuteResult::blocked(error),
                    }
                } else {
                    let params =
                        command_registry::merge_params(&request.command_id, request.params);
                    match self.set_stack_slice_label(window_label, &params) {
                        Ok(message) => command_registry::CommandExecuteResult::ok(message),
                        Err(error) => command_registry::CommandExecuteResult::blocked(error),
                    }
                }
            }
            "image.stacks.remove_slice_labels" => {
                match self.remove_stack_slice_labels(window_label) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.stacks.statistics" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.stack.statistics".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "stack statistics started",
                        json!({ "job_id": ticket.job_id, "op": "image.stack.statistics" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.stacks.make_substack" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.stack.substack".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "substack started",
                        json!({ "job_id": ticket.job_id, "op": "image.stack.substack" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.stacks.reslice" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.stack.reslice".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "reslice started",
                        json!({ "job_id": ticket.job_id, "op": "image.stack.reslice" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.stacks.stack_to_images" => match self.stack_to_images(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "image.stacks.z_project" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.stack.z_project".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "Z projection started",
                        json!({ "job_id": ticket.job_id, "op": "image.stack.z_project" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.stacks.combine" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.combine_stacks(window_label, &params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.stacks.concatenate" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.concatenate_stacks(&params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.stacks.insert" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.insert_stack(window_label, &params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.stacks.make_montage" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.stack.montage".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "montage started",
                        json!({ "job_id": ticket.job_id, "op": "image.stack.montage" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.stacks.montage_to_stack" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.stack.montage_to_stack".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "montage-to-stack started",
                        json!({ "job_id": ticket.job_id, "op": "image.stack.montage_to_stack" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.stacks.grouped_z_project" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.stack.grouped_z_project".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "grouped Z projection started",
                        json!({ "job_id": ticket.job_id, "op": "image.stack.grouped_z_project" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.stacks.reduce" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.stack.reduce".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "stack reduction started",
                        json!({ "job_id": ticket.job_id, "op": "image.stack.reduce" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.hyperstacks.stack_to_hyperstack" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.stack.to_hyperstack".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "stack-to-hyperstack started",
                        json!({ "job_id": ticket.job_id, "op": "image.stack.to_hyperstack" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.hyperstacks.hyperstack_to_stack" => {
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.hyperstack.to_stack".to_string(),
                        params: json!({}),
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "hyperstack-to-stack started",
                        json!({ "job_id": ticket.job_id, "op": "image.hyperstack.to_stack" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.hyperstacks.reduce_dimensionality" => {
                let mut params =
                    command_registry::merge_params(&request.command_id, request.params);
                if let Some(viewer) = self.viewers_ui.get(window_label)
                    && let Some(map) = params.as_object_mut()
                {
                    if map.get("channel").map_or(true, Value::is_null) {
                        map.insert("channel".to_string(), json!(viewer.channel));
                    }
                    if map.get("z").map_or(true, Value::is_null) {
                        map.insert("z".to_string(), json!(viewer.z));
                    }
                    if map.get("time").map_or(true, Value::is_null) {
                        map.insert("time".to_string(), json!(viewer.t));
                    }
                }
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.hyperstack.reduce_dimensionality".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "hyperstack dimensionality reduction started",
                        json!({ "job_id": ticket.job_id, "op": "image.hyperstack.reduce_dimensionality" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.hyperstacks.make_subset" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.hyperstack.subset".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "hyperstack subset started",
                        json!({ "job_id": ticket.job_id, "op": "image.hyperstack.subset" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.transform.flip_horizontal"
            | "image.transform.flip_vertical"
            | "image.transform.flip_z"
            | "image.stacks.reverse" => {
                let axis = match request.command_id.as_str() {
                    "image.transform.flip_horizontal" => "horizontal",
                    "image.transform.flip_vertical" => "vertical",
                    "image.transform.flip_z" | "image.stacks.reverse" => "z",
                    _ => unreachable!(),
                };
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.flip".to_string(),
                        params: json!({ "axis": axis }),
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "flip started",
                        json!({ "job_id": ticket.job_id, "axis": axis }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.transform.rotate_right" | "image.transform.rotate_left" => {
                let direction = if request.command_id.ends_with("right") {
                    "right"
                } else {
                    "left"
                };
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.rotate_90".to_string(),
                        params: json!({ "direction": direction }),
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "rotation started",
                        json!({ "job_id": ticket.job_id, "direction": direction }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.transform.rotate" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.rotate".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "rotation started",
                        json!({ "job_id": ticket.job_id, "op": "image.rotate" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.transform.translate" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.translate".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "translation started",
                        json!({ "job_id": ticket.job_id, "op": "image.translate" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.transform.bin" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.bin".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "binning started",
                        json!({ "job_id": ticket.job_id, "op": "image.bin" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.transform.image_to_results" => match self.image_to_results(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "image.crop" => {
                let params = match request.params {
                    Some(Value::Object(map)) if !map.is_empty() => Value::Object(map),
                    Some(Value::Null) | None => {
                        let Some(viewer) = self.viewers_ui.get(window_label) else {
                            return command_registry::CommandExecuteResult::blocked(
                                "a loaded image is required for crop",
                            );
                        };
                        let Some((min_x, min_y, max_x, max_y)) = selected_roi_bbox(viewer) else {
                            return command_registry::CommandExecuteResult::blocked(
                                "a selection is required for crop",
                            );
                        };
                        json!({
                            "x": min_x,
                            "y": min_y,
                            "width": max_x.saturating_sub(min_x) + 1,
                            "height": max_y.saturating_sub(min_y) + 1
                        })
                    }
                    Some(other) => other,
                };
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.crop".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "crop started",
                        json!({ "job_id": ticket.job_id, "op": "image.crop" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.duplicate" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.duplicate_viewer(window_label, &params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.rename" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.rename_viewer(window_label, &params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.scale" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.scale".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "scale started",
                        json!({ "job_id": ticket.job_id, "op": "image.scale" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.transform.results_to_image" => match self.results_to_image() {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "process.filters.show_circular_masks" => match self.show_circular_masks() {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "image.overlay.add_selection" => match self.add_selection_to_overlay(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "image.overlay.flatten" => match self.flatten_overlay(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "image.lookup.apply_lut" => match self.apply_lookup_table(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "process.binary.options" => {
                if let Some(params) = request.params.as_ref() {
                    update_binary_options_from_params(
                        &mut self.desktop_state.binary_options,
                        params,
                    );
                } else {
                    clamp_binary_options(&mut self.desktop_state.binary_options);
                }
                self.desktop_state.utility_windows.binary_options_open = true;
                self.persist_desktop_state();
                command_registry::CommandExecuteResult::ok("binary options opened")
            }
            "image.overlay.labels" => {
                self.desktop_state.utility_windows.overlay_labels_open = true;
                self.persist_desktop_state();
                command_registry::CommandExecuteResult::ok("overlay labels opened")
            }
            "image.overlay.from_roi_manager" => match self.overlay_from_roi_manager(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "image.overlay.to_roi_manager" => match self.overlay_to_roi_manager(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "image.overlay.hide" => {
                match self.set_overlay_visibility(window_label, OverlayVisibility::Hide) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.overlay.show" => {
                match self.set_overlay_visibility(window_label, OverlayVisibility::Show) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.overlay.toggle" => {
                match self.set_overlay_visibility(window_label, OverlayVisibility::Toggle) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.overlay.remove" => match self.remove_overlay(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "image.overlay.options" => {
                self.desktop_state.utility_windows.overlay_options_open = true;
                self.persist_desktop_state();
                command_registry::CommandExecuteResult::ok("overlay options opened")
            }
            "image.overlay.list" => match self.list_overlay_elements(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "image.overlay.measure" => match self.measure_all_rois(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "image.show_info" | "image.properties" => match self.image_info_payload(window_label) {
                Ok(payload) => command_registry::CommandExecuteResult::with_payload(
                    "image information complete",
                    payload,
                ),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "process.smooth"
            | "process.gaussian"
            | "process.filters.gaussian"
            | "process.filters.gaussian_3d" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "gaussian.blur".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "processing operation started",
                        json!({ "job_id": ticket.job_id }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.adjust.size" => {
                if let Some(params) = request.params {
                    if let Some(result) =
                        self.resize_current_slice_new_window_if_requested(window_label, &params)
                    {
                        return result;
                    }
                    let mode = resize_op_mode_from_params(&params);
                    return match self.viewer_start_op(
                        window_label,
                        ViewerOpRequest {
                            op: "image.resize".to_string(),
                            params,
                            mode,
                        },
                    ) {
                        Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                            "resize started",
                            json!({ "job_id": ticket.job_id, "op": "image.resize" }),
                        ),
                        Err(error) => command_registry::CommandExecuteResult::blocked(error),
                    };
                }
                if let Some(session) = self.state.label_to_session.get(window_label) {
                    self.resize_dialog.width = session.committed_summary.shape[1];
                    self.resize_dialog.height = session.committed_summary.shape[0];
                    self.resize_dialog.original_width = session.committed_summary.shape[1];
                    self.resize_dialog.original_height = session.committed_summary.shape[0];
                    self.resize_dialog.x_scale = 1.0;
                    self.resize_dialog.y_scale = 1.0;
                    self.resize_dialog.z_scale = 1.0;
                    self.resize_dialog.depth = session.committed_summary.z_slices.max(1);
                    self.resize_dialog.original_depth = session.committed_summary.z_slices.max(1);
                    self.resize_dialog.frames = session.committed_summary.times.max(1);
                    self.resize_dialog.fill_with_background_available =
                        scale_fill_with_background_available(session);
                    self.resize_dialog.fill_with_background = false;
                    self.resize_dialog.process_stack = true;
                    self.resize_dialog.process_stack_available =
                        scale_process_stack_available(session);
                    self.resize_dialog.create_new_window = true;
                    self.resize_dialog.title = session
                        .path
                        .file_stem()
                        .and_then(|name| name.to_str())
                        .map(sanitize_image_title)
                        .unwrap_or_else(|| "Untitled".to_string());
                }
                self.resize_dialog.open = true;
                command_registry::CommandExecuteResult::ok("resize dialog opened")
            }
            "image.adjust.canvas" => {
                if let Some(params) = request.params {
                    let mut params = params;
                    if params.get("fill").is_none()
                        && !params.get("zero").and_then(Value::as_bool).unwrap_or(false)
                    {
                        if let Some(map) = params.as_object_mut() {
                            let background =
                                f32::from(self.tool_options.background_color.r()) / 255.0;
                            map.insert("fill".to_string(), json!(background));
                        }
                    }
                    return match self.viewer_start_op(
                        window_label,
                        ViewerOpRequest {
                            op: "image.canvas_resize".to_string(),
                            params,
                            mode: OpRunMode::Apply,
                        },
                    ) {
                        Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                            "canvas resize started",
                            json!({ "job_id": ticket.job_id, "op": "image.canvas_resize" }),
                        ),
                        Err(error) => command_registry::CommandExecuteResult::blocked(error),
                    };
                }
                if let Some(session) = self.state.label_to_session.get(window_label) {
                    self.canvas_dialog.width = session.committed_summary.shape[1];
                    self.canvas_dialog.height = session.committed_summary.shape[0];
                    self.canvas_dialog.original_width = session.committed_summary.shape[1];
                    self.canvas_dialog.original_height = session.committed_summary.shape[0];
                    self.canvas_dialog.x_scale = 1.0;
                    self.canvas_dialog.y_scale = 1.0;
                    self.canvas_dialog.z_scale = 1.0;
                    self.canvas_dialog.depth = session.committed_summary.z_slices.max(1);
                    self.canvas_dialog.original_depth = session.committed_summary.z_slices.max(1);
                    self.canvas_dialog.frames = session.committed_summary.times.max(1);
                }
                self.canvas_dialog.open = true;
                command_registry::CommandExecuteResult::ok("canvas size dialog opened")
            }
            "image.adjust.coordinates" => {
                if request.params.is_none() {
                    return self.open_adjust_dialog(window_label, AdjustDialogKind::Coordinates);
                }
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.coordinates".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "coordinate calibration started",
                        json!({ "job_id": ticket.job_id, "op": "image.coordinates" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.adjust.brightness" => {
                if request.params.is_none() {
                    return self
                        .open_adjust_dialog(window_label, AdjustDialogKind::BrightnessContrast);
                }
                let params = command_registry::merge_params(&request.command_id, request.params);
                if params
                    .get("reset")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    return match self.reset_display_range(window_label) {
                        Ok(message) => command_registry::CommandExecuteResult::ok(message),
                        Err(error) => command_registry::CommandExecuteResult::blocked(error),
                    };
                }
                if params
                    .get("apply")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    if let Err(error) = self.ensure_apply_lut_supported(window_label) {
                        return command_registry::CommandExecuteResult::blocked(error);
                    }
                    let low = params
                        .get("min")
                        .and_then(Value::as_f64)
                        .map(|value| value as f32)
                        .unwrap_or(0.0);
                    let high = params
                        .get("max")
                        .and_then(Value::as_f64)
                        .map(|value| value as f32)
                        .unwrap_or(1.0);
                    return match self.viewer_start_op(
                        window_label,
                        ViewerOpRequest {
                            op: "intensity.window".to_string(),
                            params: json!({ "low": low, "high": high }),
                            mode: OpRunMode::Apply,
                        },
                    ) {
                        Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                            "apply LUT started",
                            json!({ "job_id": ticket.job_id, "op": "intensity.window" }),
                        ),
                        Err(error) => command_registry::CommandExecuteResult::blocked(error),
                    };
                }
                match self.apply_display_range(window_label, &params, "min", "max") {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "process.enhance_contrast" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "intensity.enhance_contrast".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "enhance contrast started",
                        json!({ "job_id": ticket.job_id, "op": "intensity.enhance_contrast" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "process.subtract_background" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.subtract_background".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "subtract background started",
                        json!({ "job_id": ticket.job_id, "op": "image.subtract_background" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.adjust.window_level" => {
                if request.params.is_none() {
                    return self.open_adjust_dialog(window_label, AdjustDialogKind::WindowLevel);
                }
                let params = command_registry::merge_params(&request.command_id, request.params);
                if params
                    .get("apply")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    if let Err(error) = self.ensure_apply_lut_supported(window_label) {
                        return command_registry::CommandExecuteResult::blocked(error);
                    }
                    let low = params
                        .get("low")
                        .and_then(Value::as_f64)
                        .map(|value| value as f32)
                        .unwrap_or(0.0);
                    let high = params
                        .get("high")
                        .and_then(Value::as_f64)
                        .map(|value| value as f32)
                        .unwrap_or(1.0);
                    return match self.viewer_start_op(
                        window_label,
                        ViewerOpRequest {
                            op: "intensity.window".to_string(),
                            params: json!({ "low": low, "high": high }),
                            mode: OpRunMode::Apply,
                        },
                    ) {
                        Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                            "apply LUT started",
                            json!({ "job_id": ticket.job_id, "op": "intensity.window" }),
                        ),
                        Err(error) => command_registry::CommandExecuteResult::blocked(error),
                    };
                }
                match self.apply_display_range(window_label, &params, "low", "high") {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "image.adjust.threshold" => {
                if request.params.is_none() {
                    if let Err(error) = self.threshold_adjuster_supports_viewer(window_label) {
                        return command_registry::CommandExecuteResult::blocked(error);
                    }
                    return self.open_adjust_dialog(window_label, AdjustDialogKind::Threshold);
                }
                let params = command_registry::merge_params(&request.command_id, request.params);
                if !params
                    .get("apply")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    return match self.set_threshold_overlay(window_label, &params) {
                        Ok(message) => command_registry::CommandExecuteResult::ok(message),
                        Err(error) => command_registry::CommandExecuteResult::blocked(error),
                    };
                }
                if params
                    .get("background_to_nan")
                    .or_else(|| params.get("nan_background"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    let (lower, upper) = match self.threshold_apply_bounds(window_label, &params) {
                        Ok(bounds) => bounds,
                        Err(error) => {
                            return command_registry::CommandExecuteResult::blocked(error);
                        }
                    };
                    return match self.viewer_start_op(
                        window_label,
                        ViewerOpRequest {
                            op: "intensity.nan_background".to_string(),
                            params: json!({ "lower": lower, "upper": upper }),
                            mode: OpRunMode::Apply,
                        },
                    ) {
                        Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                            "nan background started",
                            json!({ "job_id": ticket.job_id, "op": "intensity.nan_background" }),
                        ),
                        Err(error) => command_registry::CommandExecuteResult::blocked(error),
                    };
                }
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "threshold.make_binary".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "binary threshold started",
                        json!({ "job_id": ticket.job_id, "op": "threshold.make_binary" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "process.binary.make" | "process.binary.convert_mask" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "threshold.make_binary".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "binary mask conversion started",
                        json!({ "job_id": ticket.job_id, "op": "threshold.make_binary" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
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
            | "process.binary.voronoi" => {
                let op = match request.command_id.as_str() {
                    "process.binary.erode" => "morphology.erode",
                    "process.binary.dilate" => "morphology.dilate",
                    "process.binary.open" => "morphology.open",
                    "process.binary.close" => "morphology.close",
                    "process.binary.median" => "morphology.binary_median",
                    "process.binary.outline" => "morphology.outline",
                    "process.binary.fill_holes" => "morphology.fill_holes",
                    "process.binary.skeletonize" => "morphology.skeletonize",
                    "process.binary.distance_map" => "morphology.distance_map",
                    "process.binary.ultimate_points" => "morphology.ultimate_points",
                    "process.binary.watershed" => "morphology.watershed",
                    "process.binary.voronoi" => "morphology.voronoi",
                    _ => unreachable!(),
                };
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: op.to_string(),
                        params: binary_morphology_params(
                            &request.command_id,
                            request.params,
                            &self.desktop_state.binary_options,
                        ),
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "binary morphology started",
                        json!({ "job_id": ticket.job_id, "op": op }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            command_id if command_id.starts_with("process.math.") => {
                if command_id == "process.math.nan_background" {
                    let params =
                        command_registry::merge_params(&request.command_id, request.params);
                    return match self.viewer_start_op(
                        window_label,
                        ViewerOpRequest {
                            op: "intensity.nan_background".to_string(),
                            params,
                            mode: OpRunMode::Apply,
                        },
                    ) {
                        Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                            "nan background started",
                            json!({ "job_id": ticket.job_id, "op": "intensity.nan_background" }),
                        ),
                        Err(error) => command_registry::CommandExecuteResult::blocked(error),
                    };
                }
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "intensity.math".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "math operation started",
                        json!({ "job_id": ticket.job_id, "op": "intensity.math" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "process.noise.add" | "process.noise.specified" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "noise.gaussian".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "noise operation started",
                        json!({ "job_id": ticket.job_id, "op": "noise.gaussian" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "process.noise.salt_pepper" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "noise.salt_and_pepper".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "salt and pepper noise started",
                        json!({ "job_id": ticket.job_id, "op": "noise.salt_and_pepper" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "process.noise.despeckle" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.median_filter".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "despeckle started",
                        json!({ "job_id": ticket.job_id, "op": "image.median_filter" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "process.noise.remove_nans" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.remove_nans".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "remove NaNs started",
                        json!({ "job_id": ticket.job_id, "op": "image.remove_nans" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "process.noise.remove_outliers" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.remove_outliers".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "remove outliers started",
                        json!({ "job_id": ticket.job_id, "op": "image.remove_outliers" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "process.shadows.demo" => match self.viewer_start_op(
                window_label,
                ViewerOpRequest {
                    op: "image.shadow_demo".to_string(),
                    params: command_registry::merge_params(&request.command_id, request.params),
                    mode: OpRunMode::Apply,
                },
            ) {
                Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                    "shadows demo started",
                    json!({ "job_id": ticket.job_id, "op": "image.shadow_demo" }),
                ),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            command_id if command_id.starts_with("process.shadows.") => {
                let direction = command_id
                    .strip_prefix("process.shadows.")
                    .unwrap_or_default()
                    .to_string();
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.shadow".to_string(),
                        params: json!({ "direction": direction }),
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "shadow filter started",
                        json!({ "job_id": ticket.job_id, "op": "image.shadow" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "process.filters.median" => match self.viewer_start_op(
                window_label,
                ViewerOpRequest {
                    op: "image.median_filter".to_string(),
                    params: command_registry::merge_params(&request.command_id, request.params),
                    mode: OpRunMode::Apply,
                },
            ) {
                Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                    "rank filter started",
                    json!({ "job_id": ticket.job_id, "op": "image.median_filter" }),
                ),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "process.filters.convolve" => match self.viewer_start_op(
                window_label,
                ViewerOpRequest {
                    op: "image.convolve".to_string(),
                    params: command_registry::merge_params(&request.command_id, request.params),
                    mode: OpRunMode::Apply,
                },
            ) {
                Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                    "convolve filter started",
                    json!({ "job_id": ticket.job_id, "op": "image.convolve" }),
                ),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "process.filters.median_3d"
            | "process.filters.mean_3d"
            | "process.filters.minimum_3d"
            | "process.filters.maximum_3d"
            | "process.filters.variance_3d" => {
                let filter = request
                    .command_id
                    .strip_prefix("process.filters.")
                    .and_then(|value| value.strip_suffix("_3d"))
                    .unwrap_or_default()
                    .to_string();
                let mut params =
                    command_registry::merge_params(&request.command_id, request.params);
                if let Some(params) = params.as_object_mut() {
                    params.insert("filter".to_string(), json!(filter));
                }
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.rank_filter_3d".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "3D rank filter started",
                        json!({ "job_id": ticket.job_id, "op": "image.rank_filter_3d" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "process.filters.unsharp_mask" => match self.viewer_start_op(
                window_label,
                ViewerOpRequest {
                    op: "image.unsharp_mask".to_string(),
                    params: command_registry::merge_params(&request.command_id, request.params),
                    mode: OpRunMode::Apply,
                },
            ) {
                Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                    "unsharp mask started",
                    json!({ "job_id": ticket.job_id, "op": "image.unsharp_mask" }),
                ),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            command_id if command_id.starts_with("process.filters.") => {
                let filter = command_id
                    .strip_prefix("process.filters.")
                    .unwrap_or_default()
                    .to_string();
                let mut params =
                    command_registry::merge_params(&request.command_id, request.params);
                if let Some(params) = params.as_object_mut() {
                    params.insert("filter".to_string(), json!(filter));
                }
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.rank_filter".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "rank filter started",
                        json!({ "job_id": ticket.job_id, "op": "image.rank_filter" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "process.sharpen" | "process.find_edges" => {
                let op = if request.command_id == "process.sharpen" {
                    "image.sharpen"
                } else {
                    "image.find_edges"
                };
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: op.to_string(),
                        params: json!({}),
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "filter started",
                        json!({ "job_id": ticket.job_id, "op": op }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "process.find_maxima" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.find_maxima".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "find maxima started",
                        json!({ "job_id": ticket.job_id, "op": "image.find_maxima" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "process.fft.swap_quadrants" => match self.viewer_start_op(
                window_label,
                ViewerOpRequest {
                    op: "image.swap_quadrants".to_string(),
                    params: json!({}),
                    mode: OpRunMode::Apply,
                },
            ) {
                Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                    "swap quadrants started",
                    json!({ "job_id": ticket.job_id, "op": "image.swap_quadrants" }),
                ),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "process.fft.fft" => match self.viewer_start_op(
                window_label,
                ViewerOpRequest {
                    op: "image.fft_power_spectrum".to_string(),
                    params: json!({}),
                    mode: OpRunMode::Apply,
                },
            ) {
                Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                    "FFT power spectrum started",
                    json!({ "job_id": ticket.job_id, "op": "image.fft_power_spectrum" }),
                ),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "process.fft.bandpass" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.fft_bandpass".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "FFT bandpass started",
                        json!({ "job_id": ticket.job_id, "op": "image.fft_bandpass" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "analyze.measure" => match self.measure_active_viewer(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "analyze.set_scale" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.set_scale".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "set scale started",
                        json!({ "job_id": ticket.job_id, "op": "image.set_scale" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "analyze.calibrate" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "image.calibrate".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "calibrate started",
                        json!({ "job_id": ticket.job_id, "op": "image.calibrate" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "analyze.histogram" => {
                let Some(viewer) = self.viewers_ui.get(window_label) else {
                    return command_registry::CommandExecuteResult::blocked(
                        "a loaded image is required for histogram",
                    );
                };
                let mut params =
                    command_registry::merge_params(&request.command_id, request.params);
                if let Some(map) = params.as_object_mut() {
                    map.entry("z".to_string())
                        .or_insert_with(|| json!(viewer.z));
                    map.entry("t".to_string())
                        .or_insert_with(|| json!(viewer.t));
                    map.entry("channel".to_string())
                        .or_insert_with(|| json!(viewer.channel));
                }
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "measurements.histogram".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "histogram started",
                        json!({ "job_id": ticket.job_id, "op": "measurements.histogram" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "analyze.set_measurements" => {
                self.desktop_state.utility_windows.measurements_open = true;
                self.persist_desktop_state();
                command_registry::CommandExecuteResult::ok("measurement settings opened")
            }
            "analyze.summarize" => match self.summarize_results() {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "analyze.distribution" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.results_distribution_payload(&params) {
                    Ok(payload) => command_registry::CommandExecuteResult::with_payload(
                        "distribution complete",
                        payload,
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "analyze.clear_results" => {
                command_registry::CommandExecuteResult::ok(self.clear_results())
            }
            "analyze.label" => match self.label_active_selection(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "analyze.surface_plot" => {
                let Some(viewer) = self.viewers_ui.get(window_label) else {
                    return command_registry::CommandExecuteResult::blocked(
                        "a loaded image is required for surface plot",
                    );
                };
                let mut params =
                    command_registry::merge_params(&request.command_id, request.params);
                if let Some(map) = params.as_object_mut() {
                    map.entry("z".to_string())
                        .or_insert_with(|| json!(viewer.z));
                    map.entry("time".to_string())
                        .or_insert_with(|| json!(viewer.t));
                    map.entry("channel".to_string())
                        .or_insert_with(|| json!(viewer.channel));
                }
                match self.surface_plot_viewer(window_label, &params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "analyze.tools.results" => {
                self.desktop_state.utility_windows.results_open = true;
                self.persist_desktop_state();
                command_registry::CommandExecuteResult::ok("results table opened")
            }
            "analyze.tools.save_xy_coordinates" => {
                let params = command_registry::merge_params(&request.command_id, request.params);
                match self.save_xy_coordinates(window_label, &params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "analyze.tools.roi_manager" => {
                self.desktop_state.utility_windows.roi_manager_open = true;
                self.persist_desktop_state();
                command_registry::CommandExecuteResult::ok("ROI manager opened")
            }
            "analyze.plot_profile" => {
                let params = match self
                    .viewers_ui
                    .get(window_label)
                    .ok_or_else(|| "a loaded image is required for plot profile".to_string())
                    .and_then(|viewer| selected_roi_profile_params(viewer, request.params.as_ref()))
                {
                    Ok(params) => params,
                    Err(error) => return command_registry::CommandExecuteResult::blocked(error),
                };
                match self.viewer_start_op(
                    window_label,
                    ViewerOpRequest {
                        op: "measurements.profile".to_string(),
                        params,
                        mode: OpRunMode::Apply,
                    },
                ) {
                    Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                        "profile plot started",
                        json!({ "job_id": ticket.job_id, "op": "measurements.profile" }),
                    ),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "analyze.analyze_particles" => match self.analyze_particles(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "edit.copy" => match self.copy_selection(window_label, false) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "edit.cut" => match self.copy_selection(window_label, true) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "edit.paste" => match self.paste_selection(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "window.tile" => match self.layout_viewers(LayoutMode::Tile) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "window.cascade" => match self.layout_viewers(LayoutMode::Cascade) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "window.show_all" => match self.show_all_windows() {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "help.about" => {
                self.desktop_state.utility_windows.help_about_open = true;
                self.persist_desktop_state();
                command_registry::CommandExecuteResult::ok("about window opened")
            }
            "help.docs" => {
                self.desktop_state.utility_windows.help_docs_open = true;
                self.persist_desktop_state();
                command_registry::CommandExecuteResult::ok("documentation window opened")
            }
            "help.shortcuts" => {
                self.desktop_state.utility_windows.help_shortcuts_open = true;
                self.persist_desktop_state();
                command_registry::CommandExecuteResult::ok("shortcuts window opened")
            }
            "plugins.commands.find" => {
                self.desktop_state.utility_windows.command_finder_open = true;
                self.persist_desktop_state();
                command_registry::CommandExecuteResult::ok("command finder opened")
            }
            "plugins.macros.run"
            | "plugins.macros.record"
            | "plugins.macros.install"
            | "plugins.utilities.startup" => {
                if matches!(request.command_id.as_str(), "plugins.macros.run") {
                    let Some(path) = FileDialog::new()
                        .add_filter("ImageJ Macros", &["ijm", "txt"])
                        .set_title("Run macro")
                        .pick_file()
                    else {
                        self.set_fallback_status("macro run canceled");
                        self.macro_compatibility_command = Some("plugins.macros.run".to_string());
                        self.macro_compatibility_open = true;
                        self.macro_compatibility_path = None;
                        self.macro_compatibility_preview =
                            "Select an installed macro below or use Install... to add one."
                                .to_string();
                        self.macro_compatibility_run_log = String::new();
                        return command_registry::CommandExecuteResult::ok("macro run canceled");
                    };
                    self.open_macro_compatibility_window_for_file("plugins.macros.run", &path);
                    self.set_fallback_status("macro file selected for review");
                    return command_registry::CommandExecuteResult::ok(format!(
                        "macro file selected: {}",
                        path.display()
                    ));
                }

                if matches!(request.command_id.as_str(), "plugins.macros.install") {
                    let Some(path) = FileDialog::new()
                        .add_filter("ImageJ Macro", &["ijm", "txt"])
                        .set_title("Install macro plugin")
                        .pick_file()
                    else {
                        self.set_fallback_status("macro install canceled");
                        return command_registry::CommandExecuteResult::ok(
                            "macro install canceled",
                        );
                    };
                    let installed_path = match self.install_macro_file(&path) {
                        Ok(path) => path,
                        Err(error) => {
                            self.set_fallback_status(error.clone());
                            return command_registry::CommandExecuteResult::blocked(error);
                        }
                    };
                    self.open_macro_compatibility_window_for_file(
                        "plugins.macros.install",
                        &installed_path,
                    );
                    self.set_fallback_status("macro installed");
                    return command_registry::CommandExecuteResult::ok(format!(
                        "macro installed: {}",
                        installed_path.display()
                    ));
                }
                if matches!(request.command_id.as_str(), "plugins.macros.record") {
                    self.macro_recorder.open = true;
                    self.macro_recorder.recording = true;
                    self.set_fallback_status("macro recorder opened");
                    return command_registry::CommandExecuteResult::ok("macro recorder opened");
                }
                if matches!(request.command_id.as_str(), "plugins.utilities.startup") {
                    self.open_startup_macro_window();
                    self.set_fallback_status("startup macro editor opened");
                    return command_registry::CommandExecuteResult::ok(
                        "startup macro editor opened",
                    );
                }

                self.macro_compatibility_command = Some(request.command_id.clone());
                self.macro_compatibility_open = true;
                self.macro_compatibility_path = None;
                self.macro_compatibility_preview = String::new();
                self.set_fallback_status(format!(
                    "{} is not yet fully supported",
                    request.command_id
                ));
                command_registry::CommandExecuteResult::ok(format!(
                    "{} command opened in compatibility window",
                    request.command_id
                ))
            }
            _ => command_registry::CommandExecuteResult::unimplemented(format!(
                "command `{}` has no backend handler yet",
                request.command_id
            )),
        }
    }

    fn viewer_start_op(
        &mut self,
        window_label: &str,
        request: ViewerOpRequest,
    ) -> Result<JobTicket, String> {
        let job_id = self.state.next_job_id();
        let preview_key = match request.mode {
            OpRunMode::Preview => Some(preview_cache_key(&request.op, &request.params)?),
            OpRunMode::Apply | OpRunMode::NewWindow => None,
        };
        let new_window_title = request
            .params
            .get("title")
            .and_then(Value::as_str)
            .map(str::to_string);

        if let Some(key) = &preview_key
            && let Some(session) = self.state.label_to_session.get_mut(window_label)
            && session.preview_cache.contains_key(key)
        {
            session.generation = session.generation.saturating_add(1);
            session.active_job = None;
            session.set_active_preview(Some(key.clone()));

            if let Some(viewer) = self.viewers_ui.get_mut(window_label) {
                viewer.status_message =
                    format!("Preview cached for {} (job {})", request.op, job_id);
                viewer.last_generation = 0;
                viewer.last_request = None;
            }

            return Ok(JobTicket { job_id });
        }

        let (input_dataset, generation) = {
            let session = self
                .state
                .label_to_session
                .get_mut(window_label)
                .ok_or_else(|| format!("no viewer session for `{window_label}`"))?;

            session.generation = session.generation.saturating_add(1);
            let generation = session.generation;
            session.active_job = Some(ActiveJob {
                job_id,
                generation,
                mode: request.mode.clone(),
                op: request.op.clone(),
            });
            (session.ensure_committed_dataset()?, generation)
        };

        if let Some(viewer) = self.viewers_ui.get_mut(window_label) {
            viewer.status_message = format!("Running {} ({})", request.op, request.mode.as_str());
        }

        let mode = request.mode.clone();
        let op_name = request.op.clone();
        let params = request.params.clone();
        let preview_key_for_task = preview_key.clone();
        let tx = self.worker_tx.clone();
        let window_label = window_label.to_string();
        let ops_service = self.state.app.ops_service().clone();

        std::thread::spawn(move || {
            let result = ops_service
                .execute(&op_name, input_dataset.as_ref(), &params)
                .map(|output| OpRunOutput {
                    dataset: Arc::new(output.dataset),
                    measurements: output.measurements,
                })
                .map_err(|error| error.to_string());

            let _ = tx.send(WorkerEvent::OpFinished {
                window_label,
                job_id,
                generation,
                mode,
                op: op_name,
                new_window_title,
                preview_key: preview_key_for_task,
                result,
            });
        });

        Ok(JobTicket { job_id })
    }

    fn ensure_frame_for_viewer(&mut self, ctx: &egui::Context, label: &str) {
        let Some(viewer) = self.viewers_ui.get(label) else {
            return;
        };

        let request = ViewerFrameRequest {
            z: viewer.z,
            t: viewer.t,
            channel: viewer.channel,
        };

        let generation = self
            .state
            .label_to_session
            .get(label)
            .map(|session| session.generation)
            .unwrap_or_default();

        let needs_refresh = viewer.last_request.as_ref() != Some(&request)
            || viewer.last_generation != generation
            || viewer.frame.is_none();

        if !needs_refresh {
            return;
        }

        match compute_viewer_frame(&mut self.state, label, &request, None) {
            Ok(frame) => {
                let threshold_overlay = self
                    .state
                    .label_to_session
                    .get(label)
                    .and_then(|session| session.threshold_overlay);
                let color =
                    to_color_image_with_threshold(&frame, viewer.lookup_table, threshold_overlay);
                let viewer_state = self
                    .viewers_ui
                    .get_mut(label)
                    .expect("viewer ui state should exist");

                if let Some(texture) = &mut viewer_state.texture {
                    texture.set(color, egui::TextureOptions::NEAREST);
                } else {
                    viewer_state.texture = Some(ctx.load_texture(
                        format!("viewer-texture-{label}"),
                        color,
                        egui::TextureOptions::NEAREST,
                    ));
                }

                viewer_state.frame = Some(frame);
                viewer_state.last_request = Some(request);
                viewer_state.last_generation = generation;
                viewer_state.status_message = "Rendered".to_string();
            }
            Err(error) => {
                if let Some(viewer_state) = self.viewers_ui.get_mut(label) {
                    viewer_state.status_message = error;
                }
            }
        }
    }

    fn command_is_enabled(&self, window_label: &str, command_id: &str, menu_enabled: bool) -> bool {
        if !menu_enabled {
            return false;
        }
        if command_id == "process.repeat_command" {
            return self.last_repeatable_command.is_some();
        }
        let metadata = command_registry::metadata(command_id);
        let effective_label = if window_label == LAUNCHER_LABEL
            && matches!(metadata.scope, command_registry::CommandScope::Viewer)
        {
            match self.active_viewer_label.as_deref() {
                Some(label) => label,
                None => return false,
            }
        } else {
            window_label
        };
        if !metadata.implemented || !metadata.scope.contains(effective_label) {
            return false;
        }
        if metadata.requires_image && !self.state.label_to_session.contains_key(effective_label) {
            return false;
        }
        if let Some(session) = self.state.label_to_session.get(effective_label) {
            match command_id {
                "edit.undo" => return session.can_undo(),
                "edit.redo" => return session.can_redo(),
                "file.revert" => {
                    return !source_ptr_eq(&session.committed_source, &session.base_source);
                }
                _ => {}
            }
        }
        if command_id == "file.open_recent.none" {
            return self
                .desktop_state
                .recent_files
                .iter()
                .any(|path| normalize_path(path).exists());
        }
        true
    }

    fn draw_menu_bar(&self, ui: &mut egui::Ui, window_label: &str, actions: &mut Vec<UiAction>) {
        egui::menu::bar(ui, |ui| {
            for top in &self.menus {
                ui.menu_button(&top.label, |ui| {
                    self.draw_menu_items(ui, window_label, &top.items, actions);
                });
            }
        });
    }

    fn draw_menu_items(
        &self,
        ui: &mut egui::Ui,
        window_label: &str,
        items: &[menu::MenuManifestItem],
        actions: &mut Vec<UiAction>,
    ) {
        for item in items {
            match item.kind.as_str() {
                "separator" => {
                    ui.separator();
                }
                "submenu" => {
                    let label = item.label.as_deref().unwrap_or("Submenu");
                    ui.menu_button(label, |ui| {
                        if item.id.as_deref() == Some("file.open_recent") {
                            let mut any = false;
                            for path in &self.desktop_state.recent_files {
                                let normalized = normalize_path(path);
                                if !normalized.exists() {
                                    continue;
                                }
                                any = true;
                                let caption = normalized.display().to_string();
                                if ui.button(&caption).clicked() {
                                    actions.push(UiAction::Command {
                                        window_label: window_label.to_string(),
                                        command_id: format!(
                                            "file.open_recent:{}",
                                            normalized.display()
                                        ),
                                        params: None,
                                    });
                                    ui.close_menu();
                                }
                            }
                            if !any {
                                ui.add_enabled(false, egui::Button::new("(empty)"));
                            }
                        } else {
                            let children = item.items.clone().unwrap_or_default();
                            self.draw_menu_items(ui, window_label, &children, actions);
                            if item.id.as_deref() == Some("plugins.macros") {
                                self.draw_installed_macro_menu_items(ui, window_label, actions);
                            }
                        }
                    });
                }
                "item" => {
                    let command_id = item
                        .command
                        .as_ref()
                        .or(item.id.as_ref())
                        .cloned()
                        .unwrap_or_default();
                    if command_id.is_empty() {
                        continue;
                    }

                    let label = item.label.clone().unwrap_or_else(|| command_id.clone());
                    let enabled = self.command_is_enabled(
                        window_label,
                        &command_id,
                        item.enabled.unwrap_or(true),
                    );
                    let caption = if let Some(shortcut) = &item.shortcut {
                        format!("{label}    {shortcut}")
                    } else {
                        label
                    };

                    if ui
                        .add_enabled(enabled, egui::Button::new(caption))
                        .clicked()
                    {
                        actions.push(UiAction::Command {
                            window_label: window_label.to_string(),
                            command_id,
                            params: None,
                        });
                        ui.close_menu();
                    }
                }
                _ => {
                    if let Some(children) = &item.items {
                        self.draw_menu_items(ui, window_label, children, actions);
                    }
                }
            }
        }
    }

    fn draw_installed_macro_menu_items(
        &self,
        ui: &mut egui::Ui,
        window_label: &str,
        actions: &mut Vec<UiAction>,
    ) {
        let entries = self.installed_macro_menu_entries();
        if entries.is_empty() {
            return;
        }

        ui.separator();
        let mut submenus: BTreeMap<String, Vec<InstalledMacroMenuEntry>> = BTreeMap::new();
        for entry in entries {
            if let Some(submenu) = entry.submenu.clone() {
                submenus.entry(submenu).or_default().push(entry);
            } else {
                self.draw_installed_macro_menu_entry(ui, window_label, actions, &entry);
            }
        }
        for (submenu, entries) in submenus {
            ui.menu_button(submenu, |ui| {
                for entry in entries {
                    self.draw_installed_macro_menu_entry(ui, window_label, actions, &entry);
                }
            });
        }
    }

    fn draw_installed_macro_menu_entry(
        &self,
        ui: &mut egui::Ui,
        window_label: &str,
        actions: &mut Vec<UiAction>,
        entry: &InstalledMacroMenuEntry,
    ) {
        let caption = if let Some(shortcut) = &entry.shortcut {
            format!("{}    {}", entry.label, shortcut)
        } else {
            entry.label.clone()
        };
        if ui.button(caption).clicked() {
            actions.push(UiAction::RunInstalledMacro {
                window_label: self.macro_shortcut_window_label(window_label),
                path: entry.path.clone(),
                macro_name: entry.macro_name.clone(),
            });
            ui.close_menu();
        }
    }

    fn draw_launcher_toolbar(&mut self, ui: &mut egui::Ui, actions: &mut Vec<UiAction>) {
        for item in launcher_toolbar_items() {
            if item.kind == ToolbarKind::Separator {
                ui.separator();
                continue;
            }

            let enabled = self.command_is_enabled(LAUNCHER_LABEL, item.command_id, true);
            let icon_texture = self.toolbar_texture(ui.ctx(), item.icon).clone();
            let icon = egui::Image::new((icon_texture.id(), icon_texture.size_vec2()))
                .fit_to_exact_size(egui::vec2(16.0, 16.0))
                .tint(if enabled {
                    egui::Color32::WHITE
                } else {
                    egui::Color32::GRAY
                });

            let is_tool_selected = tool_from_command_id(item.command_id)
                .is_some_and(|tool| self.tool_state.selected == tool);
            let mut button = egui::Button::image(icon)
                .small()
                .frame(true)
                .min_size(egui::vec2(22.0, 22.0));
            if is_tool_selected {
                button = button.fill(egui::Color32::from_rgb(52, 91, 146));
            }

            let response = ui.add_enabled(enabled, button).on_hover_text(item.label);
            if response.clicked() {
                actions.push(UiAction::Command {
                    window_label: LAUNCHER_LABEL.to_string(),
                    command_id: item.command_id.to_string(),
                    params: None,
                });
            }
            response.context_menu(|ui| match item.command_id {
                "launcher.tool.rect" => {
                    if ui.button("Rectangle").clicked() {
                        actions.push(UiAction::Command {
                            window_label: LAUNCHER_LABEL.to_string(),
                            command_id: "launcher.tool.rect.mode.rectangle".to_string(),
                            params: None,
                        });
                        ui.close_menu();
                    }
                    if ui.button("Rounded Rectangle").clicked() {
                        actions.push(UiAction::Command {
                            window_label: LAUNCHER_LABEL.to_string(),
                            command_id: "launcher.tool.rect.mode.rounded".to_string(),
                            params: None,
                        });
                        ui.close_menu();
                    }
                    if ui.button("Rotated Rectangle").clicked() {
                        actions.push(UiAction::Command {
                            window_label: LAUNCHER_LABEL.to_string(),
                            command_id: "launcher.tool.rect.mode.rotated".to_string(),
                            params: None,
                        });
                        ui.close_menu();
                    }
                }
                "launcher.tool.oval" => {
                    if ui.button("Oval").clicked() {
                        actions.push(UiAction::Command {
                            window_label: LAUNCHER_LABEL.to_string(),
                            command_id: "launcher.tool.oval.mode.oval".to_string(),
                            params: None,
                        });
                        ui.close_menu();
                    }
                    if ui.button("Ellipse").clicked() {
                        actions.push(UiAction::Command {
                            window_label: LAUNCHER_LABEL.to_string(),
                            command_id: "launcher.tool.oval.mode.ellipse".to_string(),
                            params: None,
                        });
                        ui.close_menu();
                    }
                    if ui.button("Brush").clicked() {
                        actions.push(UiAction::Command {
                            window_label: LAUNCHER_LABEL.to_string(),
                            command_id: "launcher.tool.oval.mode.brush".to_string(),
                            params: None,
                        });
                        ui.close_menu();
                    }
                }
                "launcher.tool.line" => {
                    if ui.button("Straight").clicked() {
                        actions.push(UiAction::Command {
                            window_label: LAUNCHER_LABEL.to_string(),
                            command_id: "launcher.tool.line.mode.straight".to_string(),
                            params: None,
                        });
                        ui.close_menu();
                    }
                    if ui.button("Segmented").clicked() {
                        actions.push(UiAction::Command {
                            window_label: LAUNCHER_LABEL.to_string(),
                            command_id: "launcher.tool.line.mode.segmented".to_string(),
                            params: None,
                        });
                        ui.close_menu();
                    }
                    if ui.button("Freehand").clicked() {
                        actions.push(UiAction::Command {
                            window_label: LAUNCHER_LABEL.to_string(),
                            command_id: "launcher.tool.line.mode.freehand".to_string(),
                            params: None,
                        });
                        ui.close_menu();
                    }
                    if ui.button("Arrow").clicked() {
                        actions.push(UiAction::Command {
                            window_label: LAUNCHER_LABEL.to_string(),
                            command_id: "launcher.tool.line.mode.arrow".to_string(),
                            params: None,
                        });
                        ui.close_menu();
                    }
                }
                "launcher.tool.point" => {
                    if ui.button("Point").clicked() {
                        actions.push(UiAction::Command {
                            window_label: LAUNCHER_LABEL.to_string(),
                            command_id: "launcher.tool.point.mode.point".to_string(),
                            params: None,
                        });
                        ui.close_menu();
                    }
                    if ui.button("Multi-point").clicked() {
                        actions.push(UiAction::Command {
                            window_label: LAUNCHER_LABEL.to_string(),
                            command_id: "launcher.tool.point.mode.multipoint".to_string(),
                            params: None,
                        });
                        ui.close_menu();
                    }
                }
                "launcher.tool.zoom" => {
                    if ui.button("Original Scale").clicked() {
                        actions.push(UiAction::Command {
                            window_label: self
                                .active_viewer_label
                                .clone()
                                .unwrap_or_else(|| LAUNCHER_LABEL.to_string()),
                            command_id: "image.zoom.original".to_string(),
                            params: None,
                        });
                        ui.close_menu();
                    }
                    if ui.button("View 100%").clicked() {
                        actions.push(UiAction::Command {
                            window_label: self
                                .active_viewer_label
                                .clone()
                                .unwrap_or_else(|| LAUNCHER_LABEL.to_string()),
                            command_id: "image.zoom.view100".to_string(),
                            params: None,
                        });
                        ui.close_menu();
                    }
                    if ui.button("To Selection").clicked() {
                        actions.push(UiAction::Command {
                            window_label: self
                                .active_viewer_label
                                .clone()
                                .unwrap_or_else(|| LAUNCHER_LABEL.to_string()),
                            command_id: "image.zoom.to_selection".to_string(),
                            params: None,
                        });
                        ui.close_menu();
                    }
                    if ui.button("Scale to Fit").clicked() {
                        actions.push(UiAction::Command {
                            window_label: self
                                .active_viewer_label
                                .clone()
                                .unwrap_or_else(|| LAUNCHER_LABEL.to_string()),
                            command_id: "image.zoom.scale_to_fit".to_string(),
                            params: None,
                        });
                        ui.close_menu();
                    }
                }
                "launcher.tool.dropper" => {
                    for (label, command_id) in [
                        ("White/Black", "tool.dropper.palette.white_black"),
                        ("Black/White", "tool.dropper.palette.black_white"),
                        ("Red", "tool.dropper.palette.red"),
                        ("Green", "tool.dropper.palette.green"),
                        ("Blue", "tool.dropper.palette.blue"),
                        ("Yellow", "tool.dropper.palette.yellow"),
                        ("Cyan", "tool.dropper.palette.cyan"),
                        ("Magenta", "tool.dropper.palette.magenta"),
                        ("Foreground...", "tool.dropper.palette.foreground"),
                        ("Background...", "tool.dropper.palette.background"),
                        ("Colors...", "tool.dropper.palette.colors"),
                        ("Color Picker...", "tool.dropper.palette.color_picker"),
                    ] {
                        if ui.button(label).clicked() {
                            actions.push(UiAction::Command {
                                window_label: self
                                    .active_viewer_label
                                    .clone()
                                    .unwrap_or_else(|| LAUNCHER_LABEL.to_string()),
                                command_id: command_id.to_string(),
                                params: None,
                            });
                            ui.close_menu();
                        }
                    }
                }
                _ => {}
            });
        }
    }

    fn toolbar_texture(&mut self, ctx: &egui::Context, icon: ToolbarIcon) -> &egui::TextureHandle {
        self.toolbar_icons.entry(icon).or_insert_with(|| {
            let (name, bytes) = toolbar_icon_asset(icon);
            let decoded = load_from_memory(bytes)
                .map(|image| image.to_rgba8())
                .ok()
                .map(|rgba| {
                    egui::ColorImage::from_rgba_unmultiplied(
                        [rgba.width() as usize, rgba.height() as usize],
                        rgba.as_raw(),
                    )
                })
                .unwrap_or_else(|| egui::ColorImage::from_rgba_unmultiplied([1, 1], &[0, 0, 0, 0]));

            ctx.load_texture(
                format!("toolbar-icon-{name}"),
                decoded,
                egui::TextureOptions::LINEAR,
            )
        })
    }

    fn draw_viewer_toolbar(&self, ui: &mut egui::Ui, label: &str, actions: &mut Vec<UiAction>) {
        for item in viewer_toolbar_items() {
            if item.kind == ToolbarKind::Separator {
                ui.separator();
                continue;
            }

            let enabled = self.command_is_enabled(label, item.command_id, true);
            if ui
                .add_enabled(enabled, egui::Button::new(item.glyph).small())
                .on_hover_text(item.label)
                .clicked()
            {
                actions.push(UiAction::Command {
                    window_label: label.to_string(),
                    command_id: item.command_id.to_string(),
                    params: None,
                });
            }
        }
    }

    fn handle_shortcuts(
        &self,
        ctx: &egui::Context,
        window_label: &str,
        actions: &mut Vec<UiAction>,
    ) {
        let wants_keyboard = ctx.wants_keyboard_input();
        let input = ctx.input(|i| i.clone());

        if !wants_keyboard {
            if (input.modifiers.command || input.modifiers.ctrl) && input.key_pressed(egui::Key::O)
            {
                actions.push(UiAction::Command {
                    window_label: window_label.to_string(),
                    command_id: "file.open".to_string(),
                    params: None,
                });
            }

            if (input.modifiers.command || input.modifiers.ctrl) && input.key_pressed(egui::Key::W)
            {
                actions.push(UiAction::Command {
                    window_label: window_label.to_string(),
                    command_id: "file.close".to_string(),
                    params: None,
                });
            }

            if input.modifiers.alt && input.key_pressed(egui::Key::F4) {
                actions.push(UiAction::Command {
                    window_label: window_label.to_string(),
                    command_id: "file.quit".to_string(),
                    params: None,
                });
            }

            if input.key_pressed(egui::Key::Tab) {
                let command_id = if input.modifiers.shift {
                    "window.previous"
                } else {
                    "window.next"
                };
                actions.push(UiAction::Command {
                    window_label: window_label.to_string(),
                    command_id: command_id.to_string(),
                    params: None,
                });
            }

            if window_label.starts_with(VIEWER_PREFIX) {
                if input.key_pressed(egui::Key::ArrowRight) || input.key_pressed(egui::Key::Period)
                {
                    actions.push(UiAction::Command {
                        window_label: window_label.to_string(),
                        command_id: "viewer.slice.next".to_string(),
                        params: None,
                    });
                }
                if input.key_pressed(egui::Key::ArrowLeft) || input.key_pressed(egui::Key::Comma) {
                    actions.push(UiAction::Command {
                        window_label: window_label.to_string(),
                        command_id: "viewer.slice.previous".to_string(),
                        params: None,
                    });
                }
                if input.key_pressed(egui::Key::ArrowUp) {
                    actions.push(UiAction::Command {
                        window_label: window_label.to_string(),
                        command_id: "image.zoom.in".to_string(),
                        params: None,
                    });
                }
                if input.key_pressed(egui::Key::ArrowDown) {
                    actions.push(UiAction::Command {
                        window_label: window_label.to_string(),
                        command_id: "image.zoom.out".to_string(),
                        params: None,
                    });
                }
                if input.key_pressed(egui::Key::Escape) {
                    actions.push(UiAction::Command {
                        window_label: window_label.to_string(),
                        command_id: "viewer.roi.abort".to_string(),
                        params: None,
                    });
                }
                if input.key_pressed(egui::Key::Delete) || input.key_pressed(egui::Key::Backspace) {
                    actions.push(UiAction::Command {
                        window_label: window_label.to_string(),
                        command_id: "viewer.roi.delete".to_string(),
                        params: None,
                    });
                }
            }

            if input.modifiers.is_none() {
                if let Some((path, macro_name)) = self.installed_macro_shortcut(|shortcut| {
                    macro_shortcut_matches_function_key(shortcut, &input)
                }) {
                    actions.push(UiAction::RunInstalledMacro {
                        window_label: self.macro_shortcut_window_label(window_label),
                        path,
                        macro_name,
                    });
                }
            }
        }

        if wants_keyboard {
            return;
        }

        for event in &input.events {
            let egui::Event::Text(text) = event else {
                continue;
            };

            if let Some((path, macro_name)) = self
                .installed_macro_shortcut(|shortcut| macro_shortcut_matches_text(shortcut, text))
            {
                actions.push(UiAction::RunInstalledMacro {
                    window_label: self.macro_shortcut_window_label(window_label),
                    path,
                    macro_name,
                });
                continue;
            }

            if let Some(tool_command) = tool_shortcut_command(text) {
                actions.push(UiAction::Command {
                    window_label: window_label.to_string(),
                    command_id: tool_command.to_string(),
                    params: None,
                });
                continue;
            }

            if !window_label.starts_with(VIEWER_PREFIX) {
                continue;
            }

            match text.as_str() {
                "+" => actions.push(UiAction::Command {
                    window_label: window_label.to_string(),
                    command_id: "image.zoom.in".to_string(),
                    params: None,
                }),
                "-" => actions.push(UiAction::Command {
                    window_label: window_label.to_string(),
                    command_id: "image.zoom.out".to_string(),
                    params: None,
                }),
                "0" => actions.push(UiAction::Command {
                    window_label: window_label.to_string(),
                    command_id: "image.zoom.original".to_string(),
                    params: None,
                }),
                "5" => actions.push(UiAction::Command {
                    window_label: window_label.to_string(),
                    command_id: "image.zoom.view100".to_string(),
                    params: None,
                }),
                "M" | "m" => actions.push(UiAction::Command {
                    window_label: window_label.to_string(),
                    command_id: "analyze.measure".to_string(),
                    params: None,
                }),
                "<" | "," => actions.push(UiAction::Command {
                    window_label: window_label.to_string(),
                    command_id: "viewer.slice.previous".to_string(),
                    params: None,
                }),
                ">" | "." | ";" => actions.push(UiAction::Command {
                    window_label: window_label.to_string(),
                    command_id: "viewer.slice.next".to_string(),
                    params: None,
                }),
                _ => {}
            }
        }
    }

    fn macro_shortcut_window_label(&self, window_label: &str) -> String {
        if window_label == LAUNCHER_LABEL {
            self.active_viewer_label
                .clone()
                .unwrap_or_else(|| LAUNCHER_LABEL.to_string())
        } else {
            window_label.to_string()
        }
    }

    fn installed_macro_shortcut(
        &self,
        matches_shortcut: impl Fn(&str) -> bool,
    ) -> Option<(PathBuf, String)> {
        for path in self.list_installed_macro_files() {
            for block in installed_macro_blocks(&path) {
                let Some(shortcut) = block.shortcut.as_deref() else {
                    continue;
                };
                if matches_shortcut(shortcut) {
                    return Some((path, block.name));
                }
            }
        }
        None
    }

    fn installed_macro_menu_entries(&self) -> Vec<InstalledMacroMenuEntry> {
        let mut entries = Vec::new();
        for path in self.list_installed_macro_files() {
            for block in installed_macro_blocks(&path) {
                if let Some(entry) = installed_macro_menu_entry_from_block(&path, &block) {
                    entries.push(entry);
                }
            }
        }
        entries
    }

    fn draw_launcher(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        let shortcut_label = self
            .active_viewer_label
            .as_deref()
            .unwrap_or(LAUNCHER_LABEL);
        self.handle_shortcuts(ctx, shortcut_label, actions);
        self.refresh_launcher_status();

        egui::TopBottomPanel::top("launcher-header").show(ctx, |ui| {
            self.draw_menu_bar(ui, LAUNCHER_LABEL, actions);
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                self.draw_launcher_toolbar(ui, actions);
            });
        });

        egui::TopBottomPanel::bottom("launcher-status").show(ctx, |ui| {
            ui.label(&self.launcher_ui.status.text);
        });

        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        let dropped_paths = dropped
            .into_iter()
            .filter_map(|file| file.path)
            .collect::<Vec<_>>();
        if !dropped_paths.is_empty() {
            actions.push(UiAction::OpenPaths {
                paths: dropped_paths,
            });
        }

        self.draw_utility_windows(ctx, actions);
    }

    fn draw_utility_windows(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        self.draw_new_image_dialog(ctx, actions);
        self.draw_adjust_dialog(ctx, actions);
        self.draw_threshold_apply_dialog(ctx, actions);
        self.draw_threshold_set_dialog(ctx, actions);
        self.draw_apply_lut_dialog(ctx, actions);
        self.draw_set_display_range_dialog(ctx, actions);
        self.draw_set_window_level_dialog(ctx, actions);
        self.draw_resize_dialog(ctx, actions, false);
        self.draw_resize_dialog(ctx, actions, true);
        self.draw_stack_position_dialog(ctx, actions);
        self.draw_stack_label_dialog(ctx, actions);
        self.draw_zoom_set_dialog(ctx, actions);
        self.draw_color_dialog(ctx);
        self.draw_raw_import_dialog(ctx);
        self.draw_url_import_dialog(ctx, actions);
        self.draw_results_window(ctx);
        self.draw_measurement_settings_window(ctx);
        self.draw_binary_options_window(ctx);
        self.draw_overlay_settings_windows(ctx);
        self.draw_roi_manager_window(ctx, actions);
        self.draw_profile_plot_window(ctx);
        self.draw_help_windows(ctx);
        self.draw_command_finder_window(ctx, actions);
        self.draw_macro_recorder_window(ctx);
        self.draw_startup_macro_window(ctx);
        self.draw_macro_compatibility_window(ctx);
    }

    fn draw_macro_recorder_window(&mut self, ctx: &egui::Context) {
        if !self.macro_recorder.open {
            return;
        }

        let mut open = self.macro_recorder.open;
        let active_label = self
            .active_viewer_label
            .clone()
            .unwrap_or_else(|| LAUNCHER_LABEL.to_string());
        let mut run_requested = false;
        let mut save_requested = false;

        egui::Window::new("Recorder")
            .open(&mut open)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let label = if self.macro_recorder.recording {
                        "Pause"
                    } else {
                        "Record"
                    };
                    if ui.button(label).clicked() {
                        self.macro_recorder.recording = !self.macro_recorder.recording;
                    }
                    if ui.button("Clear").clicked() {
                        self.macro_recorder.text.clear();
                        self.macro_recorder.last_run_log.clear();
                    }
                    if ui.button("Run").clicked() {
                        run_requested = true;
                    }
                    if ui.button("Save").clicked() {
                        save_requested = true;
                    }
                });
                let status = if self.macro_recorder.recording {
                    "Recording"
                } else {
                    "Paused"
                };
                ui.label(status);
                ui.add(
                    egui::TextEdit::multiline(&mut self.macro_recorder.text)
                        .font(egui::TextStyle::Monospace)
                        .desired_rows(16)
                        .desired_width(f32::INFINITY),
                );
                if !self.macro_recorder.last_run_log.is_empty() {
                    ui.separator();
                    ui.label("Execution log:");
                    egui::ScrollArea::vertical()
                        .max_height(140.0)
                        .show(ui, |ui| {
                            ui.monospace(&self.macro_recorder.last_run_log);
                        });
                }
            });

        self.macro_recorder.open = open;

        if run_requested {
            let source = self.macro_recorder.text.clone();
            self.macro_recorder.last_run_log =
                self.run_simple_macro_source("Recorder", &source, &active_label);
        }
        if save_requested {
            self.macro_recorder.last_run_log = self.save_recorded_macro();
        }
    }

    fn open_startup_macro_window(&mut self) {
        self.startup_macro.text = match load_startup_macro() {
            Ok(contents) => contents,
            Err(error) => format!("// Failed to load startup macro: {error}\n"),
        };
        self.startup_macro.last_run_log.clear();
        self.startup_macro.open = true;
    }

    fn draw_startup_macro_window(&mut self, ctx: &egui::Context) {
        if !self.startup_macro.open {
            return;
        }

        let mut open = self.startup_macro.open;
        let path = startup_macro_path();
        let active_label = self
            .active_viewer_label
            .clone()
            .unwrap_or_else(|| LAUNCHER_LABEL.to_string());
        let mut reload_requested = false;
        let mut run_requested = false;
        let mut save_requested = false;

        egui::Window::new("Startup Macro")
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label(format!("Source: {}", path.display()));
                ui.horizontal(|ui| {
                    if ui.button("Reload").clicked() {
                        reload_requested = true;
                    }
                    if ui.button("Run").clicked() {
                        run_requested = true;
                    }
                    if ui.button("Save").clicked() {
                        save_requested = true;
                    }
                });
                ui.add(
                    egui::TextEdit::multiline(&mut self.startup_macro.text)
                        .font(egui::TextStyle::Monospace)
                        .desired_rows(16)
                        .desired_width(f32::INFINITY),
                );
                if !self.startup_macro.last_run_log.is_empty() {
                    ui.separator();
                    ui.label("Status:");
                    egui::ScrollArea::vertical()
                        .max_height(140.0)
                        .show(ui, |ui| {
                            ui.monospace(&self.startup_macro.last_run_log);
                        });
                }
            });

        self.startup_macro.open = open;

        if reload_requested {
            match load_startup_macro() {
                Ok(contents) => {
                    self.startup_macro.text = contents;
                    self.startup_macro.last_run_log =
                        format!("reloaded startup macro: {}", path.display());
                }
                Err(error) => {
                    self.startup_macro.last_run_log =
                        format!("startup macro reload failed: {error}");
                }
            }
        }
        if run_requested {
            let source = self.startup_macro.text.clone();
            self.startup_macro.last_run_log =
                self.run_simple_macro_source("Startup Macro", &source, &active_label);
        }
        if save_requested {
            self.startup_macro.last_run_log = match save_startup_macro(&self.startup_macro.text) {
                Ok(()) => format!("saved startup macro: {}", path.display()),
                Err(error) => format!("startup macro save failed: {error}"),
            };
        }
    }

    fn draw_macro_compatibility_window(&mut self, ctx: &egui::Context) {
        if !self.macro_compatibility_open {
            return;
        }
        let mut open = self.macro_compatibility_open;
        let command = self
            .macro_compatibility_command
            .as_deref()
            .unwrap_or("plugins/macros command")
            .to_string();
        let has_macro_path = self.macro_compatibility_path.is_some();
        let active_label = self
            .active_viewer_label
            .clone()
            .unwrap_or_else(|| LAUNCHER_LABEL.to_string());
        let installed_macros = self.list_installed_macro_files();
        let mut review_installed_macro: Option<PathBuf> = None;
        let mut run_installed_macro: Option<PathBuf> = None;
        let mut run_named_installed_macro: Option<(PathBuf, String)> = None;

        egui::Window::new("Macros and Plugins")
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label("ImageJ-style macro and plugin support is not fully implemented.");
                ui.label(format!("Requested command: {command}"));
                ui.separator();
                ui.label("Current compatibility:");
                ui.label("• plugins.macros.run executes simple run(\"Command...\") macro files");
                ui.label("• plugins.macros.record records UI commands into macro text");
                ui.label("• plugins.macros.install copies .ijm/.txt files into the macros folder");
                ui.label(
                    "• plugins.utilities.startup edits and runs RunAtStartup.ijm and StartupMacros AutoRun macros",
                );
                ui.label("Planned support:");
                ui.label("• Java/class/jar plugin execution");
                ui.label("• full ImageJ macro language support");
                if matches!(
                    command.as_str(),
                    "plugins.macros.run" | "plugins.macros.install"
                ) && has_macro_path
                {
                    if ui
                        .button("Execute selected macro as command list")
                        .clicked()
                    {
                        if let Some(path) = self.macro_compatibility_path.clone() {
                            let report = self.run_simple_macro_file(&path, &active_label);
                            self.macro_compatibility_run_log = report;
                        }
                    }
                }
                if let Some(path) = &self.macro_compatibility_path {
                    ui.separator();
                    ui.label(format!("Source path: {}", path.display()));
                    ui.label("Preview:");
                }
                ui.separator();
                ui.label(format!(
                    "Installed macros: {}",
                    installed_macros_dir().display()
                ));
                if installed_macros.is_empty() {
                    ui.label("No installed macros found.");
                } else {
                    egui::ScrollArea::vertical()
                        .max_height(140.0)
                        .show(ui, |ui| {
                            for path in &installed_macros {
                                let name = path
                                    .file_name()
                                    .and_then(|name| name.to_str())
                                    .unwrap_or("macro");
                                ui.horizontal(|ui| {
                                    ui.label(name);
                                    if ui.button("Review").clicked() {
                                        review_installed_macro = Some(path.clone());
                                    }
                                    if ui.button("Run").clicked() {
                                        run_installed_macro = Some(path.clone());
                                    }
                                });
                                for block in installed_macro_blocks(path) {
                                    ui.horizontal(|ui| {
                                        ui.add_space(16.0);
                                        let display_name = macro_display_name(&block.name);
                                        let shortcut = block
                                            .shortcut
                                            .as_deref()
                                            .map(|shortcut| format!(" [{shortcut}]"))
                                            .unwrap_or_default();
                                        ui.label(format!("macro {display_name}{shortcut}"));
                                        if ui.button("Run").clicked() {
                                            run_named_installed_macro =
                                                Some((path.clone(), block.name.clone()));
                                        }
                                    });
                                }
                            }
                        });
                }
                if !self.macro_compatibility_preview.is_empty() {
                    ui.separator();
                    ui.label("Details:");
                    egui::ScrollArea::vertical()
                        .max_height(160.0)
                        .show(ui, |ui| {
                            ui.monospace(&self.macro_compatibility_preview);
                        });
                }
                if !self.macro_compatibility_run_log.is_empty() {
                    ui.separator();
                    ui.label("Execution log:");
                    ui.label(&self.macro_compatibility_run_log);
                }
            });
        self.macro_compatibility_open = open;

        if let Some(path) = review_installed_macro {
            self.open_macro_compatibility_window_for_file("plugins.macros.install", &path);
        }
        if let Some(path) = run_installed_macro {
            self.open_macro_compatibility_window_for_file("plugins.macros.install", &path);
            self.macro_compatibility_run_log = self.run_simple_macro_file(&path, &active_label);
        }
        if let Some((path, name)) = run_named_installed_macro {
            self.open_macro_compatibility_window_for_file("plugins.macros.install", &path);
            self.macro_compatibility_run_log =
                self.run_named_macro_block_file(&path, &name, &active_label);
        }
    }

    fn open_macro_compatibility_window_for_file(&mut self, command_id: &str, path: &Path) {
        self.macro_compatibility_command = Some(command_id.to_string());
        self.macro_compatibility_path = Some(path.to_path_buf());
        self.macro_compatibility_open = true;
        self.macro_compatibility_preview = match fs::read_to_string(path) {
            Ok(contents) => {
                let snippet = contents.lines().take(12).collect::<Vec<_>>().join("\n");
                if contents.lines().count() > 12 {
                    format!("{snippet}\n… (truncated)")
                } else {
                    snippet
                }
            }
            Err(error) => format!("Failed to read macro source: {error}"),
        };
        self.macro_compatibility_run_log = String::new();
    }

    fn install_macro_file(&self, source: &Path) -> Result<PathBuf, String> {
        install_macro_file_to_dir(source, &installed_macros_dir())
    }

    fn list_installed_macro_files(&self) -> Vec<PathBuf> {
        list_installed_macro_files_in_dir(&installed_macros_dir())
    }

    fn is_known_command_id(&self, command_id: &str) -> bool {
        self.command_catalog
            .entries
            .iter()
            .any(|entry| entry.id == command_id)
            || command_registry::metadata(command_id).implemented
    }

    fn save_recorded_macro(&mut self) -> String {
        if self.macro_recorder.text.trim().is_empty() {
            return "No recorded macro commands to save".to_string();
        }

        let Some(path) = FileDialog::new()
            .add_filter("ImageJ Macro", &["ijm"])
            .set_file_name("Macro.ijm")
            .set_title("Save Recorded Macro")
            .save_file()
        else {
            return "macro save canceled".to_string();
        };

        match fs::write(&path, self.macro_recorder.text.as_bytes()) {
            Ok(()) => format!("macro saved: {}", path.display()),
            Err(error) => format!("macro save failed: {error}"),
        }
    }

    fn run_simple_macro_file(&mut self, path: &Path, window_label: &str) -> String {
        let contents = match fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(error) => {
                return format!("Failed to read macro file {}: {error}", path.display());
            }
        };

        self.run_simple_macro_source(&path.display().to_string(), &contents, window_label)
    }

    fn run_named_macro_block_file(
        &mut self,
        path: &Path,
        name: &str,
        window_label: &str,
    ) -> String {
        let contents = match fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(error) => {
                return format!("Failed to read macro file {}: {error}", path.display());
            }
        };
        let Some(block) = macro_source_named_blocks(&contents)
            .into_iter()
            .find(|block| block.name == name)
        else {
            return format!("macro `{name}` not found in {}", path.display());
        };
        let named_blocks = macro_named_block_statement_map(&contents);
        self.run_simple_macro_lines(
            &format!("{}:{}", path.display(), block.name),
            block.statements,
            window_label,
            &named_blocks,
        )
    }

    fn run_simple_macro_source(
        &mut self,
        source_name: &str,
        contents: &str,
        window_label: &str,
    ) -> String {
        let named_blocks = macro_named_block_statement_map(contents);
        self.run_simple_macro_lines(
            source_name,
            macro_source_executable_lines(contents),
            window_label,
            &named_blocks,
        )
    }

    fn run_simple_macro_lines(
        &mut self,
        source_name: &str,
        executable_lines: Vec<(usize, String)>,
        window_label: &str,
        named_blocks: &HashMap<String, Vec<(usize, String)>>,
    ) -> String {
        let report = self.run_simple_macro_lines_inner(
            source_name,
            executable_lines,
            window_label,
            named_blocks,
            0,
        );
        if report.lines.is_empty() {
            return format!("No executable commands found in {source_name}");
        }

        format!(
            "Executed: {}, blocked: {}, unimplemented: {}, unknown: {}\n{}",
            report.executed,
            report.blocked,
            report.unimplemented,
            report.unknown,
            report.lines.join("\n")
        )
    }

    fn run_simple_macro_lines_inner(
        &mut self,
        source_name: &str,
        executable_lines: Vec<(usize, String)>,
        window_label: &str,
        named_blocks: &HashMap<String, Vec<(usize, String)>>,
        depth: usize,
    ) -> MacroRunReport {
        const MAX_NAMED_MACRO_DEPTH: usize = 8;
        let mut report = MacroRunReport::default();
        for (line_number, raw_line) in executable_lines {
            let invocation = match parse_macro_command_line(&raw_line, &self.command_catalog) {
                Ok(Some(invocation)) => invocation,
                Ok(None) => continue,
                Err(error) => {
                    report.blocked += 1;
                    report
                        .lines
                        .push(format!("{line_number}: macro parse -> blocked ({error})"));
                    continue;
                }
            };
            let command_id = invocation.command_id;
            if !self.is_known_command_id(&command_id) {
                if let Some(statements) = named_blocks.get(&command_id) {
                    if depth >= MAX_NAMED_MACRO_DEPTH {
                        report.blocked += 1;
                        report.lines.push(format!(
                            "{line_number}: {command_id} -> blocked (macro recursion limit)"
                        ));
                        continue;
                    }
                    report
                        .lines
                        .push(format!("{line_number}: {command_id} -> macro"));
                    let nested = self.run_simple_macro_lines_inner(
                        &format!("{source_name}:{command_id}"),
                        statements.clone(),
                        window_label,
                        named_blocks,
                        depth + 1,
                    );
                    report.executed += nested.executed;
                    report.blocked += nested.blocked;
                    report.unknown += nested.unknown;
                    report.unimplemented += nested.unimplemented;
                    report.lines.extend(nested.lines);
                    continue;
                }

                report.unknown += 1;
                report
                    .lines
                    .push(format!("{line_number}: {command_id} -> unknown command"));
                continue;
            }

            let result = self.dispatch_command(window_label, &command_id, invocation.params);
            match result.status {
                command_registry::CommandExecuteStatus::Ok => {
                    report.executed += 1;
                    report
                        .lines
                        .push(format!("{line_number}: {command_id} -> {}", result.message));
                }
                command_registry::CommandExecuteStatus::Unimplemented => {
                    report.unimplemented += 1;
                    report.lines.push(format!(
                        "{line_number}: {command_id} -> unimplemented ({})",
                        result.message
                    ));
                }
                command_registry::CommandExecuteStatus::Blocked => {
                    report.blocked += 1;
                    report.lines.push(format!(
                        "{line_number}: {command_id} -> blocked ({})",
                        result.message
                    ));
                }
            }
        }

        report
    }

    fn draw_new_image_dialog(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        if !self.new_image_dialog.open {
            return;
        }
        let mut open = self.new_image_dialog.open;
        egui::Window::new("New Image")
            .open(&mut open)
            .show(ctx, |ui| {
                ui.add(egui::DragValue::new(&mut self.new_image_dialog.width).prefix("Width "));
                ui.add(egui::DragValue::new(&mut self.new_image_dialog.height).prefix("Height "));
                ui.add(egui::DragValue::new(&mut self.new_image_dialog.slices).prefix("Slices "));
                ui.add(
                    egui::DragValue::new(&mut self.new_image_dialog.channels).prefix("Channels "),
                );
                ui.add(egui::DragValue::new(&mut self.new_image_dialog.frames).prefix("Frames "));
                ui.add(egui::DragValue::new(&mut self.new_image_dialog.fill).prefix("Fill "));
                pixel_type_selector(ui, &mut self.new_image_dialog.pixel_type, "Pixel Type");
                if ui.button("Create").clicked() {
                    actions.push(UiAction::Command {
                        window_label: LAUNCHER_LABEL.to_string(),
                        command_id: "file.new".to_string(),
                        params: Some(json!({
                            "width": self.new_image_dialog.width,
                            "height": self.new_image_dialog.height,
                            "slices": self.new_image_dialog.slices,
                            "channels": self.new_image_dialog.channels,
                            "frames": self.new_image_dialog.frames,
                            "fill": self.new_image_dialog.fill,
                            "pixelType": pixel_type_id(self.new_image_dialog.pixel_type),
                        })),
                    });
                    self.new_image_dialog.open = false;
                }
            });
        self.new_image_dialog.open = open;
    }

    fn draw_adjust_dialog(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        if !self.adjust_dialog.open {
            return;
        }

        let kind = self.adjust_dialog.kind;
        let mut open = self.adjust_dialog.open;
        let close = false;
        let title = adjust_dialog_window_title(&self.adjust_dialog);
        egui::Window::new(title)
            .open(&mut open)
            .show(ctx, |ui| match kind {
                AdjustDialogKind::BrightnessContrast => {
                    draw_adjust_histogram(
                        ui,
                        self.adjust_dialog.histogram.as_ref(),
                        Some((self.adjust_dialog.min, self.adjust_dialog.max)),
                        None,
                        self.adjust_dialog.log_histogram,
                    );
                    ui.checkbox(&mut self.adjust_dialog.log_histogram, "Log scale");
                    let (slider_min, slider_max) = adjust_slider_bounds(&self.adjust_dialog);
                    let mut display_range_changed = false;
                    ui.horizontal(|ui| {
                        ui.label("Minimum");
                        let slider_changed = ui
                            .add(
                                egui::Slider::new(
                                    &mut self.adjust_dialog.min,
                                    slider_min..=slider_max,
                                )
                                .show_value(false),
                            )
                            .changed();
                        let value_changed = ui
                            .add(egui::DragValue::new(&mut self.adjust_dialog.min).speed(0.1))
                            .changed();
                        if slider_changed || value_changed {
                            clamp_adjust_min_max(&mut self.adjust_dialog);
                            sync_brightness_contrast_from_min_max(&mut self.adjust_dialog);
                            display_range_changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Maximum");
                        let slider_changed = ui
                            .add(
                                egui::Slider::new(
                                    &mut self.adjust_dialog.max,
                                    slider_min..=slider_max,
                                )
                                .show_value(false),
                            )
                            .changed();
                        let value_changed = ui
                            .add(egui::DragValue::new(&mut self.adjust_dialog.max).speed(0.1))
                            .changed();
                        if slider_changed || value_changed {
                            clamp_adjust_min_max(&mut self.adjust_dialog);
                            sync_brightness_contrast_from_min_max(&mut self.adjust_dialog);
                            display_range_changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Brightness");
                        if ui
                            .add(
                                egui::Slider::new(&mut self.adjust_dialog.brightness, 0.0..=1.0)
                                    .show_value(false),
                            )
                            .changed()
                        {
                            adjust_min_max_from_brightness(&mut self.adjust_dialog);
                            clamp_adjust_min_max(&mut self.adjust_dialog);
                            sync_contrast_from_min_max(&mut self.adjust_dialog);
                            display_range_changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Contrast");
                        if ui
                            .add(
                                egui::Slider::new(&mut self.adjust_dialog.contrast, 0.0..=1.0)
                                    .show_value(false),
                            )
                            .changed()
                        {
                            adjust_min_max_from_contrast(&mut self.adjust_dialog);
                            clamp_adjust_min_max(&mut self.adjust_dialog);
                            sync_brightness_from_min_max(&mut self.adjust_dialog);
                            display_range_changed = true;
                        }
                    });
                    if display_range_changed {
                        push_adjust_display_range_action(
                            actions,
                            &self.adjust_dialog,
                            "image.adjust.brightness",
                            "min",
                            "max",
                        );
                    }
                    egui::Grid::new("brightness_contrast_buttons")
                        .num_columns(2)
                        .spacing(egui::vec2(4.0, 4.0))
                        .show(ui, |ui| {
                            if ui.button("Auto").clicked() {
                                let (min, max) =
                                    auto_contrast_range_for_dialog(&mut self.adjust_dialog)
                                        .unwrap_or((
                                            self.adjust_dialog.default_min,
                                            self.adjust_dialog.default_max,
                                        ));
                                self.adjust_dialog.min = min;
                                self.adjust_dialog.max = max;
                                sync_brightness_contrast_from_min_max(&mut self.adjust_dialog);
                                actions.push(UiAction::Command {
                                    window_label: self.adjust_dialog.window_label.clone(),
                                    command_id: "image.adjust.brightness".to_string(),
                                    params: Some(json!({
                                        "min": self.adjust_dialog.min,
                                        "max": self.adjust_dialog.max,
                                    })),
                                });
                            }
                            if ui.button("Reset").clicked() {
                                self.adjust_dialog.contrast_auto_threshold = 0;
                                self.adjust_dialog.min = self.adjust_dialog.default_min;
                                self.adjust_dialog.max = self.adjust_dialog.default_max;
                                sync_brightness_contrast_from_min_max(&mut self.adjust_dialog);
                                actions.push(UiAction::Command {
                                    window_label: self.adjust_dialog.window_label.clone(),
                                    command_id: "image.adjust.brightness".to_string(),
                                    params: Some(json!({ "reset": true })),
                                });
                            }
                            ui.end_row();
                            if ui.button("Set").clicked() {
                                self.open_set_display_range_dialog_from_adjust(
                                    "image.adjust.brightness",
                                    "min",
                                    "max",
                                );
                            }
                            if ui.button("Apply").clicked() {
                                let window_label = self.adjust_dialog.window_label.clone();
                                let command_id = "image.adjust.window_level".to_string();
                                let params = json!({
                                    "low": self.adjust_dialog.min,
                                    "high": self.adjust_dialog.max,
                                    "apply": true,
                                });
                                if self.apply_lut_needs_confirmation(&window_label) {
                                    self.apply_lut_dialog = self.apply_lut_dialog_state(
                                        window_label,
                                        command_id,
                                        params,
                                    );
                                } else {
                                    actions.push(UiAction::Command {
                                        window_label,
                                        command_id,
                                        params: Some(params),
                                    });
                                }
                            }
                        });
                }
                AdjustDialogKind::WindowLevel => {
                    draw_adjust_histogram(
                        ui,
                        self.adjust_dialog.histogram.as_ref(),
                        Some((self.adjust_dialog.min, self.adjust_dialog.max)),
                        None,
                        self.adjust_dialog.log_histogram,
                    );
                    ui.checkbox(&mut self.adjust_dialog.log_histogram, "Log scale");
                    let level = self.adjust_dialog.min
                        + (self.adjust_dialog.max - self.adjust_dialog.min) * 0.5;
                    let window = self.adjust_dialog.max - self.adjust_dialog.min;
                    let mut display_range_changed = false;
                    ui.horizontal(|ui| {
                        ui.label(format!("Level: {level:.4}"));
                        if ui
                            .add(
                                egui::Slider::new(&mut self.adjust_dialog.brightness, 0.0..=1.0)
                                    .show_value(false),
                            )
                            .changed()
                        {
                            adjust_min_max_from_brightness(&mut self.adjust_dialog);
                            clamp_adjust_min_max(&mut self.adjust_dialog);
                            sync_contrast_from_min_max(&mut self.adjust_dialog);
                            display_range_changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(format!("Window: {window:.4}"));
                        if ui
                            .add(
                                egui::Slider::new(&mut self.adjust_dialog.contrast, 0.0..=1.0)
                                    .show_value(false),
                            )
                            .changed()
                        {
                            adjust_min_max_from_contrast(&mut self.adjust_dialog);
                            clamp_adjust_min_max(&mut self.adjust_dialog);
                            sync_brightness_from_min_max(&mut self.adjust_dialog);
                            display_range_changed = true;
                        }
                    });
                    if display_range_changed {
                        push_adjust_display_range_action(
                            actions,
                            &self.adjust_dialog,
                            "image.adjust.window_level",
                            "low",
                            "high",
                        );
                    }
                    egui::Grid::new("window_level_buttons")
                        .num_columns(2)
                        .spacing(egui::vec2(4.0, 4.0))
                        .show(ui, |ui| {
                            if ui.button("Auto").clicked() {
                                let (min, max) =
                                    auto_contrast_range_for_dialog(&mut self.adjust_dialog)
                                        .unwrap_or((
                                            self.adjust_dialog.default_min,
                                            self.adjust_dialog.default_max,
                                        ));
                                self.adjust_dialog.min = min;
                                self.adjust_dialog.max = max;
                                sync_brightness_contrast_from_min_max(&mut self.adjust_dialog);
                                actions.push(UiAction::Command {
                                    window_label: self.adjust_dialog.window_label.clone(),
                                    command_id: "image.adjust.window_level".to_string(),
                                    params: Some(json!({
                                        "low": self.adjust_dialog.min,
                                        "high": self.adjust_dialog.max,
                                    })),
                                });
                            }
                            if ui.button("Reset").clicked() {
                                self.adjust_dialog.contrast_auto_threshold = 0;
                                self.adjust_dialog.min = self.adjust_dialog.default_min;
                                self.adjust_dialog.max = self.adjust_dialog.default_max;
                                sync_brightness_contrast_from_min_max(&mut self.adjust_dialog);
                                actions.push(UiAction::Command {
                                    window_label: self.adjust_dialog.window_label.clone(),
                                    command_id: "image.adjust.window_level".to_string(),
                                    params: Some(json!({
                                        "low": self.adjust_dialog.min,
                                        "high": self.adjust_dialog.max,
                                    })),
                                });
                            }
                            ui.end_row();
                            if ui.button("Set").clicked() {
                                self.open_set_window_level_dialog_from_adjust();
                            }
                            if ui.button("Apply").clicked() {
                                let window_label = self.adjust_dialog.window_label.clone();
                                let command_id = "image.adjust.brightness".to_string();
                                let params = json!({
                                    "min": self.adjust_dialog.min,
                                    "max": self.adjust_dialog.max,
                                    "apply": true,
                                });
                                if self.apply_lut_needs_confirmation(&window_label) {
                                    self.apply_lut_dialog = self.apply_lut_dialog_state(
                                        window_label,
                                        command_id,
                                        params,
                                    );
                                } else {
                                    actions.push(UiAction::Command {
                                        window_label,
                                        command_id,
                                        params: Some(params),
                                    });
                                }
                            }
                        });
                }
                AdjustDialogKind::ColorBalance => {
                    draw_adjust_histogram(
                        ui,
                        self.adjust_dialog.histogram.as_ref(),
                        Some((self.adjust_dialog.min, self.adjust_dialog.max)),
                        None,
                        self.adjust_dialog.log_histogram,
                    );
                    ui.checkbox(&mut self.adjust_dialog.log_histogram, "Log scale");
                    let mut display_range_changed = false;
                    ui.horizontal(|ui| {
                        ui.label("Brightness");
                        if ui
                            .add(
                                egui::Slider::new(&mut self.adjust_dialog.brightness, 0.0..=1.0)
                                    .show_value(false),
                            )
                            .changed()
                        {
                            adjust_min_max_from_brightness(&mut self.adjust_dialog);
                            clamp_adjust_min_max(&mut self.adjust_dialog);
                            display_range_changed = true;
                        }
                    });
                    egui::ComboBox::from_id_salt("color_balance_channel_choice")
                        .selected_text(&self.adjust_dialog.color_balance_channel)
                        .show_ui(ui, |ui| {
                            for channel in &self.adjust_dialog.color_balance_channel_labels {
                                ui.selectable_value(
                                    &mut self.adjust_dialog.color_balance_channel,
                                    channel.clone(),
                                    channel,
                                );
                            }
                        });
                    if display_range_changed {
                        actions.push(UiAction::Command {
                            window_label: self.adjust_dialog.window_label.clone(),
                            command_id: "image.adjust.color_balance".to_string(),
                            params: Some(json!(color_balance_params(&self.adjust_dialog))),
                        });
                    }
                    egui::Grid::new("color_balance_buttons")
                        .num_columns(2)
                        .spacing(egui::vec2(4.0, 4.0))
                        .show(ui, |ui| {
                            if ui.button("Auto").clicked() {
                                let (min, max) =
                                    auto_contrast_range_for_dialog(&mut self.adjust_dialog)
                                        .unwrap_or((
                                            self.adjust_dialog.default_min,
                                            self.adjust_dialog.default_max,
                                        ));
                                self.adjust_dialog.min = min;
                                self.adjust_dialog.max = max;
                                sync_brightness_contrast_from_min_max(&mut self.adjust_dialog);
                                actions.push(UiAction::Command {
                                    window_label: self.adjust_dialog.window_label.clone(),
                                    command_id: "image.adjust.color_balance".to_string(),
                                    params: Some(json!(color_balance_params(&self.adjust_dialog))),
                                });
                            }
                            if ui.button("Reset").clicked() {
                                self.adjust_dialog.contrast_auto_threshold = 0;
                                self.adjust_dialog.min = self.adjust_dialog.default_min;
                                self.adjust_dialog.max = self.adjust_dialog.default_max;
                                sync_brightness_contrast_from_min_max(&mut self.adjust_dialog);
                                actions.push(UiAction::Command {
                                    window_label: self.adjust_dialog.window_label.clone(),
                                    command_id: "image.adjust.color_balance".to_string(),
                                    params: Some(json!(color_balance_params(&self.adjust_dialog))),
                                });
                            }
                            ui.end_row();
                            if ui.button("Set").clicked() {
                                self.open_set_display_range_dialog_from_adjust(
                                    "image.adjust.color_balance",
                                    "min",
                                    "max",
                                );
                            }
                            if ui.button("Apply").clicked() {
                                let mut params = color_balance_params(&self.adjust_dialog);
                                if let Some(map) = params.as_object_mut() {
                                    map.insert("apply".to_string(), json!(true));
                                }
                                let window_label = self.adjust_dialog.window_label.clone();
                                let command_id = "image.adjust.color_balance".to_string();
                                if self.apply_lut_needs_confirmation(&window_label) {
                                    self.apply_lut_dialog = self.apply_lut_dialog_state(
                                        window_label,
                                        command_id,
                                        params,
                                    );
                                } else {
                                    actions.push(UiAction::Command {
                                        window_label,
                                        command_id,
                                        params: Some(params),
                                    });
                                }
                            }
                        });
                }
                AdjustDialogKind::Threshold => {
                    draw_adjust_histogram(
                        ui,
                        self.adjust_dialog.histogram.as_ref(),
                        Some((self.adjust_dialog.min, self.adjust_dialog.max)),
                        None,
                        false,
                    );
                    let in_range_count = self
                        .adjust_dialog
                        .histogram
                        .as_ref()
                        .map(|histogram| {
                            histogram
                                .counts
                                .iter()
                                .enumerate()
                                .filter_map(|(index, count)| {
                                    let denominator =
                                        histogram.counts.len().saturating_sub(1).max(1) as f32;
                                    let bin_value = histogram.min
                                        + (index as f32 / denominator)
                                            * (histogram.max - histogram.min);
                                    (bin_value >= self.adjust_dialog.min
                                        && bin_value <= self.adjust_dialog.max)
                                        .then_some(*count)
                                })
                                .sum::<u64>()
                        })
                        .unwrap_or(0);
                    let total_count = self
                        .adjust_dialog
                        .histogram
                        .as_ref()
                        .map(|histogram| histogram.pixel_count as u64)
                        .unwrap_or(0);
                    if total_count > 0 {
                        ui.label(format!(
                            "{:.1}% ({}/{})",
                            (in_range_count as f32 / total_count as f32) * 100.0,
                            in_range_count,
                            total_count
                        ));
                    }
                    let (slider_min, slider_max) = adjust_slider_bounds(&self.adjust_dialog);
                    let mut threshold_range_changed = false;
                    egui::Grid::new("threshold_adjust_grid")
                        .num_columns(2)
                        .spacing(egui::vec2(8.0, 4.0))
                        .show(ui, |ui| {
                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.adjust_dialog.min,
                                        slider_min..=slider_max,
                                    )
                                    .show_value(false),
                                )
                                .changed()
                            {
                                threshold_range_changed = true;
                            }
                            if ui
                                .add(egui::DragValue::new(&mut self.adjust_dialog.min).speed(0.1))
                                .changed()
                            {
                                threshold_range_changed = true;
                            }
                            ui.end_row();
                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.adjust_dialog.max,
                                        slider_min..=slider_max,
                                    )
                                    .show_value(false),
                                )
                                .changed()
                            {
                                threshold_range_changed = true;
                            }
                            if ui
                                .add(egui::DragValue::new(&mut self.adjust_dialog.max).speed(0.1))
                                .changed()
                            {
                                threshold_range_changed = true;
                            }
                            ui.end_row();
                        });
                    clamp_adjust_min_max(&mut self.adjust_dialog);
                    if threshold_range_changed {
                        actions.push(UiAction::Command {
                            window_label: self.adjust_dialog.window_label.clone(),
                            command_id: "image.adjust.threshold".to_string(),
                            params: Some(json!({
                                "min": self.adjust_dialog.min,
                                "max": self.adjust_dialog.max,
                                "mode": self.adjust_dialog.threshold_mode,
                                "background": threshold_background_param(
                                    self.adjust_dialog.dark_background
                                ),
                            })),
                        });
                    }
                    let method_changed = egui::ComboBox::from_id_salt("threshold_method_choice")
                        .selected_text(&self.adjust_dialog.threshold_method)
                        .show_ui(ui, |ui| {
                            for method in threshold_method_labels() {
                                ui.selectable_value(
                                    &mut self.adjust_dialog.threshold_method,
                                    (*method).to_string(),
                                    *method,
                                );
                            }
                        })
                        .response
                        .changed();
                    let mode_changed = egui::ComboBox::from_id_salt("threshold_mode_choice")
                        .selected_text(&self.adjust_dialog.threshold_mode)
                        .show_ui(ui, |ui| {
                            for mode in ["Red", "B&W", "Over/Under"] {
                                ui.selectable_value(
                                    &mut self.adjust_dialog.threshold_mode,
                                    mode.to_string(),
                                    mode,
                                );
                            }
                        })
                        .response
                        .changed();
                    if mode_changed {
                        actions.push(UiAction::Command {
                            window_label: self.adjust_dialog.window_label.clone(),
                            command_id: "image.adjust.threshold".to_string(),
                            params: Some(json!({
                                "min": self.adjust_dialog.min,
                                "max": self.adjust_dialog.max,
                                "mode": self.adjust_dialog.threshold_mode,
                                "background": threshold_background_param(
                                    self.adjust_dialog.dark_background
                                ),
                            })),
                        });
                    }
                    let mut histogram_options_changed = false;
                    let mut dark_background_changed = false;
                    egui::Grid::new("threshold_checkbox_grid")
                        .num_columns(2)
                        .show(ui, |ui| {
                            dark_background_changed |= ui
                                .checkbox(
                                    &mut self.adjust_dialog.dark_background,
                                    "Dark background",
                                )
                                .changed();
                            histogram_options_changed |= ui
                                .checkbox(
                                    &mut self.adjust_dialog.threshold_stack_histogram,
                                    "Stack histogram",
                                )
                                .changed();
                            ui.end_row();
                            ui.checkbox(
                                &mut self.adjust_dialog.threshold_no_reset,
                                "Don't reset range",
                            );
                            ui.checkbox(&mut self.adjust_dialog.threshold_raw_values, "Raw values");
                            ui.end_row();
                            let mut sixteen_bit_histogram =
                                self.adjust_dialog.threshold_sixteen_bit_histogram;
                            if ui
                                .checkbox(&mut sixteen_bit_histogram, "16-bit histogram")
                                .changed()
                            {
                                set_threshold_sixteen_bit_histogram(
                                    &mut self.adjust_dialog,
                                    sixteen_bit_histogram,
                                );
                                histogram_options_changed = true;
                            }
                            ui.end_row();
                        });
                    if histogram_options_changed {
                        let bins = threshold_histogram_bins(&self.adjust_dialog);
                        if let Ok(histogram) = self.adjust_histogram_for_viewer(
                            &self.adjust_dialog.window_label,
                            bins,
                            self.adjust_dialog.threshold_stack_histogram,
                        ) {
                            self.adjust_dialog.default_min = histogram.min;
                            self.adjust_dialog.default_max = histogram.max;
                            self.adjust_dialog.histogram = Some(histogram);
                            clamp_adjust_min_max(&mut self.adjust_dialog);
                        }
                    }
                    if method_changed || dark_background_changed {
                        let method = threshold_method_param(&self.adjust_dialog.threshold_method);
                        actions.push(UiAction::Command {
                            window_label: self.adjust_dialog.window_label.clone(),
                            command_id: "image.adjust.threshold".to_string(),
                            params: Some(json!({
                                "method": method,
                                "mode": self.adjust_dialog.threshold_mode,
                                "stack": self.adjust_dialog.threshold_stack_histogram,
                                "sixteen_bit": self
                                    .adjust_dialog
                                    .threshold_sixteen_bit_histogram,
                                "background": threshold_background_param(
                                    self.adjust_dialog.dark_background
                                ),
                            })),
                        });
                    }
                    ui.horizontal(|ui| {
                        if ui.button("Auto").clicked() {
                            let method =
                                threshold_method_param(&self.adjust_dialog.threshold_method);
                            actions.push(UiAction::Command {
                                window_label: self.adjust_dialog.window_label.clone(),
                                command_id: "image.adjust.threshold".to_string(),
                                params: Some(json!({
                                    "method": method,
                                    "mode": self.adjust_dialog.threshold_mode,
                                    "stack": self.adjust_dialog.threshold_stack_histogram,
                                    "sixteen_bit": self
                                        .adjust_dialog
                                        .threshold_sixteen_bit_histogram,
                                    "background": threshold_background_param(
                                        self.adjust_dialog.dark_background
                                    ),
                                })),
                            });
                        }
                        if ui.button("Apply").clicked() {
                            if self.threshold_apply_needs_float_prompt(
                                &self.adjust_dialog.window_label,
                            ) {
                                self.threshold_apply_dialog = ThresholdApplyDialogState {
                                    open: true,
                                    window_label: self.adjust_dialog.window_label.clone(),
                                    min: self.adjust_dialog.min,
                                    max: self.adjust_dialog.max,
                                    dark_background: self.adjust_dialog.dark_background,
                                };
                            } else {
                                actions.push(UiAction::Command {
                                    window_label: self.adjust_dialog.window_label.clone(),
                                    command_id: "image.adjust.threshold".to_string(),
                                    params: Some(json!({
                                        "apply": true,
                                        "min": self.adjust_dialog.min,
                                        "max": self.adjust_dialog.max,
                                        "background": threshold_background_param(
                                            self.adjust_dialog.dark_background
                                        ),
                                    })),
                                });
                            }
                        }
                        if ui.button("Reset").clicked() {
                            self.adjust_dialog.min = self.adjust_dialog.default_min;
                            self.adjust_dialog.max = self.adjust_dialog.default_max;
                            actions.push(UiAction::Command {
                                window_label: self.adjust_dialog.window_label.clone(),
                                command_id: "image.adjust.threshold".to_string(),
                                params: Some(json!({
                                    "reset": true,
                                    "no_reset": self.adjust_dialog.threshold_no_reset,
                                })),
                            });
                        }
                        if ui.button("Set").clicked() {
                            self.open_threshold_set_dialog_from_adjust();
                        }
                    });
                }
                AdjustDialogKind::ColorThreshold => {
                    let band_labels = match self.adjust_dialog.color_threshold_space.as_str() {
                        "RGB" => ["Red", "Green", "Blue"],
                        "Lab" => ["L*", "a*", "b*"],
                        "YUV" => ["Y", "U", "V"],
                        _ => ["Hue", "Saturation", "Brightness"],
                    };
                    draw_color_threshold_band(
                        ui,
                        "color_threshold_hue",
                        band_labels[0],
                        &mut self.adjust_dialog.hue_min,
                        &mut self.adjust_dialog.hue_max,
                        &mut self.adjust_dialog.hue_pass,
                        self.adjust_dialog.histogram.as_ref(),
                    );
                    draw_color_threshold_band(
                        ui,
                        "color_threshold_saturation",
                        band_labels[1],
                        &mut self.adjust_dialog.saturation_min,
                        &mut self.adjust_dialog.saturation_max,
                        &mut self.adjust_dialog.saturation_pass,
                        self.adjust_dialog.histogram.as_ref(),
                    );
                    draw_color_threshold_band(
                        ui,
                        "color_threshold_brightness",
                        band_labels[2],
                        &mut self.adjust_dialog.brightness_min,
                        &mut self.adjust_dialog.brightness_max,
                        &mut self.adjust_dialog.brightness_pass,
                        self.adjust_dialog.histogram.as_ref(),
                    );
                    let [method_label, mode_label, space_label] = color_threshold_choice_labels();
                    let method_changed = egui::ComboBox::from_label(method_label)
                        .selected_text(&self.adjust_dialog.color_threshold_method)
                        .show_ui(ui, |ui| {
                            for method in threshold_method_labels() {
                                ui.selectable_value(
                                    &mut self.adjust_dialog.color_threshold_method,
                                    (*method).to_string(),
                                    *method,
                                );
                            }
                        })
                        .response
                        .changed();
                    let mode_changed = egui::ComboBox::from_label(mode_label)
                        .selected_text(&self.adjust_dialog.color_threshold_mode)
                        .show_ui(ui, |ui| {
                            for mode in ["Red", "White", "Black", "B&W"] {
                                ui.selectable_value(
                                    &mut self.adjust_dialog.color_threshold_mode,
                                    mode.to_string(),
                                    mode,
                                );
                            }
                        })
                        .response
                        .changed();
                    let color_space_changed = egui::ComboBox::from_label(space_label)
                        .selected_text(&self.adjust_dialog.color_threshold_space)
                        .show_ui(ui, |ui| {
                            for space in ["HSB", "RGB", "Lab", "YUV"] {
                                ui.selectable_value(
                                    &mut self.adjust_dialog.color_threshold_space,
                                    space.to_string(),
                                    space,
                                );
                            }
                        })
                        .response
                        .changed();
                    let dark_background_changed = ui
                        .checkbox(&mut self.adjust_dialog.dark_background, "Dark background")
                        .changed();
                    if color_space_changed {
                        reset_color_threshold_bands_for_space(&mut self.adjust_dialog);
                    }
                    if color_space_changed || method_changed || dark_background_changed {
                        let mut params = color_threshold_params(&self.adjust_dialog);
                        if let Value::Object(map) = &mut params {
                            map.insert("action".to_string(), json!("auto"));
                        }
                        actions.push(UiAction::Command {
                            window_label: self.adjust_dialog.window_label.clone(),
                            command_id: "image.adjust.color_threshold".to_string(),
                            params: Some(params),
                        });
                    } else if mode_changed {
                        let mut params = color_threshold_params(&self.adjust_dialog);
                        if let Value::Object(map) = &mut params {
                            map.insert("action".to_string(), json!("filtered"));
                        }
                        actions.push(UiAction::Command {
                            window_label: self.adjust_dialog.window_label.clone(),
                            command_id: "image.adjust.color_threshold".to_string(),
                            params: Some(params),
                        });
                    }
                    egui::Grid::new("color_threshold_buttons")
                        .num_columns(4)
                        .spacing(egui::vec2(6.0, 4.0))
                        .show(ui, |ui| {
                            if ui.button("Original").clicked() {
                                actions.push(UiAction::Command {
                                    window_label: self.adjust_dialog.window_label.clone(),
                                    command_id: "image.adjust.color_threshold".to_string(),
                                    params: Some(json!({ "action": "original" })),
                                });
                            }
                            if ui.button("Filtered").clicked() {
                                let mut params = color_threshold_params(&self.adjust_dialog);
                                if let Value::Object(map) = &mut params {
                                    map.insert("action".to_string(), json!("filtered"));
                                }
                                actions.push(UiAction::Command {
                                    window_label: self.adjust_dialog.window_label.clone(),
                                    command_id: "image.adjust.color_threshold".to_string(),
                                    params: Some(params),
                                });
                            }
                            if ui.button("Select").clicked() {
                                let mut params = color_threshold_params(&self.adjust_dialog);
                                if let Value::Object(map) = &mut params {
                                    map.insert("action".to_string(), json!("select"));
                                }
                                actions.push(UiAction::Command {
                                    window_label: self.adjust_dialog.window_label.clone(),
                                    command_id: "image.adjust.color_threshold".to_string(),
                                    params: Some(params),
                                });
                            }
                            if ui.button("Sample").clicked() {
                                let mut params = color_threshold_params(&self.adjust_dialog);
                                if let Value::Object(map) = &mut params {
                                    map.insert("action".to_string(), json!("sample"));
                                }
                                actions.push(UiAction::Command {
                                    window_label: self.adjust_dialog.window_label.clone(),
                                    command_id: "image.adjust.color_threshold".to_string(),
                                    params: Some(params),
                                });
                            }
                            ui.end_row();
                            if ui.button("Stack").clicked() {
                                let mut params = color_threshold_params(&self.adjust_dialog);
                                if let Value::Object(map) = &mut params {
                                    map.insert("action".to_string(), json!("stack"));
                                }
                                actions.push(UiAction::Command {
                                    window_label: self.adjust_dialog.window_label.clone(),
                                    command_id: "image.adjust.color_threshold".to_string(),
                                    params: Some(params),
                                });
                            }
                            if ui.button("Macro").clicked() {
                                let mut params = color_threshold_params(&self.adjust_dialog);
                                if let Value::Object(map) = &mut params {
                                    map.insert("action".to_string(), json!("macro"));
                                }
                                actions.push(UiAction::Command {
                                    window_label: self.adjust_dialog.window_label.clone(),
                                    command_id: "image.adjust.color_threshold".to_string(),
                                    params: Some(params),
                                });
                            }
                            if ui.button("Help").clicked() {
                                actions.push(UiAction::Command {
                                    window_label: self.adjust_dialog.window_label.clone(),
                                    command_id: "image.adjust.color_threshold".to_string(),
                                    params: Some(json!({ "action": "help" })),
                                });
                            }
                        });
                }
                AdjustDialogKind::LineWidth => {
                    let mut changed = false;
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.adjust_dialog.line_width, 1.0..=300.0)
                                .integer()
                                .text("Width"),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.adjust_dialog.line_width)
                                .range(1.0..=f32::INFINITY)
                                .speed(1.0),
                        )
                        .changed();
                    if changed {
                        self.adjust_dialog.line_width =
                            self.adjust_dialog.line_width.round().max(1.0);
                        actions.push(UiAction::Command {
                            window_label: self.adjust_dialog.window_label.clone(),
                            command_id: "image.adjust.line_width".to_string(),
                            params: Some(json!({
                                "width": self.adjust_dialog.line_width
                            })),
                        });
                    }
                    let can_spline_fit = self
                        .viewers_ui
                        .get(&self.adjust_dialog.window_label)
                        .is_some_and(selected_roi_can_spline_fit);
                    if !can_spline_fit {
                        self.adjust_dialog.spline_fit = false;
                    }
                    if ui
                        .add_enabled(
                            can_spline_fit,
                            egui::Checkbox::new(&mut self.adjust_dialog.spline_fit, "Spline fit"),
                        )
                        .changed()
                    {
                        actions.push(UiAction::Command {
                            window_label: self.adjust_dialog.window_label.clone(),
                            command_id: "image.adjust.line_width".to_string(),
                            params: Some(json!({
                                "width": self.adjust_dialog.line_width,
                                "spline_fit": self.adjust_dialog.spline_fit
                            })),
                        });
                    }
                }
                AdjustDialogKind::Coordinates => {
                    let is_stack = coordinates_dialog_is_stack(&self.adjust_dialog);
                    egui::Grid::new("coordinates_adjust_grid")
                        .num_columns(2)
                        .spacing(egui::vec2(8.0, 4.0))
                        .show(ui, |ui| {
                            if self.adjust_dialog.coordinates_mode == "point" {
                                for (label, value) in [
                                    ("X:", &mut self.adjust_dialog.left),
                                    ("Y:", &mut self.adjust_dialog.top),
                                ] {
                                    ui.label(label);
                                    ui.add(egui::DragValue::new(value).speed(1.0));
                                    ui.end_row();
                                }
                                if is_stack {
                                    ui.label("Z:");
                                    ui.add(
                                        egui::DragValue::new(&mut self.adjust_dialog.front)
                                            .speed(1.0),
                                    );
                                    ui.end_row();
                                }
                            } else {
                                for (label, value) in [
                                    ("Left:", &mut self.adjust_dialog.left),
                                    ("Right:", &mut self.adjust_dialog.right),
                                    ("Top:", &mut self.adjust_dialog.top),
                                    ("Bottom:", &mut self.adjust_dialog.bottom),
                                ] {
                                    ui.label(label);
                                    ui.add(egui::DragValue::new(value).speed(1.0));
                                    ui.end_row();
                                }
                                if is_stack {
                                    for (label, value) in [
                                        ("Front:", &mut self.adjust_dialog.front),
                                        ("Back:", &mut self.adjust_dialog.back),
                                    ] {
                                        ui.label(label);
                                        ui.add(egui::DragValue::new(value).speed(1.0));
                                        ui.end_row();
                                    }
                                }
                            }
                            ui.label("X_unit:");
                            ui.text_edit_singleline(&mut self.adjust_dialog.x_unit);
                            ui.end_row();
                            ui.label("Y_unit:");
                            ui.text_edit_singleline(&mut self.adjust_dialog.y_unit);
                            ui.end_row();
                            if is_stack {
                                ui.label("Z_unit:");
                                ui.text_edit_singleline(&mut self.adjust_dialog.z_unit);
                                ui.end_row();
                            }
                        });
                    if ui.button("OK").clicked() {
                        let params = if self.adjust_dialog.coordinates_mode == "point" {
                            json!({
                                "mode": "point",
                                "point_x_coordinate": self.adjust_dialog.left,
                                "point_y_coordinate": self.adjust_dialog.top,
                                "point_z_coordinate": self.adjust_dialog.front,
                                "point_x_pixel": self.adjust_dialog.coordinates_x_pixel,
                                "point_y_pixel": self.adjust_dialog.coordinates_y_pixel,
                                "point_z_pixel": self.adjust_dialog.coordinates_z_pixel,
                                "x_unit": self.adjust_dialog.x_unit,
                                "y_unit": self.adjust_dialog.y_unit,
                                "z_unit": self.adjust_dialog.z_unit,
                            })
                        } else {
                            json!({
                                "mode": self.adjust_dialog.coordinates_mode,
                                "left": self.adjust_dialog.left,
                                "right": self.adjust_dialog.right,
                                "top": self.adjust_dialog.top,
                                "bottom": self.adjust_dialog.bottom,
                                "front": self.adjust_dialog.front,
                                "back": self.adjust_dialog.back,
                                "x_pixel_start": self.adjust_dialog.coordinates_x_pixel,
                                "x_pixel_size": self.adjust_dialog.coordinates_width,
                                "y_pixel_start": self.adjust_dialog.coordinates_y_pixel,
                                "y_pixel_size": self.adjust_dialog.coordinates_height,
                                "z_pixel_start": 0.0,
                                "z_pixel_size": self.adjust_dialog.coordinates_depth,
                                "x_unit": self.adjust_dialog.x_unit,
                                "y_unit": self.adjust_dialog.y_unit,
                                "z_unit": self.adjust_dialog.z_unit,
                            })
                        };
                        actions.push(UiAction::Command {
                            window_label: self.adjust_dialog.window_label.clone(),
                            command_id: "image.adjust.coordinates".to_string(),
                            params: Some(params),
                        });
                    }
                }
            });
        self.adjust_dialog.open = open && !close;
    }

    fn draw_threshold_apply_dialog(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        if !self.threshold_apply_dialog.open {
            return;
        }
        let mut open = self.threshold_apply_dialog.open;
        let mut convert_to_mask = false;
        let mut set_to_nan = false;
        let mut cancel = false;
        egui::Window::new("Thresholder")
            .open(&mut open)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.label("Convert to 8-bit mask or set background pixels to NaN?");
                ui.horizontal(|ui| {
                    if ui.button("Convert to Mask").clicked() {
                        convert_to_mask = true;
                    }
                    if ui.button("Set to NaN").clicked() {
                        set_to_nan = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            });

        if convert_to_mask || set_to_nan {
            let params = json!({
                "apply": true,
                "min": self.threshold_apply_dialog.min,
                "max": self.threshold_apply_dialog.max,
                "background": threshold_background_param(
                    self.threshold_apply_dialog.dark_background
                ),
                "background_to_nan": set_to_nan,
            });
            actions.push(UiAction::Command {
                window_label: self.threshold_apply_dialog.window_label.clone(),
                command_id: "image.adjust.threshold".to_string(),
                params: Some(params),
            });
            self.threshold_apply_dialog.open = false;
        } else if cancel {
            self.threshold_apply_dialog.open = false;
        } else {
            self.threshold_apply_dialog.open = open;
        }
    }

    fn draw_threshold_set_dialog(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        if !self.threshold_set_dialog.open {
            return;
        }

        let mut open = self.threshold_set_dialog.open;
        let mut ok = false;
        let mut cancel = false;
        egui::Window::new("Set Threshold Levels")
            .open(&mut open)
            .collapsible(false)
            .show(ctx, |ui| {
                egui::Grid::new("set_threshold_levels_grid")
                    .num_columns(2)
                    .spacing(egui::vec2(8.0, 4.0))
                    .show(ui, |ui| {
                        ui.label("Lower threshold level:");
                        ui.add(egui::DragValue::new(&mut self.threshold_set_dialog.min).speed(0.1));
                        ui.end_row();

                        ui.label("Upper threshold level:");
                        ui.add(egui::DragValue::new(&mut self.threshold_set_dialog.max).speed(0.1));
                        ui.end_row();
                    });

                ui.horizontal(|ui| {
                    if ui.button("OK").clicked() {
                        ok = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            });

        if ok {
            if self.threshold_set_dialog.max < self.threshold_set_dialog.min {
                self.threshold_set_dialog.max = self.threshold_set_dialog.min;
            }
            actions.push(UiAction::Command {
                window_label: self.threshold_set_dialog.window_label.clone(),
                command_id: "image.adjust.threshold".to_string(),
                params: Some(json!({
                    "min": self.threshold_set_dialog.min,
                    "max": self.threshold_set_dialog.max,
                    "mode": self.threshold_set_dialog.mode,
                    "background": threshold_background_param(
                        self.threshold_set_dialog.dark_background
                    ),
                })),
            });
            self.threshold_set_dialog.open = false;
        } else if cancel {
            self.threshold_set_dialog.open = false;
        } else {
            self.threshold_set_dialog.open = open;
        }
    }

    fn draw_apply_lut_dialog(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        if !self.apply_lut_dialog.open {
            return;
        }
        let mut open = self.apply_lut_dialog.open;
        let mut ok = false;
        let mut yes = false;
        let mut no = false;
        let mut cancel = false;
        let title = if self.apply_lut_dialog.stack_prompt {
            "Entire Stack?"
        } else {
            "Apply Lookup Table?"
        };
        egui::Window::new(title)
            .open(&mut open)
            .collapsible(false)
            .show(ctx, |ui| {
                if self.apply_lut_dialog.stack_prompt {
                    ui.label(format!(
                        "Apply LUT to all {} stack slices?",
                        self.apply_lut_dialog.stack_slices
                    ));
                    ui.horizontal(|ui| {
                        if ui.button("Yes").clicked() {
                            yes = true;
                        }
                        if ui.button("No").clicked() {
                            no = true;
                        }
                        if ui.button("Cancel").clicked() {
                            cancel = true;
                        }
                    });
                } else {
                    ui.label("WARNING: the pixel values will");
                    ui.label("change if you click \"OK\".");
                    ui.horizontal(|ui| {
                        if ui.button("OK").clicked() {
                            ok = true;
                        }
                        if ui.button("Cancel").clicked() {
                            cancel = true;
                        }
                    });
                }
            });

        if ok && self.apply_lut_dialog.stack_slices > 1 {
            self.apply_lut_dialog.stack_prompt = true;
            self.apply_lut_dialog.open = open;
        } else if ok || yes {
            actions.push(UiAction::Command {
                window_label: self.apply_lut_dialog.window_label.clone(),
                command_id: self.apply_lut_dialog.command_id.clone(),
                params: Some(self.apply_lut_dialog.params.clone()),
            });
            self.apply_lut_dialog.open = false;
        } else if no {
            actions.push(UiAction::Command {
                window_label: self.apply_lut_dialog.window_label.clone(),
                command_id: self.apply_lut_dialog.command_id.clone(),
                params: Some(self.apply_lut_dialog.slice_params.clone()),
            });
            self.apply_lut_dialog.open = false;
        } else if cancel {
            self.apply_lut_dialog.open = false;
        } else {
            self.apply_lut_dialog.open = open;
        }
    }

    fn draw_set_display_range_dialog(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        if !self.set_display_range_dialog.open {
            return;
        }

        let mut open = self.set_display_range_dialog.open;
        let mut ok = false;
        let mut cancel = false;
        egui::Window::new("Set Display Range")
            .open(&mut open)
            .collapsible(false)
            .show(ctx, |ui| {
                egui::Grid::new("set_display_range_grid")
                    .num_columns(2)
                    .spacing(egui::vec2(8.0, 4.0))
                    .show(ui, |ui| {
                        ui.label("Minimum displayed value:");
                        ui.add(
                            egui::DragValue::new(&mut self.set_display_range_dialog.minimum)
                                .speed(0.1),
                        );
                        ui.end_row();

                        ui.label("Maximum displayed value:");
                        ui.add(
                            egui::DragValue::new(&mut self.set_display_range_dialog.maximum)
                                .speed(0.1),
                        );
                        ui.end_row();
                    });

                egui::ComboBox::from_label("Unsigned 16-bit range:")
                    .selected_text(&self.set_display_range_dialog.unsigned_16bit_range)
                    .show_ui(ui, |ui| {
                        for range in [
                            "Automatic",
                            "8-bit (0-255)",
                            "10-bit (0-1023)",
                            "12-bit (0-4095)",
                            "14-bit (0-16383)",
                            "15-bit (0-32767)",
                            "16-bit (0-65535)",
                        ] {
                            ui.selectable_value(
                                &mut self.set_display_range_dialog.unsigned_16bit_range,
                                range.to_string(),
                                range,
                            );
                        }
                    });

                let propagate_label = if self.set_display_range_dialog.channel_count > 1 {
                    format!(
                        "Propagate to all other {} channel images",
                        self.set_display_range_dialog.channel_count
                    )
                } else {
                    "Propagate to all other open images".to_string()
                };
                ui.checkbox(
                    &mut self.set_display_range_dialog.propagate,
                    propagate_label,
                );
                if self.set_display_range_dialog.show_all_channels {
                    let count = self.set_display_range_dialog.channel_count;
                    let label = if count == 2 {
                        "Propagate to the other channel of this image".to_string()
                    } else {
                        format!(
                            "Propagate to the other {} channels of this image",
                            count - 1
                        )
                    };
                    ui.checkbox(&mut self.set_display_range_dialog.all_channels, label);
                }

                ui.horizontal(|ui| {
                    if ui.button("OK").clicked() {
                        ok = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            });

        if ok {
            let mut params = Map::new();
            params.insert(
                self.set_display_range_dialog.low_key.clone(),
                json!(self.set_display_range_dialog.minimum),
            );
            params.insert(
                self.set_display_range_dialog.high_key.clone(),
                json!(self.set_display_range_dialog.maximum),
            );
            params.insert(
                "unsigned_16bit_range".to_string(),
                json!(self.set_display_range_dialog.unsigned_16bit_range.clone()),
            );
            params.insert(
                "propagate".to_string(),
                json!(self.set_display_range_dialog.propagate),
            );
            if !self.set_display_range_dialog.channel.is_empty() {
                params.insert(
                    "channel".to_string(),
                    json!(self.set_display_range_dialog.channel.clone()),
                );
            }
            if self.set_display_range_dialog.all_channels {
                params.insert("all_channels".to_string(), json!(true));
            }
            actions.push(UiAction::Command {
                window_label: self.set_display_range_dialog.window_label.clone(),
                command_id: self.set_display_range_dialog.command_id.clone(),
                params: Some(Value::Object(params)),
            });
            self.set_display_range_dialog.open = false;
        } else if cancel {
            self.set_display_range_dialog.open = false;
        } else {
            self.set_display_range_dialog.open = open;
        }
    }

    fn draw_set_window_level_dialog(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        if !self.set_window_level_dialog.open {
            return;
        }

        let mut open = self.set_window_level_dialog.open;
        let mut ok = false;
        let mut cancel = false;
        egui::Window::new("Set W&L")
            .open(&mut open)
            .collapsible(false)
            .show(ctx, |ui| {
                egui::Grid::new("set_window_level_grid")
                    .num_columns(2)
                    .spacing(egui::vec2(8.0, 4.0))
                    .show(ui, |ui| {
                        ui.label("Window Center (Level):");
                        ui.add(
                            egui::DragValue::new(&mut self.set_window_level_dialog.level)
                                .speed(0.1),
                        );
                        ui.end_row();

                        ui.label("Window Width:");
                        ui.add(
                            egui::DragValue::new(&mut self.set_window_level_dialog.window)
                                .range(0.0..=f32::INFINITY)
                                .speed(0.1),
                        );
                        ui.end_row();
                    });
                ui.checkbox(
                    &mut self.set_window_level_dialog.propagate,
                    "Propagate to all open images",
                );
                ui.horizontal(|ui| {
                    if ui.button("OK").clicked() {
                        ok = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            });

        if ok {
            let half_window = self.set_window_level_dialog.window.max(0.0) * 0.5;
            actions.push(UiAction::Command {
                window_label: self.set_window_level_dialog.window_label.clone(),
                command_id: "image.adjust.window_level".to_string(),
                params: Some(json!({
                    "low": self.set_window_level_dialog.level - half_window,
                    "high": self.set_window_level_dialog.level + half_window,
                    "propagate": self.set_window_level_dialog.propagate,
                })),
            });
            self.set_window_level_dialog.open = false;
        } else if cancel {
            self.set_window_level_dialog.open = false;
        } else {
            self.set_window_level_dialog.open = open;
        }
    }

    fn draw_resize_dialog(
        &mut self,
        ctx: &egui::Context,
        actions: &mut Vec<UiAction>,
        canvas: bool,
    ) {
        let background_fill = f32::from(self.tool_options.background_color.r()) / 255.0;
        let dialog = if canvas {
            &mut self.canvas_dialog
        } else {
            &mut self.resize_dialog
        };
        if !dialog.open {
            return;
        }
        let mut open = dialog.open;
        let mut close = false;
        let old_width = dialog.width;
        let old_height = dialog.height;
        let old_x_scale = dialog.x_scale;
        let old_y_scale = dialog.y_scale;
        let old_z_scale = dialog.z_scale;
        let old_depth = dialog.depth;
        let title = if canvas {
            if dialog.depth > 1 || dialog.frames > 1 {
                "Resize Stack Canvas"
            } else {
                "Resize Image Canvas"
            }
        } else {
            "Scale"
        };
        egui::Window::new(title).open(&mut open).show(ctx, |ui| {
            egui::Grid::new(if canvas {
                "canvas_size_dialog_grid"
            } else {
                "resize_dialog_grid"
            })
            .num_columns(2)
            .spacing(egui::vec2(8.0, 4.0))
            .show(ui, |ui| {
                if !canvas {
                    ui.label("X Scale:");
                    ui.add(
                        egui::DragValue::new(&mut dialog.x_scale)
                            .range(0.0001..=f32::MAX)
                            .speed(0.01),
                    );
                    ui.end_row();

                    ui.label("Y Scale:");
                    ui.add(
                        egui::DragValue::new(&mut dialog.y_scale)
                            .range(0.0001..=f32::MAX)
                            .speed(0.01),
                    );
                    ui.end_row();

                    if dialog.depth > 1 {
                        ui.label("Z Scale:");
                        ui.add(
                            egui::DragValue::new(&mut dialog.z_scale)
                                .range(0.0001..=f32::MAX)
                                .speed(0.01),
                        );
                        ui.end_row();
                    }
                }

                ui.label(if canvas { "Width:" } else { "Width (pixels):" });
                ui.add(egui::DragValue::new(&mut dialog.width).range(1..=usize::MAX));
                ui.end_row();

                ui.label(if canvas {
                    "Height:"
                } else {
                    "Height (pixels):"
                });
                ui.add(egui::DragValue::new(&mut dialog.height).range(1..=usize::MAX));
                ui.end_row();

                if !canvas && dialog.depth > 1 {
                    ui.label("Depth (images):");
                    ui.add(egui::DragValue::new(&mut dialog.depth).range(1..=usize::MAX));
                    ui.end_row();
                }

                if !canvas && dialog.frames > 1 {
                    ui.label("Time (frames):");
                    ui.add(egui::DragValue::new(&mut dialog.frames).range(1..=usize::MAX));
                    ui.end_row();
                }
            });

            if canvas {
                egui::ComboBox::from_label("Position:")
                    .selected_text(&dialog.position)
                    .show_ui(ui, |ui| {
                        for position in [
                            "Top-Left",
                            "Top-Center",
                            "Top-Right",
                            "Center-Left",
                            "Center",
                            "Center-Right",
                            "Bottom-Left",
                            "Bottom-Center",
                            "Bottom-Right",
                        ] {
                            ui.selectable_value(
                                &mut dialog.position,
                                position.to_string(),
                                position,
                            );
                        }
                    });
                ui.checkbox(&mut dialog.zero_fill, "Zero Fill");
            } else {
                ui.checkbox(&mut dialog.constrain_aspect, "Constrain aspect ratio");
                if dialog.x_scale != old_x_scale && dialog.original_width > 0 {
                    dialog.width = ((dialog.original_width as f32) * dialog.x_scale)
                        .round()
                        .max(1.0) as usize;
                    if dialog.constrain_aspect && dialog.original_height > 0 {
                        dialog.y_scale = dialog.x_scale;
                        dialog.height = ((dialog.original_height as f32) * dialog.y_scale)
                            .round()
                            .max(1.0) as usize;
                    }
                } else if dialog.y_scale != old_y_scale && dialog.original_height > 0 {
                    dialog.height = ((dialog.original_height as f32) * dialog.y_scale)
                        .round()
                        .max(1.0) as usize;
                    if dialog.constrain_aspect && dialog.original_width > 0 {
                        dialog.x_scale = dialog.y_scale;
                        dialog.width = ((dialog.original_width as f32) * dialog.x_scale)
                            .round()
                            .max(1.0) as usize;
                    }
                } else if dialog.z_scale != old_z_scale && old_depth > 0 {
                    let original_depth = if old_z_scale > 0.0 {
                        old_depth as f32 / old_z_scale
                    } else {
                        old_depth as f32
                    };
                    dialog.depth = (original_depth * dialog.z_scale).round().max(1.0) as usize;
                } else if dialog.constrain_aspect
                    && dialog.width != old_width
                    && dialog.original_width > 0
                {
                    dialog.height = ((dialog.width as f32)
                        * (dialog.original_height as f32 / dialog.original_width as f32))
                        .round()
                        .max(1.0) as usize;
                    dialog.x_scale = dialog.width as f32 / dialog.original_width as f32;
                    if dialog.original_height > 0 {
                        dialog.y_scale = dialog.height as f32 / dialog.original_height as f32;
                    }
                } else if dialog.constrain_aspect
                    && dialog.height != old_height
                    && dialog.original_height > 0
                {
                    dialog.width = ((dialog.height as f32)
                        * (dialog.original_width as f32 / dialog.original_height as f32))
                        .round()
                        .max(1.0) as usize;
                    dialog.y_scale = dialog.height as f32 / dialog.original_height as f32;
                    if dialog.original_width > 0 {
                        dialog.x_scale = dialog.width as f32 / dialog.original_width as f32;
                    }
                } else {
                    if dialog.width != old_width && dialog.original_width > 0 {
                        dialog.x_scale = dialog.width as f32 / dialog.original_width as f32;
                    }
                    if dialog.height != old_height && dialog.original_height > 0 {
                        dialog.y_scale = dialog.height as f32 / dialog.original_height as f32;
                    }
                    if dialog.depth != old_depth && old_depth > 0 {
                        let original_depth = if old_z_scale > 0.0 {
                            old_depth as f32 / old_z_scale
                        } else {
                            old_depth as f32
                        };
                        dialog.z_scale = dialog.depth as f32 / original_depth;
                    }
                }
                egui::ComboBox::from_label("Interpolation:")
                    .selected_text(&dialog.interpolation)
                    .show_ui(ui, |ui| {
                        for method in ["None", "Bilinear", "Bicubic"] {
                            ui.selectable_value(
                                &mut dialog.interpolation,
                                method.to_string(),
                                method,
                            );
                        }
                    });
                if dialog.fill_with_background_available {
                    ui.checkbox(
                        &mut dialog.fill_with_background,
                        "Fill with background color",
                    );
                } else {
                    dialog.fill_with_background = false;
                }
                ui.checkbox(
                    &mut dialog.average_when_downsizing,
                    "Average when downsizing",
                );
                if dialog.process_stack_available {
                    ui.checkbox(&mut dialog.process_stack, "Process entire stack");
                } else {
                    dialog.process_stack = true;
                }
                ui.checkbox(&mut dialog.create_new_window, "Create new window");
                ui.horizontal(|ui| {
                    ui.label("Title:");
                    ui.text_edit_singleline(&mut dialog.title);
                });
            }
            ui.horizontal(|ui| {
                if ui.button("OK").clicked() {
                    let command_id = if canvas {
                        "__dialog.canvas_resize"
                    } else {
                        "__dialog.resize"
                    };
                    let fill = if canvas && dialog.zero_fill {
                        0.0
                    } else if canvas {
                        background_fill
                    } else {
                        dialog.fill
                    };
                    let params = if canvas {
                        json!({
                            "width": dialog.width,
                            "height": dialog.height,
                            "position": dialog.position,
                            "fill": fill,
                            "zero": dialog.zero_fill,
                        })
                    } else {
                        let z_scaling = dialog.depth != dialog.original_depth;
                        let process_stack =
                            z_scaling || !dialog.process_stack_available || dialog.process_stack;
                        let create_new_window = dialog.create_new_window || z_scaling;
                        let fill = if dialog.fill_with_background {
                            background_fill
                        } else {
                            0.0
                        };
                        json!({
                            "width": dialog.width,
                            "height": dialog.height,
                            "depth": dialog.depth,
                            "frames": dialog.frames,
                            "constrain": dialog.constrain_aspect,
                            "average_when_downsizing": dialog.average_when_downsizing,
                            "interpolation": dialog.interpolation,
                            "fill": fill,
                            "fill_with_background": dialog.fill_with_background,
                            "process_stack": process_stack,
                            "create_new_window": create_new_window,
                            "title": dialog.title,
                        })
                    };
                    actions.push(UiAction::Command {
                        window_label: self
                            .active_viewer_label
                            .clone()
                            .unwrap_or_else(|| LAUNCHER_LABEL.to_string()),
                        command_id: command_id.to_string(),
                        params: Some(params),
                    });
                    close = true;
                }
                if ui.button("Cancel").clicked() {
                    close = true;
                }
            });
        });
        dialog.open = open && !close;
    }

    fn draw_stack_position_dialog(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        if !self.stack_position_dialog.open {
            return;
        }
        let target = self.stack_position_dialog.window_label.clone();
        let summary = self
            .state
            .label_to_session
            .get(&target)
            .map(|session| session.committed_summary.clone());
        let mut open = self.stack_position_dialog.open;
        egui::Window::new("Set Position")
            .open(&mut open)
            .show(ctx, |ui| {
                if let Some(summary) = &summary {
                    ui.add(
                        egui::DragValue::new(&mut self.stack_position_dialog.channel)
                            .prefix("Channel ")
                            .range(1..=summary.channels.max(1)),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.stack_position_dialog.slice)
                            .prefix("Slice ")
                            .range(1..=summary.z_slices.max(1)),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.stack_position_dialog.frame)
                            .prefix("Frame ")
                            .range(1..=summary.times.max(1)),
                    );
                    if ui.button("Apply").clicked() {
                        actions.push(UiAction::Command {
                            window_label: target.clone(),
                            command_id: "image.stacks.set".to_string(),
                            params: Some(json!({
                                "channel": self.stack_position_dialog.channel,
                                "slice": self.stack_position_dialog.slice,
                                "frame": self.stack_position_dialog.frame,
                            })),
                        });
                        self.stack_position_dialog.open = false;
                    }
                } else {
                    ui.label("No active stack.");
                }
            });
        self.stack_position_dialog.open = open && self.stack_position_dialog.open;
    }

    fn draw_stack_label_dialog(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        if !self.stack_label_dialog.open {
            return;
        }
        let target = self.stack_label_dialog.window_label.clone();
        let mut open = self.stack_label_dialog.open;
        egui::Window::new(format!(
            "Set Slice Label ({})",
            self.stack_label_dialog.slice
        ))
        .open(&mut open)
        .show(ctx, |ui| {
            ui.text_edit_singleline(&mut self.stack_label_dialog.label);
            if ui.button("Apply").clicked() {
                actions.push(UiAction::Command {
                    window_label: target.clone(),
                    command_id: "image.stacks.set_label".to_string(),
                    params: Some(json!({
                        "slice": self.stack_label_dialog.slice,
                        "label": self.stack_label_dialog.label,
                    })),
                });
                self.stack_label_dialog.open = false;
            }
        });
        self.stack_label_dialog.open = open && self.stack_label_dialog.open;
    }

    fn draw_zoom_set_dialog(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        if !self.zoom_set_dialog.open {
            return;
        }
        let target = self.zoom_set_dialog.window_label.clone();
        let mut open = self.zoom_set_dialog.open;
        egui::Window::new("Set Zoom")
            .open(&mut open)
            .show(ctx, |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.zoom_set_dialog.zoom_percent)
                        .prefix("Zoom ")
                        .suffix("%")
                        .speed(1.0)
                        .range(1.0..=3200.0),
                );
                ui.add(
                    egui::DragValue::new(&mut self.zoom_set_dialog.x_center)
                        .prefix("X center ")
                        .speed(1.0),
                );
                ui.add(
                    egui::DragValue::new(&mut self.zoom_set_dialog.y_center)
                        .prefix("Y center ")
                        .speed(1.0),
                );
                if ui.button("Apply").clicked() {
                    actions.push(UiAction::Command {
                        window_label: target.clone(),
                        command_id: "image.zoom.set".to_string(),
                        params: Some(json!({
                            "zoom_percent": self.zoom_set_dialog.zoom_percent,
                            "x": self.zoom_set_dialog.x_center,
                            "y": self.zoom_set_dialog.y_center,
                        })),
                    });
                    self.zoom_set_dialog.open = false;
                }
            });
        self.zoom_set_dialog.open = open && self.zoom_set_dialog.open;
    }

    fn draw_color_dialog(&mut self, ctx: &egui::Context) {
        if !self.color_dialog.open {
            return;
        }
        let title = match self.color_dialog.mode {
            ColorDialogMode::Colors => "Colors",
            ColorDialogMode::Foreground => "Select Foreground Color",
            ColorDialogMode::Background => "Select Background Color",
            ColorDialogMode::Picker => "Color Picker",
        };
        let mut open = self.color_dialog.open;
        let mut apply = false;
        let mut cancel = false;
        egui::Window::new(title).open(&mut open).show(ctx, |ui| {
            match self.color_dialog.mode {
                ColorDialogMode::Colors => {
                    named_color_combo(ui, "Foreground", &mut self.color_dialog.foreground);
                    named_color_combo(ui, "Background", &mut self.color_dialog.background);
                }
                ColorDialogMode::Foreground => {
                    ui.horizontal(|ui| {
                        ui.label("Foreground");
                        egui::color_picker::color_edit_button_srgba(
                            ui,
                            &mut self.color_dialog.foreground,
                            egui::color_picker::Alpha::Opaque,
                        );
                    });
                }
                ColorDialogMode::Background => {
                    ui.horizontal(|ui| {
                        ui.label("Background");
                        egui::color_picker::color_edit_button_srgba(
                            ui,
                            &mut self.color_dialog.background,
                            egui::color_picker::Alpha::Opaque,
                        );
                    });
                }
                ColorDialogMode::Picker => {
                    ui.horizontal(|ui| {
                        ui.label("Foreground");
                        egui::color_picker::color_edit_button_srgba(
                            ui,
                            &mut self.color_dialog.foreground,
                            egui::color_picker::Alpha::Opaque,
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Background");
                        egui::color_picker::color_edit_button_srgba(
                            ui,
                            &mut self.color_dialog.background,
                            egui::color_picker::Alpha::Opaque,
                        );
                    });
                }
            }
            ui.horizontal(|ui| {
                if ui.button("Apply").clicked() {
                    apply = true;
                }
                if ui.button("Cancel").clicked() {
                    cancel = true;
                }
            });
        });

        if apply {
            self.tool_options.foreground_color = self.color_dialog.foreground;
            self.tool_options.background_color = self.color_dialog.background;
            self.color_dialog.open = false;
        } else if cancel {
            self.color_dialog.open = false;
        } else {
            self.color_dialog.open = open;
        }
    }

    fn draw_raw_import_dialog(&mut self, ctx: &egui::Context) {
        if !self.raw_import_dialog.open {
            return;
        }
        let mut open = self.raw_import_dialog.open;
        egui::Window::new("Raw Import")
            .open(&mut open)
            .show(ctx, |ui| {
                if ui.button("Choose File").clicked() {
                    self.raw_import_dialog.path = FileDialog::new().pick_file();
                }
                ui.label(
                    self.raw_import_dialog
                        .path
                        .as_ref()
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "No file selected".to_string()),
                );
                ui.add(egui::DragValue::new(&mut self.raw_import_dialog.width).prefix("Width "));
                ui.add(egui::DragValue::new(&mut self.raw_import_dialog.height).prefix("Height "));
                ui.add(egui::DragValue::new(&mut self.raw_import_dialog.slices).prefix("Slices "));
                ui.add(
                    egui::DragValue::new(&mut self.raw_import_dialog.channels).prefix("Channels "),
                );
                ui.add(
                    egui::DragValue::new(&mut self.raw_import_dialog.byte_offset)
                        .prefix("Byte Offset "),
                );
                ui.checkbox(&mut self.raw_import_dialog.little_endian, "Little Endian");
                pixel_type_selector(ui, &mut self.raw_import_dialog.pixel_type, "Pixel Type");
                if ui.button("Import").clicked() {
                    let result = self.finish_raw_import();
                    self.set_fallback_status(result);
                }
            });
        self.raw_import_dialog.open = open;
    }

    fn draw_url_import_dialog(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        if !self.url_import_dialog.open {
            return;
        }
        let mut open = self.url_import_dialog.open;
        egui::Window::new("Import URL")
            .open(&mut open)
            .show(ctx, |ui| {
                ui.text_edit_singleline(&mut self.url_import_dialog.url);
                if ui.button("Import").clicked() {
                    actions.push(UiAction::Command {
                        window_label: LAUNCHER_LABEL.to_string(),
                        command_id: "file.import.url".to_string(),
                        params: Some(json!({ "url": self.url_import_dialog.url })),
                    });
                    self.url_import_dialog.open = false;
                }
            });
        self.url_import_dialog.open = open;
    }

    fn draw_results_window(&mut self, ctx: &egui::Context) {
        if !self.desktop_state.utility_windows.results_open {
            return;
        }
        let mut open = self.desktop_state.utility_windows.results_open;
        let columns = self.results_table.columns();
        egui::Window::new("Results")
            .open(&mut open)
            .vscroll(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Clear").clicked() {
                        self.results_table.clear();
                    }
                    if ui.button("Copy JSON").clicked() {
                        let payload = serde_json::to_string_pretty(&self.results_table.rows)
                            .unwrap_or_default();
                        ui.ctx().copy_text(payload);
                    }
                });
                if self.results_table.rows.is_empty() {
                    ui.label("No results yet.");
                    return;
                }
                egui::Grid::new("results-grid")
                    .striped(true)
                    .show(ui, |ui| {
                        for column in &columns {
                            ui.strong(column);
                        }
                        ui.end_row();
                        for row in &self.results_table.rows {
                            for column in &columns {
                                ui.label(
                                    row.get(column)
                                        .map(value_to_display)
                                        .unwrap_or_else(|| "-".to_string()),
                                );
                            }
                            ui.end_row();
                        }
                    });
            });
        self.desktop_state.utility_windows.results_open = open;
    }

    fn draw_measurement_settings_window(&mut self, ctx: &egui::Context) {
        if !self.desktop_state.utility_windows.measurements_open {
            return;
        }
        let mut open = self.desktop_state.utility_windows.measurements_open;
        egui::Window::new("Set Measurements")
            .open(&mut open)
            .show(ctx, |ui| {
                measurement_checkbox(
                    ui,
                    &mut self.desktop_state.measurement_settings.area,
                    "Area",
                );
                measurement_checkbox(ui, &mut self.desktop_state.measurement_settings.min, "Min");
                measurement_checkbox(ui, &mut self.desktop_state.measurement_settings.max, "Max");
                measurement_checkbox(
                    ui,
                    &mut self.desktop_state.measurement_settings.mean,
                    "Mean",
                );
                measurement_checkbox(
                    ui,
                    &mut self.desktop_state.measurement_settings.centroid,
                    "Centroid",
                );
                measurement_checkbox(
                    ui,
                    &mut self.desktop_state.measurement_settings.bbox,
                    "BBox",
                );
                measurement_checkbox(
                    ui,
                    &mut self.desktop_state.measurement_settings.integrated_density,
                    "Integrated Density",
                );
                measurement_checkbox(
                    ui,
                    &mut self.desktop_state.measurement_settings.slice,
                    "Slice",
                );
                measurement_checkbox(
                    ui,
                    &mut self.desktop_state.measurement_settings.channel,
                    "Channel",
                );
                measurement_checkbox(
                    ui,
                    &mut self.desktop_state.measurement_settings.time,
                    "Time",
                );
            });
        self.desktop_state.utility_windows.measurements_open = open;
    }

    fn draw_binary_options_window(&mut self, ctx: &egui::Context) {
        if !self.desktop_state.utility_windows.binary_options_open {
            return;
        }

        let mut open = self.desktop_state.utility_windows.binary_options_open;
        let was_open = open;
        let mut changed = false;
        egui::Window::new("Binary Options")
            .open(&mut open)
            .show(ctx, |ui| {
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.desktop_state.binary_options.iterations)
                            .prefix("Iterations ")
                            .range(1..=BINARY_MAX_ITERATIONS),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.desktop_state.binary_options.count)
                            .prefix("Count ")
                            .range(1..=BINARY_MAX_COUNT),
                    )
                    .changed();
            });

        self.desktop_state.utility_windows.binary_options_open = open;
        clamp_binary_options(&mut self.desktop_state.binary_options);
        if changed || open != was_open {
            self.persist_desktop_state();
        }
    }

    fn draw_overlay_settings_windows(&mut self, ctx: &egui::Context) {
        if self.desktop_state.utility_windows.overlay_labels_open {
            let mut open = self.desktop_state.utility_windows.overlay_labels_open;
            egui::Window::new("Labels").open(&mut open).show(ctx, |ui| {
                overlay_settings_controls(ui, &mut self.desktop_state.overlay_settings);
            });
            self.desktop_state.utility_windows.overlay_labels_open = open;
            self.persist_desktop_state();
        }

        if self.desktop_state.utility_windows.overlay_options_open {
            let mut open = self.desktop_state.utility_windows.overlay_options_open;
            egui::Window::new("Overlay Options")
                .open(&mut open)
                .show(ctx, |ui| {
                    overlay_settings_controls(ui, &mut self.desktop_state.overlay_settings);
                });
            self.desktop_state.utility_windows.overlay_options_open = open;
            self.persist_desktop_state();
        }
    }

    fn draw_roi_manager_window(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        if !self.desktop_state.utility_windows.roi_manager_open {
            return;
        }
        let mut open = self.desktop_state.utility_windows.roi_manager_open;
        egui::Window::new("ROI Manager")
            .open(&mut open)
            .vscroll(true)
            .show(ctx, |ui| {
                let Some(active_label) = self.active_viewer_label.clone() else {
                    ui.label("No active viewer.");
                    return;
                };
                let Some(viewer) = self.viewers_ui.get_mut(&active_label) else {
                    ui.label("No active viewer.");
                    return;
                };
                ui.horizontal(|ui| {
                    if ui.button("Add Current").clicked()
                        && let Some(selected) = viewer.rois.selected_roi_id
                        && let Some(roi) = viewer
                            .rois
                            .overlay_rois
                            .iter()
                            .find(|roi| roi.id == selected)
                            .cloned()
                    {
                        viewer.rois.overlay_rois.push(roi);
                    }
                    if ui.button("Delete").clicked() {
                        actions.push(UiAction::Command {
                            window_label: active_label.clone(),
                            command_id: "viewer.roi.delete".to_string(),
                            params: None,
                        });
                    }
                    if ui.button("Measure Selected").clicked() {
                        actions.push(UiAction::Command {
                            window_label: active_label.clone(),
                            command_id: "analyze.measure".to_string(),
                            params: None,
                        });
                    }
                    if ui.button("Measure All").clicked() {
                        actions.push(UiAction::Command {
                            window_label: active_label.clone(),
                            command_id: "__roi.measure_all".to_string(),
                            params: None,
                        });
                    }
                });
                for roi in &mut viewer.rois.overlay_rois {
                    ui.horizontal(|ui| {
                        let selected = viewer.rois.selected_roi_id == Some(roi.id);
                        if ui.selectable_label(selected, &roi.name).clicked() {
                            viewer.rois.selected_roi_id = Some(roi.id);
                            self.roi_manager.rename_buffer = roi.name.clone();
                        }
                        ui.checkbox(&mut roi.visible, "Show");
                        ui.checkbox(&mut roi.locked, "Lock");
                    });
                    if viewer.rois.selected_roi_id == Some(roi.id) {
                        ui.text_edit_singleline(&mut self.roi_manager.rename_buffer);
                        if ui.button("Rename").clicked() {
                            roi.name = self.roi_manager.rename_buffer.clone();
                        }
                    }
                    ui.separator();
                }
            });
        self.desktop_state.utility_windows.roi_manager_open = open;
    }

    fn draw_profile_plot_window(&mut self, ctx: &egui::Context) {
        if !self.desktop_state.utility_windows.profile_plot_open {
            return;
        }
        let mut open = self.desktop_state.utility_windows.profile_plot_open;
        egui::Window::new(if self.profile_plot.title.is_empty() {
            "Profile Plot"
        } else {
            &self.profile_plot.title
        })
        .open(&mut open)
        .show(ctx, |ui| {
            if self.profile_plot.samples.is_empty() {
                ui.label("No profile data.");
                return;
            }
            let available = egui::vec2(ui.available_width().max(240.0), 180.0);
            let (rect, _) = ui.allocate_exact_size(available, egui::Sense::hover());
            ui.painter().rect_stroke(
                rect,
                0.0,
                egui::Stroke::new(1.0, egui::Color32::GRAY),
                egui::StrokeKind::Outside,
            );
            let (min, max) = min_max(&self.profile_plot.samples);
            let span = (max - min).max(f32::EPSILON);
            let points = self
                .profile_plot
                .samples
                .iter()
                .enumerate()
                .map(|(i, value)| {
                    let x = rect.left()
                        + i as f32 * rect.width()
                            / (self.profile_plot.samples.len().max(2) - 1) as f32;
                    let y = rect.bottom() - ((*value - min) / span) * rect.height();
                    egui::pos2(x, y)
                })
                .collect::<Vec<_>>();
            ui.painter().add(egui::Shape::line(
                points,
                egui::Stroke::new(1.5, egui::Color32::LIGHT_GREEN),
            ));
        });
        self.desktop_state.utility_windows.profile_plot_open = open;
    }

    fn draw_help_windows(&mut self, ctx: &egui::Context) {
        draw_simple_window(
            ctx,
            "About image-rs",
            &mut self.desktop_state.utility_windows.help_about_open,
            |ui| {
                ui.label("image-rs");
                ui.label("Rust-first ImageJ-inspired desktop shell.");
            },
        );
        draw_simple_window(
            ctx,
            "Documentation",
            &mut self.desktop_state.utility_windows.help_docs_open,
            |ui| {
                ui.label("README and `image ops list` document the current CLI/runtime surface.");
            },
        );
        draw_simple_window(
            ctx,
            "Keyboard Shortcuts",
            &mut self.desktop_state.utility_windows.help_shortcuts_open,
            |ui| {
                ui.label("R/O/G/F/L/A/P/W/T/Z/H/D select tools.");
                ui.label("M measures, +/- zoom, </> move slices, Tab cycles windows.");
            },
        );
    }

    fn draw_command_finder_window(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        if !self.desktop_state.utility_windows.command_finder_open {
            return;
        }
        let mut open = self.desktop_state.utility_windows.command_finder_open;
        egui::Window::new("Command Finder")
            .open(&mut open)
            .vscroll(true)
            .show(ctx, |ui| {
                ui.text_edit_singleline(&mut self.command_finder.query);
                let needle = self.command_finder.query.to_ascii_lowercase();
                for entry in &self.command_catalog.entries {
                    if !needle.is_empty()
                        && !entry.label.to_ascii_lowercase().contains(&needle)
                        && !entry.id.to_ascii_lowercase().contains(&needle)
                    {
                        continue;
                    }
                    if ui
                        .button(format!("{} ({})", entry.label, entry.id))
                        .clicked()
                    {
                        let target = self
                            .active_viewer_label
                            .clone()
                            .unwrap_or_else(|| LAUNCHER_LABEL.to_string());
                        actions.push(UiAction::Command {
                            window_label: target,
                            command_id: entry.id.clone(),
                            params: None,
                        });
                    }
                }
                for path in &self.desktop_state.recent_files {
                    if needle.is_empty()
                        || path
                            .display()
                            .to_string()
                            .to_ascii_lowercase()
                            .contains(&needle)
                    {
                        if ui
                            .button(format!("Open Recent: {}", path.display()))
                            .clicked()
                        {
                            actions.push(UiAction::Command {
                                window_label: LAUNCHER_LABEL.to_string(),
                                command_id: format!("file.open_recent:{}", path.display()),
                                params: None,
                            });
                        }
                    }
                }
            });
        self.desktop_state.utility_windows.command_finder_open = open;
    }

    fn draw_viewer_tabs(&mut self, ui: &mut egui::Ui, actions: &mut Vec<UiAction>) {
        let mut labels = self.state.label_to_path.keys().cloned().collect::<Vec<_>>();
        labels.sort_by_key(|label| viewer_sort_key(label));

        let mut selected: Option<String> = None;
        let mut close_requested: Option<String> = None;

        egui::ScrollArea::horizontal()
            .id_salt("viewer-tabs-scroll")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for label in &labels {
                        let title = self
                            .state
                            .label_to_path
                            .get(label)
                            .and_then(|path| path.file_name())
                            .and_then(|name| name.to_str())
                            .map(str::to_string)
                            .or_else(|| {
                                self.viewers_ui
                                    .get(label)
                                    .map(|viewer| viewer.title.clone())
                            })
                            .unwrap_or_else(|| label.clone());
                        let is_active = self.active_viewer_label.as_deref() == Some(label);

                        if ui.selectable_label(is_active, title).clicked() {
                            selected = Some(label.clone());
                        }
                        if ui.small_button("x").on_hover_text("Close tab").clicked() {
                            close_requested = Some(label.clone());
                        }
                    }
                });
            });

        if let Some(label) = selected {
            self.set_active_viewer(Some(label));
        }
        if let Some(label) = close_requested {
            actions.push(UiAction::CloseViewer { label });
        }
    }

    fn draw_viewer_area(&mut self, ctx: &egui::Context, label: &str, actions: &mut Vec<UiAction>) {
        self.ensure_frame_for_viewer(ctx, label);

        egui::TopBottomPanel::top(format!("viewer-header-{label}")).show(ctx, |ui| {
            self.draw_viewer_tabs(ui, actions);
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                self.draw_viewer_toolbar(ui, label, actions);
            });
        });

        let summary = summary_for_window(&self.state, label)
            .ok()
            .map(|(_, summary)| summary);

        let mut hovered_viewer = false;
        let mut telemetry: Option<ViewerTelemetry> = None;
        let selected_tool = self.tool_state.selected;

        if let Some(viewer) = self.viewers_ui.get_mut(label) {
            if let Some(summary) = &summary {
                viewer.z = viewer.z.min(summary.z_slices.saturating_sub(1));
                viewer.t = viewer.t.min(summary.times.saturating_sub(1));
                viewer.channel = viewer.channel.min(summary.channels.saturating_sub(1));
            }

            egui::CentralPanel::default().show(ctx, |ui| {
                let frame = viewer.frame.clone();
                let texture_id = viewer.texture.as_ref().map(|texture| texture.id());
                let available = ui.available_size();
                let safe_available = egui::vec2(available.x.max(1.0), available.y.max(1.0));
                let (rect, response) =
                    ui.allocate_exact_size(safe_available, egui::Sense::click_and_drag());
                if selected_tool != ToolId::Zoom {
                    response.context_menu(|ui| {
                        draw_imagej_viewer_popup_menu(ui, label, actions);
                    });
                }
                ui.painter()
                    .rect_filled(rect, 0.0, egui::Color32::from_gray(16));

                if let (Some(frame), Some(texture_id)) = (frame, texture_id) {
                    if viewer.fit_requested {
                        initialize_view_to_open_state(viewer, rect, frame.width, frame.height);
                        viewer.fit_requested = false;
                    }

                    let image_rect =
                        image_draw_rect(rect, &viewer.transform, frame.width, frame.height);

                    if let Some(zoom_command) = viewer.pending_zoom.take() {
                        apply_zoom_command(viewer, zoom_command, rect, image_rect, &frame);
                    }

                    let image_rect =
                        image_draw_rect(rect, &viewer.transform, frame.width, frame.height);

                    let hover_sample = response
                        .hover_pos()
                        .and_then(|pointer| sample_at_pointer(viewer, &frame, image_rect, pointer));
                    viewer.hover = hover_sample;
                    hovered_viewer = response.hovered() || response.dragged() || response.clicked();
                    viewer.zoom = viewer.transform.magnification;

                    let input = ui.input(|i| i.clone());
                    let pointer = response.hover_pos();
                    let pointer_image = pointer
                        .and_then(|screen| viewer.transform.screen_to_image(image_rect, screen));
                    if let Some(pointer) = pointer {
                        if let Some(last) = viewer.last_status_pointer {
                            if (pointer - last).length_sq() > 144.0 {
                                viewer.tool_message = None;
                                viewer.last_status_pointer = Some(pointer);
                            }
                        } else {
                            viewer.last_status_pointer = Some(pointer);
                        }
                    }

                    if response.hovered() {
                        let scroll_delta = effective_scroll_delta(
                            input.smooth_scroll_delta,
                            input.raw_scroll_delta,
                        );
                        let zoom_modifier = input.modifiers.ctrl || input.modifiers.shift;
                        let zoom_scroll = dominant_scroll_component(scroll_delta);
                        let scroll = scroll_delta.y;
                        let has_scroll = if zoom_modifier {
                            zoom_scroll.abs() > f32::EPSILON
                        } else {
                            scroll.abs() > f32::EPSILON
                        };
                        if has_scroll {
                            if zoom_modifier {
                                if let Some(pointer) = pointer {
                                    let zoom_factor =
                                        if zoom_scroll > 0.0 { 1.12 } else { 1.0 / 1.12 };
                                    zoom_about_pointer(
                                        viewer,
                                        rect,
                                        image_rect,
                                        pointer,
                                        zoom_factor,
                                        frame.width,
                                        frame.height,
                                    );
                                }
                            } else if summary
                                .as_ref()
                                .is_some_and(|meta| meta.z_slices.saturating_sub(1) > 0)
                            {
                                if scroll < 0.0 {
                                    if let Some(meta) = &summary {
                                        viewer.z =
                                            (viewer.z + 1).min(meta.z_slices.saturating_sub(1));
                                    }
                                } else {
                                    viewer.z = viewer.z.saturating_sub(1);
                                }
                                viewer.last_request = None;
                            } else {
                                let horizontal = input.key_down(egui::Key::Space)
                                    || viewer.transform.src_rect.height >= frame.height as f32;
                                viewer.transform.wheel_pan(
                                    -scroll.signum(),
                                    horizontal,
                                    frame.width,
                                    frame.height,
                                );
                            }
                        }
                    }

                    match selected_tool {
                        ToolId::Hand => {
                            if response.dragged() {
                                viewer.transform.scroll_by_screen_delta(
                                    input.pointer.delta(),
                                    frame.width,
                                    frame.height,
                                );
                            }
                        }
                        ToolId::Zoom => {
                            if let Some(pointer) = pointer {
                                if response.clicked_by(egui::PointerButton::Primary) {
                                    let zoom_factor = if input.modifiers.shift {
                                        1.0 / 1.2
                                    } else {
                                        1.2
                                    };
                                    zoom_about_pointer(
                                        viewer,
                                        rect,
                                        image_rect,
                                        pointer,
                                        zoom_factor,
                                        frame.width,
                                        frame.height,
                                    );
                                }
                                if response.clicked_by(egui::PointerButton::Secondary) {
                                    zoom_about_pointer(
                                        viewer,
                                        rect,
                                        image_rect,
                                        pointer,
                                        1.0 / 1.2,
                                        frame.width,
                                        frame.height,
                                    );
                                }
                            }
                        }
                        ToolId::Point => {
                            if response.clicked_by(egui::PointerButton::Primary) {
                                viewer.pinned = hover_sample;
                                if let Some(point) = pointer_image {
                                    let mut points = vec![point];
                                    let multi =
                                        self.tool_options.point_mode == PointMode::MultiPoint;
                                    if multi {
                                        points = viewer
                                            .rois
                                            .overlay_rois
                                            .iter()
                                            .find_map(|roi| match &roi.kind {
                                                RoiKind::Point { points, .. } => {
                                                    Some(points.clone())
                                                }
                                                _ => None,
                                            })
                                            .unwrap_or_default();
                                        points.push(point);
                                    }
                                    let roi = RoiKind::Point { points, multi };
                                    viewer.rois.begin_active(
                                        roi,
                                        interaction::roi::RoiPosition {
                                            channel: viewer.channel,
                                            z: viewer.z,
                                            t: viewer.t,
                                        },
                                    );
                                    viewer.rois.commit_active(multi || input.modifiers.shift);
                                }
                            }
                        }
                        ToolId::Dropper => {
                            if response.clicked_by(egui::PointerButton::Primary) {
                                if let Some(sample) = hover_sample {
                                    let v = sample.value.clamp(0.0, 255.0) as u8;
                                    if input.modifiers.alt {
                                        self.tool_options.background_color =
                                            egui::Color32::from_gray(v);
                                    } else {
                                        self.tool_options.foreground_color =
                                            egui::Color32::from_gray(v);
                                    }
                                }
                            }
                        }
                        ToolId::Poly => {
                            if response.clicked_by(egui::PointerButton::Primary)
                                && let Some(point) = pointer_image
                            {
                                viewer.active_polygon_points.push(point);
                            }
                            if response.double_clicked() && viewer.active_polygon_points.len() >= 2
                            {
                                viewer.rois.begin_active(
                                    RoiKind::Polygon {
                                        points: viewer.active_polygon_points.clone(),
                                        closed: true,
                                        spline_fit: false,
                                    },
                                    interaction::roi::RoiPosition {
                                        channel: viewer.channel,
                                        z: viewer.z,
                                        t: viewer.t,
                                    },
                                );
                                viewer.rois.commit_active(!input.modifiers.shift);
                                viewer.active_polygon_points.clear();
                            }
                        }
                        ToolId::Angle => {
                            if response.clicked_by(egui::PointerButton::Primary)
                                && let Some(point) = pointer_image
                            {
                                viewer.active_polygon_points.push(point);
                                if viewer.active_polygon_points.len() == 3 {
                                    let points = viewer.active_polygon_points.clone();
                                    viewer.rois.begin_active(
                                        RoiKind::Angle {
                                            a: points[0],
                                            b: points[1],
                                            c: points[2],
                                        },
                                        interaction::roi::RoiPosition {
                                            channel: viewer.channel,
                                            z: viewer.z,
                                            t: viewer.t,
                                        },
                                    );
                                    viewer.rois.commit_active(!input.modifiers.shift);
                                    viewer.active_polygon_points.clear();
                                }
                            }
                        }
                        ToolId::Wand => {
                            if response.clicked_by(egui::PointerButton::Primary)
                                && let Some(point) = pointer_image
                            {
                                let radius = self.tool_options.wand_tolerance.max(4.0);
                                let mut points = Vec::with_capacity(16);
                                for step in 0..16 {
                                    let theta = step as f32 * std::f32::consts::TAU / 16.0;
                                    points.push(egui::pos2(
                                        point.x + radius * theta.cos(),
                                        point.y + radius * theta.sin(),
                                    ));
                                }
                                viewer.rois.begin_active(
                                    RoiKind::WandTrace { points },
                                    interaction::roi::RoiPosition {
                                        channel: viewer.channel,
                                        z: viewer.z,
                                        t: viewer.t,
                                    },
                                );
                                viewer.rois.commit_active(!input.modifiers.shift);
                            }
                        }
                        ToolId::Text => {
                            if response.clicked_by(egui::PointerButton::Primary)
                                && let Some(point) = pointer_image
                            {
                                viewer.rois.begin_active(
                                    RoiKind::Text {
                                        at: point,
                                        text: "Text".to_string(),
                                    },
                                    interaction::roi::RoiPosition {
                                        channel: viewer.channel,
                                        z: viewer.z,
                                        t: viewer.t,
                                    },
                                );
                                viewer.rois.commit_active(!input.modifiers.shift);
                            }
                        }
                        ToolId::Rect | ToolId::Oval | ToolId::Line | ToolId::Free => {
                            let primary_down =
                                input.pointer.button_down(egui::PointerButton::Primary);
                            if response.dragged()
                                && viewer.active_drag_started.is_none()
                                && let Some(start) = pointer_image
                            {
                                viewer.active_drag_started = Some(start);
                            }
                            if response.dragged()
                                && let (Some(start), Some(end)) =
                                    (viewer.active_drag_started, pointer_image)
                            {
                                let roi = drag_roi_for_tool(
                                    selected_tool,
                                    &self.tool_options,
                                    start,
                                    end,
                                    &mut viewer.active_polygon_points,
                                );
                                viewer.rois.begin_active(
                                    roi,
                                    interaction::roi::RoiPosition {
                                        channel: viewer.channel,
                                        z: viewer.z,
                                        t: viewer.t,
                                    },
                                );
                            }
                            if !primary_down && viewer.active_drag_started.is_some() {
                                viewer.active_drag_started = None;
                                viewer.active_polygon_points.clear();
                                viewer.rois.commit_active(!input.modifiers.shift);
                            }
                        }
                        _ => {}
                    }

                    ui.painter().image(
                        texture_id,
                        image_rect,
                        viewer.transform.src_rect.uv_rect(frame.width, frame.height),
                        egui::Color32::WHITE,
                    );

                    if let Some(pinned) = viewer.pinned {
                        draw_point_marker(ui.painter(), viewer, image_rect, pinned);
                    }
                    draw_rois(
                        ui.painter(),
                        viewer,
                        image_rect,
                        self.tool_options.line_width_px,
                        &self.desktop_state.overlay_settings,
                        interaction::roi::RoiPosition {
                            channel: viewer.channel,
                            z: viewer.z,
                            t: viewer.t,
                        },
                    );
                }
            });

            egui::TopBottomPanel::bottom(format!("viewer-status-{label}")).show(ctx, |ui| {
                ui.label(viewer.status_text(selected_tool));
            });

            let active_job = self
                .state
                .label_to_session
                .get(label)
                .and_then(|session| session.active_job.as_ref())
                .map(|job| format!("{} ({})", job.op, job.mode.as_str()));
            telemetry = Some(viewer.telemetry(active_job));
        }

        if let Some(telemetry) = telemetry {
            self.update_viewer_telemetry(label, telemetry);
        }

        if hovered_viewer {
            self.set_active_viewer(Some(label.to_string()));
        }
    }

    fn apply_actions(&mut self, actions: Vec<UiAction>) {
        for action in actions {
            match action {
                UiAction::Command {
                    window_label,
                    command_id,
                    params,
                } => {
                    let repeat_params = params.clone();
                    let label = self
                        .command_catalog
                        .entries
                        .iter()
                        .find(|entry| entry.id == command_id)
                        .map(|entry| entry.label.clone())
                        .unwrap_or_else(|| command_id.clone());
                    let result = self.dispatch_command(&window_label, &command_id, params);
                    self.remember_repeatable_command(&command_id, repeat_params.as_ref(), &result);
                    self.record_macro_command(&command_id, repeat_params.as_ref(), &result);
                    if window_label == LAUNCHER_LABEL {
                        self.set_fallback_status(format!("{label}: {}", result.message));
                    } else if let Some(viewer) = self.viewers_ui.get_mut(&window_label) {
                        viewer.tool_message = Some(result.message.clone());
                        viewer.status_message = result.message;
                    }
                }
                UiAction::RunInstalledMacro {
                    window_label,
                    path,
                    macro_name,
                } => {
                    let display_name = macro_display_name(&macro_name);
                    let message =
                        self.run_named_macro_block_file(&path, &macro_name, &window_label);
                    if window_label == LAUNCHER_LABEL {
                        self.set_fallback_status(format!("Macro {display_name}: {message}"));
                    } else if let Some(viewer) = self.viewers_ui.get_mut(&window_label) {
                        viewer.tool_message = Some(message.clone());
                        viewer.status_message = message;
                    }
                }
                UiAction::OpenPaths { paths } => {
                    let result = self.open_paths(paths);
                    self.apply_open_result(&result);
                }
                UiAction::CloseViewer { label } => {
                    self.remove_viewer_by_label(&label);
                }
            }
        }
    }

    fn maybe_focus_windows(&mut self, ctx: &egui::Context) {
        if self.focus_launcher {
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            self.focus_launcher = false;
        }

        if let Some(label) = self.focus_viewer_label.take()
            && self.viewers_ui.contains_key(&label)
        {
            self.set_active_viewer(Some(label));
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
        }
    }
}

impl eframe::App for ImageUiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let worker_state_changed = self.poll_worker_events();

        let mut actions = Vec::new();
        self.draw_launcher(ctx, &mut actions);

        let mut existing_labels = self
            .state
            .label_to_path
            .keys()
            .filter(|label| self.viewers_ui.contains_key(*label))
            .cloned()
            .collect::<Vec<_>>();
        existing_labels.sort_by_key(|label| viewer_sort_key(label));

        if self
            .active_viewer_label
            .as_ref()
            .is_none_or(|label| !self.viewers_ui.contains_key(label))
        {
            self.set_active_viewer(existing_labels.first().cloned());
        }

        if let Some(label) = self.active_viewer_label.clone() {
            self.draw_viewer_area(ctx, &label, &mut actions);
        } else {
            egui::CentralPanel::default().show(ctx, |_ui| {});
        }

        let stale_viewers = self
            .viewers_ui
            .keys()
            .filter(|label| !existing_labels.contains(*label))
            .cloned()
            .collect::<Vec<_>>();
        for label in stale_viewers {
            self.viewers_ui.remove(&label);
            self.viewer_telemetry.remove(&label);
            if self.active_viewer_label.as_deref() == Some(&label) {
                self.active_viewer_label = None;
            }
        }

        let has_pending_actions = !actions.is_empty();
        self.apply_actions(actions);

        let has_focus_or_close_command =
            self.focus_launcher || self.focus_viewer_label.is_some() || self.should_quit;
        self.maybe_focus_windows(ctx);
        self.refresh_launcher_status();

        if self.should_quit {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            self.should_quit = false;
        }

        let has_active_jobs = self.has_active_jobs();
        let repaint_inputs = ctx.input(|input| RepaintDecisionInputs {
            worker_state_changed,
            has_pending_actions,
            has_focus_or_close_command,
            has_pointer_activity: input.pointer.any_down()
                || input.pointer.delta() != egui::Vec2::ZERO,
            has_scroll_activity: input.raw_scroll_delta != egui::Vec2::ZERO
                || input.smooth_scroll_delta != egui::Vec2::ZERO,
            has_input_events: !input.events.is_empty(),
            has_active_jobs,
        });

        if should_request_repaint_now(repaint_inputs) {
            ctx.request_repaint();
        } else if should_request_periodic_repaint(repaint_inputs) {
            ctx.request_repaint_after(Duration::from_millis(16));
        }

        self.persist_desktop_state();
    }
}

fn format_launcher_status(
    selected_tool: ToolId,
    status: &LauncherStatusModel,
    telemetry: Option<&ViewerTelemetry>,
    label_to_path: &HashMap<String, PathBuf>,
    fallback_text: &str,
) -> String {
    let tool_text = format!("Tool:{}", selected_tool.label());
    let Some(active_viewer) = status.active_viewer.as_ref() else {
        return format!("{tool_text} | No image | {fallback_text}");
    };

    let context = label_to_path
        .get(active_viewer)
        .map(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Image")
                .to_string()
        })
        .unwrap_or_else(|| active_viewer.clone());

    let Some(telemetry) = telemetry else {
        return format!("{tool_text} | {context} | {fallback_text}");
    };

    let sample = telemetry
        .pinned
        .or(telemetry.hover)
        .map(|sample| format!("X:{} Y:{} Value:{:.4}", sample.x, sample.y, sample.value))
        .unwrap_or_else(|| "X:- Y:- Value:-".to_string());
    let progress = match &status.progress {
        ProgressState::Idle => "idle".to_string(),
        ProgressState::Running(message) => message.clone(),
    };

    format!(
        "{tool_text} | {context} | {sample} | Z:{} T:{} C:{} | Zoom:{:.0}% | {}",
        telemetry.z,
        telemetry.t,
        telemetry.c,
        telemetry.zoom * 100.0,
        progress
    )
}

fn sample_at_pointer(
    viewer: &ViewerUiState,
    frame: &ViewerFrameBuffer,
    rect: egui::Rect,
    pointer: egui::Pos2,
) -> Option<HoverInfo> {
    let image_pos = viewer.transform.screen_to_image(rect, pointer)?;
    let image_x = image_pos.x.floor() as isize;
    let image_y = image_pos.y.floor() as isize;
    if image_x < 0 || image_y < 0 {
        return None;
    }

    let x = image_x as usize;
    let y = image_y as usize;
    if x >= frame.width || y >= frame.height {
        return None;
    }

    let index = y * frame.width + x;
    let value = frame.values.get(index).copied().unwrap_or_else(|| {
        let gray = frame.pixels_u8[index] as f32;
        frame.min + (gray / 255.0) * (frame.max - frame.min)
    });
    Some(HoverInfo { x, y, value })
}

fn zoom_about_pointer(
    viewer: &mut ViewerUiState,
    viewport_rect: egui::Rect,
    image_rect: egui::Rect,
    pointer: egui::Pos2,
    factor: f32,
    image_width: usize,
    image_height: usize,
) {
    if factor >= 1.0 {
        let next = zoom_level_up(viewer.transform.magnification);
        viewer.transform.set_magnification_at_with_viewport(
            image_rect,
            viewport_rect,
            pointer,
            next,
            image_width,
            image_height,
        );
    } else {
        let next = zoom_level_down(viewer.transform.magnification);
        viewer.transform.set_magnification_at_with_viewport(
            image_rect,
            viewport_rect,
            pointer,
            next,
            image_width,
            image_height,
        );
    }
    viewer.zoom = viewer.transform.magnification;
    viewer.pan = egui::Vec2::ZERO;
}

fn draw_point_marker(
    painter: &egui::Painter,
    viewer: &ViewerUiState,
    rect: egui::Rect,
    pinned: HoverInfo,
) {
    let center = viewer.transform.image_to_screen(
        rect,
        egui::pos2(pinned.x as f32 + 0.5, pinned.y as f32 + 0.5),
    );
    let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 212, 26));
    painter.circle_stroke(center, 5.0, stroke);
    painter.line_segment(
        [
            center + egui::vec2(-8.0, 0.0),
            center + egui::vec2(8.0, 0.0),
        ],
        stroke,
    );
    painter.line_segment(
        [
            center + egui::vec2(0.0, -8.0),
            center + egui::vec2(0.0, 8.0),
        ],
        stroke,
    );
}

fn apply_zoom_command(
    viewer: &mut ViewerUiState,
    command: ZoomCommand,
    viewport_rect: egui::Rect,
    image_rect: egui::Rect,
    frame: &ViewerFrameBuffer,
) {
    let pointer = image_rect.center();
    match command {
        ZoomCommand::In => zoom_about_pointer(
            viewer,
            viewport_rect,
            image_rect,
            pointer,
            1.2,
            frame.width,
            frame.height,
        ),
        ZoomCommand::Out => zoom_about_pointer(
            viewer,
            viewport_rect,
            image_rect,
            pointer,
            1.0 / 1.2,
            frame.width,
            frame.height,
        ),
        ZoomCommand::Original => {
            viewer.transform.src_rect =
                interaction::transform::SourceRect::full(frame.width, frame.height);
            viewer.transform.magnification = viewer.initial_magnification;
            viewer.zoom = viewer.transform.magnification;
            viewer.pan = egui::Vec2::ZERO;
        }
        ZoomCommand::View100 => {
            viewer
                .transform
                .zoom_view_100(viewport_rect, frame.width, frame.height);
            viewer.zoom = viewer.transform.magnification;
        }
        ZoomCommand::ScaleToFit => {
            fit_to_rect(viewer, viewport_rect, frame.width, frame.height);
        }
        ZoomCommand::ToSelection => {
            if let Some(roi) = viewer.rois.active_roi.as_ref() {
                if let Some(bounds) = roi_bounds(&roi.kind) {
                    let mut src_rect = interaction::transform::SourceRect {
                        x: bounds.min.x,
                        y: bounds.min.y,
                        width: bounds.width().max(1.0),
                        height: bounds.height().max(1.0),
                    };
                    src_rect.clamp_to_image(frame.width, frame.height);
                    viewer.transform.src_rect = src_rect;
                    viewer.transform.magnification = (viewport_rect.width() / src_rect.width)
                        .min(viewport_rect.height() / src_rect.height)
                        .clamp(
                            interaction::transform::MIN_MAGNIFICATION,
                            interaction::transform::MAX_MAGNIFICATION,
                        );
                    viewer.zoom = viewer.transform.magnification;
                }
            }
        }
        ZoomCommand::Set {
            magnification,
            x_center,
            y_center,
        } => {
            let center = egui::pos2(
                x_center
                    .unwrap_or(viewer.transform.src_rect.x + viewer.transform.src_rect.width * 0.5),
                y_center.unwrap_or(
                    viewer.transform.src_rect.y + viewer.transform.src_rect.height * 0.5,
                ),
            );
            let pointer = viewer.transform.image_to_screen(image_rect, center);
            viewer.transform.set_magnification_at_with_viewport(
                image_rect,
                viewport_rect,
                pointer,
                magnification,
                frame.width,
                frame.height,
            );
            viewer.zoom = viewer.transform.magnification;
            viewer.pan = egui::Vec2::ZERO;
        }
        ZoomCommand::Maximize => {
            let src_rect = interaction::transform::SourceRect {
                x: viewer.transform.src_rect.x,
                y: viewer.transform.src_rect.y,
                width: (viewer.transform.src_rect.width * 0.5).max(1.0),
                height: (viewer.transform.src_rect.height * 0.5).max(1.0),
            };
            let mut src_rect = src_rect;
            src_rect.clamp_to_image(frame.width, frame.height);
            viewer.transform.src_rect = src_rect;
            viewer.transform.magnification = (viewport_rect.width() / src_rect.width)
                .min(viewport_rect.height() / src_rect.height)
                .clamp(
                    interaction::transform::MIN_MAGNIFICATION,
                    interaction::transform::MAX_MAGNIFICATION,
                );
            viewer.zoom = viewer.transform.magnification;
        }
    }
}

fn image_draw_rect(
    canvas_rect: egui::Rect,
    transform: &ViewerTransformState,
    image_width: usize,
    image_height: usize,
) -> egui::Rect {
    let draw_width = (transform.src_rect.width * transform.magnification)
        .max(1.0)
        .min(image_width.max(1) as f32 * interaction::transform::MAX_MAGNIFICATION);
    let draw_height = (transform.src_rect.height * transform.magnification)
        .max(1.0)
        .min(image_height.max(1) as f32 * interaction::transform::MAX_MAGNIFICATION);
    let offset_x = ((canvas_rect.width() - draw_width) * 0.5).max(0.0);
    let offset_y = ((canvas_rect.height() - draw_height) * 0.5).max(0.0);

    egui::Rect::from_min_size(
        egui::pos2(canvas_rect.min.x + offset_x, canvas_rect.min.y + offset_y),
        egui::vec2(draw_width, draw_height),
    )
}

#[cfg(test)]
fn compute_initial_viewport_size(
    summary: &ImageSummary,
    monitor_size: Option<egui::Vec2>,
) -> egui::Vec2 {
    let image_width = summary.shape.get(1).copied().unwrap_or(1) as f32;
    let image_height = summary.shape.first().copied().unwrap_or(1) as f32;
    let max_inner = monitor_size
        .map(|size| {
            egui::vec2(
                (size.x - VIEWER_WINDOW_EXTRA_SIZE[0] - 20.0).max(VIEWER_MIN_WINDOW_SIZE[0]),
                (size.y - VIEWER_WINDOW_EXTRA_SIZE[1] - 40.0).max(VIEWER_MIN_WINDOW_SIZE[1]),
            )
        })
        .unwrap_or(egui::vec2(1200.0, 900.0));
    let available_canvas = egui::vec2(
        (max_inner.x - VIEWER_WINDOW_EXTRA_SIZE[0]).max(1.0),
        (max_inner.y - VIEWER_WINDOW_EXTRA_SIZE[1]).max(1.0),
    );
    let initial_mag = (available_canvas.x / image_width)
        .min(available_canvas.y / image_height)
        .min(1.0);

    egui::vec2(
        (image_width * initial_mag + VIEWER_WINDOW_EXTRA_SIZE[0])
            .clamp(VIEWER_MIN_WINDOW_SIZE[0], max_inner.x),
        (image_height * initial_mag + VIEWER_WINDOW_EXTRA_SIZE[1])
            .clamp(VIEWER_MIN_WINDOW_SIZE[1], max_inner.y),
    )
}

fn dominant_scroll_component(delta: egui::Vec2) -> f32 {
    if delta.y.abs() >= delta.x.abs() {
        delta.y
    } else {
        delta.x
    }
}

fn effective_scroll_delta(smooth: egui::Vec2, raw: egui::Vec2) -> egui::Vec2 {
    if smooth == egui::Vec2::ZERO {
        raw
    } else if raw == egui::Vec2::ZERO {
        smooth
    } else {
        egui::vec2(
            if smooth.x.abs() >= raw.x.abs() {
                smooth.x
            } else {
                raw.x
            },
            if smooth.y.abs() >= raw.y.abs() {
                smooth.y
            } else {
                raw.y
            },
        )
    }
}

fn roi_bounds(kind: &RoiKind) -> Option<egui::Rect> {
    match kind {
        RoiKind::Rect { start, end, .. }
        | RoiKind::Oval { start, end, .. }
        | RoiKind::Line { start, end, .. } => Some(egui::Rect::from_two_pos(*start, *end)),
        RoiKind::Angle { a, b, c } => {
            let min_x = a.x.min(b.x).min(c.x);
            let min_y = a.y.min(b.y).min(c.y);
            let max_x = a.x.max(b.x).max(c.x);
            let max_y = a.y.max(b.y).max(c.y);
            Some(egui::Rect::from_min_max(
                egui::pos2(min_x, min_y),
                egui::pos2(max_x, max_y),
            ))
        }
        RoiKind::Polygon { points, .. }
        | RoiKind::Freehand { points }
        | RoiKind::WandTrace { points } => {
            let first = points.first()?;
            let mut min_x = first.x;
            let mut min_y = first.y;
            let mut max_x = first.x;
            let mut max_y = first.y;
            for point in points.iter().skip(1) {
                min_x = min_x.min(point.x);
                min_y = min_y.min(point.y);
                max_x = max_x.max(point.x);
                max_y = max_y.max(point.y);
            }
            Some(egui::Rect::from_min_max(
                egui::pos2(min_x, min_y),
                egui::pos2(max_x, max_y),
            ))
        }
        RoiKind::Point { points, .. } => {
            let first = points.first()?;
            Some(egui::Rect::from_center_size(*first, egui::vec2(1.0, 1.0)))
        }
        RoiKind::Text { at, .. } => Some(egui::Rect::from_center_size(*at, egui::vec2(1.0, 1.0))),
    }
}

fn roi_label_anchor(kind: &RoiKind) -> Option<egui::Pos2> {
    Some(roi_bounds(kind)?.center())
}

fn centered_circular_roi(shape: &[usize], radius: Option<f32>) -> Result<RoiKind, String> {
    if shape.len() < 2 {
        return Err("circular selection requires X/Y dimensions".to_string());
    }
    let height = shape[0] as f32;
    let width = shape[1] as f32;
    if width <= 0.0 || height <= 0.0 {
        return Err("circular selection requires non-empty X/Y dimensions".to_string());
    }
    let mut radius = radius.unwrap_or(width / 4.0);
    if !radius.is_finite() || radius < 0.0 {
        return Err("circular selection radius must be a finite non-negative value".to_string());
    }
    radius = radius.min(width / 2.0).min(height / 2.0);
    let center = egui::pos2(width / 2.0, height / 2.0);
    Ok(RoiKind::Oval {
        start: egui::pos2(center.x - radius, center.y - radius),
        end: egui::pos2(center.x + radius, center.y + radius),
        ellipse: false,
        brush: false,
    })
}

fn full_image_rect_roi(shape: &[usize]) -> Result<RoiKind, String> {
    if shape.len() < 2 {
        return Err("select all requires X/Y dimensions".to_string());
    }
    let height = shape[0] as f32;
    let width = shape[1] as f32;
    if width <= 0.0 || height <= 0.0 {
        return Err("select all requires non-empty X/Y dimensions".to_string());
    }
    Ok(RoiKind::Rect {
        start: egui::pos2(0.0, 0.0),
        end: egui::pos2(width, height),
        rounded: false,
        rotated: false,
    })
}

fn interpolate_roi_kind(
    kind: &RoiKind,
    interval: f32,
    smooth: bool,
    adjust: bool,
) -> Result<RoiKind, String> {
    if !interval.is_finite() || interval <= 0.0 {
        return Err("interpolation interval must be a finite positive number".to_string());
    }
    let (points, closed, output) = match kind {
        RoiKind::Line { start, end, .. } => (vec![*start, *end], false, "freehand"),
        RoiKind::Polygon { points, closed, .. } => (points.clone(), *closed, "polygon"),
        RoiKind::Freehand { points } => (points.clone(), false, "freehand"),
        RoiKind::WandTrace { points } => (points.clone(), points.len() > 2, "wand"),
        _ => {
            return Err(
                "Interpolate supports line, polygon, freehand, and wand selections".to_string(),
            );
        }
    };
    let points = resample_roi_points(&points, closed, interval, adjust)?;
    let points = if smooth {
        smooth_roi_points(&points, closed)
    } else {
        points
    };

    Ok(match output {
        "polygon" => RoiKind::Polygon {
            points,
            closed,
            spline_fit: false,
        },
        "wand" => RoiKind::WandTrace { points },
        _ => RoiKind::Freehand { points },
    })
}

fn resample_roi_points(
    points: &[egui::Pos2],
    closed: bool,
    interval: f32,
    adjust: bool,
) -> Result<Vec<egui::Pos2>, String> {
    if points.len() < 2 {
        return Err("Interpolate requires at least two selection points".to_string());
    }

    let mut path = points.to_vec();
    if closed {
        path.push(points[0]);
    }
    let mut distances = Vec::with_capacity(path.len());
    distances.push(0.0);
    for segment in path.windows(2) {
        let last = *distances.last().unwrap_or(&0.0);
        distances.push(last + segment[0].distance(segment[1]));
    }
    let total = *distances.last().unwrap_or(&0.0);
    if total <= f32::EPSILON {
        return Ok(points.to_vec());
    }

    let segment_count = if adjust {
        (total / interval).round().max(1.0)
    } else {
        (total / interval).ceil().max(1.0)
    };
    if segment_count > 50_000.0 {
        return Err("Interpolate would create too many points".to_string());
    }
    let step = if adjust {
        total / segment_count
    } else {
        interval
    };

    let mut output = Vec::new();
    let mut target = 0.0;
    while target < total {
        output.push(interpolate_point_at_distance(&path, &distances, target));
        target += step;
    }
    if !closed {
        output.push(*points.last().expect("points checked"));
    }

    Ok(output)
}

fn interpolate_point_at_distance(
    path: &[egui::Pos2],
    distances: &[f32],
    target: f32,
) -> egui::Pos2 {
    let segment_index = distances
        .windows(2)
        .position(|pair| target <= pair[1])
        .unwrap_or_else(|| distances.len().saturating_sub(2));
    let start_distance = distances[segment_index];
    let end_distance = distances[segment_index + 1];
    let span = (end_distance - start_distance).max(f32::EPSILON);
    let t = ((target - start_distance) / span).clamp(0.0, 1.0);
    path[segment_index].lerp(path[segment_index + 1], t)
}

fn smooth_roi_points(points: &[egui::Pos2], closed: bool) -> Vec<egui::Pos2> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut smoothed = Vec::with_capacity(points.len());
    for index in 0..points.len() {
        if !closed && (index == 0 || index + 1 == points.len()) {
            smoothed.push(points[index]);
            continue;
        }
        let previous = if index == 0 {
            points[points.len() - 1]
        } else {
            points[index - 1]
        };
        let next = if index + 1 == points.len() {
            points[0]
        } else {
            points[index + 1]
        };
        smoothed.push(egui::pos2(
            (previous.x + points[index].x + next.x) / 3.0,
            (previous.y + points[index].y + next.y) / 3.0,
        ));
    }
    smoothed
}

fn spline_fit_roi_points(points: &[egui::Pos2], closed: bool) -> Vec<egui::Pos2> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let segment_count = if closed {
        points.len()
    } else {
        points.len().saturating_sub(1)
    };
    let mut output = Vec::with_capacity(segment_count.saturating_mul(8) + 1);
    for index in 0..segment_count {
        let p0 = if index == 0 {
            if closed {
                points[points.len() - 1]
            } else {
                points[0]
            }
        } else {
            points[index - 1]
        };
        let p1 = points[index];
        let p2 = points[(index + 1) % points.len()];
        let p3 = if index + 2 < points.len() {
            points[index + 2]
        } else if closed {
            points[(index + 2) % points.len()]
        } else {
            points[points.len() - 1]
        };

        for step in 0..8 {
            let t = step as f32 / 8.0;
            let t2 = t * t;
            let t3 = t2 * t;
            let x = 0.5
                * ((2.0 * p1.x)
                    + (-p0.x + p2.x) * t
                    + (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2
                    + (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3);
            let y = 0.5
                * ((2.0 * p1.y)
                    + (-p0.y + p2.y) * t
                    + (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2
                    + (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3);
            output.push(egui::pos2(x, y));
        }
    }
    if !closed {
        output.push(*points.last().expect("points checked"));
    }
    output
}

fn drag_roi_for_tool(
    tool: ToolId,
    options: &ToolOptionsState,
    start: egui::Pos2,
    end: egui::Pos2,
    active_polygon_points: &mut Vec<egui::Pos2>,
) -> RoiKind {
    match tool {
        ToolId::Rect => RoiKind::Rect {
            start,
            end,
            rounded: options.rect_mode == RectMode::Rounded,
            rotated: options.rect_mode == RectMode::Rotated,
        },
        ToolId::Oval => RoiKind::Oval {
            start,
            end,
            ellipse: options.oval_mode == OvalMode::Ellipse,
            brush: options.oval_mode == OvalMode::Brush,
        },
        ToolId::Line => {
            if options.line_mode == LineMode::Freehand {
                if active_polygon_points.is_empty() {
                    active_polygon_points.push(start);
                }
                active_polygon_points.push(end);
                RoiKind::Freehand {
                    points: active_polygon_points.clone(),
                }
            } else {
                RoiKind::Line {
                    start,
                    end,
                    arrow: options.line_mode == LineMode::Arrow,
                }
            }
        }
        ToolId::Free => {
            if active_polygon_points.is_empty() {
                active_polygon_points.push(start);
            }
            active_polygon_points.push(end);
            RoiKind::Freehand {
                points: active_polygon_points.clone(),
            }
        }
        _ => RoiKind::Rect {
            start,
            end,
            rounded: false,
            rotated: false,
        },
    }
}

fn draw_rois(
    painter: &egui::Painter,
    viewer: &ViewerUiState,
    canvas_rect: egui::Rect,
    line_width_px: f32,
    overlay_settings: &OverlaySettings,
    position: interaction::roi::RoiPosition,
) {
    for (index, roi) in viewer.rois.visible_rois(position).enumerate() {
        let selected = viewer.rois.selected_roi_id == Some(roi.id);
        let width = roi_stroke_width(line_width_px, selected);
        let stroke = if selected {
            egui::Stroke::new(width, egui::Color32::from_rgb(255, 212, 26))
        } else {
            egui::Stroke::new(width, egui::Color32::from_rgb(52, 212, 255))
        };

        match &roi.kind {
            RoiKind::Rect { start, end, .. } => {
                let rect = egui::Rect::from_two_pos(
                    viewer.transform.image_to_screen(canvas_rect, *start),
                    viewer.transform.image_to_screen(canvas_rect, *end),
                );
                painter.rect_stroke(rect, 0.0, stroke, egui::StrokeKind::Outside);
            }
            RoiKind::Oval { start, end, .. } => {
                let rect = egui::Rect::from_two_pos(
                    viewer.transform.image_to_screen(canvas_rect, *start),
                    viewer.transform.image_to_screen(canvas_rect, *end),
                );
                painter.circle_stroke(rect.center(), rect.width().min(rect.height()) * 0.5, stroke);
            }
            RoiKind::Line { start, end, arrow } => {
                let p1 = viewer.transform.image_to_screen(canvas_rect, *start);
                let p2 = viewer.transform.image_to_screen(canvas_rect, *end);
                painter.line_segment([p1, p2], stroke);
                if *arrow {
                    let direction = (p1 - p2).normalized();
                    let left = egui::vec2(-direction.y, direction.x);
                    painter.line_segment([p2, p2 + (direction + left) * 8.0], stroke);
                    painter.line_segment([p2, p2 + (direction - left) * 8.0], stroke);
                }
            }
            RoiKind::Polygon {
                points,
                closed,
                spline_fit,
            } => {
                let display_points = if *spline_fit {
                    spline_fit_roi_points(points, *closed)
                } else {
                    points.clone()
                };
                let mut line_points = display_points
                    .iter()
                    .map(|point| viewer.transform.image_to_screen(canvas_rect, *point))
                    .collect::<Vec<_>>();
                if *closed && line_points.len() > 2 {
                    line_points.push(line_points[0]);
                }
                painter.add(egui::Shape::line(line_points, stroke));
            }
            RoiKind::Freehand { points } | RoiKind::WandTrace { points } => {
                let line_points = points
                    .iter()
                    .map(|point| viewer.transform.image_to_screen(canvas_rect, *point))
                    .collect::<Vec<_>>();
                painter.add(egui::Shape::line(line_points, stroke));
            }
            RoiKind::Angle { a, b, c } => {
                let pa = viewer.transform.image_to_screen(canvas_rect, *a);
                let pb = viewer.transform.image_to_screen(canvas_rect, *b);
                let pc = viewer.transform.image_to_screen(canvas_rect, *c);
                painter.line_segment([pa, pb], stroke);
                painter.line_segment([pb, pc], stroke);
            }
            RoiKind::Point { points, .. } => {
                for point in points {
                    let center = viewer.transform.image_to_screen(canvas_rect, *point);
                    painter.circle_stroke(center, 4.0, stroke);
                }
            }
            RoiKind::Text { at, text } => {
                let point = viewer.transform.image_to_screen(canvas_rect, *at);
                painter.text(
                    point,
                    egui::Align2::LEFT_TOP,
                    text,
                    egui::FontId::proportional(14.0),
                    stroke.color,
                );
            }
        }

        if let Some((anchor, text)) = overlay_label_for_roi(roi, index, overlay_settings) {
            let point = viewer.transform.image_to_screen(canvas_rect, anchor);
            let font_id = if overlay_settings.bold {
                egui::FontId::proportional(overlay_settings.font_size + 1.0)
            } else {
                egui::FontId::proportional(overlay_settings.font_size)
            };
            let color = imagej_color_from_name(&overlay_settings.label_color)
                .unwrap_or(egui::Color32::WHITE);
            if overlay_settings.draw_backgrounds {
                let width = text.chars().count() as f32 * overlay_settings.font_size * 0.62;
                let height = overlay_settings.font_size * 1.25;
                let bg = egui::Rect::from_min_size(point, egui::vec2(width, height));
                painter.rect_filled(bg, 0.0, egui::Color32::from_black_alpha(180));
            }
            painter.text(point, egui::Align2::LEFT_TOP, text, font_id, color);
        }
    }
}

fn compute_viewer_frame(
    state: &mut UiState,
    window_label: &str,
    request: &ViewerFrameRequest,
    preview_override: Option<&PreviewRequest>,
) -> Result<Arc<ViewerFrameBuffer>, String> {
    let (source_kind, source) = source_for_window(state, window_label, preview_override)?;
    let frame_key = FrameKey {
        source_kind: source_kind.clone(),
        z: request.z,
        t: request.t,
        channel: request.channel,
    };

    if let Some(frame) = state
        .label_to_session
        .get(window_label)
        .and_then(|session| session.frame_cache.get(&frame_key).cloned())
    {
        return Ok(frame);
    }

    let display_range = state
        .label_to_session
        .get(window_label)
        .and_then(|session| {
            session
                .channel_display_ranges
                .get(&request.channel)
                .copied()
                .or(session.display_range)
        });
    let frame = Arc::new(build_frame(&source, request, display_range)?);

    if let Some(session) = state.label_to_session.get_mut(window_label) {
        let can_store = preview_override.is_some() || session.current_source_kind() == source_kind;
        if can_store {
            session.frame_cache.insert(frame_key, frame.clone());
        }
    }

    Ok(frame)
}

fn source_for_window(
    state: &mut UiState,
    window_label: &str,
    preview_override: Option<&PreviewRequest>,
) -> Result<(String, ViewerImageSource), String> {
    if let Some(preview) = preview_override {
        let (preview_key, dataset) = ensure_preview_dataset(state, window_label, preview)?;
        return Ok((
            format!("{SOURCE_PREVIEW_PREFIX}{preview_key}"),
            ViewerImageSource::Dataset(dataset),
        ));
    }

    let session = state
        .label_to_session
        .get(window_label)
        .ok_or_else(|| format!("no viewer session for `{window_label}`"))?;
    let source_kind = session.current_source_kind();
    let source = session
        .source_for_kind(&source_kind)
        .ok_or_else(|| format!("no source found for `{source_kind}`"))?;

    Ok((source_kind, source))
}

fn ensure_preview_dataset(
    state: &mut UiState,
    window_label: &str,
    preview: &PreviewRequest,
) -> Result<(String, Arc<DatasetF32>), String> {
    let key = preview_cache_key(&preview.op, &preview.params)?;

    let committed = {
        let session = state
            .label_to_session
            .get(window_label)
            .ok_or_else(|| format!("no viewer session for `{window_label}`"))?;

        if let Some(dataset) = session.preview_cache.get(&key) {
            return Ok((key, dataset.clone()));
        }

        session.committed_source.to_dataset()?
    };

    let generated = state
        .app
        .ops_service()
        .execute(&preview.op, committed.as_ref(), &preview.params)
        .map(|output| Arc::new(output.dataset))
        .map_err(|error| error.to_string())?;

    let session = state
        .label_to_session
        .get_mut(window_label)
        .ok_or_else(|| format!("no viewer session for `{window_label}`"))?;
    let entry = session
        .preview_cache
        .entry(key.clone())
        .or_insert_with(|| generated.clone())
        .clone();

    Ok((key, entry))
}

fn summary_for_window(
    state: &UiState,
    window_label: &str,
) -> Result<(PathBuf, ImageSummary), String> {
    let session = state
        .label_to_session
        .get(window_label)
        .ok_or_else(|| format!("no viewer session for `{window_label}`"))?;

    let summary = session.committed_summary.clone();
    Ok((session.path.clone(), summary))
}

fn preview_cache_key(op: &str, params: &Value) -> Result<String, String> {
    let normalized = canonical_json(params);
    let encoded = serde_json::to_string(&normalized)
        .map_err(|error| format!("invalid preview params: {error}"))?;
    Ok(format!("{op}:{encoded}"))
}

fn line_width_from_params(params: &Value) -> Result<f32, String> {
    let width = params.get("width").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    if !width.is_finite() || width <= 0.0 {
        return Err("line width must be a finite positive number".to_string());
    }
    Ok(width.round().max(1.0))
}

fn resize_op_mode_from_params(params: &Value) -> OpRunMode {
    if params
        .get("create_new_window")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        OpRunMode::NewWindow
    } else {
        OpRunMode::Apply
    }
}

fn scale_fill_with_background_available(session: &ViewerSession) -> bool {
    session.committed_summary.channels >= 3
        || session
            .committed_source
            .to_dataset()
            .ok()
            .is_some_and(|dataset| dataset.metadata.pixel_type == PixelType::U8)
}

fn scale_process_stack_available(session: &ViewerSession) -> bool {
    session.committed_summary.z_slices > 1 && session.committed_summary.times <= 1
}

fn current_zt_plane_dataset(
    dataset: &DatasetF32,
    z: usize,
    t: usize,
) -> Result<DatasetF32, String> {
    if dataset.ndim() < 2 {
        return Err("dataset must have at least two dimensions".to_string());
    }
    let y_axis = dataset
        .axis_index(AxisKind::Y)
        .unwrap_or(0)
        .min(dataset.ndim() - 1);
    let x_axis = dataset
        .axis_index(AxisKind::X)
        .unwrap_or(1.min(dataset.ndim() - 1));
    if x_axis == y_axis {
        return Err("could not infer distinct X/Y axes".to_string());
    }
    let z_axis = dataset.axis_index(AxisKind::Z);
    let t_axis = dataset.axis_index(AxisKind::Time);
    let channel_axis = dataset.axis_index(AxisKind::Channel);
    let height = dataset.shape()[y_axis];
    let width = dataset.shape()[x_axis];
    let channels = channel_axis.map(|axis| dataset.shape()[axis]).unwrap_or(1);

    let mut index = vec![0usize; dataset.ndim()];
    if let Some(axis) = z_axis {
        index[axis] = z.min(dataset.shape()[axis].saturating_sub(1));
    }
    if let Some(axis) = t_axis {
        index[axis] = t.min(dataset.shape()[axis].saturating_sub(1));
    }

    let mut values = Vec::with_capacity(height * width * channels);
    for y in 0..height {
        index[y_axis] = y;
        for x in 0..width {
            index[x_axis] = x;
            if let Some(axis) = channel_axis {
                for channel in 0..channels {
                    index[axis] = channel;
                    values.push(dataset.data[IxDyn(&index)]);
                }
            } else {
                values.push(dataset.data[IxDyn(&index)]);
            }
        }
    }

    let mut dims = vec![
        plane_dim(&dataset.metadata.dims[y_axis], AxisKind::Y, height),
        plane_dim(&dataset.metadata.dims[x_axis], AxisKind::X, width),
    ];
    let shape = if let Some(axis) = channel_axis {
        dims.push(plane_dim(
            &dataset.metadata.dims[axis],
            AxisKind::Channel,
            channels,
        ));
        vec![height, width, channels]
    } else {
        vec![height, width]
    };
    let metadata = Metadata {
        dims,
        pixel_type: dataset.metadata.pixel_type,
        channel_names: dataset.metadata.channel_names.clone(),
        source: dataset.metadata.source.clone(),
        extras: dataset.metadata.extras.clone(),
    };
    let data = ArrayD::from_shape_vec(IxDyn(&shape), values)
        .map_err(|error| format!("current slice shape error: {error}"))?;
    Dataset::new(data, metadata).map_err(|error| error.to_string())
}

fn plane_dim(source: &Dim, axis: AxisKind, size: usize) -> Dim {
    let mut dim = source.clone();
    dim.axis = axis;
    dim.size = size;
    dim
}

fn scale_current_zt_slice_in_place(
    dataset: &DatasetF32,
    z: usize,
    t: usize,
    params: &Value,
) -> Result<DatasetF32, String> {
    if dataset.ndim() < 2 {
        return Err("dataset must have at least two dimensions".to_string());
    }
    let y_axis = dataset
        .axis_index(AxisKind::Y)
        .unwrap_or(0)
        .min(dataset.ndim() - 1);
    let x_axis = dataset
        .axis_index(AxisKind::X)
        .unwrap_or(1.min(dataset.ndim() - 1));
    if x_axis == y_axis {
        return Err("could not infer distinct X/Y axes".to_string());
    }
    let z_axis = dataset.axis_index(AxisKind::Z);
    let t_axis = dataset.axis_index(AxisKind::Time);
    let width = dataset.shape()[x_axis];
    let height = dataset.shape()[y_axis];
    let scaled_width = params
        .get("width")
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .unwrap_or(width)
        .max(1);
    let scaled_height = params
        .get("height")
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .unwrap_or(height)
        .max(1);
    let x_scale = scaled_width as f32 / width as f32;
    let y_scale = scaled_height as f32 / height as f32;
    let fill = params
        .get("fill")
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .unwrap_or(0.0);
    let nearest = params
        .get("interpolation")
        .and_then(Value::as_str)
        .is_some_and(|method| method.eq_ignore_ascii_case("none"));
    let original = dataset.data.clone();
    let mut values = original.iter().copied().collect::<Vec<_>>();
    for (flat_index, (index, _)) in original.indexed_iter().enumerate() {
        let z_matches = z_axis
            .map(|axis| index[axis] == z.min(dataset.shape()[axis].saturating_sub(1)))
            .unwrap_or(true);
        let t_matches = t_axis
            .map(|axis| index[axis] == t.min(dataset.shape()[axis].saturating_sub(1)))
            .unwrap_or(true);
        if !z_matches || !t_matches {
            continue;
        }
        let src_x = index[x_axis] as f32 / x_scale;
        let src_y = index[y_axis] as f32 / y_scale;
        values[flat_index] = sample_scaled_dataset_value(
            &original,
            index.slice(),
            x_axis,
            y_axis,
            width,
            height,
            src_x,
            src_y,
            fill,
            nearest,
        );
    }
    let data = ArrayD::from_shape_vec(IxDyn(dataset.shape()), values)
        .map_err(|error| format!("scaled slice shape error: {error}"))?;
    Dataset::new(data, dataset.metadata.clone()).map_err(|error| error.to_string())
}

fn sample_scaled_dataset_value(
    data: &ArrayD<f32>,
    base_index: &[usize],
    x_axis: usize,
    y_axis: usize,
    width: usize,
    height: usize,
    src_x: f32,
    src_y: f32,
    fill: f32,
    nearest: bool,
) -> f32 {
    if src_x < 0.0 || src_y < 0.0 || src_x >= width as f32 || src_y >= height as f32 {
        return fill;
    }
    if nearest {
        let x = src_x.round() as usize;
        let y = src_y.round() as usize;
        if x >= width || y >= height {
            return fill;
        }
        return dataset_value_at_xy(data, base_index, x_axis, y_axis, x, y);
    }

    let x0 = src_x.floor() as usize;
    let y0 = src_y.floor() as usize;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let wx = src_x - x0 as f32;
    let wy = src_y - y0 as f32;
    let top = dataset_value_at_xy(data, base_index, x_axis, y_axis, x0, y0) * (1.0 - wx)
        + dataset_value_at_xy(data, base_index, x_axis, y_axis, x1, y0) * wx;
    let bottom = dataset_value_at_xy(data, base_index, x_axis, y_axis, x0, y1) * (1.0 - wx)
        + dataset_value_at_xy(data, base_index, x_axis, y_axis, x1, y1) * wx;
    top * (1.0 - wy) + bottom * wy
}

fn dataset_value_at_xy(
    data: &ArrayD<f32>,
    base_index: &[usize],
    x_axis: usize,
    y_axis: usize,
    x: usize,
    y: usize,
) -> f32 {
    let mut index = base_index.to_vec();
    index[x_axis] = x;
    index[y_axis] = y;
    data[IxDyn(&index)]
}

fn roi_stroke_width(line_width_px: f32, selected: bool) -> f32 {
    let width = if line_width_px.is_finite() {
        line_width_px
    } else {
        1.0
    }
    .max(1.0);
    if selected { width + 0.5 } else { width }
}

fn value_to_display(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => value.to_string(),
    }
}

fn measurement_rows_from_table(table: &MeasurementTable) -> Vec<BTreeMap<String, Value>> {
    if let Some(rows) = table.values.get("rows").and_then(Value::as_array) {
        let parsed = rows
            .iter()
            .filter_map(Value::as_object)
            .map(|row| {
                row.iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect::<BTreeMap<_, _>>()
            })
            .collect::<Vec<_>>();
        if !parsed.is_empty() {
            return parsed;
        }
    }
    if table.values.is_empty() {
        Vec::new()
    } else {
        vec![table.values.clone()]
    }
}

fn profile_samples_from_table(table: &MeasurementTable) -> Option<Vec<f32>> {
    let samples = table.values.get("profile")?.as_array()?;
    Some(
        samples
            .iter()
            .filter_map(Value::as_f64)
            .map(|value| value as f32)
            .collect(),
    )
}

fn named_color_combo(ui: &mut egui::Ui, label: &str, color: &mut egui::Color32) {
    egui::ComboBox::from_label(label)
        .selected_text(imagej_color_to_string(*color))
        .show_ui(ui, |ui| {
            for (name, option) in imagej_named_colors() {
                ui.selectable_value(color, option, name);
            }
        });
}

fn imagej_named_colors() -> [(&'static str, egui::Color32); 13] {
    [
        ("Red", egui::Color32::from_rgb(255, 0, 0)),
        ("Green", egui::Color32::from_rgb(0, 255, 0)),
        ("Blue", egui::Color32::from_rgb(0, 0, 255)),
        ("Magenta", egui::Color32::from_rgb(255, 0, 255)),
        ("Cyan", egui::Color32::from_rgb(0, 255, 255)),
        ("Yellow", egui::Color32::from_rgb(255, 255, 0)),
        ("Orange", egui::Color32::from_rgb(255, 200, 0)),
        ("Black", egui::Color32::from_rgb(0, 0, 0)),
        ("White", egui::Color32::from_rgb(255, 255, 255)),
        ("Gray", egui::Color32::from_rgb(128, 128, 128)),
        ("lightGray", egui::Color32::from_rgb(192, 192, 192)),
        ("darkGray", egui::Color32::from_rgb(64, 64, 64)),
        ("Pink", egui::Color32::from_rgb(255, 175, 175)),
    ]
}

fn imagej_color_from_name(name: &str) -> Option<egui::Color32> {
    let name = name.to_ascii_lowercase();
    if name.contains("black") {
        Some(egui::Color32::from_rgb(0, 0, 0))
    } else if name.contains("white") {
        Some(egui::Color32::from_rgb(255, 255, 255))
    } else if name.contains("red") {
        Some(egui::Color32::from_rgb(255, 0, 0))
    } else if name.contains("blue") {
        Some(egui::Color32::from_rgb(0, 0, 255))
    } else if name.contains("yellow") {
        Some(egui::Color32::from_rgb(255, 255, 0))
    } else if name.contains("green") {
        Some(egui::Color32::from_rgb(0, 255, 0))
    } else if name.contains("magenta") {
        Some(egui::Color32::from_rgb(255, 0, 255))
    } else if name.contains("cyan") {
        Some(egui::Color32::from_rgb(0, 255, 255))
    } else if name.contains("orange") {
        Some(egui::Color32::from_rgb(255, 200, 0))
    } else if name.contains("pink") {
        Some(egui::Color32::from_rgb(255, 175, 175))
    } else if name.contains("gray") || name.contains("grey") {
        if name.contains("light") {
            Some(egui::Color32::from_rgb(192, 192, 192))
        } else if name.contains("dark") {
            Some(egui::Color32::from_rgb(64, 64, 64))
        } else {
            Some(egui::Color32::from_rgb(128, 128, 128))
        }
    } else {
        None
    }
}

fn imagej_color_to_string(color: egui::Color32) -> String {
    for (name, option) in imagej_named_colors() {
        if color == option {
            return name.to_string();
        }
    }
    format!("#{:02x}{:02x}{:02x}", color.r(), color.g(), color.b())
}

fn value_to_csv(value: &Value) -> String {
    let text = value_to_display(value);
    if text.contains(',') || text.contains('"') {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text
    }
}

fn pixel_type_id(pixel_type: PixelType) -> &'static str {
    match pixel_type {
        PixelType::U8 => "u8",
        PixelType::U16 => "u16",
        PixelType::F32 => "f32",
    }
}

fn pixel_type_selector(ui: &mut egui::Ui, pixel_type: &mut PixelType, label: &str) {
    egui::ComboBox::from_label(label)
        .selected_text(pixel_type_id(*pixel_type))
        .show_ui(ui, |ui| {
            ui.selectable_value(pixel_type, PixelType::U8, "u8");
            ui.selectable_value(pixel_type, PixelType::U16, "u16");
            ui.selectable_value(pixel_type, PixelType::F32, "f32");
        });
}

fn measurement_checkbox(ui: &mut egui::Ui, value: &mut bool, label: &str) {
    ui.checkbox(value, label);
}

fn overlay_settings_controls(ui: &mut egui::Ui, settings: &mut OverlaySettings) {
    named_overlay_color_combo(ui, &mut settings.label_color);
    ui.add(egui::Slider::new(&mut settings.font_size, 7.0..=72.0).text("Font size"));
    ui.checkbox(&mut settings.show_labels, "Show labels");
    ui.checkbox(&mut settings.use_names_as_labels, "Use names as labels");
    ui.checkbox(&mut settings.draw_backgrounds, "Draw backgrounds");
    ui.checkbox(&mut settings.bold, "Bold");
    if settings.use_names_as_labels {
        settings.show_labels = true;
    }
}

fn named_overlay_color_combo(ui: &mut egui::Ui, color: &mut String) {
    egui::ComboBox::from_label("Color")
        .selected_text(color.as_str())
        .show_ui(ui, |ui| {
            for (name, _) in imagej_named_colors() {
                ui.selectable_value(color, name.to_string(), name);
            }
        });
}

fn draw_simple_window(
    ctx: &egui::Context,
    title: &str,
    open: &mut bool,
    mut draw: impl FnMut(&mut egui::Ui),
) {
    if !*open {
        return;
    }
    egui::Window::new(title).open(open).show(ctx, |ui| draw(ui));
}

fn adjust_dialog_window_title(dialog: &AdjustDialogState) -> String {
    if dialog.kind == AdjustDialogKind::ColorBalance && dialog.color_balance_lut_color {
        return "LUT Color".to_string();
    }
    if dialog.kind != AdjustDialogKind::Coordinates {
        return dialog.kind.title().to_string();
    }
    match dialog.coordinates_mode.as_str() {
        "point" => "Point Coordinates".to_string(),
        "selection" => "Selection Coordinates".to_string(),
        _ => "Image Coordinates".to_string(),
    }
}

fn coordinates_dialog_is_stack(dialog: &AdjustDialogState) -> bool {
    dialog.coordinates_depth > 1.0
}

fn selected_roi_bbox(viewer: &ViewerUiState) -> Option<(usize, usize, usize, usize)> {
    let roi = viewer
        .rois
        .selected_roi_id
        .and_then(|id| viewer.rois.overlay_rois.iter().find(|roi| roi.id == id))
        .or(viewer.rois.active_roi.as_ref())?;
    roi_kind_bbox(&roi.kind)
}

fn selected_roi_model(viewer: &ViewerUiState) -> Option<&RoiModel> {
    viewer
        .rois
        .selected_roi_id
        .and_then(|id| viewer.rois.overlay_rois.iter().find(|roi| roi.id == id))
        .or(viewer.rois.active_roi.as_ref())
}

fn selected_roi_model_mut(viewer: &mut ViewerUiState) -> Option<&mut RoiModel> {
    if let Some(id) = viewer.rois.selected_roi_id
        && let Some(index) = viewer.rois.overlay_rois.iter().position(|roi| roi.id == id)
    {
        return viewer.rois.overlay_rois.get_mut(index);
    }
    viewer.rois.active_roi.as_mut()
}

fn selected_roi_can_spline_fit(viewer: &ViewerUiState) -> bool {
    matches!(
        selected_roi_model(viewer).map(|roi| &roi.kind),
        Some(RoiKind::Polygon { .. })
    )
}

fn selected_roi_spline_fit(viewer: &ViewerUiState) -> bool {
    match selected_roi_model(viewer).map(|roi| &roi.kind) {
        Some(RoiKind::Polygon { spline_fit, .. }) => *spline_fit,
        _ => false,
    }
}

fn set_selected_roi_spline_fit(viewer: &mut ViewerUiState, enabled: bool) -> bool {
    let Some(roi) = selected_roi_model_mut(viewer) else {
        return false;
    };
    let RoiKind::Polygon { spline_fit, .. } = &mut roi.kind else {
        return false;
    };
    *spline_fit = enabled;
    true
}

fn init_coordinates_dialog(
    dialog: &mut AdjustDialogState,
    metadata: &Metadata,
    viewer: Option<&ViewerUiState>,
    z_slices: usize,
    image_width: f32,
    image_height: f32,
) {
    let x_axis = metadata.axis_index(AxisKind::X);
    let y_axis = metadata.axis_index(AxisKind::Y);
    let z_axis = metadata.axis_index(AxisKind::Z);
    let x_unit = x_axis
        .and_then(|axis| metadata.dims[axis].unit.clone())
        .unwrap_or_else(|| "pixel".to_string());
    let y_unit = y_axis
        .and_then(|axis| metadata.dims[axis].unit.clone())
        .unwrap_or_else(|| x_unit.clone());
    let z_unit = z_axis
        .and_then(|axis| metadata.dims[axis].unit.clone())
        .unwrap_or_else(|| x_unit.clone());

    dialog.x_unit = x_unit.clone();
    dialog.y_unit = if y_unit == x_unit {
        "<same as x unit>".to_string()
    } else {
        y_unit
    };
    dialog.z_unit = if z_unit == x_unit {
        "<same as x unit>".to_string()
    } else {
        z_unit
    };
    dialog.coordinates_z_pixel = viewer.map(|viewer| viewer.z as f32).unwrap_or(0.0);
    dialog.coordinates_depth = z_slices.max(1) as f32;

    let selected = viewer.and_then(selected_roi_model);
    let single_point = selected.and_then(single_point_roi_position);
    if let Some(point) = single_point {
        dialog.coordinates_mode = "point".to_string();
        dialog.coordinates_x_pixel = point.x;
        dialog.coordinates_y_pixel = point.y;
        dialog.left = calibrated_axis_coordinate(metadata, x_axis, "x", point.x);
        dialog.top = calibrated_axis_coordinate(metadata, y_axis, "y", point.y);
        dialog.front =
            calibrated_axis_coordinate(metadata, z_axis, "z", dialog.coordinates_z_pixel);
        return;
    }

    let rect = selected
        .and_then(|roi| roi_bounds(&roi.kind))
        .unwrap_or_else(|| {
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(image_width, image_height))
        });
    let min = rect.min;
    let max = rect.max;
    dialog.coordinates_mode = if selected.is_some() {
        "selection".to_string()
    } else {
        "image".to_string()
    };
    dialog.coordinates_x_pixel = min.x.max(0.0);
    dialog.coordinates_y_pixel = min.y.max(0.0);
    dialog.coordinates_width = (max.x - min.x).abs().max(1.0);
    dialog.coordinates_height = (max.y - min.y).abs().max(1.0);
    dialog.left = calibrated_axis_coordinate(metadata, x_axis, "x", dialog.coordinates_x_pixel);
    dialog.right = calibrated_axis_coordinate(
        metadata,
        x_axis,
        "x",
        dialog.coordinates_x_pixel + dialog.coordinates_width,
    );
    dialog.top = calibrated_axis_coordinate(metadata, y_axis, "y", dialog.coordinates_y_pixel);
    dialog.bottom = calibrated_axis_coordinate(
        metadata,
        y_axis,
        "y",
        dialog.coordinates_y_pixel + dialog.coordinates_height,
    );
    dialog.front = calibrated_axis_coordinate(metadata, z_axis, "z", 0.0);
    dialog.back = calibrated_axis_coordinate(metadata, z_axis, "z", dialog.coordinates_depth);
}

fn single_point_roi_position(roi: &RoiModel) -> Option<egui::Pos2> {
    match &roi.kind {
        RoiKind::Point { points, .. } if points.len() == 1 => points.first().copied(),
        _ => None,
    }
}

fn calibrated_axis_coordinate(
    metadata: &Metadata,
    axis: Option<usize>,
    label: &str,
    pixel: f32,
) -> f32 {
    let Some(axis) = axis else {
        return pixel;
    };
    let origin = metadata
        .extras
        .get(&format!("{label}_origin_coordinate"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0) as f32;
    let spacing = metadata.dims[axis].spacing.unwrap_or(1.0);
    let inverted = metadata
        .extras
        .get(&format!("{label}_coordinate_inverted"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let direction = if inverted { -1.0 } else { 1.0 };
    origin + direction * spacing * pixel
}

fn selected_roi_profile_params(
    viewer: &ViewerUiState,
    base_params: Option<&Value>,
) -> Result<Value, String> {
    let mut params = base_params
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    params.insert("z".to_string(), json!(viewer.z));
    params.insert("time".to_string(), json!(viewer.t));
    params.insert("channel".to_string(), json!(viewer.channel));

    let explicit_selection = ["left", "top", "width", "height", "x0", "y0", "x1", "y1"]
        .iter()
        .any(|key| params.get(*key).is_some_and(|value| !value.is_null()));
    if explicit_selection {
        return Ok(Value::Object(params));
    }

    let roi = viewer
        .rois
        .selected_roi_id
        .and_then(|id| viewer.rois.overlay_rois.iter().find(|roi| roi.id == id))
        .or(viewer.rois.active_roi.as_ref())
        .ok_or_else(|| "Plot Profile requires a line or rectangular selection".to_string())?;

    match &roi.kind {
        RoiKind::Rect { .. } => {
            let (min_x, min_y, max_x, max_y) = roi_kind_bbox(&roi.kind)
                .ok_or_else(|| "Plot Profile requires a valid rectangular selection".to_string())?;
            params.insert("left".to_string(), json!(min_x));
            params.insert("top".to_string(), json!(min_y));
            params.insert(
                "width".to_string(),
                json!(max_x.saturating_sub(min_x).saturating_add(1)),
            );
            params.insert(
                "height".to_string(),
                json!(max_y.saturating_sub(min_y).saturating_add(1)),
            );
        }
        RoiKind::Line { start, end, .. } => {
            params.insert("x0".to_string(), json!(start.x));
            params.insert("y0".to_string(), json!(start.y));
            params.insert("x1".to_string(), json!(end.x));
            params.insert("y1".to_string(), json!(end.y));
        }
        _ => {
            return Err("Plot Profile requires a line or rectangular selection".to_string());
        }
    }

    Ok(Value::Object(params))
}

fn stack_to_image_datasets(dataset: &DatasetF32) -> Result<Vec<DatasetF32>, String> {
    let z_axis = dataset
        .axis_index(AxisKind::Z)
        .ok_or_else(|| "Stack to Images requires a Z stack".to_string())?;
    let slices = dataset.shape()[z_axis];
    if slices <= 1 {
        return Err("Stack to Images requires more than one Z slice".to_string());
    }

    let mut output = Vec::with_capacity(slices);
    for z in 0..slices {
        let data = dataset.data.view().index_axis(Axis(z_axis), z).to_owned();
        let mut metadata = dataset.metadata.clone();
        metadata.dims.remove(z_axis);
        metadata.extras.insert(
            "stack_to_images_source_slice".to_string(),
            json!(z.saturating_add(1)),
        );
        metadata.extras.insert(
            "stack_to_images_source_shape".to_string(),
            json!(dataset.shape().to_vec()),
        );
        output.push(
            Dataset::new(data.into_dyn(), metadata)
                .map_err(|error| format!("failed to create stack slice image: {error}"))?,
        );
    }
    Ok(output)
}

fn stack_slice_path(source_path: &Path, one_based_slice: usize) -> PathBuf {
    let stem = source_path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("Stack");
    let extension = source_path.extension().and_then(|name| name.to_str());
    let file_name = match extension {
        Some(extension) if !extension.is_empty() => {
            format!("{stem}-slice-{one_based_slice:03}.{extension}")
        }
        _ => format!("{stem}-slice-{one_based_slice:03}.tif"),
    };
    source_path.with_file_name(file_name)
}

fn images_to_stack_dataset(images: &[(&str, &DatasetF32)]) -> Result<DatasetF32, String> {
    if images.len() < 2 {
        return Err("Images to Stack requires at least two open 2D images".to_string());
    }

    let first = images[0].1;
    if first.axis_index(AxisKind::Z).is_some() {
        return Err("Images to Stack requires 2D images, not stacks".to_string());
    }
    let first_shape = first.shape().to_vec();
    let first_axes = first
        .metadata
        .dims
        .iter()
        .map(|dim| dim.axis)
        .collect::<Vec<_>>();

    for (title, dataset) in images {
        if dataset.axis_index(AxisKind::Z).is_some() {
            return Err(format!("`{title}` is already a stack"));
        }
        if dataset.shape() != first_shape.as_slice() {
            return Err("Images to Stack requires images with matching dimensions".to_string());
        }
        let axes = dataset
            .metadata
            .dims
            .iter()
            .map(|dim| dim.axis)
            .collect::<Vec<_>>();
        if axes != first_axes {
            return Err("Images to Stack requires images with matching axis order".to_string());
        }
    }

    let z_axis = images_to_stack_z_axis(&first.metadata);
    let views = images
        .iter()
        .map(|(_, dataset)| dataset.data.view())
        .collect::<Vec<_>>();
    let data = stack(Axis(z_axis), &views)
        .map_err(|error| format!("failed to build image stack: {error}"))?;

    let mut metadata = first.metadata.clone();
    metadata
        .dims
        .insert(z_axis, Dim::new(AxisKind::Z, images.len()));
    metadata.extras.insert(
        "images_to_stack_titles".to_string(),
        json!(images.iter().map(|(title, _)| *title).collect::<Vec<_>>()),
    );
    Ok(Dataset::new(data, metadata)
        .map_err(|error| format!("failed to create image stack dataset: {error}"))?)
}

fn images_to_stack_z_axis(metadata: &Metadata) -> usize {
    metadata
        .axis_index(AxisKind::Channel)
        .or_else(|| metadata.axis_index(AxisKind::Time))
        .unwrap_or(metadata.dims.len())
}

fn combine_stack_datasets(
    first: &DatasetF32,
    second: &DatasetF32,
    vertical: bool,
    fill: f32,
) -> Result<DatasetF32, String> {
    if first.axis_index(AxisKind::Channel).is_some()
        || second.axis_index(AxisKind::Channel).is_some()
        || first.axis_index(AxisKind::Time).is_some()
        || second.axis_index(AxisKind::Time).is_some()
    {
        return Err("Combine currently supports X/Y/Z images only".to_string());
    }
    if first.metadata.pixel_type != second.metadata.pixel_type {
        return Err("Combine requires images with matching pixel types".to_string());
    }

    let (first_width, first_height, first_depth) = stack_xyz_extent(first)?;
    let (second_width, second_height, second_depth) = stack_xyz_extent(second)?;
    let output_width = if vertical {
        first_width.max(second_width)
    } else {
        first_width + second_width
    };
    let output_height = if vertical {
        first_height + second_height
    } else {
        first_height.max(second_height)
    };
    let output_depth = first_depth.max(second_depth);
    let mut values = vec![fill; output_width * output_height * output_depth];

    copy_stack_into(
        &mut values,
        output_width,
        output_height,
        first,
        0,
        0,
        output_depth,
    )?;
    let second_x = if vertical { 0 } else { first_width };
    let second_y = if vertical { first_height } else { 0 };
    copy_stack_into(
        &mut values,
        output_width,
        output_height,
        second,
        second_x,
        second_y,
        output_depth,
    )?;

    let data = ArrayD::from_shape_vec(IxDyn(&[output_height, output_width, output_depth]), values)
        .map_err(|error| format!("combined stack shape error: {error}"))?;
    let mut metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, output_height),
            Dim::new(AxisKind::X, output_width),
            Dim::new(AxisKind::Z, output_depth),
        ],
        pixel_type: first.metadata.pixel_type,
        ..Metadata::default()
    };
    metadata.extras.insert(
        "stack_combine_orientation".to_string(),
        json!(if vertical { "vertical" } else { "horizontal" }),
    );
    metadata
        .extras
        .insert("stack_combine_fill".to_string(), json!(fill));
    Ok(Dataset::new(data, metadata)
        .map_err(|error| format!("failed to create combined stack: {error}"))?)
}

fn stack_xyz_extent(dataset: &DatasetF32) -> Result<(usize, usize, usize), String> {
    let y_axis = dataset
        .axis_index(AxisKind::Y)
        .ok_or_else(|| "Combine requires a Y axis".to_string())?;
    let x_axis = dataset
        .axis_index(AxisKind::X)
        .ok_or_else(|| "Combine requires an X axis".to_string())?;
    let width = dataset.shape()[x_axis];
    let height = dataset.shape()[y_axis];
    let depth = stack_slice_count(dataset);
    Ok((width, height, depth))
}

fn copy_stack_into(
    output: &mut [f32],
    output_width: usize,
    output_height: usize,
    dataset: &DatasetF32,
    x_offset: usize,
    y_offset: usize,
    output_depth: usize,
) -> Result<(), String> {
    let (width, height, depth) = stack_xyz_extent(dataset)?;
    for z in 0..depth.min(output_depth) {
        let slice = extract_slice(dataset, z, 0, 0)?;
        for y in 0..height {
            for x in 0..width {
                let out_x = x + x_offset;
                let out_y = y + y_offset;
                if out_x < output_width && out_y < output_height {
                    let index = (out_y * output_width + out_x) * output_depth + z;
                    output[index] = slice.values[y * width + x];
                }
            }
        }
    }
    Ok(())
}

fn concatenate_stack_datasets(
    datasets: &[(&str, &DatasetF32)],
    fill: f32,
) -> Result<DatasetF32, String> {
    if datasets.len() < 2 {
        return Err("Concatenate requires at least two open images or stacks".to_string());
    }
    let first_pixel_type = datasets[0].1.metadata.pixel_type;
    let mut output_width = 0usize;
    let mut output_height = 0usize;
    let mut output_depth = 0usize;
    for (title, dataset) in datasets {
        if dataset.axis_index(AxisKind::Channel).is_some()
            || dataset.axis_index(AxisKind::Time).is_some()
        {
            return Err(format!("`{title}` is not an X/Y/Z image"));
        }
        if dataset.metadata.pixel_type != first_pixel_type {
            return Err("Concatenate requires images with matching pixel types".to_string());
        }
        let (width, height, depth) = stack_xyz_extent(dataset)?;
        output_width = output_width.max(width);
        output_height = output_height.max(height);
        output_depth += depth;
    }

    let mut values = vec![fill; output_width * output_height * output_depth];
    let mut z_offset = 0usize;
    for (_, dataset) in datasets {
        let (_, _, depth) = stack_xyz_extent(dataset)?;
        copy_stack_into_z_offset(
            &mut values,
            output_width,
            output_height,
            output_depth,
            dataset,
            z_offset,
        )?;
        z_offset += depth;
    }

    let data = ArrayD::from_shape_vec(IxDyn(&[output_height, output_width, output_depth]), values)
        .map_err(|error| format!("concatenated stack shape error: {error}"))?;
    let mut metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, output_height),
            Dim::new(AxisKind::X, output_width),
            Dim::new(AxisKind::Z, output_depth),
        ],
        pixel_type: first_pixel_type,
        ..Metadata::default()
    };
    metadata.extras.insert(
        "stack_concatenate_titles".to_string(),
        json!(datasets.iter().map(|(title, _)| *title).collect::<Vec<_>>()),
    );
    metadata
        .extras
        .insert("stack_concatenate_fill".to_string(), json!(fill));
    Ok(Dataset::new(data, metadata)
        .map_err(|error| format!("failed to create concatenated stack: {error}"))?)
}

fn copy_stack_into_z_offset(
    output: &mut [f32],
    output_width: usize,
    output_height: usize,
    output_depth: usize,
    dataset: &DatasetF32,
    z_offset: usize,
) -> Result<(), String> {
    let (width, height, depth) = stack_xyz_extent(dataset)?;
    for z in 0..depth {
        let slice = extract_slice(dataset, z, 0, 0)?;
        for y in 0..height {
            for x in 0..width {
                let out_z = z + z_offset;
                if out_z < output_depth && x < output_width && y < output_height {
                    let index = (y * output_width + x) * output_depth + out_z;
                    output[index] = slice.values[y * width + x];
                }
            }
        }
    }
    Ok(())
}

fn insert_stack_dataset(
    source: &DatasetF32,
    destination: &DatasetF32,
    x_offset: isize,
    y_offset: isize,
) -> Result<DatasetF32, String> {
    ensure_xyz_only(source, "source")?;
    ensure_xyz_only(destination, "destination")?;
    if source.metadata.pixel_type != destination.metadata.pixel_type {
        return Err(
            "Insert requires source and destination to have matching pixel types".to_string(),
        );
    }

    let (source_width, source_height, source_depth) = stack_xyz_extent(source)?;
    let (dest_width, dest_height, dest_depth) = stack_xyz_extent(destination)?;
    let dest_x_axis = destination
        .axis_index(AxisKind::X)
        .ok_or_else(|| "Insert destination requires an X axis".to_string())?;
    let dest_y_axis = destination
        .axis_index(AxisKind::Y)
        .ok_or_else(|| "Insert destination requires a Y axis".to_string())?;
    let dest_z_axis = destination.axis_index(AxisKind::Z);
    let mut data = destination.data.clone();

    for dest_z in 0..dest_depth {
        let source_z = dest_z.min(source_depth.saturating_sub(1));
        let source_slice = extract_slice(source, source_z, 0, 0)?;
        for sy in 0..source_height {
            let dy = sy as isize + y_offset;
            if !(0..dest_height as isize).contains(&dy) {
                continue;
            }
            for sx in 0..source_width {
                let dx = sx as isize + x_offset;
                if !(0..dest_width as isize).contains(&dx) {
                    continue;
                }
                let mut coord = vec![0usize; destination.ndim()];
                coord[dest_y_axis] = dy as usize;
                coord[dest_x_axis] = dx as usize;
                if let Some(axis) = dest_z_axis {
                    coord[axis] = dest_z;
                }
                data[IxDyn(&coord)] = source_slice.values[sy * source_width + sx];
            }
        }
    }

    let mut metadata = destination.metadata.clone();
    metadata.extras.insert(
        "stack_insert_offset".to_string(),
        json!([x_offset, y_offset]),
    );
    Ok(Dataset::new(data, metadata)
        .map_err(|error| format!("failed to create inserted stack: {error}"))?)
}

fn ensure_xyz_only(dataset: &DatasetF32, label: &str) -> Result<(), String> {
    if dataset.axis_index(AxisKind::Channel).is_some()
        || dataset.axis_index(AxisKind::Time).is_some()
    {
        return Err(format!("Insert {label} must be an X/Y/Z image"));
    }
    Ok(())
}

fn stack_measurement_rows(
    dataset: &DatasetF32,
    settings: &MeasurementSettings,
    bbox: Option<(usize, usize, usize, usize)>,
    channel: usize,
    time: usize,
) -> Result<Vec<BTreeMap<String, Value>>, String> {
    let z_axis = dataset
        .axis_index(AxisKind::Z)
        .ok_or_else(|| "Measure Stack requires a Z stack".to_string())?;
    let slices = dataset.shape()[z_axis];
    if slices <= 1 {
        return Err("Measure Stack requires more than one Z slice".to_string());
    }

    let mut rows = Vec::with_capacity(slices);
    for z in 0..slices {
        let slice = extract_slice(dataset, z, time, channel)?;
        let (values, width, height) = if let Some((min_x, min_y, max_x, max_y)) =
            clamped_bbox(slice.width, slice.height, bbox)
        {
            let width = max_x.saturating_sub(min_x).saturating_add(1);
            let height = max_y.saturating_sub(min_y).saturating_add(1);
            let mut values = Vec::with_capacity(width * height);
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    values.push(slice.values[x + y * slice.width]);
                }
            }
            (values, width, height)
        } else {
            (slice.values, slice.width, slice.height)
        };
        rows.push(measurement_row_from_slice(
            &values, width, height, bbox, settings, z, time, channel,
        ));
    }
    Ok(rows)
}

fn stack_xy_profile_rows(
    dataset: &DatasetF32,
    params: &Value,
) -> Result<Vec<BTreeMap<String, Value>>, String> {
    let z_axis = dataset
        .axis_index(AxisKind::Z)
        .ok_or_else(|| "Plot XY Profile requires a Z stack".to_string())?;
    let slices = dataset.shape()[z_axis];
    if slices <= 1 {
        return Err("Plot XY Profile requires more than one Z slice".to_string());
    }
    let base_params = params
        .as_object()
        .ok_or_else(|| "Plot XY Profile requires profile parameters".to_string())?;

    let mut rows = Vec::with_capacity(slices);
    for z in 0..slices {
        let mut slice_params = base_params.clone();
        slice_params.insert("z".to_string(), json!(z));
        let measurements = crate::commands::execute_operation(
            "measurements.profile",
            dataset,
            &Value::Object(slice_params),
        )
        .map_err(|error| error.to_string())?
        .measurements
        .ok_or_else(|| "Plot XY Profile produced no profile data".to_string())?;
        let samples = profile_samples_from_table(&measurements)
            .ok_or_else(|| "Plot XY Profile produced no profile samples".to_string())?;

        let mut row = BTreeMap::new();
        row.insert("slice".to_string(), json!(z));
        row.insert(
            "profile_axis".to_string(),
            measurements
                .values
                .get("profile_axis")
                .cloned()
                .unwrap_or_else(|| json!("profile")),
        );
        row.insert("sample_count".to_string(), json!(samples.len()));
        for (index, sample) in samples.into_iter().enumerate() {
            row.insert(format!("P{index}"), json!(sample));
        }
        rows.push(row);
    }
    Ok(rows)
}

fn stack_slice_label(dataset: &DatasetF32, slice: usize) -> Option<String> {
    let labels = stack_slice_labels(dataset, stack_slice_count(dataset));
    labels.get(slice).cloned().flatten()
}

fn set_stack_slice_label_dataset(
    dataset: &DatasetF32,
    slice: usize,
    label: &str,
) -> Result<DatasetF32, String> {
    let slices = stack_slice_count(dataset);
    if slice >= slices {
        return Err(format!(
            "slice {} is outside the stack range 1-{slices}",
            slice + 1
        ));
    }

    let mut metadata = dataset.metadata.clone();
    let mut labels = stack_slice_labels(dataset, slices);
    let trimmed = label.trim();
    labels[slice] = if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    };
    write_stack_slice_labels(&mut metadata, labels);
    Dataset::new(dataset.data.clone(), metadata)
        .map_err(|error| format!("failed to set slice label: {error}"))
}

fn remove_stack_slice_labels_dataset(dataset: &DatasetF32) -> Result<DatasetF32, String> {
    let mut metadata = dataset.metadata.clone();
    metadata.extras.remove(SLICE_LABELS_KEY);
    metadata.extras.remove(CURRENT_SLICE_LABEL_KEY);
    Dataset::new(dataset.data.clone(), metadata)
        .map_err(|error| format!("failed to remove slice labels: {error}"))
}

fn stack_slice_count(dataset: &DatasetF32) -> usize {
    dataset
        .axis_index(AxisKind::Z)
        .map(|axis| dataset.shape()[axis])
        .unwrap_or(1)
}

fn stack_slice_labels(dataset: &DatasetF32, slices: usize) -> Vec<Option<String>> {
    let mut labels = vec![None; slices];
    if let Some(array) = dataset
        .metadata
        .extras
        .get(SLICE_LABELS_KEY)
        .and_then(Value::as_array)
    {
        for (index, value) in array.iter().take(slices).enumerate() {
            labels[index] = value
                .as_str()
                .filter(|label| !label.is_empty())
                .map(str::to_string);
        }
    } else if slices == 1 {
        labels[0] = dataset
            .metadata
            .extras
            .get(CURRENT_SLICE_LABEL_KEY)
            .and_then(Value::as_str)
            .filter(|label| !label.is_empty())
            .map(str::to_string);
    }
    labels
}

fn write_stack_slice_labels(metadata: &mut Metadata, labels: Vec<Option<String>>) {
    metadata.extras.remove(CURRENT_SLICE_LABEL_KEY);
    if labels.iter().all(Option::is_none) {
        metadata.extras.remove(SLICE_LABELS_KEY);
        return;
    }
    if labels.len() == 1 {
        if let Some(label) = labels.into_iter().next().flatten() {
            metadata
                .extras
                .insert(CURRENT_SLICE_LABEL_KEY.to_string(), json!(label));
        }
        metadata.extras.remove(SLICE_LABELS_KEY);
        return;
    }
    metadata.extras.insert(
        SLICE_LABELS_KEY.to_string(),
        Value::Array(
            labels
                .into_iter()
                .map(|label| label.map(Value::String).unwrap_or(Value::Null))
                .collect(),
        ),
    );
}

fn stack_position_from_params(
    params: &Value,
    current_channel: usize,
    current_z: usize,
    current_t: usize,
    channels: usize,
    z_slices: usize,
    times: usize,
) -> Result<(usize, usize, usize), String> {
    if channels == 0 || z_slices == 0 || times == 0 {
        return Err("set slice requires a non-empty image".to_string());
    }

    if !has_stack_position_param(params) {
        return Err("set slice requires `slice`, `channel` or `frame`".to_string());
    }

    let channel = one_based_index_param(params, &["channel", "c"], channels, "channel")?;
    let z = one_based_index_param(params, &["slice", "z"], z_slices, "slice")?;
    let t = one_based_index_param(params, &["frame", "time", "t"], times, "frame")?;

    Ok((
        channel.unwrap_or_else(|| current_channel.min(channels - 1)),
        z.unwrap_or_else(|| current_z.min(z_slices - 1)),
        t.unwrap_or_else(|| current_t.min(times - 1)),
    ))
}

fn has_stack_position_param(params: &Value) -> bool {
    ["channel", "c", "slice", "z", "frame", "time", "t"]
        .iter()
        .any(|key| params.get(*key).is_some_and(|value| !value.is_null()))
}

fn zoom_set_params(params: &Value) -> Result<(f32, Option<f32>, Option<f32>), String> {
    let Some(percent) = optional_f32_param_any(params, &["zoom_percent", "zoom", "percent"])?
    else {
        return Err("set zoom requires `zoom_percent`".to_string());
    };
    if percent <= 0.0 {
        return Err("`zoom_percent` must be positive".to_string());
    }
    let magnification = (percent / 100.0).clamp(
        interaction::transform::MIN_MAGNIFICATION,
        interaction::transform::MAX_MAGNIFICATION,
    );
    let x = optional_f32_param_any(params, &["x", "x_center"])?;
    let y = optional_f32_param_any(params, &["y", "y_center"])?;
    if x.is_some() != y.is_some() {
        return Err("set zoom requires both `x` and `y` center coordinates".to_string());
    }
    Ok((magnification, x, y))
}

fn has_zoom_set_param(params: &Value) -> bool {
    [
        "zoom_percent",
        "zoom",
        "percent",
        "x",
        "x_center",
        "y",
        "y_center",
    ]
    .iter()
    .any(|key| params.get(*key).is_some_and(|value| !value.is_null()))
}

fn optional_f32_param_any(params: &Value, keys: &[&str]) -> Result<Option<f32>, String> {
    for key in keys {
        let Some(value) = params.get(*key) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        let Some(value) = value.as_f64() else {
            return Err(format!("`{key}` must be a number"));
        };
        if !value.is_finite() {
            return Err(format!("`{key}` must be finite"));
        }
        return Ok(Some(value as f32));
    }
    Ok(None)
}

fn one_based_index_param(
    params: &Value,
    keys: &[&str],
    max: usize,
    label: &str,
) -> Result<Option<usize>, String> {
    for key in keys {
        let Some(value) = params.get(*key) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        let Some(index) = value.as_u64() else {
            return Err(format!("`{label}` must be a positive integer"));
        };
        if index == 0 || index as usize > max {
            return Err(format!("`{label}` must be between 1 and {max}"));
        }
        return Ok(Some(index as usize - 1));
    }
    Ok(None)
}

fn roi_kind_bbox(kind: &RoiKind) -> Option<(usize, usize, usize, usize)> {
    fn bounds(points: &[egui::Pos2]) -> Option<(usize, usize, usize, usize)> {
        let first = points.first()?;
        let (mut min_x, mut max_x) = (first.x, first.x);
        let (mut min_y, mut max_y) = (first.y, first.y);
        for point in &points[1..] {
            min_x = min_x.min(point.x);
            max_x = max_x.max(point.x);
            min_y = min_y.min(point.y);
            max_y = max_y.max(point.y);
        }
        Some((
            min_x.floor().max(0.0) as usize,
            min_y.floor().max(0.0) as usize,
            max_x.ceil().max(0.0) as usize,
            max_y.ceil().max(0.0) as usize,
        ))
    }

    match kind {
        RoiKind::Rect { start, end, .. }
        | RoiKind::Oval { start, end, .. }
        | RoiKind::Line { start, end, .. } => bounds(&[*start, *end]),
        RoiKind::Polygon { points, .. }
        | RoiKind::Freehand { points }
        | RoiKind::WandTrace { points }
        | RoiKind::Point { points, .. } => bounds(points),
        RoiKind::Angle { a, b, c } => bounds(&[*a, *b, *c]),
        RoiKind::Text { at, .. } => bounds(&[*at]),
    }
}

fn add_selection_to_overlay(rois: &mut RoiStore) -> Result<&'static str, String> {
    if rois.active_roi.is_some() {
        rois.commit_active(true);
        return Ok("selection added to overlay");
    }

    if let Some(selected) = rois.selected_roi_id {
        if let Some(roi) = rois.overlay_rois.iter_mut().find(|roi| roi.id == selected) {
            roi.visible = true;
            return Ok("selection already in overlay");
        }
    }

    Err("a selection is required for overlay add".to_string())
}

fn overlay_from_roi_manager(rois: &mut RoiStore) -> Result<usize, String> {
    if rois.overlay_rois.is_empty() {
        return Err("ROI Manager has no elements".to_string());
    }
    for roi in &mut rois.overlay_rois {
        roi.visible = true;
    }
    if rois.selected_roi_id.is_none() {
        rois.selected_roi_id = rois.overlay_rois.first().map(|roi| roi.id);
    }
    Ok(rois.overlay_rois.len())
}

fn overlay_to_roi_manager(rois: &mut RoiStore) -> Result<usize, String> {
    if rois.overlay_rois.is_empty() {
        return Err("no overlay elements".to_string());
    }
    if rois
        .selected_roi_id
        .is_none_or(|id| rois.overlay_rois.iter().all(|roi| roi.id != id))
    {
        rois.selected_roi_id = rois.overlay_rois.first().map(|roi| roi.id);
    }
    Ok(rois.overlay_rois.len())
}

fn flatten_overlay_slice(slice: &SliceImage, rois: &[RoiModel]) -> Result<DatasetF32, String> {
    if rois.is_empty() {
        return Err("overlay required".to_string());
    }
    let mut values = slice.values.clone();
    let (min, max) = min_max(&values);
    let draw_value = if max > min { max } else { 1.0 };
    for roi in rois {
        rasterize_roi_outline(
            &mut values,
            slice.width,
            slice.height,
            &roi.kind,
            draw_value,
        );
    }
    let data = ArrayD::from_shape_vec(IxDyn(&[slice.height, slice.width]), values)
        .map_err(|error| format!("flattened overlay shape error: {error}"))?;
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, slice.height),
            Dim::new(AxisKind::X, slice.width),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    Dataset::new(data, metadata).map_err(|error| error.to_string())
}

fn rasterize_roi_outline(
    values: &mut [f32],
    width: usize,
    height: usize,
    kind: &RoiKind,
    draw_value: f32,
) {
    if width == 0 || height == 0 {
        return;
    }
    match kind {
        RoiKind::Rect { .. } => {
            if let Some((min_x, min_y, max_x, max_y)) = roi_kind_bbox(kind) {
                let max_x = max_x.min(width.saturating_sub(1));
                let max_y = max_y.min(height.saturating_sub(1));
                for x in min_x.min(width)..=max_x {
                    put_flatten_pixel(
                        values,
                        width,
                        height,
                        x as isize,
                        min_y as isize,
                        draw_value,
                    );
                    put_flatten_pixel(
                        values,
                        width,
                        height,
                        x as isize,
                        max_y as isize,
                        draw_value,
                    );
                }
                for y in min_y.min(height)..=max_y {
                    put_flatten_pixel(
                        values,
                        width,
                        height,
                        min_x as isize,
                        y as isize,
                        draw_value,
                    );
                    put_flatten_pixel(
                        values,
                        width,
                        height,
                        max_x as isize,
                        y as isize,
                        draw_value,
                    );
                }
            }
        }
        RoiKind::Oval { start, end, .. } => {
            let rect = egui::Rect::from_two_pos(*start, *end);
            let center = rect.center();
            let radius = rect.width().abs().min(rect.height().abs()) * 0.5;
            let steps = ((radius * std::f32::consts::TAU).ceil() as usize).max(16);
            for step in 0..steps {
                let theta = step as f32 / steps as f32 * std::f32::consts::TAU;
                let x = center.x + radius * theta.cos();
                let y = center.y + radius * theta.sin();
                put_flatten_pixel(
                    values,
                    width,
                    height,
                    x.round() as isize,
                    y.round() as isize,
                    draw_value,
                );
            }
        }
        RoiKind::Line { start, end, .. } => {
            draw_flatten_line(values, width, height, *start, *end, draw_value);
        }
        RoiKind::Polygon {
            points,
            closed,
            spline_fit,
        } => {
            let display_points = if *spline_fit {
                spline_fit_roi_points(points, *closed)
            } else {
                points.clone()
            };
            for pair in display_points.windows(2) {
                draw_flatten_line(values, width, height, pair[0], pair[1], draw_value);
            }
            if *closed && display_points.len() > 2 {
                draw_flatten_line(
                    values,
                    width,
                    height,
                    *display_points.last().unwrap(),
                    display_points[0],
                    draw_value,
                );
            }
        }
        RoiKind::Freehand { points } | RoiKind::WandTrace { points } => {
            for pair in points.windows(2) {
                draw_flatten_line(values, width, height, pair[0], pair[1], draw_value);
            }
        }
        RoiKind::Angle { a, b, c } => {
            draw_flatten_line(values, width, height, *a, *b, draw_value);
            draw_flatten_line(values, width, height, *b, *c, draw_value);
        }
        RoiKind::Point { points, .. } => {
            for point in points {
                let x = point.x.round() as isize;
                let y = point.y.round() as isize;
                put_flatten_pixel(values, width, height, x, y, draw_value);
                put_flatten_pixel(values, width, height, x - 1, y, draw_value);
                put_flatten_pixel(values, width, height, x + 1, y, draw_value);
                put_flatten_pixel(values, width, height, x, y - 1, draw_value);
                put_flatten_pixel(values, width, height, x, y + 1, draw_value);
            }
        }
        RoiKind::Text { at, .. } => {
            put_flatten_pixel(
                values,
                width,
                height,
                at.x.round() as isize,
                at.y.round() as isize,
                draw_value,
            );
        }
    }
}

fn draw_flatten_line(
    values: &mut [f32],
    width: usize,
    height: usize,
    start: egui::Pos2,
    end: egui::Pos2,
    draw_value: f32,
) {
    let mut x0 = start.x.round() as isize;
    let mut y0 = start.y.round() as isize;
    let x1 = end.x.round() as isize;
    let y1 = end.y.round() as isize;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        put_flatten_pixel(values, width, height, x0, y0, draw_value);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = error * 2;
        if e2 >= dy {
            error += dy;
            x0 += sx;
        }
        if e2 <= dx {
            error += dx;
            y0 += sy;
        }
    }
}

fn put_flatten_pixel(
    values: &mut [f32],
    width: usize,
    height: usize,
    x: isize,
    y: isize,
    draw_value: f32,
) {
    if x < 0 || y < 0 {
        return;
    }
    let (x, y) = (x as usize, y as usize);
    if x >= width || y >= height {
        return;
    }
    values[x + y * width] = draw_value;
}

fn overlay_label_for_roi(
    roi: &RoiModel,
    index: usize,
    settings: &OverlaySettings,
) -> Option<(egui::Pos2, String)> {
    if !settings.show_labels {
        return None;
    }
    let anchor = roi_label_anchor(&roi.kind)?;
    let text = if settings.use_names_as_labels {
        roi.name.clone()
    } else {
        (index + 1).to_string()
    };
    Some((anchor, text))
}

fn roi_kind_name(kind: &RoiKind) -> &'static str {
    match kind {
        RoiKind::Rect { .. } => "Rectangle",
        RoiKind::Oval { .. } => "Oval",
        RoiKind::Polygon { .. } => "Polygon",
        RoiKind::Freehand { .. } => "Freehand",
        RoiKind::Line { .. } => "Line",
        RoiKind::Angle { .. } => "Angle",
        RoiKind::Point { .. } => "Point",
        RoiKind::Text { .. } => "Text",
        RoiKind::WandTrace { .. } => "Wand",
    }
}

fn apply_overlay_visibility(
    rois: &mut [RoiModel],
    mode: OverlayVisibility,
) -> Result<usize, String> {
    if rois.is_empty() {
        return Err("no overlay elements".to_string());
    }
    let visible = match mode {
        OverlayVisibility::Show => true,
        OverlayVisibility::Hide => false,
        OverlayVisibility::Toggle => !rois.iter().any(|roi| roi.visible),
    };
    for roi in rois.iter_mut() {
        roi.visible = visible;
    }
    Ok(rois.len())
}

fn overlay_element_rows(rois: &[RoiModel]) -> Result<Vec<BTreeMap<String, Value>>, String> {
    if rois.is_empty() {
        return Err("no overlay elements".to_string());
    }
    Ok(rois
        .iter()
        .enumerate()
        .map(|(index, roi)| {
            let bbox = roi_kind_bbox(&roi.kind);
            let mut row = BTreeMap::new();
            row.insert("Index".to_string(), json!(index + 1));
            row.insert("Name".to_string(), json!(roi.name));
            row.insert("Type".to_string(), json!(roi_kind_name(&roi.kind)));
            if let Some((min_x, min_y, max_x, max_y)) = bbox {
                row.insert("X".to_string(), json!(min_x));
                row.insert("Y".to_string(), json!(min_y));
                row.insert("Width".to_string(), json!(max_x - min_x + 1));
                row.insert("Height".to_string(), json!(max_y - min_y + 1));
            }
            row.insert("Channel".to_string(), json!(roi.position.channel + 1));
            row.insert("Slice".to_string(), json!(roi.position.z + 1));
            row.insert("Frame".to_string(), json!(roi.position.t + 1));
            row.insert("Visible".to_string(), json!(roi.visible));
            row.insert("Locked".to_string(), json!(roi.locked));
            row
        })
        .collect())
}

fn selection_properties_row(roi: &RoiModel) -> Result<BTreeMap<String, Value>, String> {
    let mut row = BTreeMap::new();
    row.insert("Name".to_string(), json!(roi.name));
    row.insert("Type".to_string(), json!(roi_kind_name(&roi.kind)));
    let Some((min_x, min_y, max_x, max_y)) = roi_kind_bbox(&roi.kind) else {
        return Err("selection properties require a valid selection".to_string());
    };
    row.insert("X".to_string(), json!(min_x));
    row.insert("Y".to_string(), json!(min_y));
    row.insert("Width".to_string(), json!(max_x - min_x + 1));
    row.insert("Height".to_string(), json!(max_y - min_y + 1));
    row.insert("Channel".to_string(), json!(roi.position.channel + 1));
    row.insert("Slice".to_string(), json!(roi.position.z + 1));
    row.insert("Frame".to_string(), json!(roi.position.t + 1));
    row.insert("Visible".to_string(), json!(roi.visible));
    row.insert("Locked".to_string(), json!(roi.locked));
    Ok(row)
}

fn threshold_slice_otsu(values: &[f32]) -> Vec<u8> {
    let (min, max) = min_max(values);
    let span = (max - min).max(f32::EPSILON);
    let mut histogram = [0_u64; 256];
    for value in values {
        let normalized = ((*value - min) / span).clamp(0.0, 1.0);
        histogram[(normalized * 255.0).round() as usize] += 1;
    }
    let total = values.len() as f64;
    let mut weighted_sum = 0.0;
    for (index, count) in histogram.iter().enumerate() {
        weighted_sum += index as f64 * *count as f64;
    }
    let mut sum_background = 0.0;
    let mut weight_background = 0.0;
    let mut best_threshold = 0usize;
    let mut best_variance = -1.0_f64;
    for (index, count) in histogram.iter().enumerate() {
        weight_background += *count as f64;
        if weight_background == 0.0 {
            continue;
        }
        let weight_foreground = total - weight_background;
        if weight_foreground == 0.0 {
            break;
        }
        sum_background += index as f64 * *count as f64;
        let mean_background = sum_background / weight_background;
        let mean_foreground = (weighted_sum - sum_background) / weight_foreground;
        let variance =
            weight_background * weight_foreground * (mean_background - mean_foreground).powi(2);
        if variance > best_variance {
            best_variance = variance;
            best_threshold = index;
        }
    }
    let threshold = (best_threshold as f32 / 255.0) * span + min;
    values
        .iter()
        .map(|value| if *value >= threshold { 1 } else { 0 })
        .collect()
}

#[derive(Debug, Clone)]
struct Particle {
    label: usize,
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    values: Vec<f32>,
}

impl Particle {
    fn width(&self) -> usize {
        self.max_x.saturating_sub(self.min_x) + 1
    }

    fn height(&self) -> usize {
        self.max_y.saturating_sub(self.min_y) + 1
    }
}

fn connected_components_2d(width: usize, height: usize, binary: &[u8]) -> Vec<Particle> {
    let mut visited = vec![false; binary.len()];
    let mut particles = Vec::new();
    let mut label = 0usize;
    for y in 0..height {
        for x in 0..width {
            let idx = x + y * width;
            if visited[idx] || binary[idx] == 0 {
                continue;
            }
            label += 1;
            let mut queue = std::collections::VecDeque::from([(x, y)]);
            visited[idx] = true;
            let mut points = Vec::new();
            let (mut min_x, mut min_y, mut max_x, mut max_y) = (x, y, x, y);
            while let Some((cx, cy)) = queue.pop_front() {
                points.push((cx, cy));
                min_x = min_x.min(cx);
                min_y = min_y.min(cy);
                max_x = max_x.max(cx);
                max_y = max_y.max(cy);
                for (nx, ny) in [
                    (cx.wrapping_sub(1), cy),
                    (cx + 1, cy),
                    (cx, cy.wrapping_sub(1)),
                    (cx, cy + 1),
                ] {
                    if nx >= width || ny >= height {
                        continue;
                    }
                    let nidx = nx + ny * width;
                    if visited[nidx] || binary[nidx] == 0 {
                        continue;
                    }
                    visited[nidx] = true;
                    queue.push_back((nx, ny));
                }
            }
            let particle_width = max_x - min_x + 1;
            let particle_height = max_y - min_y + 1;
            let mut values = vec![0.0_f32; particle_width * particle_height];
            for (px, py) in points {
                values[(px - min_x) + (py - min_y) * particle_width] = 1.0;
            }
            particles.push(Particle {
                label,
                min_x,
                min_y,
                max_x,
                max_y,
                values,
            });
        }
    }
    particles
}

fn measurement_row_from_slice(
    values: &[f32],
    width: usize,
    height: usize,
    bbox: Option<(usize, usize, usize, usize)>,
    settings: &MeasurementSettings,
    z: usize,
    t: usize,
    channel: usize,
) -> BTreeMap<String, Value> {
    let mut row = BTreeMap::new();
    let (min, max) = min_max(values);
    let pixel_count = values.len().max(1);
    let sum = values.iter().sum::<f32>();
    let mean = sum / pixel_count as f32;
    let mut area = 0usize;
    let mut centroid_x = 0.0_f32;
    let mut centroid_y = 0.0_f32;
    for y in 0..height {
        for x in 0..width {
            let value = values[x + y * width];
            if value > 0.0 {
                area += 1;
                centroid_x += x as f32;
                centroid_y += y as f32;
            }
        }
    }
    if settings.area {
        row.insert("area".into(), json!(area));
    }
    if settings.min {
        row.insert("min".into(), json!(min));
    }
    if settings.max {
        row.insert("max".into(), json!(max));
    }
    if settings.mean {
        row.insert("mean".into(), json!(mean));
    }
    if settings.integrated_density {
        row.insert("integrated_density".into(), json!(sum));
    }
    if settings.centroid {
        let denom = area.max(1) as f32;
        row.insert(
            "centroid".into(),
            json!([centroid_x / denom, centroid_y / denom]),
        );
    }
    if settings.bbox {
        if let Some((min_x, min_y, max_x, max_y)) = bbox {
            row.insert("bbox".into(), json!([min_x, min_y, max_x, max_y]));
        }
    }
    if settings.slice {
        row.insert("slice".into(), json!(z));
    }
    if settings.time {
        row.insert("time".into(), json!(t));
    }
    if settings.channel {
        row.insert("channel".into(), json!(channel));
    }
    row
}

fn image_slice_to_results_rows(
    slice: &SliceImage,
    bbox: Option<(usize, usize, usize, usize)>,
) -> Result<Vec<BTreeMap<String, Value>>, String> {
    let (min_x, min_y, max_x, max_y) = clamped_bbox(slice.width, slice.height, bbox)
        .ok_or_else(|| "image slice is empty".to_string())?;
    let mut rows = Vec::with_capacity(max_y - min_y + 1);
    for y in min_y..=max_y {
        let mut row = BTreeMap::new();
        row.insert("Label".to_string(), json!(format!("Y{y}")));
        for x in min_x..=max_x {
            row.insert(format!("X{x}"), json!(slice.values[x + y * slice.width]));
        }
        rows.push(row);
    }
    Ok(rows)
}

fn xy_coordinate_rows(
    slice: &SliceImage,
    bbox: Option<(usize, usize, usize, usize)>,
    params: &Value,
) -> Result<Vec<BTreeMap<String, Value>>, String> {
    let (min_x, min_y, max_x, max_y) = clamped_bbox(slice.width, slice.height, bbox)
        .ok_or_else(|| "image slice is empty".to_string())?;
    let has_selection = bbox.is_some();
    let background = params
        .get("background")
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .unwrap_or_else(|| slice.values.first().copied().unwrap_or(f32::NAN));
    let invert_y = params
        .get("invert_y")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut rows = Vec::new();

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let value = slice.values[x + y * slice.width];
            let matches_background = if background.is_nan() {
                value.is_nan()
            } else {
                value == background
            };
            if !has_selection && matches_background {
                continue;
            }

            let output_y = if invert_y { y } else { slice.height - 1 - y };
            let mut row = BTreeMap::new();
            row.insert("X".to_string(), json!(x));
            row.insert("Y".to_string(), json!(output_y));
            row.insert("Value".to_string(), json!(value));
            rows.push(row);
        }
    }

    if rows.is_empty() {
        return Err("no non-background pixels found".to_string());
    }
    Ok(rows)
}

fn results_rows_to_dataset(rows: &[BTreeMap<String, Value>]) -> Result<DatasetF32, String> {
    if rows.is_empty() {
        return Err("results table is empty".to_string());
    }
    let columns = numeric_results_columns(rows);
    if columns.is_empty() {
        return Err("results table has no numeric columns".to_string());
    }
    let width = columns.len();
    let height = rows.len();
    let mut values = Vec::with_capacity(width * height);
    for row in rows {
        for column in &columns {
            values.push(row.get(column).and_then(Value::as_f64).unwrap_or(f64::NAN) as f32);
        }
    }
    let data = ArrayD::from_shape_vec(IxDyn(&[height, width]), values)
        .map_err(|error| format!("results image shape error: {error}"))?;
    let metadata = Metadata {
        dims: vec![Dim::new(AxisKind::Y, height), Dim::new(AxisKind::X, width)],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    Dataset::new(data, metadata).map_err(|error| error.to_string())
}

fn results_summary_rows(
    rows: &[BTreeMap<String, Value>],
) -> Result<Vec<BTreeMap<String, Value>>, String> {
    if rows.len() < 2 {
        return Err("at least two results rows are required to summarize".to_string());
    }
    if rows
        .last()
        .and_then(|row| row.get("Label"))
        .and_then(Value::as_str)
        == Some("Max")
    {
        return Err("results table is already summarized".to_string());
    }

    let columns = numeric_results_columns(rows);
    if columns.is_empty() {
        return Err("results table has no numeric columns".to_string());
    }

    let mut summary = ["Mean", "SD", "Min", "Max"]
        .into_iter()
        .map(|label| {
            let mut row = BTreeMap::new();
            row.insert("Label".to_string(), json!(label));
            row
        })
        .collect::<Vec<_>>();

    for column in columns {
        let values = rows
            .iter()
            .filter_map(|row| row.get(&column).and_then(Value::as_f64))
            .collect::<Vec<_>>();
        if values.len() < 2 {
            continue;
        }
        let count = values.len() as f64;
        let sum = values.iter().sum::<f64>();
        let sum2 = values.iter().map(|value| value * value).sum::<f64>();
        let mean = sum / count;
        let variance = ((sum2 - sum * sum / count) / (count - 1.0)).max(0.0);
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        summary[0].insert(column.clone(), json!(mean));
        summary[1].insert(column.clone(), json!(variance.sqrt()));
        summary[2].insert(column.clone(), json!(min));
        summary[3].insert(column, json!(max));
    }

    Ok(summary)
}

fn results_distribution(
    rows: &[BTreeMap<String, Value>],
    column: Option<&str>,
    bins: usize,
) -> Result<Value, String> {
    if bins == 0 {
        return Err("distribution bins must be > 0".to_string());
    }
    let columns = numeric_results_columns(rows);
    let column = match column {
        Some(column) if columns.iter().any(|candidate| candidate == column) => column.to_string(),
        Some(column) => return Err(format!("results table has no numeric column `{column}`")),
        None => columns
            .first()
            .cloned()
            .ok_or_else(|| "results table has no numeric columns".to_string())?,
    };
    let values = rows
        .iter()
        .filter_map(|row| row.get(&column).and_then(Value::as_f64))
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return Err(format!("results column `{column}` has no finite values"));
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (max - min).max(f64::EPSILON);
    let mut counts = vec![0_u64; bins];
    for value in &values {
        let mut bin = (((*value - min) / span) * bins as f64).floor() as usize;
        if bin >= bins {
            bin = bins - 1;
        }
        counts[bin] += 1;
    }
    let bin_width = span / bins as f64;
    let bins_payload = counts
        .iter()
        .enumerate()
        .map(|(index, count)| {
            json!({
                "min": min + bin_width * index as f64,
                "max": if index + 1 == bins { max } else { min + bin_width * (index + 1) as f64 },
                "count": count
            })
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "column": column,
        "count": values.len(),
        "min": min,
        "max": max,
        "bins": bins_payload
    }))
}

fn create_circular_masks_dataset() -> Result<DatasetF32, String> {
    const WIDTH: usize = 150;
    const HEIGHT: usize = 150;
    const SLICES: usize = 99;

    let mut values = vec![0.0_f32; WIDTH * HEIGHT * SLICES];
    for slice in 0..SLICES {
        let radius = 0.5 + slice as f32 * 0.5;
        let line_radii = circular_mask_line_radii(radius);
        let kernel_height = line_radii.len();
        let y0 = HEIGHT / 2 - kernel_height / 2;
        for (line, (left, right)) in line_radii.iter().enumerate() {
            let y = y0 + line;
            for x in (WIDTH as isize / 2 + left)..=(WIDTH as isize / 2 + right) {
                let x = x as usize;
                values[(y * WIDTH + x) * SLICES + slice] = 1.0;
            }
        }
    }

    let data = ArrayD::from_shape_vec(IxDyn(&[HEIGHT, WIDTH, SLICES]), values)
        .map_err(|error| format!("circular masks shape error: {error}"))?;
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, HEIGHT),
            Dim::new(AxisKind::X, WIDTH),
            Dim::new(AxisKind::Z, SLICES),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    Dataset::new(data, metadata).map_err(|error| error.to_string())
}

fn circular_mask_line_radii(radius: f32) -> Vec<(isize, isize)> {
    let radius = if (1.5..1.75).contains(&radius) {
        1.75
    } else if (2.5..2.85).contains(&radius) {
        2.85
    } else {
        radius
    };
    let r2 = (radius * radius) as isize + 1;
    let kernel_radius = ((r2 as f32 + 1.0e-10).sqrt()) as isize;
    let mut line_radii = Vec::with_capacity(2 * kernel_radius as usize + 1);
    for y in -kernel_radius..=kernel_radius {
        let dx = ((r2 - y * y) as f32 + 1.0e-10).sqrt() as isize;
        line_radii.push((-dx, dx));
    }
    line_radii
}

fn numeric_results_columns(rows: &[BTreeMap<String, Value>]) -> Vec<String> {
    let mut columns = Vec::new();
    for row in rows {
        for (key, value) in row {
            if key == "Label" || !value.is_number() || columns.contains(key) {
                continue;
            }
            columns.push(key.clone());
        }
    }
    columns.sort_by(
        |left, right| match (x_column_index(left), x_column_index(right)) {
            (Some(left), Some(right)) => left.cmp(&right),
            _ => left.cmp(right),
        },
    );
    columns
}

fn x_column_index(column: &str) -> Option<usize> {
    column.strip_prefix('X')?.parse().ok()
}

fn clamped_bbox(
    width: usize,
    height: usize,
    bbox: Option<(usize, usize, usize, usize)>,
) -> Option<(usize, usize, usize, usize)> {
    if width == 0 || height == 0 {
        return None;
    }
    let (min_x, min_y, max_x, max_y) =
        bbox.unwrap_or((0, 0, width.saturating_sub(1), height.saturating_sub(1)));
    let min_x = min_x.min(width - 1);
    let min_y = min_y.min(height - 1);
    let max_x = max_x.min(width - 1);
    let max_y = max_y.min(height - 1);
    if max_x < min_x || max_y < min_y {
        return None;
    }
    Some((min_x, min_y, max_x, max_y))
}

fn slice_values_in_bbox(
    slice: &SliceImage,
    bbox: Option<(usize, usize, usize, usize)>,
) -> Result<Vec<f32>, String> {
    let (min_x, min_y, max_x, max_y) = clamped_bbox(slice.width, slice.height, bbox)
        .ok_or_else(|| "image slice is empty".to_string())?;
    let mut values = Vec::with_capacity((max_x - min_x + 1) * (max_y - min_y + 1));
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            values.push(slice.values[x + y * slice.width]);
        }
    }
    Ok(values)
}

fn adjust_histogram(values: &[f32], bins: usize) -> Result<AdjustHistogram, String> {
    if bins == 0 {
        return Err("histogram bins must be > 0".to_string());
    }

    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return Err("histogram requires finite image values".to_string());
    }

    let (min, max) = min_max(&finite);
    let span = max - min;
    let mut counts = vec![0_u64; bins];
    for value in finite.drain(..) {
        let bin = if span <= f32::EPSILON {
            0
        } else {
            (((value - min) / span) * bins as f32).floor().max(0.0) as usize
        }
        .min(bins - 1);
        counts[bin] += 1;
    }

    Ok(AdjustHistogram {
        pixel_count: counts.iter().sum::<u64>() as usize,
        counts,
        min,
        max,
    })
}

fn threshold_values_dataset(values: Vec<f32>, pixel_type: PixelType) -> Result<DatasetF32, String> {
    if values.is_empty() {
        return Err("threshold requires image values".to_string());
    }
    let shape = [values.len()];
    let mut metadata = Metadata::from_shape(&shape, pixel_type);
    metadata.pixel_type = pixel_type;
    let data = ArrayD::from_shape_vec(IxDyn(&shape), values)
        .map_err(|_| "failed to build threshold sample dataset".to_string())?;
    Dataset::new(data, metadata).map_err(|error| error.to_string())
}

fn adjust_slider_bounds(dialog: &AdjustDialogState) -> (f32, f32) {
    let mut min = dialog.default_min.min(dialog.default_max);
    let mut max = dialog.default_min.max(dialog.default_max);
    if !min.is_finite() || !max.is_finite() {
        min = 0.0;
        max = 1.0;
    }
    if max <= min {
        max = min + 1.0;
    }
    (min, max)
}

fn adjust_default_range(dialog: &AdjustDialogState) -> f32 {
    let (min, max) = adjust_slider_bounds(dialog);
    (max - min).max(f32::EPSILON)
}

fn clamp_adjust_min_max(dialog: &mut AdjustDialogState) {
    let (slider_min, slider_max) = adjust_slider_bounds(dialog);
    dialog.min = dialog.min.clamp(slider_min, slider_max);
    dialog.max = dialog.max.clamp(slider_min, slider_max);
    if dialog.max < dialog.min {
        std::mem::swap(&mut dialog.min, &mut dialog.max);
    }
    if (dialog.max - dialog.min).abs() < f32::EPSILON {
        dialog.max = (dialog.min + f32::EPSILON).min(slider_max);
        if dialog.max <= dialog.min {
            dialog.min = (dialog.max - f32::EPSILON).max(slider_min);
        }
    }
}

fn sync_brightness_from_min_max(dialog: &mut AdjustDialogState) {
    let range = adjust_default_range(dialog);
    let center = dialog.min + (dialog.max - dialog.min) * 0.5;
    dialog.brightness = (1.0 - (center - dialog.default_min) / range).clamp(0.0, 1.0);
}

fn sync_contrast_from_min_max(dialog: &mut AdjustDialogState) {
    let range = adjust_default_range(dialog);
    let width = (dialog.max - dialog.min).abs().max(f32::EPSILON);
    dialog.contrast = if width <= range {
        1.0 - (width / range) * 0.5
    } else {
        (range / width) * 0.5
    }
    .clamp(0.0, 1.0);
}

fn sync_brightness_contrast_from_min_max(dialog: &mut AdjustDialogState) {
    sync_brightness_from_min_max(dialog);
    sync_contrast_from_min_max(dialog);
}

fn adjust_min_max_from_brightness(dialog: &mut AdjustDialogState) {
    let range = adjust_default_range(dialog);
    let width = (dialog.max - dialog.min).abs().max(f32::EPSILON);
    let center = dialog.default_min + range * (1.0 - dialog.brightness.clamp(0.0, 1.0));
    dialog.min = center - width * 0.5;
    dialog.max = center + width * 0.5;
}

fn adjust_min_max_from_contrast(dialog: &mut AdjustDialogState) {
    let range = adjust_default_range(dialog);
    let center = dialog.min + (dialog.max - dialog.min) * 0.5;
    let contrast = dialog.contrast.clamp(0.0, 0.999_999);
    let slope = if contrast <= 0.5 {
        contrast / 0.5
    } else {
        0.5 / (1.0 - contrast)
    };
    if slope > f32::EPSILON {
        let width = range / slope;
        dialog.min = center - width * 0.5;
        dialog.max = center + width * 0.5;
    }
}

fn auto_contrast_range_for_dialog(dialog: &mut AdjustDialogState) -> Option<(f32, f32)> {
    auto_contrast_range(
        dialog.histogram.as_ref(),
        &mut dialog.contrast_auto_threshold,
    )
}

fn auto_contrast_range(
    histogram: Option<&AdjustHistogram>,
    auto_threshold: &mut u32,
) -> Option<(f32, f32)> {
    let histogram = histogram?;
    let total = histogram.pixel_count as u64;
    if total == 0 || histogram.max <= histogram.min {
        return Some((histogram.min, histogram.max));
    }

    if *auto_threshold < 10 {
        *auto_threshold = 5_000;
    } else {
        *auto_threshold /= 2;
    }
    let limit = total / 10;
    let threshold = total / u64::from(*auto_threshold);
    let low_bin = histogram.counts.iter().position(|count| {
        let count = if *count > limit { 0 } else { *count };
        count > threshold
    })?;
    let high_bin = histogram.counts.iter().rposition(|count| {
        let count = if *count > limit { 0 } else { *count };
        count > threshold
    })?;
    if high_bin < low_bin {
        return None;
    }

    let bin_width = (histogram.max - histogram.min) / histogram.counts.len().max(1) as f32;
    let low = histogram.min + low_bin as f32 * bin_width;
    let high = histogram.min + high_bin as f32 * bin_width;
    if (high - low).abs() <= f32::EPSILON {
        Some((histogram.min, histogram.max))
    } else {
        Some((low, high))
    }
}

fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries = map.iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            let mut output = Map::new();
            for (key, value) in entries {
                output.insert(key.clone(), canonical_json(value));
            }
            Value::Object(output)
        }
        Value::Array(values) => Value::Array(values.iter().map(canonical_json).collect()),
        _ => value.clone(),
    }
}

fn threshold_method_labels() -> &'static [&'static str] {
    &[
        "Default",
        "Huang",
        "Intermodes",
        "IsoData",
        "IJ_IsoData",
        "Li",
        "MaxEntropy",
        "Mean",
        "MinError",
        "Minimum",
        "Moments",
        "Otsu",
        "Percentile",
        "RenyiEntropy",
        "Shanbhag",
        "Triangle",
        "Yen",
    ]
}

fn threshold_method_param(method: &str) -> &'static str {
    match method {
        "Huang" => "huang",
        "Intermodes" => "intermodes",
        "IsoData" => "isodata",
        "IJ_IsoData" => "ij_isodata",
        "Li" => "li",
        "MaxEntropy" => "max_entropy",
        "MinError" => "min_error",
        "Minimum" => "minimum",
        "Moments" => "moments",
        "Otsu" => "otsu",
        "Mean" => "mean",
        "Percentile" => "percentile",
        "RenyiEntropy" => "renyi_entropy",
        "Shanbhag" => "shanbhag",
        "Triangle" => "triangle",
        "Yen" => "yen",
        "Default" => "default",
        _ => "ij_isodata",
    }
}

fn threshold_background_param(dark_background: bool) -> &'static str {
    if dark_background { "dark" } else { "light" }
}

fn color_balance_channel_index(channel: &str) -> Option<usize> {
    match channel {
        "Red" | "Cyan" | "red" | "cyan" => Some(0),
        "Green" | "Magenta" | "green" | "magenta" => Some(1),
        "Blue" | "Yellow" | "blue" | "yellow" => Some(2),
        _ => channel
            .strip_prefix("Channel ")
            .and_then(|value| value.parse::<usize>().ok())
            .and_then(|value| value.checked_sub(1)),
    }
}

fn color_balance_color_labels() -> Vec<String> {
    ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "All"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn color_balance_channel_labels(channel_count: usize) -> Vec<String> {
    let mut labels = (1..=channel_count.max(1))
        .map(|channel| format!("Channel {channel}"))
        .collect::<Vec<_>>();
    if channel_count > 1 {
        labels.push("All".to_string());
    }
    labels
}

fn color_balance_channel_labels_for_session(session: &ViewerSession) -> Vec<String> {
    if color_balance_uses_lut_color(session) {
        return vec!["LUT level".to_string()];
    }
    let channel_count = session.committed_summary.channels.max(1);
    if channel_count == 3 {
        color_balance_color_labels()
    } else {
        color_balance_channel_labels(channel_count)
    }
}

fn color_balance_uses_lut_color(session: &ViewerSession) -> bool {
    session.committed_summary.channels <= 1
}

fn to_color_image_with_threshold(
    frame: &ViewerFrameBuffer,
    lut: LookupTable,
    threshold: Option<ThresholdOverlay>,
) -> egui::ColorImage {
    let Some(threshold) = threshold else {
        return to_color_image(frame, lut);
    };

    let mut rgba = Vec::with_capacity(frame.pixels_u8.len() * 4);
    for (index, gray) in frame.pixels_u8.iter().enumerate() {
        let value = frame.values.get(index).copied().unwrap_or_default();
        let in_range = value.is_finite() && value >= threshold.low && value <= threshold.high;
        let color = match threshold.mode {
            ThresholdOverlayMode::Red if in_range => egui::Color32::RED,
            ThresholdOverlayMode::BlackAndWhite => {
                if in_range {
                    egui::Color32::WHITE
                } else {
                    egui::Color32::BLACK
                }
            }
            ThresholdOverlayMode::OverUnder if value.is_finite() && value < threshold.low => {
                egui::Color32::BLUE
            }
            ThresholdOverlayMode::OverUnder if value.is_finite() && value > threshold.high => {
                egui::Color32::RED
            }
            _ => lookup_table_color(lut, *gray),
        };
        rgba.extend_from_slice(&[color.r(), color.g(), color.b(), 255]);
    }
    egui::ColorImage::from_rgba_unmultiplied([frame.width, frame.height], &rgba)
}

fn push_adjust_display_range_action(
    actions: &mut Vec<UiAction>,
    dialog: &AdjustDialogState,
    command_id: &str,
    low_key: &str,
    high_key: &str,
) {
    let mut params = Map::new();
    params.insert(low_key.to_string(), json!(dialog.min));
    params.insert(high_key.to_string(), json!(dialog.max));
    actions.push(UiAction::Command {
        window_label: dialog.window_label.clone(),
        command_id: command_id.to_string(),
        params: Some(Value::Object(params)),
    });
}

fn color_balance_params(dialog: &AdjustDialogState) -> Value {
    json!({
        "min": dialog.min,
        "max": dialog.max,
        "brightness": dialog.brightness,
        "channel": dialog.color_balance_channel,
        "log_histogram": dialog.log_histogram,
    })
}

fn threshold_histogram_bins(dialog: &AdjustDialogState) -> usize {
    if dialog.threshold_sixteen_bit_histogram {
        65_536
    } else {
        256
    }
}

fn set_threshold_sixteen_bit_histogram(dialog: &mut AdjustDialogState, enabled: bool) {
    dialog.threshold_sixteen_bit_histogram = enabled;
    if enabled {
        dialog.threshold_no_reset = true;
    }
}

fn color_threshold_params(dialog: &AdjustDialogState) -> Value {
    json!({
        "color_space": dialog.color_threshold_space,
        "method": dialog.color_threshold_method,
        "mode": dialog.color_threshold_mode,
        "dark_background": dialog.dark_background,
        "hue": {
            "min": dialog.hue_min,
            "max": dialog.hue_max,
            "pass": dialog.hue_pass,
        },
        "saturation": {
            "min": dialog.saturation_min,
            "max": dialog.saturation_max,
            "pass": dialog.saturation_pass,
        },
        "brightness": {
            "min": dialog.brightness_min,
            "max": dialog.brightness_max,
            "pass": dialog.brightness_pass,
        },
    })
}

fn color_threshold_choice_labels() -> [&'static str; 3] {
    ["Thresholding method:", "Threshold color:", "Color space:"]
}

fn reset_color_threshold_bands_for_space(dialog: &mut AdjustDialogState) {
    dialog.hue_min = 0.0;
    dialog.hue_max = 255.0;
    dialog.saturation_min = 0.0;
    dialog.saturation_max = 255.0;
    dialog.brightness_min = 0.0;
    dialog.brightness_max = 255.0;
    dialog.hue_pass = true;
    dialog.saturation_pass = true;
    dialog.brightness_pass = true;
}

fn color_threshold_space_key(space: &str) -> String {
    match space
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
        .as_str()
    {
        "cielab" | "lab" => "lab".to_string(),
        "hsb" => "hsb".to_string(),
        "rgb" => "rgb".to_string(),
        "yuv" => "yuv".to_string(),
        other => other.to_string(),
    }
}

fn mask_foreground_bbox(slice: &SliceImage) -> Option<(usize, usize, usize, usize)> {
    let mut bbox: Option<(usize, usize, usize, usize)> = None;
    for y in 0..slice.height {
        for x in 0..slice.width {
            let value = slice.values[x + y * slice.width];
            if value <= 0.0 || !value.is_finite() {
                continue;
            }
            bbox = Some(match bbox {
                Some((min_x, min_y, max_x, max_y)) => {
                    (min_x.min(x), min_y.min(y), max_x.max(x), max_y.max(y))
                }
                None => (x, y, x, y),
            });
        }
    }
    bbox
}

fn mask_foreground_roi(slice: &SliceImage) -> Option<RoiKind> {
    let outline = mask_foreground_outline_points(slice)?;
    if outline.len() >= 3 {
        Some(RoiKind::Polygon {
            points: outline,
            closed: true,
            spline_fit: false,
        })
    } else {
        let (min_x, min_y, max_x, max_y) = mask_foreground_bbox(slice)?;
        Some(RoiKind::Rect {
            start: egui::pos2(min_x as f32, min_y as f32),
            end: egui::pos2((max_x + 1) as f32, (max_y + 1) as f32),
            rounded: false,
            rotated: false,
        })
    }
}

fn mask_foreground_outline_points(slice: &SliceImage) -> Option<Vec<egui::Pos2>> {
    type Point = (i32, i32);
    type Edge = (Point, Point);

    fn foreground(slice: &SliceImage, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 || x >= slice.width as i32 || y >= slice.height as i32 {
            return false;
        }
        let value = slice.values[x as usize + y as usize * slice.width];
        value.is_finite() && value > 0.0
    }

    let mut outgoing: HashMap<Point, Vec<Point>> = HashMap::new();
    let mut unused: HashSet<Edge> = HashSet::new();
    for y in 0..slice.height as i32 {
        for x in 0..slice.width as i32 {
            if !foreground(slice, x, y) {
                continue;
            }
            for edge in [
                (!foreground(slice, x, y - 1)).then_some(((x, y), (x + 1, y))),
                (!foreground(slice, x + 1, y)).then_some(((x + 1, y), (x + 1, y + 1))),
                (!foreground(slice, x, y + 1)).then_some(((x + 1, y + 1), (x, y + 1))),
                (!foreground(slice, x - 1, y)).then_some(((x, y + 1), (x, y))),
            ]
            .into_iter()
            .flatten()
            {
                outgoing.entry(edge.0).or_default().push(edge.1);
                unused.insert(edge);
            }
        }
    }

    let mut best_loop: Vec<Point> = Vec::new();
    while let Some(edge) = unused.iter().next().copied() {
        let start = edge.0;
        let mut current = edge;
        let mut loop_points = vec![current.0, current.1];
        unused.remove(&current);

        while current.1 != start {
            let Some(next) = outgoing.get(&current.1).and_then(|targets| {
                targets
                    .iter()
                    .copied()
                    .find(|target| unused.contains(&(current.1, *target)))
            }) else {
                break;
            };
            current = (current.1, next);
            unused.remove(&current);
            loop_points.push(current.1);
        }

        if loop_points.last().copied() == Some(start) {
            loop_points.pop();
        }
        if loop_points.len() > best_loop.len() {
            best_loop = loop_points;
        }
    }

    if best_loop.len() < 3 {
        return None;
    }
    Some(
        best_loop
            .into_iter()
            .map(|(x, y)| egui::pos2(x as f32, y as f32))
            .collect(),
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ColorThresholdSampleRanges {
    ranges: [(f32, f32); 3],
    passes: [bool; 3],
}

fn sample_color_threshold_ranges_with_passes(
    dataset: &DatasetF32,
    bbox: (usize, usize, usize, usize),
    z: usize,
    t: usize,
    color_space: &str,
) -> Result<ColorThresholdSampleRanges, String> {
    let channel_axis = dataset
        .axis_index(AxisKind::Channel)
        .ok_or_else(|| "Color Threshold Sample requires an RGB channel image".to_string())?;
    if dataset.shape()[channel_axis] < 3 {
        return Err("Color Threshold Sample requires at least 3 channels".to_string());
    }
    let y_axis = dataset
        .axis_index(AxisKind::Y)
        .unwrap_or(0)
        .min(dataset.ndim().saturating_sub(1));
    let x_axis = dataset
        .axis_index(AxisKind::X)
        .unwrap_or(1.min(dataset.ndim().saturating_sub(1)));
    if y_axis == x_axis {
        return Err("Color Threshold Sample could not infer X/Y axes".to_string());
    }
    let width = dataset.shape()[x_axis];
    let height = dataset.shape()[y_axis];
    if width == 0 || height == 0 {
        return Err("Color Threshold Sample requires a non-empty image".to_string());
    }
    let (min_x, min_y, max_x, max_y) = bbox;
    let min_x = min_x.min(width - 1);
    let max_x = max_x.min(width - 1);
    let min_y = min_y.min(height - 1);
    let max_y = max_y.min(height - 1);
    if max_x < min_x || max_y < min_y {
        return Err("Color Threshold Sample selection is outside the image".to_string());
    }

    let mut mins = [f32::INFINITY; 3];
    let mut maxes = [f32::NEG_INFINITY; 3];
    let mut hue_bin = [false; 256];
    let mut index = vec![0usize; dataset.ndim()];
    if let Some(z_axis) = dataset.axis_index(AxisKind::Z) {
        index[z_axis] = z.min(dataset.shape()[z_axis].saturating_sub(1));
    }
    if let Some(t_axis) = dataset.axis_index(AxisKind::Time) {
        index[t_axis] = t.min(dataset.shape()[t_axis].saturating_sub(1));
    }
    let color_space = color_threshold_space_key(color_space);
    for y in min_y..=max_y {
        index[y_axis] = y;
        for x in min_x..=max_x {
            index[x_axis] = x;
            let components =
                sampled_color_components(dataset, &mut index, channel_axis, &color_space)?;
            if color_space == "hsb" {
                let hue = components[0].round().clamp(0.0, 255.0) as usize;
                hue_bin[hue] = true;
            }
            for band in 0..3 {
                mins[band] = mins[band].min(components[band]);
                maxes[band] = maxes[band].max(components[band]);
            }
        }
    }

    if mins.iter().any(|value| !value.is_finite()) || maxes.iter().any(|value| !value.is_finite()) {
        return Err("Color Threshold Sample found no finite RGB pixels".to_string());
    }

    let mut sample = ColorThresholdSampleRanges {
        ranges: [
            (
                mins[0].floor().clamp(0.0, 255.0),
                maxes[0].ceil().clamp(0.0, 255.0),
            ),
            (
                mins[1].floor().clamp(0.0, 255.0),
                maxes[1].ceil().clamp(0.0, 255.0),
            ),
            (
                mins[2].floor().clamp(0.0, 255.0),
                maxes[2].ceil().clamp(0.0, 255.0),
            ),
        ],
        passes: [true, true, true],
    };
    if color_space == "hsb" {
        apply_imagej_hsb_sample_hue_gap_rule(&mut sample, &hue_bin);
    }
    Ok(sample)
}

fn apply_imagej_hsb_sample_hue_gap_rule(
    sample: &mut ColorThresholdSampleRanges,
    hue_bin: &[bool; 256],
) {
    let mut gap = 0usize;
    let mut max_gap = 0usize;
    let mut max_gap_start = 0usize;
    let mut max_gap_end = 0usize;
    let mut gap_start = 0usize;
    if !hue_bin[0] {
        gap_start = 0;
        gap = 1;
    }
    for hue in 1..256 {
        if !hue_bin[hue] {
            if hue_bin[hue - 1] {
                gap = 1;
                gap_start = hue;
            } else {
                gap += 1;
            }
            if gap > max_gap {
                max_gap = gap;
                max_gap_start = gap_start;
                max_gap_end = hue;
            }
        }
    }
    let Some(range_pass_low) = hue_bin.iter().position(|occupied| *occupied) else {
        return;
    };
    let Some(range_pass_high) = hue_bin.iter().rposition(|occupied| *occupied) else {
        return;
    };
    if range_pass_high.saturating_sub(range_pass_low) < max_gap {
        sample.passes[0] = true;
        sample.ranges[0] = (range_pass_low as f32, range_pass_high as f32);
    } else {
        sample.passes[0] = false;
        sample.ranges[0] = (max_gap_start as f32, max_gap_end as f32);
    }
}

fn color_threshold_auto_ranges(
    dataset: &DatasetF32,
    z: usize,
    t: usize,
    color_space: &str,
    method: &str,
    background: &str,
    app: &AppContext,
) -> Result<[(f32, f32); 3], String> {
    let values = color_threshold_component_values(dataset, None, z, t, color_space)?;
    let mut ranges = [(0.0, 255.0); 3];
    let color_space = color_threshold_space_key(color_space);
    let active_bands = match color_space.as_str() {
        "rgb" => [true, true, true],
        "lab" | "yuv" => [true, false, false],
        _ => [false, false, true],
    };
    for band in 0..3 {
        if !active_bands[band] {
            continue;
        }
        ranges[band] = auto_threshold_range_for_values(&values[band], method, background, app)?;
    }
    Ok(ranges)
}

fn color_threshold_component_values(
    dataset: &DatasetF32,
    bbox: Option<(usize, usize, usize, usize)>,
    z: usize,
    t: usize,
    color_space: &str,
) -> Result<[Vec<f32>; 3], String> {
    let channel_axis = dataset
        .axis_index(AxisKind::Channel)
        .ok_or_else(|| "Color Threshold requires an RGB channel image".to_string())?;
    if dataset.shape()[channel_axis] < 3 {
        return Err("Color Threshold requires at least 3 channels".to_string());
    }
    let y_axis = dataset
        .axis_index(AxisKind::Y)
        .unwrap_or(0)
        .min(dataset.ndim().saturating_sub(1));
    let x_axis = dataset
        .axis_index(AxisKind::X)
        .unwrap_or(1.min(dataset.ndim().saturating_sub(1)));
    if y_axis == x_axis {
        return Err("Color Threshold could not infer X/Y axes".to_string());
    }
    let width = dataset.shape()[x_axis];
    let height = dataset.shape()[y_axis];
    if width == 0 || height == 0 {
        return Err("Color Threshold requires a non-empty image".to_string());
    }

    let (min_x, min_y, max_x, max_y) = bbox.unwrap_or((0, 0, width - 1, height - 1));
    let min_x = min_x.min(width - 1);
    let max_x = max_x.min(width - 1);
    let min_y = min_y.min(height - 1);
    let max_y = max_y.min(height - 1);
    if max_x < min_x || max_y < min_y {
        return Err("Color Threshold selection is outside the image".to_string());
    }

    let capacity = (max_x - min_x + 1) * (max_y - min_y + 1);
    let mut values = [
        Vec::with_capacity(capacity),
        Vec::with_capacity(capacity),
        Vec::with_capacity(capacity),
    ];
    let mut index = vec![0usize; dataset.ndim()];
    if let Some(z_axis) = dataset.axis_index(AxisKind::Z) {
        index[z_axis] = z.min(dataset.shape()[z_axis].saturating_sub(1));
    }
    if let Some(t_axis) = dataset.axis_index(AxisKind::Time) {
        index[t_axis] = t.min(dataset.shape()[t_axis].saturating_sub(1));
    }
    let color_space = color_threshold_space_key(color_space);
    for y in min_y..=max_y {
        index[y_axis] = y;
        for x in min_x..=max_x {
            index[x_axis] = x;
            let components =
                sampled_color_components(dataset, &mut index, channel_axis, &color_space)?;
            for band in 0..3 {
                if components[band].is_finite() {
                    values[band].push(components[band]);
                }
            }
        }
    }
    if values.iter().any(Vec::is_empty) {
        return Err("Color Threshold found no finite RGB pixels".to_string());
    }
    Ok(values)
}

fn auto_threshold_range_for_values(
    values: &[f32],
    method: &str,
    background: &str,
    app: &AppContext,
) -> Result<(f32, f32), String> {
    let data = ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec())
        .map_err(|error| format!("color threshold auto histogram shape error: {error}"))?;
    let dataset = DatasetF32::new(
        data,
        Metadata {
            dims: vec![Dim::new(AxisKind::X, values.len())],
            pixel_type: PixelType::U8,
            ..Metadata::default()
        },
    )
    .map_err(|error| error.to_string())?;
    let output = app
        .ops_service()
        .execute(
            "threshold.make_binary",
            &dataset,
            &json!({
                "method": method,
                "background": background,
            }),
        )
        .map_err(|error| error.to_string())?;
    let measurements = output
        .measurements
        .ok_or_else(|| "Color Threshold auto did not report threshold range".to_string())?;
    let min = measurements
        .values
        .get("threshold_min")
        .and_then(Value::as_f64)
        .ok_or_else(|| "Color Threshold auto did not report threshold_min".to_string())?
        as f32;
    let max = measurements
        .values
        .get("threshold_max")
        .and_then(Value::as_f64)
        .ok_or_else(|| "Color Threshold auto did not report threshold_max".to_string())?
        as f32;
    Ok((min.floor().clamp(0.0, 255.0), max.ceil().clamp(0.0, 255.0)))
}

fn sampled_color_components(
    dataset: &DatasetF32,
    index: &mut [usize],
    channel_axis: usize,
    color_space: &str,
) -> Result<[f32; 3], String> {
    index[channel_axis] = 0;
    let r = component_to_color_u8(dataset.data[IxDyn(index)], dataset.metadata.pixel_type);
    index[channel_axis] = 1;
    let g = component_to_color_u8(dataset.data[IxDyn(index)], dataset.metadata.pixel_type);
    index[channel_axis] = 2;
    let b = component_to_color_u8(dataset.data[IxDyn(index)], dataset.metadata.pixel_type);
    match color_space {
        "hsb" => Ok(rgb_to_hsb_255_local(r, g, b)),
        "rgb" => Ok([r, g, b]),
        "lab" => Ok(rgb_to_lab_255_local(r, g, b)),
        "yuv" => Ok(rgb_to_yuv_255_local(r, g, b)),
        other => Err(format!("unsupported color threshold space `{other}`")),
    }
}

fn component_to_color_u8(value: f32, pixel_type: PixelType) -> f32 {
    match pixel_type {
        PixelType::U8 => value.clamp(0.0, 255.0),
        PixelType::U16 => (value.clamp(0.0, 65_535.0) / 257.0).round(),
        PixelType::F32 if value <= 1.0 => (value.clamp(0.0, 1.0) * 255.0).round(),
        PixelType::F32 => value.clamp(0.0, 255.0),
    }
}

fn rgb_to_hsb_255_local(r: f32, g: f32, b: f32) -> [f32; 3] {
    let r = r / 255.0;
    let g = g / 255.0;
    let b = b / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let hue = if delta <= f32::EPSILON {
        0.0
    } else if (max - r).abs() <= f32::EPSILON {
        60.0 * (((g - b) / delta) % 6.0)
    } else if (max - g).abs() <= f32::EPSILON {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    let hue = if hue < 0.0 { hue + 360.0 } else { hue };
    let saturation = if max <= f32::EPSILON {
        0.0
    } else {
        delta / max
    };
    [hue / 360.0 * 255.0, saturation * 255.0, max * 255.0]
}

fn rgb_to_yuv_255_local(r: f32, g: f32, b: f32) -> [f32; 3] {
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let u = (0.492 * (b - y) + 128.0).clamp(0.0, 255.0);
    let v = (0.877 * (r - y) + 128.0).clamp(0.0, 255.0);
    [y.clamp(0.0, 255.0), u, v]
}

fn rgb_to_lab_255_local(r: f32, g: f32, b: f32) -> [f32; 3] {
    let r = srgb_to_linear_local(r / 255.0);
    let g = srgb_to_linear_local(g / 255.0);
    let b = srgb_to_linear_local(b / 255.0);
    let x = (0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b) / 0.950_47;
    let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175 * b;
    let z = (0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b) / 1.088_83;
    let fx = lab_pivot_local(x);
    let fy = lab_pivot_local(y);
    let fz = lab_pivot_local(z);
    [
        (116.0 * fy - 16.0).clamp(0.0, 100.0) * 2.55,
        (500.0 * (fx - fy) + 128.0).clamp(0.0, 255.0),
        (200.0 * (fy - fz) + 128.0).clamp(0.0, 255.0),
    ]
}

fn srgb_to_linear_local(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn lab_pivot_local(value: f32) -> f32 {
    if value > 0.008_856 {
        value.powf(1.0 / 3.0)
    } else {
        7.787 * value + 16.0 / 116.0
    }
}

fn color_threshold_macro_text(params: &Value) -> String {
    let space = params
        .get("color_space")
        .or_else(|| params.get("space"))
        .and_then(Value::as_str)
        .unwrap_or("HSB");
    let space_key = color_threshold_space_key(space).to_ascii_uppercase();
    let mode = params.get("mode").and_then(Value::as_str).unwrap_or("Red");
    let method = params
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("Default");
    let band = |index: usize, key: &str| {
        let object = params.get(key).and_then(Value::as_object);
        let min = object
            .and_then(|object| object.get("min"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let max = object
            .and_then(|object| object.get("max"))
            .and_then(Value::as_f64)
            .unwrap_or(255.0);
        let pass = object
            .and_then(|object| object.get("pass"))
            .and_then(Value::as_bool)
            .unwrap_or(true);
        format!(
            "min[{index}]={min:.0};\nmax[{index}]={max:.0};\nfilter[{index}]=\"{}\";\n",
            if pass { "pass" } else { "stop" }
        )
    };
    let setup = match space_key.as_str() {
        "HSB" => concat!(
            "run(\"HSB Stack\");\n",
            "run(\"Convert Stack to Images\");\n",
            "selectWindow(\"Hue\");\nrename(\"0\");\n",
            "selectWindow(\"Saturation\");\nrename(\"1\");\n",
            "selectWindow(\"Brightness\");\nrename(\"2\");\n",
        )
        .to_string(),
        "LAB" => concat!(
            "call(\"ij.plugin.frame.ColorThresholder.RGBtoLab\");\n",
            "run(\"RGB Stack\");\n",
            "run(\"Convert Stack to Images\");\n",
            "selectWindow(\"Red\");\nrename(\"0\");\n",
            "selectWindow(\"Green\");\nrename(\"1\");\n",
            "selectWindow(\"Blue\");\nrename(\"2\");\n",
        )
        .to_string(),
        "YUV" => concat!(
            "call(\"ij.plugin.frame.ColorThresholder.RGBtoYUV\");\n",
            "run(\"RGB Stack\");\n",
            "run(\"Convert Stack to Images\");\n",
            "selectWindow(\"Red\");\nrename(\"0\");\n",
            "selectWindow(\"Green\");\nrename(\"1\");\n",
            "selectWindow(\"Blue\");\nrename(\"2\");\n",
        )
        .to_string(),
        _ => concat!(
            "run(\"RGB Stack\");\n",
            "run(\"Convert Stack to Images\");\n",
            "selectWindow(\"Red\");\nrename(\"0\");\n",
            "selectWindow(\"Green\");\nrename(\"1\");\n",
            "selectWindow(\"Blue\");\nrename(\"2\");\n",
        )
        .to_string(),
    };
    format!(
        concat!(
            "// Color Thresholder\n",
            "// Autogenerated macro, single images only!\n",
            "// color_space={space} method={method} threshold_color={mode}\n",
            "min=newArray(3);\n",
            "max=newArray(3);\n",
            "filter=newArray(3);\n",
            "a=getTitle();\n",
            "{setup}",
            "{band0}",
            "{band1}",
            "{band2}",
            "for (i=0;i<3;i++){{\n",
            "  selectWindow(\"\"+i);\n",
            "  setThreshold(min[i], max[i]);\n",
            "  run(\"Convert to Mask\");\n",
            "  if (filter[i]==\"stop\")  run(\"Invert\");\n",
            "}}\n",
            "imageCalculator(\"AND create\", \"0\",\"1\");\n",
            "imageCalculator(\"AND create\", \"Result of 0\",\"2\");\n",
            "for (i=0;i<3;i++){{\n",
            "  selectWindow(\"\"+i);\n",
            "  close();\n",
            "}}\n",
            "selectWindow(\"Result of 0\");\n",
            "close();\n",
            "selectWindow(\"Result of Result of 0\");\n",
            "rename(a);\n",
            "// Colour Thresholding-------------\n",
        ),
        space = space,
        method = method,
        mode = mode,
        setup = setup,
        band0 = band(0, "hue"),
        band1 = band(1, "saturation"),
        band2 = band(2, "brightness"),
    )
}

fn draw_color_threshold_band(
    ui: &mut egui::Ui,
    id: &str,
    label: &str,
    min: &mut f32,
    max: &mut f32,
    pass: &mut bool,
    histogram: Option<&AdjustHistogram>,
) {
    ui.separator();
    ui.horizontal(|ui| {
        ui.label(label);
        ui.checkbox(pass, "Pass");
    });
    draw_adjust_histogram(ui, histogram, Some((*min, *max)), None, false);
    egui::Grid::new(id)
        .num_columns(2)
        .spacing(egui::vec2(8.0, 4.0))
        .show(ui, |ui| {
            ui.add(egui::Slider::new(min, 0.0..=255.0).show_value(false));
            ui.add(egui::DragValue::new(min).range(0.0..=255.0).speed(1.0));
            ui.end_row();
            ui.add(egui::Slider::new(max, 0.0..=255.0).show_value(false));
            ui.add(egui::DragValue::new(max).range(0.0..=255.0).speed(1.0));
            ui.end_row();
        });
    if *max < *min {
        std::mem::swap(min, max);
    }
}

fn draw_adjust_histogram(
    ui: &mut egui::Ui,
    histogram: Option<&AdjustHistogram>,
    range: Option<(f32, f32)>,
    marker: Option<f32>,
    log_scale: bool,
) {
    let available = egui::vec2(ui.available_width().max(240.0), 120.0);
    let (rect, _) = ui.allocate_exact_size(available, egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 0.0, egui::Color32::WHITE);
    painter.rect_stroke(
        rect,
        0.0,
        egui::Stroke::new(1.0, egui::Color32::BLACK),
        egui::StrokeKind::Outside,
    );

    let Some(histogram) = histogram else {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No histogram",
            egui::FontId::proportional(13.0),
            egui::Color32::DARK_GRAY,
        );
        return;
    };

    let max_count = display_histogram_max(&histogram.counts).max(1) as f32;
    let max_height = histogram_count_height(max_count, log_scale).max(f32::EPSILON);
    let bin_width = rect.width() / histogram.counts.len().max(1) as f32;
    for (index, count) in histogram.counts.iter().enumerate() {
        if *count == 0 {
            continue;
        }
        let x = rect.left() + index as f32 * bin_width + bin_width * 0.5;
        let normalized =
            histogram_count_height((*count as f32).min(max_count), log_scale) / max_height;
        let y = rect.bottom() - normalized * rect.height();
        painter.line_segment(
            [egui::pos2(x, rect.bottom()), egui::pos2(x, y)],
            egui::Stroke::new(bin_width.max(1.0), egui::Color32::from_rgb(110, 110, 150)),
        );
        painter.circle_filled(egui::pos2(x, y), 1.0, egui::Color32::BLACK);
    }

    let value_to_x = |value: f32| {
        let span = (histogram.max - histogram.min).max(f32::EPSILON);
        rect.left() + ((value - histogram.min) / span).clamp(0.0, 1.0) * rect.width()
    };

    if let Some((low, high)) = range
        && low.is_finite()
        && high.is_finite()
    {
        let raw_low = low;
        let raw_high = high;
        let low = raw_low.min(raw_high);
        let high = raw_low.max(raw_high);
        let left = value_to_x(low);
        let right = value_to_x(high);
        let range_rect = egui::Rect::from_min_max(
            egui::pos2(left, rect.top()),
            egui::pos2(right, rect.bottom()),
        );
        painter.rect_filled(
            range_rect,
            0.0,
            egui::Color32::from_rgba_premultiplied(110, 110, 150, 28),
        );
        painter.line_segment(
            [
                egui::pos2(left, rect.bottom()),
                egui::pos2(right, rect.top()),
            ],
            egui::Stroke::new(2.0, egui::Color32::BLACK),
        );
        painter.line_segment(
            [
                egui::pos2(right, rect.bottom() - 5.0),
                egui::pos2(right, rect.bottom()),
            ],
            egui::Stroke::new(2.0, egui::Color32::BLACK),
        );
        painter.circle_filled(egui::pos2(left, rect.bottom()), 4.0, egui::Color32::BLACK);
        painter.circle_filled(egui::pos2(right, rect.top()), 4.0, egui::Color32::BLACK);
    }

    if let Some(marker) = marker
        && marker.is_finite()
    {
        let x = value_to_x(marker);
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            egui::Stroke::new(2.0, egui::Color32::from_rgb(240, 196, 64)),
        );
    }

    ui.horizontal(|ui| {
        ui.small(format!("min {:.4}", histogram.min));
        ui.small(format!("max {:.4}", histogram.max));
        ui.small(format!("n {}", histogram.pixel_count));
        if log_scale {
            ui.small("log");
        }
    });
}

fn histogram_count_height(count: f32, log_scale: bool) -> f32 {
    if log_scale { count.ln_1p() } else { count }
}

fn display_histogram_max(counts: &[u64]) -> u64 {
    let mut max_count = 0_u64;
    let mut mode = 0usize;
    for (index, count) in counts.iter().enumerate() {
        if *count > max_count {
            max_count = *count;
            mode = index;
        }
    }
    let mut second_max = 0_u64;
    for (index, count) in counts.iter().enumerate() {
        if index != mode {
            second_max = second_max.max(*count);
        }
    }
    if second_max != 0 && max_count > second_max * 2 {
        ((second_max as f32) * 1.5) as u64
    } else {
        max_count
    }
}

fn draw_imagej_viewer_popup_menu(
    ui: &mut egui::Ui,
    window_label: &str,
    actions: &mut Vec<UiAction>,
) {
    for (label, command_id) in [
        ("Show Info...", Some("image.show_info")),
        ("Properties...", Some("image.properties")),
        ("Rename...", Some("image.rename")),
        ("Measure", Some("analyze.measure")),
        ("Histogram", Some("analyze.histogram")),
        ("Duplicate Image...", Some("image.duplicate")),
        ("Original Scale", Some("image.zoom.original")),
        ("-", None),
        ("Record...", Some("plugins.macros.record")),
        ("Find Commands...", Some("plugins.commands.find")),
        ("Capture Screen", None),
    ] {
        if label == "-" {
            ui.separator();
            continue;
        }
        let Some(command_id) = command_id else {
            ui.add_enabled(false, egui::Button::new(label));
            continue;
        };
        if ui.button(label).clicked() {
            actions.push(UiAction::Command {
                window_label: window_label.to_string(),
                command_id: command_id.to_string(),
                params: None,
            });
            ui.close_menu();
        }
    }
}

fn lookup_table_slice_to_rgb(slice: &SliceImage, lut: LookupTable) -> Result<DatasetF32, String> {
    let mut values = Vec::with_capacity(slice.width * slice.height * 3);
    for gray in to_u8_samples(&slice.values, slice.pixel_type, None) {
        let color = lookup_table_color(lut, gray);
        values.push(f32::from(color.r()) / 255.0);
        values.push(f32::from(color.g()) / 255.0);
        values.push(f32::from(color.b()) / 255.0);
    }
    let data = ArrayD::from_shape_vec(IxDyn(&[slice.height, slice.width, 3]), values)
        .map_err(|error| format!("LUT-applied image shape error: {error}"))?;
    let mut metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, slice.height),
            Dim::new(AxisKind::X, slice.width),
            Dim::new(AxisKind::Channel, 3),
        ],
        pixel_type: PixelType::U8,
        channel_names: vec!["R".to_string(), "G".to_string(), "B".to_string()],
        ..Metadata::default()
    };
    metadata
        .extras
        .insert("applied_lut".to_string(), json!(lut.label()));
    Dataset::new(data, metadata).map_err(|error| error.to_string())
}

fn fit_to_rect(viewer: &mut ViewerUiState, rect: egui::Rect, width: usize, height: usize) {
    viewer.transform.fit_to_canvas(rect, width, height);
    viewer.zoom = viewer.transform.magnification;
    viewer.pan = egui::Vec2::ZERO;
}

fn initialize_view_to_open_state(
    viewer: &mut ViewerUiState,
    rect: egui::Rect,
    width: usize,
    height: usize,
) {
    let fits_without_scaling = width as f32 <= rect.width() && height as f32 <= rect.height();
    if fits_without_scaling {
        viewer.transform = ViewerTransformState::new(width, height);
        viewer.initial_magnification = viewer.transform.magnification;
        viewer.zoom = viewer.transform.magnification;
        viewer.pan = egui::Vec2::ZERO;
    } else {
        fit_to_rect(viewer, rect, width, height);
        viewer.initial_magnification = viewer.transform.magnification;
    }
}

fn is_supported_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            let extension = extension.to_ascii_lowercase();
            supported_formats().contains(&extension.as_str())
        })
        .unwrap_or(false)
}

fn viewer_sort_key(label: &str) -> u64 {
    label
        .strip_prefix(VIEWER_PREFIX)
        .and_then(|suffix| suffix.parse::<u64>().ok())
        .unwrap_or(u64::MAX)
}

fn normalize_path(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    std::fs::canonicalize(&absolute).unwrap_or(absolute)
}

fn renamed_image_path(current_path: &Path, title: &str) -> PathBuf {
    let sanitized = sanitize_image_title(title);
    let extension = current_path
        .extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| !extension.is_empty())
        .unwrap_or("tif");
    let file_name = if Path::new(&sanitized).extension().is_some() {
        sanitized
    } else {
        format!("{sanitized}.{extension}")
    };
    current_path
        .parent()
        .map(|parent| parent.join(&file_name))
        .unwrap_or_else(|| PathBuf::from(file_name))
}

fn macro_save_path(path: &str, format: Option<&str>) -> PathBuf {
    let path = PathBuf::from(path);
    if path.extension().is_some() {
        return path;
    }

    let Some(extension) = format.and_then(macro_save_extension) else {
        return path;
    };
    path.with_extension(extension)
}

fn macro_save_extension(format: &str) -> Option<&'static str> {
    let format = format.trim().to_ascii_lowercase();
    if format.contains("tif") {
        Some("tif")
    } else if format.contains("jpeg") || format.contains("jpg") {
        Some("jpg")
    } else if format.contains("png") {
        Some("png")
    } else if format.contains("bmp") {
        Some("bmp")
    } else if format.contains("pgm") {
        Some("pgm")
    } else if format.contains("text") || format.contains("txt") {
        Some("txt")
    } else {
        None
    }
}

fn sanitize_image_title(title: &str) -> String {
    let sanitized = title
        .chars()
        .map(|ch| match ch {
            '/' | '\\' | ':' | '\0' => '_',
            other => other,
        })
        .collect::<String>()
        .trim()
        .to_string();
    if sanitized.is_empty() {
        "Untitled".to_string()
    } else {
        sanitized
    }
}

fn new_image_dataset(params: &Value) -> Result<DatasetF32, String> {
    let width = params.get("width").and_then(Value::as_u64).unwrap_or(512) as usize;
    let height = params.get("height").and_then(Value::as_u64).unwrap_or(512) as usize;
    let slices = params.get("slices").and_then(Value::as_u64).unwrap_or(1) as usize;
    let channels = params.get("channels").and_then(Value::as_u64).unwrap_or(1) as usize;
    let frames = params.get("frames").and_then(Value::as_u64).unwrap_or(1) as usize;
    if width == 0 || height == 0 || slices == 0 || channels == 0 || frames == 0 {
        return Err("new image dimensions must be positive".to_string());
    }
    let fill = params.get("fill").and_then(Value::as_f64).unwrap_or(0.0) as f32;
    let pixel_type = match params
        .get("pixelType")
        .and_then(Value::as_str)
        .unwrap_or("f32")
    {
        "u8" => PixelType::U8,
        "u16" => PixelType::U16,
        _ => PixelType::F32,
    };

    let mut shape = vec![height, width];
    let mut dims = vec![Dim::new(AxisKind::Y, height), Dim::new(AxisKind::X, width)];
    if slices > 1 {
        shape.push(slices);
        dims.push(Dim::new(AxisKind::Z, slices));
    }
    if channels > 1 {
        shape.push(channels);
        dims.push(Dim::new(AxisKind::Channel, channels));
    }
    if frames > 1 {
        shape.push(frames);
        dims.push(Dim::new(AxisKind::Time, frames));
    }

    let len: usize = shape.iter().product();
    let data = ArrayD::from_shape_vec(IxDyn(&shape), vec![fill; len])
        .map_err(|error| format!("new image shape error: {error}"))?;
    let metadata = Metadata {
        dims,
        pixel_type,
        ..Metadata::default()
    };
    Dataset::new(data, metadata).map_err(|error| error.to_string())
}

fn source_ptr_eq(left: &ViewerImageSource, right: &ViewerImageSource) -> bool {
    match (left, right) {
        (ViewerImageSource::Native(left), ViewerImageSource::Native(right)) => {
            Arc::ptr_eq(left, right)
        }
        (ViewerImageSource::Dataset(left), ViewerImageSource::Dataset(right)) => {
            Arc::ptr_eq(left, right)
        }
        _ => false,
    }
}

pub fn run(startup_input: Option<PathBuf>) -> Result<(), String> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("image-rs Control")
            .with_inner_size(LAUNCHER_MIN_WINDOW_SIZE)
            .with_min_inner_size(LAUNCHER_MIN_WINDOW_SIZE)
            .with_resizable(true),
        ..Default::default()
    };

    eframe::run_native(
        "image-rs Control",
        options,
        Box::new(move |cc| Ok(Box::new(ImageUiApp::new(cc, startup_input.clone())))),
    )
    .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use crate::formats::NativeRasterImage;
    use crate::model::{AxisKind, DatasetF32, Dim, Metadata, PixelType};
    use crate::runtime::AppContext;
    use crate::ui::interaction::roi::{RoiPosition, RoiStore};
    use crate::ui::interaction::transform::ViewerTransformState;
    use crate::ui::state::{BinaryOptions, MeasurementSettings, OverlaySettings};
    use eframe::egui;
    use ndarray::{Array, IxDyn};
    use serde_json::{Value, json};

    use super::{
        AdjustDialogState, AdjustHistogram, HoverInfo, ImageSummary, ImageUiApp,
        LauncherStatusModel, LookupTable, OpRunMode, OverlayVisibility, ProgressState,
        RepaintDecisionInputs, RoiKind, RoiModel, SliceImage, ThresholdOverlay,
        ThresholdOverlayMode, ToolId, UiState, ViewerFrameBuffer, ViewerFrameRequest,
        ViewerImageSource, ViewerSession, ViewerTelemetry, ViewerUiState, ZoomCommand,
        add_selection_to_overlay, adjust_dialog_window_title, adjust_histogram,
        apply_overlay_visibility, apply_zoom_command, auto_contrast_range,
        binary_morphology_params, build_frame, canonical_json, centered_circular_roi,
        color_balance_channel_index, color_balance_channel_labels_for_session,
        color_balance_uses_lut_color, color_threshold_auto_ranges, color_threshold_choice_labels,
        color_threshold_macro_text, combine_stack_datasets, compute_initial_viewport_size,
        compute_viewer_frame, concatenate_stack_datasets, coordinates_dialog_is_stack,
        create_circular_masks_dataset, dominant_scroll_component, effective_scroll_delta,
        first_report_line, flatten_overlay_slice, format_launcher_status, full_image_rect_roi,
        function_key_for_macro_shortcut, image_draw_rect, image_slice_to_results_rows,
        imagej_color_from_name, imagej_color_to_string, images_to_stack_dataset,
        init_coordinates_dialog, initialize_view_to_open_state, insert_stack_dataset,
        install_macro_file_to_dir, installed_macro_file_name,
        installed_macro_menu_entry_from_block, interpolate_roi_kind, line_width_from_params,
        list_installed_macro_files_in_dir, lookup_table_color, lookup_table_from_command,
        lookup_table_slice_to_rgb, macro_display_name, macro_name_shortcut,
        macro_named_block_statement_map, macro_options_to_json, macro_record_line_for_command,
        macro_save_path, macro_shortcut_matches_text, macro_source_executable_lines,
        macro_source_named_blocks, mask_foreground_bbox, mask_foreground_roi, new_image_dataset,
        overlay_element_rows, overlay_from_roi_manager, overlay_label_for_roi,
        overlay_to_roi_manager, parse_macro_command_line, preview_cache_key,
        remove_stack_slice_labels_dataset, renamed_image_path,
        reset_color_threshold_bands_for_space, resize_op_mode_from_params, results_distribution,
        results_rows_to_dataset, results_summary_rows, roi_kind_bbox, roi_label_anchor,
        roi_stroke_width, sample_color_threshold_ranges_with_passes, sanitize_image_title,
        scale_fill_with_background_available, selection_properties_row,
        set_selected_roi_spline_fit, set_stack_slice_label_dataset,
        set_threshold_sixteen_bit_histogram, should_request_periodic_repaint,
        should_request_repaint_now, source_ptr_eq, spline_fit_roi_points, stack_measurement_rows,
        stack_position_from_params, stack_slice_label, stack_slice_path, stack_to_image_datasets,
        stack_xy_profile_rows, startup_auto_run_macro_block, strip_macro_line_comment,
        threshold_method_labels, threshold_method_param, to_color_image_with_threshold,
        tool_from_command_id, tool_shortcut_command, viewer_sort_key, xy_coordinate_rows,
        zoom_set_params,
    };
    use tempfile::tempdir;

    fn dataset_2x2(values: [f32; 4]) -> Arc<DatasetF32> {
        dataset_2x2_with_pixel_type(values, PixelType::F32)
    }

    fn dataset_2x2_with_pixel_type(values: [f32; 4], pixel_type: PixelType) -> Arc<DatasetF32> {
        let data = Array::from_shape_vec((2, 2), values.to_vec())
            .expect("shape")
            .into_dyn();
        Arc::new(DatasetF32::from_data_with_default_metadata(
            data, pixel_type,
        ))
    }

    fn native_gray8_2x2(values: [u8; 4]) -> Arc<NativeRasterImage> {
        Arc::new(NativeRasterImage::Gray8 {
            width: 2,
            height: 2,
            pixels: values.to_vec(),
            source: Some(PathBuf::from("/tmp/native.png")),
        })
    }

    #[test]
    fn canonical_json_sorts_object_keys_recursively() {
        let input = json!({
            "z": 2,
            "a": {"b": 1, "a": 0},
            "list": [
                {"d": 4, "c": 3}
            ]
        });

        let normalized = canonical_json(&input);
        assert_eq!(
            normalized,
            json!({
                "a": {"a": 0, "b": 1},
                "list": [{"c": 3, "d": 4}],
                "z": 2
            })
        );
    }

    #[test]
    fn preview_cache_key_is_stable_for_equivalent_param_ordering() {
        let left = preview_cache_key("gaussian.blur", &json!({"sigma": 1.0, "radius": 2}))
            .expect("left key");
        let right = preview_cache_key("gaussian.blur", &json!({"radius": 2, "sigma": 1.0}))
            .expect("right key");
        assert_eq!(left, right);
    }

    #[test]
    fn binary_morphology_params_use_saved_options() {
        let options = BinaryOptions {
            iterations: 3,
            count: 4,
        };

        assert_eq!(
            binary_morphology_params("process.binary.erode", None, &options),
            json!({
                "iterations": 3,
                "count": 4
            })
        );
    }

    #[test]
    fn binary_morphology_params_allow_request_overrides() {
        let options = BinaryOptions {
            iterations: 3,
            count: 4,
        };

        assert_eq!(
            binary_morphology_params(
                "process.binary.open",
                Some(json!({
                    "iterations": 2,
                    "radius": 1
                })),
                &options
            ),
            json!({
                "iterations": 2,
                "count": 4,
                "radius": 1
            })
        );
    }

    #[test]
    fn binary_morphology_params_skip_options_for_other_binary_ops() {
        let options = BinaryOptions {
            iterations: 3,
            count: 4,
        };

        assert_eq!(
            binary_morphology_params("process.binary.median", None, &options),
            json!({})
        );
    }

    #[test]
    fn images_to_stack_dataset_combines_matching_2d_images() {
        let first = dataset_2x2([0.0, 1.0, 2.0, 3.0]);
        let second = dataset_2x2([4.0, 5.0, 6.0, 7.0]);

        let stack = images_to_stack_dataset(&[
            ("first.tif", first.as_ref()),
            ("second.tif", second.as_ref()),
        ])
        .expect("images to stack");

        assert_eq!(stack.shape(), &[2, 2, 2]);
        assert_eq!(stack.metadata.dims[2].axis, AxisKind::Z);
        assert_eq!(
            stack.data.iter().copied().collect::<Vec<_>>(),
            vec![0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]
        );
        assert_eq!(
            stack.metadata.extras.get("images_to_stack_titles"),
            Some(&json!(["first.tif", "second.tif"]))
        );
    }

    #[test]
    fn images_to_stack_dataset_inserts_z_before_channels() {
        let metadata = Metadata {
            dims: vec![
                Dim::new(AxisKind::Y, 1),
                Dim::new(AxisKind::X, 2),
                Dim::new(AxisKind::Channel, 3),
            ],
            pixel_type: PixelType::F32,
            ..Metadata::default()
        };
        let first = DatasetF32::new(
            Array::from_shape_vec((1, 2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .expect("first shape")
                .into_dyn(),
            metadata.clone(),
        )
        .expect("first dataset");
        let second = DatasetF32::new(
            Array::from_shape_vec((1, 2, 3), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
                .expect("second shape")
                .into_dyn(),
            metadata,
        )
        .expect("second dataset");

        let stack = images_to_stack_dataset(&[("first", &first), ("second", &second)])
            .expect("images to stack");

        assert_eq!(stack.shape(), &[1, 2, 2, 3]);
        assert_eq!(
            stack
                .metadata
                .dims
                .iter()
                .map(|dim| dim.axis)
                .collect::<Vec<_>>(),
            vec![AxisKind::Y, AxisKind::X, AxisKind::Z, AxisKind::Channel]
        );
        assert_eq!(stack.data[IxDyn(&[0, 0, 0, 1])], 2.0);
        assert_eq!(stack.data[IxDyn(&[0, 0, 1, 1])], 8.0);
    }

    #[test]
    fn images_to_stack_dataset_rejects_single_or_mismatched_images() {
        let first = dataset_2x2([0.0, 1.0, 2.0, 3.0]);
        let different = DatasetF32::from_data_with_default_metadata(
            Array::from_shape_vec((1, 4), vec![0.0, 1.0, 2.0, 3.0])
                .expect("shape")
                .into_dyn(),
            PixelType::F32,
        );

        assert!(
            images_to_stack_dataset(&[("first", first.as_ref())])
                .expect_err("single image")
                .contains("at least two")
        );
        assert!(
            images_to_stack_dataset(&[("first", first.as_ref()), ("different", &different)])
                .expect_err("mismatched")
                .contains("matching dimensions")
        );
    }

    #[test]
    fn combine_stack_datasets_combines_horizontally_and_fills_missing_depth() {
        let first_data =
            Array::from_shape_vec((2, 2, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0])
                .expect("first shape")
                .into_dyn();
        let second_data = Array::from_shape_vec((1, 1, 1), vec![9.0])
            .expect("second shape")
            .into_dyn();
        let first = DatasetF32::from_data_with_default_metadata(first_data, PixelType::F32);
        let second = DatasetF32::from_data_with_default_metadata(second_data, PixelType::F32);

        let combined = combine_stack_datasets(&first, &second, false, -1.0).expect("combine");

        assert_eq!(combined.shape(), &[2, 3, 2]);
        assert_eq!(combined.data[IxDyn(&[0, 0, 0])], 1.0);
        assert_eq!(combined.data[IxDyn(&[0, 0, 1])], 10.0);
        assert_eq!(combined.data[IxDyn(&[0, 2, 0])], 9.0);
        assert_eq!(combined.data[IxDyn(&[0, 2, 1])], -1.0);
        assert_eq!(combined.data[IxDyn(&[1, 2, 0])], -1.0);
        assert_eq!(
            combined.metadata.extras.get("stack_combine_orientation"),
            Some(&json!("horizontal"))
        );
    }

    #[test]
    fn combine_stack_datasets_combines_vertically() {
        let first = dataset_2x2([1.0, 2.0, 3.0, 4.0]);
        let second = dataset_2x2([5.0, 6.0, 7.0, 8.0]);

        let combined =
            combine_stack_datasets(first.as_ref(), second.as_ref(), true, 0.0).expect("combine");

        assert_eq!(combined.shape(), &[4, 2, 1]);
        assert_eq!(combined.data[IxDyn(&[0, 0, 0])], 1.0);
        assert_eq!(combined.data[IxDyn(&[1, 1, 0])], 4.0);
        assert_eq!(combined.data[IxDyn(&[2, 0, 0])], 5.0);
        assert_eq!(combined.data[IxDyn(&[3, 1, 0])], 8.0);
        assert_eq!(
            combined.metadata.extras.get("stack_combine_orientation"),
            Some(&json!("vertical"))
        );
    }

    #[test]
    fn combine_stack_datasets_rejects_channel_images() {
        let metadata = Metadata {
            dims: vec![
                Dim::new(AxisKind::Y, 1),
                Dim::new(AxisKind::X, 1),
                Dim::new(AxisKind::Channel, 3),
            ],
            pixel_type: PixelType::F32,
            ..Metadata::default()
        };
        let rgb = DatasetF32::new(
            Array::from_shape_vec((1, 1, 3), vec![1.0, 0.0, 0.0])
                .expect("shape")
                .into_dyn(),
            metadata,
        )
        .expect("dataset");
        let gray = dataset_2x2([0.0, 1.0, 2.0, 3.0]);

        let error =
            combine_stack_datasets(&rgb, gray.as_ref(), false, 0.0).expect_err("channels rejected");

        assert!(error.contains("X/Y/Z"));
    }

    #[test]
    fn concatenate_stack_datasets_appends_depth_and_pads_larger_canvas() {
        let first_data = Array::from_shape_vec((1, 2), vec![1.0, 2.0])
            .expect("first shape")
            .into_dyn();
        let second_data = Array::from_shape_vec((2, 1, 2), vec![3.0, 30.0, 4.0, 40.0])
            .expect("second shape")
            .into_dyn();
        let first = DatasetF32::from_data_with_default_metadata(first_data, PixelType::F32);
        let second = DatasetF32::from_data_with_default_metadata(second_data, PixelType::F32);

        let output = concatenate_stack_datasets(&[("first", &first), ("second", &second)], -1.0)
            .expect("concatenate");

        assert_eq!(output.shape(), &[2, 2, 3]);
        assert_eq!(output.data[IxDyn(&[0, 0, 0])], 1.0);
        assert_eq!(output.data[IxDyn(&[0, 1, 0])], 2.0);
        assert_eq!(output.data[IxDyn(&[1, 0, 0])], -1.0);
        assert_eq!(output.data[IxDyn(&[0, 0, 1])], 3.0);
        assert_eq!(output.data[IxDyn(&[0, 0, 2])], 30.0);
        assert_eq!(output.data[IxDyn(&[1, 0, 1])], 4.0);
        assert_eq!(output.data[IxDyn(&[1, 0, 2])], 40.0);
        assert_eq!(output.data[IxDyn(&[0, 1, 1])], -1.0);
        assert_eq!(
            output.metadata.extras.get("stack_concatenate_titles"),
            Some(&json!(["first", "second"]))
        );
    }

    #[test]
    fn concatenate_stack_datasets_rejects_single_or_channel_images() {
        let first = dataset_2x2([0.0, 1.0, 2.0, 3.0]);
        assert!(
            concatenate_stack_datasets(&[("first", first.as_ref())], 0.0)
                .expect_err("single image")
                .contains("at least two")
        );

        let metadata = Metadata {
            dims: vec![
                Dim::new(AxisKind::Y, 1),
                Dim::new(AxisKind::X, 1),
                Dim::new(AxisKind::Channel, 3),
            ],
            pixel_type: PixelType::F32,
            ..Metadata::default()
        };
        let rgb = DatasetF32::new(
            Array::from_shape_vec((1, 1, 3), vec![1.0, 0.0, 0.0])
                .expect("shape")
                .into_dyn(),
            metadata,
        )
        .expect("dataset");
        assert!(
            concatenate_stack_datasets(&[("first", first.as_ref()), ("rgb", &rgb)], 0.0)
                .expect_err("channel image")
                .contains("X/Y/Z")
        );
    }

    #[test]
    fn insert_stack_dataset_inserts_2d_source_with_clipping() {
        let source = dataset_2x2([1.0, 2.0, 3.0, 4.0]);
        let destination = DatasetF32::from_data_with_default_metadata(
            Array::from_shape_vec((3, 3), vec![0.0; 9])
                .expect("shape")
                .into_dyn(),
            PixelType::F32,
        );

        let inserted =
            insert_stack_dataset(source.as_ref(), &destination, 1, 1).expect("insert stack");

        assert_eq!(inserted.shape(), &[3, 3]);
        assert_eq!(inserted.data[IxDyn(&[1, 1])], 1.0);
        assert_eq!(inserted.data[IxDyn(&[1, 2])], 2.0);
        assert_eq!(inserted.data[IxDyn(&[2, 1])], 3.0);
        assert_eq!(inserted.data[IxDyn(&[2, 2])], 4.0);
        assert_eq!(
            inserted.metadata.extras.get("stack_insert_offset"),
            Some(&json!([1, 1]))
        );

        let clipped =
            insert_stack_dataset(source.as_ref(), &destination, -1, -1).expect("clipped insert");
        assert_eq!(clipped.data[IxDyn(&[0, 0])], 4.0);
        assert_eq!(clipped.data[IxDyn(&[0, 1])], 0.0);
    }

    #[test]
    fn insert_stack_dataset_repeats_source_last_slice() {
        let source = DatasetF32::from_data_with_default_metadata(
            Array::from_shape_vec((1, 1, 2), vec![5.0, 6.0])
                .expect("source shape")
                .into_dyn(),
            PixelType::F32,
        );
        let destination = DatasetF32::from_data_with_default_metadata(
            Array::from_shape_vec((1, 2, 3), vec![0.0; 6])
                .expect("destination shape")
                .into_dyn(),
            PixelType::F32,
        );

        let inserted = insert_stack_dataset(&source, &destination, 1, 0).expect("insert stack");

        assert_eq!(inserted.data[IxDyn(&[0, 1, 0])], 5.0);
        assert_eq!(inserted.data[IxDyn(&[0, 1, 1])], 6.0);
        assert_eq!(inserted.data[IxDyn(&[0, 1, 2])], 6.0);
    }

    #[test]
    fn insert_stack_dataset_rejects_channel_images() {
        let metadata = Metadata {
            dims: vec![
                Dim::new(AxisKind::Y, 1),
                Dim::new(AxisKind::X, 1),
                Dim::new(AxisKind::Channel, 3),
            ],
            pixel_type: PixelType::F32,
            ..Metadata::default()
        };
        let source = DatasetF32::new(
            Array::from_shape_vec((1, 1, 3), vec![1.0, 0.0, 0.0])
                .expect("shape")
                .into_dyn(),
            metadata,
        )
        .expect("dataset");
        let destination = dataset_2x2([0.0, 1.0, 2.0, 3.0]);

        let error =
            insert_stack_dataset(&source, destination.as_ref(), 0, 0).expect_err("channel image");

        assert!(error.contains("X/Y/Z"));
    }

    #[test]
    fn stack_measurement_rows_reports_each_z_slice() {
        let data = Array::from_shape_vec(
            (2, 2, 2),
            vec![
                0.0, 10.0, //
                2.0, 12.0, //
                4.0, 14.0, //
                6.0, 16.0,
            ],
        )
        .expect("shape")
        .into_dyn();
        let dataset = DatasetF32::from_data_with_default_metadata(data, PixelType::F32);

        let rows = stack_measurement_rows(&dataset, &MeasurementSettings::default(), None, 0, 0)
            .expect("measure stack");

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("slice"), Some(&json!(0)));
        assert_eq!(rows[1].get("slice"), Some(&json!(1)));
        assert_eq!(rows[0].get("mean"), Some(&json!(3.0)));
        assert_eq!(rows[1].get("mean"), Some(&json!(13.0)));
        assert_eq!(rows[0].get("area"), Some(&json!(3)));
        assert_eq!(rows[1].get("area"), Some(&json!(4)));
    }

    #[test]
    fn stack_measurement_rows_applies_selection_to_each_slice() {
        let data = Array::from_shape_vec(
            (2, 2, 2),
            vec![
                0.0, 10.0, //
                2.0, 12.0, //
                4.0, 14.0, //
                6.0, 16.0,
            ],
        )
        .expect("shape")
        .into_dyn();
        let dataset = DatasetF32::from_data_with_default_metadata(data, PixelType::F32);

        let rows = stack_measurement_rows(
            &dataset,
            &MeasurementSettings::default(),
            Some((1, 0, 1, 1)),
            0,
            0,
        )
        .expect("measure selected stack");

        assert_eq!(rows[0].get("mean"), Some(&json!(4.0)));
        assert_eq!(rows[1].get("mean"), Some(&json!(14.0)));
        assert_eq!(rows[0].get("bbox"), Some(&json!([1, 0, 1, 1])));
    }

    #[test]
    fn stack_measurement_rows_rejects_non_stack_images() {
        let dataset = dataset_2x2([0.0, 1.0, 2.0, 3.0]);

        let error = stack_measurement_rows(
            dataset.as_ref(),
            &MeasurementSettings::default(),
            None,
            0,
            0,
        )
        .expect_err("not a stack");

        assert!(error.contains("Z stack"));
    }

    #[test]
    fn stack_xy_profile_rows_reports_rect_profile_for_each_z_slice() {
        let data = Array::from_shape_vec(
            (2, 3, 2),
            vec![
                1.0, 11.0, //
                2.0, 12.0, //
                3.0, 13.0, //
                4.0, 14.0, //
                5.0, 15.0, //
                6.0, 16.0,
            ],
        )
        .expect("shape")
        .into_dyn();
        let dataset = DatasetF32::from_data_with_default_metadata(data, PixelType::F32);
        let params = json!({
            "left": 0,
            "top": 0,
            "width": 3,
            "height": 2,
            "vertical": false
        });

        let rows = stack_xy_profile_rows(&dataset, &params).expect("stack xy profile");

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("slice"), Some(&json!(0)));
        assert_eq!(rows[1].get("slice"), Some(&json!(1)));
        assert_eq!(rows[0].get("profile_axis"), Some(&json!("x")));
        assert_eq!(rows[0].get("sample_count"), Some(&json!(3)));
        assert_eq!(rows[0].get("P0"), Some(&json!(2.5)));
        assert_eq!(rows[0].get("P1"), Some(&json!(3.5)));
        assert_eq!(rows[0].get("P2"), Some(&json!(4.5)));
        assert_eq!(rows[1].get("P0"), Some(&json!(12.5)));
        assert_eq!(rows[1].get("P1"), Some(&json!(13.5)));
        assert_eq!(rows[1].get("P2"), Some(&json!(14.5)));
    }

    #[test]
    fn stack_xy_profile_rows_reports_line_profile_for_each_z_slice() {
        let data = Array::from_shape_vec(
            (2, 2, 2),
            vec![
                1.0, 10.0, //
                2.0, 20.0, //
                3.0, 30.0, //
                4.0, 40.0,
            ],
        )
        .expect("shape")
        .into_dyn();
        let dataset = DatasetF32::from_data_with_default_metadata(data, PixelType::F32);
        let params = json!({
            "x0": 0.0,
            "y0": 0.0,
            "x1": 1.0,
            "y1": 1.0
        });

        let rows = stack_xy_profile_rows(&dataset, &params).expect("stack xy profile");

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("profile_axis"), Some(&json!("line")));
        assert_eq!(rows[0].get("P0"), Some(&json!(1.0)));
        assert_eq!(rows[0].get("P1"), Some(&json!(4.0)));
        assert_eq!(rows[1].get("P0"), Some(&json!(10.0)));
        assert_eq!(rows[1].get("P1"), Some(&json!(40.0)));
    }

    #[test]
    fn stack_xy_profile_rows_rejects_non_stack_or_missing_selection() {
        let dataset = dataset_2x2([0.0, 1.0, 2.0, 3.0]);
        let non_stack_error = stack_xy_profile_rows(
            dataset.as_ref(),
            &json!({
                "left": 0,
                "top": 0,
                "width": 1,
                "height": 1
            }),
        )
        .expect_err("not a stack");
        assert!(non_stack_error.contains("Z stack"));

        let stack = DatasetF32::from_data_with_default_metadata(
            Array::from_shape_vec((1, 1, 2), vec![1.0, 2.0])
                .expect("shape")
                .into_dyn(),
            PixelType::F32,
        );
        let selection_error =
            stack_xy_profile_rows(&stack, &json!({})).expect_err("missing selection");
        assert!(selection_error.contains("line or rectangular selection"));
    }

    #[test]
    fn set_stack_slice_label_dataset_writes_per_slice_labels() {
        let data = Array::from_shape_vec((2, 2, 3), (0..12).map(|value| value as f32).collect())
            .expect("shape")
            .into_dyn();
        let dataset = DatasetF32::from_data_with_default_metadata(data, PixelType::F32);

        let labeled =
            set_stack_slice_label_dataset(&dataset, 1, "middle").expect("set slice label");

        assert_eq!(labeled.shape(), dataset.shape());
        assert_eq!(stack_slice_label(&labeled, 0), None);
        assert_eq!(stack_slice_label(&labeled, 1), Some("middle".to_string()));
        assert_eq!(stack_slice_label(&labeled, 2), None);
        assert_eq!(
            labeled.metadata.extras.get("slice_labels"),
            Some(&json!([null, "middle", null]))
        );
    }

    #[test]
    fn set_stack_slice_label_dataset_clears_empty_labels_and_validates_range() {
        let data = Array::from_shape_vec((2, 2, 2), (0..8).map(|value| value as f32).collect())
            .expect("shape")
            .into_dyn();
        let dataset = DatasetF32::from_data_with_default_metadata(data, PixelType::F32);
        let labeled = set_stack_slice_label_dataset(&dataset, 0, "first").expect("set label");
        let cleared = set_stack_slice_label_dataset(&labeled, 0, "").expect("clear label");

        assert!(cleared.metadata.extras.get("slice_labels").is_none());
        assert!(
            set_stack_slice_label_dataset(&dataset, 4, "bad")
                .expect_err("slice out of range")
                .contains("outside")
        );
    }

    #[test]
    fn stack_slice_labels_support_single_image_label_property() {
        let dataset = dataset_2x2([0.0, 1.0, 2.0, 3.0]);
        let labeled =
            set_stack_slice_label_dataset(dataset.as_ref(), 0, "single").expect("single label");

        assert_eq!(stack_slice_label(&labeled, 0), Some("single".to_string()));
        assert_eq!(
            labeled.metadata.extras.get("Slice_Label"),
            Some(&json!("single"))
        );
        assert!(labeled.metadata.extras.get("slice_labels").is_none());
    }

    #[test]
    fn remove_stack_slice_labels_dataset_removes_label_metadata() {
        let data = Array::from_shape_vec((2, 2, 2), (0..8).map(|value| value as f32).collect())
            .expect("shape")
            .into_dyn();
        let dataset = DatasetF32::from_data_with_default_metadata(data, PixelType::F32);
        let labeled = set_stack_slice_label_dataset(&dataset, 1, "second").expect("set label");

        let cleaned = remove_stack_slice_labels_dataset(&labeled).expect("remove labels");

        assert!(cleaned.metadata.extras.get("slice_labels").is_none());
        assert!(cleaned.metadata.extras.get("Slice_Label").is_none());
        assert_eq!(
            cleaned.data.iter().copied().collect::<Vec<_>>(),
            dataset.data.iter().copied().collect::<Vec<_>>()
        );
    }

    #[test]
    fn stack_to_image_datasets_splits_z_stack_into_2d_images() {
        let data = Array::from_shape_vec((2, 2, 3), (0..12).map(|value| value as f32).collect())
            .expect("shape")
            .into_dyn();
        let dataset = DatasetF32::from_data_with_default_metadata(data, PixelType::F32);

        let images = stack_to_image_datasets(&dataset).expect("stack to images");

        assert_eq!(images.len(), 3);
        assert_eq!(images[0].shape(), &[2, 2]);
        assert!(
            images[0]
                .metadata
                .dims
                .iter()
                .all(|dim| dim.axis != AxisKind::Z)
        );
        assert_eq!(
            images[0].data.iter().copied().collect::<Vec<_>>(),
            vec![0.0, 3.0, 6.0, 9.0]
        );
        assert_eq!(
            images[2].data.iter().copied().collect::<Vec<_>>(),
            vec![2.0, 5.0, 8.0, 11.0]
        );
        assert_eq!(
            images[2]
                .metadata
                .extras
                .get("stack_to_images_source_slice"),
            Some(&json!(3))
        );
        assert_eq!(
            images[2]
                .metadata
                .extras
                .get("stack_to_images_source_shape"),
            Some(&json!([2, 2, 3]))
        );
    }

    #[test]
    fn stack_to_image_datasets_rejects_non_stack_images() {
        let dataset = dataset_2x2([0.0, 1.0, 2.0, 3.0]);

        let error = stack_to_image_datasets(dataset.as_ref()).expect_err("not a stack");

        assert!(error.contains("Z stack"));
    }

    #[test]
    fn stack_slice_path_preserves_extension_and_adds_slice_number() {
        assert_eq!(
            stack_slice_path(Path::new("/tmp/source.tif"), 7),
            PathBuf::from("/tmp/source-slice-007.tif")
        );
        assert_eq!(
            stack_slice_path(Path::new("/tmp/source"), 2),
            PathBuf::from("/tmp/source-slice-002.tif")
        );
    }

    #[test]
    fn line_width_from_params_accepts_positive_width() {
        assert_eq!(line_width_from_params(&json!({})).expect("default"), 1.0);
        assert_eq!(
            line_width_from_params(&json!({"width": 5.5})).expect("custom"),
            6.0
        );
        assert!(line_width_from_params(&json!({"width": 0.0})).is_err());
        assert!(line_width_from_params(&json!({"width": -1.0})).is_err());
    }

    #[test]
    fn line_width_spline_fit_toggles_selected_polygon_roi_like_imagej() {
        let mut viewer = ViewerUiState::new("viewer-1", "test".to_string());
        viewer.rois.begin_active(
            RoiKind::Polygon {
                points: vec![
                    egui::pos2(0.0, 0.0),
                    egui::pos2(5.0, 10.0),
                    egui::pos2(10.0, 0.0),
                ],
                closed: false,
                spline_fit: false,
            },
            RoiPosition::default(),
        );
        viewer.rois.commit_active(false);

        assert!(set_selected_roi_spline_fit(&mut viewer, true));
        let RoiKind::Polygon { spline_fit, .. } = &viewer.rois.overlay_rois[0].kind else {
            panic!("expected polygon");
        };
        assert!(*spline_fit);
    }

    #[test]
    fn spline_fit_roi_points_preserves_endpoints_for_open_polygons() {
        let points = vec![
            egui::pos2(0.0, 0.0),
            egui::pos2(5.0, 10.0),
            egui::pos2(10.0, 0.0),
        ];
        let fitted = spline_fit_roi_points(&points, false);

        assert_eq!(fitted.first().copied(), Some(points[0]));
        assert_eq!(fitted.last().copied(), Some(points[2]));
        assert!(fitted.len() > points.len());
    }

    #[test]
    fn coordinates_dialog_defaults_to_selection_bounds_like_imagej() {
        let mut metadata = Metadata::from_shape(&[10, 10], PixelType::F32);
        metadata.dims[1].spacing = Some(2.0);
        metadata.dims[0].spacing = Some(5.0);
        metadata
            .extras
            .insert("x_origin_coordinate".to_string(), json!(10.0));
        metadata
            .extras
            .insert("y_origin_coordinate".to_string(), json!(20.0));
        let mut viewer = ViewerUiState::new("viewer-1", "test".to_string());
        viewer.rois.begin_active(
            RoiKind::Rect {
                start: egui::pos2(2.0, 3.0),
                end: egui::pos2(6.0, 7.0),
                rounded: false,
                rotated: false,
            },
            RoiPosition::default(),
        );
        viewer.rois.commit_active(true);
        let mut dialog = AdjustDialogState::default();

        init_coordinates_dialog(&mut dialog, &metadata, Some(&viewer), 1, 10.0, 10.0);

        assert_eq!(dialog.coordinates_mode, "selection");
        assert_eq!(dialog.coordinates_x_pixel, 2.0);
        assert_eq!(dialog.coordinates_y_pixel, 3.0);
        assert_eq!(dialog.coordinates_width, 4.0);
        assert_eq!(dialog.coordinates_height, 4.0);
        assert_eq!(dialog.left, 14.0);
        assert_eq!(dialog.right, 22.0);
        assert_eq!(dialog.top, 35.0);
        assert_eq!(dialog.bottom, 55.0);
    }

    #[test]
    fn coordinates_dialog_uses_point_mode_for_single_point_selection() {
        let mut metadata = Metadata::from_shape(&[10, 10, 3], PixelType::F32);
        metadata.dims[1].spacing = Some(2.0);
        metadata.dims[0].spacing = Some(5.0);
        metadata.dims[2].spacing = Some(10.0);
        let mut viewer = ViewerUiState::new("viewer-1", "test".to_string());
        viewer.z = 1;
        viewer.rois.begin_active(
            RoiKind::Point {
                points: vec![egui::pos2(4.0, 6.0)],
                multi: false,
            },
            RoiPosition::default(),
        );
        viewer.rois.commit_active(true);
        let mut dialog = AdjustDialogState::default();

        init_coordinates_dialog(&mut dialog, &metadata, Some(&viewer), 3, 10.0, 10.0);

        assert_eq!(dialog.coordinates_mode, "point");
        assert_eq!(dialog.coordinates_x_pixel, 4.0);
        assert_eq!(dialog.coordinates_y_pixel, 6.0);
        assert_eq!(dialog.coordinates_z_pixel, 1.0);
        assert_eq!(dialog.left, 8.0);
        assert_eq!(dialog.top, 30.0);
        assert_eq!(dialog.front, 10.0);
    }

    #[test]
    fn coordinates_dialog_only_shows_z_controls_for_stacks_like_imagej() {
        let mut dialog = AdjustDialogState::default();
        dialog.coordinates_depth = 1.0;
        assert!(!coordinates_dialog_is_stack(&dialog));

        dialog.coordinates_depth = 2.0;
        assert!(coordinates_dialog_is_stack(&dialog));
    }

    #[test]
    fn stack_position_from_params_uses_imagej_one_based_indices() {
        let position = stack_position_from_params(
            &json!({"channel": 2, "slice": 3, "frame": 4}),
            0,
            0,
            0,
            3,
            5,
            6,
        )
        .expect("position");

        assert_eq!(position, (1, 2, 3));
    }

    #[test]
    fn stack_position_from_params_keeps_unspecified_axes_and_validates_bounds() {
        let position =
            stack_position_from_params(&json!({"slice": 2}), 1, 3, 4, 3, 5, 6).expect("position");

        assert_eq!(position, (1, 1, 4));
        assert!(stack_position_from_params(&json!({}), 0, 0, 0, 1, 1, 1).is_err());
        assert!(stack_position_from_params(&json!({"slice": 0}), 0, 0, 0, 1, 3, 1).is_err());
        assert!(stack_position_from_params(&json!({"slice": 4}), 0, 0, 0, 1, 3, 1).is_err());
    }

    #[test]
    fn zoom_set_params_accepts_percent_and_optional_center() {
        let params = zoom_set_params(&json!({"zoom_percent": 250.0, "x": 10.0, "y": 20.0}))
            .expect("zoom params");

        assert_eq!(params, (2.5, Some(10.0), Some(20.0)));
    }

    #[test]
    fn zoom_set_params_validates_percent_and_paired_center() {
        assert!(zoom_set_params(&json!({})).is_err());
        assert!(zoom_set_params(&json!({"zoom_percent": 0.0})).is_err());
        assert!(zoom_set_params(&json!({"zoom_percent": 100.0, "x": 10.0})).is_err());
    }

    #[test]
    fn imagej_color_helpers_match_named_colors_and_hex_fallback() {
        assert_eq!(
            imagej_color_from_name("lightGray"),
            Some(egui::Color32::from_rgb(192, 192, 192))
        );
        assert_eq!(
            imagej_color_from_name("grey"),
            Some(egui::Color32::from_rgb(128, 128, 128))
        );
        assert_eq!(
            imagej_color_to_string(egui::Color32::from_rgb(255, 255, 0)),
            "Yellow"
        );
        assert_eq!(
            imagej_color_to_string(egui::Color32::from_rgb(1, 2, 3)),
            "#010203"
        );
    }

    #[test]
    fn lookup_table_commands_map_to_display_colors() {
        assert_eq!(
            lookup_table_from_command("image.lookup.red"),
            Some(LookupTable::Red)
        );
        assert_eq!(lookup_table_from_command("image.lookup.unknown"), None);
        assert_eq!(
            lookup_table_color(LookupTable::Grays, 128),
            egui::Color32::from_rgb(128, 128, 128)
        );
        assert_eq!(
            lookup_table_color(LookupTable::Inverted, 10),
            egui::Color32::from_rgb(245, 245, 245)
        );
        assert_eq!(
            lookup_table_color(LookupTable::Cyan, 200),
            egui::Color32::from_rgb(0, 200, 200)
        );
        assert_eq!(
            lookup_table_color(LookupTable::RedGreen, 0),
            egui::Color32::from_rgb(255, 0, 0)
        );
        assert_eq!(
            lookup_table_color(LookupTable::RedGreen, 255),
            egui::Color32::from_rgb(0, 255, 0)
        );
    }

    #[test]
    fn lookup_table_slice_to_rgb_bakes_display_lut() {
        let slice = SliceImage {
            width: 2,
            height: 1,
            pixel_type: PixelType::F32,
            values: vec![0.0, 1.0],
        };

        let dataset = lookup_table_slice_to_rgb(&slice, LookupTable::RedGreen).expect("apply lut");

        assert_eq!(dataset.shape(), &[1, 2, 3]);
        assert_eq!(dataset.metadata.pixel_type, PixelType::U8);
        assert_eq!(dataset.metadata.channel_names, ["R", "G", "B"]);
        assert_eq!(
            dataset.metadata.extras.get("applied_lut"),
            Some(&json!("Red/Green"))
        );
        assert_eq!(dataset.data[IxDyn(&[0, 0, 0])], 1.0);
        assert_eq!(dataset.data[IxDyn(&[0, 0, 1])], 0.0);
        assert_eq!(dataset.data[IxDyn(&[0, 1, 0])], 0.0);
        assert_eq!(dataset.data[IxDyn(&[0, 1, 1])], 1.0);
    }

    #[test]
    fn roi_stroke_width_uses_line_width_and_selection_emphasis() {
        assert_eq!(roi_stroke_width(3.0, false), 3.0);
        assert_eq!(roi_stroke_width(3.0, true), 3.5);
        assert_eq!(roi_stroke_width(0.0, false), 1.0);
    }

    #[test]
    fn roi_label_anchor_uses_selection_center() {
        let rect = RoiKind::Rect {
            start: egui::pos2(2.0, 4.0),
            end: egui::pos2(10.0, 12.0),
            rounded: false,
            rotated: false,
        };
        assert_eq!(roi_label_anchor(&rect), Some(egui::pos2(6.0, 8.0)));

        let polygon = RoiKind::Polygon {
            points: vec![
                egui::pos2(1.0, 2.0),
                egui::pos2(7.0, 4.0),
                egui::pos2(3.0, 10.0),
            ],
            closed: true,
            spline_fit: false,
        };
        assert_eq!(roi_label_anchor(&polygon), Some(egui::pos2(4.0, 6.0)));
    }

    #[test]
    fn add_selection_to_overlay_commits_active_roi_and_preserves_existing() {
        let mut rois = RoiStore::default();
        rois.begin_active(
            RoiKind::Rect {
                start: egui::pos2(0.0, 0.0),
                end: egui::pos2(2.0, 2.0),
                rounded: false,
                rotated: false,
            },
            RoiPosition::default(),
        );

        assert_eq!(
            add_selection_to_overlay(&mut rois).expect("add active"),
            "selection added to overlay"
        );
        assert_eq!(rois.overlay_rois.len(), 1);
        assert!(rois.active_roi.is_none());
        assert_eq!(rois.selected_roi_id, Some(rois.overlay_rois[0].id));

        rois.begin_active(
            RoiKind::Point {
                points: vec![egui::pos2(4.0, 4.0)],
                multi: false,
            },
            RoiPosition::default(),
        );
        add_selection_to_overlay(&mut rois).expect("add second");
        assert_eq!(rois.overlay_rois.len(), 2);
    }

    #[test]
    fn add_selection_to_overlay_marks_existing_selection_visible() {
        let mut rois = RoiStore::default();
        rois.begin_active(
            RoiKind::Point {
                points: vec![egui::pos2(1.0, 1.0)],
                multi: false,
            },
            RoiPosition::default(),
        );
        rois.commit_active(true);
        rois.overlay_rois[0].visible = false;

        assert_eq!(
            add_selection_to_overlay(&mut rois).expect("existing"),
            "selection already in overlay"
        );
        assert_eq!(rois.overlay_rois.len(), 1);
        assert!(rois.overlay_rois[0].visible);

        rois.selected_roi_id = None;
        assert!(add_selection_to_overlay(&mut rois).is_err());
    }

    #[test]
    fn overlay_roi_manager_bridge_uses_shared_store() {
        let mut rois = RoiStore::default();
        rois.begin_active(
            RoiKind::Point {
                points: vec![egui::pos2(1.0, 1.0)],
                multi: false,
            },
            RoiPosition::default(),
        );
        rois.commit_active(true);
        rois.overlay_rois[0].visible = false;
        rois.selected_roi_id = None;

        assert_eq!(
            overlay_from_roi_manager(&mut rois).expect("from manager"),
            1
        );
        assert!(rois.overlay_rois[0].visible);
        assert_eq!(rois.selected_roi_id, Some(rois.overlay_rois[0].id));

        rois.selected_roi_id = Some(999);
        assert_eq!(overlay_to_roi_manager(&mut rois).expect("to manager"), 1);
        assert_eq!(rois.selected_roi_id, Some(rois.overlay_rois[0].id));

        let mut empty = RoiStore::default();
        assert!(overlay_from_roi_manager(&mut empty).is_err());
        assert!(overlay_to_roi_manager(&mut empty).is_err());
    }

    #[test]
    fn overlay_label_for_roi_uses_index_or_name() {
        let roi = RoiModel {
            id: 1,
            name: "Nucleus".to_string(),
            kind: RoiKind::Rect {
                start: egui::pos2(2.0, 4.0),
                end: egui::pos2(6.0, 8.0),
                rounded: false,
                rotated: false,
            },
            position: RoiPosition::default(),
            visible: true,
            locked: false,
        };
        let mut settings = OverlaySettings {
            show_labels: true,
            ..OverlaySettings::default()
        };

        let (anchor, label) = overlay_label_for_roi(&roi, 4, &settings).expect("numbered label");
        assert_eq!(anchor, egui::pos2(4.0, 6.0));
        assert_eq!(label, "5");

        settings.use_names_as_labels = true;
        let (_, label) = overlay_label_for_roi(&roi, 4, &settings).expect("named label");
        assert_eq!(label, "Nucleus");

        settings.show_labels = false;
        assert!(overlay_label_for_roi(&roi, 4, &settings).is_none());
    }

    #[test]
    fn flatten_overlay_slice_burns_visible_roi_outlines() {
        let slice = SliceImage {
            width: 4,
            height: 3,
            pixel_type: PixelType::F32,
            values: vec![0.0; 12],
        };
        let rois = vec![RoiModel {
            id: 1,
            name: "Box".to_string(),
            kind: RoiKind::Rect {
                start: egui::pos2(1.0, 0.0),
                end: egui::pos2(3.0, 2.0),
                rounded: false,
                rotated: false,
            },
            position: RoiPosition::default(),
            visible: true,
            locked: false,
        }];

        let flattened = flatten_overlay_slice(&slice, &rois).expect("flatten");

        assert_eq!(flattened.shape(), &[3, 4]);
        assert_eq!(flattened.data[IxDyn(&[0, 1])], 1.0);
        assert_eq!(flattened.data[IxDyn(&[0, 2])], 1.0);
        assert_eq!(flattened.data[IxDyn(&[0, 3])], 1.0);
        assert_eq!(flattened.data[IxDyn(&[1, 1])], 1.0);
        assert_eq!(flattened.data[IxDyn(&[1, 2])], 0.0);
        assert_eq!(flattened.data[IxDyn(&[2, 3])], 1.0);
    }

    #[test]
    fn flatten_overlay_slice_rejects_empty_overlay() {
        let slice = SliceImage {
            width: 1,
            height: 1,
            pixel_type: PixelType::F32,
            values: vec![0.0],
        };

        assert!(flatten_overlay_slice(&slice, &[]).is_err());
    }

    #[test]
    fn overlay_element_rows_report_imagej_style_metadata() {
        let rois = vec![RoiModel {
            id: 7,
            name: "Cell 7".to_string(),
            kind: RoiKind::Rect {
                start: egui::pos2(2.2, 4.0),
                end: egui::pos2(5.0, 8.6),
                rounded: false,
                rotated: false,
            },
            position: RoiPosition {
                channel: 1,
                z: 2,
                t: 3,
            },
            visible: true,
            locked: false,
        }];

        let rows = overlay_element_rows(&rois).expect("overlay rows");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("Index"), Some(&json!(1)));
        assert_eq!(rows[0].get("Name"), Some(&json!("Cell 7")));
        assert_eq!(rows[0].get("Type"), Some(&json!("Rectangle")));
        assert_eq!(rows[0].get("X"), Some(&json!(2)));
        assert_eq!(rows[0].get("Y"), Some(&json!(4)));
        assert_eq!(rows[0].get("Width"), Some(&json!(4)));
        assert_eq!(rows[0].get("Height"), Some(&json!(6)));
        assert_eq!(rows[0].get("Channel"), Some(&json!(2)));
        assert_eq!(rows[0].get("Slice"), Some(&json!(3)));
        assert_eq!(rows[0].get("Frame"), Some(&json!(4)));
        assert_eq!(rows[0].get("Visible"), Some(&json!(true)));
    }

    #[test]
    fn selection_properties_row_reports_bounds_and_position() {
        let roi = RoiModel {
            id: 7,
            name: "Selection".to_string(),
            kind: RoiKind::Line {
                start: egui::pos2(1.0, 2.0),
                end: egui::pos2(6.0, 8.0),
                arrow: false,
            },
            position: RoiPosition {
                channel: 1,
                z: 2,
                t: 3,
            },
            visible: true,
            locked: false,
        };

        let row = selection_properties_row(&roi).expect("selection row");

        assert_eq!(row.get("Type"), Some(&json!("Line")));
        assert_eq!(row.get("Width"), Some(&json!(6)));
        assert_eq!(row.get("Height"), Some(&json!(7)));
        assert_eq!(row.get("Channel"), Some(&json!(2)));
        assert_eq!(row.get("Slice"), Some(&json!(3)));
        assert_eq!(row.get("Frame"), Some(&json!(4)));
    }

    #[test]
    fn apply_overlay_visibility_toggles_all_elements() {
        let mut rois = vec![
            RoiModel {
                id: 1,
                name: "A".to_string(),
                kind: RoiKind::Point {
                    points: vec![egui::pos2(1.0, 1.0)],
                    multi: false,
                },
                position: RoiPosition::default(),
                visible: true,
                locked: false,
            },
            RoiModel {
                id: 2,
                name: "B".to_string(),
                kind: RoiKind::Point {
                    points: vec![egui::pos2(2.0, 2.0)],
                    multi: false,
                },
                position: RoiPosition::default(),
                visible: false,
                locked: false,
            },
        ];

        assert_eq!(
            apply_overlay_visibility(&mut rois, OverlayVisibility::Toggle).expect("toggle"),
            2
        );
        assert!(rois.iter().all(|roi| !roi.visible));

        apply_overlay_visibility(&mut rois, OverlayVisibility::Toggle).expect("toggle");
        assert!(rois.iter().all(|roi| roi.visible));

        apply_overlay_visibility(&mut rois, OverlayVisibility::Hide).expect("hide");
        assert!(rois.iter().all(|roi| !roi.visible));
        assert!(apply_overlay_visibility(&mut [], OverlayVisibility::Show).is_err());
    }

    #[test]
    fn image_slice_to_results_rows_uses_imagej_y_labels_and_x_columns() {
        let slice = SliceImage {
            width: 3,
            height: 2,
            pixel_type: PixelType::F32,
            values: vec![
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0,
            ],
        };

        let rows =
            image_slice_to_results_rows(&slice, Some((1, 0, 2, 1))).expect("image to results");

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("Label"), Some(&json!("Y0")));
        assert_eq!(rows[0].get("X1"), Some(&json!(2.0)));
        assert_eq!(rows[0].get("X2"), Some(&json!(3.0)));
        assert_eq!(rows[1].get("Label"), Some(&json!("Y1")));
        assert_eq!(rows[1].get("X1"), Some(&json!(5.0)));
        assert_eq!(rows[1].get("X2"), Some(&json!(6.0)));
    }

    #[test]
    fn xy_coordinate_rows_skip_background_without_selection() {
        let slice = SliceImage {
            width: 3,
            height: 2,
            pixel_type: PixelType::F32,
            values: vec![
                0.0, 2.0, 0.0, //
                4.0, 0.0, 6.0,
            ],
        };

        let rows = xy_coordinate_rows(&slice, None, &json!({})).expect("xy rows");

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].get("X"), Some(&json!(1)));
        assert_eq!(rows[0].get("Y"), Some(&json!(1)));
        assert_eq!(rows[0].get("Value"), Some(&json!(2.0)));
        assert_eq!(rows[1].get("X"), Some(&json!(0)));
        assert_eq!(rows[1].get("Y"), Some(&json!(0)));
        assert_eq!(rows[2].get("Value"), Some(&json!(6.0)));
    }

    #[test]
    fn xy_coordinate_rows_include_selection_and_invert_y() {
        let slice = SliceImage {
            width: 3,
            height: 2,
            pixel_type: PixelType::F32,
            values: vec![
                0.0, 2.0, 0.0, //
                4.0, 0.0, 6.0,
            ],
        };

        let rows = xy_coordinate_rows(
            &slice,
            Some((1, 0, 2, 0)),
            &json!({"background": 0.0, "invert_y": true}),
        )
        .expect("xy rows");

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("X"), Some(&json!(1)));
        assert_eq!(rows[0].get("Y"), Some(&json!(0)));
        assert_eq!(rows[0].get("Value"), Some(&json!(2.0)));
        assert_eq!(rows[1].get("X"), Some(&json!(2)));
        assert_eq!(rows[1].get("Y"), Some(&json!(0)));
        assert_eq!(rows[1].get("Value"), Some(&json!(0.0)));
    }

    #[test]
    fn results_rows_to_dataset_sorts_imagej_x_columns_numerically() {
        let mut row0 = BTreeMap::new();
        row0.insert("Label".to_string(), json!("Y0"));
        row0.insert("X10".to_string(), json!(10.0));
        row0.insert("X2".to_string(), json!(2.0));
        row0.insert("X1".to_string(), json!(1.0));
        let mut row1 = BTreeMap::new();
        row1.insert("Label".to_string(), json!("Y1"));
        row1.insert("X10".to_string(), json!(20.0));
        row1.insert("X2".to_string(), json!(4.0));
        row1.insert("X1".to_string(), json!(3.0));

        let dataset = results_rows_to_dataset(&[row0, row1]).expect("results to image");

        assert_eq!(dataset.shape(), &[2, 3]);
        assert_eq!(
            dataset.data.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 10.0, 3.0, 4.0, 20.0]
        );
    }

    #[test]
    fn results_rows_to_dataset_rejects_empty_or_non_numeric_results() {
        assert!(results_rows_to_dataset(&[]).is_err());
        let mut row = BTreeMap::new();
        row.insert("Label".to_string(), json!("Y0"));
        row.insert("Name".to_string(), json!("cell"));
        let error = results_rows_to_dataset(&[row]).expect_err("non-numeric results");
        assert!(error.contains("no numeric columns"));
    }

    #[test]
    fn results_summary_rows_adds_imagej_mean_sd_min_max_rows() {
        let mut row0 = BTreeMap::new();
        row0.insert("Label".to_string(), json!("cell-1"));
        row0.insert("Area".to_string(), json!(2.0));
        row0.insert("Mean".to_string(), json!(4.0));
        let mut row1 = BTreeMap::new();
        row1.insert("Label".to_string(), json!("cell-2"));
        row1.insert("Area".to_string(), json!(4.0));
        row1.insert("Mean".to_string(), json!(8.0));

        let summary = results_summary_rows(&[row0, row1]).expect("summary rows");

        assert_eq!(summary.len(), 4);
        assert_eq!(summary[0].get("Label"), Some(&json!("Mean")));
        assert_eq!(summary[0].get("Area"), Some(&json!(3.0)));
        assert_eq!(summary[0].get("Mean"), Some(&json!(6.0)));
        assert_eq!(summary[2].get("Area"), Some(&json!(2.0)));
        assert_eq!(summary[3].get("Mean"), Some(&json!(8.0)));
        let sd = summary[1].get("Area").and_then(Value::as_f64).unwrap();
        assert!((sd - 2.0_f64.sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn results_summary_rows_rejects_small_or_already_summarized_tables() {
        let mut row = BTreeMap::new();
        row.insert("Area".to_string(), json!(2.0));
        assert!(results_summary_rows(&[row.clone()]).is_err());

        row.insert("Label".to_string(), json!("Max"));
        let mut row0 = BTreeMap::new();
        row0.insert("Area".to_string(), json!(1.0));
        let error = results_summary_rows(&[row0, row]).expect_err("already summarized");
        assert!(error.contains("already summarized"));
    }

    #[test]
    fn results_distribution_bins_numeric_results_column() {
        let rows = [1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(|value| {
                let mut row = BTreeMap::new();
                row.insert("Area".to_string(), json!(value));
                row
            })
            .collect::<Vec<_>>();

        let distribution = results_distribution(&rows, Some("Area"), 2).expect("distribution");

        assert_eq!(distribution.get("column"), Some(&json!("Area")));
        assert_eq!(distribution.get("count"), Some(&json!(4)));
        let bins = distribution
            .get("bins")
            .and_then(Value::as_array)
            .expect("bins");
        assert_eq!(bins[0].get("count"), Some(&json!(2)));
        assert_eq!(bins[1].get("count"), Some(&json!(2)));
    }

    #[test]
    fn results_distribution_rejects_missing_column_or_zero_bins() {
        let mut row = BTreeMap::new();
        row.insert("Area".to_string(), json!(2.0));

        assert!(results_distribution(&[row.clone()], Some("Mean"), 10).is_err());
        assert!(results_distribution(&[row], None, 0).is_err());
    }

    #[test]
    fn renamed_image_path_preserves_parent_and_extension() {
        let path = renamed_image_path(Path::new("/tmp/source.tif"), "Renamed Image");
        assert_eq!(path, PathBuf::from("/tmp/Renamed Image.tif"));

        let explicit = renamed_image_path(Path::new("/tmp/source.tif"), "renamed.png");
        assert_eq!(explicit, PathBuf::from("/tmp/renamed.png"));
    }

    #[test]
    fn macro_save_path_adds_format_extension_when_missing() {
        assert_eq!(
            macro_save_path("/tmp/output", Some("Tiff")),
            PathBuf::from("/tmp/output.tif")
        );
        assert_eq!(
            macro_save_path("/tmp/output", Some("PNG")),
            PathBuf::from("/tmp/output.png")
        );
        assert_eq!(
            macro_save_path("/tmp/output.jpg", Some("Tiff")),
            PathBuf::from("/tmp/output.jpg")
        );
    }

    #[test]
    fn sanitize_image_title_replaces_path_separators() {
        assert_eq!(sanitize_image_title("a/b\\c:d"), "a_b_c_d");
        assert_eq!(sanitize_image_title("   "), "Untitled");
    }

    #[test]
    fn new_image_dataset_creates_time_hyperstack() {
        let dataset = new_image_dataset(&json!({
            "width": 2,
            "height": 3,
            "slices": 4,
            "channels": 2,
            "frames": 5,
            "fill": 7.0,
            "pixelType": "u16"
        }))
        .expect("new hyperstack");

        assert_eq!(dataset.shape(), &[3, 2, 4, 2, 5]);
        assert_eq!(
            dataset
                .metadata
                .dims
                .iter()
                .map(|dim| dim.axis)
                .collect::<Vec<_>>(),
            vec![
                AxisKind::Y,
                AxisKind::X,
                AxisKind::Z,
                AxisKind::Channel,
                AxisKind::Time
            ]
        );
        assert_eq!(dataset.metadata.pixel_type, PixelType::U16);
        assert_eq!(dataset.data[IxDyn(&[2, 1, 3, 1, 4])], 7.0);
    }

    #[test]
    fn new_image_dataset_rejects_zero_dimensions() {
        let error = new_image_dataset(&json!({
            "width": 0,
            "height": 3,
            "slices": 1,
            "channels": 1,
            "frames": 1
        }))
        .expect_err("zero width");

        assert!(error.contains("positive"));
    }

    #[test]
    fn create_circular_masks_dataset_matches_imagej_stack_shape() {
        let dataset = create_circular_masks_dataset().expect("circular masks");

        assert_eq!(dataset.shape(), &[150, 150, 99]);
        assert_eq!(dataset.metadata.dims[0].axis, AxisKind::Y);
        assert_eq!(dataset.metadata.dims[1].axis, AxisKind::X);
        assert_eq!(dataset.metadata.dims[2].axis, AxisKind::Z);
        assert_eq!(dataset.data[IxDyn(&[75, 75, 0])], 1.0);
        assert_eq!(dataset.data[IxDyn(&[75, 76, 0])], 1.0);
        assert_eq!(dataset.data[IxDyn(&[75, 77, 0])], 0.0);
    }

    #[test]
    fn full_image_rect_roi_matches_imagej_select_all_bounds() {
        let roi = full_image_rect_roi(&[20, 30]).expect("full image roi");
        let RoiKind::Rect {
            start,
            end,
            rounded,
            rotated,
        } = roi
        else {
            panic!("expected rectangular ROI");
        };

        assert_eq!(start, egui::pos2(0.0, 0.0));
        assert_eq!(end, egui::pos2(30.0, 20.0));
        assert!(!rounded);
        assert!(!rotated);
    }

    #[test]
    fn interpolate_roi_kind_resamples_wand_trace_points() {
        let roi = RoiKind::WandTrace {
            points: vec![
                egui::pos2(0.0, 0.0),
                egui::pos2(4.0, 0.0),
                egui::pos2(4.0, 4.0),
            ],
        };
        let RoiKind::WandTrace { points } =
            interpolate_roi_kind(&roi, 2.0, false, true).expect("interpolate")
        else {
            panic!("expected wand trace");
        };

        assert!(points.len() > 3);
        assert_eq!(points[0], egui::pos2(0.0, 0.0));
        assert!(points.iter().any(|point| point.x == 4.0 && point.y > 0.0));
    }

    #[test]
    fn viewer_sort_key_uses_numeric_suffix() {
        assert_eq!(viewer_sort_key("viewer-42"), 42);
        assert_eq!(viewer_sort_key("viewer-abc"), u64::MAX);
    }

    #[test]
    fn imagej_macro_run_line_maps_menu_label_to_command() {
        let catalog = crate::ui::command_registry::command_catalog();
        let invocation = parse_macro_command_line(
            r#"run("Enhance Contrast...", "saturated=0.35 normalize");"#,
            &catalog,
        )
        .expect("macro line should parse")
        .expect("macro line should produce command");

        assert_eq!(invocation.command_id, "process.enhance_contrast");
        assert_eq!(
            invocation.params,
            Some(json!({
                "saturated": 0.35,
                "normalize": true
            }))
        );
    }

    #[test]
    fn macro_command_line_keeps_internal_command_json_format() {
        let catalog = crate::ui::command_registry::command_catalog();
        let invocation =
            parse_macro_command_line(r#"process.filters.gaussian|{"sigma":2.0}"#, &catalog)
                .expect("macro line should parse")
                .expect("macro line should produce command");

        assert_eq!(invocation.command_id, "process.filters.gaussian");
        assert_eq!(invocation.params, Some(json!({ "sigma": 2.0 })));
    }

    #[test]
    fn macro_command_line_tolerates_inline_comments_and_metadata_calls() {
        let catalog = crate::ui::command_registry::command_catalog();
        let invocation = parse_macro_command_line(
            r#"run("URL...", "url=http://example.com/a//b"); // inline comment"#,
            &catalog,
        )
        .expect("macro line should parse")
        .expect("macro line should produce command");

        assert_eq!(invocation.command_id, "file.import.url");
        assert_eq!(
            invocation.params,
            Some(json!({"url": "http://example.com/a//b"}))
        );
        assert!(
            parse_macro_command_line(r#"requires("1.53");"#, &catalog)
                .expect("macro line should parse")
                .is_none()
        );
        assert!(
            parse_macro_command_line("setBatchMode(true);", &catalog)
                .expect("macro line should parse")
                .is_none()
        );
        let set_option =
            parse_macro_command_line(r#"setOption("Stack position", true);"#, &catalog)
                .expect("macro line should parse")
                .expect("macro line should produce command");
        assert_eq!(set_option.command_id, "macro.set_option");
        assert_eq!(
            set_option.params,
            Some(json!({"option": "Stack position", "state": true}))
        );
        let call = parse_macro_command_line(
            r#"call("ij.plugin.frame.Recorder.recordString", "Roi.setDefaultGroup(1);\n");"#,
            &catalog,
        )
        .expect("macro line should parse")
        .expect("macro line should produce command");
        assert_eq!(call.command_id, "macro.call");
        assert_eq!(
            call.params,
            Some(json!({"target": "ij.plugin.frame.Recorder.recordString"}))
        );
        let dialog = parse_macro_command_line(r#"Dialog.create("Set Montage Layout");"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(dialog.command_id, "macro.builtin_call");
        assert_eq!(dialog.params, Some(json!({"target": "Dialog.create"})));
        let stack_slice = parse_macro_command_line("Stack.setSlice(4);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(stack_slice.command_id, "image.stacks.set");
        assert_eq!(stack_slice.params, Some(json!({"slice": 4})));
        let dynamic_stack_slice = parse_macro_command_line("Stack.setSlice(n);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(dynamic_stack_slice.command_id, "macro.builtin_call");
        assert_eq!(
            dynamic_stack_slice.params,
            Some(json!({"target": "Stack.setSlice"}))
        );
        let stroke_width = parse_macro_command_line("Roi.setStrokeWidth(3);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(stroke_width.command_id, "edit.options.line_width");
        assert_eq!(stroke_width.params, Some(json!({"width": 3.0})));
        let roi_name = parse_macro_command_line(r#"Roi.setName("label-1");"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(roi_name.command_id, "macro.set_roi_name");
        assert_eq!(roi_name.params, Some(json!({"name": "label-1"})));
        let selection_name =
            parse_macro_command_line(r#"setSelectionName("selection-1");"#, &catalog)
                .expect("macro line should parse")
                .expect("macro line should produce command");
        assert_eq!(selection_name.command_id, "macro.set_roi_name");
        assert_eq!(selection_name.params, Some(json!({"name": "selection-1"})));
        let metadata = parse_macro_command_line(r#"setMetadata("Info", "x=1");"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(metadata.command_id, "macro.set_metadata");
        assert_eq!(
            metadata.params,
            Some(json!({"key": "Info", "value": "x=1"}))
        );
        let dynamic_metadata =
            parse_macro_command_line(r#"setMetadata("Info", "x="+width);"#, &catalog)
                .expect("macro line should parse")
                .expect("macro line should produce command");
        assert_eq!(dynamic_metadata.command_id, "macro.builtin_call");
        assert_eq!(
            dynamic_metadata.params,
            Some(json!({"target": "setMetadata"}))
        );
        let property =
            parse_macro_command_line(r#"Property.set("CompositeProjection", "Sum");"#, &catalog)
                .expect("macro line should parse")
                .expect("macro line should produce command");
        assert_eq!(property.command_id, "macro.set_metadata");
        assert_eq!(
            property.params,
            Some(json!({"key": "CompositeProjection", "value": "Sum"}))
        );
        let overlay_add =
            parse_macro_command_line(r##"Overlay.addSelection("#ff0000", 2);"##, &catalog)
                .expect("macro line should parse")
                .expect("macro line should produce command");
        assert_eq!(overlay_add.command_id, "image.overlay.add_selection");
        assert_eq!(overlay_add.params, None);
        let overlay_show = parse_macro_command_line("Overlay.show;", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(overlay_show.command_id, "image.overlay.show");
        assert_eq!(overlay_show.params, None);
        let overlay_hide = parse_macro_command_line("Overlay.hide();", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(overlay_hide.command_id, "image.overlay.hide");
        assert_eq!(overlay_hide.params, None);
        let overlay_remove = parse_macro_command_line(
            r#"Overlay.removeRois("ToolSelectedOverlayElement");"#,
            &catalog,
        )
        .expect("macro line should parse")
        .expect("macro line should produce command");
        assert_eq!(overlay_remove.command_id, "macro.remove_overlay_rois");
        assert_eq!(
            overlay_remove.params,
            Some(json!({"name": "ToolSelectedOverlayElement"}))
        );
        let overlay_remove_index =
            parse_macro_command_line("Overlay.removeSelection(2);", &catalog)
                .expect("macro line should parse")
                .expect("macro line should produce command");
        assert_eq!(
            overlay_remove_index.command_id,
            "macro.remove_overlay_selection"
        );
        assert_eq!(overlay_remove_index.params, Some(json!({"index": 2})));
        let overlay_activate = parse_macro_command_line("Overlay.activateSelection(1);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(
            overlay_activate.command_id,
            "macro.activate_overlay_selection"
        );
        assert_eq!(overlay_activate.params, Some(json!({"index": 1})));
        let select_none = parse_macro_command_line("selectNone();", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(select_none.command_id, "edit.selection.none");
        let set_slice = parse_macro_command_line("setSlice(3);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(set_slice.command_id, "image.stacks.set");
        assert_eq!(set_slice.params, Some(json!({"slice": 3})));
        let dynamic_set_slice = parse_macro_command_line("setSlice(n);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(dynamic_set_slice.command_id, "macro.builtin_call");
        assert_eq!(
            dynamic_set_slice.params,
            Some(json!({"target": "setSlice"}))
        );
        let dynamic_select_image = parse_macro_command_line("selectImage(id);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(dynamic_select_image.command_id, "macro.builtin_call");
        assert_eq!(
            dynamic_select_image.params,
            Some(json!({"target": "selectImage"}))
        );
        let literal_select_image = parse_macro_command_line("selectImage(3);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(literal_select_image.command_id, "macro.select_image");
        assert_eq!(literal_select_image.params, Some(json!({"id": 3})));
        let arrow_tool = parse_macro_command_line(r#"setTool("arrow");"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(arrow_tool.command_id, "macro.set_tool");
        assert_eq!(
            arrow_tool.params,
            Some(json!({
                "tool": "launcher.tool.line",
                "mode": "launcher.tool.line.mode.arrow"
            }))
        );
        let numeric_tool = parse_macro_command_line("setTool(7);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(numeric_tool.command_id, "macro.set_tool");
        assert_eq!(
            numeric_tool.params,
            Some(json!({
                "tool": "launcher.tool.point",
                "mode": "launcher.tool.point.mode.point"
            }))
        );
        let unknown_tool = parse_macro_command_line(r#"setTool("not-a-tool");"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(unknown_tool.command_id, "macro.builtin_call");
        assert_eq!(unknown_tool.params, Some(json!({"target": "setTool"})));
        let paste_mode = parse_macro_command_line(r#"setPasteMode("add");"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(paste_mode.command_id, "macro.set_paste_mode");
        assert_eq!(paste_mode.params, Some(json!({"mode": "add"})));
        let make_rect = parse_macro_command_line("makeRectangle(x, y, w, h);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(make_rect.command_id, "macro.builtin_call");
        assert_eq!(make_rect.params, Some(json!({"target": "makeRectangle"})));
        let literal_rect = parse_macro_command_line("makeRectangle(0, 1, 256, 32);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(literal_rect.command_id, "macro.make_rectangle");
        assert_eq!(
            literal_rect.params,
            Some(json!({"x": 0.0, "y": 1.0, "width": 256.0, "height": 32.0}))
        );
        let literal_oval = parse_macro_command_line("makeOval(1, 2, 3, 4);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(literal_oval.command_id, "macro.make_oval");
        let literal_line = parse_macro_command_line("makeLine(1, 2, 3, 4);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(literal_line.command_id, "macro.make_line");
        assert_eq!(
            literal_line.params,
            Some(json!({"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0}))
        );
        let literal_polyline = parse_macro_command_line("makeLine(1, 2, 3, 4, 5, 6);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(literal_polyline.command_id, "macro.make_selection");
        assert_eq!(
            literal_polyline.params,
            Some(json!({
                "selection_type": "polyline",
                "points": [
                    {"x": 1.0, "y": 2.0},
                    {"x": 3.0, "y": 4.0},
                    {"x": 5.0, "y": 6.0}
                ]
            }))
        );
        let literal_polygon = parse_macro_command_line("makePolygon(1, 2, 3, 4);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(literal_polygon.command_id, "macro.make_selection");
        let literal_point = parse_macro_command_line("makePoint(1, 2);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(literal_point.command_id, "macro.make_selection");
        assert_eq!(
            literal_point.params,
            Some(json!({
                "selection_type": "point",
                "points": [
                    {"x": 1.0, "y": 2.0}
                ]
            }))
        );
        let literal_selection = parse_macro_command_line(
            r#"makeSelection("polygon", 0, 0, 10, 0, 10, 10);"#,
            &catalog,
        )
        .expect("macro line should parse")
        .expect("macro line should produce command");
        assert_eq!(literal_selection.command_id, "macro.make_selection");
        assert_eq!(
            literal_selection.params,
            Some(json!({
                "selection_type": "polygon",
                "points": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 10.0, "y": 0.0},
                    {"x": 10.0, "y": 10.0}
                ]
            }))
        );
        let new_image =
            parse_macro_command_line(r#"newImage("luts", "RGB White", 256, 48, 1);"#, &catalog)
                .expect("macro line should parse")
                .expect("macro line should produce command");
        assert_eq!(new_image.command_id, "file.new");
        assert_eq!(
            new_image.params,
            Some(json!({
                "title": "luts",
                "width": 256,
                "height": 48,
                "slices": 1,
                "channels": 3,
                "frames": 1,
                "fill": 1.0,
                "pixelType": "u8"
            }))
        );
        let select_window = parse_macro_command_line(r#"selectWindow("luts");"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(select_window.command_id, "macro.select_window");
        assert_eq!(select_window.params, Some(json!({"title": "luts"})));
        let foreground = parse_macro_command_line("setForegroundColor(255, 128, 0);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(foreground.command_id, "macro.set_color");
        assert_eq!(
            foreground.params,
            Some(json!({
                "target": "foreground",
                "red": 255,
                "green": 128,
                "blue": 0
            }))
        );
        let background = parse_macro_command_line("setBackgroundColor(r, g, b);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(background.command_id, "macro.builtin_call");
        assert_eq!(
            background.params,
            Some(json!({"target": "setBackgroundColor"}))
        );
        let rename = parse_macro_command_line(r#"rename("Lookup Tables");"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(rename.command_id, "image.rename");
        assert_eq!(rename.params, Some(json!({"title": "Lookup Tables"})));
        let dynamic_rename = parse_macro_command_line("rename(title);", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(dynamic_rename.command_id, "macro.builtin_call");
        assert_eq!(dynamic_rename.params, Some(json!({"target": "rename"})));
        let open = parse_macro_command_line(r#"open("/tmp/source.tif");"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(open.command_id, "file.open");
        assert_eq!(open.params, Some(json!({"path": "/tmp/source.tif"})));
        let open_url = parse_macro_command_line(r#"open("https://example.com/a.tif");"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(open_url.command_id, "file.import.url");
        assert_eq!(
            open_url.params,
            Some(json!({"url": "https://example.com/a.tif"}))
        );
        let save = parse_macro_command_line(r#"save("/tmp/output.tif");"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(save.command_id, "file.save_as");
        assert_eq!(save.params, Some(json!({"path": "/tmp/output.tif"})));
        let save_as = parse_macro_command_line(r#"saveAs("Tiff", "/tmp/output");"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(save_as.command_id, "file.save_as");
        assert_eq!(
            save_as.params,
            Some(json!({"format": "Tiff", "path": "/tmp/output"}))
        );
        let close = parse_macro_command_line("close();", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(close.command_id, "file.close");
        assert_eq!(close.params, None);
        let bare_close = parse_macro_command_line("close;", &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(bare_close.command_id, "file.close");
        assert_eq!(bare_close.params, None);
        let close_title = parse_macro_command_line(r#"close("About ImageJ");"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(close_title.command_id, "macro.close_window");
        assert_eq!(close_title.params, Some(json!({"title": "About ImageJ"})));
    }

    #[test]
    fn macro_command_line_accepts_imagej_string_spacing_and_single_quotes() {
        let catalog = crate::ui::command_registry::command_catalog();
        let spaced = parse_macro_command_line(r#"run( "Select None" );"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        let select_all = parse_macro_command_line(r#"run("Select All");"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        let single_quoted = parse_macro_command_line(r#"run('Smooth');"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");

        assert_eq!(spaced.command_id, "edit.selection.none");
        assert_eq!(select_all.command_id, "edit.selection.all");
        assert_eq!(single_quoted.command_id, "process.smooth");
        assert_eq!(
            parse_macro_command_line(r#"run("Clear", "slice");"#, &catalog)
                .expect("macro line should parse")
                .expect("macro line should produce command")
                .command_id,
            "edit.clear"
        );
        assert_eq!(
            parse_macro_command_line(r#"run("Fill");"#, &catalog)
                .expect("macro line should parse")
                .expect("macro line should produce command")
                .command_id,
            "edit.fill"
        );
        assert_eq!(
            parse_macro_command_line(r#"run("Internal Clipboard");"#, &catalog)
                .expect("macro line should parse")
                .expect("macro line should produce command")
                .command_id,
            "edit.internal_clipboard"
        );
        assert_eq!(
            parse_macro_command_line(r#"run("Find Commands...");"#, &catalog)
                .expect("macro line should parse")
                .expect("macro line should produce command")
                .command_id,
            "plugins.commands.find"
        );
        assert_eq!(
            parse_macro_command_line(r#"run("Interpolate", "interval=1 adjust");"#, &catalog)
                .expect("macro line should parse")
                .expect("macro line should produce command")
                .command_id,
            "edit.selection.interpolate"
        );
        assert_eq!(
            parse_macro_command_line(r#"run("Properties... ");"#, &catalog)
                .expect("macro line should parse")
                .expect("macro line should produce command")
                .command_id,
            "edit.selection.properties"
        );
        assert_eq!(
            parse_macro_command_line(r#"run("Make Composite");"#, &catalog)
                .expect("macro line should parse")
                .expect("macro line should produce command")
                .command_id,
            "image.type.make_composite"
        );
        let duplicate = parse_macro_command_line(r#"run("Duplicate...", "title=temp");"#, &catalog)
            .expect("macro line should parse")
            .expect("macro line should produce command");
        assert_eq!(duplicate.command_id, "image.duplicate");
        assert_eq!(duplicate.params, Some(json!({"title": "temp"})));
        let resize = parse_macro_command_line(
            r#"run("Size...", "width=128 height=64 constrain interpolate");"#,
            &catalog,
        )
        .expect("macro line should parse")
        .expect("macro line should produce command");
        assert_eq!(resize.command_id, "image.adjust.size");
        assert_eq!(
            resize.params,
            Some(json!({
                "width": 128,
                "height": 64,
                "constrain": true,
                "interpolate": true
            }))
        );
        let canvas = parse_macro_command_line(
            r#"run("Canvas Size...", "width=258 height=50 position=Center zero");"#,
            &catalog,
        )
        .expect("macro line should parse")
        .expect("macro line should produce command");
        assert_eq!(canvas.command_id, "image.adjust.canvas");
        assert_eq!(
            canvas.params,
            Some(json!({
                "width": 258,
                "height": 50,
                "position": "Center",
                "zero": true
            }))
        );
        assert_eq!(
            strip_macro_line_comment(r#"run('URL...', 'url=http://x//y'); // comment"#),
            r#"run('URL...', 'url=http://x//y'); "#
        );
    }

    #[test]
    fn macro_source_lines_skip_macro_set_wrappers() {
        let lines = macro_source_executable_lines(
            r#"
                macro "Smooth Once" {
                    run("Smooth");
                }
                macro "Two Commands" { run("Sharpen"); run("Find Edges"); }
            "#,
        );

        assert_eq!(
            lines,
            vec![
                (3, r#"run("Smooth")"#.to_string()),
                (5, r#"run("Sharpen")"#.to_string()),
                (5, r#"run("Find Edges")"#.to_string()),
            ]
        );
    }

    #[test]
    fn macro_source_named_blocks_extract_individual_macros() {
        let blocks = macro_source_named_blocks(
            r#"
                macro "Smooth Once [F1]" {
                    run("Smooth");
                }
                macro 'Single Quoted Tool' {
                    run("Find Edges");
                }
                macro "Two Commands" { run("Sharpen"); run("Find Edges"); }
            "#,
        );

        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].name, "Smooth Once [F1]");
        assert_eq!(blocks[0].shortcut.as_deref(), Some("F1"));
        assert_eq!(
            blocks[0].statements,
            vec![(3, r#"run("Smooth")"#.to_string())]
        );
        assert_eq!(blocks[1].name, "Single Quoted Tool");
        assert_eq!(
            blocks[1].statements,
            vec![(6, r#"run("Find Edges")"#.to_string())]
        );
        assert_eq!(blocks[2].name, "Two Commands");
        assert_eq!(
            blocks[2].statements,
            vec![
                (8, r#"run("Sharpen")"#.to_string()),
                (8, r#"run("Find Edges")"#.to_string()),
            ]
        );
    }

    #[test]
    fn macro_named_block_statement_map_resolves_local_macro_calls() {
        let blocks = macro_named_block_statement_map(
            r#"
                macro "Set Montage Layout [F2]" { run("Make Montage..."); }
                macro "<Stacks> Change Montage Layout" { run("Set Montage Layout"); }
            "#,
        );

        assert_eq!(
            blocks.get("Set Montage Layout").cloned(),
            Some(vec![(2, r#"run("Make Montage...")"#.to_string())])
        );
        assert_eq!(
            blocks.get("Change Montage Layout").cloned(),
            Some(vec![(3, r#"run("Set Montage Layout")"#.to_string())])
        );
    }

    #[test]
    fn startup_auto_run_macro_block_selects_first_autorun_macro() {
        let block = startup_auto_run_macro_block(
            r#"
                macro "Regular Startup" {
                    run("Smooth");
                }
                macro "AutoRunAndHide" { run("Find Edges"); }
                macro "AutoRun" { run("Sharpen"); }
            "#,
        )
        .expect("AutoRun macro");

        assert_eq!(block.name, "AutoRunAndHide");
        assert_eq!(
            block.statements,
            vec![(5, r#"run("Find Edges")"#.to_string())]
        );
    }

    #[test]
    fn installed_macro_menu_entry_uses_submenus_and_skips_startup_helpers() {
        let path = Path::new("StartupMacros.ijm");
        let blocks = macro_source_named_blocks(
            r#"
                macro "<Stacks> Z Project [F2]" { run("Z Project..."); }
                macro "AutoRun" { run("Smooth"); }
                macro "Line Tool Selected" { run("Measure"); }
            "#,
        );

        let entry = installed_macro_menu_entry_from_block(path, &blocks[0]).expect("menu entry");
        assert_eq!(entry.label, "Z Project");
        assert_eq!(entry.shortcut.as_deref(), Some("F2"));
        assert_eq!(entry.submenu.as_deref(), Some("Stacks"));
        assert_eq!(entry.macro_name, "<Stacks> Z Project [F2]");

        assert!(installed_macro_menu_entry_from_block(path, &blocks[1]).is_none());
        assert!(installed_macro_menu_entry_from_block(path, &blocks[2]).is_none());
    }

    #[test]
    fn macro_name_shortcut_matches_imagej_bracket_rules() {
        assert_eq!(macro_name_shortcut("Smooth [a]").as_deref(), Some("a"));
        assert_eq!(macro_name_shortcut("Smooth [f1]").as_deref(), Some("F1"));
        assert_eq!(macro_name_shortcut("Smooth [n+]").as_deref(), Some("N+"));
        assert_eq!(macro_name_shortcut("Smooth [&1]").as_deref(), Some("&1"));
        assert_eq!(macro_name_shortcut("Smooth [F13]").as_deref(), None);
        assert_eq!(macro_name_shortcut("Smooth [.]").as_deref(), None);
        assert_eq!(macro_name_shortcut("Smooth [ctrl]").as_deref(), None);
        assert_eq!(macro_display_name("Smooth [F1]"), "Smooth");
    }

    #[test]
    fn macro_shortcut_matching_accepts_text_and_function_keys() {
        assert!(macro_shortcut_matches_text("a", "a"));
        assert!(macro_shortcut_matches_text("a", "A"));
        assert!(macro_shortcut_matches_text("N+", "+"));
        assert!(macro_shortcut_matches_text("&1", "1"));
        assert!(!macro_shortcut_matches_text("F1", "F1"));
        assert_eq!(function_key_for_macro_shortcut("f1"), Some(egui::Key::F1));
        assert_eq!(function_key_for_macro_shortcut("F13"), None);
    }

    #[test]
    fn macro_options_keep_bracketed_text_values_together() {
        assert_eq!(
            macro_options_to_json("title=[My Image] depth=16 modal=false"),
            json!({
                "title": "My Image",
                "depth": 16,
                "modal": false
            })
        );
        assert_eq!(
            macro_options_to_json("columns=4 rows=3 scale=1 border=2"),
            json!({
                "columns": 4,
                "rows": 3,
                "scale": 1,
                "border_width": 2
            })
        );
    }

    #[test]
    fn macro_recorder_writes_imagej_run_calls_with_options() {
        let catalog = crate::ui::command_registry::command_catalog();
        let line = macro_record_line_for_command(
            "process.enhance_contrast",
            Some(&json!({
                "saturated": 0.35,
                "normalize": true,
                "title": "My Image",
                "optional": null
            })),
            &catalog,
        )
        .expect("recordable command");

        assert_eq!(
            line,
            r#"run("Enhance Contrast...", "normalize saturated=0.35 title=[My Image]");"#
        );
    }

    #[test]
    fn macro_recorder_omits_empty_options() {
        let catalog = crate::ui::command_registry::command_catalog();
        let line = macro_record_line_for_command("process.smooth", None, &catalog)
            .expect("recordable command");

        assert_eq!(line, r#"run("Smooth");"#);
    }

    #[test]
    fn installed_macro_file_name_matches_imagej_txt_rule() {
        assert_eq!(
            installed_macro_file_name(Path::new("Example.txt")).expect("txt macro"),
            "Example.ijm"
        );
        assert_eq!(
            installed_macro_file_name(Path::new("Tool_Macro.txt")).expect("underscore txt macro"),
            "Tool_Macro.txt"
        );
        assert_eq!(
            installed_macro_file_name(Path::new("Already.ijm")).expect("ijm macro"),
            "Already.ijm"
        );
        assert!(installed_macro_file_name(Path::new("Plugin.jar")).is_err());
    }

    #[test]
    fn install_macro_file_copies_to_install_dir() {
        let dir = tempdir().expect("tempdir");
        let source = dir.path().join("Example.txt");
        let install_dir = dir.path().join("installed");
        fs::write(&source, r#"run("Smooth");"#).expect("write source macro");

        let installed = install_macro_file_to_dir(&source, &install_dir).expect("install macro");

        assert_eq!(installed, install_dir.join("Example.ijm"));
        assert_eq!(
            fs::read_to_string(installed).expect("read installed macro"),
            r#"run("Smooth");"#
        );
    }

    #[test]
    fn installed_macro_listing_filters_and_sorts_macros() {
        let dir = tempdir().expect("tempdir");
        fs::write(dir.path().join("B.ijm"), "").expect("write B");
        fs::write(dir.path().join("A.txt"), "").expect("write A");
        fs::write(dir.path().join("plugin.jar"), "").expect("write jar");

        let macros = list_installed_macro_files_in_dir(dir.path())
            .into_iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().to_string())
            .collect::<Vec<_>>();

        assert_eq!(macros, vec!["A.txt", "B.ijm"]);
    }

    #[test]
    fn startup_macro_status_uses_first_report_line() {
        assert_eq!(
            first_report_line("Executed: 1, blocked: 0\n1: process.smooth -> ok"),
            "Executed: 1, blocked: 0"
        );
        assert_eq!(first_report_line(""), "");
    }

    #[test]
    fn tool_command_mapping_matches_expected_tool_ids() {
        assert_eq!(
            tool_from_command_id("launcher.tool.rect"),
            Some(ToolId::Rect)
        );
        assert_eq!(
            tool_from_command_id("launcher.tool.zoom"),
            Some(ToolId::Zoom)
        );
        assert_eq!(
            tool_from_command_id("launcher.tool.hand"),
            Some(ToolId::Hand)
        );
        assert_eq!(
            tool_from_command_id("launcher.tool.point"),
            Some(ToolId::Point)
        );
        assert_eq!(
            tool_from_command_id("launcher.tool.angle"),
            Some(ToolId::Angle)
        );
        assert_eq!(tool_from_command_id("launcher.tool.unknown"), None);
    }

    #[test]
    fn tool_shortcut_mapping_is_deterministic() {
        assert_eq!(tool_shortcut_command("r"), Some("launcher.tool.rect"));
        assert_eq!(tool_shortcut_command("R"), Some("launcher.tool.rect"));
        assert_eq!(tool_shortcut_command("h"), Some("launcher.tool.hand"));
        assert_eq!(tool_shortcut_command("a"), Some("launcher.tool.angle"));
        assert_eq!(tool_shortcut_command("."), Some("launcher.tool.point"));
        assert_eq!(tool_shortcut_command("?"), None);
    }

    #[test]
    fn centered_circular_roi_uses_imagej_default_radius() {
        let roi = centered_circular_roi(&[80, 120], None).expect("roi");
        let RoiKind::Oval {
            start,
            end,
            ellipse,
            brush,
        } = roi
        else {
            panic!("expected oval ROI");
        };

        assert_eq!(start, egui::pos2(30.0, 10.0));
        assert_eq!(end, egui::pos2(90.0, 70.0));
        assert!(!ellipse);
        assert!(!brush);
    }

    #[test]
    fn centered_circular_roi_caps_radius_to_image_bounds() {
        let roi = centered_circular_roi(&[40, 120], Some(100.0)).expect("roi");
        let RoiKind::Oval { start, end, .. } = roi else {
            panic!("expected oval ROI");
        };

        assert_eq!(start, egui::pos2(40.0, 0.0));
        assert_eq!(end, egui::pos2(80.0, 40.0));
        assert!(centered_circular_roi(&[40, 120], Some(-1.0)).is_err());
    }

    #[test]
    fn launcher_status_formatter_no_active_viewer() {
        let status = LauncherStatusModel {
            active_viewer: None,
            text: String::new(),
            progress: ProgressState::Idle,
        };
        let text = format_launcher_status(
            ToolId::Rect,
            &status,
            None,
            &HashMap::new(),
            "Ready for file open",
        );
        assert!(text.contains("Tool:Rect"));
        assert!(text.contains("No image"));
    }

    #[test]
    fn launcher_status_formatter_active_hover_sample() {
        let status = LauncherStatusModel {
            active_viewer: Some("viewer-1".to_string()),
            text: String::new(),
            progress: ProgressState::Idle,
        };
        let mut paths = HashMap::new();
        paths.insert("viewer-1".to_string(), PathBuf::from("/tmp/sample.tif"));
        let telemetry = ViewerTelemetry {
            hover: Some(HoverInfo {
                x: 9,
                y: 4,
                value: 0.75,
            }),
            pinned: None,
            zoom: 1.5,
            z: 2,
            t: 0,
            c: 1,
            active_job: None,
        };

        let text = format_launcher_status(ToolId::Point, &status, Some(&telemetry), &paths, "");
        assert!(text.contains("sample.tif"));
        assert!(text.contains("X:9 Y:4"));
        assert!(text.contains("Z:2 T:0 C:1"));
    }

    #[test]
    fn launcher_status_formatter_prefers_pinned_and_job() {
        let status = LauncherStatusModel {
            active_viewer: Some("viewer-2".to_string()),
            text: String::new(),
            progress: ProgressState::Running("gaussian.blur (apply)".to_string()),
        };
        let mut paths = HashMap::new();
        paths.insert("viewer-2".to_string(), PathBuf::from("/tmp/stack.tiff"));
        let telemetry = ViewerTelemetry {
            hover: Some(HoverInfo {
                x: 1,
                y: 2,
                value: 3.0,
            }),
            pinned: Some(HoverInfo {
                x: 5,
                y: 6,
                value: 7.0,
            }),
            zoom: 2.0,
            z: 4,
            t: 1,
            c: 0,
            active_job: Some("gaussian.blur (apply)".to_string()),
        };

        let text = format_launcher_status(ToolId::Zoom, &status, Some(&telemetry), &paths, "");
        assert!(text.contains("X:5 Y:6"));
        assert!(text.contains("200%"));
        assert!(text.contains("gaussian.blur (apply)"));
    }

    #[test]
    fn viewer_session_summary_updates_on_commit() {
        let path = PathBuf::from("/tmp/summary-update.tif");
        let mut session = ViewerSession::new(
            path.clone(),
            ViewerImageSource::Dataset(dataset_2x2([0.0, 1.0, 2.0, 3.0])),
        );

        assert_eq!(session.committed_summary.min, 0.0);
        assert_eq!(session.committed_summary.max, 3.0);
        let summary_before_preview = session.committed_summary.clone();

        session.set_active_preview(Some("preview-1".to_string()));
        assert_eq!(session.committed_summary, summary_before_preview);

        session.commit_dataset(dataset_2x2([10.0, 11.0, 12.0, 13.0]));
        assert_eq!(session.committed_summary.min, 10.0);
        assert_eq!(session.committed_summary.max, 13.0);
        assert_eq!(session.committed_summary.source, path.display().to_string());
    }

    #[test]
    fn viewer_session_tracks_undo_redo_and_revert() {
        let path = PathBuf::from("/tmp/history.tif");
        let original = dataset_2x2([0.0, 1.0, 2.0, 3.0]);
        let mut session = ViewerSession::new(path, ViewerImageSource::Dataset(original.clone()));
        let updated = dataset_2x2([10.0, 11.0, 12.0, 13.0]);

        session.commit_dataset(updated.clone());
        assert!(session.can_undo());
        assert!(!session.can_redo());
        assert_eq!(session.committed_summary.min, 10.0);

        assert!(session.undo());
        assert_eq!(session.committed_summary.min, 0.0);
        assert!(!session.can_undo());
        assert!(session.can_redo());

        assert!(session.redo());
        assert_eq!(session.committed_summary.min, 10.0);
        assert!(session.can_undo());

        assert!(session.revert_to_base());
        assert_eq!(session.committed_summary.min, 0.0);
        assert!(!session.can_undo());
        assert!(!session.can_redo());
        assert!(!session.revert_to_base());

        session.commit_dataset(updated);
        session.mark_saved(PathBuf::from("/tmp/history-saved.tif"));
        assert_eq!(session.committed_summary.source, "/tmp/history-saved.tif");
        assert!(!session.can_undo());
        assert!(!session.can_redo());
        assert!(source_ptr_eq(
            &session.base_source,
            &session.committed_source
        ));
    }

    #[test]
    fn native_viewer_session_promotes_once_for_processing() {
        let path = PathBuf::from("/tmp/native-history.png");
        let native = native_gray8_2x2([0, 64, 128, 255]);
        let mut session = ViewerSession::new(path, ViewerImageSource::Native(native));

        assert!(matches!(
            session.committed_source,
            ViewerImageSource::Native(_)
        ));

        let first = session
            .ensure_committed_dataset()
            .expect("first materialization");
        let second = session
            .ensure_committed_dataset()
            .expect("second materialization");

        assert!(Arc::ptr_eq(&first, &second));
        assert!(matches!(
            session.committed_source,
            ViewerImageSource::Dataset(_)
        ));
        assert!(matches!(session.base_source, ViewerImageSource::Native(_)));
    }

    #[test]
    fn compute_viewer_frame_reuses_arc_from_cache() {
        let mut state = UiState::new(None);
        let label = "viewer-1".to_string();
        let session = ViewerSession::new(
            PathBuf::from("/tmp/frame-cache.tif"),
            ViewerImageSource::Dataset(dataset_2x2([0.0, 1.0, 2.0, 3.0])),
        );
        state.label_to_session.insert(label.clone(), session);

        let request = ViewerFrameRequest::default();
        let first = compute_viewer_frame(&mut state, &label, &request, None).expect("first frame");
        let second =
            compute_viewer_frame(&mut state, &label, &request, None).expect("cached second frame");

        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn native_and_dataset_frames_match_for_gray8() {
        let native = ViewerImageSource::Native(native_gray8_2x2([0, 64, 128, 255]));
        let dataset = ViewerImageSource::Dataset(
            native_gray8_2x2([0, 64, 128, 255])
                .to_dataset()
                .map(Arc::new)
                .expect("dataset"),
        );
        let request = ViewerFrameRequest::default();

        let native_frame = build_frame(&native, &request, None).expect("native frame");
        let dataset_frame = build_frame(&dataset, &request, None).expect("dataset frame");

        assert_eq!(native_frame.width, dataset_frame.width);
        assert_eq!(native_frame.height, dataset_frame.height);
        assert_eq!(native_frame.pixels_u8, dataset_frame.pixels_u8);
        assert_eq!(native_frame.min, dataset_frame.min);
        assert_eq!(native_frame.max, dataset_frame.max);
        assert_eq!(dataset_frame.values, vec![0.0, 64.0, 128.0, 255.0]);
    }

    #[test]
    fn dataset_frame_does_not_auto_stretch_display_range() {
        let dataset = ViewerImageSource::Dataset(dataset_2x2([0.25, 0.5, 0.75, 1.0]));
        let frame = build_frame(&dataset, &ViewerFrameRequest::default(), None).expect("frame");

        assert_eq!(frame.pixels_u8, vec![64, 128, 191, 255]);
        assert_eq!(frame.values, vec![0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn dataset_frame_maps_raw_integer_samples_for_display() {
        let dataset = ViewerImageSource::Dataset(dataset_2x2_with_pixel_type(
            [0.0, 64.0, 128.0, 255.0],
            PixelType::U8,
        ));
        let frame = build_frame(&dataset, &ViewerFrameRequest::default(), None).expect("frame");

        assert_eq!(frame.pixels_u8, vec![0, 64, 128, 255]);
        assert_eq!(frame.values, vec![0.0, 64.0, 128.0, 255.0]);
        assert_eq!(frame.min, 0.0);
        assert_eq!(frame.max, 255.0);
    }

    #[test]
    fn dataset_frame_applies_display_range_without_changing_values() {
        let dataset = ViewerImageSource::Dataset(dataset_2x2_with_pixel_type(
            [0.0, 5.0, 10.0, 20.0],
            PixelType::U8,
        ));
        let frame = build_frame(&dataset, &ViewerFrameRequest::default(), Some((0.0, 10.0)))
            .expect("frame");

        assert_eq!(frame.pixels_u8, vec![0, 128, 255, 255]);
        assert_eq!(frame.values, vec![0.0, 5.0, 10.0, 20.0]);
        assert_eq!(frame.min, 0.0);
        assert_eq!(frame.max, 20.0);
    }

    #[test]
    fn compute_viewer_frame_uses_session_display_range_without_committing_pixels() {
        let mut state = UiState::new(None);
        let label = "viewer-1".to_string();
        let original = dataset_2x2_with_pixel_type([0.0, 5.0, 10.0, 20.0], PixelType::U8);
        let mut session = ViewerSession::new(
            PathBuf::from("/tmp/display-range.tif"),
            ViewerImageSource::Dataset(original.clone()),
        );
        session.set_display_range(Some((0.0, 10.0)));
        state.label_to_session.insert(label.clone(), session);

        let frame = compute_viewer_frame(&mut state, &label, &ViewerFrameRequest::default(), None)
            .expect("frame");

        assert_eq!(frame.pixels_u8, vec![0, 128, 255, 255]);
        assert_eq!(frame.values, vec![0.0, 5.0, 10.0, 20.0]);
        let session = state.label_to_session.get(&label).expect("session");
        let committed = session.committed_dataset().expect("committed dataset");
        assert!(Arc::ptr_eq(&committed, &original));
        assert!(!session.can_undo());
        assert_eq!(session.committed_summary.min, 0.0);
        assert_eq!(session.committed_summary.max, 20.0);
    }

    #[test]
    fn compute_viewer_frame_prefers_channel_display_range() {
        let data = Array::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                0.0, 0.0, 0.0, //
                0.0, 5.0, 0.0, //
                0.0, 10.0, 0.0, //
                0.0, 20.0, 0.0,
            ],
        )
        .expect("shape");
        let dataset = Arc::new(
            DatasetF32::new(
                data,
                Metadata {
                    dims: vec![
                        Dim::new(AxisKind::Y, 2),
                        Dim::new(AxisKind::X, 2),
                        Dim::new(AxisKind::Channel, 3),
                    ],
                    pixel_type: PixelType::U8,
                    ..Metadata::default()
                },
            )
            .expect("dataset"),
        );
        let mut state = UiState::new(None);
        let label = "viewer-1".to_string();
        let mut session = ViewerSession::new(
            PathBuf::from("/tmp/channel-display-range.tif"),
            ViewerImageSource::Dataset(dataset.clone()),
        );
        session.set_display_range(Some((0.0, 255.0)));
        session.set_channel_display_range(1, Some((0.0, 10.0)));
        state.label_to_session.insert(label.clone(), session);

        let frame = compute_viewer_frame(
            &mut state,
            &label,
            &ViewerFrameRequest {
                channel: 1,
                ..ViewerFrameRequest::default()
            },
            None,
        )
        .expect("channel frame");

        assert_eq!(frame.values, vec![0.0, 5.0, 10.0, 20.0]);
        assert_eq!(frame.pixels_u8, vec![0, 128, 255, 255]);
        let session = state.label_to_session.get(&label).expect("session");
        assert!(Arc::ptr_eq(
            &session.committed_dataset().expect("committed"),
            &dataset
        ));
        assert!(!session.can_undo());
    }

    #[test]
    fn adjust_apply_lut_rejects_float_images_like_imagej() {
        let label = "viewer-1".to_string();
        let dataset = dataset_2x2_with_pixel_type([0.0, 0.25, 0.5, 1.0], PixelType::F32);
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/float-apply-lut.tif"),
                ViewerImageSource::Dataset(dataset),
            ),
        );
        app.viewers_ui.insert(
            label.clone(),
            ViewerUiState::new(&label, "float-apply-lut".to_string()),
        );

        let brightness = app.dispatch_command(
            &label,
            "image.adjust.brightness",
            Some(json!({"min": 0.0, "max": 1.0, "apply": true})),
        );
        assert!(matches!(
            brightness.status,
            crate::ui::command_registry::CommandExecuteStatus::Blocked
        ));
        assert!(brightness.message.contains("32-bit"));

        let window_level = app.dispatch_command(
            &label,
            "image.adjust.window_level",
            Some(json!({"low": 0.0, "high": 1.0, "apply": true})),
        );
        assert!(matches!(
            window_level.status,
            crate::ui::command_registry::CommandExecuteStatus::Blocked
        ));
        assert!(window_level.message.contains("32-bit"));
    }

    #[test]
    fn adjust_apply_lut_prompts_before_changing_integer_pixels() {
        let byte_label = "viewer-1".to_string();
        let float_label = "viewer-2".to_string();
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            byte_label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/apply-lut-u8.tif"),
                ViewerImageSource::Dataset(dataset_2x2_with_pixel_type(
                    [0.0, 64.0, 128.0, 255.0],
                    PixelType::U8,
                )),
            ),
        );
        app.state.label_to_session.insert(
            float_label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/apply-lut-f32.tif"),
                ViewerImageSource::Dataset(dataset_2x2_with_pixel_type(
                    [0.0, 0.25, 0.5, 1.0],
                    PixelType::F32,
                )),
            ),
        );

        assert!(app.apply_lut_needs_confirmation(&byte_label));
        assert!(!app.apply_lut_needs_confirmation(&float_label));
    }

    #[test]
    fn adjust_apply_lut_stack_prompt_can_target_current_slice_like_imagej() {
        let label = "viewer-1".to_string();
        let data = Array::from_shape_vec(IxDyn(&[1, 1, 2]), vec![10.0, 20.0]).expect("shape");
        let metadata = Metadata {
            dims: vec![
                Dim::new(AxisKind::Y, 1),
                Dim::new(AxisKind::X, 1),
                Dim::new(AxisKind::Z, 2),
            ],
            pixel_type: PixelType::U8,
            ..Metadata::default()
        };
        let dataset = Arc::new(DatasetF32::new(data, metadata).expect("dataset"));
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/apply-lut-stack.tif"),
                ViewerImageSource::Dataset(dataset),
            ),
        );
        let mut viewer = ViewerUiState::new(&label, "apply-lut-stack".to_string());
        viewer.z = 1;
        app.viewers_ui.insert(label.clone(), viewer);

        let state = app.apply_lut_dialog_state(
            label.clone(),
            "image.adjust.brightness".to_string(),
            json!({"min": 10.0, "max": 20.0, "apply": true}),
        );

        assert!(state.open);
        assert_eq!(state.stack_slices, 2);
        assert_eq!(state.params.get("slice"), None);
        assert_eq!(state.slice_params.get("slice"), Some(&json!(1)));
    }

    #[test]
    fn adjust_apply_lut_rgb_stack_prompts_like_imagej() {
        let label = "viewer-1".to_string();
        let data = Array::from_shape_vec(
            IxDyn(&[1, 1, 2, 3]),
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )
        .expect("shape");
        let metadata = Metadata {
            dims: vec![
                Dim::new(AxisKind::Y, 1),
                Dim::new(AxisKind::X, 1),
                Dim::new(AxisKind::Z, 2),
                Dim::new(AxisKind::Channel, 3),
            ],
            pixel_type: PixelType::U8,
            ..Metadata::default()
        };
        let dataset = Arc::new(DatasetF32::new(data, metadata).expect("dataset"));
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/apply-lut-rgb-stack.tif"),
                ViewerImageSource::Dataset(dataset),
            ),
        );
        let mut viewer = ViewerUiState::new(&label, "apply-lut-rgb-stack".to_string());
        viewer.z = 1;
        viewer.channel = 2;
        app.viewers_ui.insert(label.clone(), viewer);

        let state = app.apply_lut_dialog_state(
            label.clone(),
            "image.adjust.color_balance".to_string(),
            json!({"min": 10.0, "max": 60.0, "channel": "Blue", "apply": true}),
        );

        assert!(state.open);
        assert_eq!(state.stack_slices, 2);
        assert_eq!(state.params.get("slice"), None);
        assert_eq!(state.slice_params.get("slice"), Some(&json!(1)));
        assert_eq!(state.slice_params.get("channel"), Some(&json!("Blue")));
    }

    #[test]
    fn adjust_set_opens_imagej_set_display_range_dialog_state() {
        let label = "viewer-1".to_string();
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/set-range.tif"),
                ViewerImageSource::Dataset(dataset_2x2_with_pixel_type(
                    [0.0, 64.0, 128.0, 255.0],
                    PixelType::U16,
                )),
            ),
        );

        app.open_adjust_dialog(&label, super::AdjustDialogKind::BrightnessContrast);
        app.adjust_dialog.min = 10.0;
        app.adjust_dialog.max = 200.0;
        app.open_set_display_range_dialog_from_adjust("image.adjust.brightness", "min", "max");

        assert!(app.set_display_range_dialog.open);
        assert_eq!(app.set_display_range_dialog.window_label, label);
        assert_eq!(
            app.set_display_range_dialog.command_id,
            "image.adjust.brightness"
        );
        assert_eq!(app.set_display_range_dialog.minimum, 10.0);
        assert_eq!(app.set_display_range_dialog.maximum, 200.0);
        assert_eq!(
            app.set_display_range_dialog.unsigned_16bit_range,
            "Automatic"
        );
    }

    #[test]
    fn window_level_set_opens_imagej_set_wl_dialog_state() {
        let label = "viewer-1".to_string();
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/set-window-level.tif"),
                ViewerImageSource::Dataset(dataset_2x2_with_pixel_type(
                    [0.0, 64.0, 128.0, 255.0],
                    PixelType::U8,
                )),
            ),
        );

        app.open_adjust_dialog(&label, super::AdjustDialogKind::WindowLevel);
        app.adjust_dialog.min = 50.0;
        app.adjust_dialog.max = 150.0;
        app.open_set_window_level_dialog_from_adjust();

        assert!(app.set_window_level_dialog.open);
        assert_eq!(app.set_window_level_dialog.window_label, label);
        assert_eq!(app.set_window_level_dialog.level, 100.0);
        assert_eq!(app.set_window_level_dialog.window, 100.0);
        assert!(!app.set_window_level_dialog.propagate);
    }

    #[test]
    fn adjust_set_display_range_propagates_to_other_open_images() {
        let first = "viewer-1".to_string();
        let second = "viewer-2".to_string();
        let mut app = ImageUiApp::new_for_test();
        for label in [&first, &second] {
            app.state.label_to_session.insert(
                label.clone(),
                ViewerSession::new(
                    PathBuf::from(format!("/tmp/{label}.tif")),
                    ViewerImageSource::Dataset(dataset_2x2([0.0, 1.0, 2.0, 3.0])),
                ),
            );
        }

        let result = app.dispatch_command(
            &first,
            "image.adjust.brightness",
            Some(json!({"min": 0.5, "max": 2.5, "propagate": true})),
        );

        assert!(matches!(
            result.status,
            crate::ui::command_registry::CommandExecuteStatus::Ok
        ));
        assert_eq!(
            app.state.label_to_session[&first].display_range,
            Some((0.5, 2.5))
        );
        assert_eq!(
            app.state.label_to_session[&second].display_range,
            Some((0.5, 2.5))
        );
    }

    #[test]
    fn adjust_set_display_range_can_propagate_to_other_channels() {
        let label = "viewer-1".to_string();
        let data = Array::from_shape_vec(IxDyn(&[1, 1, 3]), vec![10.0, 20.0, 30.0]).expect("shape");
        let metadata = Metadata {
            dims: vec![
                Dim::new(AxisKind::Y, 1),
                Dim::new(AxisKind::X, 1),
                Dim::new(AxisKind::Channel, 3),
            ],
            pixel_type: PixelType::U8,
            ..Metadata::default()
        };
        let dataset = Arc::new(DatasetF32::new(data, metadata).expect("dataset"));
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/channel-set-range.tif"),
                ViewerImageSource::Dataset(dataset),
            ),
        );

        let result = app.dispatch_command(
            &label,
            "image.adjust.color_balance",
            Some(json!({"min": 5.0, "max": 25.0, "channel": "Red", "all_channels": true})),
        );

        assert!(matches!(
            result.status,
            crate::ui::command_registry::CommandExecuteStatus::Ok
        ));
        let session = &app.state.label_to_session[&label];
        assert_eq!(session.channel_display_ranges.get(&0), Some(&(5.0, 25.0)));
        assert_eq!(session.channel_display_ranges.get(&1), Some(&(5.0, 25.0)));
        assert_eq!(session.channel_display_ranges.get(&2), Some(&(5.0, 25.0)));
    }

    #[test]
    fn color_balance_uses_lut_color_for_grayscale_like_imagej() {
        let label = "viewer-1".to_string();
        let mut app = ImageUiApp::new_for_test();
        let session = ViewerSession::new(
            PathBuf::from("/tmp/lut-color.tif"),
            ViewerImageSource::Dataset(dataset_2x2_with_pixel_type(
                [0.0, 1.0, 2.0, 3.0],
                PixelType::U8,
            )),
        );
        assert!(color_balance_uses_lut_color(&session));
        app.state.label_to_session.insert(label.clone(), session);

        let result = app.dispatch_command(&label, "image.adjust.color_balance", None);

        assert!(matches!(
            result.status,
            crate::ui::command_registry::CommandExecuteStatus::Ok
        ));
        assert!(app.adjust_dialog.color_balance_lut_color);
        assert_eq!(app.adjust_dialog.color_balance_channel, "LUT level");
        assert_eq!(
            app.adjust_dialog.color_balance_channel_labels,
            vec!["LUT level".to_string()]
        );
        assert_eq!(
            color_balance_channel_labels_for_session(&app.state.label_to_session[&label]),
            vec!["LUT level".to_string()]
        );
        assert_eq!(adjust_dialog_window_title(&app.adjust_dialog), "LUT Color");
    }

    #[test]
    fn color_balance_uses_channel_labels_for_multichannel_images_like_imagej() {
        let label = "viewer-1".to_string();
        let data =
            Array::from_shape_vec(IxDyn(&[1, 1, 4]), vec![1.0, 2.0, 3.0, 4.0]).expect("shape");
        let dataset = Arc::new(
            DatasetF32::new(
                data,
                Metadata {
                    dims: vec![
                        Dim::new(AxisKind::Y, 1),
                        Dim::new(AxisKind::X, 1),
                        Dim::new(AxisKind::Channel, 4),
                    ],
                    pixel_type: PixelType::F32,
                    ..Metadata::default()
                },
            )
            .expect("dataset"),
        );
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/composite-color-balance.tif"),
                ViewerImageSource::Dataset(dataset),
            ),
        );

        let result = app.dispatch_command(&label, "image.adjust.color_balance", None);

        assert!(matches!(
            result.status,
            crate::ui::command_registry::CommandExecuteStatus::Ok
        ));
        assert!(!app.adjust_dialog.color_balance_lut_color);
        assert_eq!(adjust_dialog_window_title(&app.adjust_dialog), "Color");
        assert_eq!(app.adjust_dialog.color_balance_channel, "Channel 1");
        assert_eq!(
            app.adjust_dialog.color_balance_channel_labels,
            vec![
                "Channel 1".to_string(),
                "Channel 2".to_string(),
                "Channel 3".to_string(),
                "Channel 4".to_string(),
                "All".to_string(),
            ]
        );
        assert_eq!(color_balance_channel_index("Channel 4"), Some(3));
        assert_eq!(color_balance_channel_index("All"), None);
    }

    #[test]
    fn adjust_size_dialog_initializes_imagej_scale_fields() {
        let label = "viewer-1".to_string();
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/scale-dialog.tif"),
                ViewerImageSource::Dataset(dataset_2x2([0.0, 1.0, 2.0, 3.0])),
            ),
        );

        let result = app.dispatch_command(&label, "image.adjust.size", None);

        assert!(matches!(
            result.status,
            crate::ui::command_registry::CommandExecuteStatus::Ok
        ));
        assert!(app.resize_dialog.open);
        assert_eq!(app.resize_dialog.width, 2);
        assert_eq!(app.resize_dialog.height, 2);
        assert_eq!(app.resize_dialog.x_scale, 1.0);
        assert_eq!(app.resize_dialog.y_scale, 1.0);
        assert_eq!(app.resize_dialog.z_scale, 1.0);
        assert!(!app.resize_dialog.fill_with_background_available);
        assert!(!app.resize_dialog.fill_with_background);
        assert!(!app.resize_dialog.process_stack_available);
        assert!(app.resize_dialog.process_stack);
        assert!(app.resize_dialog.create_new_window);
        assert_eq!(app.resize_dialog.title, "scale-dialog");
        assert_eq!(
            resize_op_mode_from_params(&json!({"create_new_window": true})),
            OpRunMode::NewWindow
        );
    }

    #[test]
    fn adjust_size_dialog_exposes_process_stack_for_stacks_like_imagej() {
        let label = "viewer-1".to_string();
        let data = Array::from_shape_vec(IxDyn(&[1, 1, 2]), vec![10.0, 20.0]).expect("shape");
        let metadata = Metadata {
            dims: vec![
                Dim::new(AxisKind::Y, 1),
                Dim::new(AxisKind::X, 1),
                Dim::new(AxisKind::Z, 2),
            ],
            pixel_type: PixelType::U8,
            ..Metadata::default()
        };
        let dataset = Arc::new(DatasetF32::new(data, metadata).expect("dataset"));
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/scale-stack-dialog.tif"),
                ViewerImageSource::Dataset(dataset),
            ),
        );

        let result = app.dispatch_command(&label, "image.adjust.size", None);

        assert!(matches!(
            result.status,
            crate::ui::command_registry::CommandExecuteStatus::Ok
        ));
        assert_eq!(app.resize_dialog.depth, 2);
        assert_eq!(app.resize_dialog.original_depth, 2);
        assert!(app.resize_dialog.process_stack_available);
        assert!(app.resize_dialog.process_stack);
    }

    #[test]
    fn adjust_size_process_current_slice_new_window_like_imagej() {
        let label = "viewer-1".to_string();
        let data = Array::from_shape_vec(
            IxDyn(&[1, 1, 2, 3]),
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )
        .expect("shape");
        let metadata = Metadata {
            dims: vec![
                Dim::new(AxisKind::Y, 1),
                Dim::new(AxisKind::X, 1),
                Dim::new(AxisKind::Z, 2),
                Dim::new(AxisKind::Channel, 3),
            ],
            pixel_type: PixelType::U8,
            ..Metadata::default()
        };
        let dataset = Arc::new(DatasetF32::new(data, metadata).expect("dataset"));
        let mut app = ImageUiApp::new_for_test();
        app.state.next_window_id = 1;
        app.state.label_to_session.insert(
            label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/scale-current-slice.tif"),
                ViewerImageSource::Dataset(dataset),
            ),
        );
        let mut viewer = ViewerUiState::new(&label, "scale-current-slice".to_string());
        viewer.z = 1;
        app.viewers_ui.insert(label.clone(), viewer);

        let result = app.dispatch_command(
            &label,
            "image.adjust.size",
            Some(json!({
                "width": 2,
                "height": 2,
                "depth": 2,
                "interpolation": "None",
                "process_stack": false,
                "create_new_window": true,
                "title": "slice-copy"
            })),
        );

        assert!(matches!(
            result.status,
            crate::ui::command_registry::CommandExecuteStatus::Ok
        ));
        let new_label = app
            .state
            .label_to_session
            .keys()
            .find(|candidate| *candidate != &label)
            .cloned()
            .expect("new current-slice viewer");
        let new_session = &app.state.label_to_session[&new_label];
        assert_eq!(new_session.committed_summary.shape, vec![2, 2, 3]);
        assert_eq!(new_session.committed_summary.z_slices, 1);
        assert_eq!(new_session.committed_summary.channels, 3);
        let output = new_session.committed_dataset().expect("dataset");
        assert_eq!(
            output.data.iter().copied().collect::<Vec<_>>(),
            vec![
                40.0, 50.0, 60.0, 40.0, 50.0, 60.0, 40.0, 50.0, 60.0, 40.0, 50.0, 60.0,
            ]
        );
    }

    #[test]
    fn adjust_size_process_current_slice_in_place_like_imagej() {
        let label = "viewer-1".to_string();
        let data =
            Array::from_shape_vec(IxDyn(&[1, 2, 2]), vec![10.0, 30.0, 20.0, 40.0]).expect("shape");
        let metadata = Metadata {
            dims: vec![
                Dim::new(AxisKind::Y, 1),
                Dim::new(AxisKind::X, 2),
                Dim::new(AxisKind::Z, 2),
            ],
            pixel_type: PixelType::U8,
            ..Metadata::default()
        };
        let dataset = Arc::new(DatasetF32::new(data, metadata).expect("dataset"));
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/scale-current-slice-in-place.tif"),
                ViewerImageSource::Dataset(dataset),
            ),
        );
        let mut viewer = ViewerUiState::new(&label, "scale-current-slice-in-place".to_string());
        viewer.z = 1;
        app.viewers_ui.insert(label.clone(), viewer);

        let result = app.dispatch_command(
            &label,
            "image.adjust.size",
            Some(json!({
                "width": 1,
                "height": 1,
                "interpolation": "None",
                "fill": 0.0,
                "process_stack": false,
                "create_new_window": false,
            })),
        );

        assert!(matches!(
            result.status,
            crate::ui::command_registry::CommandExecuteStatus::Ok
        ));
        assert_eq!(app.state.label_to_session.len(), 1);
        let output = app.state.label_to_session[&label]
            .committed_dataset()
            .expect("dataset");
        assert_eq!(output.shape(), &[1, 2, 2]);
        assert_eq!(
            output.data.iter().copied().collect::<Vec<_>>(),
            vec![10.0, 30.0, 20.0, 0.0]
        );
    }

    #[test]
    fn adjust_size_dialog_exposes_background_fill_for_u8_or_rgb_like_imagej() {
        let label = "viewer-1".to_string();
        let u8_session = ViewerSession::new(
            PathBuf::from("/tmp/u8-scale-dialog.tif"),
            ViewerImageSource::Dataset(dataset_2x2_with_pixel_type(
                [0.0, 1.0, 2.0, 3.0],
                PixelType::U8,
            )),
        );
        assert!(scale_fill_with_background_available(&u8_session));

        let rgb_data =
            Array::from_shape_vec(IxDyn(&[1, 1, 3]), vec![0.0, 1.0, 2.0]).expect("shape");
        let rgb_dataset = Arc::new(
            DatasetF32::new(
                rgb_data,
                Metadata {
                    dims: vec![
                        Dim::new(AxisKind::Y, 1),
                        Dim::new(AxisKind::X, 1),
                        Dim::new(AxisKind::Channel, 3),
                    ],
                    pixel_type: PixelType::F32,
                    ..Metadata::default()
                },
            )
            .expect("dataset"),
        );
        let rgb_session = ViewerSession::new(
            PathBuf::from("/tmp/rgb-scale-dialog.tif"),
            ViewerImageSource::Dataset(rgb_dataset),
        );
        assert!(scale_fill_with_background_available(&rgb_session));

        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(label.clone(), u8_session);

        let result = app.dispatch_command(&label, "image.adjust.size", None);

        assert!(matches!(
            result.status,
            crate::ui::command_registry::CommandExecuteStatus::Ok
        ));
        assert!(app.resize_dialog.fill_with_background_available);
        assert!(!app.resize_dialog.fill_with_background);
    }

    #[test]
    fn threshold_apply_can_route_float_images_to_nan_background() {
        let label = "viewer-1".to_string();
        let dataset = dataset_2x2_with_pixel_type([0.0, 0.25, 0.75, 1.0], PixelType::F32);
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/threshold-nan-background.tif"),
                ViewerImageSource::Dataset(dataset),
            ),
        );
        app.viewers_ui.insert(
            label.clone(),
            ViewerUiState::new(&label, "threshold-nan-background".to_string()),
        );

        let result = app.dispatch_command(
            &label,
            "image.adjust.threshold",
            Some(json!({
                "apply": true,
                "min": 0.25,
                "max": 0.75,
                "background_to_nan": true
            })),
        );

        assert!(matches!(
            result.status,
            crate::ui::command_registry::CommandExecuteStatus::Ok
        ));
        assert_eq!(
            result
                .payload
                .as_ref()
                .and_then(|payload| payload.get("op")),
            Some(&json!("intensity.nan_background"))
        );
    }

    #[test]
    fn threshold_apply_prompts_for_float_images_only() {
        let float_label = "viewer-1".to_string();
        let byte_label = "viewer-2".to_string();
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            float_label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/threshold-float-prompt.tif"),
                ViewerImageSource::Dataset(dataset_2x2_with_pixel_type(
                    [0.0, 0.25, 0.75, 1.0],
                    PixelType::F32,
                )),
            ),
        );
        app.state.label_to_session.insert(
            byte_label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/threshold-byte-prompt.tif"),
                ViewerImageSource::Dataset(dataset_2x2_with_pixel_type(
                    [0.0, 64.0, 128.0, 255.0],
                    PixelType::U8,
                )),
            ),
        );

        assert!(app.threshold_apply_needs_float_prompt(&float_label));
        assert!(!app.threshold_apply_needs_float_prompt(&byte_label));
    }

    #[test]
    fn threshold_set_opens_imagej_threshold_levels_dialog_state() {
        let label = "viewer-1".to_string();
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/threshold-set.tif"),
                ViewerImageSource::Dataset(dataset_2x2_with_pixel_type(
                    [0.0, 64.0, 128.0, 255.0],
                    PixelType::U8,
                )),
            ),
        );

        app.open_adjust_dialog(&label, super::AdjustDialogKind::Threshold);
        app.adjust_dialog.min = 64.0;
        app.adjust_dialog.max = 192.0;
        app.adjust_dialog.threshold_mode = "Over/Under".to_string();
        app.adjust_dialog.dark_background = false;
        app.open_threshold_set_dialog_from_adjust();

        assert!(app.threshold_set_dialog.open);
        assert_eq!(app.threshold_set_dialog.window_label, label);
        assert_eq!(app.threshold_set_dialog.min, 64.0);
        assert_eq!(app.threshold_set_dialog.max, 192.0);
        assert_eq!(app.threshold_set_dialog.mode, "Over/Under");
        assert!(!app.threshold_set_dialog.dark_background);
    }

    #[test]
    fn threshold_adjuster_rejects_rgb_images_like_imagej() {
        let label = "viewer-1".to_string();
        let data = Array::from_shape_vec(IxDyn(&[1, 1, 3]), vec![10.0, 20.0, 30.0]).expect("shape");
        let metadata = Metadata {
            dims: vec![
                Dim::new(AxisKind::Y, 1),
                Dim::new(AxisKind::X, 1),
                Dim::new(AxisKind::Channel, 3),
            ],
            pixel_type: PixelType::U8,
            ..Metadata::default()
        };
        let dataset = Arc::new(DatasetF32::new(data, metadata).expect("dataset"));
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/rgb-threshold.tif"),
                ViewerImageSource::Dataset(dataset),
            ),
        );

        let result = app.dispatch_command(&label, "image.adjust.threshold", None);

        assert!(matches!(
            result.status,
            crate::ui::command_registry::CommandExecuteStatus::Blocked
        ));
        assert!(result.message.contains("grayscale"));
        assert!(result.message.contains("Color Threshold"));
    }

    #[test]
    fn threshold_overlay_renders_without_committing_pixels() {
        let original = dataset_2x2_with_pixel_type([0.0, 5.0, 10.0, 20.0], PixelType::U8);
        let mut session = ViewerSession::new(
            PathBuf::from("/tmp/threshold-overlay.tif"),
            ViewerImageSource::Dataset(original.clone()),
        );
        session.set_threshold_overlay(Some(ThresholdOverlay {
            low: 5.0,
            high: 10.0,
            mode: ThresholdOverlayMode::Red,
        }));

        let frame = build_frame(
            &session.committed_source(),
            &ViewerFrameRequest::default(),
            session.display_range,
        )
        .expect("frame");
        let image =
            to_color_image_with_threshold(&frame, LookupTable::Grays, session.threshold_overlay);

        assert_eq!(image.pixels[0], egui::Color32::BLACK);
        assert_eq!(image.pixels[1], egui::Color32::RED);
        assert_eq!(image.pixels[2], egui::Color32::RED);
        assert_eq!(image.pixels[3], egui::Color32::from_gray(20));
        assert!(Arc::ptr_eq(
            &session.committed_dataset().expect("committed"),
            &original
        ));
        assert!(!session.can_undo());
    }

    #[test]
    fn threshold_reset_honors_no_reset_range_checkbox() {
        let label = "viewer-1".to_string();
        let dataset = dataset_2x2_with_pixel_type([0.0, 5.0, 10.0, 20.0], PixelType::U8);
        let mut app = ImageUiApp::new_for_test();
        let mut session = ViewerSession::new(
            PathBuf::from("/tmp/threshold-reset.tif"),
            ViewerImageSource::Dataset(dataset),
        );
        session.set_display_range(Some((0.0, 10.0)));
        session.set_threshold_overlay(Some(ThresholdOverlay {
            low: 5.0,
            high: 10.0,
            mode: ThresholdOverlayMode::Red,
        }));
        app.state.label_to_session.insert(label.clone(), session);
        app.viewers_ui.insert(
            label.clone(),
            ViewerUiState::new(&label, "threshold-reset".to_string()),
        );

        app.set_threshold_overlay(&label, &json!({"reset": true, "no_reset": true}))
            .expect("reset threshold only");
        let session = app.state.label_to_session.get_mut(&label).expect("session");
        assert!(session.threshold_overlay.is_none());
        assert_eq!(session.display_range, Some((0.0, 10.0)));
        session.set_threshold_overlay(Some(ThresholdOverlay {
            low: 5.0,
            high: 10.0,
            mode: ThresholdOverlayMode::Red,
        }));

        app.set_threshold_overlay(&label, &json!({"reset": true, "no_reset": false}))
            .expect("reset threshold and display");
        let session = app.state.label_to_session.get(&label).expect("session");
        assert!(session.threshold_overlay.is_none());
        assert_eq!(session.display_range, None);
    }

    #[test]
    fn threshold_auto_uses_current_slice_unless_stack_histogram_enabled() {
        let data = Array::from_shape_vec(
            (2, 2, 2),
            vec![0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 10.0, 100.0],
        )
        .expect("shape")
        .into_dyn();
        let dataset = Arc::new(DatasetF32::from_data_with_default_metadata(
            data,
            PixelType::F32,
        ));
        let label = "viewer-1".to_string();
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/threshold-stack.tif"),
                ViewerImageSource::Dataset(dataset),
            ),
        );
        app.viewers_ui.insert(
            label.clone(),
            ViewerUiState::new(&label, "stack".to_string()),
        );

        app.set_threshold_overlay(&label, &json!({"method": "mean", "background": "dark"}))
            .expect("slice threshold");
        let slice_low = app
            .state
            .label_to_session
            .get(&label)
            .and_then(|session| session.threshold_overlay.as_ref())
            .map(|threshold| threshold.low)
            .expect("slice overlay");

        app.set_threshold_overlay(
            &label,
            &json!({"method": "mean", "background": "dark", "stack": true}),
        )
        .expect("stack threshold");
        let stack_low = app
            .state
            .label_to_session
            .get(&label)
            .and_then(|session| session.threshold_overlay.as_ref())
            .map(|threshold| threshold.low)
            .expect("stack overlay");

        assert!(slice_low < 10.0, "slice threshold was {slice_low}");
        assert!(stack_low > 40.0, "stack threshold was {stack_low}");
    }

    #[test]
    fn color_threshold_selection_bbox_uses_foreground_mask_pixels() {
        let slice = SliceImage {
            width: 4,
            height: 3,
            pixel_type: PixelType::U8,
            values: vec![
                0.0, 0.0, 0.0, 0.0, //
                0.0, 255.0, 255.0, 0.0, //
                0.0, 0.0, 255.0, 0.0,
            ],
        };

        assert_eq!(mask_foreground_bbox(&slice), Some((1, 1, 2, 2)));
    }

    #[test]
    fn color_threshold_selection_traces_mask_outline_instead_of_bbox() {
        let slice = SliceImage {
            width: 4,
            height: 4,
            pixel_type: PixelType::U8,
            values: vec![
                0.0, 0.0, 0.0, 0.0, //
                0.0, 255.0, 0.0, 0.0, //
                0.0, 255.0, 255.0, 0.0, //
                0.0, 0.0, 0.0, 0.0,
            ],
        };

        let roi = mask_foreground_roi(&slice).expect("roi");
        let RoiKind::Polygon { points, closed, .. } = roi else {
            panic!("expected traced polygon");
        };

        assert!(closed);
        assert!(points.len() > 4, "outline points: {points:?}");
        let bounds = roi_kind_bbox(&RoiKind::Polygon {
            points: points.clone(),
            closed,
            spline_fit: false,
        })
        .expect("bounds");
        assert_eq!(bounds, (1, 1, 3, 3));
    }

    #[test]
    fn color_threshold_sample_ranges_use_selected_rgb_pixels() {
        let data = Array::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                10.0, 20.0, 30.0, //
                40.0, 50.0, 60.0, //
                70.0, 80.0, 90.0, //
                100.0, 110.0, 120.0,
            ],
        )
        .expect("shape");
        let dataset = DatasetF32::new(
            data,
            Metadata {
                dims: vec![
                    Dim::new(AxisKind::Y, 2),
                    Dim::new(AxisKind::X, 2),
                    Dim::new(AxisKind::Channel, 3),
                ],
                pixel_type: PixelType::U8,
                ..Metadata::default()
            },
        )
        .expect("dataset");

        let sample = sample_color_threshold_ranges_with_passes(&dataset, (1, 0, 1, 1), 0, 0, "RGB")
            .expect("sample ranges");

        assert_eq!(sample.ranges, [(40.0, 100.0), (50.0, 110.0), (60.0, 120.0)]);
        assert_eq!(sample.passes, [true, true, true]);
    }

    #[test]
    fn color_threshold_sample_hsb_hue_wrap_uses_stop_band_like_imagej() {
        let data = Array::from_shape_vec(IxDyn(&[1, 2, 3]), vec![255.0, 0.0, 0.0, 255.0, 0.0, 1.0])
            .expect("shape");
        let dataset = DatasetF32::new(
            data,
            Metadata {
                dims: vec![
                    Dim::new(AxisKind::Y, 1),
                    Dim::new(AxisKind::X, 2),
                    Dim::new(AxisKind::Channel, 3),
                ],
                pixel_type: PixelType::U8,
                ..Metadata::default()
            },
        )
        .expect("dataset");

        let sample = sample_color_threshold_ranges_with_passes(&dataset, (0, 0, 1, 0), 0, 0, "HSB")
            .expect("sample ranges");

        assert_eq!(sample.ranges[0], (1.0, 254.0));
        assert_eq!(sample.passes, [false, true, true]);
    }

    #[test]
    fn color_threshold_auto_ranges_follow_imagej_rgb_band_rule() {
        let data = Array::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                0.0, 0.0, 0.0, //
                10.0, 20.0, 30.0, //
                240.0, 220.0, 200.0, //
                250.0, 240.0, 230.0,
            ],
        )
        .expect("shape");
        let dataset = DatasetF32::new(
            data,
            Metadata {
                dims: vec![
                    Dim::new(AxisKind::Y, 2),
                    Dim::new(AxisKind::X, 2),
                    Dim::new(AxisKind::Channel, 3),
                ],
                pixel_type: PixelType::U8,
                ..Metadata::default()
            },
        )
        .expect("dataset");
        let app = AppContext::new();

        let ranges = color_threshold_auto_ranges(&dataset, 0, 0, "RGB", "mean", "dark", &app)
            .expect("auto ranges");

        assert!(ranges[0].0 > 100.0 && ranges[0].1 == 250.0);
        assert!(ranges[1].0 > 100.0 && ranges[1].1 == 240.0);
        assert!(ranges[2].0 > 100.0 && ranges[2].1 == 230.0);
    }

    #[test]
    fn color_threshold_auto_ranges_follow_imagej_hsb_brightness_rule() {
        let data = Array::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                0.0, 0.0, 0.0, //
                32.0, 32.0, 32.0, //
                224.0, 224.0, 224.0, //
                255.0, 255.0, 255.0,
            ],
        )
        .expect("shape");
        let dataset = DatasetF32::new(
            data,
            Metadata {
                dims: vec![
                    Dim::new(AxisKind::Y, 2),
                    Dim::new(AxisKind::X, 2),
                    Dim::new(AxisKind::Channel, 3),
                ],
                pixel_type: PixelType::U8,
                ..Metadata::default()
            },
        )
        .expect("dataset");
        let app = AppContext::new();

        let ranges = color_threshold_auto_ranges(&dataset, 0, 0, "HSB", "mean", "dark", &app)
            .expect("auto ranges");

        assert_eq!(ranges[0], (0.0, 255.0));
        assert_eq!(ranges[1], (0.0, 255.0));
        assert!(ranges[2].0 > 100.0 && ranges[2].1 == 255.0);
    }

    #[test]
    fn color_threshold_space_change_resets_bands_like_imagej() {
        let mut dialog = AdjustDialogState {
            hue_min: 10.0,
            hue_max: 20.0,
            saturation_min: 30.0,
            saturation_max: 40.0,
            brightness_min: 50.0,
            brightness_max: 60.0,
            hue_pass: false,
            saturation_pass: false,
            brightness_pass: false,
            ..AdjustDialogState::default()
        };

        reset_color_threshold_bands_for_space(&mut dialog);

        assert_eq!(dialog.hue_min, 0.0);
        assert_eq!(dialog.hue_max, 255.0);
        assert_eq!(dialog.saturation_min, 0.0);
        assert_eq!(dialog.saturation_max, 255.0);
        assert_eq!(dialog.brightness_min, 0.0);
        assert_eq!(dialog.brightness_max, 255.0);
        assert!(dialog.hue_pass);
        assert!(dialog.saturation_pass);
        assert!(dialog.brightness_pass);
    }

    #[test]
    fn color_threshold_choice_labels_follow_imagej_order() {
        assert_eq!(
            color_threshold_choice_labels(),
            ["Thresholding method:", "Threshold color:", "Color space:"]
        );
    }

    #[test]
    fn color_threshold_macro_text_records_current_ranges() {
        let text = color_threshold_macro_text(&json!({
            "color_space": "RGB",
            "method": "Otsu",
            "mode": "B&W",
            "hue": {"min": 10, "max": 200, "pass": true},
            "saturation": {"min": 20, "max": 180, "pass": false},
            "brightness": {"min": 30, "max": 160, "pass": true}
        }));

        assert!(text.contains("color_space=RGB"));
        assert!(text.contains("method=Otsu"));
        assert!(text.contains("threshold_color=B&W"));
        assert!(text.contains("min[1]=20;"));
        assert!(text.contains("max[1]=180;"));
        assert!(text.contains("filter[1]=\"stop\";"));
        assert!(text.contains("run(\"RGB Stack\");"));
        assert!(text.contains("imageCalculator(\"AND create\", \"0\",\"1\");"));
        assert!(text.contains("// Colour Thresholding-------------"));

        let lab_text = color_threshold_macro_text(&json!({
            "color_space": "CIE Lab"
        }));
        assert!(lab_text.contains("call(\"ij.plugin.frame.ColorThresholder.RGBtoLab\");"));
    }

    #[test]
    fn color_threshold_help_action_reports_imagej_controls() {
        let label = "viewer-1".to_string();
        let dataset = dataset_2x2_with_pixel_type([0.0, 5.0, 10.0, 20.0], PixelType::U8);
        let mut app = ImageUiApp::new_for_test();
        app.state.label_to_session.insert(
            label.clone(),
            ViewerSession::new(
                PathBuf::from("/tmp/color-threshold-help.tif"),
                ViewerImageSource::Dataset(dataset),
            ),
        );
        app.viewers_ui.insert(
            label.clone(),
            ViewerUiState::new(&label, "color-threshold-help".to_string()),
        );

        let result = app.dispatch_command(
            &label,
            "image.adjust.color_threshold",
            Some(json!({"action": "help"})),
        );

        assert!(matches!(
            result.status,
            crate::ui::command_registry::CommandExecuteStatus::Ok
        ));
        let viewer = app.viewers_ui.get(&label).expect("viewer");
        let help = viewer.tool_message.as_deref().expect("help message");
        assert!(help.contains("Pass"));
        assert!(help.contains("Stack"));
        assert!(help.contains("Sample"));
    }

    #[test]
    fn threshold_method_labels_include_imagej_ij_isodata() {
        assert!(threshold_method_labels().contains(&"IJ_IsoData"));
        assert_eq!(threshold_method_param("IJ_IsoData"), "ij_isodata");
    }

    #[test]
    fn threshold_sixteen_bit_histogram_forces_no_reset_range() {
        let mut dialog = AdjustDialogState {
            threshold_no_reset: false,
            ..AdjustDialogState::default()
        };

        set_threshold_sixteen_bit_histogram(&mut dialog, true);

        assert!(dialog.threshold_sixteen_bit_histogram);
        assert!(dialog.threshold_no_reset);

        set_threshold_sixteen_bit_histogram(&mut dialog, false);

        assert!(!dialog.threshold_sixteen_bit_histogram);
        assert!(dialog.threshold_no_reset);
    }

    #[test]
    fn adjust_histogram_counts_finite_values_into_bins() {
        let histogram = adjust_histogram(&[0.0, 0.2, 0.8, 1.0, f32::NAN], 2).expect("histogram");

        assert_eq!(histogram.counts, vec![2, 2]);
        assert_eq!(histogram.min, 0.0);
        assert_eq!(histogram.max, 1.0);
        assert_eq!(histogram.pixel_count, 4);
    }

    #[test]
    fn auto_contrast_range_matches_imagej_contrast_adjuster_threshold() {
        let histogram = AdjustHistogram {
            counts: vec![50, 1, 20, 1, 50],
            min: 0.0,
            max: 100.0,
            pixel_count: 122,
        };
        let mut auto_threshold = 0;

        let first = auto_contrast_range(Some(&histogram), &mut auto_threshold).expect("auto");
        let second = auto_contrast_range(Some(&histogram), &mut auto_threshold).expect("auto");

        assert_eq!(auto_threshold, 2500);
        assert_eq!(first, (20.0, 60.0));
        assert_eq!(second, (20.0, 60.0));
    }

    #[test]
    fn image_draw_rect_keeps_native_size_at_100_percent() {
        let canvas = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(400.0, 300.0));
        let transform = ViewerTransformState::new(100, 50);

        let image_rect = image_draw_rect(canvas, &transform, 100, 50);

        assert_eq!(image_rect.size(), egui::vec2(100.0, 50.0));
        assert_eq!(image_rect.center(), canvas.center());
    }

    #[test]
    fn oversized_images_scale_down_on_initial_open() {
        let canvas = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(400.0, 300.0));
        let mut viewer = ViewerUiState::new("viewer-1", "Test".to_string());

        initialize_view_to_open_state(&mut viewer, canvas, 1200, 300);

        assert!(viewer.transform.magnification < 1.0);
        assert_eq!(viewer.initial_magnification, viewer.transform.magnification);
        let image_rect = image_draw_rect(canvas, &viewer.transform, 1200, 300);
        assert!(image_rect.width() <= canvas.width() + f32::EPSILON);
        assert!(image_rect.height() <= canvas.height() + f32::EPSILON);
    }

    #[test]
    fn original_scale_restores_initial_fit_while_view_100_forces_native_size() {
        let canvas = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(400.0, 300.0));
        let frame = ViewerFrameBuffer {
            width: 1200,
            height: 300,
            values: vec![0.0; 1200 * 300],
            pixels_u8: vec![0; 1200 * 300],
            min: 0.0,
            max: 0.0,
        };
        let mut viewer = ViewerUiState::new("viewer-1", "Test".to_string());
        initialize_view_to_open_state(&mut viewer, canvas, frame.width, frame.height);

        let initial = viewer.initial_magnification;
        assert!(initial < 1.0);
        let image_rect = image_draw_rect(canvas, &viewer.transform, frame.width, frame.height);

        apply_zoom_command(
            &mut viewer,
            ZoomCommand::View100,
            canvas,
            image_rect,
            &frame,
        );
        assert!((viewer.transform.magnification - 1.0).abs() < f32::EPSILON);

        let image_rect = image_draw_rect(canvas, &viewer.transform, frame.width, frame.height);
        apply_zoom_command(
            &mut viewer,
            ZoomCommand::Original,
            canvas,
            image_rect,
            &frame,
        );
        assert!((viewer.transform.magnification - initial).abs() < f32::EPSILON);
    }

    #[test]
    fn set_zoom_applies_exact_magnification_around_image_center() {
        let canvas = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(400.0, 300.0));
        let frame = ViewerFrameBuffer {
            width: 1200,
            height: 300,
            values: vec![0.0; 1200 * 300],
            pixels_u8: vec![0; 1200 * 300],
            min: 0.0,
            max: 0.0,
        };
        let mut viewer = ViewerUiState::new("viewer-1", "Test".to_string());
        initialize_view_to_open_state(&mut viewer, canvas, frame.width, frame.height);
        let image_rect = image_draw_rect(canvas, &viewer.transform, frame.width, frame.height);

        apply_zoom_command(
            &mut viewer,
            ZoomCommand::Set {
                magnification: 2.0,
                x_center: Some(600.0),
                y_center: Some(150.0),
            },
            canvas,
            image_rect,
            &frame,
        );

        assert!((viewer.transform.magnification - 2.0).abs() < f32::EPSILON);
        assert!((viewer.transform.src_rect.x - 500.0).abs() < f32::EPSILON);
        assert!((viewer.transform.src_rect.y - 75.0).abs() < f32::EPSILON);
        assert!((viewer.transform.src_rect.width - 200.0).abs() < f32::EPSILON);
        assert!((viewer.transform.src_rect.height - 150.0).abs() < f32::EPSILON);
    }

    #[test]
    fn initial_viewport_size_scales_large_images_to_monitor() {
        let summary = ImageSummary {
            shape: vec![300, 1200],
            axes: vec!["Y".to_string(), "X".to_string()],
            channels: 1,
            z_slices: 1,
            times: 1,
            min: 0.0,
            max: 1.0,
            source: "/tmp/wide.png".to_string(),
        };

        let size = compute_initial_viewport_size(&summary, Some(egui::vec2(800.0, 600.0)));

        assert!(size.x <= 800.0);
        assert!(size.y <= 600.0);
        assert!(size.x > size.y);
    }

    #[test]
    fn dominant_scroll_component_prefers_larger_axis() {
        assert_eq!(dominant_scroll_component(egui::vec2(8.0, 2.0)), 8.0);
        assert_eq!(dominant_scroll_component(egui::vec2(1.0, -4.0)), -4.0);
    }

    #[test]
    fn effective_scroll_delta_prefers_populated_stream() {
        assert_eq!(
            effective_scroll_delta(egui::Vec2::ZERO, egui::vec2(1.0, 2.0)),
            egui::vec2(1.0, 2.0)
        );
        assert_eq!(
            effective_scroll_delta(egui::vec2(3.0, 4.0), egui::Vec2::ZERO),
            egui::vec2(3.0, 4.0)
        );
    }

    #[test]
    fn repaint_policy_idle_does_not_request_repaint() {
        let inputs = RepaintDecisionInputs::default();
        assert!(!should_request_repaint_now(inputs));
        assert!(!should_request_periodic_repaint(inputs));
    }
}
