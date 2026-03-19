use super::state::{
    DesktopState, MeasurementSettings, ResultsTableState, load_desktop_state, push_recent_file,
    save_desktop_state,
};
use super::{command_registry, interaction};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    mpsc::{self, Receiver, Sender},
};
use std::time::Duration;

use super::interaction::roi::RoiKind;
use super::interaction::tooling::{
    LineMode, OvalMode, PointMode, RectMode, ToolId, ToolOptionsState, ToolState,
};
use super::interaction::transform::{ViewerTransformState, zoom_level_down, zoom_level_up};
use crate::formats::{NativeRasterImage, supported_formats};
use crate::model::{AxisKind, Dataset, DatasetF32, Dim, Metadata, PixelType};
use crate::runtime::AppContext;
use eframe::egui;
use image::load_from_memory;
use ndarray::{ArrayD, IxDyn};
use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

const LAUNCHER_LABEL: &str = "main";
const VIEWER_PREFIX: &str = "viewer-";
const SOURCE_COMMITTED: &str = "committed";
const SOURCE_PREVIEW_PREFIX: &str = "preview:";
const VIEWER_MIN_WINDOW_SIZE: [f32; 2] = [220.0, 160.0];
const VIEWER_WINDOW_EXTRA_SIZE: [f32; 2] = [24.0, 120.0];
const LAUNCHER_MIN_WINDOW_SIZE: [f32; 2] = [600.0, 200.0];

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MenuManifestTopLevel {
    id: String,
    label: String,
    items: Vec<MenuManifestItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone)]
enum ViewerImageSource {
    Native(Arc<NativeRasterImage>),
    Dataset(Arc<DatasetF32>),
}

impl ViewerImageSource {
    fn to_dataset(&self) -> Result<Arc<DatasetF32>, String> {
        match self {
            Self::Native(image) => image
                .to_dataset()
                .map(Arc::new)
                .map_err(|error: crate::formats::IoError| error.to_string()),
            Self::Dataset(dataset) => Ok(dataset.clone()),
        }
    }
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

#[derive(Debug, Clone)]
struct ViewerSession {
    path: PathBuf,
    base_source: ViewerImageSource,
    committed_source: ViewerImageSource,
    committed_summary: ImageSummary,
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
            self.undo_stack.clear();
            self.redo_stack.clear();
            return false;
        }

        self.committed_source = self.base_source.clone();
        self.committed_summary = summarize_source(&self.committed_source, &self.path);
        self.active_preview = None;
        self.preview_cache.clear();
        self.frame_cache.clear();
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

#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq, Eq)]
struct ViewerFrameRequest {
    #[serde(default)]
    z: usize,
    #[serde(default)]
    t: usize,
    #[serde(default)]
    channel: usize,
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
}

impl OpRunMode {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Preview => "preview",
            Self::Apply => "apply",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct JobTicket {
    job_id: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct ImageSummary {
    shape: Vec<usize>,
    axes: Vec<String>,
    channels: usize,
    z_slices: usize,
    times: usize,
    min: f32,
    max: f32,
    source: String,
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
struct SliceImage {
    width: usize,
    height: usize,
    values: Vec<f32>,
}

#[derive(Debug, Clone)]
struct ViewerFrameBuffer {
    width: usize,
    height: usize,
    values: Vec<f32>,
    pixels_u8: Vec<u8>,
    min: f32,
    max: f32,
}

#[derive(Debug, Clone)]
enum WorkerEvent {
    OpFinished {
        window_label: String,
        job_id: u64,
        generation: u64,
        mode: OpRunMode,
        op: String,
        preview_key: Option<String>,
        result: Result<Arc<DatasetF32>, String>,
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
}

#[derive(Debug, Clone, Default)]
struct PlotProfileState {
    title: String,
    samples: Vec<f32>,
}

#[derive(Debug, Clone)]
struct NewImageDialogState {
    open: bool,
    width: usize,
    height: usize,
    slices: usize,
    channels: usize,
    pixel_type: PixelType,
    fill: f32,
}

impl Default for NewImageDialogState {
    fn default() -> Self {
        Self {
            open: false,
            width: 512,
            height: 512,
            slices: 1,
            channels: 1,
            pixel_type: PixelType::F32,
            fill: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
struct ResizeDialogState {
    open: bool,
    width: usize,
    height: usize,
    fill: f32,
}

impl Default for ResizeDialogState {
    fn default() -> Self {
        Self {
            open: false,
            width: 512,
            height: 512,
            fill: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
struct RawImportDialogState {
    open: bool,
    path: Option<PathBuf>,
    width: usize,
    height: usize,
    slices: usize,
    channels: usize,
    pixel_type: PixelType,
    little_endian: bool,
    byte_offset: usize,
}

impl Default for RawImportDialogState {
    fn default() -> Self {
        Self {
            open: false,
            path: None,
            width: 512,
            height: 512,
            slices: 1,
            channels: 1,
            pixel_type: PixelType::U8,
            little_endian: true,
            byte_offset: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct UrlImportDialogState {
    open: bool,
    url: String,
}

#[derive(Debug, Clone, Default)]
struct CommandFinderState {
    query: String,
}

#[derive(Debug, Clone, Default)]
struct RoiManagerState {
    rename_buffer: String,
}

struct ViewerUiState {
    viewport_id: egui::ViewportId,
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
    last_request: Option<ViewerFrameRequest>,
    last_generation: u64,
    initial_magnification: f32,
    initial_viewport_sized: bool,
    fit_requested: bool,
    pending_zoom: Option<ZoomCommand>,
    active_drag_started: Option<egui::Pos2>,
    active_polygon_points: Vec<egui::Pos2>,
    last_status_pointer: Option<egui::Pos2>,
}

impl ViewerUiState {
    fn new(label: &str, title: String) -> Self {
        Self {
            viewport_id: viewport_id_for_label(label),
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
            last_request: None,
            last_generation: 0,
            initial_magnification: 1.0,
            initial_viewport_sized: false,
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
    OpenPaths {
        paths: Vec<PathBuf>,
    },
    CloseViewer {
        label: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ZoomCommand {
    In,
    Out,
    Original,
    View100,
    ToSelection,
    ScaleToFit,
    Set,
    Maximize,
}

#[derive(Debug, Clone, Copy)]
enum LayoutMode {
    Tile,
    Cascade,
}

struct ImageUiApp {
    state: UiState,
    desktop_state: DesktopState,
    menus: Vec<MenuManifestTopLevel>,
    command_catalog: command_registry::CommandCatalog,
    launcher_ui: LauncherUiState,
    tool_state: ToolState,
    tool_options: ToolOptionsState,
    toolbar_icons: HashMap<ToolbarIcon, egui::TextureHandle>,
    viewers_ui: HashMap<String, ViewerUiState>,
    viewer_telemetry: HashMap<String, ViewerTelemetry>,
    viewport_positions: HashMap<String, egui::Pos2>,
    results_table: ResultsTableState,
    clipboard: ClipboardState,
    profile_plot: PlotProfileState,
    new_image_dialog: NewImageDialogState,
    resize_dialog: ResizeDialogState,
    canvas_dialog: ResizeDialogState,
    raw_import_dialog: RawImportDialogState,
    url_import_dialog: UrlImportDialogState,
    command_finder: CommandFinderState,
    roi_manager: RoiManagerState,
    active_viewer_label: Option<String>,
    worker_tx: Sender<WorkerEvent>,
    worker_rx: Receiver<WorkerEvent>,
    focus_launcher: bool,
    focus_viewer_label: Option<String>,
    should_quit: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct RepaintDecisionInputs {
    worker_state_changed: bool,
    has_pending_actions: bool,
    has_focus_or_close_command: bool,
    has_pointer_activity: bool,
    has_scroll_activity: bool,
    has_input_events: bool,
    has_active_jobs: bool,
}

fn should_request_repaint_now(inputs: RepaintDecisionInputs) -> bool {
    inputs.worker_state_changed
        || inputs.has_pending_actions
        || inputs.has_focus_or_close_command
        || inputs.has_pointer_activity
        || inputs.has_scroll_activity
        || inputs.has_input_events
}

fn should_request_periodic_repaint(inputs: RepaintDecisionInputs) -> bool {
    !should_request_repaint_now(inputs) && inputs.has_active_jobs
}

impl ImageUiApp {
    fn new(_cc: &eframe::CreationContext<'_>, startup_input: Option<PathBuf>) -> Self {
        let menus: Vec<MenuManifestTopLevel> =
            serde_json::from_str(include_str!("menu/imagej-menu-manifest.json"))
                .expect("failed to parse menu manifest");
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
            viewport_positions: HashMap::new(),
            results_table: ResultsTableState::default(),
            clipboard: ClipboardState::default(),
            profile_plot: PlotProfileState::default(),
            new_image_dialog: NewImageDialogState::default(),
            resize_dialog: ResizeDialogState::default(),
            canvas_dialog: ResizeDialogState::default(),
            raw_import_dialog: RawImportDialogState::default(),
            url_import_dialog: UrlImportDialogState::default(),
            command_finder: CommandFinderState::default(),
            roi_manager: RoiManagerState::default(),
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

    fn set_active_viewer(&mut self, label: Option<String>) {
        self.active_viewer_label = label;
        self.refresh_launcher_status();
    }

    fn update_viewer_telemetry(&mut self, label: &str, telemetry: ViewerTelemetry) {
        self.viewer_telemetry.insert(label.to_string(), telemetry);
        self.refresh_launcher_status();
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
                    preview_key,
                    result,
                } => {
                    let mut status = format!("{op} failed");
                    if let Some(session) = self.state.label_to_session.get_mut(&window_label) {
                        if !session.is_active_job(job_id, generation) {
                            continue;
                        }
                        state_changed = true;

                        match result {
                            Ok(dataset) => {
                                match mode {
                                    OpRunMode::Preview => {
                                        if let Some(key) = preview_key {
                                            session.preview_cache.insert(key.clone(), dataset);
                                            session.set_active_preview(Some(key));
                                        }
                                    }
                                    OpRunMode::Apply => {
                                        session.commit_dataset(dataset);
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
        let width = params.get("width").and_then(Value::as_u64).unwrap_or(512) as usize;
        let height = params.get("height").and_then(Value::as_u64).unwrap_or(512) as usize;
        let slices = params.get("slices").and_then(Value::as_u64).unwrap_or(1) as usize;
        let channels = params.get("channels").and_then(Value::as_u64).unwrap_or(1) as usize;
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
        let len: usize = shape.iter().product();
        let data = ArrayD::from_shape_vec(IxDyn(&shape), vec![fill; len])
            .map_err(|error| format!("new image shape error: {error}"))?;
        let metadata = Metadata {
            dims,
            pixel_type,
            ..Metadata::default()
        };
        let dataset = Dataset::new(data, metadata).map_err(|error| error.to_string())?;
        let path = normalize_path(&PathBuf::from(format!(
            "Untitled-{}.tif",
            self.state.next_window_id + 1
        )));
        let label = self.create_viewer(path, ViewerImageSource::Dataset(Arc::new(dataset)));
        Ok(format!("created new image in {label}"))
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

    fn build_profile_plot(&mut self, window_label: &str) -> Result<String, String> {
        let viewer_label = self
            .current_viewer_label(window_label)
            .ok_or_else(|| "a loaded image is required for plot profile".to_string())?
            .to_string();
        let (slice, roi_bbox, _, _, _) = self.measurement_context(&viewer_label)?;
        let samples = if let Some((min_x, min_y, max_x, max_y)) = roi_bbox {
            if max_y > min_y {
                mean_profile_rect(&slice.values, slice.width, min_x, min_y, max_x, max_y)
            } else {
                sample_line_profile(
                    &slice.values,
                    slice.width,
                    min_x as f32,
                    min_y as f32,
                    max_x as f32,
                    max_y as f32,
                )
            }
        } else {
            let y = slice.height.saturating_sub(1) / 2;
            mean_profile_rect(
                &slice.values,
                slice.width,
                0,
                y,
                slice.width.saturating_sub(1),
                y,
            )
        };
        self.profile_plot.title = format!("Profile ({viewer_label})");
        self.profile_plot.samples = samples;
        self.desktop_state.utility_windows.profile_plot_open = true;
        self.persist_desktop_state();
        Ok("profile plot opened".to_string())
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
        )?;
        Ok("clipboard pasted".to_string())
    }

    fn layout_viewers(&mut self, mode: LayoutMode) -> Result<String, String> {
        let mut labels = self.state.label_to_path.keys().cloned().collect::<Vec<_>>();
        labels.sort_by_key(|label| viewer_sort_key(label));
        if labels.is_empty() {
            return Err("no viewers are open".to_string());
        }
        for (index, label) in labels.iter().enumerate() {
            if let Some(viewer) = self.viewers_ui.get(label) {
                let position = match mode {
                    LayoutMode::Tile => {
                        let columns = (labels.len() as f32).sqrt().ceil() as usize;
                        let row = index / columns;
                        let col = index % columns;
                        egui::pos2(40.0 + col as f32 * 420.0, 60.0 + row as f32 * 360.0)
                    }
                    LayoutMode::Cascade => {
                        egui::pos2(40.0 + index as f32 * 32.0, 60.0 + index as f32 * 28.0)
                    }
                };
                self.focus_viewer_label = Some(label.clone());
                self.launcher_ui.fallback_text = match mode {
                    LayoutMode::Tile => "Tiled viewer windows".to_string(),
                    LayoutMode::Cascade => "Cascaded viewer windows".to_string(),
                };
                let _ = viewer.viewport_id;
                self.viewport_positions.insert(label.clone(), position);
            }
        }
        Ok(match mode {
            LayoutMode::Tile => "viewer windows tiled".to_string(),
            LayoutMode::Cascade => "viewer windows cascaded".to_string(),
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
                    patch.data[IxDyn(&[py, px])]
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

    fn dispatch_command(
        &mut self,
        window_label: &str,
        command_id: &str,
        params: Option<Value>,
    ) -> command_registry::CommandExecuteResult {
        if let Some(result) = self.handle_local_command(window_label, command_id, params.as_ref()) {
            return result;
        }

        let request = command_registry::CommandExecuteRequest {
            command_id: command_id.to_string(),
            params,
        };
        self.execute_command(window_label, request)
    }

    fn handle_local_command(
        &mut self,
        window_label: &str,
        command_id: &str,
        params: Option<&Value>,
    ) -> Option<command_registry::CommandExecuteResult> {
        if command_id == "file.open" {
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

        if command_id == "file.new" {
            if let Some(params) = params {
                return Some(match self.create_new_image(params) {
                    Ok(message) => command_registry::CommandExecuteResult::ok(message),
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                });
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
            return Some(
                match self.viewer_start_op(
                    &viewer_label,
                    ViewerOpRequest {
                        op: op.to_string(),
                        params,
                        mode: OpRunMode::Apply,
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
                    "{} tool selected (behavior not implemented yet)",
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
                std::mem::swap(
                    &mut self.tool_options.foreground_color,
                    &mut self.tool_options.background_color,
                );
                return Some(command_registry::CommandExecuteResult::ok(
                    "foreground/background swapped",
                ));
            }
            "tool.dropper.palette.background" => {
                std::mem::swap(
                    &mut self.tool_options.foreground_color,
                    &mut self.tool_options.background_color,
                );
                return Some(command_registry::CommandExecuteResult::ok(
                    "foreground/background swapped",
                ));
            }
            "tool.dropper.palette.colors" => {
                return Some(command_registry::CommandExecuteResult::ok(
                    "color dialog is not implemented yet",
                ));
            }
            "tool.dropper.palette.color_picker" => {
                return Some(command_registry::CommandExecuteResult::ok(
                    "color picker dialog is not implemented yet",
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
            "image.stacks.set" => Some(command_registry::CommandExecuteResult::ok(
                "set slice dialog is not implemented yet",
            )),
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
                viewer.pending_zoom = Some(ZoomCommand::Set);
                Some(command_registry::CommandExecuteResult::ok(
                    "zoom set dialog is not implemented yet",
                ))
            }
            "image.zoom.maximize" => {
                viewer.pending_zoom = Some(ZoomCommand::Maximize);
                Some(command_registry::CommandExecuteResult::ok("zoom maximized"))
            }
            "viewer.roi.clear" => {
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
            "file.close" => {
                if window_label == LAUNCHER_LABEL {
                    command_registry::CommandExecuteResult::blocked(
                        "the launcher window is always visible",
                    )
                } else {
                    self.remove_viewer_by_label(window_label);
                    command_registry::CommandExecuteResult::ok("window closed")
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
            "process.smooth" | "process.gaussian" => {
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
                if let Some(session) = self.state.label_to_session.get(window_label) {
                    self.resize_dialog.width = session.committed_summary.shape[1];
                    self.resize_dialog.height = session.committed_summary.shape[0];
                }
                self.resize_dialog.open = true;
                command_registry::CommandExecuteResult::ok("resize dialog opened")
            }
            "image.adjust.canvas" => {
                if let Some(session) = self.state.label_to_session.get(window_label) {
                    self.canvas_dialog.width = session.committed_summary.shape[1];
                    self.canvas_dialog.height = session.committed_summary.shape[0];
                }
                self.canvas_dialog.open = true;
                command_registry::CommandExecuteResult::ok("canvas size dialog opened")
            }
            "image.adjust.brightness" => match self.viewer_start_op(
                window_label,
                ViewerOpRequest {
                    op: "intensity.normalize".to_string(),
                    params: json!({}),
                    mode: OpRunMode::Apply,
                },
            ) {
                Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                    "auto contrast applied",
                    json!({ "job_id": ticket.job_id }),
                ),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "image.adjust.threshold" | "process.binary.make" => match self.viewer_start_op(
                window_label,
                ViewerOpRequest {
                    op: "threshold.otsu".to_string(),
                    params: json!({}),
                    mode: OpRunMode::Apply,
                },
            ) {
                Ok(ticket) => command_registry::CommandExecuteResult::with_payload(
                    "binary threshold started",
                    json!({ "job_id": ticket.job_id }),
                ),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "process.binary.erode" | "process.binary.dilate" => {
                let op = if request.command_id.ends_with("erode") {
                    "morphology.erode"
                } else {
                    "morphology.dilate"
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
                        "binary morphology started",
                        json!({ "job_id": ticket.job_id, "op": op }),
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
            "analyze.measure" => match self.measure_active_viewer(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
            "analyze.histogram" => {
                let Some(viewer) = self.viewers_ui.get(window_label) else {
                    return command_registry::CommandExecuteResult::blocked(
                        "a loaded image is required for histogram",
                    );
                };
                let request = ViewerFrameRequest {
                    z: viewer.z,
                    t: viewer.t,
                    channel: viewer.channel,
                };
                match self.active_dataset_slice(window_label, &request) {
                    Ok(slice) => {
                        let (min, max) = min_max(&slice.values);
                        let span = (max - min).max(f32::EPSILON);
                        let mut bins = vec![0u64; 256];
                        for value in &slice.values {
                            let normalized = ((*value - min) / span).clamp(0.0, 1.0);
                            let bin = (normalized * 255.0).round() as usize;
                            bins[bin] += 1;
                        }
                        command_registry::CommandExecuteResult::with_payload(
                            "histogram complete",
                            json!({
                                "min": min,
                                "max": max,
                                "bins": bins,
                                "pixelCount": slice.values.len(),
                                "z": viewer.z,
                                "t": viewer.t,
                                "channel": viewer.channel
                            }),
                        )
                    }
                    Err(error) => command_registry::CommandExecuteResult::blocked(error),
                }
            }
            "analyze.set_measurements" => {
                self.desktop_state.utility_windows.measurements_open = true;
                self.persist_desktop_state();
                command_registry::CommandExecuteResult::ok("measurement settings opened")
            }
            "analyze.tools.results" => {
                self.desktop_state.utility_windows.results_open = true;
                self.persist_desktop_state();
                command_registry::CommandExecuteResult::ok("results table opened")
            }
            "analyze.tools.roi_manager" => {
                self.desktop_state.utility_windows.roi_manager_open = true;
                self.persist_desktop_state();
                command_registry::CommandExecuteResult::ok("ROI manager opened")
            }
            "analyze.plot_profile" => match self.build_profile_plot(window_label) {
                Ok(message) => command_registry::CommandExecuteResult::ok(message),
                Err(error) => command_registry::CommandExecuteResult::blocked(error),
            },
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
            | "plugins.utilities.startup" => command_registry::CommandExecuteResult::ok(
                "macro/plugin compatibility is deferred in image-rs",
            ),
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
            OpRunMode::Apply => None,
        };

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
                .map(|output| Arc::new(output.dataset))
                .map_err(|error| error.to_string());

            let _ = tx.send(WorkerEvent::OpFinished {
                window_label,
                job_id,
                generation,
                mode,
                op: op_name,
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
                let color = to_color_image(&frame);
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
        let metadata = command_registry::metadata(command_id);
        if !metadata.implemented || !metadata.scope.contains(window_label) {
            return false;
        }
        if metadata.requires_image && !self.state.label_to_session.contains_key(window_label) {
            return false;
        }
        if let Some(session) = self.state.label_to_session.get(window_label) {
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
        items: &[MenuManifestItem],
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
        }

        if wants_keyboard {
            return;
        }

        for event in &input.events {
            let egui::Event::Text(text) = event else {
                continue;
            };

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

    fn draw_launcher(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        self.handle_shortcuts(ctx, LAUNCHER_LABEL, actions);
        self.refresh_launcher_status();

        egui::TopBottomPanel::top("launcher-header").show(ctx, |ui| {
            self.draw_menu_bar(ui, LAUNCHER_LABEL, actions);
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                self.draw_launcher_toolbar(ui, actions);
            });
        });

        egui::CentralPanel::default().show(ctx, |_ui| {});

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
        self.draw_resize_dialog(ctx, actions, false);
        self.draw_resize_dialog(ctx, actions, true);
        self.draw_raw_import_dialog(ctx);
        self.draw_url_import_dialog(ctx, actions);
        self.draw_results_window(ctx);
        self.draw_measurement_settings_window(ctx);
        self.draw_roi_manager_window(ctx, actions);
        self.draw_profile_plot_window(ctx);
        self.draw_help_windows(ctx);
        self.draw_command_finder_window(ctx, actions);
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
                            "fill": self.new_image_dialog.fill,
                            "pixelType": pixel_type_id(self.new_image_dialog.pixel_type),
                        })),
                    });
                    self.new_image_dialog.open = false;
                }
            });
        self.new_image_dialog.open = open;
    }

    fn draw_resize_dialog(
        &mut self,
        ctx: &egui::Context,
        actions: &mut Vec<UiAction>,
        canvas: bool,
    ) {
        let dialog = if canvas {
            &mut self.canvas_dialog
        } else {
            &mut self.resize_dialog
        };
        if !dialog.open {
            return;
        }
        let mut open = dialog.open;
        egui::Window::new(if canvas { "Canvas Size" } else { "Resize" })
            .open(&mut open)
            .show(ctx, |ui| {
                ui.add(egui::DragValue::new(&mut dialog.width).prefix("Width "));
                ui.add(egui::DragValue::new(&mut dialog.height).prefix("Height "));
                if canvas {
                    ui.add(egui::DragValue::new(&mut dialog.fill).prefix("Fill "));
                }
                if ui.button("Apply").clicked() {
                    let command_id = if canvas {
                        "__dialog.canvas_resize"
                    } else {
                        "__dialog.resize"
                    };
                    actions.push(UiAction::Command {
                        window_label: self
                            .active_viewer_label
                            .clone()
                            .unwrap_or_else(|| LAUNCHER_LABEL.to_string()),
                        command_id: command_id.to_string(),
                        params: Some(json!({
                            "width": dialog.width,
                            "height": dialog.height,
                            "fill": dialog.fill,
                        })),
                    });
                    dialog.open = false;
                }
            });
        dialog.open = open;
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

    fn draw_viewer_viewport(
        &mut self,
        ctx: &egui::Context,
        label: &str,
        actions: &mut Vec<UiAction>,
    ) {
        self.handle_shortcuts(ctx, label, actions);

        if ctx.input(|i| i.viewport().close_requested()) {
            actions.push(UiAction::CloseViewer {
                label: label.to_string(),
            });
            return;
        }

        self.ensure_frame_for_viewer(ctx, label);

        egui::TopBottomPanel::top(format!("viewer-header-{label}")).show(ctx, |ui| {
            self.draw_menu_bar(ui, label, actions);
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                self.draw_viewer_toolbar(ui, label, actions);
            });
        });

        let summary = summary_for_window(&self.state, label)
            .ok()
            .map(|(_, summary)| summary);

        let mut close_requested = false;
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

            if ui_close_requested(ctx) {
                close_requested = true;
            }

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

        if close_requested {
            actions.push(UiAction::CloseViewer {
                label: label.to_string(),
            });
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
                    let label = self
                        .command_catalog
                        .entries
                        .iter()
                        .find(|entry| entry.id == command_id)
                        .map(|entry| entry.label.clone())
                        .unwrap_or_else(|| command_id.clone());
                    let result = self.dispatch_command(&window_label, &command_id, params);
                    if window_label == LAUNCHER_LABEL {
                        self.set_fallback_status(format!("{label}: {}", result.message));
                    } else if let Some(viewer) = self.viewers_ui.get_mut(&window_label) {
                        viewer.tool_message = Some(result.message.clone());
                        viewer.status_message = result.message;
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
            && let Some(viewer) = self.viewers_ui.get(&label)
        {
            ctx.send_viewport_cmd_to(viewer.viewport_id, egui::ViewportCommand::Focus);
        }
    }
}

impl eframe::App for ImageUiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let worker_state_changed = self.poll_worker_events();

        let mut actions = Vec::new();
        self.draw_launcher(ctx, &mut actions);

        let labels = self.state.label_to_path.keys().cloned().collect::<Vec<_>>();
        let mut existing_labels = HashSet::new();
        for label in labels {
            if !self.viewers_ui.contains_key(&label) {
                continue;
            }
            existing_labels.insert(label.clone());
            let viewport_id = viewport_id_for_label(&label);
            let summary = summary_for_window(&self.state, &label)
                .ok()
                .map(|(_, summary)| summary);
            let (title, initial_size) = if let Some(viewer) = self.viewers_ui.get_mut(&label) {
                let initial_size = if !viewer.initial_viewport_sized {
                    let size = summary.as_ref().map(|summary| {
                        compute_initial_viewport_size(
                            summary,
                            ctx.input(|i| i.viewport().monitor_size),
                        )
                    });
                    viewer.initial_viewport_sized = true;
                    size
                } else {
                    None
                };
                (viewer.title.clone(), initial_size)
            } else {
                ("image-rs Viewer".to_string(), None)
            };

            let mut builder = egui::ViewportBuilder::default()
                .with_title(title)
                .with_min_inner_size(VIEWER_MIN_WINDOW_SIZE)
                .with_clamp_size_to_monitor_size(true);
            if let Some(size) = initial_size {
                builder = builder.with_inner_size(size);
            }

            ctx.show_viewport_immediate(viewport_id, builder, |ctx, _class| {
                self.draw_viewer_viewport(ctx, &label, &mut actions);
            });

            if let Some(position) = self.viewport_positions.get(&label).copied() {
                ctx.send_viewport_cmd_to(
                    viewport_id,
                    egui::ViewportCommand::OuterPosition(position),
                );
            }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolbarKind {
    Button,
    Separator,
}

#[derive(Debug, Clone, Copy)]
struct LauncherToolbarItem {
    kind: ToolbarKind,
    command_id: &'static str,
    label: &'static str,
    icon: ToolbarIcon,
}

#[derive(Debug, Clone, Copy)]
struct ViewerToolbarItem {
    kind: ToolbarKind,
    command_id: &'static str,
    label: &'static str,
    glyph: &'static str,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ToolbarIcon {
    Rect,
    Oval,
    Poly,
    Free,
    Line,
    Angle,
    Point,
    Wand,
    Text,
    Open,
    Zoom,
    Hand,
    Dropper,
    Previous,
    Next,
    Quit,
    Custom1,
    Custom2,
    Custom3,
    More,
}

const fn launcher_toolbar_button(
    command_id: &'static str,
    label: &'static str,
    icon: ToolbarIcon,
) -> LauncherToolbarItem {
    LauncherToolbarItem {
        kind: ToolbarKind::Button,
        command_id,
        label,
        icon,
    }
}

const fn launcher_toolbar_separator() -> LauncherToolbarItem {
    LauncherToolbarItem {
        kind: ToolbarKind::Separator,
        command_id: "",
        label: "",
        icon: ToolbarIcon::Rect,
    }
}

const fn viewer_toolbar_button(
    command_id: &'static str,
    label: &'static str,
    glyph: &'static str,
) -> ViewerToolbarItem {
    ViewerToolbarItem {
        kind: ToolbarKind::Button,
        command_id,
        label,
        glyph,
    }
}

const fn viewer_toolbar_separator() -> ViewerToolbarItem {
    ViewerToolbarItem {
        kind: ToolbarKind::Separator,
        command_id: "",
        label: "",
        glyph: "",
    }
}

fn launcher_toolbar_items() -> &'static [LauncherToolbarItem] {
    const ITEMS: &[LauncherToolbarItem] = &[
        launcher_toolbar_button("launcher.tool.rect", "Rectangle", ToolbarIcon::Rect),
        launcher_toolbar_button("launcher.tool.oval", "Oval", ToolbarIcon::Oval),
        launcher_toolbar_button("launcher.tool.poly", "Polygon", ToolbarIcon::Poly),
        launcher_toolbar_button("launcher.tool.free", "Freehand", ToolbarIcon::Free),
        launcher_toolbar_button("launcher.tool.line", "Line", ToolbarIcon::Line),
        launcher_toolbar_button("launcher.tool.angle", "Angle", ToolbarIcon::Angle),
        launcher_toolbar_button("launcher.tool.point", "Point", ToolbarIcon::Point),
        launcher_toolbar_button("launcher.tool.wand", "Wand", ToolbarIcon::Wand),
        launcher_toolbar_button("launcher.tool.text", "Text", ToolbarIcon::Text),
        launcher_toolbar_separator(),
        launcher_toolbar_button("launcher.tool.zoom", "Zoom", ToolbarIcon::Zoom),
        launcher_toolbar_button("launcher.tool.hand", "Hand", ToolbarIcon::Hand),
        launcher_toolbar_button("launcher.tool.dropper", "Dropper", ToolbarIcon::Dropper),
        launcher_toolbar_separator(),
        launcher_toolbar_button("launcher.tool.more", "More Tools", ToolbarIcon::More),
    ];
    ITEMS
}

fn viewer_toolbar_items() -> &'static [ViewerToolbarItem] {
    const ITEMS: &[ViewerToolbarItem] = &[
        viewer_toolbar_button("file.open", "Open", "O"),
        viewer_toolbar_button("file.close", "Close", "C"),
        viewer_toolbar_separator(),
        viewer_toolbar_button("image.zoom.in", "Zoom In", "+"),
        viewer_toolbar_button("image.zoom.out", "Zoom Out", "-"),
        viewer_toolbar_button("image.zoom.original", "Original Scale", "Orig"),
        viewer_toolbar_button("image.zoom.view100", "View 100%", "100%"),
        viewer_toolbar_button("image.zoom.to_selection", "To Selection", "Sel"),
        viewer_toolbar_separator(),
        viewer_toolbar_button("process.smooth", "Smooth", "S"),
        viewer_toolbar_button("process.gaussian", "Gaussian Blur", "G"),
        viewer_toolbar_separator(),
        viewer_toolbar_button("analyze.measure", "Measure", "M"),
    ];
    ITEMS
}

fn toolbar_icon_asset(icon: ToolbarIcon) -> (&'static str, &'static [u8]) {
    match icon {
        ToolbarIcon::Rect => ("rect", include_bytes!("assets/tools/rect.png").as_slice()),
        ToolbarIcon::Oval => ("oval", include_bytes!("assets/tools/oval.png").as_slice()),
        ToolbarIcon::Poly => ("poly", include_bytes!("assets/tools/poly.png").as_slice()),
        ToolbarIcon::Free => ("free", include_bytes!("assets/tools/free.png").as_slice()),
        ToolbarIcon::Line => ("line", include_bytes!("assets/tools/line.png").as_slice()),
        ToolbarIcon::Angle => ("angle", include_bytes!("assets/tools/line.png").as_slice()),
        ToolbarIcon::Point => ("point", include_bytes!("assets/tools/point.png").as_slice()),
        ToolbarIcon::Wand => ("wand", include_bytes!("assets/tools/wand.png").as_slice()),
        ToolbarIcon::Text => ("text", include_bytes!("assets/tools/text.png").as_slice()),
        ToolbarIcon::Open => ("open", include_bytes!("assets/tools/open.png").as_slice()),
        ToolbarIcon::Zoom => ("zoom", include_bytes!("assets/tools/zoom.png").as_slice()),
        ToolbarIcon::Hand => ("hand", include_bytes!("assets/tools/hand.png").as_slice()),
        ToolbarIcon::Dropper => (
            "dropper",
            include_bytes!("assets/tools/dropper.png").as_slice(),
        ),
        ToolbarIcon::Previous => (
            "previous",
            include_bytes!("assets/tools/previous.png").as_slice(),
        ),
        ToolbarIcon::Next => ("next", include_bytes!("assets/tools/next.png").as_slice()),
        ToolbarIcon::Quit => ("quit", include_bytes!("assets/tools/quit.png").as_slice()),
        ToolbarIcon::Custom1 => (
            "custom1",
            include_bytes!("assets/tools/custom1.png").as_slice(),
        ),
        ToolbarIcon::Custom2 => (
            "custom2",
            include_bytes!("assets/tools/custom2.png").as_slice(),
        ),
        ToolbarIcon::Custom3 => (
            "custom3",
            include_bytes!("assets/tools/custom3.png").as_slice(),
        ),
        ToolbarIcon::More => ("more", include_bytes!("assets/tools/more.png").as_slice()),
    }
}

fn tool_from_command_id(command_id: &str) -> Option<ToolId> {
    interaction::tooling::tool_from_command_id(command_id)
}

fn tool_shortcut_command(text: &str) -> Option<&'static str> {
    interaction::tooling::tool_shortcut_command(text)
}

#[allow(dead_code)]
fn tool_shortcut_tool(text: &str) -> Option<ToolId> {
    interaction::tooling::tool_shortcut_tool(text)
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
        ZoomCommand::Set => {}
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
    position: interaction::roi::RoiPosition,
) {
    for roi in viewer.rois.visible_rois(position) {
        let selected = viewer.rois.selected_roi_id == Some(roi.id);
        let stroke = if selected {
            egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 212, 26))
        } else {
            egui::Stroke::new(1.5, egui::Color32::from_rgb(52, 212, 255))
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
            RoiKind::Polygon { points, closed } => {
                let mut line_points = points
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

    let frame = Arc::new(build_frame(&source, request)?);

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

fn value_to_display(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => value.to_string(),
    }
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

fn selected_roi_bbox(viewer: &ViewerUiState) -> Option<(usize, usize, usize, usize)> {
    let roi = viewer
        .rois
        .selected_roi_id
        .and_then(|id| viewer.rois.overlay_rois.iter().find(|roi| roi.id == id))
        .or(viewer.rois.active_roi.as_ref())?;
    roi_kind_bbox(&roi.kind)
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

fn mean_profile_rect(
    values: &[f32],
    width: usize,
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
) -> Vec<f32> {
    let row_count = max_y.saturating_sub(min_y) + 1;
    let mut output = Vec::new();
    for x in min_x..=max_x {
        let mut sum = 0.0;
        for y in min_y..=max_y {
            sum += values[x + y * width];
        }
        output.push(sum / row_count as f32);
    }
    output
}

fn sample_line_profile(
    values: &[f32],
    width: usize,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
) -> Vec<f32> {
    let steps = ((x1 - x0).hypot(y1 - y0)).round().max(1.0) as usize;
    let height = values.len() / width;
    let mut output = Vec::with_capacity(steps + 1);
    for step in 0..=steps {
        let t = step as f32 / steps.max(1) as f32;
        let x = x0 + (x1 - x0) * t;
        let y = y0 + (y1 - y0) * t;
        let x_clamped = x.clamp(0.0, width.saturating_sub(1) as f32);
        let y_clamped = y.clamp(0.0, height.saturating_sub(1) as f32);
        let x_floor = x_clamped.floor() as usize;
        let y_floor = y_clamped.floor() as usize;
        output.push(values[x_floor + y_floor * width]);
    }
    output
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

fn build_frame(
    source: &ViewerImageSource,
    request: &ViewerFrameRequest,
) -> Result<ViewerFrameBuffer, String> {
    match source {
        ViewerImageSource::Native(image) => build_native_frame(image.as_ref(), request),
        ViewerImageSource::Dataset(dataset) => build_dataset_frame(dataset.as_ref(), request),
    }
}

fn build_dataset_frame(
    dataset: &DatasetF32,
    request: &ViewerFrameRequest,
) -> Result<ViewerFrameBuffer, String> {
    let slice = extract_slice(dataset, request.z, request.t, request.channel)?;
    let pixels_u8 = to_u8_samples(&slice.values);
    let (min, max) = min_max(&slice.values);

    Ok(ViewerFrameBuffer {
        width: slice.width,
        height: slice.height,
        values: slice.values.clone(),
        pixels_u8,
        min,
        max,
    })
}

fn build_native_frame(
    image: &NativeRasterImage,
    request: &ViewerFrameRequest,
) -> Result<ViewerFrameBuffer, String> {
    if request.z > 0 || request.t > 0 {
        return Err("native rasters expose only a single Z/T plane".to_string());
    }

    match image {
        NativeRasterImage::Gray8 {
            width,
            height,
            pixels,
            ..
        } => {
            let (min, max) = image.min_max();
            Ok(ViewerFrameBuffer {
                width: *width,
                height: *height,
                values: pixels
                    .iter()
                    .map(|value| f32::from(*value) / 255.0)
                    .collect(),
                pixels_u8: pixels.clone(),
                min,
                max,
            })
        }
        NativeRasterImage::Gray16 {
            width,
            height,
            pixels,
            ..
        } => {
            let (min, max) = image.min_max();
            Ok(ViewerFrameBuffer {
                width: *width,
                height: *height,
                values: pixels
                    .iter()
                    .map(|value| f32::from(*value) / 65_535.0)
                    .collect(),
                pixels_u8: pixels.iter().map(|value| (value / 257) as u8).collect(),
                min,
                max,
            })
        }
        NativeRasterImage::Rgb8 {
            width,
            height,
            pixels,
            ..
        } => {
            let channel = request.channel.min(2);
            let mut pixels_u8 = Vec::with_capacity(width * height);
            let mut min_sample = u8::MAX;
            let mut max_sample = u8::MIN;
            for chunk in pixels.chunks_exact(3) {
                let sample = chunk[channel];
                min_sample = min_sample.min(sample);
                max_sample = max_sample.max(sample);
                pixels_u8.push(sample);
            }
            Ok(ViewerFrameBuffer {
                width: *width,
                height: *height,
                values: pixels_u8
                    .iter()
                    .map(|value| f32::from(*value) / 255.0)
                    .collect(),
                pixels_u8,
                min: f32::from(min_sample) / 255.0,
                max: f32::from(max_sample) / 255.0,
            })
        }
    }
}

fn summarize_source(source: &ViewerImageSource, path: &Path) -> ImageSummary {
    match source {
        ViewerImageSource::Native(image) => summarize_native_image(image.as_ref(), path),
        ViewerImageSource::Dataset(dataset) => summarize_dataset(dataset.as_ref(), path),
    }
}

fn summarize_dataset(dataset: &DatasetF32, source: &Path) -> ImageSummary {
    let (min, max) = dataset.min_max().unwrap_or((0.0, 0.0));
    summarize_metadata(dataset.shape(), &dataset.metadata.dims, min, max, source)
}

fn summarize_native_image(image: &NativeRasterImage, source: &Path) -> ImageSummary {
    let metadata = image.metadata();
    let shape = if image.channel_count() > 1 {
        vec![image.height(), image.width(), image.channel_count()]
    } else {
        vec![image.height(), image.width()]
    };
    let (min, max) = image.min_max();
    summarize_metadata(&shape, &metadata.dims, min, max, source)
}

fn summarize_metadata(
    shape: &[usize],
    dims: &[Dim],
    min: f32,
    max: f32,
    source: &Path,
) -> ImageSummary {
    let channel_axis = dims
        .iter()
        .position(|dimension| dimension.axis == AxisKind::Channel);
    let z_axis = dims
        .iter()
        .position(|dimension| dimension.axis == AxisKind::Z);
    let t_axis = dims
        .iter()
        .position(|dimension| dimension.axis == AxisKind::Time);

    ImageSummary {
        shape: shape.to_vec(),
        axes: dims
            .iter()
            .map(|dimension| format!("{:?}", dimension.axis))
            .collect(),
        channels: channel_axis.map(|index| shape[index]).unwrap_or(1),
        z_slices: z_axis.map(|index| shape[index]).unwrap_or(1),
        times: t_axis.map(|index| shape[index]).unwrap_or(1),
        min,
        max,
        source: source.display().to_string(),
    }
}

fn extract_slice(
    dataset: &DatasetF32,
    z: usize,
    t: usize,
    channel: usize,
) -> Result<SliceImage, String> {
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
    if y_axis == x_axis {
        return Err("could not infer distinct X/Y axes".to_string());
    }

    let mut index = vec![0usize; dataset.ndim()];
    if let Some(axis) = dataset.axis_index(AxisKind::Z) {
        index[axis] = z.min(dataset.shape()[axis].saturating_sub(1));
    }
    if let Some(axis) = dataset.axis_index(AxisKind::Time) {
        index[axis] = t.min(dataset.shape()[axis].saturating_sub(1));
    }
    if let Some(axis) = dataset.axis_index(AxisKind::Channel) {
        index[axis] = channel.min(dataset.shape()[axis].saturating_sub(1));
    }

    let height = dataset.shape()[y_axis];
    let width = dataset.shape()[x_axis];
    let mut values = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            index[y_axis] = y;
            index[x_axis] = x;
            values.push(dataset.data[IxDyn(&index)]);
        }
    }

    Ok(SliceImage {
        width,
        height,
        values,
    })
}

fn extract_slice_from_source(
    source: &ViewerImageSource,
    z: usize,
    t: usize,
    channel: usize,
) -> Result<SliceImage, String> {
    match source {
        ViewerImageSource::Native(image) => {
            extract_slice_from_native(image.as_ref(), z, t, channel)
        }
        ViewerImageSource::Dataset(dataset) => extract_slice(dataset.as_ref(), z, t, channel),
    }
}

fn extract_slice_from_native(
    image: &NativeRasterImage,
    z: usize,
    t: usize,
    channel: usize,
) -> Result<SliceImage, String> {
    if z > 0 || t > 0 {
        return Err("native rasters expose only a single Z/T plane".to_string());
    }

    match image {
        NativeRasterImage::Gray8 {
            width,
            height,
            pixels,
            ..
        } => Ok(SliceImage {
            width: *width,
            height: *height,
            values: pixels
                .iter()
                .map(|value| f32::from(*value) / 255.0)
                .collect(),
        }),
        NativeRasterImage::Gray16 {
            width,
            height,
            pixels,
            ..
        } => Ok(SliceImage {
            width: *width,
            height: *height,
            values: pixels
                .iter()
                .map(|value| f32::from(*value) / 65_535.0)
                .collect(),
        }),
        NativeRasterImage::Rgb8 {
            width,
            height,
            pixels,
            ..
        } => {
            let selected = channel.min(2);
            Ok(SliceImage {
                width: *width,
                height: *height,
                values: pixels
                    .chunks_exact(3)
                    .map(|chunk| f32::from(chunk[selected]) / 255.0)
                    .collect(),
            })
        }
    }
}

fn to_u8_samples(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

fn min_max(values: &[f32]) -> (f32, f32) {
    let mut iter = values.iter().copied();
    let first = iter.next().unwrap_or(0.0);
    let mut min = first;
    let mut max = first;
    for value in iter {
        min = min.min(value);
        max = max.max(value);
    }
    (min, max)
}

fn to_color_image(frame: &ViewerFrameBuffer) -> egui::ColorImage {
    let mut rgba = Vec::with_capacity(frame.pixels_u8.len() * 4);
    for gray in &frame.pixels_u8 {
        rgba.extend_from_slice(&[*gray, *gray, *gray, 255]);
    }
    egui::ColorImage::from_rgba_unmultiplied([frame.width, frame.height], &rgba)
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

fn ui_close_requested(ctx: &egui::Context) -> bool {
    ctx.input(|i| i.viewport().close_requested())
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

fn viewport_id_for_label(label: &str) -> egui::ViewportId {
    egui::ViewportId::from_hash_of(format!("viewport-{label}"))
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
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;

    use crate::model::{DatasetF32, PixelType};
    use crate::ui::interaction::transform::ViewerTransformState;
    use eframe::egui;
    use ndarray::Array;
    use serde_json::json;

    use super::{
        HoverInfo, ImageSummary, LauncherStatusModel, NativeRasterImage, ProgressState,
        RepaintDecisionInputs, ToolId, UiState, ViewerFrameBuffer, ViewerFrameRequest,
        ViewerImageSource, ViewerSession, ViewerTelemetry, ViewerUiState, ZoomCommand,
        apply_zoom_command, build_frame, canonical_json, compute_initial_viewport_size,
        compute_viewer_frame, dominant_scroll_component, effective_scroll_delta,
        format_launcher_status, image_draw_rect, initialize_view_to_open_state, preview_cache_key,
        should_request_periodic_repaint, should_request_repaint_now, source_ptr_eq,
        tool_from_command_id, tool_shortcut_command, viewer_sort_key,
    };

    fn dataset_2x2(values: [f32; 4]) -> Arc<DatasetF32> {
        let data = Array::from_shape_vec((2, 2), values.to_vec())
            .expect("shape")
            .into_dyn();
        Arc::new(DatasetF32::from_data_with_default_metadata(
            data,
            PixelType::F32,
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
    fn viewer_sort_key_uses_numeric_suffix() {
        assert_eq!(viewer_sort_key("viewer-42"), 42);
        assert_eq!(viewer_sort_key("viewer-abc"), u64::MAX);
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

        let native_frame = build_frame(&native, &request).expect("native frame");
        let dataset_frame = build_frame(&dataset, &request).expect("dataset frame");

        assert_eq!(native_frame.width, dataset_frame.width);
        assert_eq!(native_frame.height, dataset_frame.height);
        assert_eq!(native_frame.pixels_u8, dataset_frame.pixels_u8);
        assert_eq!(native_frame.min, dataset_frame.min);
        assert_eq!(native_frame.max, dataset_frame.max);
    }

    #[test]
    fn dataset_frame_does_not_auto_stretch_display_range() {
        let dataset = ViewerImageSource::Dataset(dataset_2x2([0.25, 0.5, 0.75, 1.0]));
        let frame = build_frame(&dataset, &ViewerFrameRequest::default()).expect("frame");

        assert_eq!(frame.pixels_u8, vec![64, 128, 191, 255]);
        assert_eq!(frame.values, vec![0.25, 0.5, 0.75, 1.0]);
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
