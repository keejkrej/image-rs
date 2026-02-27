#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod command_registry;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    mpsc::{self, Receiver, Sender},
};

use eframe::egui;
use image::load_from_memory;
use image_model::{AxisKind, DatasetF32};
use image_runtime::AppContext;
use ndarray::IxDyn;
use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

const LAUNCHER_LABEL: &str = "main";
const VIEWER_PREFIX: &str = "viewer-";
const SOURCE_COMMITTED: &str = "committed";
const SOURCE_PREVIEW_PREFIX: &str = "preview:";
const VIEWER_DEFAULT_SIZE: [f32; 2] = [980.0, 980.0];
const LAUNCHER_WINDOW_SIZE: [f32; 2] = [840.0, 148.0];

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
    base_dataset: Arc<DatasetF32>,
    committed_dataset: Arc<DatasetF32>,
    active_preview: Option<String>,
    preview_cache: HashMap<String, Arc<DatasetF32>>,
    frame_cache: HashMap<FrameKey, ViewerFrameBuffer>,
    generation: u64,
    active_job: Option<ActiveJob>,
}

impl ViewerSession {
    fn new(path: PathBuf, dataset: Arc<DatasetF32>) -> Self {
        Self {
            path,
            base_dataset: dataset.clone(),
            committed_dataset: dataset,
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

    fn dataset_for_source(&self, source_kind: &str) -> Option<Arc<DatasetF32>> {
        if source_kind == SOURCE_COMMITTED {
            return Some(self.committed_dataset.clone());
        }

        if let Some(key) = source_kind.strip_prefix(SOURCE_PREVIEW_PREFIX) {
            return self.preview_cache.get(key).cloned();
        }

        None
    }

    fn set_active_preview(&mut self, key: Option<String>) {
        self.active_preview = key;
        self.frame_cache.clear();
    }

    fn commit_dataset(&mut self, dataset: Arc<DatasetF32>) {
        self.committed_dataset = dataset;
        self.active_preview = None;
        self.preview_cache.clear();
        self.frame_cache.clear();
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

#[derive(Debug, Clone, Serialize)]
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
    pixels_u8: Vec<u8>,
    histogram: Vec<u32>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolId {
    Rect,
    Oval,
    Poly,
    Free,
    Line,
    Point,
    Wand,
    Text,
    Zoom,
    Hand,
    Dropper,
    Custom1,
    Custom2,
    Custom3,
    More,
}

impl ToolId {
    const fn command_id(self) -> &'static str {
        match self {
            Self::Rect => "launcher.tool.rect",
            Self::Oval => "launcher.tool.oval",
            Self::Poly => "launcher.tool.poly",
            Self::Free => "launcher.tool.free",
            Self::Line => "launcher.tool.line",
            Self::Point => "launcher.tool.point",
            Self::Wand => "launcher.tool.wand",
            Self::Text => "launcher.tool.text",
            Self::Zoom => "launcher.tool.zoom",
            Self::Hand => "launcher.tool.hand",
            Self::Dropper => "launcher.tool.dropper",
            Self::Custom1 => "launcher.tool.custom1",
            Self::Custom2 => "launcher.tool.custom2",
            Self::Custom3 => "launcher.tool.custom3",
            Self::More => "launcher.tool.more",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Rect => "Rect",
            Self::Oval => "Oval",
            Self::Poly => "Poly",
            Self::Free => "Free",
            Self::Line => "Line",
            Self::Point => "Point",
            Self::Wand => "Wand",
            Self::Text => "Text",
            Self::Zoom => "Zoom",
            Self::Hand => "Hand",
            Self::Dropper => "Dropper",
            Self::Custom1 => "Custom 1",
            Self::Custom2 => "Custom 2",
            Self::Custom3 => "Custom 3",
            Self::More => "More",
        }
    }

    const fn has_behavior(self) -> bool {
        matches!(self, Self::Hand | Self::Zoom | Self::Point)
    }
}

#[derive(Debug, Clone, Copy)]
struct ToolState {
    selected: ToolId,
}

impl Default for ToolState {
    fn default() -> Self {
        Self {
            selected: ToolId::Rect,
        }
    }
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

struct ViewerUiState {
    viewport_id: egui::ViewportId,
    title: String,
    z: usize,
    t: usize,
    channel: usize,
    zoom: f32,
    pan: egui::Vec2,
    show_inspector: bool,
    status_message: String,
    hover: Option<HoverInfo>,
    pinned: Option<HoverInfo>,
    frame: Option<ViewerFrameBuffer>,
    texture: Option<egui::TextureHandle>,
    last_request: Option<ViewerFrameRequest>,
    last_generation: u64,
    fit_requested: bool,
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
            show_inspector: false,
            status_message: "Ready.".to_string(),
            hover: None,
            pinned: None,
            frame: None,
            texture: None,
            last_request: None,
            last_generation: 0,
            fit_requested: true,
        }
    }

    fn telemetry(&self, active_job: Option<String>) -> ViewerTelemetry {
        ViewerTelemetry {
            hover: self.hover,
            pinned: self.pinned,
            zoom: self.zoom,
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

        format!(
            "Tool:{}  {}  Z:{} T:{} C:{}  Zoom:{:.0}%  {}",
            tool.label(),
            sample_text,
            self.z,
            self.t,
            self.channel,
            self.zoom * 100.0,
            self.status_message
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

struct ImageUiApp {
    state: UiState,
    menus: Vec<MenuManifestTopLevel>,
    command_catalog: command_registry::CommandCatalog,
    launcher_ui: LauncherUiState,
    tool_state: ToolState,
    toolbar_icons: HashMap<ToolbarIcon, egui::TextureHandle>,
    viewers_ui: HashMap<String, ViewerUiState>,
    viewer_telemetry: HashMap<String, ViewerTelemetry>,
    active_viewer_label: Option<String>,
    worker_tx: Sender<WorkerEvent>,
    worker_rx: Receiver<WorkerEvent>,
    focus_launcher: bool,
    focus_viewer_label: Option<String>,
    should_quit: bool,
}

impl ImageUiApp {
    fn new(cc: &eframe::CreationContext<'_>, startup_input: Option<PathBuf>) -> Self {
        let menus: Vec<MenuManifestTopLevel> =
            serde_json::from_str(include_str!("menu/imagej-menu-manifest.json"))
                .expect("failed to parse menu manifest");
        let command_catalog = command_registry::command_catalog();
        let (worker_tx, worker_rx) = mpsc::channel();

        let mut app = Self {
            state: UiState::new(startup_input),
            menus,
            command_catalog,
            launcher_ui: LauncherUiState {
                status: LauncherStatusModel::default(),
                fallback_text: "Ready. Use File > Open or drop image files onto this window."
                    .to_string(),
            },
            tool_state: ToolState::default(),
            toolbar_icons: load_toolbar_icons(&cc.egui_ctx),
            viewers_ui: HashMap::new(),
            viewer_telemetry: HashMap::new(),
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

    fn poll_worker_events(&mut self) {
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
        self.refresh_launcher_status();
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
        if let Some(label) = self.state.path_to_label.get(&normalized_path).cloned() {
            if self.state.label_to_session.contains_key(&label) {
                self.focus_viewer_label = Some(label.clone());
                self.set_active_viewer(Some(label.clone()));
                return Ok(OpenOutcome::Focused { label });
            }
            self.remove_stale_mapping(&normalized_path, &label);
        }

        let dataset = self
            .state
            .app
            .io_service()
            .read(&normalized_path)
            .map_err(|error| error.to_string())?;
        let session = ViewerSession::new(normalized_path.clone(), Arc::new(dataset));

        let id = self.state.next_window_id();
        let label = format!("{VIEWER_PREFIX}{id}");
        let title = format!(
            "{} - image-rs",
            normalized_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Image")
        );

        self.state
            .path_to_label
            .insert(normalized_path.clone(), label.clone());
        self.state
            .label_to_path
            .insert(label.clone(), normalized_path.clone());
        self.state.label_to_session.insert(label.clone(), session);
        self.viewers_ui
            .insert(label.clone(), ViewerUiState::new(&label, title));
        self.focus_viewer_label = Some(label.clone());
        self.set_active_viewer(Some(label.clone()));

        Ok(OpenOutcome::Opened { label })
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
        if let Some(result) = self.handle_local_command(window_label, command_id) {
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
    ) -> Option<command_registry::CommandExecuteResult> {
        if command_id == "file.open" {
            let picked_paths = FileDialog::new()
                .add_filter("TIFF images", &["tif", "tiff"])
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

        let viewer = self.viewers_ui.get_mut(window_label)?;

        match command_id {
            "image.zoom.in" => {
                viewer.zoom = (viewer.zoom * 1.2).clamp(0.1, 64.0);
                Some(command_registry::CommandExecuteResult::ok("zoomed in"))
            }
            "image.zoom.out" => {
                viewer.zoom = (viewer.zoom / 1.2).clamp(0.1, 64.0);
                Some(command_registry::CommandExecuteResult::ok("zoomed out"))
            }
            "image.zoom.reset" => {
                viewer.fit_requested = true;
                Some(command_registry::CommandExecuteResult::ok("zoom reset"))
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
            "analyze.measure" => command_registry::CommandExecuteResult::unimplemented(
                "analyze.measure is not implemented in this phase",
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
            (session.committed_dataset.clone(), generation)
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
                    let children = item.items.clone().unwrap_or_default();
                    ui.menu_button(label, |ui| {
                        self.draw_menu_items(ui, window_label, &children, actions);
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

    fn draw_launcher_toolbar(&self, ui: &mut egui::Ui, actions: &mut Vec<UiAction>) {
        for item in launcher_toolbar_items() {
            if item.kind == ToolbarKind::Separator {
                ui.separator();
                continue;
            }

            let enabled = self.command_is_enabled(LAUNCHER_LABEL, item.command_id, true);
            let icon_texture = self
                .toolbar_icons
                .get(&item.icon)
                .expect("toolbar icon should be preloaded");
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
        }
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
                    command_id: "image.zoom.reset".to_string(),
                    params: None,
                }),
                "M" | "m" => actions.push(UiAction::Command {
                    window_label: window_label.to_string(),
                    command_id: "analyze.measure".to_string(),
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

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(2.0);
            ui.horizontal_wrapped(|ui| {
                ui.label("Drop TIFF images here or use File > Open...");
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

            if viewer.show_inspector {
                egui::SidePanel::right(format!("viewer-inspector-{label}"))
                    .resizable(true)
                    .default_width(300.0)
                    .show(ctx, |ui| {
                        ui.heading("Inspector");
                        if let Some(path) = self.state.label_to_path.get(label) {
                            ui.label(path.display().to_string());
                        }
                        ui.separator();

                        if let Some(summary) = &summary {
                            let max_z = summary.z_slices.saturating_sub(1);
                            let max_t = summary.times.saturating_sub(1);
                            let max_c = summary.channels.saturating_sub(1);

                            ui.label("Slice");
                            ui.add(
                                egui::Slider::new(&mut viewer.z, 0..=max_z)
                                    .text("Z")
                                    .clamping(egui::SliderClamping::Always),
                            );
                            ui.add(
                                egui::Slider::new(&mut viewer.t, 0..=max_t)
                                    .text("T")
                                    .clamping(egui::SliderClamping::Always),
                            );
                            ui.add(
                                egui::Slider::new(&mut viewer.channel, 0..=max_c)
                                    .text("C")
                                    .clamping(egui::SliderClamping::Always),
                            );

                            ui.separator();
                            ui.label("Metadata");
                            ui.monospace(
                                serde_json::to_string_pretty(summary)
                                    .unwrap_or_else(|_| "{}".to_string()),
                            );
                        }

                        ui.separator();
                        ui.label("Histogram");
                        let size = egui::vec2(ui.available_width().max(1.0), 140.0);
                        let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                        if let Some(frame) = &viewer.frame {
                            draw_histogram(ui.painter(), rect, &frame.histogram);
                        } else {
                            ui.painter().rect_stroke(
                                rect,
                                0.0,
                                egui::Stroke::new(1.0, egui::Color32::GRAY),
                                egui::StrokeKind::Outside,
                            );
                        }
                    });
            }

            egui::CentralPanel::default().show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .button(if viewer.show_inspector {
                            "Hide Panel"
                        } else {
                            "Show Panel"
                        })
                        .clicked()
                    {
                        viewer.show_inspector = !viewer.show_inspector;
                    }
                });
                ui.separator();

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
                        fit_to_rect(viewer, rect, frame.width, frame.height);
                        viewer.fit_requested = false;
                    }

                    let hover_sample = response
                        .hover_pos()
                        .and_then(|pointer| sample_at_pointer(viewer, &frame, rect, pointer));
                    viewer.hover = hover_sample;
                    hovered_viewer = response.hovered() || response.dragged() || response.clicked();

                    if response.hovered() && selected_tool == ToolId::Zoom {
                        let scroll = ui.input(|i| i.smooth_scroll_delta.y + i.raw_scroll_delta.y);
                        if scroll.abs() > f32::EPSILON
                            && let Some(pointer) = response.hover_pos()
                        {
                            let zoom_factor = if scroll > 0.0 { 1.12 } else { 1.0 / 1.12 };
                            zoom_about_pointer(viewer, rect, pointer, zoom_factor);
                        }
                    }

                    match selected_tool {
                        ToolId::Hand => {
                            if response.dragged() {
                                viewer.pan += ui.input(|i| i.pointer.delta());
                            }
                        }
                        ToolId::Zoom => {
                            if let Some(pointer) = response.hover_pos() {
                                if response.clicked_by(egui::PointerButton::Primary) {
                                    let zoom_factor = if ui.input(|i| i.modifiers.shift) {
                                        1.0 / 1.2
                                    } else {
                                        1.2
                                    };
                                    zoom_about_pointer(viewer, rect, pointer, zoom_factor);
                                }
                                if response.clicked_by(egui::PointerButton::Secondary) {
                                    zoom_about_pointer(viewer, rect, pointer, 1.0 / 1.2);
                                }
                            }
                        }
                        ToolId::Point => {
                            if response.clicked_by(egui::PointerButton::Primary) {
                                viewer.pinned = hover_sample;
                            }
                        }
                        _ => {}
                    }

                    let image_size =
                        egui::vec2(frame.width as f32, frame.height as f32) * viewer.zoom;
                    let image_rect = egui::Rect::from_min_size(rect.min + viewer.pan, image_size);
                    ui.painter().image(
                        texture_id,
                        image_rect,
                        egui::Rect::from_min_max(
                            egui::Pos2::new(0.0, 0.0),
                            egui::Pos2::new(1.0, 1.0),
                        ),
                        egui::Color32::WHITE,
                    );

                    if let Some(pinned) = viewer.pinned {
                        draw_point_marker(ui.painter(), viewer, rect, pinned);
                    }
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
                        viewer.status_message = result.message;
                        if command_id.starts_with("image.zoom") {
                            viewer.fit_requested = command_id == "image.zoom.reset";
                        }
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
        self.poll_worker_events();

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
            let title = self
                .viewers_ui
                .get(&label)
                .map(|viewer| viewer.title.clone())
                .unwrap_or_else(|| "image-rs Viewer".to_string());

            ctx.show_viewport_immediate(
                viewport_id,
                egui::ViewportBuilder::default()
                    .with_title(title)
                    .with_inner_size(VIEWER_DEFAULT_SIZE)
                    .with_min_inner_size([640.0, 420.0]),
                |ctx, _class| {
                    self.draw_viewer_viewport(ctx, &label, &mut actions);
                },
            );
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

        self.apply_actions(actions);
        self.maybe_focus_windows(ctx);
        self.refresh_launcher_status();

        if self.should_quit {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            self.should_quit = false;
        }

        ctx.request_repaint();
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ToolbarIcon {
    Rect,
    Oval,
    Poly,
    Free,
    Line,
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
        launcher_toolbar_button("launcher.tool.point", "Point", ToolbarIcon::Point),
        launcher_toolbar_button("launcher.tool.wand", "Wand", ToolbarIcon::Wand),
        launcher_toolbar_button("launcher.tool.text", "Text", ToolbarIcon::Text),
        launcher_toolbar_separator(),
        launcher_toolbar_button("file.open", "Open", ToolbarIcon::Open),
        launcher_toolbar_button("launcher.tool.zoom", "Zoom", ToolbarIcon::Zoom),
        launcher_toolbar_button("launcher.tool.hand", "Hand", ToolbarIcon::Hand),
        launcher_toolbar_button("launcher.tool.dropper", "Dropper", ToolbarIcon::Dropper),
        launcher_toolbar_separator(),
        launcher_toolbar_button("window.previous", "Previous Window", ToolbarIcon::Previous),
        launcher_toolbar_button("window.next", "Next Window", ToolbarIcon::Next),
        launcher_toolbar_button("file.quit", "Quit", ToolbarIcon::Quit),
        launcher_toolbar_button("launcher.tool.custom1", "Custom 1", ToolbarIcon::Custom1),
        launcher_toolbar_button("launcher.tool.custom2", "Custom 2", ToolbarIcon::Custom2),
        launcher_toolbar_button("launcher.tool.custom3", "Custom 3", ToolbarIcon::Custom3),
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
        viewer_toolbar_button("image.zoom.reset", "Zoom 100%", "1:1"),
        viewer_toolbar_separator(),
        viewer_toolbar_button("process.smooth", "Smooth", "S"),
        viewer_toolbar_button("process.gaussian", "Gaussian Blur", "G"),
        viewer_toolbar_separator(),
        viewer_toolbar_button("analyze.measure", "Measure", "M"),
    ];
    ITEMS
}

fn load_toolbar_icons(ctx: &egui::Context) -> HashMap<ToolbarIcon, egui::TextureHandle> {
    let specs = [
        (
            ToolbarIcon::Rect,
            "rect",
            include_bytes!("../assets/tools/rect.png").as_slice(),
        ),
        (
            ToolbarIcon::Oval,
            "oval",
            include_bytes!("../assets/tools/oval.png").as_slice(),
        ),
        (
            ToolbarIcon::Poly,
            "poly",
            include_bytes!("../assets/tools/poly.png").as_slice(),
        ),
        (
            ToolbarIcon::Free,
            "free",
            include_bytes!("../assets/tools/free.png").as_slice(),
        ),
        (
            ToolbarIcon::Line,
            "line",
            include_bytes!("../assets/tools/line.png").as_slice(),
        ),
        (
            ToolbarIcon::Point,
            "point",
            include_bytes!("../assets/tools/point.png").as_slice(),
        ),
        (
            ToolbarIcon::Wand,
            "wand",
            include_bytes!("../assets/tools/wand.png").as_slice(),
        ),
        (
            ToolbarIcon::Text,
            "text",
            include_bytes!("../assets/tools/text.png").as_slice(),
        ),
        (
            ToolbarIcon::Open,
            "open",
            include_bytes!("../assets/tools/open.png").as_slice(),
        ),
        (
            ToolbarIcon::Zoom,
            "zoom",
            include_bytes!("../assets/tools/zoom.png").as_slice(),
        ),
        (
            ToolbarIcon::Hand,
            "hand",
            include_bytes!("../assets/tools/hand.png").as_slice(),
        ),
        (
            ToolbarIcon::Dropper,
            "dropper",
            include_bytes!("../assets/tools/dropper.png").as_slice(),
        ),
        (
            ToolbarIcon::Previous,
            "previous",
            include_bytes!("../assets/tools/previous.png").as_slice(),
        ),
        (
            ToolbarIcon::Next,
            "next",
            include_bytes!("../assets/tools/next.png").as_slice(),
        ),
        (
            ToolbarIcon::Quit,
            "quit",
            include_bytes!("../assets/tools/quit.png").as_slice(),
        ),
        (
            ToolbarIcon::Custom1,
            "custom1",
            include_bytes!("../assets/tools/custom1.png").as_slice(),
        ),
        (
            ToolbarIcon::Custom2,
            "custom2",
            include_bytes!("../assets/tools/custom2.png").as_slice(),
        ),
        (
            ToolbarIcon::Custom3,
            "custom3",
            include_bytes!("../assets/tools/custom3.png").as_slice(),
        ),
        (
            ToolbarIcon::More,
            "more",
            include_bytes!("../assets/tools/more.png").as_slice(),
        ),
    ];

    let mut icons = HashMap::with_capacity(specs.len());
    for (icon, name, bytes) in specs {
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

        let texture = ctx.load_texture(
            format!("toolbar-icon-{name}"),
            decoded,
            egui::TextureOptions::LINEAR,
        );
        icons.insert(icon, texture);
    }

    icons
}

fn tool_from_command_id(command_id: &str) -> Option<ToolId> {
    match command_id {
        "launcher.tool.rect" => Some(ToolId::Rect),
        "launcher.tool.oval" => Some(ToolId::Oval),
        "launcher.tool.poly" => Some(ToolId::Poly),
        "launcher.tool.free" => Some(ToolId::Free),
        "launcher.tool.line" => Some(ToolId::Line),
        "launcher.tool.point" => Some(ToolId::Point),
        "launcher.tool.wand" => Some(ToolId::Wand),
        "launcher.tool.text" => Some(ToolId::Text),
        "launcher.tool.zoom" => Some(ToolId::Zoom),
        "launcher.tool.hand" => Some(ToolId::Hand),
        "launcher.tool.dropper" => Some(ToolId::Dropper),
        "launcher.tool.custom1" => Some(ToolId::Custom1),
        "launcher.tool.custom2" => Some(ToolId::Custom2),
        "launcher.tool.custom3" => Some(ToolId::Custom3),
        "launcher.tool.more" => Some(ToolId::More),
        _ => None,
    }
}

fn tool_shortcut_command(text: &str) -> Option<&'static str> {
    tool_shortcut_tool(text).map(ToolId::command_id)
}

fn tool_shortcut_tool(text: &str) -> Option<ToolId> {
    match text {
        "r" | "R" => Some(ToolId::Rect),
        "o" | "O" => Some(ToolId::Oval),
        "g" | "G" => Some(ToolId::Poly),
        "f" | "F" => Some(ToolId::Free),
        "l" | "L" => Some(ToolId::Line),
        "p" | "P" | "." => Some(ToolId::Point),
        "w" | "W" => Some(ToolId::Wand),
        "t" | "T" => Some(ToolId::Text),
        "z" | "Z" => Some(ToolId::Zoom),
        "h" | "H" => Some(ToolId::Hand),
        "d" | "D" => Some(ToolId::Dropper),
        "1" => Some(ToolId::Custom1),
        "2" => Some(ToolId::Custom2),
        "3" => Some(ToolId::Custom3),
        _ => None,
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
    let local = pointer - (rect.min + viewer.pan);
    let image_x = (local.x / viewer.zoom).floor() as isize;
    let image_y = (local.y / viewer.zoom).floor() as isize;
    if image_x < 0 || image_y < 0 {
        return None;
    }

    let x = image_x as usize;
    let y = image_y as usize;
    if x >= frame.width || y >= frame.height {
        return None;
    }

    let index = y * frame.width + x;
    let gray = frame.pixels_u8[index] as f32;
    let value = frame.min + (gray / 255.0) * (frame.max - frame.min);
    Some(HoverInfo { x, y, value })
}

fn zoom_about_pointer(
    viewer: &mut ViewerUiState,
    rect: egui::Rect,
    pointer: egui::Pos2,
    factor: f32,
) {
    let previous_zoom = viewer.zoom;
    let next_zoom = (viewer.zoom * factor).clamp(0.1, 64.0);
    let pointer_local = pointer - rect.min;
    let source_before = (pointer_local - viewer.pan) / previous_zoom;
    viewer.zoom = next_zoom;
    viewer.pan = pointer_local - source_before * next_zoom;
}

fn draw_point_marker(
    painter: &egui::Painter,
    viewer: &ViewerUiState,
    rect: egui::Rect,
    pinned: HoverInfo,
) {
    let center = rect.min
        + viewer.pan
        + egui::vec2(
            (pinned.x as f32 + 0.5) * viewer.zoom,
            (pinned.y as f32 + 0.5) * viewer.zoom,
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

fn compute_viewer_frame(
    state: &mut UiState,
    window_label: &str,
    request: &ViewerFrameRequest,
    preview_override: Option<&PreviewRequest>,
) -> Result<ViewerFrameBuffer, String> {
    let (source_kind, dataset) = dataset_for_window(state, window_label, preview_override)?;
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

    let frame = build_frame(dataset.as_ref(), request)?;

    if let Some(session) = state.label_to_session.get_mut(window_label) {
        let can_store = preview_override.is_some() || session.current_source_kind() == source_kind;
        if can_store {
            session.frame_cache.insert(frame_key, frame.clone());
        }
    }

    Ok(frame)
}

fn dataset_for_window(
    state: &mut UiState,
    window_label: &str,
    preview_override: Option<&PreviewRequest>,
) -> Result<(String, Arc<DatasetF32>), String> {
    if let Some(preview) = preview_override {
        let (preview_key, dataset) = ensure_preview_dataset(state, window_label, preview)?;
        return Ok((format!("{SOURCE_PREVIEW_PREFIX}{preview_key}"), dataset));
    }

    let session = state
        .label_to_session
        .get(window_label)
        .ok_or_else(|| format!("no viewer session for `{window_label}`"))?;
    let source_kind = session.current_source_kind();
    let dataset = session
        .dataset_for_source(&source_kind)
        .ok_or_else(|| format!("no dataset found for source `{source_kind}`"))?;

    Ok((source_kind, dataset))
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

        session.committed_dataset.clone()
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

    let _baseline_shape = session.base_dataset.shape();
    let summary = summarize_dataset(session.committed_dataset.as_ref(), &session.path);
    Ok((session.path.clone(), summary))
}

fn preview_cache_key(op: &str, params: &Value) -> Result<String, String> {
    let normalized = canonical_json(params);
    let encoded = serde_json::to_string(&normalized)
        .map_err(|error| format!("invalid preview params: {error}"))?;
    Ok(format!("{op}:{encoded}"))
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
    dataset: &DatasetF32,
    request: &ViewerFrameRequest,
) -> Result<ViewerFrameBuffer, String> {
    let slice = extract_slice(dataset, request.z, request.t, request.channel)?;
    let pixels_u8 = to_u8_samples(&slice.values);
    let (min, max) = min_max(&slice.values);
    let histogram = histogram(&slice.values);

    Ok(ViewerFrameBuffer {
        width: slice.width,
        height: slice.height,
        pixels_u8,
        histogram,
        min,
        max,
    })
}

fn summarize_dataset(dataset: &DatasetF32, source: &Path) -> ImageSummary {
    let (min, max) = dataset.min_max().unwrap_or((0.0, 0.0));
    let channel_axis = dataset.axis_index(AxisKind::Channel);
    let z_axis = dataset.axis_index(AxisKind::Z);
    let t_axis = dataset.axis_index(AxisKind::Time);

    ImageSummary {
        shape: dataset.shape().to_vec(),
        axes: dataset
            .metadata
            .dims
            .iter()
            .map(|dimension| format!("{:?}", dimension.axis))
            .collect(),
        channels: channel_axis
            .map(|index| dataset.shape()[index])
            .unwrap_or(1),
        z_slices: z_axis.map(|index| dataset.shape()[index]).unwrap_or(1),
        times: t_axis.map(|index| dataset.shape()[index]).unwrap_or(1),
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

fn to_u8_samples(values: &[f32]) -> Vec<u8> {
    let (min, max) = min_max(values);
    let unit_range = min >= 0.0 && max <= 1.0;
    values
        .iter()
        .map(|value| {
            let normalized = if unit_range {
                *value
            } else if (max - min).abs() < f32::EPSILON {
                0.0
            } else {
                (*value - min) / (max - min)
            };
            (normalized.clamp(0.0, 1.0) * 255.0).round() as u8
        })
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

fn histogram(values: &[f32]) -> Vec<u32> {
    let (min, max) = min_max(values);
    let span = (max - min).max(f32::EPSILON);
    let mut bins = vec![0_u32; 256];
    for value in values {
        let normalized = ((*value - min) / span).clamp(0.0, 1.0);
        let index = (normalized * 255.0).round() as usize;
        bins[index] += 1;
    }
    bins
}

fn draw_histogram(painter: &egui::Painter, rect: egui::Rect, histogram: &[u32]) {
    painter.rect_filled(rect, 0.0, egui::Color32::WHITE);
    if histogram.is_empty() {
        return;
    }

    let max = histogram.iter().copied().max().unwrap_or(1) as f32;
    let bar_width = rect.width() / histogram.len() as f32;

    for (index, value) in histogram.iter().enumerate() {
        let ratio = (*value as f32 / max).clamp(0.0, 1.0);
        let height = ratio * (rect.height() - 2.0);
        let x0 = rect.left() + index as f32 * bar_width;
        let x1 = x0 + bar_width.max(1.0);
        let y1 = rect.bottom();
        let y0 = y1 - height;

        painter.rect_filled(
            egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1)),
            0.0,
            egui::Color32::from_gray(90),
        );
    }
}

fn to_color_image(frame: &ViewerFrameBuffer) -> egui::ColorImage {
    let mut rgba = Vec::with_capacity(frame.pixels_u8.len() * 4);
    for gray in &frame.pixels_u8 {
        rgba.extend_from_slice(&[*gray, *gray, *gray, 255]);
    }
    egui::ColorImage::from_rgba_unmultiplied([frame.width, frame.height], &rgba)
}

fn fit_to_rect(viewer: &mut ViewerUiState, rect: egui::Rect, width: usize, height: usize) {
    let image_width = width as f32;
    let image_height = height as f32;
    if image_width <= 0.0 || image_height <= 0.0 {
        return;
    }

    let zoom = (rect.width() / image_width)
        .min(rect.height() / image_height)
        .clamp(0.1, 64.0);
    let pan_x = (rect.width() - image_width * zoom) * 0.5;
    let pan_y = (rect.height() - image_height * zoom) * 0.5;

    viewer.zoom = zoom;
    viewer.pan = egui::vec2(pan_x, pan_y);
}

fn ui_close_requested(ctx: &egui::Context) -> bool {
    ctx.input(|i| i.viewport().close_requested())
}

fn is_supported_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| matches!(extension.to_ascii_lowercase().as_str(), "tif" | "tiff"))
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

fn main() -> eframe::Result<()> {
    let startup_arg = std::env::args().nth(1).map(PathBuf::from);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("image-rs Control")
            .with_inner_size(LAUNCHER_WINDOW_SIZE)
            .with_min_inner_size(LAUNCHER_WINDOW_SIZE)
            .with_max_inner_size(LAUNCHER_WINDOW_SIZE)
            .with_resizable(false),
        ..Default::default()
    };

    eframe::run_native(
        "image-rs Control",
        options,
        Box::new(move |cc| Ok(Box::new(ImageUiApp::new(cc, startup_arg.clone())))),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use serde_json::json;

    use super::{
        HoverInfo, LauncherStatusModel, ProgressState, ToolId, ViewerTelemetry, canonical_json,
        format_launcher_status, preview_cache_key, tool_from_command_id, tool_shortcut_command,
        viewer_sort_key,
    };

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
        assert_eq!(tool_from_command_id("launcher.tool.unknown"), None);
    }

    #[test]
    fn tool_shortcut_mapping_is_deterministic() {
        assert_eq!(tool_shortcut_command("r"), Some("launcher.tool.rect"));
        assert_eq!(tool_shortcut_command("R"), Some("launcher.tool.rect"));
        assert_eq!(tool_shortcut_command("h"), Some("launcher.tool.hand"));
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
}
