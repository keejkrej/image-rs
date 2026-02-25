#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    collections::HashMap,
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use ijrs_app::AppContext;
use ijrs_core::{AxisKind, DatasetF32};
use image::{DynamicImage, ImageBuffer, ImageFormat, Luma};
use ndarray::IxDyn;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tauri::{
    AppHandle, DragDropEvent, Emitter, Manager, State, WebviewUrl, WebviewWindow,
    WebviewWindowBuilder, Window, WindowEvent,
};

const LAUNCHER_LABEL: &str = "main";
const VIEWER_PREFIX: &str = "viewer-";
const LAUNCHER_RESULT_EVENT: &str = "launcher-open-result";
const VIEWER_OP_EVENT: &str = "viewer-op-event";
const SOURCE_COMMITTED: &str = "committed";
const SOURCE_PREVIEW_PREFIX: &str = "preview:";

#[derive(Debug)]
struct UiState {
    app: AppContext,
    startup_input: Option<PathBuf>,
    path_to_label: Mutex<HashMap<PathBuf, String>>,
    label_to_path: Mutex<HashMap<String, PathBuf>>,
    label_to_session: Mutex<HashMap<String, ViewerSession>>,
    next_window_id: AtomicU64,
    next_job_id: AtomicU64,
}

impl UiState {
    fn new(startup_input: Option<PathBuf>) -> Self {
        Self {
            app: AppContext::new(),
            startup_input,
            path_to_label: Mutex::new(HashMap::new()),
            label_to_path: Mutex::new(HashMap::new()),
            label_to_session: Mutex::new(HashMap::new()),
            next_window_id: AtomicU64::new(0),
            next_job_id: AtomicU64::new(0),
        }
    }
}

#[derive(Debug, Clone)]
struct ViewerSession {
    path: PathBuf,
    base_dataset: Arc<DatasetF32>,
    committed_dataset: Arc<DatasetF32>,
    active_preview: Option<String>,
    preview_cache: HashMap<String, Arc<DatasetF32>>,
    frame_cache: HashMap<FrameKey, ViewerFrame>,
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

#[derive(Debug, Clone, Deserialize, Serialize)]
struct PreviewRequest {
    op: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct ViewerFrameRequest {
    #[serde(default)]
    z: usize,
    #[serde(default)]
    t: usize,
    #[serde(default)]
    channel: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ViewerSliceRequest {
    #[serde(default)]
    z: usize,
    #[serde(default)]
    t: usize,
    #[serde(default)]
    channel: usize,
    preview: Option<PreviewRequest>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ViewerExportRequest {
    output_path: String,
    preview: Option<PreviewRequest>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ViewerOpRequest {
    op: String,
    #[serde(default)]
    params: Value,
    mode: OpRunMode,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ViewerCancelRequest {
    job_id: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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
struct ViewerOpEvent {
    job_id: u64,
    mode: String,
    op: String,
    status: String,
    message: Option<String>,
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

#[derive(Debug, Clone, Serialize)]
struct ViewerContext {
    window_label: String,
    path: String,
    summary: ImageSummary,
}

#[derive(Debug, Clone, Serialize)]
struct ViewerInit {
    window_label: String,
    path: String,
    summary: ImageSummary,
    default_frame: ViewerFrame,
}

#[derive(Debug, Clone)]
struct SliceImage {
    width: usize,
    height: usize,
    values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
struct SlicePlane {
    width: usize,
    height: usize,
    values: Vec<f32>,
    min: f32,
    max: f32,
}

#[derive(Debug, Clone, Serialize)]
struct ViewerFrame {
    width: usize,
    height: usize,
    png_data_url: String,
    pixels_u8: Vec<u8>,
    histogram: Vec<u32>,
    min: f32,
    max: f32,
    values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
struct ViewerFrameBuffer {
    width: usize,
    height: usize,
    pixels_u8: Vec<u8>,
    histogram: Vec<u32>,
    min: f32,
    max: f32,
}

#[tauri::command]
fn open_images(state: State<'_, UiState>, app: AppHandle, paths: Vec<String>) -> OpenResult {
    let files = paths.into_iter().map(PathBuf::from).collect::<Vec<_>>();
    open_paths(&app, state.inner(), files)
}

#[tauri::command]
fn viewer_context(state: State<'_, UiState>, window: Window) -> Result<ViewerContext, String> {
    let (path, summary) = summary_for_window(state.inner(), window.label())?;
    Ok(ViewerContext {
        window_label: window.label().to_string(),
        path: path.display().to_string(),
        summary,
    })
}

#[tauri::command]
async fn viewer_init(state: State<'_, UiState>, window: Window) -> Result<ViewerInit, String> {
    let (path, summary) = summary_for_window(state.inner(), window.label())?;
    let default_frame = compute_viewer_frame(
        state.inner(),
        window.label(),
        &ViewerFrameRequest::default(),
        None,
    )
    .await?;

    Ok(ViewerInit {
        window_label: window.label().to_string(),
        path: path.display().to_string(),
        summary,
        default_frame,
    })
}

#[tauri::command]
async fn viewer_frame(
    state: State<'_, UiState>,
    window: Window,
    request: ViewerFrameRequest,
) -> Result<ViewerFrame, String> {
    compute_viewer_frame(state.inner(), window.label(), &request, None).await
}

#[tauri::command]
async fn viewer_frame_buffer(
    state: State<'_, UiState>,
    window: Window,
    request: ViewerFrameRequest,
) -> Result<ViewerFrameBuffer, String> {
    let frame = compute_viewer_frame(state.inner(), window.label(), &request, None).await?;
    Ok(ViewerFrameBuffer {
        width: frame.width,
        height: frame.height,
        pixels_u8: frame.pixels_u8,
        histogram: frame.histogram,
        min: frame.min,
        max: frame.max,
    })
}

#[tauri::command]
fn cycle_window(
    state: State<'_, UiState>,
    app: AppHandle,
    window: Window,
    direction: i32,
) -> String {
    let mut labels = if let Ok(map) = state.label_to_path.lock() {
        map.keys().cloned().collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    labels.sort_by_key(|label| viewer_sort_key(label));
    if labels.is_empty() {
        if let Some(launcher) = app.get_webview_window(LAUNCHER_LABEL) {
            focus_window(&launcher);
            return LAUNCHER_LABEL.to_string();
        }
        return window.label().to_string();
    }

    let current = window.label().to_string();
    let offset = if direction < 0 { -1 } else { 1 };
    let target_index = labels
        .iter()
        .position(|label| label == &current)
        .map(|index| {
            let len = labels.len() as isize;
            let raw = index as isize + offset;
            raw.rem_euclid(len) as usize
        })
        .unwrap_or_else(|| if direction < 0 { labels.len() - 1 } else { 0 });
    let target = labels[target_index].clone();

    if let Some(next_window) = app.get_webview_window(&target) {
        focus_window(&next_window);
    }

    target
}

#[tauri::command]
async fn viewer_start_op(
    state: State<'_, UiState>,
    app: AppHandle,
    window: Window,
    request: ViewerOpRequest,
) -> Result<JobTicket, String> {
    let window_label = window.label().to_string();
    let job_id = state.next_job_id.fetch_add(1, Ordering::Relaxed) + 1;
    let preview_key = match request.mode {
        OpRunMode::Preview => Some(preview_cache_key(&request.op, &request.params)?),
        OpRunMode::Apply => None,
    };

    if let Some(key) = &preview_key {
        let mut sessions = state
            .label_to_session
            .lock()
            .map_err(|_| "session map lock poisoned".to_string())?;
        let session = sessions
            .get_mut(&window_label)
            .ok_or_else(|| format!("no viewer session for `{}`", window.label()))?;

        if session.preview_cache.contains_key(key) {
            session.generation = session.generation.saturating_add(1);
            session.active_job = None;
            session.set_active_preview(Some(key.clone()));

            emit_viewer_op_event(
                &app,
                &window_label,
                ViewerOpEvent {
                    job_id,
                    mode: request.mode.as_str().to_string(),
                    op: request.op.clone(),
                    status: "completed".to_string(),
                    message: None,
                },
            );

            return Ok(JobTicket { job_id });
        }
    }

    let (input_dataset, generation) = {
        let mut sessions = state
            .label_to_session
            .lock()
            .map_err(|_| "session map lock poisoned".to_string())?;
        let session = sessions
            .get_mut(&window_label)
            .ok_or_else(|| format!("no viewer session for `{}`", window.label()))?;

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

    let app_handle = app.clone();
    let mode = request.mode.clone();
    let op_name = request.op.clone();
    let params = request.params.clone();
    let op_for_event = request.op.clone();
    let preview_key_for_task = preview_key.clone();
    let ops_service = state.app.ops_service().clone();

    tauri::async_runtime::spawn(async move {
        let run_result = tauri::async_runtime::spawn_blocking(move || {
            ops_service
                .execute(&op_name, input_dataset.as_ref(), &params)
                .map(|output| Arc::new(output.dataset))
                .map_err(|error| error.to_string())
        })
        .await;

        match run_result {
            Ok(Ok(result_dataset)) => {
                let state = app_handle.state::<UiState>();
                let mut sessions = match state.label_to_session.lock() {
                    Ok(sessions) => sessions,
                    Err(_) => {
                        emit_viewer_op_event(
                            &app_handle,
                            &window_label,
                            ViewerOpEvent {
                                job_id,
                                mode: mode.as_str().to_string(),
                                op: op_for_event.clone(),
                                status: "failed".to_string(),
                                message: Some("session map lock poisoned".to_string()),
                            },
                        );
                        return;
                    }
                };

                let Some(session) = sessions.get_mut(&window_label) else {
                    return;
                };

                if !session.is_active_job(job_id, generation) {
                    return;
                }

                match &mode {
                    OpRunMode::Preview => {
                        if let Some(key) = preview_key_for_task {
                            session.preview_cache.insert(key.clone(), result_dataset);
                            session.set_active_preview(Some(key));
                        }
                    }
                    OpRunMode::Apply => {
                        session.commit_dataset(result_dataset);
                    }
                }
                session.active_job = None;

                emit_viewer_op_event(
                    &app_handle,
                    &window_label,
                    ViewerOpEvent {
                        job_id,
                        mode: mode.as_str().to_string(),
                        op: op_for_event.clone(),
                        status: "completed".to_string(),
                        message: None,
                    },
                );
            }
            Ok(Err(error)) => {
                finish_failed_job(
                    &app_handle,
                    &window_label,
                    job_id,
                    generation,
                    &request,
                    error,
                );
            }
            Err(error) => {
                finish_failed_job(
                    &app_handle,
                    &window_label,
                    job_id,
                    generation,
                    &request,
                    format!("failed to join background task: {error}"),
                );
            }
        }
    });

    Ok(JobTicket { job_id })
}

#[tauri::command]
fn viewer_cancel_op(
    state: State<'_, UiState>,
    window: Window,
    request: ViewerCancelRequest,
) -> Result<(), String> {
    let mut cancelled = None;

    {
        let mut sessions = state
            .label_to_session
            .lock()
            .map_err(|_| "session map lock poisoned".to_string())?;
        let session = sessions
            .get_mut(window.label())
            .ok_or_else(|| format!("no viewer session for `{}`", window.label()))?;

        if let Some(active) = session.active_job.as_ref()
            && active.job_id == request.job_id
        {
            cancelled = Some(ViewerOpEvent {
                job_id: active.job_id,
                mode: active.mode.as_str().to_string(),
                op: active.op.clone(),
                status: "cancelled".to_string(),
                message: Some("Operation cancelled.".to_string()),
            });
            session.generation = session.generation.saturating_add(1);
            session.active_job = None;
        }
    }

    if let Some(payload) = cancelled {
        let _ = window.emit(VIEWER_OP_EVENT, &payload);
    }

    Ok(())
}

#[tauri::command]
fn list_preview_ops(state: State<'_, UiState>) -> Vec<String> {
    state
        .app
        .ops_service()
        .list()
        .into_iter()
        .map(|schema| schema.name)
        .filter(|name| {
            matches!(
                name.as_str(),
                "intensity.normalize"
                    | "intensity.window"
                    | "gaussian.blur"
                    | "threshold.fixed"
                    | "threshold.otsu"
                    | "morphology.erode"
                    | "morphology.dilate"
                    | "morphology.open"
                    | "morphology.close"
            )
        })
        .collect()
}

#[tauri::command]
fn inspect_image(state: State<'_, UiState>, path: String) -> Result<ImageSummary, String> {
    let path = PathBuf::from(path);
    let dataset = state
        .app
        .io_service()
        .read(&path)
        .map_err(|error| error.to_string())?;
    Ok(summarize_dataset(&dataset, &path))
}

#[tauri::command]
async fn render_slice(
    state: State<'_, UiState>,
    window: Window,
    request: ViewerSliceRequest,
) -> Result<String, String> {
    let frame_request = ViewerFrameRequest {
        z: request.z,
        t: request.t,
        channel: request.channel,
    };
    let frame = compute_viewer_frame(
        state.inner(),
        window.label(),
        &frame_request,
        request.preview.as_ref(),
    )
    .await?;
    Ok(frame.png_data_url)
}

#[tauri::command]
async fn slice_histogram(
    state: State<'_, UiState>,
    window: Window,
    request: ViewerSliceRequest,
) -> Result<Vec<u32>, String> {
    let frame_request = ViewerFrameRequest {
        z: request.z,
        t: request.t,
        channel: request.channel,
    };
    let frame = compute_viewer_frame(
        state.inner(),
        window.label(),
        &frame_request,
        request.preview.as_ref(),
    )
    .await?;
    Ok(frame.histogram)
}

#[tauri::command]
async fn slice_plane(
    state: State<'_, UiState>,
    window: Window,
    request: ViewerSliceRequest,
) -> Result<SlicePlane, String> {
    let frame_request = ViewerFrameRequest {
        z: request.z,
        t: request.t,
        channel: request.channel,
    };
    let frame = compute_viewer_frame(
        state.inner(),
        window.label(),
        &frame_request,
        request.preview.as_ref(),
    )
    .await?;
    Ok(SlicePlane {
        width: frame.width,
        height: frame.height,
        values: frame.values,
        min: frame.min,
        max: frame.max,
    })
}

#[tauri::command]
async fn export_preview(
    state: State<'_, UiState>,
    window: Window,
    request: ViewerExportRequest,
) -> Result<(), String> {
    let (_, dataset) =
        dataset_for_window(state.inner(), window.label(), request.preview.as_ref()).await?;
    let app_context = state.app.clone();
    let output_path = request.output_path;

    tauri::async_runtime::spawn_blocking(move || {
        app_context
            .io_service()
            .write(output_path, dataset.as_ref())
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| format!("failed to join export task: {error}"))??;

    Ok(())
}

fn finish_failed_job(
    app: &AppHandle,
    window_label: &str,
    job_id: u64,
    generation: u64,
    request: &ViewerOpRequest,
    message: String,
) {
    if let Ok(mut sessions) = app.state::<UiState>().label_to_session.lock()
        && let Some(session) = sessions.get_mut(window_label)
        && session.is_active_job(job_id, generation)
    {
        session.active_job = None;
        emit_viewer_op_event(
            app,
            window_label,
            ViewerOpEvent {
                job_id,
                mode: request.mode.as_str().to_string(),
                op: request.op.clone(),
                status: "failed".to_string(),
                message: Some(message),
            },
        );
    }
}

fn emit_viewer_op_event(app: &AppHandle, window_label: &str, payload: ViewerOpEvent) {
    if let Some(window) = app.get_webview_window(window_label) {
        let _ = window.emit(VIEWER_OP_EVENT, &payload);
    }
}

async fn compute_viewer_frame(
    state: &UiState,
    window_label: &str,
    request: &ViewerFrameRequest,
    preview_override: Option<&PreviewRequest>,
) -> Result<ViewerFrame, String> {
    let (source_kind, dataset) = dataset_for_window(state, window_label, preview_override).await?;
    let frame_key = FrameKey {
        source_kind: source_kind.clone(),
        z: request.z,
        t: request.t,
        channel: request.channel,
    };

    if let Some(frame) = {
        let sessions = state
            .label_to_session
            .lock()
            .map_err(|_| "session map lock poisoned".to_string())?;
        sessions
            .get(window_label)
            .and_then(|session| session.frame_cache.get(&frame_key).cloned())
    } {
        return Ok(frame);
    }

    let request = request.clone();
    let dataset_clone = dataset.clone();
    let frame =
        tauri::async_runtime::spawn_blocking(move || build_frame(dataset_clone.as_ref(), &request))
            .await
            .map_err(|error| format!("failed to join frame task: {error}"))??;

    {
        let mut sessions = state
            .label_to_session
            .lock()
            .map_err(|_| "session map lock poisoned".to_string())?;
        if let Some(session) = sessions.get_mut(window_label) {
            let can_store =
                preview_override.is_some() || session.current_source_kind() == source_kind;
            if can_store {
                session.frame_cache.insert(frame_key, frame.clone());
            }
        }
    }

    Ok(frame)
}

async fn dataset_for_window(
    state: &UiState,
    window_label: &str,
    preview_override: Option<&PreviewRequest>,
) -> Result<(String, Arc<DatasetF32>), String> {
    if let Some(preview) = preview_override {
        let (preview_key, dataset) = ensure_preview_dataset(state, window_label, preview).await?;
        return Ok((format!("{SOURCE_PREVIEW_PREFIX}{preview_key}"), dataset));
    }

    let sessions = state
        .label_to_session
        .lock()
        .map_err(|_| "session map lock poisoned".to_string())?;
    let session = sessions
        .get(window_label)
        .ok_or_else(|| format!("no viewer session for `{window_label}`"))?;
    let source_kind = session.current_source_kind();
    let dataset = session
        .dataset_for_source(&source_kind)
        .ok_or_else(|| format!("no dataset found for source `{source_kind}`"))?;

    Ok((source_kind, dataset))
}

async fn ensure_preview_dataset(
    state: &UiState,
    window_label: &str,
    preview: &PreviewRequest,
) -> Result<(String, Arc<DatasetF32>), String> {
    let key = preview_cache_key(&preview.op, &preview.params)?;

    let committed = {
        let sessions = state
            .label_to_session
            .lock()
            .map_err(|_| "session map lock poisoned".to_string())?;
        let session = sessions
            .get(window_label)
            .ok_or_else(|| format!("no viewer session for `{window_label}`"))?;

        if let Some(dataset) = session.preview_cache.get(&key) {
            return Ok((key, dataset.clone()));
        }

        session.committed_dataset.clone()
    };

    let op_name = preview.op.clone();
    let params = preview.params.clone();
    let ops_service = state.app.ops_service().clone();
    let generated = tauri::async_runtime::spawn_blocking(move || {
        ops_service
            .execute(&op_name, committed.as_ref(), &params)
            .map(|output| Arc::new(output.dataset))
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| format!("failed to join preview task: {error}"))??;

    let mut sessions = state
        .label_to_session
        .lock()
        .map_err(|_| "session map lock poisoned".to_string())?;
    let session = sessions
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
    let sessions = state
        .label_to_session
        .lock()
        .map_err(|_| "session map lock poisoned".to_string())?;
    let session = sessions
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

fn build_frame(dataset: &DatasetF32, request: &ViewerFrameRequest) -> Result<ViewerFrame, String> {
    let slice = extract_slice(dataset, request.z, request.t, request.channel)?;
    let pixels_u8 = to_u8_samples(&slice.values);
    let (min, max) = min_max(&slice.values);
    let histogram = histogram(&slice.values);
    let encoded = encode_slice_png(slice.width, slice.height, &pixels_u8)?;

    Ok(ViewerFrame {
        width: slice.width,
        height: slice.height,
        png_data_url: format!("data:image/png;base64,{}", BASE64_STANDARD.encode(encoded)),
        pixels_u8,
        histogram,
        min,
        max,
        values: slice.values,
    })
}

fn open_paths(app: &AppHandle, state: &UiState, paths: Vec<PathBuf>) -> OpenResult {
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

        match open_or_focus_viewer(app, state, &normalized) {
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

fn open_or_focus_viewer(
    app: &AppHandle,
    state: &UiState,
    path: &Path,
) -> Result<OpenOutcome, String> {
    if !is_supported_image_path(path) {
        return Err("unsupported image type".to_string());
    }

    let normalized_path = normalize_path(path);
    let existing_label = {
        let map = state
            .path_to_label
            .lock()
            .map_err(|_| "path map lock poisoned".to_string())?;
        map.get(&normalized_path).cloned()
    };

    if let Some(label) = existing_label {
        if let Some(window) = app.get_webview_window(&label) {
            focus_window(&window);
            return Ok(OpenOutcome::Focused { label });
        }
        remove_stale_mapping(state, &normalized_path, &label);
    }

    let dataset = state
        .app
        .io_service()
        .read(&normalized_path)
        .map_err(|error| error.to_string())?;
    let session = ViewerSession::new(normalized_path.clone(), Arc::new(dataset));

    let id = state.next_window_id.fetch_add(1, Ordering::Relaxed) + 1;
    let label = format!("{VIEWER_PREFIX}{id}");
    let title = format!(
        "{} - image-rs",
        normalized_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("Image")
    );

    let window = WebviewWindowBuilder::new(
        app,
        label.clone(),
        WebviewUrl::App("index.html?window=viewer".into()),
    )
    .title(title)
    .inner_size(1200.0, 860.0)
    .resizable(true)
    .build()
    .map_err(|error| format!("failed to create viewer window: {error}"))?;

    {
        let mut path_to_label = state
            .path_to_label
            .lock()
            .map_err(|_| "path map lock poisoned".to_string())?;
        let mut label_to_path = state
            .label_to_path
            .lock()
            .map_err(|_| "label map lock poisoned".to_string())?;
        let mut label_to_session = state
            .label_to_session
            .lock()
            .map_err(|_| "session map lock poisoned".to_string())?;

        path_to_label.insert(normalized_path.clone(), label.clone());
        label_to_path.insert(label.clone(), normalized_path);
        label_to_session.insert(label.clone(), session);
    }

    focus_window(&window);
    Ok(OpenOutcome::Opened { label })
}

fn focus_window(window: &WebviewWindow) {
    let _ = window.show();
    let _ = window.unminimize();
    let _ = window.set_focus();
}

fn remove_stale_mapping(state: &UiState, path: &Path, label: &str) {
    if let Ok(mut map) = state.path_to_label.lock()
        && map.get(path).is_some_and(|known| known == label)
    {
        map.remove(path);
    }
    if let Ok(mut map) = state.label_to_path.lock()
        && map.get(label).is_some_and(|known| known == path)
    {
        map.remove(label);
    }
    if let Ok(mut sessions) = state.label_to_session.lock() {
        sessions.remove(label);
    }
}

fn remove_viewer_by_label(state: &UiState, label: &str) {
    let path = if let Ok(mut label_to_path) = state.label_to_path.lock() {
        label_to_path.remove(label)
    } else {
        None
    };

    if let Some(path) = path
        && let Ok(mut path_to_label) = state.path_to_label.lock()
        && path_to_label.get(&path).is_some_and(|known| known == label)
    {
        path_to_label.remove(&path);
    }

    if let Ok(mut sessions) = state.label_to_session.lock() {
        sessions.remove(label);
    }
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

fn encode_slice_png(width: usize, height: usize, pixels: &[u8]) -> Result<Vec<u8>, String> {
    let image =
        ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(width as u32, height as u32, pixels.to_vec())
            .ok_or_else(|| "failed to build PNG buffer".to_string())?;
    let mut encoded = Vec::new();
    let mut cursor = Cursor::new(&mut encoded);
    DynamicImage::ImageLuma8(image)
        .write_to(&mut cursor, ImageFormat::Png)
        .map_err(|error| error.to_string())?;
    Ok(encoded)
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
    fs::canonicalize(&absolute).unwrap_or(absolute)
}

fn main() {
    let startup_arg = std::env::args().nth(1).map(PathBuf::from);
    let state = UiState::new(startup_arg);

    tauri::Builder::default()
        .manage(state)
        .setup(|app| {
            let state = app.state::<UiState>();
            if let Some(startup_path) = state.startup_input.clone() {
                let app_handle = app.handle().clone();
                let result = open_paths(&app_handle, state.inner(), vec![startup_path]);
                if let Some(launcher) = app.get_webview_window(LAUNCHER_LABEL) {
                    let _ = launcher.emit(LAUNCHER_RESULT_EVENT, &result);
                }
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() == LAUNCHER_LABEL {
                if let WindowEvent::DragDrop(DragDropEvent::Drop { paths, .. }) = event {
                    let app_handle = window.app_handle();
                    let state = window.state::<UiState>();
                    let result = open_paths(app_handle, state.inner(), paths.clone());
                    let _ = window.emit(LAUNCHER_RESULT_EVENT, &result);
                }

                if let WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    let _ = window.set_focus();
                }
            }

            if window.label().starts_with(VIEWER_PREFIX) && matches!(event, WindowEvent::Destroyed)
            {
                let state = window.state::<UiState>();
                remove_viewer_by_label(state.inner(), window.label());
            }
        })
        .invoke_handler(tauri::generate_handler![
            open_images,
            viewer_context,
            viewer_init,
            viewer_frame,
            viewer_frame_buffer,
            viewer_start_op,
            viewer_cancel_op,
            cycle_window,
            list_preview_ops,
            inspect_image,
            render_slice,
            slice_histogram,
            slice_plane,
            export_preview
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{canonical_json, preview_cache_key};

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
}
