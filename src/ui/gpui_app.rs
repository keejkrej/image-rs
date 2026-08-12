use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use gpui::{
    AnyWindowHandle, App, Bounds, ClickEvent, Context, ExternalPaths, FocusHandle, FontWeight,
    ImageSource, KeyBinding, KeyDownEvent, Menu, MenuItem, MouseButton, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, PathBuilder, Pixels, Point, Render, RenderImage, ScrollDelta,
    ScrollWheelEvent, SharedString, TitlebarOptions, Window, WindowBounds, WindowId, WindowKind,
    WindowOptions, actions, canvas, div, img, point, prelude::*, px, rgb, size,
};
use image::{Frame, RgbaImage};
use ndarray::{Array, IxDyn};
use rfd::FileDialog;
use serde_json::{Value, json};
use smallvec::smallvec;

use crate::formats::{read_dataset, supported_formats, write_dataset};
use crate::model::{AxisKind, Dataset, DatasetF32, Dim, Metadata, PixelType};
use crate::runtime::{
    AreaMask, CancellationToken, DatasetEffect, ExecutionControl, InvocationRequest,
    InvocationResult, OperationDescriptor, OperationScope, OpsService, PlanePosition,
    ProgressEvent, ProgressSink,
};

use super::command_registry;
use super::macros::{self, MacroCommandInvocation};
use super::menu::{self, MenuManifestItem, MenuManifestTopLevel};
use super::toolbar::{TOOLBAR_ITEMS, ToolId, icon_path, tool_from_shortcut};

const APP_TITLE: &str = "ImageJ — image-rs";
const MENU_HEIGHT: f32 = 40.0;
const TOOLBAR_HEIGHT: f32 = 52.0;
const INFO_HEIGHT: f32 = 32.0;
const STACK_HEIGHT: f32 = 32.0;
const STATUS_HEIGHT: f32 = 30.0;
const POPUP_WIDTH: f32 = 264.0;
const AUTO_THRESHOLD_DIVISOR: usize = 5_000;
const PROCESS_STACK_PARAMETER: &str = "__image_rs_process_stack";

// Native GPUI equivalents of a shadcn/Tailwind zinc palette.
const CHROME: u32 = 0xfafafa;
const CHROME_LIGHT: u32 = 0xffffff;
const CHROME_DARK: u32 = 0xe4e4e7;
const TEXT: u32 = 0x18181b;
const TEXT_MUTED: u32 = 0x71717a;
const MUTED: u32 = 0xf4f4f5;
const ACCENT: u32 = 0x2563eb;
const ACCENT_SOFT: u32 = 0xeff6ff;
const CANVAS: u32 = 0x18181b;

fn normalized_path_identity(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }

    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|directory| directory.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    if let (Some(parent), Some(file_name)) = (absolute.parent(), absolute.file_name())
        && let Ok(canonical_parent) = parent.canonicalize()
    {
        return canonical_parent.join(file_name);
    }
    absolute
}

fn other_path_owner<'a>(
    paths: impl IntoIterator<Item = (u64, &'a Path)>,
    current_tab_id: u64,
    requested_path: &Path,
) -> Option<u64> {
    let requested_path = normalized_path_identity(requested_path);
    paths.into_iter().find_map(|(tab_id, path)| {
        (tab_id != current_tab_id && normalized_path_identity(path) == requested_path)
            .then_some(tab_id)
    })
}

actions!(
    imagej,
    [
        NewImage,
        OpenImage,
        Save,
        SaveAs,
        CloseImage,
        Cut,
        Copy,
        Paste,
        Undo,
        Redo,
        ZoomIn,
        ZoomOut,
        ZoomActual,
        ZoomFit,
        NextTab,
        PreviousTab,
        Escape,
        Quit
    ]
);

#[derive(Debug, Clone)]
enum DialogState {
    About,
    ImageInfo {
        title: String,
        lines: Vec<String>,
    },
    Message {
        title: String,
        body: String,
    },
    ConfirmClose {
        tab_id: u64,
        title: String,
        continuation: CloseContinuation,
    },
    MacroRecorder,
    Operation {
        command_id: String,
        target_tab_id: Option<u64>,
        title: String,
        fields: Vec<ParameterField>,
        focused: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CloseContinuation {
    None,
    CloseAll,
    Quit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParameterKind {
    Boolean,
    Number,
    Text,
    Json,
}

fn dialog_window_spec(dialog: &DialogState) -> (String, f32) {
    match dialog {
        DialogState::About => ("About ImageJ — image-rs".to_string(), 260.0),
        DialogState::ImageInfo { title, lines } => (
            title.clone(),
            (170.0 + lines.len() as f32 * 28.0).min(640.0),
        ),
        DialogState::Message { title, .. } => (title.clone(), 240.0),
        DialogState::ConfirmClose { title, .. } => (format!("Save changes to {title}?"), 250.0),
        DialogState::MacroRecorder => ("Recorder — ImageJ / image-rs".to_string(), 440.0),
        DialogState::Operation { title, fields, .. } => (
            title.clone(),
            (170.0 + fields.len() as f32 * 38.0).clamp(240.0, 680.0),
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LookupTable {
    Grays,
    Fire,
    Ice,
    Spectrum,
    Rgb332,
    Red,
    Green,
    Blue,
    Cyan,
    Magenta,
    Yellow,
    RedGreen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DisplayAdjustMode {
    BrightnessContrast,
    WindowLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApplyLutScope {
    Slice,
    Stack,
}

#[derive(Debug, Clone)]
struct ParameterField {
    key: String,
    label: String,
    value: String,
    kind: ParameterKind,
}

/// Application-wide ImageJ measurement options.
///
/// ImageJ shares these choices across every image window. Keep them on the
/// launcher-owned application state rather than on individual viewers so
/// Analyze > Measure, Measure Stack, overlays, and the ROI Manager all cross
/// the same interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MeasurementSettings {
    area: bool,
    mean: bool,
    standard_deviation: bool,
    min_max: bool,
    centroid: bool,
    perimeter: bool,
    bounding_rectangle: bool,
    integrated_density: bool,
    median: bool,
    stack_position: bool,
    display_label: bool,
    invert_y_coordinates: bool,
    decimal_places: u8,
}

impl Default for MeasurementSettings {
    fn default() -> Self {
        // Matches ImageJ's Analyzer default: AREA + MEAN + MIN_MAX.
        Self {
            area: true,
            mean: true,
            standard_deviation: false,
            min_max: true,
            centroid: false,
            perimeter: false,
            bounding_rectangle: false,
            integrated_density: false,
            median: false,
            stack_position: false,
            display_label: false,
            invert_y_coordinates: false,
            decimal_places: 3,
        }
    }
}

impl MeasurementSettings {
    #[cfg(test)]
    fn all_supported() -> Self {
        Self {
            area: true,
            mean: true,
            standard_deviation: true,
            min_max: true,
            centroid: true,
            perimeter: true,
            bounding_rectangle: true,
            integrated_density: true,
            median: true,
            stack_position: true,
            display_label: true,
            invert_y_coordinates: false,
            decimal_places: 3,
        }
    }
}

#[derive(Debug, Clone)]
struct RoiSelection {
    tool: ToolId,
    points: Vec<(f32, f32)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RasterizedAreaMask {
    left: usize,
    top: usize,
    width: usize,
    height: usize,
    members: Vec<u8>,
}

#[derive(Clone)]
struct ActiveOperation {
    job_id: u64,
    revision: u64,
    input: Arc<DatasetF32>,
    cancellation: CancellationToken,
    progress: f32,
    message: String,
}

fn active_operation_matches(
    operation: &ActiveOperation,
    job_id: u64,
    revision: u64,
    input: &Arc<DatasetF32>,
) -> bool {
    operation.job_id == job_id
        && operation.revision == revision
        && Arc::ptr_eq(&operation.input, input)
}

fn choose_operation_scope(
    descriptor: &OperationDescriptor,
    process_stack: bool,
) -> Option<OperationScope> {
    if process_stack {
        return descriptor
            .supports(OperationScope::ZStack)
            .then_some(OperationScope::ZStack);
    }
    [
        OperationScope::ActivePlane,
        OperationScope::WholeDataset,
        OperationScope::ZStack,
        OperationScope::AllPlanes,
    ]
    .into_iter()
    .find(|scope| descriptor.supports(*scope))
}

fn take_process_stack_parameter(params: &mut Value) -> bool {
    params
        .as_object_mut()
        .and_then(|values| values.remove(PROCESS_STACK_PARAMETER))
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

struct UiProgressSink(Arc<Mutex<Option<ProgressEvent>>>);

impl ProgressSink for UiProgressSink {
    fn report(&self, event: ProgressEvent) {
        // The UI needs only the newest available update. Overwriting one shared slot prevents a
        // noisy native kernel or untrusted guest from queueing unbounded application work.
        if let Ok(mut latest) = self.0.lock() {
            *latest = Some(event);
        }
    }
}

struct MacroRunState {
    name: String,
    pending: VecDeque<MacroCommandInvocation>,
    lines: Vec<String>,
    executed: usize,
    awaiting_job_id: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct RoiPosition {
    channel: usize,
    z: usize,
    t: usize,
}

#[derive(Debug, Clone)]
struct ManagedRoi {
    id: u64,
    name: String,
    selection: RoiSelection,
    position: RoiPosition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManagedRoiSelectionGesture {
    Single,
    Toggle,
    Range { additive: bool },
}

fn apply_managed_roi_selection(
    order: &[u64],
    selected: &mut BTreeSet<u64>,
    anchor: &mut Option<u64>,
    clicked: u64,
    gesture: ManagedRoiSelectionGesture,
) -> bool {
    let Some(clicked_index) = order.iter().position(|id| *id == clicked) else {
        return false;
    };
    selected.retain(|id| order.contains(id));
    match gesture {
        ManagedRoiSelectionGesture::Single => {
            selected.clear();
            selected.insert(clicked);
            *anchor = Some(clicked);
        }
        ManagedRoiSelectionGesture::Toggle => {
            if !selected.remove(&clicked) {
                selected.insert(clicked);
            }
            *anchor = Some(clicked);
        }
        ManagedRoiSelectionGesture::Range { additive } => {
            let anchor_id = anchor
                .filter(|anchor_id| order.contains(anchor_id))
                .unwrap_or(clicked);
            let anchor_index = order
                .iter()
                .position(|id| *id == anchor_id)
                .unwrap_or(clicked_index);
            if !additive {
                selected.clear();
            }
            let start = anchor_index.min(clicked_index);
            let end = anchor_index.max(clicked_index);
            selected.extend(order[start..=end].iter().copied());
            *anchor = Some(anchor_id);
        }
    }
    true
}

fn effective_managed_roi_selection(order: &[u64], selected: &BTreeSet<u64>) -> BTreeSet<u64> {
    let live_selection = order
        .iter()
        .copied()
        .filter(|id| selected.contains(id))
        .collect::<BTreeSet<_>>();
    if live_selection.is_empty() {
        order.iter().copied().collect()
    } else {
        live_selection
    }
}

#[derive(Debug, Clone, Copy)]
struct RoiDrag {
    tab_id: u64,
    tool: ToolId,
}

#[derive(Debug, Clone)]
struct ClipboardPatch {
    width: usize,
    height: usize,
    pixels: Vec<f32>,
    pixel_type: PixelType,
}

#[derive(Clone)]
struct ImageTab {
    id: u64,
    /// Monotonic dataset identity used to reject stale background operation results.
    revision: u64,
    internal_label: String,
    title: String,
    path: Option<PathBuf>,
    dataset: Arc<DatasetF32>,
    render_image: Arc<RenderImage>,
    width: usize,
    height: usize,
    channels: usize,
    slices: usize,
    frames: usize,
    z: usize,
    t: usize,
    channel: usize,
    zoom: f32,
    scale_to_fit: bool,
    dirty: bool,
    undo: Vec<Arc<DatasetF32>>,
    redo: Vec<Arc<DatasetF32>>,
    roi: Option<RoiSelection>,
    overlays: Vec<RoiSelection>,
    overlays_hidden: bool,
    lut: LookupTable,
    lut_inverted: bool,
    display_ranges: Vec<(f32, f32)>,
}

impl ImageTab {
    fn from_dataset(
        id: u64,
        path: Option<PathBuf>,
        title: String,
        dataset: DatasetF32,
    ) -> Result<Self, String> {
        let dataset = Arc::new(dataset);
        let display_ranges = default_display_ranges(dataset.as_ref(), 0, 0);
        let (display_min, display_max) = display_ranges[0];
        let plane = render_dataset_plane(
            dataset.as_ref(),
            0,
            0,
            0,
            LookupTable::Grays,
            false,
            display_min,
            display_max,
        )?;
        let channels = axis_len(dataset.as_ref(), AxisKind::Channel);
        let slices = axis_len(dataset.as_ref(), AxisKind::Z);
        let frames = axis_len(dataset.as_ref(), AxisKind::Time);
        Ok(Self {
            id,
            revision: 0,
            internal_label: format!("viewer-{id}"),
            title,
            path,
            dataset,
            render_image: plane.image,
            width: plane.width,
            height: plane.height,
            channels,
            slices,
            frames,
            z: 0,
            t: 0,
            channel: 0,
            zoom: 1.0,
            scale_to_fit: true,
            dirty: false,
            undo: Vec::new(),
            redo: Vec::new(),
            roi: None,
            overlays: Vec::new(),
            overlays_hidden: false,
            lut: LookupTable::Grays,
            lut_inverted: false,
            display_ranges,
        })
    }

    fn refresh_render_image(&mut self) -> Result<(), String> {
        let (display_min, display_max) = self.display_range();
        let plane = render_dataset_plane(
            self.dataset.as_ref(),
            self.z,
            self.t,
            self.channel,
            self.lut,
            self.lut_inverted,
            display_min,
            display_max,
        )?;
        self.render_image = plane.image;
        self.width = plane.width;
        self.height = plane.height;
        Ok(())
    }

    fn display_range_index(&self) -> usize {
        if dataset_is_true_rgb(self.dataset.as_ref()) || self.display_ranges.len() <= 1 {
            0
        } else {
            self.channel
                .min(self.display_ranges.len().saturating_sub(1))
        }
    }

    fn display_range(&self) -> (f32, f32) {
        self.display_ranges
            .get(self.display_range_index())
            .copied()
            .unwrap_or_else(|| {
                default_display_range(self.dataset.as_ref(), self.z, self.t, self.channel)
            })
    }

    fn set_display_range(&mut self, minimum: f32, maximum: f32) {
        let expected = display_range_count(self.dataset.as_ref());
        if self.display_ranges.len() != expected {
            self.reset_display_ranges();
        }
        let index = self.display_range_index();
        self.display_ranges[index] = (minimum, maximum);
    }

    fn reset_display_range(&mut self) {
        let range = default_display_range(self.dataset.as_ref(), self.z, self.t, self.channel);
        self.set_display_range(range.0, range.1);
    }

    fn reset_display_ranges(&mut self) {
        self.display_ranges = default_display_ranges(self.dataset.as_ref(), self.z, self.t);
    }

    fn info_text(&self) -> String {
        let pixel_kind = match self.dataset.metadata.pixel_type {
            PixelType::U8 if dataset_is_true_rgb(self.dataset.as_ref()) => "RGB",
            PixelType::U8 => "8-bit",
            PixelType::U16 => "16-bit",
            PixelType::F32 => "32-bit",
        };
        let bytes = self
            .width
            .saturating_mul(self.height)
            .saturating_mul(self.channels.max(1))
            .saturating_mul(match self.dataset.metadata.pixel_type {
                PixelType::U8 => 1,
                PixelType::U16 => 2,
                PixelType::F32 => 4,
            });
        // ImageJ rounds a 512x512 RGB image (768 KiB) up to "1MB".
        let memory = if bytes >= 512 * 1024 {
            format!("{}MB", (bytes as f32 / (1024.0 * 1024.0)).ceil() as usize)
        } else {
            format!("{}K", (bytes as f32 / 1024.0).ceil() as usize)
        };
        let stack = if self.slices > 1 || self.frames > 1 {
            format!(
                "; C:{}/{} Z:{}/{} T:{}/{}",
                self.channel + 1,
                self.channels,
                self.z + 1,
                self.slices,
                self.t + 1,
                self.frames
            )
        } else {
            String::new()
        };
        format!(
            "{}x{} pixels; {}; {}{}",
            self.width, self.height, pixel_kind, memory, stack
        )
    }
}

fn rollback_failed_tab_state(
    tabs: &mut Vec<ImageTab>,
    activation_order: &mut Vec<u64>,
    active_tab: &mut Option<u64>,
    failed_tab_id: u64,
) -> Option<ImageTab> {
    let index = tabs.iter().position(|tab| tab.id == failed_tab_id)?;
    let removed = tabs.remove(index);
    activation_order.retain(|tab_id| *tab_id != failed_tab_id);
    if *active_tab == Some(failed_tab_id) {
        *active_tab = activation_order
            .iter()
            .rev()
            .copied()
            .find(|tab_id| tabs.iter().any(|tab| tab.id == *tab_id));
    }
    Some(removed)
}

struct RenderedPlane {
    width: usize,
    height: usize,
    image: Arc<RenderImage>,
}

#[derive(Debug, Clone, Copy)]
struct ViewerGeometry {
    zoom: f32,
    image_left: f32,
    image_top: f32,
    display_width: f32,
    display_height: f32,
}

struct ImageJApp {
    ops_service: OpsService,
    tabs: Vec<ImageTab>,
    active_tab: Option<u64>,
    activation_order: Vec<u64>,
    next_tab_id: u64,
    next_operation_job_id: u64,
    selected_tool: ToolId,
    open_menu: Option<usize>,
    open_submenu: Option<String>,
    menus: Vec<MenuManifestTopLevel>,
    status: String,
    progress: Option<f32>,
    active_operations: HashMap<u64, ActiveOperation>,
    dialog: Option<DialogState>,
    focus_handle: FocusHandle,
    last_pointer: HashMap<u64, (usize, usize, f32)>,
    last_repeatable_command: Option<(String, Value)>,
    roi_drag: Option<RoiDrag>,
    internal_clipboard: Option<ClipboardPatch>,
    roi_manager: Vec<ManagedRoi>,
    roi_manager_selected: BTreeSet<u64>,
    roi_manager_selection_anchor: Option<u64>,
    next_managed_roi_id: u64,
    roi_manager_show_all_target: Option<u64>,
    results: Vec<BTreeMap<String, Value>>,
    measurement_settings: MeasurementSettings,
    results_window_pending: bool,
    macro_recording: bool,
    macro_recorded: String,
    macro_run: Option<MacroRunState>,
    launcher_window: AnyWindowHandle,
    viewer_windows: HashMap<WindowId, u64>,
    viewer_handles: HashMap<u64, AnyWindowHandle>,
    menu_popup: Option<AnyWindowHandle>,
    dialog_window: Option<AnyWindowHandle>,
    results_window: Option<AnyWindowHandle>,
    roi_manager_window: Option<AnyWindowHandle>,
    display_adjust_mode: DisplayAdjustMode,
    display_auto_divisor: usize,
    display_adjust_window: Option<AnyWindowHandle>,
}

struct ImageViewerWindow {
    app: gpui::Entity<ImageJApp>,
    tab_id: u64,
    focus_handle: FocusHandle,
    ready: bool,
}

struct MenuPopupWindow {
    app: gpui::Entity<ImageJApp>,
    menu_index: usize,
    focus_handle: FocusHandle,
    ready: bool,
    wide: bool,
    height: f32,
}

struct AppDialogWindow {
    app: gpui::Entity<ImageJApp>,
    focus_handle: FocusHandle,
    ready: bool,
}

struct ResultsWindow {
    app: gpui::Entity<ImageJApp>,
    focus_handle: FocusHandle,
    ready: bool,
}

struct RoiManagerWindow {
    app: gpui::Entity<ImageJApp>,
    focus_handle: FocusHandle,
    ready: bool,
}

struct DisplayAdjustWindow {
    app: gpui::Entity<ImageJApp>,
    focus_handle: FocusHandle,
    ready: bool,
}

impl ImageViewerWindow {
    fn new(
        app: gpui::Entity<ImageJApp>,
        tab_id: u64,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus_handle = cx.focus_handle();
        window.focus(&focus_handle, cx);
        let weak_app = app.downgrade();
        window.on_window_should_close(cx, move |_, cx| {
            let Some(app) = weak_app.upgrade() else {
                return true;
            };
            let dirty = app.read(cx).tab(tab_id).is_some_and(|tab| tab.dirty);
            if !dirty {
                let _ = app.update(cx, |app, _| {
                    if let Some(operation) = app.active_operations.remove(&tab_id) {
                        operation.cancellation.cancel();
                    }
                });
                return true;
            }
            let _ = app.update(cx, |app, cx| app.request_close(tab_id, cx));
            false
        });
        cx.observe_window_activation(window, |viewer, window, cx| {
            if window.is_window_active() {
                let tab_id = viewer.tab_id;
                let app = viewer.app.downgrade();
                cx.defer(move |cx| {
                    if let Some(app) = app.upgrade() {
                        let _ = app.update(cx, |app, cx| {
                            app.activate_tab(tab_id);
                            cx.notify();
                        });
                    }
                });
            }
        })
        .detach();
        cx.observe(&app, |_, _, cx| cx.notify()).detach();
        Self {
            app,
            tab_id,
            focus_handle,
            ready: false,
        }
    }
}

impl MenuPopupWindow {
    fn new(
        app: gpui::Entity<ImageJApp>,
        menu_index: usize,
        height: f32,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus_handle = cx.focus_handle();
        window.focus(&focus_handle, cx);
        cx.observe_window_activation(window, |popup, window, cx| {
            if popup.ready && !window.is_window_active() {
                let app = popup.app.downgrade();
                cx.defer(move |cx| {
                    if let Some(app) = app.upgrade() {
                        let _ = app.update(cx, |app, cx| app.close_menu_popup(cx));
                    }
                });
            }
        })
        .detach();
        cx.observe(&app, |_, _, cx| cx.notify()).detach();
        Self {
            app,
            menu_index,
            focus_handle,
            ready: false,
            wide: false,
            height,
        }
    }
}

impl AppDialogWindow {
    fn new(app: gpui::Entity<ImageJApp>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();
        window.focus(&focus_handle, cx);
        cx.observe(&app, |_, _, cx| cx.notify()).detach();
        Self {
            app,
            focus_handle,
            ready: false,
        }
    }
}

impl ResultsWindow {
    fn new(app: gpui::Entity<ImageJApp>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();
        window.focus(&focus_handle, cx);
        cx.observe(&app, |_, _, cx| cx.notify()).detach();
        Self {
            app,
            focus_handle,
            ready: false,
        }
    }
}

impl RoiManagerWindow {
    fn new(app: gpui::Entity<ImageJApp>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();
        window.focus(&focus_handle, cx);
        cx.observe(&app, |_, _, cx| cx.notify()).detach();
        Self {
            app,
            focus_handle,
            ready: false,
        }
    }
}

impl DisplayAdjustWindow {
    fn new(app: gpui::Entity<ImageJApp>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();
        window.focus(&focus_handle, cx);
        cx.observe(&app, |_, _, cx| cx.notify()).detach();
        Self {
            app,
            focus_handle,
            ready: false,
        }
    }
}

impl ImageJApp {
    fn new(window: &mut Window, ops_service: OpsService, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();
        window.focus(&focus_handle, cx);
        let weak_app = cx.entity().downgrade();
        window.on_window_should_close(cx, move |_, cx| {
            let Some(app) = weak_app.upgrade() else {
                return true;
            };
            let has_dirty_tabs = app.read(cx).tabs.iter().any(|tab| tab.dirty);
            if !has_dirty_tabs {
                let _ = app.update(cx, |app, _| {
                    for (_, operation) in app.active_operations.drain() {
                        operation.cancellation.cancel();
                    }
                });
                return true;
            }
            let _ = app.update(cx, |app, cx| app.request_quit(cx));
            false
        });
        Self {
            ops_service,
            tabs: Vec::new(),
            active_tab: None,
            activation_order: Vec::new(),
            next_tab_id: 0,
            next_operation_job_id: 1,
            selected_tool: ToolId::Rect,
            open_menu: None,
            open_submenu: None,
            menus: menu::manifest().clone(),
            status: format!(
                "ImageJ-compatible GPUI shell; {} commands",
                menu::manifest_commands().len()
            ),
            progress: None,
            active_operations: HashMap::new(),
            dialog: None,
            focus_handle,
            last_pointer: HashMap::new(),
            last_repeatable_command: None,
            roi_drag: None,
            internal_clipboard: None,
            roi_manager: Vec::new(),
            roi_manager_selected: BTreeSet::new(),
            roi_manager_selection_anchor: None,
            next_managed_roi_id: 0,
            roi_manager_show_all_target: None,
            results: Vec::new(),
            measurement_settings: MeasurementSettings::default(),
            results_window_pending: false,
            macro_recording: false,
            macro_recorded: String::new(),
            macro_run: None,
            launcher_window: window.window_handle(),
            viewer_windows: HashMap::new(),
            viewer_handles: HashMap::new(),
            menu_popup: None,
            dialog_window: None,
            results_window: None,
            roi_manager_window: None,
            display_adjust_mode: DisplayAdjustMode::BrightnessContrast,
            display_auto_divisor: 0,
            display_adjust_window: None,
        }
    }

    fn active_index(&self) -> Option<usize> {
        let active = self.active_tab?;
        self.tabs.iter().position(|tab| tab.id == active)
    }

    fn active_tab(&self) -> Option<&ImageTab> {
        self.active_index().and_then(|index| self.tabs.get(index))
    }

    fn active_tab_mut(&mut self) -> Option<&mut ImageTab> {
        let index = self.active_index()?;
        self.tabs.get_mut(index)
    }

    fn cancel_active_operation(&mut self) -> bool {
        let Some(tab_id) = self.active_tab else {
            return false;
        };
        let Some(operation) = self.active_operations.get_mut(&tab_id) else {
            return false;
        };
        operation.cancellation.cancel();
        operation.message = "Cancelling operation…".into();
        self.status = "Cancelling operation…".into();
        true
    }

    fn tab(&self, id: u64) -> Option<&ImageTab> {
        self.tabs.iter().find(|tab| tab.id == id)
    }

    fn tab_mut(&mut self, id: u64) -> Option<&mut ImageTab> {
        self.tabs.iter_mut().find(|tab| tab.id == id)
    }

    fn viewer_for_window(&self, window: &Window) -> Option<u64> {
        let window_id = window.window_handle().window_id();
        self.viewer_windows.get(&window_id).copied()
    }

    fn activate_session_for_window(&mut self, window: &Window) -> Option<u64> {
        let id = self.viewer_for_window(window)?;
        self.activate_tab(id);
        Some(id)
    }

    fn activate_tab(&mut self, id: u64) {
        if self.tab(id).is_none() {
            return;
        }
        if self.active_tab != Some(id) {
            if self.roi_manager_show_all_target.is_some() {
                self.roi_manager_show_all_target = None;
            }
            self.display_auto_divisor = 0;
        }
        self.active_tab = Some(id);
        self.activation_order.retain(|candidate| *candidate != id);
        self.activation_order.push(id);
    }

    fn focus_launcher(&mut self, cx: &mut Context<Self>) {
        let _ = self
            .launcher_window
            .update(cx, |_, window, _| window.activate_window());
    }

    fn focus_viewer(&mut self, id: u64, cx: &mut Context<Self>) {
        self.activate_tab(id);
        if let Some(handle) = self.viewer_handles.get(&id).copied() {
            let _ = handle.update(cx, |_, window, _| window.activate_window());
        }
    }

    fn show_all_windows(&mut self, cx: &mut Context<Self>) {
        for id in self.activation_order.clone() {
            if let Some(handle) = self.viewer_handles.get(&id).copied() {
                let _ = handle.update(cx, |_, window, _| window.activate_window());
            }
        }
        self.focus_launcher(cx);
        self.status = format!("Raised {} image window(s)", self.viewer_handles.len());
    }

    fn maximize_active_viewer(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.active_tab else {
            self.status = "No image is open".into();
            return;
        };
        if let Some(handle) = self.viewer_handles.get(&id).copied() {
            let _ = handle.update(cx, |_, window, _| window.zoom_window());
            self.status = "Toggled the active viewer’s maximized state".into();
        }
    }

    fn close_menu_popup(&mut self, cx: &mut Context<Self>) {
        self.open_menu = None;
        self.open_submenu = None;
        if let Some(handle) = self.menu_popup.take() {
            let _ = handle.update(cx, |_, window, _| window.remove_window());
        }
    }

    fn open_menu_popup(&mut self, menu_index: usize, launcher: &Window, cx: &mut Context<Self>) {
        if self.open_menu == Some(menu_index) {
            self.close_menu_popup(cx);
            return;
        }
        self.close_menu_popup(cx);
        let Some(menu) = self.menus.get(menu_index) else {
            return;
        };
        let live_window_height = if menu._id == "window" && !self.tabs.is_empty() {
            7.0 + self.tabs.len() as f32 * 31.0
        } else {
            0.0
        };
        let height = (menu
            .items
            .iter()
            .map(|item| if item.kind == "separator" { 7.0 } else { 31.0 })
            .sum::<f32>()
            + live_window_height
            + 8.0)
            .clamp(48.0, 620.0);
        let launcher_bounds = launcher.bounds();
        let bounds = Bounds::new(
            point(
                launcher_bounds.origin.x + px(menu_left(&self.menus, menu_index)),
                launcher_bounds.origin.y + px(MENU_HEIGHT),
            ),
            size(px(POPUP_WIDTH), px(height)),
        );
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            titlebar: None,
            kind: WindowKind::PopUp,
            is_resizable: false,
            is_minimizable: false,
            app_id: Some("image-rs".into()),
            ..Default::default()
        };
        let app = cx.entity();
        self.open_menu = Some(menu_index);
        match cx.open_window(options, move |window, cx| {
            cx.new(|cx| MenuPopupWindow::new(app, menu_index, height, window, cx))
        }) {
            Ok(handle) => {
                let any: AnyWindowHandle = handle.into();
                self.menu_popup = Some(any);
                let _ = handle.update(cx, |popup, _, cx| {
                    popup.ready = true;
                    cx.notify();
                });
            }
            Err(error) => {
                self.open_menu = None;
                self.status = format!("Could not open menu: {error}");
            }
        }
    }

    fn open_dialog_window(&mut self, cx: &mut Context<Self>) {
        let Some(dialog) = self.dialog.as_ref() else {
            return;
        };
        if let Some(handle) = self.dialog_window {
            let _ = handle.update(cx, |_, window, _| window.activate_window());
            return;
        }
        let (title, height) = dialog_window_spec(dialog);
        let bounds = Bounds::new(point(px(220.0), px(180.0)), size(px(520.0), px(height)));
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            window_min_size: Some(size(px(420.0), px(200.0))),
            kind: WindowKind::Floating,
            titlebar: Some(TitlebarOptions {
                title: Some(SharedString::from(title)),
                appears_transparent: false,
                ..Default::default()
            }),
            app_id: Some("image-rs".into()),
            ..Default::default()
        };
        let app = cx.entity();
        match cx.open_window(options, move |window, cx| {
            cx.new(|cx| AppDialogWindow::new(app, window, cx))
        }) {
            Ok(handle) => {
                let any: AnyWindowHandle = handle.into();
                self.dialog_window = Some(any);
                let _ = handle.update(cx, |dialog, _, cx| {
                    dialog.ready = true;
                    cx.notify();
                });
            }
            Err(error) => self.status = format!("Could not open dialog: {error}"),
        }
    }

    fn dismiss_dialog_window(&mut self, cx: &mut Context<Self>) {
        self.dialog = None;
        let Some(handle) = self.dialog_window.take() else {
            return;
        };
        cx.defer(move |cx| {
            let _ = handle.update(cx, |_, window, _| window.remove_window());
        });
    }

    fn cancel_dialog(&mut self, cx: &mut Context<Self>) {
        let target_tab_id = match self.dialog.as_ref() {
            Some(DialogState::ConfirmClose { tab_id, .. }) => Some(*tab_id),
            Some(DialogState::Operation { target_tab_id, .. }) => *target_tab_id,
            _ => None,
        };
        self.status = match self.dialog.as_ref() {
            Some(DialogState::ConfirmClose { .. }) => "Close canceled".into(),
            Some(DialogState::Operation { .. }) => "Command canceled".into(),
            Some(DialogState::MacroRecorder) => {
                self.macro_recording = false;
                "Macro Recorder stopped".into()
            }
            _ => self.status.clone(),
        };
        self.dismiss_dialog_window(cx);
        if let Some(tab_id) = target_tab_id {
            self.focus_viewer(tab_id, cx);
        }
    }

    fn open_results_window(&mut self, cx: &mut Context<Self>) {
        self.results_window_pending = false;
        if let Some(handle) = self.results_window {
            let _ = handle.update(cx, |_, window, _| window.activate_window());
            return;
        }
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(Bounds::new(
                point(px(260.0), px(220.0)),
                size(px(780.0), px(420.0)),
            ))),
            window_min_size: Some(size(px(480.0), px(240.0))),
            kind: WindowKind::Floating,
            titlebar: Some(TitlebarOptions {
                title: Some("Results — ImageJ / image-rs".into()),
                appears_transparent: false,
                ..Default::default()
            }),
            app_id: Some("image-rs".into()),
            ..Default::default()
        };
        let app = cx.entity();
        match cx.open_window(options, move |window, cx| {
            cx.new(|cx| ResultsWindow::new(app, window, cx))
        }) {
            Ok(handle) => {
                let any: AnyWindowHandle = handle.into();
                self.results_window = Some(any);
                let _ = handle.update(cx, |results, _, cx| {
                    results.ready = true;
                    cx.notify();
                });
            }
            Err(error) => self.status = format!("Could not open Results: {error}"),
        }
    }

    fn open_roi_manager_window(&mut self, cx: &mut Context<Self>) {
        if let Some(handle) = self.roi_manager_window {
            let _ = handle.update(cx, |_, window, _| window.activate_window());
            return;
        }
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(Bounds::new(
                point(px(300.0), px(180.0)),
                size(px(440.0), px(560.0)),
            ))),
            window_min_size: Some(size(px(360.0), px(360.0))),
            kind: WindowKind::Floating,
            titlebar: Some(TitlebarOptions {
                title: Some("ROI Manager — ImageJ / image-rs".into()),
                appears_transparent: false,
                ..Default::default()
            }),
            app_id: Some("image-rs".into()),
            ..Default::default()
        };
        let app = cx.entity();
        match cx.open_window(options, move |window, cx| {
            cx.new(|cx| RoiManagerWindow::new(app, window, cx))
        }) {
            Ok(handle) => {
                let any: AnyWindowHandle = handle.into();
                self.roi_manager_window = Some(any);
                let _ = handle.update(cx, |manager, _, cx| {
                    manager.ready = true;
                    cx.notify();
                });
            }
            Err(error) => self.status = format!("Could not open ROI Manager: {error}"),
        }
    }

    fn open_display_adjuster(&mut self, mode: DisplayAdjustMode, cx: &mut Context<Self>) {
        if self.active_tab.is_none() {
            self.status = "Display adjustment requires an open image".into();
            return;
        }
        self.display_adjust_mode = mode;
        if let Some(handle) = self.display_adjust_window {
            let _ = handle.update(cx, |_, window, _| window.activate_window());
            cx.notify();
            return;
        }
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(Bounds::new(
                point(px(340.0), px(200.0)),
                size(px(480.0), px(520.0)),
            ))),
            window_min_size: Some(size(px(420.0), px(480.0))),
            kind: WindowKind::Floating,
            titlebar: Some(TitlebarOptions {
                title: Some("Brightness & Contrast — ImageJ / image-rs".into()),
                appears_transparent: false,
                ..Default::default()
            }),
            app_id: Some("image-rs".into()),
            ..Default::default()
        };
        let app = cx.entity();
        match cx.open_window(options, move |window, cx| {
            cx.new(|cx| DisplayAdjustWindow::new(app, window, cx))
        }) {
            Ok(handle) => {
                let any: AnyWindowHandle = handle.into();
                self.display_adjust_window = Some(any);
                let _ = handle.update(cx, |adjuster, _, cx| {
                    adjuster.ready = true;
                    cx.notify();
                });
                self.status = "Display adjustment is live; image pixels remain unchanged".into();
            }
            Err(error) => self.status = format!("Could not open display adjuster: {error}"),
        }
    }

    fn open_brightness_contrast_dialog(&mut self, cx: &mut Context<Self>) {
        self.open_display_adjuster(DisplayAdjustMode::BrightnessContrast, cx);
    }

    fn open_window_level_dialog(&mut self, cx: &mut Context<Self>) {
        self.open_display_adjuster(DisplayAdjustMode::WindowLevel, cx);
    }

    fn rollback_unopened_tab(&mut self, id: u64) {
        let _ = rollback_failed_tab_state(
            &mut self.tabs,
            &mut self.activation_order,
            &mut self.active_tab,
            id,
        );
        self.last_pointer.remove(&id);
        if self.roi_manager_show_all_target == Some(id) {
            self.roi_manager_show_all_target = None;
        }
    }

    fn open_viewer_window(&mut self, id: u64, cx: &mut Context<Self>) -> bool {
        if self.viewer_handles.contains_key(&id) {
            self.focus_viewer(id, cx);
            return true;
        }
        let Some(tab) = self.tab(id) else {
            self.status = format!("Could not open viewer: session viewer-{id} is missing");
            return false;
        };
        let title = tab.title.clone();
        let stack_height = if has_stack_controls(tab) {
            STACK_HEIGHT
        } else {
            0.0
        };
        let width = (tab.width as f32 + 24.0).clamp(360.0, 1_240.0);
        let height = (tab.height as f32 + INFO_HEIGHT + stack_height + STATUS_HEIGHT + 24.0)
            .clamp(280.0, 1_020.0);
        let mut bounds = Bounds::centered(None, size(px(width), px(height)), cx);
        let cascade = ((id.saturating_sub(1) % 8) as f32) * 28.0;
        bounds.origin.x += px(cascade);
        bounds.origin.y += px(cascade);
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            window_min_size: Some(size(px(300.0), px(220.0))),
            titlebar: Some(TitlebarOptions {
                title: Some(SharedString::from(format!("{title} — ImageJ / image-rs"))),
                appears_transparent: false,
                ..Default::default()
            }),
            app_id: Some("image-rs".into()),
            ..Default::default()
        };
        let app = cx.entity();
        match cx.open_window(options, move |window, cx| {
            cx.new(|cx| ImageViewerWindow::new(app, id, window, cx))
        }) {
            Ok(handle) => {
                let handle: AnyWindowHandle = handle.into();
                self.viewer_windows.insert(handle.window_id(), id);
                self.viewer_handles.insert(id, handle);
                if let Some(handle) = handle.downcast::<ImageViewerWindow>() {
                    let _ = handle.update(cx, |viewer, _, cx| {
                        viewer.ready = true;
                        cx.notify();
                    });
                }
                self.focus_viewer(id, cx);
                true
            }
            Err(error) => {
                self.status = format!("Could not open viewer: {error}");
                false
            }
        }
    }

    fn handle_window_closed(&mut self, window_id: WindowId, cx: &mut Context<Self>) {
        if self
            .menu_popup
            .is_some_and(|handle| handle.window_id() == window_id)
        {
            self.menu_popup = None;
            self.open_menu = None;
            self.open_submenu = None;
            cx.notify();
            return;
        }
        if self
            .dialog_window
            .is_some_and(|handle| handle.window_id() == window_id)
        {
            let target_tab_id = match self.dialog.as_ref() {
                Some(DialogState::ConfirmClose { tab_id, .. }) => Some(*tab_id),
                Some(DialogState::Operation { target_tab_id, .. }) => *target_tab_id,
                _ => None,
            };
            self.status = match self.dialog.as_ref() {
                Some(DialogState::ConfirmClose { .. }) => "Close canceled".into(),
                Some(DialogState::Operation { .. }) => "Command canceled".into(),
                _ => self.status.clone(),
            };
            self.dialog_window = None;
            if matches!(self.dialog, Some(DialogState::MacroRecorder)) {
                self.macro_recording = false;
            }
            self.dialog = None;
            if let Some(tab_id) = target_tab_id {
                self.focus_viewer(tab_id, cx);
            }
            cx.notify();
            return;
        }
        if self
            .results_window
            .is_some_and(|handle| handle.window_id() == window_id)
        {
            self.results_window = None;
            cx.notify();
            return;
        }
        if self
            .roi_manager_window
            .is_some_and(|handle| handle.window_id() == window_id)
        {
            self.roi_manager_window = None;
            self.roi_manager_show_all_target = None;
            cx.notify();
            return;
        }
        if self
            .display_adjust_window
            .is_some_and(|handle| handle.window_id() == window_id)
        {
            self.display_adjust_window = None;
            cx.notify();
            return;
        }
        if window_id == self.launcher_window.window_id() {
            for operation in self.active_operations.values() {
                operation.cancellation.cancel();
            }
            cx.quit();
            return;
        }
        let Some(tab_id) = self.viewer_windows.remove(&window_id) else {
            return;
        };
        if let Some(operation) = self.active_operations.remove(&tab_id) {
            operation.cancellation.cancel();
        }
        self.viewer_handles.remove(&tab_id);
        if self.roi_manager_show_all_target == Some(tab_id) {
            self.roi_manager_show_all_target = None;
        }
        self.last_pointer.remove(&tab_id);
        self.activation_order
            .retain(|candidate| *candidate != tab_id);
        if let Some(index) = self.tabs.iter().position(|tab| tab.id == tab_id) {
            let closed = self.tabs.remove(index);
            if self.active_tab == Some(tab_id) {
                self.active_tab = self.activation_order.last().copied();
            }
            self.status = format!("Closed {}", closed.title);
        }
        cx.notify();
    }

    fn open_paths(&mut self, paths: impl IntoIterator<Item = PathBuf>, cx: &mut Context<Self>) {
        let mut opened = 0usize;
        let mut focused = 0usize;
        let mut errors = Vec::new();
        for path in paths {
            let path = normalized_path_identity(&path);
            if let Some(existing_id) = self
                .tabs
                .iter()
                .find(|tab| tab.path.as_deref() == Some(path.as_path()))
                .map(|tab| tab.id)
            {
                self.focus_viewer(existing_id, cx);
                focused += 1;
                continue;
            }
            match read_dataset(&path) {
                Ok(dataset) => {
                    self.next_tab_id = self.next_tab_id.saturating_add(1);
                    let title = path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("Untitled")
                        .to_string();
                    match ImageTab::from_dataset(
                        self.next_tab_id,
                        Some(path.clone()),
                        title,
                        dataset,
                    ) {
                        Ok(tab) => {
                            let id = tab.id;
                            self.tabs.push(tab);
                            if self.open_viewer_window(id, cx) {
                                opened += 1;
                            } else {
                                let error = self.status.clone();
                                self.rollback_unopened_tab(id);
                                errors.push(format!("{}: {error}", path.display()));
                            }
                        }
                        Err(error) => errors.push(format!("{}: {error}", path.display())),
                    }
                }
                Err(error) => errors.push(format!("{}: {error}", path.display())),
            }
        }
        self.status = if errors.is_empty() {
            format!("Opened {opened}; focused {focused}")
        } else {
            format!(
                "Opened {opened}; focused {focused}; {} error(s): {}",
                errors.len(),
                errors.join(" | ")
            )
        };
    }

    fn pick_and_open(&mut self, cx: &mut Context<Self>) {
        let mut picker = FileDialog::new();
        picker = picker.add_filter("Images", supported_formats());
        if let Some(paths) = picker.pick_files() {
            self.open_paths(paths, cx);
        }
    }

    fn open_new_image_dialog(&mut self) {
        self.dialog = Some(DialogState::Operation {
            command_id: "__new_image".into(),
            target_tab_id: None,
            title: "New Image".into(),
            fields: vec![
                ParameterField {
                    key: "title".into(),
                    label: "Name".into(),
                    value: format!("Untitled-{}", self.next_tab_id + 1),
                    kind: ParameterKind::Text,
                },
                ParameterField {
                    key: "width".into(),
                    label: "Width".into(),
                    value: "512".into(),
                    kind: ParameterKind::Number,
                },
                ParameterField {
                    key: "height".into(),
                    label: "Height".into(),
                    value: "512".into(),
                    kind: ParameterKind::Number,
                },
                ParameterField {
                    key: "slices".into(),
                    label: "Slices".into(),
                    value: "1".into(),
                    kind: ParameterKind::Number,
                },
                ParameterField {
                    key: "channels".into(),
                    label: "Channels".into(),
                    value: "1".into(),
                    kind: ParameterKind::Number,
                },
                ParameterField {
                    key: "frames".into(),
                    label: "Frames".into(),
                    value: "1".into(),
                    kind: ParameterKind::Number,
                },
                ParameterField {
                    key: "pixel_type".into(),
                    label: "Pixel type (u8/u16/f32)".into(),
                    value: "u8".into(),
                    kind: ParameterKind::Text,
                },
                ParameterField {
                    key: "fill".into(),
                    label: "Fill value".into(),
                    value: "0".into(),
                    kind: ParameterKind::Number,
                },
            ],
            focused: 0,
        });
        self.status = "Create an 8-bit image or stack".into();
    }

    fn open_measurement_settings_dialog(&mut self) {
        let settings = self.measurement_settings;
        let field = |key: &str, label: &str, enabled: bool| ParameterField {
            key: key.into(),
            label: label.into(),
            value: enabled.to_string(),
            kind: ParameterKind::Boolean,
        };
        self.dialog = Some(DialogState::Operation {
            command_id: "__measurement_settings".into(),
            target_tab_id: None,
            title: "Set Measurements".into(),
            fields: vec![
                field("area", "Area", settings.area),
                field("mean", "Mean gray value", settings.mean),
                field(
                    "standard_deviation",
                    "Standard deviation",
                    settings.standard_deviation,
                ),
                field("min_max", "Min & max gray value", settings.min_max),
                field("centroid", "Centroid", settings.centroid),
                field("perimeter", "Perimeter", settings.perimeter),
                field(
                    "bounding_rectangle",
                    "Bounding rectangle",
                    settings.bounding_rectangle,
                ),
                field(
                    "integrated_density",
                    "Integrated density",
                    settings.integrated_density,
                ),
                field("median", "Median", settings.median),
                field("stack_position", "Stack position", settings.stack_position),
                field("display_label", "Display label", settings.display_label),
                field(
                    "invert_y_coordinates",
                    "Invert Y coordinates",
                    settings.invert_y_coordinates,
                ),
                ParameterField {
                    key: "decimal_places".into(),
                    label: "Decimal places (0-9)".into(),
                    value: settings.decimal_places.to_string(),
                    kind: ParameterKind::Number,
                },
            ],
            focused: 0,
        });
        self.status = "Choose the columns shared by Measure, Measure Stack, and ROI Manager".into();
    }

    fn open_display_range_set_dialog(&mut self) {
        let Some(tab) = self.active_tab() else {
            self.status = "Brightness/Contrast requires an open image".into();
            return;
        };
        let (display_min, display_max) = tab.display_range();
        self.dialog = Some(DialogState::Operation {
            command_id: "__display_range".into(),
            target_tab_id: Some(tab.id),
            title: "Brightness/Contrast".into(),
            fields: vec![
                ParameterField {
                    key: "minimum".into(),
                    label: "Minimum".into(),
                    value: format_compact_number(display_min),
                    kind: ParameterKind::Number,
                },
                ParameterField {
                    key: "maximum".into(),
                    label: "Maximum".into(),
                    value: format_compact_number(display_max),
                    kind: ParameterKind::Number,
                },
            ],
            focused: 0,
        });
        self.status = "Adjust the viewer display range; image pixels remain unchanged".into();
    }

    fn open_window_level_set_dialog(&mut self) {
        let Some(tab) = self.active_tab() else {
            self.status = "Window/Level requires an open image".into();
            return;
        };
        let (display_min, display_max) = tab.display_range();
        let window_width = display_max - display_min;
        let level = (display_max + display_min) * 0.5;
        self.dialog = Some(DialogState::Operation {
            command_id: "__window_level".into(),
            target_tab_id: Some(tab.id),
            title: "Window/Level".into(),
            fields: vec![
                ParameterField {
                    key: "window".into(),
                    label: "Window".into(),
                    value: format_compact_number(window_width),
                    kind: ParameterKind::Number,
                },
                ParameterField {
                    key: "level".into(),
                    label: "Level".into(),
                    value: format_compact_number(level),
                    kind: ParameterKind::Number,
                },
            ],
            focused: 0,
        });
        self.status = "Adjust window width and center level; image pixels remain unchanged".into();
    }

    fn apply_display_range(
        &mut self,
        minimum: f32,
        maximum: f32,
        record: bool,
    ) -> Result<String, String> {
        if !minimum.is_finite() || !maximum.is_finite() || maximum < minimum {
            let message = "Display maximum must not be less than minimum".to_string();
            self.status = message.clone();
            return Err(message);
        }
        let Some(tab) = self.active_tab_mut() else {
            let message = "The image that opened this dialog has been closed".to_string();
            self.status = message.clone();
            return Err(message);
        };
        let previous = tab.display_range();
        tab.set_display_range(minimum, maximum);
        if let Err(error) = tab.refresh_render_image() {
            tab.set_display_range(previous.0, previous.1);
            self.status = error.clone();
            return Err(error);
        }
        let message = format!(
            "Display range: {}–{} (pixels unchanged)",
            format_compact_number(minimum),
            format_compact_number(maximum)
        );
        self.status = message.clone();
        if record {
            self.record_command(
                "macro.set_min_and_max",
                Some(&json!({"minimum": minimum, "maximum": maximum})),
            );
        }
        Ok(message)
    }

    fn adjust_display_from_slider(&mut self, control: usize, fraction: f32) {
        let Some(tab) = self.active_tab() else {
            self.status = "Open an image to adjust its display".into();
            return;
        };
        let fraction = fraction.clamp(0.0, 1.0);
        let (domain_min, domain_max) = display_control_domain(tab);
        let domain_span = (domain_max - domain_min).max(f32::EPSILON);
        let (current_min, current_max) = tab.display_range();
        let mode = self.display_adjust_mode;
        let (minimum, maximum, _command_id) = match mode {
            DisplayAdjustMode::BrightnessContrast => {
                let value = domain_min + domain_span * fraction;
                match control {
                    0 => {
                        let maximum = current_max.min(domain_max).max(value);
                        (value, maximum, "image.adjust.brightness")
                    }
                    1 => {
                        let minimum = current_min.max(domain_min).min(value);
                        (minimum, value, "image.adjust.brightness")
                    }
                    2 => {
                        let window_width = (current_max - current_min).max(0.0);
                        // Moving right brightens the image by lowering the display center,
                        // matching ImageJ's Brightness scrollbar semantics.
                        let level = domain_max - domain_span * fraction;
                        (
                            level - window_width * 0.5,
                            level + window_width * 0.5,
                            "image.adjust.brightness",
                        )
                    }
                    3 => {
                        let window_width = contrast_window_from_fraction(domain_span, fraction);
                        let level = (current_max + current_min) * 0.5;
                        (
                            level - window_width * 0.5,
                            level + window_width * 0.5,
                            "image.adjust.brightness",
                        )
                    }
                    _ => return,
                }
            }
            DisplayAdjustMode::WindowLevel => {
                let current_window = (current_max - current_min).max(0.0);
                let current_level = (current_max + current_min) * 0.5;
                match control {
                    0 => {
                        let window_width = contrast_window_from_fraction(domain_span, fraction);
                        (
                            current_level - window_width * 0.5,
                            current_level + window_width * 0.5,
                            "image.adjust.window_level",
                        )
                    }
                    1 => {
                        let level = domain_min + domain_span * fraction;
                        (
                            level - current_window * 0.5,
                            level + current_window * 0.5,
                            "image.adjust.window_level",
                        )
                    }
                    _ => return,
                }
            }
        };
        let _ = self.apply_display_range(minimum, maximum, false);
    }

    fn reset_display_adjustment(&mut self, record: bool) -> Result<String, String> {
        let Some(tab) = self.active_tab_mut() else {
            let message = "Open an image to reset its display range".to_string();
            self.status = message.clone();
            return Err(message);
        };
        let previous = tab.display_range();
        tab.reset_display_range();
        let (minimum, maximum) = tab.display_range();
        if let Err(error) = tab.refresh_render_image() {
            tab.set_display_range(previous.0, previous.1);
            self.status = error.clone();
            return Err(error);
        }
        self.display_auto_divisor = 0;
        let message = format!(
            "Display reset to {}–{} (pixels unchanged)",
            format_compact_number(minimum),
            format_compact_number(maximum)
        );
        self.status = message.clone();
        if record {
            self.record_command("macro.reset_min_and_max", None);
        }
        Ok(message)
    }

    fn auto_display_adjustment(&mut self) {
        self.display_auto_divisor = if self.display_auto_divisor < 10 {
            AUTO_THRESHOLD_DIVISOR
        } else {
            self.display_auto_divisor / 2
        };
        let divisor = self.display_auto_divisor;
        let Some((minimum, maximum)) = self
            .active_tab()
            .and_then(|tab| auto_display_range(tab, divisor))
        else {
            self.status = "Auto contrast could not find finite pixels in the active plane".into();
            return;
        };
        let _ = self.apply_display_range(minimum, maximum, true);
    }

    fn open_apply_lut_dialog(&mut self) {
        let Some(tab) = self.active_tab() else {
            self.status = "Apply LUT requires an open image".into();
            return;
        };
        let pixel_maximum = match tab.dataset.metadata.pixel_type {
            PixelType::U8 => 255.0,
            PixelType::U16 => 65_535.0,
            PixelType::F32 => {
                self.status = "Apply LUT does not support 32-bit images".into();
                return;
            }
        };
        let (minimum, maximum) = tab.display_range();
        if !minimum.is_finite() || !maximum.is_finite() || maximum < minimum {
            self.status = "Choose a valid display range before applying the LUT".into();
            return;
        }
        if minimum == 0.0 && maximum == pixel_maximum {
            self.status =
                "The display range is unchanged; adjust Brightness/Contrast before Apply LUT"
                    .into();
            return;
        }
        let has_stack = tab.slices > 1 || tab.frames > 1;
        self.dialog = Some(DialogState::Operation {
            command_id: "__apply_lut".into(),
            target_tab_id: Some(tab.id),
            title: "Apply Lookup Table?".into(),
            fields: if has_stack {
                vec![ParameterField {
                    key: "stack".into(),
                    label: "Apply to every Z/T plane".into(),
                    value: "false".into(),
                    kind: ParameterKind::Boolean,
                }]
            } else {
                Vec::new()
            },
            focused: 0,
        });
        self.status = "Apply LUT will change pixel values; choose OK to continue".into();
    }

    fn apply_lut_to_pixels(&mut self, requested_scope: ApplyLutScope) -> Result<String, String> {
        let Some(index) = self.active_index() else {
            let message = "Apply LUT requires an open image".to_string();
            self.status = message.clone();
            return Err(message);
        };
        let pixel_maximum = match self.tabs[index].dataset.metadata.pixel_type {
            PixelType::U8 => 255.0,
            PixelType::U16 => 65_535.0,
            PixelType::F32 => {
                let message = "Apply LUT does not support 32-bit images".to_string();
                self.status = message.clone();
                return Err(message);
            }
        };
        let (minimum, maximum) = self.tabs[index].display_range();
        if !minimum.is_finite() || !maximum.is_finite() || maximum < minimum {
            let message = "Choose a valid display range before applying the LUT".to_string();
            self.status = message.clone();
            return Err(message);
        }
        if minimum == 0.0 && maximum == pixel_maximum {
            let message =
                "The display range is unchanged; adjust Brightness/Contrast before Apply LUT"
                    .to_string();
            self.status = message.clone();
            return Err(message);
        }

        let current = self.tabs[index].dataset.clone();
        let mut output = current.as_ref().clone();
        let map_value = |value: f32| apply_lut_sample(value, minimum, maximum, pixel_maximum);
        let (left, top, width, height) = selection_bounds(&self.tabs[index]);
        let selected_pixels = self.tabs[index].roi.as_ref().map(|selection| {
            roi_sample_pixels(selection, self.tabs[index].width, self.tabs[index].height)
                .into_iter()
                .collect::<HashSet<_>>()
        });
        if selected_pixels.as_ref().is_some_and(HashSet::is_empty) {
            let message = "The active selection contains no image pixels".to_string();
            self.status = message.clone();
            return Err(message);
        }
        let channels = if dataset_is_true_rgb(self.tabs[index].dataset.as_ref()) {
            (0..3).collect::<Vec<_>>()
        } else {
            vec![self.tabs[index].channel]
        };
        let has_stack = self.tabs[index].slices > 1 || self.tabs[index].frames > 1;
        let scope = if requested_scope == ApplyLutScope::Stack && has_stack {
            ApplyLutScope::Stack
        } else {
            ApplyLutScope::Slice
        };
        let z_planes = if scope == ApplyLutScope::Stack {
            (0..self.tabs[index].slices).collect::<Vec<_>>()
        } else {
            vec![self.tabs[index].z]
        };
        let time_points = if scope == ApplyLutScope::Stack {
            (0..self.tabs[index].frames).collect::<Vec<_>>()
        } else {
            vec![self.tabs[index].t]
        };
        let mut changed_samples = 0usize;
        for t in time_points {
            for z in &z_planes {
                for channel in &channels {
                    for y in top..top.saturating_add(height) {
                        for x in left..left.saturating_add(width) {
                            if selected_pixels
                                .as_ref()
                                .is_some_and(|pixels| !pixels.contains(&(x, y)))
                            {
                                continue;
                            }
                            if let Some(value) =
                                sample_dataset(current.as_ref(), x, y, *z, t, *channel)
                            {
                                set_dataset_sample(
                                    &mut output,
                                    x,
                                    y,
                                    *z,
                                    t,
                                    *channel,
                                    map_value(value),
                                );
                                changed_samples = changed_samples.saturating_add(1);
                            }
                        }
                    }
                }
            }
        }
        if changed_samples == 0 {
            let message = "Apply LUT found no pixels in the requested scope".to_string();
            self.status = message.clone();
            return Err(message);
        }

        let z = self.tabs[index].z;
        let t = self.tabs[index].t;
        let channel = self.tabs[index].channel;
        let lut = self.tabs[index].lut;
        let lut_inverted = self.tabs[index].lut_inverted;
        let rendered = render_dataset_plane(
            &output,
            z,
            t,
            channel,
            lut,
            lut_inverted,
            0.0,
            pixel_maximum,
        )
        .inspect_err(|error| {
            self.status = error.clone();
        })?;

        self.tabs[index].undo.push(current);
        self.tabs[index].redo.clear();
        self.tabs[index].dataset = Arc::new(output);
        self.tabs[index].revision = self.tabs[index].revision.saturating_add(1);
        self.tabs[index].dirty = true;
        self.tabs[index].set_display_range(0.0, pixel_maximum);
        self.tabs[index].render_image = rendered.image;
        self.tabs[index].width = rendered.width;
        self.tabs[index].height = rendered.height;
        let message = if scope == ApplyLutScope::Stack {
            format!(
                "Applied the display range to all {} stack plane(s)",
                self.tabs[index]
                    .slices
                    .saturating_mul(self.tabs[index].frames)
            )
        } else {
            if has_stack {
                "Applied the display range to the current stack plane".into()
            } else {
                "Applied the display range to image pixels".into()
            }
        };
        self.status = message.clone();
        let params = match scope {
            ApplyLutScope::Slice => json!({ "slice": true }),
            ApplyLutScope::Stack => json!({ "stack": true }),
        };
        self.record_command("image.lookup.apply_lut", Some(&params));
        Ok(message)
    }

    fn open_single_text_dialog(
        &mut self,
        command_id: &str,
        title: &str,
        label: &str,
        value: String,
    ) {
        let Some(tab_id) = self.active_tab else {
            self.status = "This command requires an open image".into();
            return;
        };
        self.dialog = Some(DialogState::Operation {
            command_id: command_id.into(),
            target_tab_id: Some(tab_id),
            title: title.into(),
            fields: vec![ParameterField {
                key: "title".into(),
                label: label.into(),
                value,
                kind: ParameterKind::Text,
            }],
            focused: 0,
        });
    }

    fn create_blank_image_with_size(
        &mut self,
        title: String,
        width: usize,
        height: usize,
        slices: usize,
        channels: usize,
        frames: usize,
        pixel_type: PixelType,
        fill: f32,
        cx: &mut Context<Self>,
    ) {
        let width = width.clamp(1, 32_768);
        let height = height.clamp(1, 32_768);
        let slices = slices.clamp(1, 4_096);
        let channels = channels.clamp(1, 64);
        let frames = frames.clamp(1, 4_096);
        let element_count = width
            .checked_mul(height)
            .and_then(|count| count.checked_mul(slices))
            .and_then(|count| count.checked_mul(channels))
            .and_then(|count| count.checked_mul(frames));
        if element_count.is_none_or(|count| count > 134_217_728) {
            self.status = "New image would exceed the 512 MiB safety limit".into();
            return;
        }
        let mut shape = Vec::new();
        let mut dims = Vec::new();
        if frames > 1 {
            shape.push(frames);
            dims.push(Dim::new(AxisKind::Time, frames));
        }
        if slices > 1 {
            shape.push(slices);
            dims.push(Dim::new(AxisKind::Z, slices));
        }
        shape.extend([height, width]);
        dims.extend([Dim::new(AxisKind::Y, height), Dim::new(AxisKind::X, width)]);
        if channels > 1 {
            shape.push(channels);
            dims.push(Dim::new(AxisKind::Channel, channels));
        }
        let data = Array::from_elem(IxDyn(&shape), fill);
        let metadata = Metadata {
            dims,
            pixel_type,
            ..Metadata::default()
        };
        let dataset = Dataset::new(data, metadata).expect("valid blank image shape");
        self.next_tab_id = self.next_tab_id.saturating_add(1);
        match ImageTab::from_dataset(
            self.next_tab_id,
            None,
            if title.trim().is_empty() {
                format!("Untitled-{}", self.next_tab_id)
            } else {
                title
            },
            dataset,
        ) {
            Ok(mut tab) => {
                tab.dirty = true;
                let id = tab.id;
                self.tabs.push(tab);
                if self.open_viewer_window(id, cx) {
                    self.status = format!(
                        "Created {width}x{height}, C:{channels} Z:{slices} T:{frames}, {pixel_type:?}"
                    );
                } else {
                    self.rollback_unopened_tab(id);
                }
            }
            Err(error) => self.status = error,
        }
    }

    fn request_close(&mut self, tab_id: u64, cx: &mut Context<Self>) {
        self.request_close_then(tab_id, CloseContinuation::None, cx);
    }

    fn request_close_then(
        &mut self,
        tab_id: u64,
        continuation: CloseContinuation,
        cx: &mut Context<Self>,
    ) {
        if self.dialog.is_some() {
            self.status =
                "Finish or cancel the current dialog before closing another window".into();
            if let Some(handle) = self.dialog_window {
                let _ = handle.update(cx, |_, window, _| window.activate_window());
            }
            return;
        }
        let Some((dirty, title)) = self.tab(tab_id).map(|tab| (tab.dirty, tab.title.clone()))
        else {
            self.status = "No image is open".into();
            return;
        };
        self.focus_viewer(tab_id, cx);
        if dirty {
            self.dialog = Some(DialogState::ConfirmClose {
                tab_id,
                title: title.clone(),
                continuation,
            });
            self.status = format!("Save changes to {title} before closing?");
            let app = cx.entity().downgrade();
            cx.defer(move |cx| {
                if let Some(app) = app.upgrade() {
                    let _ = app.update(cx, |app, cx| app.open_dialog_window(cx));
                }
            });
            return;
        }
        self.close_tab(tab_id, cx);
        self.continue_close(continuation, cx);
    }

    fn close_tab(&mut self, tab_id: u64, cx: &mut Context<Self>) {
        let Some(index) = self.tabs.iter().position(|tab| tab.id == tab_id) else {
            return;
        };
        if let Some(operation) = self.active_operations.remove(&tab_id) {
            operation.cancellation.cancel();
        }
        let closed = self.tabs.remove(index);
        self.last_pointer.remove(&tab_id);
        if self.roi_manager_show_all_target == Some(tab_id) {
            self.roi_manager_show_all_target = None;
        }
        self.activation_order
            .retain(|candidate| *candidate != tab_id);
        if let Some(handle) = self.viewer_handles.remove(&closed.id) {
            self.viewer_windows.remove(&handle.window_id());
            let _ = handle.update(cx, |_, window, _| window.remove_window());
        }
        self.active_tab = self.activation_order.last().copied();
        if let Some(active_tab) = self.active_tab {
            self.focus_viewer(active_tab, cx);
        }
        self.status = format!("Closed {}", closed.title);
    }

    fn close_active(&mut self, cx: &mut Context<Self>) {
        let Some(tab_id) = self.active_tab else {
            self.status = "No image is open".into();
            return;
        };
        self.request_close(tab_id, cx);
    }

    fn confirm_close(
        &mut self,
        tab_id: u64,
        continuation: CloseContinuation,
        save: bool,
        cx: &mut Context<Self>,
    ) {
        if save && !self.save_tab(tab_id, false) {
            return;
        }
        self.dismiss_dialog_window(cx);
        self.close_tab(tab_id, cx);
        if continuation != CloseContinuation::None {
            let app = cx.entity().downgrade();
            cx.defer(move |cx| {
                if let Some(app) = app.upgrade() {
                    let _ = app.update(cx, |app, cx| app.continue_close(continuation, cx));
                }
            });
        }
    }

    fn continue_close(&mut self, continuation: CloseContinuation, cx: &mut Context<Self>) {
        match continuation {
            CloseContinuation::None => {}
            CloseContinuation::CloseAll => self.close_all(false, cx),
            CloseContinuation::Quit => self.close_all(true, cx),
        }
    }

    fn close_all(&mut self, quit_after: bool, cx: &mut Context<Self>) {
        while let Some(tab_id) = self.tabs.last().map(|tab| tab.id) {
            if self.tab(tab_id).is_some_and(|tab| tab.dirty) {
                self.request_close_then(
                    tab_id,
                    if quit_after {
                        CloseContinuation::Quit
                    } else {
                        CloseContinuation::CloseAll
                    },
                    cx,
                );
                return;
            }
            self.close_tab(tab_id, cx);
        }
        if quit_after {
            cx.quit();
        } else {
            self.status = "Closed all images".into();
        }
    }

    fn request_quit(&mut self, cx: &mut Context<Self>) {
        self.macro_run = None;
        for operation in self.active_operations.values_mut() {
            operation.cancellation.cancel();
            operation.message = "Cancelling operation…".into();
        }
        self.close_all(true, cx);
    }

    fn save_active(&mut self) -> bool {
        let Some(tab_id) = self.active_tab else {
            self.status = "No image is open".into();
            return false;
        };
        self.save_tab(tab_id, false)
    }

    fn save_active_as(&mut self) -> bool {
        let Some(tab_id) = self.active_tab else {
            self.status = "No image is open".into();
            return false;
        };
        self.save_tab(tab_id, true)
    }

    fn save_tab(&mut self, tab_id: u64, force_picker: bool) -> bool {
        let Some(index) = self.tabs.iter().position(|tab| tab.id == tab_id) else {
            self.status = "No image is open".into();
            return false;
        };
        let path = if !force_picker {
            self.tabs[index].path.clone()
        } else {
            None
        }
        .or_else(|| {
            FileDialog::new()
                .set_file_name(self.tabs[index].title.clone())
                .add_filter("PNG image", &["png"])
                .add_filter("TIFF image", &["tif", "tiff"])
                .add_filter("JPEG image", &["jpg", "jpeg"])
                .save_file()
        });
        let Some(path) = path else {
            self.status = "Save canceled".into();
            return false;
        };
        self.save_tab_to_path(tab_id, path)
    }

    fn save_tab_to_path(&mut self, tab_id: u64, path: PathBuf) -> bool {
        let Some(index) = self.tabs.iter().position(|tab| tab.id == tab_id) else {
            self.status = "No image is open".into();
            return false;
        };
        let path = normalized_path_identity(&path);
        if let Some(owner_id) = other_path_owner(
            self.tabs
                .iter()
                .filter_map(|tab| tab.path.as_deref().map(|path| (tab.id, path))),
            tab_id,
            &path,
        ) {
            let owner = self
                .tab(owner_id)
                .map(|tab| format!("{} ({})", tab.title, tab.internal_label))
                .unwrap_or_else(|| format!("viewer-{owner_id}"));
            self.status = format!(
                "Cannot save {}: that path is already open in {owner}",
                path.display()
            );
            return false;
        }
        match write_dataset(&path, self.tabs[index].dataset.as_ref()) {
            Ok(()) => {
                self.tabs[index].path = Some(path.clone());
                self.tabs[index].title = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("Untitled")
                    .to_string();
                self.tabs[index].dirty = false;
                self.status = format!("Saved {}", path.display());
                true
            }
            Err(error) => {
                self.status = format!("Save failed: {error}");
                false
            }
        }
    }

    fn show_image_info(&mut self) {
        let Some(tab) = self.active_tab() else {
            self.status = "No image is open".into();
            return;
        };
        let (min, max) = tab.dataset.min_max().unwrap_or((0.0, 0.0));
        let source = tab
            .path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<memory>".into());
        self.dialog = Some(DialogState::ImageInfo {
            title: tab.title.clone(),
            lines: vec![
                tab.info_text(),
                format!("Dimensions: {:?}", tab.dataset.shape()),
                format!("Pixel type: {:?}", tab.dataset.metadata.pixel_type),
                format!("Display range: {min:.4} – {max:.4}"),
                format!("Source: {source}"),
                format!("Session: {}", tab.internal_label),
            ],
        });
    }

    fn zoom(&mut self, factor: f32) {
        let Some(tab) = self.active_tab_mut() else {
            self.status = "No image is open".into();
            return;
        };
        tab.scale_to_fit = false;
        tab.zoom = (tab.zoom * factor).clamp(1.0 / 72.0, 32.0);
        let zoom = tab.zoom;
        self.status = format!("Magnification: {:.0}%", zoom * 100.0);
    }

    fn zoom_actual(&mut self) {
        let Some(tab) = self.active_tab_mut() else {
            return;
        };
        tab.scale_to_fit = false;
        tab.zoom = 1.0;
        self.status = "Magnification: 100%".into();
    }

    fn zoom_fit(&mut self) {
        let Some(tab) = self.active_tab_mut() else {
            return;
        };
        tab.scale_to_fit = true;
        self.status = "Scale to fit".into();
    }

    fn cycle_tab(&mut self, delta: isize, cx: &mut Context<Self>) {
        if self.tabs.is_empty() {
            return;
        }
        let current = self.active_index().unwrap_or(0) as isize;
        let len = self.tabs.len() as isize;
        let next = (current + delta).rem_euclid(len) as usize;
        let id = self.tabs[next].id;
        self.focus_viewer(id, cx);
        self.status = format!("Active image: {}", self.tabs[next].title);
    }

    fn undo(&mut self) {
        let Some(tab) = self.active_tab_mut() else {
            return;
        };
        let Some(previous) = tab.undo.pop() else {
            self.status = "Nothing to undo".into();
            return;
        };
        tab.redo.push(tab.dataset.clone());
        tab.dataset = previous;
        tab.revision = tab.revision.saturating_add(1);
        tab.reset_display_ranges();
        tab.dirty = true;
        if let Err(error) = tab.refresh_render_image() {
            self.status = error;
        } else {
            self.status = "Undo".into();
        }
    }

    fn redo(&mut self) {
        let Some(tab) = self.active_tab_mut() else {
            return;
        };
        let Some(next) = tab.redo.pop() else {
            self.status = "Nothing to redo".into();
            return;
        };
        tab.undo.push(tab.dataset.clone());
        tab.dataset = next;
        tab.revision = tab.revision.saturating_add(1);
        tab.reset_display_ranges();
        tab.dirty = true;
        if let Err(error) = tab.refresh_render_image() {
            self.status = error;
        } else {
            self.status = "Redo".into();
        }
    }

    fn begin_operation(&mut self, command_id: &str, cx: &mut Context<Self>) {
        if operation_for_command(command_id).is_none() {
            self.run_operation_with_params(command_id, None);
            return;
        }
        let title = command_label(command_id);
        let mut defaults = command_registry::merge_params(command_id, None);
        if let Some(tab) = self.active_tab() {
            match command_id {
                "image.adjust.size" => overlay_json_objects(
                    &mut defaults,
                    json!({
                        "width": tab.width,
                        "height": tab.height,
                        "average_when_downsizing": true,
                        "interpolation": "Bilinear"
                    }),
                ),
                "image.adjust.canvas" => overlay_json_objects(
                    &mut defaults,
                    json!({
                        "width": tab.width,
                        "height": tab.height,
                        "position": "center",
                        "fill": 0.0
                    }),
                ),
                "image.adjust.threshold" => {
                    let (minimum, maximum) = tab.dataset.min_max().unwrap_or((0.0, 255.0));
                    overlay_json_objects(
                        &mut defaults,
                        json!({ "lower": minimum, "upper": maximum }),
                    );
                }
                "analyze.analyze_particles" => {
                    let (_, maximum) = tab.dataset.min_max().unwrap_or((0.0, 255.0));
                    let spacing = |axis| {
                        tab.dataset
                            .axis_index(axis)
                            .and_then(|index| tab.dataset.metadata.dims.get(index))
                            .and_then(|dim| dim.spacing)
                            .unwrap_or(1.0) as f64
                    };
                    let max_size = tab.width as f64
                        * tab.height as f64
                        * spacing(AxisKind::X)
                        * spacing(AxisKind::Y);
                    overlay_json_objects(
                        &mut defaults,
                        json!({
                            "max_threshold": maximum,
                            "max_size": max_size
                        }),
                    );
                }
                "analyze.plot_profile" => {
                    if let Some(selection) = tab.roi.as_ref()
                        && selection.points.len() >= 2
                    {
                        let first = selection.points[0];
                        let last = *selection.points.last().unwrap_or(&first);
                        if matches!(selection.tool, ToolId::Line | ToolId::Angle | ToolId::Free) {
                            overlay_json_objects(
                                &mut defaults,
                                json!({
                                    "x0": first.0,
                                    "y0": first.1,
                                    "x1": last.0,
                                    "y1": last.1
                                }),
                            );
                        } else {
                            overlay_json_objects(
                                &mut defaults,
                                json!({
                                    "left": first.0.min(last.0),
                                    "top": first.1.min(last.1),
                                    "width": (last.0 - first.0).abs().max(1.0),
                                    "height": (last.1 - first.1).abs().max(1.0)
                                }),
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        let mut fields = parameter_fields(&defaults);
        let offers_stack = operation_for_command(command_id)
            .and_then(|(operation, _)| self.ops_service.describe(operation))
            .is_some_and(|descriptor| {
                descriptor.supports(OperationScope::ZStack)
                    && self.active_tab().is_some_and(|tab| tab.slices > 1)
            });
        if offers_stack {
            fields.push(ParameterField {
                key: PROCESS_STACK_PARAMETER.into(),
                label: "Process stack".into(),
                value: "false".into(),
                kind: ParameterKind::Boolean,
            });
        }
        if ((title.ends_with("...") || title.ends_with('…')) || offers_stack) && !fields.is_empty()
        {
            self.dialog = Some(DialogState::Operation {
                command_id: command_id.to_string(),
                target_tab_id: self.active_tab,
                title,
                fields,
                focused: 0,
            });
            self.status = "Adjust parameters, then choose OK".into();
        } else {
            self.start_operation(command_id, Some(defaults), cx);
        }
    }

    fn apply_operation_dialog(&mut self, cx: &mut Context<Self>) {
        let Some(DialogState::Operation {
            command_id,
            target_tab_id,
            fields,
            ..
        }) = self.dialog.clone()
        else {
            return;
        };
        let mut params = serde_json::Map::new();
        for field in fields {
            let value = match field.kind {
                ParameterKind::Boolean => Value::Bool(field.value == "true"),
                ParameterKind::Number => match field.value.parse::<f64>() {
                    Ok(number) if number.is_finite() => Value::Number(
                        serde_json::Number::from_f64(number).expect("finite JSON number"),
                    ),
                    _ => {
                        self.status = format!("{} must be a number", field.label);
                        return;
                    }
                },
                ParameterKind::Text => Value::String(field.value),
                ParameterKind::Json => match serde_json::from_str(&field.value) {
                    Ok(value) => value,
                    Err(error) => {
                        self.status = format!("{} must be valid JSON: {error}", field.label);
                        return;
                    }
                },
            };
            params.insert(field.key, value);
        }
        self.dismiss_dialog_window(cx);
        if command_id == "__new_image" {
            let number = |key: &str, fallback: usize| {
                params
                    .get(key)
                    .and_then(Value::as_f64)
                    .map(|value| value.round().max(1.0) as usize)
                    .unwrap_or(fallback)
            };
            let title = params
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("Untitled")
                .to_string();
            let pixel_type = parse_pixel_type(
                params
                    .get("pixel_type")
                    .and_then(Value::as_str)
                    .unwrap_or("u8"),
            );
            self.create_blank_image_with_size(
                title,
                number("width", 512),
                number("height", 512),
                number("slices", 1),
                number("channels", 1),
                number("frames", 1),
                pixel_type,
                params.get("fill").and_then(Value::as_f64).unwrap_or(0.0) as f32,
                cx,
            );
            return;
        }
        if let Some(tab_id) = target_tab_id {
            if self.tab(tab_id).is_none() {
                self.status = "The image that opened this dialog has been closed".into();
                return;
            }
            self.focus_viewer(tab_id, cx);
        }
        if command_id == "__measurement_settings" {
            self.measurement_settings = measurement_settings_from_params(
                &Value::Object(params.clone()),
                self.measurement_settings,
                true,
            );
            let recorded = Value::Object(params);
            self.record_command("analyze.set_measurements", Some(&recorded));
            self.status = "Measurement settings updated for all image windows".into();
            return;
        }
        if command_id == "__apply_lut" {
            let scope = if params.get("stack").and_then(Value::as_bool) == Some(true) {
                ApplyLutScope::Stack
            } else {
                ApplyLutScope::Slice
            };
            let _ = self.apply_lut_to_pixels(scope);
            return;
        }
        if command_id == "__display_range" {
            let minimum = params.get("minimum").and_then(Value::as_f64).unwrap_or(0.0) as f32;
            let maximum = params
                .get("maximum")
                .and_then(Value::as_f64)
                .unwrap_or(255.0) as f32;
            let _ = self.apply_display_range(minimum, maximum, true);
            return;
        }
        if command_id == "__window_level" {
            let window_width = params.get("window").and_then(Value::as_f64).unwrap_or(0.0) as f32;
            let level = params.get("level").and_then(Value::as_f64).unwrap_or(0.0) as f32;
            let _ = self.apply_display_range(
                level - window_width * 0.5,
                level + window_width * 0.5,
                true,
            );
            return;
        }
        if command_id == "__rename" {
            let title = params
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim();
            if title.is_empty() {
                self.status = "Image title cannot be empty".into();
                return;
            }
            if let Some(tab) = self.active_tab_mut() {
                tab.title = title.to_string();
                tab.dirty = true;
                self.status = format!("Renamed image to {title}");
            }
            return;
        }
        if command_id == "__duplicate" {
            let title = params
                .get("title")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|title| !title.is_empty())
                .map(str::to_string);
            self.duplicate_active_as(title, cx);
            return;
        }
        self.start_operation(&command_id, Some(Value::Object(params)), cx);
    }

    fn run_operation(&mut self, command_id: &str, cx: &mut Context<Self>) {
        self.start_operation(command_id, None, cx);
    }

    fn start_operation(
        &mut self,
        command_id: &str,
        overrides: Option<Value>,
        cx: &mut Context<Self>,
    ) {
        // Analyze Particles still uses the legacy active-plane measurement adapter. It does not
        // change pixels and will move onto the shared plane lifecycle when native measurement
        // aggregation is represented by the scoped adapter.
        if command_id == "analyze.analyze_particles" {
            self.run_operation_with_params(command_id, overrides);
            if self.results_window_pending {
                self.open_results_window(cx);
            }
            return;
        }

        let Some((operation, mut params)) = operation_for_command(command_id) else {
            self.run_operation_with_params(command_id, overrides);
            return;
        };
        let Some(index) = self.active_index() else {
            self.status = "This command requires an open image".into();
            return;
        };
        if self.active_operations.contains_key(&self.tabs[index].id) {
            self.status =
                "An operation is already running for this image; press Escape to cancel it".into();
            return;
        }

        let defaults = command_registry::merge_params(command_id, None);
        merge_json_objects(&mut params, defaults);
        if let Some(overrides) = overrides {
            overlay_json_objects(&mut params, overrides);
        }
        let process_stack = take_process_stack_parameter(&mut params);
        let mut repeat_params = params.clone();
        if process_stack && let Some(values) = repeat_params.as_object_mut() {
            values.insert(PROCESS_STACK_PARAMETER.into(), Value::Bool(true));
        }
        let Some(descriptor) = self.ops_service.describe(operation) else {
            self.status = format!(
                "{} failed: unknown operation {operation}",
                command_label(command_id)
            );
            return;
        };
        let Some(scope) = choose_operation_scope(&descriptor, process_stack) else {
            self.status = if process_stack {
                format!(
                    "{} does not support processing the active Z stack",
                    command_label(command_id)
                )
            } else {
                format!(
                    "{} does not expose a compatible execution scope",
                    command_label(command_id)
                )
            };
            return;
        };
        let tab = &self.tabs[index];
        let area_mask = if scope.is_plane_wise() {
            match rasterize_processing_area(tab.roi.as_ref(), tab.width, tab.height).and_then(
                |mask| {
                    mask.map(|mask| {
                        AreaMask::new(mask.left, mask.top, mask.width, mask.height, mask.members)
                            .map_err(|error| error.to_string())
                    })
                    .transpose()
                },
            ) {
                Ok(mask) => mask,
                Err(error) => {
                    self.status = format!("{} failed: {error}", command_label(command_id));
                    return;
                }
            }
        } else {
            None
        };
        let tab_id = tab.id;
        let revision = tab.revision;
        let input = tab.dataset.clone();
        let request = InvocationRequest {
            operation: operation.to_string(),
            input: input.clone(),
            parameters: params.clone(),
            scope,
            active: PlanePosition {
                channel: tab.channel,
                z: tab.z,
                time: tab.t,
            },
            area_mask,
        };
        let cancellation = CancellationToken::default();
        let latest_progress = Arc::new(Mutex::new(None));
        let control = ExecutionControl::new(
            cancellation.clone(),
            Arc::new(UiProgressSink(latest_progress.clone())),
        );
        let operation_message = format!("{} running…", command_label(command_id));
        let job_id = self.next_operation_job_id;
        let Some(next_operation_job_id) = job_id.checked_add(1) else {
            self.status = "Cannot start another operation: job identifier space exhausted".into();
            return;
        };
        self.next_operation_job_id = next_operation_job_id;
        self.active_operations.insert(
            tab_id,
            ActiveOperation {
                job_id,
                revision,
                input: input.clone(),
                cancellation,
                progress: 0.05,
                message: operation_message.clone(),
            },
        );
        self.status = operation_message;

        let progress_input = input.clone();
        let progress_command = command_id.to_string();
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(50))
                    .await;
                let event = latest_progress
                    .lock()
                    .ok()
                    .and_then(|mut latest| latest.take());
                let still_running = this
                    .update(cx, |app, cx| {
                        let Some(active) = app.active_operations.get_mut(&tab_id) else {
                            return false;
                        };
                        let current =
                            active_operation_matches(active, job_id, revision, &progress_input);
                        if current
                            && !active.cancellation.is_cancelled()
                            && let Some(event) = event
                        {
                            active.progress = if event.total_planes == 0 {
                                0.0
                            } else {
                                event.completed_planes as f32 / event.total_planes as f32
                            };
                            if let Some(message) = event
                                .detail
                                .as_ref()
                                .and_then(|detail| detail.message.as_deref())
                            {
                                active.message =
                                    format!("{}: {message}", command_label(&progress_command));
                            }
                            cx.notify();
                        }
                        current
                    })
                    .unwrap_or(false);
                if !still_running {
                    break;
                }
            }
        })
        .detach();

        let ops_service = self.ops_service.clone();
        let command_id = command_id.to_string();
        cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move { ops_service.invoke(request, &control) })
                .await;
            this.update(cx, |app, cx| {
                app.finish_operation(
                    tab_id,
                    job_id,
                    revision,
                    input,
                    command_id,
                    repeat_params,
                    result,
                    cx,
                );
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn finish_operation(
        &mut self,
        tab_id: u64,
        job_id: u64,
        revision: u64,
        input: Arc<DatasetF32>,
        command_id: String,
        repeat_params: Value,
        result: crate::runtime::Result<InvocationResult>,
        cx: &mut Context<Self>,
    ) {
        let outcome = self.finish_operation_result(
            tab_id,
            job_id,
            revision,
            input,
            &command_id,
            repeat_params,
            result,
            cx,
        );
        let resumes_macro = self
            .macro_run
            .as_ref()
            .is_some_and(|run| run.awaiting_job_id == Some(job_id));
        if resumes_macro {
            if let Some(run) = self.macro_run.as_mut() {
                run.awaiting_job_id = None;
            }
            self.record_macro_step(&command_id, outcome);
            self.continue_macro_run(cx);
        }
    }

    fn finish_operation_result(
        &mut self,
        tab_id: u64,
        job_id: u64,
        revision: u64,
        input: Arc<DatasetF32>,
        command_id: &str,
        repeat_params: Value,
        result: crate::runtime::Result<InvocationResult>,
        cx: &mut Context<Self>,
    ) -> Result<String, String> {
        let Some(active) = self.active_operations.get(&tab_id) else {
            return Err(format!(
                "{} result discarded because its viewer job is no longer current",
                command_label(command_id)
            ));
        };
        if !active_operation_matches(active, job_id, revision, &input) {
            return Err(format!(
                "{} result discarded because its viewer job changed",
                command_label(command_id)
            ));
        }
        let cancelled = active.cancellation.is_cancelled();
        self.active_operations.remove(&tab_id);
        if cancelled {
            self.status = format!("{} canceled", command_label(command_id));
            return Err(self.status.clone());
        }

        let result = match result {
            Ok(result) => result,
            Err(crate::runtime::AppError::Ops(crate::commands::OpsError::Cancelled)) => {
                self.status = format!("{} canceled", command_label(command_id));
                return Err(self.status.clone());
            }
            Err(error) => {
                self.status = format!("{} failed: {error}", command_label(command_id));
                return Err(self.status.clone());
            }
        };
        let Some(index) = self.tabs.iter().position(|tab| tab.id == tab_id) else {
            return Err(format!(
                "{} result discarded because its viewer closed",
                command_label(command_id)
            ));
        };
        if self.tabs[index].revision != revision || !Arc::ptr_eq(&self.tabs[index].dataset, &input)
        {
            self.status = format!(
                "{} result discarded because the source image changed",
                command_label(command_id)
            );
            return Err(self.status.clone());
        }

        let InvocationResult {
            dataset_effect,
            measurements,
            status,
        } = result;
        if let DatasetEffect::Replaced { before, after } = dataset_effect {
            if !Arc::ptr_eq(&before, &input) {
                self.status = format!(
                    "{} failed: replacement did not reference its source image",
                    command_label(command_id)
                );
                return Err(self.status.clone());
            }
            let channels = axis_len(after.as_ref(), AxisKind::Channel);
            let slices = axis_len(after.as_ref(), AxisKind::Z);
            let frames = axis_len(after.as_ref(), AxisKind::Time);
            let z = self.tabs[index].z.min(slices.saturating_sub(1));
            let t = self.tabs[index].t.min(frames.saturating_sub(1));
            let channel = self.tabs[index].channel.min(channels.saturating_sub(1));
            let display_ranges = default_display_ranges(after.as_ref(), z, t);
            let display_index = if dataset_is_true_rgb(after.as_ref()) || display_ranges.len() <= 1
            {
                0
            } else {
                channel.min(display_ranges.len().saturating_sub(1))
            };
            let (display_min, display_max) = display_ranges
                .get(display_index)
                .copied()
                .unwrap_or((0.0, 255.0));
            let rendered = match render_dataset_plane(
                after.as_ref(),
                z,
                t,
                channel,
                self.tabs[index].lut,
                self.tabs[index].lut_inverted,
                display_min,
                display_max,
            ) {
                Ok(rendered) => rendered,
                Err(error) => {
                    self.status = format!("Render failed: {error}");
                    return Err(self.status.clone());
                }
            };
            let tab = &mut self.tabs[index];
            tab.undo.push(before);
            tab.redo.clear();
            tab.dataset = after;
            tab.revision = tab.revision.saturating_add(1);
            tab.render_image = rendered.image;
            tab.width = rendered.width;
            tab.height = rendered.height;
            tab.channels = channels;
            tab.slices = slices;
            tab.frames = frames;
            tab.z = z;
            tab.t = t;
            tab.channel = channel;
            tab.display_ranges = display_ranges;
            tab.dirty = true;
        }

        let image_title = self.tabs[index].title.clone();
        let label = command_label(command_id);
        self.status = status.unwrap_or_else(|| format!("{label} complete"));
        self.last_repeatable_command = Some((command_id.to_string(), repeat_params.clone()));
        self.record_command(command_id, Some(&repeat_params));
        if let Some(measurements) = measurements {
            let added = self.append_measurement_values(measurements.values, &image_title, &label);
            self.status = format!(
                "{} · {added} new row(s), {} total",
                self.status,
                self.results.len()
            );
        }
        if self.results_window_pending {
            self.open_results_window(cx);
        }
        Ok(self.status.clone())
    }

    fn run_operation_with_params(&mut self, command_id: &str, overrides: Option<Value>) {
        let Some((op, mut params)) = operation_for_command(command_id) else {
            let label = menu::manifest_commands()
                .iter()
                .find(|entry| entry.id == command_id)
                .map(|entry| entry.label.clone())
                .unwrap_or_else(|| command_id.to_string());
            self.dialog = Some(DialogState::Message {
                title: label,
                body: format!(
                    "“{command_id}” is reserved for ImageJ compatibility but is not backed by a native core operation yet."
                ),
            });
            self.status = format!("{command_id}: compatibility route unavailable");
            return;
        };
        let Some(index) = self.active_index() else {
            self.status = "This command requires an open image".into();
            return;
        };
        let defaults = command_registry::merge_params(command_id, None);
        merge_json_objects(&mut params, defaults);
        if let Some(overrides) = overrides {
            overlay_json_objects(&mut params, overrides);
        }
        if let Some(values) = params.as_object_mut() {
            values.remove(PROCESS_STACK_PARAMETER);
        }
        self.progress = Some(0.15);
        let current = self.tabs[index].dataset.clone();
        let plane_input = if command_id == "analyze.analyze_particles" {
            match active_plane_dataset(&self.tabs[index]) {
                Ok(dataset) => Some(dataset),
                Err(error) => {
                    self.status = error;
                    self.progress = None;
                    return;
                }
            }
        } else {
            None
        };
        let operation_input = plane_input.as_ref().unwrap_or(current.as_ref());
        match self.ops_service.execute(op, operation_input, &params) {
            Ok(output) => {
                let measurements = output.measurements;
                if measurements.is_none() {
                    self.tabs[index].undo.push(current);
                    self.tabs[index].redo.clear();
                    self.tabs[index].dataset = Arc::new(output.dataset);
                    self.tabs[index].revision = self.tabs[index].revision.saturating_add(1);
                    self.tabs[index].reset_display_ranges();
                    self.tabs[index].dirty = true;
                    self.tabs[index].channels =
                        axis_len(self.tabs[index].dataset.as_ref(), AxisKind::Channel);
                    self.tabs[index].slices =
                        axis_len(self.tabs[index].dataset.as_ref(), AxisKind::Z);
                    self.tabs[index].frames =
                        axis_len(self.tabs[index].dataset.as_ref(), AxisKind::Time);
                    self.tabs[index].z = self.tabs[index]
                        .z
                        .min(self.tabs[index].slices.saturating_sub(1));
                    self.tabs[index].t = self.tabs[index]
                        .t
                        .min(self.tabs[index].frames.saturating_sub(1));
                    self.tabs[index].channel = self.tabs[index]
                        .channel
                        .min(self.tabs[index].channels.saturating_sub(1));
                    if let Err(error) = self.tabs[index].refresh_render_image() {
                        self.status = format!("Render failed: {error}");
                        self.progress = None;
                        return;
                    }
                }
                self.status = format!("{} complete", command_label(command_id));
                self.last_repeatable_command = Some((command_id.to_string(), params.clone()));
                self.record_command(command_id, Some(&params));
                if let Some(measurements) = measurements {
                    let image_title = self.tabs[index].title.clone();
                    let command = command_label(command_id);
                    let added =
                        self.append_measurement_values(measurements.values, &image_title, &command);
                    self.status = format!(
                        "{} complete · {added} new row(s), {} total",
                        command,
                        self.results.len()
                    );
                }
            }
            Err(error) => self.status = format!("{} failed: {error}", command_label(command_id)),
        }
        self.progress = None;
    }

    fn dispatch_command(&mut self, command_id: &str, window: &mut Window, cx: &mut Context<Self>) {
        self.activate_session_for_window(window);
        self.close_menu_popup(cx);
        if operation_for_command(command_id).is_none()
            && !matches!(
                command_id,
                "image.adjust.brightness"
                    | "image.adjust.window_level"
                    | "image.lookup.apply_lut"
                    | "analyze.set_measurements"
            )
        {
            self.record_command(command_id, None);
        }
        match command_id {
            "file.new" => self.open_new_image_dialog(),
            "file.open" => self.pick_and_open(cx),
            "file.close" => self.close_active(cx),
            "file.close_all" => self.close_all(false, cx),
            "file.save" => {
                self.save_active();
            }
            "file.save_as" => {
                self.save_active_as();
            }
            "file.export.results" => self.export_results(),
            "file.revert" => self.revert_active(),
            "file.quit" => self.request_quit(cx),
            "edit.undo" => self.undo(),
            "edit.redo" => self.redo(),
            "edit.cut" => self.cut_active(),
            "edit.copy" => self.copy_active(),
            "edit.paste" => self.paste_clipboard(cx),
            "edit.internal_clipboard" => self.show_internal_clipboard(cx),
            "edit.selection.all" => self.select_all(),
            "edit.clear" => self.fill_selection(false),
            "edit.fill" => self.fill_selection(true),
            "edit.invert" => self.run_operation(command_id, cx),
            "image.show_info" | "image.properties" => self.show_image_info(),
            "image.adjust.brightness" => self.open_brightness_contrast_dialog(cx),
            "image.adjust.window_level" => self.open_window_level_dialog(cx),
            "image.duplicate" => {
                let value = self
                    .active_tab()
                    .map(|tab| format!("{} copy", tab.title))
                    .unwrap_or_else(|| "Untitled copy".into());
                self.open_single_text_dialog("__duplicate", "Duplicate", "Title", value);
            }
            "image.rename" => {
                let value = self
                    .active_tab()
                    .map(|tab| tab.title.clone())
                    .unwrap_or_default();
                self.open_single_text_dialog("__rename", "Rename", "Title", value);
            }
            "image.crop" => self.crop_to_selection(),
            "image.zoom.in" => self.zoom(1.5),
            "image.zoom.out" => self.zoom(1.0 / 1.5),
            "image.zoom.reset" | "image.zoom.original" | "image.zoom.view100" => self.zoom_actual(),
            "image.zoom.scale_to_fit" => self.zoom_fit(),
            "image.zoom.maximize" => self.maximize_active_viewer(cx),
            "image.lookup.invert_lut" | "image.color.invert_luts" => self.invert_lut(),
            "image.lookup.fire" => self.set_lut(LookupTable::Fire, "Fire"),
            "image.lookup.grays" => self.set_lut(LookupTable::Grays, "Grays"),
            "image.lookup.ice" => self.set_lut(LookupTable::Ice, "Ice"),
            "image.lookup.spectrum" => self.set_lut(LookupTable::Spectrum, "Spectrum"),
            "image.lookup.rgb332" => self.set_lut(LookupTable::Rgb332, "3-3-2 RGB"),
            "image.lookup.red" => self.set_lut(LookupTable::Red, "Red"),
            "image.lookup.green" => self.set_lut(LookupTable::Green, "Green"),
            "image.lookup.blue" => self.set_lut(LookupTable::Blue, "Blue"),
            "image.lookup.cyan" => self.set_lut(LookupTable::Cyan, "Cyan"),
            "image.lookup.magenta" => self.set_lut(LookupTable::Magenta, "Magenta"),
            "image.lookup.yellow" => self.set_lut(LookupTable::Yellow, "Yellow"),
            "image.lookup.red_green" => self.set_lut(LookupTable::RedGreen, "Red/Green"),
            "image.lookup.apply_lut" => self.open_apply_lut_dialog(),
            "image.overlay.add_selection" => self.add_selection_to_overlay(),
            "image.overlay.hide" => {
                if let Some(tab) = self.active_tab_mut() {
                    tab.overlays_hidden = true;
                    self.status = "Overlay hidden".into();
                }
            }
            "image.overlay.show" => {
                if let Some(tab) = self.active_tab_mut() {
                    tab.overlays_hidden = false;
                    self.status = "Overlay shown".into();
                }
            }
            "image.overlay.toggle" => {
                if let Some(tab) = self.active_tab_mut() {
                    tab.overlays_hidden = !tab.overlays_hidden;
                    self.status = if tab.overlays_hidden {
                        "Overlay hidden".into()
                    } else {
                        "Overlay shown".into()
                    };
                }
            }
            "image.overlay.remove" => {
                if let Some(tab) = self.active_tab_mut() {
                    tab.overlays.clear();
                    self.status = "Overlay removed".into();
                }
            }
            "image.overlay.list" => self.show_overlay_list(),
            "image.overlay.measure" => self.measure_overlays(),
            "image.overlay.from_roi_manager" => {
                let selections = self
                    .roi_manager
                    .iter()
                    .map(|roi| roi.selection.clone())
                    .collect::<Vec<_>>();
                if let Some(tab) = self.active_tab_mut() {
                    tab.overlays.extend(selections);
                    tab.overlays_hidden = false;
                    self.status = format!("Overlay now contains {} ROI(s)", tab.overlays.len());
                }
            }
            "image.overlay.to_roi_manager" => self.replace_roi_manager_from_overlay(),
            "analyze.measure" => self.measure_active_image(),
            "analyze.set_measurements" => self.open_measurement_settings_dialog(),
            "analyze.tools.roi_manager" => self.open_roi_manager_window(cx),
            "analyze.tools.results" => self.open_results_window(cx),
            "analyze.clear_results" => self.clear_results(),
            "analyze.summarize" => self.summarize_results(),
            "plugins.macros.run" => self.pick_and_run_macro(cx),
            "plugins.macros.install" => self.install_macro(),
            "plugins.macros.record" => {
                self.macro_recording = true;
                self.dialog = Some(DialogState::MacroRecorder);
                self.status = "Macro Recorder is running".into();
            }
            "plugins.utilities.startup" => self.run_startup_macro(cx),
            command if command.starts_with("plugins.macros.installed.") => {
                if let Ok(index) = command
                    .trim_start_matches("plugins.macros.installed.")
                    .parse::<usize>()
                    && let Some(path) = macros::list_installed_macro_files().get(index).cloned()
                {
                    self.run_macro_path(&path, cx);
                }
            }
            "window.next" | "window.put_behind" => self.cycle_tab(1, cx),
            "window.previous" => self.cycle_tab(-1, cx),
            "window.main" => {
                self.focus_launcher(cx);
                self.status = "Main window active".into();
            }
            "window.show_all" => self.show_all_windows(cx),
            command if command.starts_with("window.viewer.") => {
                if let Ok(id) = command.trim_start_matches("window.viewer.").parse::<u64>() {
                    let title = self.tab(id).map(|tab| tab.title.clone());
                    self.focus_viewer(id, cx);
                    if let Some(title) = title {
                        self.status = format!("{title} active");
                    }
                }
            }
            "window.cascade" | "window.tile" => {
                self.status = "Window arrangement is managed by the desktop".into();
            }
            "process.repeat_command" => {
                if let Some((command, params)) = self.last_repeatable_command.clone() {
                    self.start_operation(&command, Some(params), cx);
                } else {
                    self.status = "No command to repeat".into();
                }
            }
            "image.stacks.next" => self.step_stack(AxisKind::Z, 1),
            "image.stacks.previous" => self.step_stack(AxisKind::Z, -1),
            "image.stacks.measure_stack" => self.measure_active_stack(),
            "edit.selection.none" => self.clear_selection(),
            "help.about" => self.dialog = Some(DialogState::About),
            "help.shortcuts" => {
                self.dialog = Some(DialogState::ImageInfo {
                    title: "Keyboard Shortcuts".into(),
                    lines: vec![
                        "Ctrl/Cmd+O  Open image".into(),
                        "Ctrl/Cmd+S  Save image".into(),
                        "Ctrl/Cmd+Shift+S  Save As".into(),
                        "Ctrl/Cmd+W  Close image".into(),
                        "Ctrl/Cmd+Z  Undo · Ctrl/Cmd+Shift+Z  Redo".into(),
                        "+ / −  Zoom · 4 or 5  Actual size · Ctrl/Cmd+0  Fit".into(),
                        "R O G F L A P W T Z H D  Select ImageJ tools".into(),
                        "Ctrl+Tab / Ctrl+Shift+Tab  Cycle image windows".into(),
                    ],
                });
            }
            "help.docs" => {
                self.dialog = Some(DialogState::Message {
                    title: "ImageJ Documentation".into(),
                    body: "ImageJ reference documentation: https://imagej.net/ij/docs/\n\nimage-rs keeps the original command vocabulary while using native GPUI windows and a modern interface.".into(),
                });
            }
            command if command.starts_with("launcher.tool.") => {
                if let Some(tool) = tool_for_command(command) {
                    self.select_tool(tool);
                }
            }
            _ => self.begin_operation(command_id, cx),
        }
        if self.dialog.is_some() {
            let app = cx.entity().downgrade();
            cx.defer(move |cx| {
                if let Some(app) = app.upgrade() {
                    let _ = app.update(cx, |app, cx| app.open_dialog_window(cx));
                }
            });
        }
        if self.results_window_pending {
            self.open_results_window(cx);
        }
        cx.notify();
    }

    fn select_tool(&mut self, tool: ToolId) {
        self.selected_tool = tool;
        self.status = tool.label().to_string();
    }

    fn clear_selection(&mut self) {
        if let Some(tab) = self.active_tab_mut() {
            tab.roi = None;
            self.status = "Selection cleared".into();
        }
    }

    fn select_all(&mut self) {
        let Some(tab) = self.active_tab_mut() else {
            self.status = "No image is open".into();
            return;
        };
        tab.roi = Some(RoiSelection {
            tool: ToolId::Rect,
            points: vec![(0.0, 0.0), (tab.width as f32, tab.height as f32)],
        });
        self.status = "Selected the entire image".into();
    }

    fn fill_selection(&mut self, foreground: bool) {
        let Some(index) = self.active_index() else {
            self.status = "No image is open".into();
            return;
        };
        let (left, top, width, height) = selection_bounds(&self.tabs[index]);
        let value = if foreground {
            match self.tabs[index].dataset.metadata.pixel_type {
                PixelType::U8 => 255.0,
                PixelType::U16 => 65_535.0,
                PixelType::F32 => 1.0,
            }
        } else {
            0.0
        };
        let current = self.tabs[index].dataset.clone();
        let mut output = current.as_ref().clone();
        for y in top..top + height {
            for x in left..left + width {
                set_dataset_sample(
                    &mut output,
                    x,
                    y,
                    self.tabs[index].z,
                    self.tabs[index].t,
                    self.tabs[index].channel,
                    value,
                );
            }
        }
        self.tabs[index].undo.push(current);
        self.tabs[index].redo.clear();
        self.tabs[index].dataset = Arc::new(output);
        self.tabs[index].revision = self.tabs[index].revision.saturating_add(1);
        self.tabs[index].dirty = true;
        if let Err(error) = self.tabs[index].refresh_render_image() {
            self.status = error;
        } else {
            self.status = if foreground {
                format!("Filled {width}x{height} pixels")
            } else {
                format!("Cleared {width}x{height} pixels")
            };
        }
    }

    fn crop_to_selection(&mut self) {
        let Some(tab) = self.active_tab() else {
            self.status = "No image is open".into();
            return;
        };
        let Some(selection) = tab.roi.as_ref().filter(|roi| !roi.points.is_empty()) else {
            self.status = "Crop requires an active selection".into();
            return;
        };
        let min_x = selection
            .points
            .iter()
            .map(|point| point.0)
            .fold(f32::INFINITY, f32::min)
            .floor()
            .clamp(0.0, tab.width.saturating_sub(1) as f32) as usize;
        let min_y = selection
            .points
            .iter()
            .map(|point| point.1)
            .fold(f32::INFINITY, f32::min)
            .floor()
            .clamp(0.0, tab.height.saturating_sub(1) as f32) as usize;
        let max_x = selection
            .points
            .iter()
            .map(|point| point.0)
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil()
            .clamp((min_x + 1) as f32, tab.width as f32) as usize;
        let max_y = selection
            .points
            .iter()
            .map(|point| point.1)
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil()
            .clamp((min_y + 1) as f32, tab.height as f32) as usize;
        self.run_operation_with_params(
            "image.crop",
            Some(json!({
                "x": min_x,
                "y": min_y,
                "width": max_x - min_x,
                "height": max_y - min_y
            })),
        );
        if let Some(tab) = self.active_tab_mut() {
            tab.roi = None;
        }
    }

    fn step_stack(&mut self, axis: AxisKind, delta: isize) {
        let Some(tab) = self.active_tab_mut() else {
            self.status = "No image is open".into();
            return;
        };
        let (position, length, axis_name) = match axis {
            AxisKind::Channel => (&mut tab.channel, tab.channels, "channel"),
            AxisKind::Z => (&mut tab.z, tab.slices, "slice"),
            AxisKind::Time => (&mut tab.t, tab.frames, "frame"),
            _ => return,
        };
        *position =
            (*position as isize + delta).clamp(0, length.saturating_sub(1) as isize) as usize;
        let current = *position + 1;
        if let Err(error) = tab.refresh_render_image() {
            self.status = error;
        } else {
            self.status = format!("{} {current}/{length}", humanize_key(axis_name));
        }
    }

    fn viewer_geometry(window: &Window, tab: &ImageTab) -> ViewerGeometry {
        let bounds = window.bounds().size;
        let viewport_width = f32::from(bounds.width);
        let stack_height = if has_stack_controls(tab) {
            STACK_HEIGHT
        } else {
            0.0
        };
        let content_top = INFO_HEIGHT + stack_height;
        let viewport_height = (f32::from(bounds.height) - content_top - STATUS_HEIGHT).max(1.0);
        let fit_zoom = ((viewport_width - 18.0).max(1.0) / tab.width.max(1) as f32)
            .min((viewport_height - 18.0).max(1.0) / tab.height.max(1) as f32)
            .clamp(1.0 / 72.0, 32.0);
        let zoom = if tab.scale_to_fit { fit_zoom } else { tab.zoom };
        let display_width = tab.width as f32 * zoom;
        let display_height = tab.height as f32 * zoom;
        ViewerGeometry {
            zoom,
            image_left: (viewport_width - display_width) * 0.5,
            image_top: content_top + (viewport_height - display_height) * 0.5,
            display_width,
            display_height,
        }
    }

    fn image_position(
        &self,
        tab_id: u64,
        position: Point<Pixels>,
        window: &Window,
    ) -> Option<(f32, f32)> {
        let tab = self.tab(tab_id)?;
        let geometry = Self::viewer_geometry(window, tab);
        let local_x = f32::from(position.x) - geometry.image_left;
        let local_y = f32::from(position.y) - geometry.image_top;
        if local_x < 0.0
            || local_y < 0.0
            || local_x >= geometry.display_width
            || local_y >= geometry.display_height
        {
            return None;
        }
        Some((local_x / geometry.zoom, local_y / geometry.zoom))
    }

    fn begin_roi(&mut self, tab_id: u64, event: &MouseDownEvent, window: &Window) {
        self.activate_tab(tab_id);
        let Some(position) = self.image_position(tab_id, event.position, window) else {
            return;
        };
        let tool = self.selected_tool;
        let Some(tab) = self.tab_mut(tab_id) else {
            return;
        };
        match tool {
            ToolId::Rect | ToolId::Oval | ToolId::Line | ToolId::Angle | ToolId::Free => {
                tab.roi = Some(RoiSelection {
                    tool,
                    points: vec![position, position],
                });
                self.roi_drag = Some(RoiDrag { tab_id, tool });
                self.status = format!("{} selection", tool.label());
            }
            ToolId::Poly => {
                if let Some(selection) = tab.roi.as_mut().filter(|roi| roi.tool == ToolId::Poly) {
                    selection.points.push(position);
                } else {
                    tab.roi = Some(RoiSelection {
                        tool,
                        points: vec![position],
                    });
                }
                self.status = "Polygon selection: click to add vertices".into();
            }
            ToolId::Point | ToolId::Wand | ToolId::Text => {
                tab.roi = Some(RoiSelection {
                    tool,
                    points: vec![position],
                });
                self.status = format!(
                    "{} at x={:.0}, y={:.0}",
                    tool.label(),
                    position.0,
                    position.1
                );
            }
            ToolId::Zoom => self.zoom(1.5),
            ToolId::Hand | ToolId::Dropper | ToolId::More => {}
        }
    }

    fn update_roi(&mut self, tab_id: u64, event: &MouseMoveEvent, window: &Window) {
        let Some(position) = self.image_position(tab_id, event.position, window) else {
            self.update_pointer_status(tab_id, event, window);
            return;
        };
        let drag = self.roi_drag;
        if let Some(drag) = drag
            && tab_id == drag.tab_id
            && let Some(selection) = self.tab_mut(tab_id).and_then(|tab| tab.roi.as_mut())
        {
            if drag.tool == ToolId::Free {
                selection.points.push(position);
            } else if let Some(last) = selection.points.last_mut() {
                *last = position;
            }
        }
        self.update_pointer_status(tab_id, event, window);
    }

    fn end_roi(&mut self, tab_id: u64, event: &MouseUpEvent, window: &Window) {
        self.activate_tab(tab_id);
        if self.roi_drag.is_some_and(|drag| drag.tab_id == tab_id) {
            if let Some(position) = self.image_position(tab_id, event.position, window)
                && let Some(selection) = self.tab_mut(tab_id).and_then(|tab| tab.roi.as_mut())
                && selection.tool != ToolId::Free
                && let Some(last) = selection.points.last_mut()
            {
                *last = position;
            }
            self.roi_drag = None;
            if let Some(selection) = self.tab(tab_id).and_then(|tab| tab.roi.as_ref()) {
                self.status = roi_status(selection);
            }
        }
    }

    fn revert_active(&mut self) {
        let Some(index) = self.active_index() else {
            return;
        };
        let Some(path) = self.tabs[index].path.clone() else {
            self.status = "Untitled images cannot be reverted".into();
            return;
        };
        match read_dataset(&path) {
            Ok(dataset) => {
                let current = self.tabs[index].dataset.clone();
                self.tabs[index].undo.push(current);
                self.tabs[index].redo.clear();
                self.tabs[index].dataset = Arc::new(dataset);
                self.tabs[index].revision = self.tabs[index].revision.saturating_add(1);
                self.tabs[index].reset_display_ranges();
                self.tabs[index].dirty = false;
                if let Err(error) = self.tabs[index].refresh_render_image() {
                    self.status = error;
                } else {
                    self.status = format!("Reverted {}", self.tabs[index].title);
                }
            }
            Err(error) => self.status = format!("Revert failed: {error}"),
        }
    }

    fn duplicate_active_as(&mut self, title: Option<String>, cx: &mut Context<Self>) {
        let Some(source) = self.active_tab().cloned() else {
            return;
        };
        self.next_tab_id = self.next_tab_id.saturating_add(1);
        let mut duplicate = source;
        duplicate.id = self.next_tab_id;
        duplicate.revision = 0;
        duplicate.internal_label = format!("viewer-{}", duplicate.id);
        duplicate.title = title.unwrap_or_else(|| format!("{} copy", duplicate.title));
        duplicate.path = None;
        duplicate.dirty = true;
        duplicate.undo.clear();
        duplicate.redo.clear();
        let id = duplicate.id;
        self.tabs.push(duplicate);
        if self.open_viewer_window(id, cx) {
            self.status = "Image duplicated".into();
        } else {
            self.rollback_unopened_tab(id);
        }
    }

    fn copy_active(&mut self) {
        let Some(patch) = self.active_tab().map(clipboard_patch) else {
            self.status = "No image is open".into();
            return;
        };
        self.status = format!(
            "Copied {}x{} pixels to the internal clipboard",
            patch.width, patch.height
        );
        self.internal_clipboard = Some(patch);
    }

    fn cut_active(&mut self) {
        let Some(index) = self.active_index() else {
            self.status = "No image is open".into();
            return;
        };
        let patch = clipboard_patch(&self.tabs[index]);
        let (left, top, width, height) = selection_bounds(&self.tabs[index]);
        let current = self.tabs[index].dataset.clone();
        let mut cleared = current.as_ref().clone();
        for y in top..top.saturating_add(height) {
            for x in left..left.saturating_add(width) {
                set_dataset_sample(
                    &mut cleared,
                    x,
                    y,
                    self.tabs[index].z,
                    self.tabs[index].t,
                    self.tabs[index].channel,
                    0.0,
                );
            }
        }
        self.internal_clipboard = Some(patch);
        self.tabs[index].undo.push(current);
        self.tabs[index].redo.clear();
        self.tabs[index].dataset = Arc::new(cleared);
        self.tabs[index].revision = self.tabs[index].revision.saturating_add(1);
        self.tabs[index].dirty = true;
        if let Err(error) = self.tabs[index].refresh_render_image() {
            self.status = error;
        } else {
            self.status = format!("Cut {width}x{height} pixels to the internal clipboard");
        }
    }

    fn paste_clipboard(&mut self, cx: &mut Context<Self>) {
        let Some(patch) = self.internal_clipboard.clone() else {
            self.status = "The internal clipboard is empty".into();
            return;
        };
        if let Some(index) = self.active_index() {
            let (left, top) = self.tabs[index]
                .roi
                .as_ref()
                .map(|_| {
                    let (left, top, _, _) = selection_bounds(&self.tabs[index]);
                    (left, top)
                })
                .or_else(|| {
                    self.last_pointer
                        .get(&self.tabs[index].id)
                        .map(|(x, y, _)| (*x, *y))
                })
                .unwrap_or((0, 0));
            let current = self.tabs[index].dataset.clone();
            let mut pasted = current.as_ref().clone();
            let mut pasted_count = 0usize;
            for source_y in 0..patch.height {
                for source_x in 0..patch.width {
                    let target_x = left + source_x;
                    let target_y = top + source_y;
                    if target_x >= self.tabs[index].width || target_y >= self.tabs[index].height {
                        continue;
                    }
                    if set_dataset_sample(
                        &mut pasted,
                        target_x,
                        target_y,
                        self.tabs[index].z,
                        self.tabs[index].t,
                        self.tabs[index].channel,
                        patch.pixels[source_y * patch.width + source_x],
                    ) {
                        pasted_count += 1;
                    }
                }
            }
            self.tabs[index].undo.push(current);
            self.tabs[index].redo.clear();
            self.tabs[index].dataset = Arc::new(pasted);
            self.tabs[index].revision = self.tabs[index].revision.saturating_add(1);
            self.tabs[index].dirty = true;
            self.tabs[index].roi = Some(RoiSelection {
                tool: ToolId::Rect,
                points: vec![
                    (left as f32, top as f32),
                    (
                        (left + patch.width).min(self.tabs[index].width) as f32,
                        (top + patch.height).min(self.tabs[index].height) as f32,
                    ),
                ],
            });
            if let Err(error) = self.tabs[index].refresh_render_image() {
                self.status = error;
            } else {
                self.status = format!("Pasted {pasted_count} pixels into the active image");
            }
            return;
        }

        let data = match Array::from_shape_vec((patch.height, patch.width), patch.pixels) {
            Ok(data) => data.into_dyn(),
            Err(error) => {
                self.status = format!("Clipboard image is invalid: {error}");
                return;
            }
        };
        let dataset = match Dataset::new(
            data,
            Metadata {
                dims: vec![
                    Dim::new(AxisKind::Y, patch.height),
                    Dim::new(AxisKind::X, patch.width),
                ],
                pixel_type: patch.pixel_type,
                ..Metadata::default()
            },
        ) {
            Ok(dataset) => dataset,
            Err(error) => {
                self.status = format!("Clipboard image is invalid: {error}");
                return;
            }
        };
        self.next_tab_id = self.next_tab_id.saturating_add(1);
        match ImageTab::from_dataset(
            self.next_tab_id,
            None,
            format!("Clipboard-{}", self.next_tab_id),
            dataset,
        ) {
            Ok(mut tab) => {
                tab.dirty = true;
                let id = tab.id;
                self.tabs.push(tab);
                if self.open_viewer_window(id, cx) {
                    self.status = "Pasted clipboard into a new image window".into();
                } else {
                    self.rollback_unopened_tab(id);
                }
            }
            Err(error) => self.status = error,
        }
    }

    fn show_internal_clipboard(&mut self, cx: &mut Context<Self>) {
        let Some(patch) = self.internal_clipboard.clone() else {
            self.status = "The internal clipboard is empty".into();
            return;
        };
        let dataset = match clipboard_dataset(&patch) {
            Ok(dataset) => dataset,
            Err(error) => {
                self.status = error;
                return;
            }
        };
        self.next_tab_id = self.next_tab_id.saturating_add(1);
        match ImageTab::from_dataset(
            self.next_tab_id,
            None,
            format!("Clipboard-{}", self.next_tab_id),
            dataset,
        ) {
            Ok(tab) => {
                let id = tab.id;
                self.tabs.push(tab);
                if self.open_viewer_window(id, cx) {
                    self.status = "Opened the internal clipboard in a new image window".into();
                } else {
                    self.rollback_unopened_tab(id);
                }
            }
            Err(error) => self.status = error,
        }
    }

    fn set_lut(&mut self, lut: LookupTable, label: &str) {
        let Some(tab) = self.active_tab_mut() else {
            self.status = "No image is open".into();
            return;
        };
        tab.lut = lut;
        tab.lut_inverted = false;
        if let Err(error) = tab.refresh_render_image() {
            self.status = error;
        } else {
            self.status = format!("{label} lookup table");
        }
    }

    fn invert_lut(&mut self) {
        let Some(tab) = self.active_tab_mut() else {
            self.status = "No image is open".into();
            return;
        };
        tab.lut_inverted = !tab.lut_inverted;
        if let Err(error) = tab.refresh_render_image() {
            self.status = error;
        } else {
            self.status = if tab.lut_inverted {
                "Lookup table inverted".into()
            } else {
                "Lookup table restored".into()
            };
        }
    }

    fn add_selection_to_overlay(&mut self) {
        let Some(selection) = self.active_tab().and_then(|tab| tab.roi.clone()) else {
            self.status = "Add Selection requires an active ROI".into();
            return;
        };
        if let Some(tab) = self.active_tab_mut() {
            tab.overlays.push(selection);
            self.status = format!("Overlay now contains {} element(s)", tab.overlays.len());
        }
    }

    fn measure_active_image(&mut self) {
        let settings = self.measurement_settings;
        let roi_number = self.results.len().saturating_add(1);
        let Some(tab) = self.active_tab() else {
            self.status = "Measure requires an open image".into();
            return;
        };
        let selection = active_measurement_selection(tab);
        let row = match measure_roi_on_tab(
            tab,
            &selection,
            &tab.title,
            "Measure",
            roi_number,
            RoiPosition {
                channel: tab.channel,
                z: tab.z,
                t: tab.t,
            },
            &settings,
        ) {
            Ok(row) => row,
            Err(error) => {
                self.status = error;
                return;
            }
        };
        self.results.push(row);
        self.results_window_pending = true;
        self.status = format!(
            "Measured active C/Z/T plane · {} result row(s)",
            self.results.len()
        );
    }

    fn measure_active_stack(&mut self) {
        let settings = self.measurement_settings;
        let first_roi_number = self.results.len().saturating_add(1);
        let (rows, channel, time) = {
            let Some(tab) = self.active_tab() else {
                self.status = "Measure Stack requires an open image".into();
                return;
            };
            let selection = active_measurement_selection(tab);
            let rows = match measure_stack_rows(tab, &selection, &settings, first_roi_number) {
                Ok(rows) => rows,
                Err(error) => {
                    self.status = error;
                    return;
                }
            };
            (rows, tab.channel, tab.t)
        };
        let added = rows.len();
        self.results.extend(rows);
        self.results_window_pending = true;
        self.status = format!(
            "Measured {added} Z slice(s) at C={} T={} · {} result row(s)",
            channel + 1,
            time + 1,
            self.results.len()
        );
    }

    fn show_overlay_list(&mut self) {
        let Some(tab) = self.active_tab() else {
            self.status = "No image is open".into();
            return;
        };
        let lines = if tab.overlays.is_empty() {
            vec!["No overlay elements".into()]
        } else {
            tab.overlays
                .iter()
                .enumerate()
                .map(|(index, roi)| format!("{}: {}", index + 1, roi_status(roi)))
                .collect()
        };
        self.dialog = Some(DialogState::ImageInfo {
            title: "Overlay Elements".into(),
            lines,
        });
    }

    fn measure_overlays(&mut self) {
        let settings = self.measurement_settings;
        let Some(tab) = self.active_tab() else {
            self.status = "No image is open".into();
            return;
        };
        if tab.overlays.is_empty() {
            self.status = "Measure Overlay requires at least one overlay element".into();
            return;
        }
        let overlays = tab.overlays.clone();
        let position = RoiPosition {
            channel: tab.channel,
            z: tab.z,
            t: tab.t,
        };
        let rows = overlays
            .iter()
            .enumerate()
            .map(|(index, selection)| {
                measure_roi_on_tab(
                    tab,
                    selection,
                    &format!("{}: ROI {}", tab.title, index + 1),
                    "Measure Overlay",
                    index + 1,
                    position,
                    &settings,
                )
            })
            .collect::<Result<Vec<_>, _>>();
        let rows = match rows {
            Ok(rows) => rows,
            Err(error) => {
                self.status = error;
                return;
            }
        };
        self.results.extend(rows);
        self.results_window_pending = true;
        self.status = format!(
            "Measured {} overlay element(s) · {} result row(s)",
            overlays.len(),
            self.results.len()
        );
    }

    fn add_active_roi_to_manager(&mut self) {
        let Some(tab) = self.active_tab() else {
            self.status = "ROI Manager Add requires an open image".into();
            return;
        };
        let Some(selection) = tab.roi.clone() else {
            self.status = "Draw a selection before adding it to the ROI Manager".into();
            return;
        };
        let position = RoiPosition {
            channel: tab.channel,
            z: tab.z,
            t: tab.t,
        };
        let (x, y) = selection.points.first().copied().unwrap_or((0.0, 0.0));
        self.next_managed_roi_id = self.next_managed_roi_id.saturating_add(1);
        let id = self.next_managed_roi_id;
        self.roi_manager.push(ManagedRoi {
            id,
            name: format!(
                "{:04}-{:04}",
                y.max(0.0).round() as usize,
                x.max(0.0).round() as usize
            ),
            selection,
            position,
        });
        self.roi_manager_selected.clear();
        self.roi_manager_selected.insert(id);
        self.roi_manager_selection_anchor = Some(id);
        self.status = format!("Added ROI {} to the ROI Manager", self.roi_manager.len());
    }

    fn update_selected_managed_roi(&mut self) {
        let Some(id) = (self.roi_manager_selected.len() == 1)
            .then(|| self.roi_manager_selected.iter().next().copied())
            .flatten()
        else {
            self.status = "Select one ROI Manager entry to update".into();
            return;
        };
        let Some(tab) = self.active_tab() else {
            self.status = "ROI Manager Update requires an open image".into();
            return;
        };
        let Some(selection) = tab.roi.clone() else {
            self.status = "Draw a selection before updating the ROI Manager".into();
            return;
        };
        let position = RoiPosition {
            channel: tab.channel,
            z: tab.z,
            t: tab.t,
        };
        let Some(entry) = self.roi_manager.iter_mut().find(|entry| entry.id == id) else {
            self.roi_manager_selected.remove(&id);
            if self.roi_manager_selection_anchor == Some(id) {
                self.roi_manager_selection_anchor = None;
            }
            self.status = "The selected ROI Manager entry no longer exists".into();
            return;
        };
        entry.selection = selection;
        entry.position = position;
        self.status = format!("Updated {}", entry.name);
    }

    fn delete_selected_managed_roi(&mut self) {
        if self.roi_manager.is_empty() {
            self.status = "The ROI Manager is empty".into();
            return;
        }
        let order = self
            .roi_manager
            .iter()
            .map(|entry| entry.id)
            .collect::<Vec<_>>();
        let targets = effective_managed_roi_selection(&order, &self.roi_manager_selected);
        let before = self.roi_manager.len();
        self.roi_manager
            .retain(|entry| !targets.contains(&entry.id));
        self.roi_manager_selected.clear();
        self.roi_manager_selection_anchor = None;
        let deleted = before.saturating_sub(self.roi_manager.len());
        self.status = if deleted > 0 {
            format!(
                "Deleted {deleted} ROI(s) · {} remaining",
                self.roi_manager.len()
            )
        } else {
            "No ROI Manager entries were deleted".into()
        };
    }

    fn select_managed_roi(&mut self, id: u64, gesture: ManagedRoiSelectionGesture) {
        let order = self
            .roi_manager
            .iter()
            .map(|entry| entry.id)
            .collect::<Vec<_>>();
        if !apply_managed_roi_selection(
            &order,
            &mut self.roi_manager_selected,
            &mut self.roi_manager_selection_anchor,
            id,
            gesture,
        ) {
            return;
        }
        if self.roi_manager_selected.len() != 1 || !self.roi_manager_selected.contains(&id) {
            self.status = format!("{} ROI(s) selected", self.roi_manager_selected.len());
            return;
        }
        let Some(entry) = self
            .roi_manager
            .iter()
            .find(|entry| entry.id == id)
            .cloned()
        else {
            return;
        };
        let Some(tab) = self.active_tab_mut() else {
            self.status = format!("Selected {} (open an image to restore it)", entry.name);
            return;
        };
        tab.channel = entry.position.channel.min(tab.channels.saturating_sub(1));
        tab.z = entry.position.z.min(tab.slices.saturating_sub(1));
        tab.t = entry.position.t.min(tab.frames.saturating_sub(1));
        tab.roi = Some(entry.selection);
        if let Err(error) = tab.refresh_render_image() {
            self.status = error;
        } else {
            self.status = format!("Restored {}", entry.name);
        }
    }

    fn replace_roi_manager_from_overlay(&mut self) {
        let Some(tab_id) = self.active_tab else {
            self.status = "To ROI Manager requires an open image".into();
            return;
        };
        let Some(tab) = self.tab(tab_id) else {
            return;
        };
        let selections = tab.overlays.clone();
        let position = RoiPosition {
            channel: tab.channel,
            z: tab.z,
            t: tab.t,
        };
        self.roi_manager.clear();
        self.roi_manager_selected.clear();
        self.roi_manager_selection_anchor = None;
        for (index, selection) in selections.into_iter().enumerate() {
            self.next_managed_roi_id = self.next_managed_roi_id.saturating_add(1);
            self.roi_manager.push(ManagedRoi {
                id: self.next_managed_roi_id,
                name: format!("ROI-{:03}", index + 1),
                selection,
                position,
            });
        }
        if let Some(tab) = self.tab_mut(tab_id) {
            tab.overlays.clear();
        }
        self.status = format!(
            "Moved {} overlay ROI(s) to the ROI Manager",
            self.roi_manager.len()
        );
    }

    fn add_managed_rois_to_overlay(&mut self) {
        let order = self
            .roi_manager
            .iter()
            .map(|entry| entry.id)
            .collect::<Vec<_>>();
        let targets = effective_managed_roi_selection(&order, &self.roi_manager_selected);
        let selections = self
            .roi_manager
            .iter()
            .filter(|entry| targets.contains(&entry.id))
            .map(|entry| entry.selection.clone())
            .collect::<Vec<_>>();
        if selections.is_empty() {
            self.status = "The ROI Manager is empty".into();
            return;
        }
        let Some(tab) = self.active_tab_mut() else {
            self.status = "Open an image before adding ROI Manager entries to an overlay".into();
            return;
        };
        let added = selections.len();
        tab.overlays.extend(selections);
        tab.overlays_hidden = false;
        self.status = format!("Added {added} ROI(s) to the overlay");
    }

    fn toggle_roi_manager_show_all(&mut self) {
        let Some(tab_id) = self.active_tab else {
            self.status = "Open an image before using Show All".into();
            return;
        };
        self.roi_manager_show_all_target = if self.roi_manager_show_all_target == Some(tab_id) {
            None
        } else {
            Some(tab_id)
        };
        self.status = if self.roi_manager_show_all_target.is_some() {
            format!("Showing all {} managed ROI(s)", self.roi_manager.len())
        } else {
            "ROI Manager preview hidden".into()
        };
    }

    fn measure_managed_rois(&mut self, cx: &mut Context<Self>) {
        let settings = self.measurement_settings;
        let Some(tab) = self.active_tab() else {
            self.status = "Open an image before measuring managed ROIs".into();
            return;
        };
        let order = self
            .roi_manager
            .iter()
            .map(|entry| entry.id)
            .collect::<Vec<_>>();
        let targets = effective_managed_roi_selection(&order, &self.roi_manager_selected);
        let entries = self
            .roi_manager
            .iter()
            .filter(|entry| targets.contains(&entry.id))
            .cloned()
            .collect::<Vec<_>>();
        if entries.is_empty() {
            self.status = "The ROI Manager is empty".into();
            return;
        }
        let rows = entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                measure_roi_on_tab(
                    tab,
                    &entry.selection,
                    &entry.name,
                    "ROI Manager Measure",
                    index + 1,
                    entry.position,
                    &settings,
                )
            })
            .collect::<Result<Vec<_>, _>>();
        let rows = match rows {
            Ok(rows) => rows,
            Err(error) => {
                self.status = error;
                return;
            }
        };
        let added = rows.len();
        self.results.extend(rows);
        self.results_window_pending = true;
        self.status = format!("Measured {added} managed ROI(s)");
        self.open_results_window(cx);
    }

    fn clear_results(&mut self) {
        self.results.clear();
        self.status = "Results cleared".into();
    }

    fn append_measurement_values(
        &mut self,
        mut values: BTreeMap<String, Value>,
        image_title: &str,
        command: &str,
    ) -> usize {
        let structured_rows = values.remove("rows").and_then(|value| match value {
            Value::Array(rows) => Some(rows),
            _ => None,
        });
        let has_structured_rows = structured_rows.is_some();
        let mut rows = structured_rows
            .unwrap_or_default()
            .into_iter()
            .filter_map(|value| value.as_object().cloned())
            .map(|row| row.into_iter().collect::<BTreeMap<_, _>>())
            .collect::<Vec<_>>();
        let shared_units = ["area_unit", "length_unit"]
            .into_iter()
            .filter_map(|key| values.get(key).cloned().map(|value| (key, value)))
            .collect::<Vec<_>>();
        if rows.is_empty() && !has_structured_rows {
            rows.push(values);
        }
        for row in &mut rows {
            for (key, value) in &shared_units {
                row.entry((*key).into()).or_insert_with(|| value.clone());
            }
            if row.contains_key("Label") {
                row.entry("Image".into())
                    .or_insert_with(|| Value::String(image_title.into()));
            } else {
                row.insert("Label".into(), Value::String(image_title.into()));
            }
            row.entry("Command".into())
                .or_insert_with(|| Value::String(command.into()));
        }
        let added = rows.len();
        self.results.extend(rows);
        self.results_window_pending = true;
        added
    }

    fn summarize_results(&mut self) {
        if self.results.len() < 2 {
            self.status = "Summarize requires at least two result rows".into();
            return;
        }
        if self
            .results
            .last()
            .and_then(|row| row.get("Label"))
            .and_then(Value::as_str)
            == Some("Max")
        {
            self.status = "Results are already summarized".into();
            return;
        }
        let source_rows = self.results.clone();
        let shared_units = match common_result_units(&source_rows) {
            Ok(units) => units,
            Err(message) => {
                self.status = message;
                return;
            }
        };
        let columns = result_columns(&source_rows)
            .into_iter()
            .filter(|column| {
                column != "Label"
                    && column != "Command"
                    && source_rows
                        .iter()
                        .any(|row| row.get(column).and_then(Value::as_f64).is_some())
            })
            .collect::<Vec<_>>();
        if columns.is_empty() {
            self.status = "Results contain no numeric columns to summarize".into();
            return;
        }

        let mut summaries = [
            ("Mean", BTreeMap::new()),
            ("SD", BTreeMap::new()),
            ("Min", BTreeMap::new()),
            ("Max", BTreeMap::new()),
        ];
        for (label, row) in &mut summaries {
            row.insert("Label".into(), json!(label));
            row.insert("Command".into(), json!("Summarize"));
            row.extend(shared_units.clone());
        }
        for column in columns {
            let values = source_rows
                .iter()
                .filter_map(|row| row.get(&column).and_then(Value::as_f64))
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            if values.is_empty() {
                continue;
            }
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let deviation = if values.len() > 1 {
                (values
                    .iter()
                    .map(|value| (value - mean).powi(2))
                    .sum::<f64>()
                    / (values.len() - 1) as f64)
                    .sqrt()
            } else {
                0.0
            };
            summaries[0].1.insert(column.clone(), json!(mean));
            summaries[1].1.insert(column.clone(), json!(deviation));
            summaries[2].1.insert(
                column.clone(),
                json!(values.iter().copied().fold(f64::INFINITY, f64::min)),
            );
            summaries[3].1.insert(
                column,
                json!(values.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
            );
        }
        self.results
            .extend(summaries.into_iter().map(|(_, row)| row));
        self.results_window_pending = true;
        self.status = format!("Summarized {} result rows", source_rows.len());
    }

    fn export_results(&mut self) {
        if self.results.is_empty() {
            self.status = "There are no results to export".into();
            return;
        }
        let Some(path) = FileDialog::new()
            .set_file_name("Results.csv")
            .add_filter("CSV table", &["csv"])
            .save_file()
        else {
            self.status = "Results export canceled".into();
            return;
        };
        let columns = result_columns(&self.results);
        let decimal_places = self.measurement_settings.decimal_places;
        let mut csv = String::new();
        csv.push_str("Row");
        for column in &columns {
            csv.push(',');
            csv.push_str(&csv_cell(column));
        }
        csv.push('\n');
        for (index, row) in self.results.iter().enumerate() {
            csv.push_str(&(index + 1).to_string());
            for column in &columns {
                csv.push(',');
                csv.push_str(&csv_cell(
                    &row.get(column)
                        .map(|value| format_json_value(value, decimal_places))
                        .unwrap_or_default(),
                ));
            }
            csv.push('\n');
        }
        match fs::write(&path, csv) {
            Ok(()) => self.status = format!("Exported results to {}", path.display()),
            Err(error) => self.status = format!("Results export failed: {error}"),
        }
    }

    fn render_results_window(
        &mut self,
        focus_handle: &FocusHandle,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let columns = result_columns(&self.results);
        let decimal_places = self.measurement_settings.decimal_places;
        let column_widths = columns
            .iter()
            .map(|column| result_column_width(&self.results, column, decimal_places))
            .collect::<Vec<_>>();
        let header_cells = std::iter::once(
            div()
                .w(px(56.0))
                .flex_none()
                .px_2()
                .whitespace_nowrap()
                .font_weight(FontWeight::SEMIBOLD)
                .child("Row")
                .into_any_element(),
        )
        .chain(
            columns
                .iter()
                .zip(column_widths.iter())
                .map(|(column, width)| {
                    div()
                        .w(px(*width))
                        .flex_none()
                        .px_2()
                        .whitespace_nowrap()
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(column.clone())
                        .into_any_element()
                }),
        )
        .collect::<Vec<_>>();
        let rows = self
            .results
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let cells = std::iter::once(
                    div()
                        .w(px(56.0))
                        .flex_none()
                        .px_2()
                        .whitespace_nowrap()
                        .text_color(rgb(TEXT_MUTED))
                        .child((index + 1).to_string())
                        .into_any_element(),
                )
                .chain(
                    columns
                        .iter()
                        .zip(column_widths.iter())
                        .map(|(column, width)| {
                            div()
                                .w(px(*width))
                                .flex_none()
                                .px_2()
                                .whitespace_nowrap()
                                .child(
                                    row.get(column)
                                        .map(|value| format_json_value(value, decimal_places))
                                        .unwrap_or_default(),
                                )
                                .into_any_element()
                        }),
                )
                .collect::<Vec<_>>();
                div()
                    .h(px(30.0))
                    .w_full()
                    .flex()
                    .items_center()
                    .border_b_1()
                    .border_color(rgb(CHROME_DARK))
                    .when(index % 2 == 1, |row| row.bg(rgb(MUTED)))
                    .children(cells)
            })
            .collect::<Vec<_>>();
        div()
            .id("results-window")
            .track_focus(focus_handle)
            .key_context("ImageJ")
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(CHROME_LIGHT))
            .text_color(rgb(TEXT))
            .child(
                div()
                    .h(px(46.0))
                    .flex_none()
                    .flex()
                    .items_center()
                    .justify_between()
                    .px_3()
                    .bg(rgb(CHROME))
                    .border_b_1()
                    .border_color(rgb(CHROME_DARK))
                    .child(format!("{} measurement row(s)", self.results.len()))
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(
                                div()
                                    .id("results-clear")
                                    .px_3()
                                    .py_1()
                                    .rounded_md()
                                    .border_1()
                                    .border_color(rgb(CHROME_DARK))
                                    .cursor_pointer()
                                    .hover(|style| style.bg(rgb(MUTED)))
                                    .child("Clear")
                                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                        this.clear_results();
                                        cx.notify();
                                    })),
                            )
                            .child(
                                div()
                                    .id("results-save")
                                    .px_3()
                                    .py_1()
                                    .rounded_md()
                                    .bg(rgb(ACCENT))
                                    .text_color(rgb(0xffffff))
                                    .cursor_pointer()
                                    .hover(|style| style.opacity(0.88))
                                    .child("Save CSV…")
                                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                        this.export_results();
                                        cx.notify();
                                    })),
                            ),
                    ),
            )
            .child(
                div().id("results-scroll").flex_1().overflow_scroll().child(
                    div()
                        .min_w(px(56.0 + column_widths.iter().sum::<f32>()))
                        .child(
                            div()
                                .h(px(32.0))
                                .w_full()
                                .flex()
                                .items_center()
                                .bg(rgb(ACCENT_SOFT))
                                .border_b_1()
                                .border_color(rgb(CHROME_DARK))
                                .children(header_cells),
                        )
                        .children(if rows.is_empty() {
                            vec![
                                div()
                                    .p_5()
                                    .text_color(rgb(TEXT_MUTED))
                                    .child("No measurements yet"),
                            ]
                        } else {
                            rows
                        }),
                ),
            )
            .into_any_element()
    }

    fn render_roi_manager_window(
        &mut self,
        focus_handle: &FocusHandle,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let active_title = self
            .active_tab()
            .map(|tab| tab.title.clone())
            .unwrap_or_else(|| "No active image".into());
        let can_add = self.active_tab().and_then(|tab| tab.roi.as_ref()).is_some();
        let can_update = can_add && self.roi_manager_selected.len() == 1;
        let can_target_image = self.active_tab.is_some() && !self.roi_manager.is_empty();
        let can_delete = !self.roi_manager.is_empty();
        let show_all = self
            .active_tab
            .is_some_and(|tab_id| self.roi_manager_show_all_target == Some(tab_id));
        let selected_ids = self.roi_manager_selected.clone();
        let selected_count = selected_ids.len();
        let rows = self
            .roi_manager
            .clone()
            .into_iter()
            .enumerate()
            .map(|(index, entry)| {
                let id = entry.id;
                let selected = selected_ids.contains(&id);
                let detail = roi_status(&entry.selection);
                div()
                    .id(("roi-manager-row", id as usize))
                    .w_full()
                    .min_h(px(58.0))
                    .flex()
                    .items_center()
                    .gap_3()
                    .px_3()
                    .py_2()
                    .border_b_1()
                    .border_color(rgb(CHROME_DARK))
                    .bg(rgb(if selected { ACCENT_SOFT } else { CHROME_LIGHT }))
                    .when(!selected && index % 2 == 1, |row| row.bg(rgb(MUTED)))
                    .cursor_pointer()
                    .hover(|style| style.bg(rgb(ACCENT_SOFT)))
                    .on_click(cx.listener(move |this, event: &ClickEvent, _, cx| {
                        let modifiers = event.modifiers();
                        let toggle = modifiers.control || modifiers.platform;
                        let gesture = if modifiers.shift {
                            ManagedRoiSelectionGesture::Range { additive: toggle }
                        } else if toggle {
                            ManagedRoiSelectionGesture::Toggle
                        } else {
                            ManagedRoiSelectionGesture::Single
                        };
                        this.select_managed_roi(id, gesture);
                        cx.notify();
                    }))
                    .child(
                        div()
                            .size_7()
                            .flex_none()
                            .flex()
                            .items_center()
                            .justify_center()
                            .rounded_full()
                            .bg(rgb(if selected { ACCENT } else { CHROME_DARK }))
                            .text_color(rgb(if selected { 0xffffff } else { TEXT_MUTED }))
                            .text_size(px(12.0))
                            .font_weight(FontWeight::SEMIBOLD)
                            .child((index + 1).to_string()),
                    )
                    .child(
                        div()
                            .min_w_0()
                            .flex_1()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .gap_2()
                                    .child(
                                        div().font_weight(FontWeight::SEMIBOLD).child(entry.name),
                                    )
                                    .child(
                                        div()
                                            .flex_none()
                                            .px_2()
                                            .py(px(2.0))
                                            .rounded_full()
                                            .bg(rgb(CHROME))
                                            .border_1()
                                            .border_color(rgb(CHROME_DARK))
                                            .text_size(px(11.0))
                                            .text_color(rgb(TEXT_MUTED))
                                            .child(format!(
                                                "C{}  Z{}  T{}",
                                                entry.position.channel + 1,
                                                entry.position.z + 1,
                                                entry.position.t + 1
                                            )),
                                    ),
                            )
                            .child(
                                div()
                                    .text_size(px(12.0))
                                    .text_color(rgb(TEXT_MUTED))
                                    .overflow_hidden()
                                    .child(detail),
                            ),
                    )
                    .into_any_element()
            })
            .collect::<Vec<_>>();

        let add_button = div()
            .id("roi-manager-add")
            .h(px(32.0))
            .px_4()
            .flex()
            .items_center()
            .justify_center()
            .rounded_md()
            .bg(rgb(ACCENT))
            .text_color(rgb(0xffffff))
            .font_weight(FontWeight::SEMIBOLD)
            .when(can_add, |button| {
                button
                    .cursor_pointer()
                    .hover(|style| style.opacity(0.88))
                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.add_active_roi_to_manager();
                        cx.notify();
                    }))
            })
            .when(!can_add, |button| button.opacity(0.42))
            .child("Add");
        let update_button = div()
            .id("roi-manager-update")
            .h(px(32.0))
            .px_3()
            .flex()
            .items_center()
            .justify_center()
            .rounded_md()
            .border_1()
            .border_color(rgb(CHROME_DARK))
            .bg(rgb(CHROME_LIGHT))
            .when(can_update, |button| {
                button
                    .cursor_pointer()
                    .hover(|style| style.bg(rgb(MUTED)))
                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.update_selected_managed_roi();
                        cx.notify();
                    }))
            })
            .when(!can_update, |button| button.opacity(0.42))
            .child("Update");
        let delete_button = div()
            .id("roi-manager-delete")
            .h(px(32.0))
            .px_3()
            .flex()
            .items_center()
            .justify_center()
            .rounded_md()
            .border_1()
            .border_color(rgb(CHROME_DARK))
            .bg(rgb(CHROME_LIGHT))
            .when(can_delete, |button| {
                button
                    .cursor_pointer()
                    .hover(|style| style.bg(rgb(0xfef2f2)).text_color(rgb(0xb91c1c)))
                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.delete_selected_managed_roi();
                        cx.notify();
                    }))
            })
            .when(!can_delete, |button| button.opacity(0.42))
            .child("Delete");
        let measure_button = div()
            .id("roi-manager-measure")
            .h(px(32.0))
            .px_3()
            .flex()
            .items_center()
            .justify_center()
            .rounded_md()
            .border_1()
            .border_color(rgb(CHROME_DARK))
            .bg(rgb(CHROME_LIGHT))
            .when(can_target_image, |button| {
                button
                    .cursor_pointer()
                    .hover(|style| style.bg(rgb(MUTED)))
                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.measure_managed_rois(cx);
                        cx.notify();
                    }))
            })
            .when(!can_target_image, |button| button.opacity(0.42))
            .child("Measure");
        let overlay_button = div()
            .id("roi-manager-overlay")
            .h(px(32.0))
            .px_3()
            .flex()
            .items_center()
            .justify_center()
            .rounded_md()
            .border_1()
            .border_color(rgb(CHROME_DARK))
            .bg(rgb(CHROME_LIGHT))
            .when(can_target_image, |button| {
                button
                    .cursor_pointer()
                    .hover(|style| style.bg(rgb(MUTED)))
                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.add_managed_rois_to_overlay();
                        cx.notify();
                    }))
            })
            .when(!can_target_image, |button| button.opacity(0.42))
            .child("Overlay");
        let show_all_button = div()
            .id("roi-manager-show-all")
            .h(px(32.0))
            .px_3()
            .flex()
            .items_center()
            .justify_center()
            .rounded_md()
            .border_1()
            .border_color(rgb(if show_all { ACCENT } else { CHROME_DARK }))
            .bg(rgb(if show_all { ACCENT_SOFT } else { CHROME_LIGHT }))
            .text_color(rgb(if show_all { ACCENT } else { TEXT }))
            .when(can_target_image, |button| {
                button
                    .cursor_pointer()
                    .hover(|style| style.bg(rgb(ACCENT_SOFT)))
                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                        this.toggle_roi_manager_show_all();
                        cx.notify();
                    }))
            })
            .when(!can_target_image, |button| button.opacity(0.42))
            .child(if show_all { "Hide all" } else { "Show all" });

        div()
            .id("roi-manager-window")
            .track_focus(focus_handle)
            .key_context("ImageJ")
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(CHROME))
            .text_color(rgb(TEXT))
            .child(
                div()
                    .min_h(px(64.0))
                    .flex_none()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_3()
                    .px_4()
                    .py_3()
                    .bg(rgb(CHROME_LIGHT))
                    .border_b_1()
                    .border_color(rgb(CHROME_DARK))
                    .child(
                        div()
                            .min_w_0()
                            .flex_1()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(div().font_weight(FontWeight::SEMIBOLD).child(
                                if selected_count == 0 {
                                    format!("{} ROI(s)", self.roi_manager.len())
                                } else {
                                    format!(
                                        "{} ROI(s) · {selected_count} selected",
                                        self.roi_manager.len()
                                    )
                                },
                            ))
                            .child(
                                div()
                                    .text_size(px(12.0))
                                    .text_color(rgb(TEXT_MUTED))
                                    .overflow_hidden()
                                    .child(active_title),
                            ),
                    )
                    .child(show_all_button),
            )
            .child(
                div()
                    .id("roi-manager-list")
                    .flex_1()
                    .overflow_y_scroll()
                    .bg(rgb(CHROME_LIGHT))
                    .children(if rows.is_empty() {
                        vec![
                            div()
                                .h(px(180.0))
                                .w_full()
                                .flex()
                                .flex_col()
                                .items_center()
                                .justify_center()
                                .gap_2()
                                .px_6()
                                .text_color(rgb(TEXT_MUTED))
                                .child(
                                    div()
                                        .text_size(px(16.0))
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .text_color(rgb(TEXT))
                                        .child("No saved selections"),
                                )
                                .child("Draw an ROI in a viewer, then choose Add.")
                                .into_any_element(),
                        ]
                    } else {
                        rows
                    }),
            )
            .child(
                div()
                    .flex_none()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .p_3()
                    .border_t_1()
                    .border_color(rgb(CHROME_DARK))
                    .bg(rgb(CHROME))
                    .child(
                        div()
                            .w_full()
                            .flex()
                            .gap_2()
                            .child(add_button)
                            .child(update_button)
                            .child(delete_button),
                    )
                    .child(
                        div()
                            .w_full()
                            .flex()
                            .gap_2()
                            .child(measure_button)
                            .child(overlay_button)
                            .child(
                                div()
                                    .id("roi-manager-deselect")
                                    .h(px(32.0))
                                    .px_3()
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded_md()
                                    .border_1()
                                    .border_color(rgb(CHROME_DARK))
                                    .bg(rgb(CHROME_LIGHT))
                                    .when(selected_count > 0, |button| {
                                        button
                                            .cursor_pointer()
                                            .hover(|style| style.bg(rgb(MUTED)))
                                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                                this.roi_manager_selected.clear();
                                                this.roi_manager_selection_anchor = None;
                                                this.status =
                                                    "ROI Manager selection cleared".into();
                                                cx.notify();
                                            }))
                                    })
                                    .when(selected_count == 0, |button| button.opacity(0.42))
                                    .child("Deselect"),
                            ),
                    ),
            )
            .into_any_element()
    }

    fn render_display_adjust_window(
        &mut self,
        focus_handle: &FocusHandle,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let mode = self.display_adjust_mode;
        let can_adjust = self.active_tab.is_some();
        let active_title = self
            .active_tab()
            .map(|tab| {
                format!(
                    "{} · C{} Z{} T{}",
                    tab.title,
                    tab.channel + 1,
                    tab.z + 1,
                    tab.t + 1
                )
            })
            .unwrap_or_else(|| "No active image".into());
        let (domain_min, domain_max, display_min, display_max) = self
            .active_tab()
            .map(|tab| {
                let (domain_min, domain_max) = display_control_domain(tab);
                let (display_min, display_max) = tab.display_range();
                (domain_min, domain_max, display_min, display_max)
            })
            .unwrap_or((0.0, 255.0, 0.0, 255.0));
        let domain_span = (domain_max - domain_min).max(f32::EPSILON);
        let window_width = (display_max - display_min).max(0.0);
        let level = (display_max + display_min) * 0.5;
        let contrast_ratio = if window_width > 0.0 {
            domain_span / window_width
        } else {
            f32::INFINITY
        };
        let contrast_value = if contrast_ratio.is_finite() {
            format!("{}×", format_compact_number(contrast_ratio))
        } else {
            "∞×".into()
        };
        let slider_specs = match mode {
            DisplayAdjustMode::BrightnessContrast => vec![
                (
                    0,
                    "Minimum",
                    format_compact_number(display_min),
                    ((display_min - domain_min) / domain_span).clamp(0.0, 1.0),
                ),
                (
                    1,
                    "Maximum",
                    format_compact_number(display_max),
                    ((display_max - domain_min) / domain_span).clamp(0.0, 1.0),
                ),
                (
                    2,
                    "Brightness",
                    format_compact_number(level),
                    ((domain_max - level) / domain_span).clamp(0.0, 1.0),
                ),
                (
                    3,
                    "Contrast",
                    contrast_value,
                    contrast_fraction_from_window(domain_span, window_width),
                ),
            ],
            DisplayAdjustMode::WindowLevel => vec![
                (
                    0,
                    "Window",
                    format_compact_number(window_width),
                    contrast_fraction_from_window(domain_span, window_width),
                ),
                (
                    1,
                    "Level",
                    format_compact_number(level),
                    ((level - domain_min) / domain_span).clamp(0.0, 1.0),
                ),
            ],
        };
        let histogram = self
            .active_tab()
            .map(|tab| display_histogram(tab, 64))
            .unwrap_or_else(|| vec![0; 64]);
        let histogram_max = histogram.iter().copied().max().unwrap_or(1).max(1) as f32;
        let histogram_bars = histogram
            .into_iter()
            .map(|count| {
                div()
                    .flex_1()
                    .h(px(70.0 * count as f32 / histogram_max))
                    .bg(rgb(if count == 0 { CHROME_DARK } else { ACCENT }))
                    .opacity(if count == 0 { 0.35 } else { 0.78 })
            })
            .collect::<Vec<_>>();
        let slider_rows = slider_specs
            .into_iter()
            .map(|(control, label, value, fraction)| {
                let slider = div()
                    .id(("display-slider", control))
                    .relative()
                    .h(px(20.0))
                    .w_full()
                    .cursor_pointer()
                    .when(can_adjust, |slider| {
                        slider.on_click(cx.listener(move |this, event: &ClickEvent, window, cx| {
                            let width = f32::from(window.bounds().size.width);
                            let fraction =
                                (f32::from(event.position().x) - 20.0) / (width - 40.0).max(1.0);
                            this.adjust_display_from_slider(control, fraction);
                            cx.notify();
                        }))
                    })
                    .child(
                        div()
                            .absolute()
                            .left_0()
                            .right_0()
                            .top(px(8.0))
                            .h(px(4.0))
                            .rounded_full()
                            .bg(rgb(CHROME_DARK)),
                    )
                    .child(
                        div()
                            .absolute()
                            .left_0()
                            .top(px(8.0))
                            .h(px(4.0))
                            .w(gpui::relative(fraction))
                            .rounded_full()
                            .bg(rgb(ACCENT)),
                    )
                    .child(
                        div()
                            .absolute()
                            .left(gpui::relative(fraction))
                            .top(px(2.0))
                            .ml(px(-7.0))
                            .size(px(16.0))
                            .rounded_full()
                            .border_2()
                            .border_color(rgb(ACCENT))
                            .bg(rgb(CHROME_LIGHT))
                            .shadow_sm(),
                    );
                div()
                    .w_full()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(label)
                            .child(
                                div()
                                    .font_family("monospace")
                                    .text_size(px(13.0))
                                    .text_color(rgb(TEXT_MUTED))
                                    .child(value),
                            ),
                    )
                    .child(slider)
            })
            .collect::<Vec<_>>();

        div()
            .id("display-adjust-window")
            .track_focus(focus_handle)
            .key_context("ImageJ")
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(CHROME))
            .text_color(rgb(TEXT))
            .child(
                div()
                    .h(px(58.0))
                    .flex_none()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_3()
                    .px_4()
                    .bg(rgb(CHROME_LIGHT))
                    .border_b_1()
                    .border_color(rgb(CHROME_DARK))
                    .child(
                        div()
                            .min_w_0()
                            .flex_1()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .child("Live display range"),
                            )
                            .child(
                                div()
                                    .text_size(px(12.0))
                                    .text_color(rgb(TEXT_MUTED))
                                    .overflow_hidden()
                                    .child(active_title),
                            ),
                    )
                    .child(
                        div()
                            .px_2()
                            .py_1()
                            .rounded_full()
                            .bg(rgb(ACCENT_SOFT))
                            .text_color(rgb(ACCENT))
                            .text_size(px(12.0))
                            .child("Pixels unchanged"),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .px_4()
                    .py_3()
                    .child(
                        div()
                            .h(px(34.0))
                            .w_full()
                            .flex()
                            .p_1()
                            .rounded_lg()
                            .bg(rgb(MUTED))
                            .child(
                                div()
                                    .id("display-mode-bc")
                                    .flex_1()
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded_md()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .bg(rgb(if mode == DisplayAdjustMode::BrightnessContrast {
                                        CHROME_LIGHT
                                    } else {
                                        MUTED
                                    }))
                                    .text_color(rgb(
                                        if mode == DisplayAdjustMode::BrightnessContrast {
                                            TEXT
                                        } else {
                                            TEXT_MUTED
                                        },
                                    ))
                                    .cursor_pointer()
                                    .hover(|style| style.bg(rgb(CHROME_LIGHT)))
                                    .child("Brightness / Contrast")
                                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                        this.display_adjust_mode =
                                            DisplayAdjustMode::BrightnessContrast;
                                        cx.notify();
                                    })),
                            )
                            .child(
                                div()
                                    .id("display-mode-wl")
                                    .flex_1()
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded_md()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .bg(rgb(if mode == DisplayAdjustMode::WindowLevel {
                                        CHROME_LIGHT
                                    } else {
                                        MUTED
                                    }))
                                    .text_color(rgb(if mode == DisplayAdjustMode::WindowLevel {
                                        TEXT
                                    } else {
                                        TEXT_MUTED
                                    }))
                                    .cursor_pointer()
                                    .hover(|style| style.bg(rgb(CHROME_LIGHT)))
                                    .child("Window / Level")
                                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                        this.display_adjust_mode = DisplayAdjustMode::WindowLevel;
                                        cx.notify();
                                    })),
                            ),
                    )
                    .child(
                        div()
                            .h(px(84.0))
                            .w_full()
                            .flex()
                            .items_end()
                            .gap(px(1.0))
                            .px_2()
                            .pt_2()
                            .rounded_lg()
                            .border_1()
                            .border_color(rgb(CHROME_DARK))
                            .bg(rgb(CHROME_LIGHT))
                            .children(histogram_bars),
                    )
                    .children(slider_rows),
            )
            .child(
                div()
                    .h(px(54.0))
                    .flex_none()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .px_4()
                    .border_t_1()
                    .border_color(rgb(CHROME_DARK))
                    .bg(rgb(CHROME_LIGHT))
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(
                                div()
                                    .id("display-auto")
                                    .px_3()
                                    .py_1()
                                    .rounded_md()
                                    .border_1()
                                    .border_color(rgb(CHROME_DARK))
                                    .when(can_adjust, |button| {
                                        button
                                            .cursor_pointer()
                                            .hover(|style| style.bg(rgb(MUTED)))
                                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                                this.auto_display_adjustment();
                                                cx.notify();
                                            }))
                                    })
                                    .when(!can_adjust, |button| button.opacity(0.42))
                                    .child("Auto"),
                            )
                            .child(
                                div()
                                    .id("display-reset")
                                    .px_3()
                                    .py_1()
                                    .rounded_md()
                                    .border_1()
                                    .border_color(rgb(CHROME_DARK))
                                    .when(can_adjust, |button| {
                                        button
                                            .cursor_pointer()
                                            .hover(|style| style.bg(rgb(MUTED)))
                                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                                let _ = this.reset_display_adjustment(true);
                                                cx.notify();
                                            }))
                                    })
                                    .when(!can_adjust, |button| button.opacity(0.42))
                                    .child("Reset"),
                            )
                            .child(
                                div()
                                    .id("display-set")
                                    .px_3()
                                    .py_1()
                                    .rounded_md()
                                    .border_1()
                                    .border_color(rgb(CHROME_DARK))
                                    .when(can_adjust, |button| {
                                        button
                                            .cursor_pointer()
                                            .hover(|style| style.bg(rgb(MUTED)))
                                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                                match this.display_adjust_mode {
                                                    DisplayAdjustMode::BrightnessContrast => {
                                                        this.open_display_range_set_dialog()
                                                    }
                                                    DisplayAdjustMode::WindowLevel => {
                                                        this.open_window_level_set_dialog()
                                                    }
                                                }
                                                this.open_dialog_window(cx);
                                                cx.notify();
                                            }))
                                    })
                                    .when(!can_adjust, |button| button.opacity(0.42))
                                    .child("Set…"),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(
                                div()
                                    .id("display-apply")
                                    .px_3()
                                    .py_1()
                                    .rounded_md()
                                    .bg(rgb(ACCENT))
                                    .text_color(rgb(0xffffff))
                                    .when(can_adjust, |button| {
                                        button
                                            .cursor_pointer()
                                            .hover(|style| style.opacity(0.88))
                                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                                this.open_apply_lut_dialog();
                                                this.open_dialog_window(cx);
                                                cx.notify();
                                            }))
                                    })
                                    .when(!can_adjust, |button| button.opacity(0.42))
                                    .child("Apply"),
                            )
                            .child(
                                div()
                                    .id("display-close")
                                    .px_3()
                                    .py_1()
                                    .rounded_md()
                                    .border_1()
                                    .border_color(rgb(CHROME_DARK))
                                    .cursor_pointer()
                                    .hover(|style| style.bg(rgb(MUTED)))
                                    .child("Close")
                                    .on_click(cx.listener(|_, _: &ClickEvent, window, _| {
                                        window.remove_window()
                                    })),
                            ),
                    ),
            )
            .into_any_element()
    }

    fn record_command(&mut self, command_id: &str, params: Option<&Value>) {
        let recordable_macro_builtin = matches!(
            command_id,
            "macro.set_min_and_max" | "macro.reset_min_and_max"
        );
        if !self.macro_recording
            || command_id.starts_with("plugins.macros.")
            || (command_id.starts_with("macro.") && !recordable_macro_builtin)
            || command_id.starts_with("window.viewer.")
        {
            return;
        }
        if let Some(line) = macros::macro_record_line_for_command(
            command_id,
            params,
            &command_registry::command_catalog(),
        ) {
            if !self.macro_recorded.is_empty() && !self.macro_recorded.ends_with('\n') {
                self.macro_recorded.push('\n');
            }
            self.macro_recorded.push_str(&line);
            self.macro_recorded.push('\n');
        }
    }

    fn save_macro_recording(&mut self) {
        let Some(path) = FileDialog::new()
            .set_file_name("Recorded.ijm")
            .add_filter("ImageJ macro", &["ijm", "txt"])
            .save_file()
        else {
            self.status = "Macro save canceled".into();
            return;
        };
        match fs::write(&path, &self.macro_recorded) {
            Ok(()) => self.status = format!("Saved macro recording to {}", path.display()),
            Err(error) => self.status = format!("Macro save failed: {error}"),
        }
    }

    fn install_macro(&mut self) {
        let Some(path) = FileDialog::new()
            .add_filter("ImageJ macro", &["ijm", "txt"])
            .set_title("Install Macro")
            .pick_file()
        else {
            self.status = "Macro install canceled".into();
            return;
        };
        match macros::install_macro_file(&path) {
            Ok(installed) => {
                self.dialog = Some(DialogState::Message {
                    title: "Macro installed".into(),
                    body: format!(
                        "Installed {}. It is now available under Plugins › Macros.",
                        installed.display()
                    ),
                });
                self.status = format!("Installed macro {}", installed.display());
            }
            Err(error) => {
                self.dialog = Some(DialogState::Message {
                    title: "Macro install failed".into(),
                    body: error.clone(),
                });
                self.status = error;
            }
        }
    }

    fn pick_and_run_macro(&mut self, cx: &mut Context<Self>) {
        let Some(path) = FileDialog::new()
            .add_filter("ImageJ macro", &["ijm", "txt"])
            .set_title("Run Macro")
            .pick_file()
        else {
            self.status = "Macro run canceled".into();
            return;
        };
        self.run_macro_path(&path, cx);
    }

    fn run_macro_path(&mut self, path: &std::path::Path, cx: &mut Context<Self>) {
        match fs::read_to_string(path) {
            Ok(source) => self.run_macro_source(&path.display().to_string(), &source, cx),
            Err(error) => {
                self.dialog = Some(DialogState::Message {
                    title: "Macro read failed".into(),
                    body: format!("{}: {error}", path.display()),
                });
                self.status = format!("Macro read failed: {error}");
            }
        }
    }

    fn run_startup_macro(&mut self, cx: &mut Context<Self>) {
        let path = macros::startup_macro_path();
        if path.exists() {
            self.run_macro_path(&path, cx);
            return;
        }
        let Some(source) = FileDialog::new()
            .add_filter("ImageJ macro", &["ijm", "txt"])
            .set_title("Choose RunAtStartup Macro")
            .pick_file()
        else {
            self.dialog = Some(DialogState::Message {
                title: "Startup Macro".into(),
                body: format!(
                    "No startup macro is installed yet. Create or copy one to {}.",
                    path.display()
                ),
            });
            return;
        };
        if let Some(parent) = path.parent()
            && let Err(error) = fs::create_dir_all(parent)
        {
            self.status = format!("Startup macro directory failed: {error}");
            return;
        }
        match fs::copy(&source, &path) {
            Ok(_) => self.run_macro_path(&path, cx),
            Err(error) => self.status = format!("Startup macro install failed: {error}"),
        }
    }

    fn run_macro_source(&mut self, name: &str, source: &str, cx: &mut Context<Self>) {
        if self.macro_run.is_some() {
            self.status = "Wait for the current macro to finish before starting another".into();
            return;
        }
        let catalog = command_registry::command_catalog();
        let invocations = match macros::parse_macro_source(source, &catalog) {
            Ok(invocations) => invocations,
            Err(error) => {
                self.dialog = Some(DialogState::Message {
                    title: "Macro parse failed".into(),
                    body: error.clone(),
                });
                self.status = error;
                return;
            }
        };
        self.macro_run = Some(MacroRunState {
            name: name.to_string(),
            pending: invocations.into(),
            lines: Vec::new(),
            executed: 0,
            awaiting_job_id: None,
        });
        self.continue_macro_run(cx);
    }

    fn continue_macro_run(&mut self, cx: &mut Context<Self>) {
        loop {
            let invocation = {
                let Some(run) = self.macro_run.as_mut() else {
                    return;
                };
                if run.awaiting_job_id.is_some() {
                    return;
                }
                run.pending.pop_front()
            };
            let Some(invocation) = invocation else {
                self.finish_macro_run(cx);
                return;
            };
            let command_id = invocation.command_id.clone();
            if operation_for_command(&command_id).is_some()
                && command_id != "analyze.analyze_particles"
            {
                let expected_job_id = self.next_operation_job_id;
                self.start_operation(&command_id, invocation.params, cx);
                if self.next_operation_job_id != expected_job_id {
                    if let Some(run) = self.macro_run.as_mut() {
                        run.awaiting_job_id = Some(expected_job_id);
                    }
                    return;
                }
                self.record_macro_step(&command_id, Err(self.status.clone()));
                continue;
            }
            let outcome = self.execute_macro_invocation(invocation, cx);
            self.record_macro_step(&command_id, outcome);
        }
    }

    fn record_macro_step(&mut self, command_id: &str, outcome: Result<String, String>) {
        let Some(run) = self.macro_run.as_mut() else {
            return;
        };
        match outcome {
            Ok(message) => {
                run.executed += 1;
                run.lines
                    .push(format!("✓ {} — {message}", command_label(command_id)));
            }
            Err(error) => run
                .lines
                .push(format!("× {} — {error}", command_label(command_id))),
        }
    }

    fn finish_macro_run(&mut self, cx: &mut Context<Self>) {
        let Some(mut run) = self.macro_run.take() else {
            return;
        };
        if self.results_window_pending {
            self.open_results_window(cx);
        }
        if run.lines.is_empty() {
            run.lines.push("No executable statements found".into());
        }
        self.dialog = Some(DialogState::ImageInfo {
            title: format!("Macro — {}", run.name),
            lines: run.lines,
        });
        self.status = format!("Macro finished: {} command(s) executed", run.executed);
        let app = cx.entity().downgrade();
        cx.defer(move |cx| {
            if let Some(app) = app.upgrade() {
                let _ = app.update(cx, |app, cx| app.open_dialog_window(cx));
            }
        });
    }

    fn execute_macro_invocation(
        &mut self,
        invocation: MacroCommandInvocation,
        cx: &mut Context<Self>,
    ) -> Result<String, String> {
        let command_id = invocation.command_id;
        let params = invocation.params.unwrap_or_else(|| json!({}));
        match command_id.as_str() {
            "file.close" => {
                self.close_active(cx);
                return Ok("close requested".into());
            }
            "file.open" => {
                let path = params
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "open(...) requires a path".to_string())?;
                self.open_paths([PathBuf::from(path)], cx);
                return Ok(self.status.clone());
            }
            "file.save" | "file.save_as" => {
                let tab_id = self
                    .active_tab
                    .ok_or_else(|| "save requires an open image".to_string())?;
                if let Some(path) = params.get("path").and_then(Value::as_str) {
                    let path = macro_save_path(path, params.get("format").and_then(Value::as_str));
                    if !self.save_tab_to_path(tab_id, path) {
                        return Err(self.status.clone());
                    }
                } else if !self.save_tab(tab_id, false) {
                    return Err(self.status.clone());
                }
                return Ok(self.status.clone());
            }
            "file.new" => {
                self.create_blank_image_with_size(
                    params
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or("Untitled")
                        .to_string(),
                    json_usize(&params, "width", 512),
                    json_usize(&params, "height", 512),
                    json_usize(&params, "slices", 1),
                    json_usize(&params, "channels", 1),
                    json_usize(&params, "frames", 1),
                    parse_pixel_type(
                        params
                            .get("pixelType")
                            .or_else(|| params.get("pixel_type"))
                            .and_then(Value::as_str)
                            .unwrap_or("u8"),
                    ),
                    params.get("fill").and_then(Value::as_f64).unwrap_or(0.0) as f32,
                    cx,
                );
                return Ok(self.status.clone());
            }
            "macro.select_window" => {
                let title = params
                    .get("title")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "selectWindow requires a title".to_string())?;
                let normalized = macros::normalize_macro_command_label(title);
                let id = self
                    .tabs
                    .iter()
                    .find(|tab| {
                        macros::normalize_macro_command_label(&tab.title) == normalized
                            || tab
                                .path
                                .as_ref()
                                .and_then(|path| path.file_stem())
                                .and_then(|stem| stem.to_str())
                                .is_some_and(|stem| {
                                    macros::normalize_macro_command_label(stem) == normalized
                                })
                    })
                    .map(|tab| tab.id)
                    .ok_or_else(|| format!("image window not found: {title}"))?;
                self.focus_viewer(id, cx);
                return Ok(format!("selected viewer-{id}"));
            }
            "macro.select_image" => {
                let id = params
                    .get("id")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| "selectImage requires a numeric id".to_string())?;
                if self.tab(id).is_none() {
                    return Err(format!("viewer-{id} is not open"));
                }
                self.focus_viewer(id, cx);
                return Ok(format!("selected viewer-{id}"));
            }
            "macro.close_window" => {
                let title = params
                    .get("title")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "close(title) requires a title".to_string())?;
                let normalized = macros::normalize_macro_command_label(title);
                let id = self
                    .tabs
                    .iter()
                    .find(|tab| macros::normalize_macro_command_label(&tab.title) == normalized)
                    .map(|tab| tab.id)
                    .ok_or_else(|| format!("image window not found: {title}"))?;
                self.request_close(id, cx);
                return Ok(format!("close requested for viewer-{id}"));
            }
            "macro.set_tool" => {
                let command = params
                    .get("tool")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "setTool requires a supported tool".to_string())?;
                let tool = tool_for_command(command)
                    .ok_or_else(|| format!("unsupported macro tool: {command}"))?;
                self.select_tool(tool);
                return Ok(format!("selected {}", tool.label()));
            }
            "macro.make_rectangle" | "macro.make_oval" => {
                let tool = if command_id == "macro.make_oval" {
                    ToolId::Oval
                } else {
                    ToolId::Rect
                };
                let x = json_f32(&params, "x")?;
                let y = json_f32(&params, "y")?;
                let width = json_f32(&params, "width")?;
                let height = json_f32(&params, "height")?;
                let tab = self
                    .active_tab_mut()
                    .ok_or_else(|| "selection requires an open image".to_string())?;
                tab.roi = Some(RoiSelection {
                    tool,
                    points: vec![(x, y), (x + width, y + height)],
                });
                return Ok(roi_status(tab.roi.as_ref().expect("ROI just assigned")));
            }
            "macro.make_line" => {
                let points = vec![
                    (json_f32(&params, "x1")?, json_f32(&params, "y1")?),
                    (json_f32(&params, "x2")?, json_f32(&params, "y2")?),
                ];
                let tab = self
                    .active_tab_mut()
                    .ok_or_else(|| "line requires an open image".to_string())?;
                tab.roi = Some(RoiSelection {
                    tool: ToolId::Line,
                    points,
                });
                return Ok(roi_status(tab.roi.as_ref().expect("ROI just assigned")));
            }
            "macro.make_selection" => {
                let tool = match params
                    .get("selection_type")
                    .and_then(Value::as_str)
                    .unwrap_or("polygon")
                    .to_ascii_lowercase()
                    .as_str()
                {
                    kind if kind.contains("free") => ToolId::Free,
                    kind if kind.contains("point") => ToolId::Point,
                    kind if kind.contains("line") => ToolId::Line,
                    _ => ToolId::Poly,
                };
                let points = params
                    .get("points")
                    .and_then(Value::as_array)
                    .ok_or_else(|| "makeSelection requires literal points".to_string())?
                    .iter()
                    .map(|point| Ok((json_f32(point, "x")?, json_f32(point, "y")?)))
                    .collect::<Result<Vec<_>, String>>()?;
                let tab = self
                    .active_tab_mut()
                    .ok_or_else(|| "selection requires an open image".to_string())?;
                tab.roi = Some(RoiSelection { tool, points });
                return Ok(roi_status(tab.roi.as_ref().expect("ROI just assigned")));
            }
            "macro.roi_manager" => {
                let action = params
                    .get("action")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "roiManager requires an action".to_string())?
                    .trim()
                    .to_ascii_lowercase();
                match action.as_str() {
                    "add" => self.add_active_roi_to_manager(),
                    "update" => self.update_selected_managed_roi(),
                    "delete" => self.delete_selected_managed_roi(),
                    "measure" => self.measure_managed_rois(cx),
                    "show all" => {
                        if self.active_tab.is_none()
                            || self.roi_manager_show_all_target != self.active_tab
                        {
                            self.toggle_roi_manager_show_all();
                        }
                    }
                    "show none" | "hide all" => {
                        self.roi_manager_show_all_target = None;
                        self.status = "ROI Manager preview hidden".into();
                    }
                    "deselect" => {
                        self.roi_manager_selected.clear();
                        self.roi_manager_selection_anchor = None;
                        self.status = "ROI Manager selection cleared".into();
                    }
                    "reset" => {
                        self.roi_manager.clear();
                        self.roi_manager_selected.clear();
                        self.roi_manager_selection_anchor = None;
                        self.roi_manager_show_all_target = None;
                        self.status = "ROI Manager reset".into();
                    }
                    "open" => self.open_roi_manager_window(cx),
                    "add to overlay" | "overlay" => self.add_managed_rois_to_overlay(),
                    _ => return Err(format!("unsupported roiManager action: {action}")),
                }
                return Ok(self.status.clone());
            }
            "macro.set_min_and_max" => {
                let minimum = json_f32(&params, "minimum")?;
                let maximum = json_f32(&params, "maximum")?;
                return self.apply_display_range(minimum, maximum, false);
            }
            "macro.reset_min_and_max" => {
                return self.reset_display_adjustment(false);
            }
            "macro.set_option" | "macro.builtin_call" | "macro.call" => {
                return Ok("compatibility call acknowledged".into());
            }
            "edit.undo" => self.undo(),
            "edit.redo" => self.redo(),
            "edit.cut" => self.cut_active(),
            "edit.copy" => self.copy_active(),
            "edit.paste" => self.paste_clipboard(cx),
            "edit.selection.none" => self.clear_selection(),
            "edit.selection.all" => self.select_all(),
            "edit.clear" => self.fill_selection(false),
            "edit.fill" => self.fill_selection(true),
            "image.duplicate" => self.duplicate_active_as(
                params
                    .get("title")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                cx,
            ),
            "image.crop" => self.crop_to_selection(),
            "image.adjust.brightness" => {
                match (
                    params.get("minimum").and_then(Value::as_f64),
                    params.get("maximum").and_then(Value::as_f64),
                ) {
                    (Some(minimum), Some(maximum)) => {
                        return self.apply_display_range(minimum as f32, maximum as f32, false);
                    }
                    _ => self.open_display_adjuster(DisplayAdjustMode::BrightnessContrast, cx),
                }
            }
            "image.adjust.window_level" => {
                match (
                    params.get("window").and_then(Value::as_f64),
                    params.get("level").and_then(Value::as_f64),
                ) {
                    (Some(window_width), Some(level)) if window_width >= 0.0 => {
                        let minimum = level - window_width * 0.5;
                        let maximum = level + window_width * 0.5;
                        return self.apply_display_range(minimum as f32, maximum as f32, false);
                    }
                    (Some(_), Some(_)) => return Err("window must be non-negative".into()),
                    _ => self.open_display_adjuster(DisplayAdjustMode::WindowLevel, cx),
                }
            }
            "image.lookup.apply_lut" => {
                let scope = if params.get("stack").and_then(Value::as_bool) == Some(true)
                    || params.get("scope").and_then(Value::as_str) == Some("stack")
                {
                    ApplyLutScope::Stack
                } else {
                    ApplyLutScope::Slice
                };
                return self.apply_lut_to_pixels(scope);
            }
            "analyze.measure" => {
                let before = self.results.len();
                self.measure_active_image();
                if self.results.len() == before {
                    return Err(self.status.clone());
                }
                return Ok(self.status.clone());
            }
            "analyze.set_measurements" => {
                if params.as_object().is_none_or(serde_json::Map::is_empty) {
                    self.open_measurement_settings_dialog();
                    return Ok("measurement settings dialog opened".into());
                }
                self.measurement_settings =
                    measurement_settings_from_params(&params, self.measurement_settings, true);
                self.status = "Measurement settings updated for all image windows".into();
                return Ok(self.status.clone());
            }
            "image.stacks.measure_stack" => {
                let before = self.results.len();
                self.measure_active_stack();
                if self.results.len() == before {
                    return Err(self.status.clone());
                }
                return Ok(self.status.clone());
            }
            "analyze.tools.roi_manager" => self.open_roi_manager_window(cx),
            "analyze.tools.results" => self.open_results_window(cx),
            command if command.starts_with("launcher.tool.") => {
                let tool = tool_for_command(command)
                    .ok_or_else(|| format!("unsupported tool command: {command}"))?;
                self.select_tool(tool);
            }
            "analyze.analyze_particles" => {
                self.run_operation_with_params(&command_id, Some(params));
                if self.status.contains(" failed:") {
                    return Err(self.status.clone());
                }
            }
            command if operation_for_command(command).is_some() => {
                return Err(
                    "operation-backed macro commands must run through the sequential job queue"
                        .into(),
                );
            }
            _ => return Err(format!("unsupported macro command: {command_id}")),
        }
        Ok(self.status.clone())
    }

    fn render_menu_bar(&self, cx: &mut Context<Self>) -> gpui::Div {
        let labels = self
            .menus
            .iter()
            .enumerate()
            .map(|(index, top)| {
                let active = self.open_menu == Some(index);
                div()
                    .id(("menu", index))
                    .h_full()
                    .px_3()
                    .flex()
                    .items_center()
                    .rounded_md()
                    .text_size(px(17.0))
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(rgb(TEXT))
                    .bg(if active { rgb(MUTED) } else { rgb(CHROME) })
                    .hover(|style| style.bg(rgb(MUTED)))
                    .cursor_pointer()
                    .child(top.label.clone())
                    .on_mouse_move(cx.listener(move |this, _: &MouseMoveEvent, window, cx| {
                        if this.open_menu.is_some() && this.open_menu != Some(index) {
                            this.open_menu_popup(index, window, cx);
                            cx.notify();
                        }
                    }))
                    .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                        this.open_menu_popup(index, window, cx);
                        cx.notify();
                    }))
            })
            .collect::<Vec<_>>();
        div()
            .h(px(MENU_HEIGHT))
            .w_full()
            .flex_none()
            .flex()
            .items_center()
            .px_2()
            .gap_1()
            .bg(rgb(CHROME))
            .border_b_1()
            .border_color(rgb(CHROME_DARK))
            .children(labels)
    }

    fn render_toolbar(&self, cx: &mut Context<Self>) -> gpui::Div {
        let tools = TOOLBAR_ITEMS
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let selected = self.selected_tool == item.tool;
                let tool = item.tool;
                div()
                    .id(("tool", index))
                    .size(px(38.0))
                    .flex_none()
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded_md()
                    .bg(if selected {
                        rgb(ACCENT_SOFT)
                    } else {
                        rgb(CHROME)
                    })
                    .border_1()
                    .border_color(if selected {
                        rgb(0x93c5fd)
                    } else {
                        rgb(CHROME_DARK)
                    })
                    .hover(|style| style.bg(rgb(MUTED)).border_color(rgb(0xa1a1aa)))
                    .cursor_pointer()
                    .child(img(icon_path(item.icon)).size(px(22.0)))
                    .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                        this.close_menu_popup(cx);
                        this.select_tool(tool);
                        cx.notify();
                    }))
            })
            .collect::<Vec<_>>();
        div()
            .h(px(TOOLBAR_HEIGHT))
            .w_full()
            .flex_none()
            .flex()
            .items_center()
            .px_3()
            .gap_2()
            .bg(rgb(CHROME))
            .border_b_1()
            .border_color(rgb(CHROME_DARK))
            .children(tools)
    }

    fn render_stack_controls(&self, tab_id: u64, cx: &mut Context<Self>) -> Option<gpui::Div> {
        let tab = self.tab(tab_id)?;
        if !has_stack_controls(tab) {
            return None;
        }
        let positions = [
            (AxisKind::Channel, "C", tab.channel + 1, tab.channels),
            (AxisKind::Z, "Z", tab.z + 1, tab.slices),
            (AxisKind::Time, "T", tab.t + 1, tab.frames),
        ];
        let groups = positions
            .into_iter()
            .filter(|(_, _, _, length)| *length > 1)
            .enumerate()
            .map(|(index, (axis, label, position, length))| {
                let previous_axis = axis;
                let next_axis = axis;
                div()
                    .id(("stack-axis", index))
                    .h_full()
                    .flex()
                    .items_center()
                    .gap_1()
                    .child(
                        div()
                            .id(("stack-previous", index))
                            .size_5()
                            .flex()
                            .items_center()
                            .justify_center()
                            .border_1()
                            .border_color(rgb(CHROME_DARK))
                            .rounded_md()
                            .bg(rgb(CHROME_LIGHT))
                            .hover(|style| style.bg(rgb(MUTED)))
                            .cursor_pointer()
                            .child("‹")
                            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                this.activate_tab(tab_id);
                                this.step_stack(previous_axis, -1);
                                cx.notify();
                            })),
                    )
                    .child(format!("{label}:{position}/{length}"))
                    .child(
                        div()
                            .id(("stack-next", index))
                            .size_5()
                            .flex()
                            .items_center()
                            .justify_center()
                            .border_1()
                            .border_color(rgb(CHROME_DARK))
                            .rounded_md()
                            .bg(rgb(CHROME_LIGHT))
                            .hover(|style| style.bg(rgb(MUTED)))
                            .cursor_pointer()
                            .child("›")
                            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                this.activate_tab(tab_id);
                                this.step_stack(next_axis, 1);
                                cx.notify();
                            })),
                    )
            })
            .collect::<Vec<_>>();
        Some(
            div()
                .h(px(STACK_HEIGHT))
                .w_full()
                .flex_none()
                .flex()
                .items_center()
                .justify_center()
                .gap_4()
                .bg(rgb(MUTED))
                .border_t_1()
                .border_b_1()
                .border_color(rgb(CHROME_DARK))
                .text_size(px(15.0))
                .children(groups),
        )
    }

    fn render_viewer(
        &self,
        tab_id: u64,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let Some(tab) = self.tab(tab_id) else {
            return div()
                .id("closed-viewer")
                .flex_1()
                .w_full()
                .flex()
                .flex_col()
                .items_center()
                .justify_center()
                .gap_3()
                .bg(rgb(CHROME))
                .text_color(rgb(TEXT_MUTED))
                .child(
                    div()
                        .text_size(px(24.0))
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(rgb(TEXT))
                        .child("Image closed"),
                )
                .child("This viewer session is no longer available")
                .into_any_element();
        };

        let geometry = Self::viewer_geometry(window, tab);
        let zoom = geometry.zoom;
        let display_width = geometry.display_width;
        let display_height = geometry.display_height;
        let image = ImageSource::from(tab.render_image.clone());
        let mut selections = if tab.overlays_hidden {
            Vec::new()
        } else {
            tab.overlays
                .iter()
                .cloned()
                .map(|selection| (selection, 0xffef00))
                .collect::<Vec<_>>()
        };
        if self.roi_manager_show_all_target == Some(tab_id) {
            selections.extend(
                self.roi_manager
                    .iter()
                    .cloned()
                    .map(|entry| (entry.selection, 0x60a5fa)),
            );
        }
        if let Some(selection) = tab.roi.clone() {
            selections.push((selection, 0xffef00));
        }

        let mut viewer = div()
            .id("viewer")
            .flex_1()
            .w_full()
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(
                div()
                    .h(px(INFO_HEIGHT))
                    .w_full()
                    .flex_none()
                    .flex()
                    .items_center()
                    .px_2()
                    .bg(rgb(CHROME_LIGHT))
                    .border_b_1()
                    .border_color(rgb(CHROME_DARK))
                    .text_size(px(15.0))
                    .text_color(rgb(TEXT))
                    .child(tab.info_text()),
            );
        if let Some(controls) = self.render_stack_controls(tab_id, cx) {
            viewer = viewer.child(controls);
        }
        viewer
            .child(
                div()
                    .id("image-canvas")
                    .relative()
                    .flex_1()
                    .w_full()
                    .flex()
                    .items_center()
                    .justify_center()
                    .overflow_hidden()
                    .bg(rgb(CANVAS))
                    .cursor_crosshair()
                    .on_drop(cx.listener(|this, paths: &ExternalPaths, _, cx| {
                        this.open_paths(paths.paths().iter().cloned(), cx);
                        cx.notify();
                    }))
                    .on_scroll_wheel(cx.listener(move |this, event: &ScrollWheelEvent, _, cx| {
                        this.activate_tab(tab_id);
                        let delta = match event.delta {
                            ScrollDelta::Pixels(point) => f32::from(point.y),
                            ScrollDelta::Lines(point) => point.y,
                        };
                        let scrolls_stack = this
                            .tab(tab_id)
                            .is_some_and(|tab| tab.slices > 1 || tab.frames > 1)
                            && !event.modifiers.control
                            && !event.modifiers.platform
                            && !event.modifiers.alt;
                        if scrolls_stack && delta != 0.0 {
                            let axis = if this.tab(tab_id).is_some_and(|tab| tab.slices > 1) {
                                AxisKind::Z
                            } else {
                                AxisKind::Time
                            };
                            this.step_stack(axis, if delta < 0.0 { 1 } else { -1 });
                        } else if delta < 0.0 {
                            this.zoom(1.5);
                        } else if delta > 0.0 {
                            this.zoom(1.0 / 1.5);
                        }
                        cx.notify();
                    }))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                            this.begin_roi(tab_id, event, window);
                            cx.notify();
                        }),
                    )
                    .on_mouse_move(
                        cx.listener(move |this, event: &MouseMoveEvent, window, cx| {
                            this.update_roi(tab_id, event, window);
                            cx.notify();
                        }),
                    )
                    .on_mouse_up(
                        MouseButton::Left,
                        cx.listener(move |this, event: &MouseUpEvent, window, cx| {
                            this.end_roi(tab_id, event, window);
                            cx.notify();
                        }),
                    )
                    .child(
                        img(image)
                            .id("active-image")
                            .w(px(display_width))
                            .h(px(display_height))
                            .flex_none(),
                    )
                    .child(
                        div().absolute().top_0().left_0().size_full().child(
                            canvas(
                                move |_, _, _| {},
                                move |bounds, _, window, _| {
                                    for (selection, color) in &selections {
                                        paint_roi(
                                            window,
                                            bounds,
                                            selection,
                                            zoom,
                                            display_width,
                                            display_height,
                                            *color,
                                        );
                                    }
                                },
                            )
                            .size_full(),
                        ),
                    ),
            )
            .into_any_element()
    }

    fn update_pointer_status(&mut self, tab_id: u64, event: &MouseMoveEvent, window: &Window) {
        let Some(tab) = self.tab(tab_id) else {
            return;
        };
        let geometry = Self::viewer_geometry(window, tab);
        let local_x = f32::from(event.position.x) - geometry.image_left;
        let local_y = f32::from(event.position.y) - geometry.image_top;
        if local_x < 0.0
            || local_y < 0.0
            || local_x >= geometry.display_width
            || local_y >= geometry.display_height
        {
            self.last_pointer.remove(&tab_id);
            return;
        }
        let x = (local_x / geometry.zoom).floor() as usize;
        let y = (local_y / geometry.zoom).floor() as usize;
        if let Some(value) = sample_dataset(tab.dataset.as_ref(), x, y, tab.z, tab.t, tab.channel) {
            self.last_pointer.insert(tab_id, (x, y, value));
        }
    }

    fn render_status(&self, tab_id: Option<u64>) -> gpui::Div {
        let pointer = tab_id
            .and_then(|id| self.last_pointer.get(&id).copied())
            .map(|(x, y, value)| format!("  x={x}, y={y}, value={value:.4}"))
            .unwrap_or_default();
        let zoom = tab_id
            .and_then(|id| self.tab(id))
            .map(|tab| {
                if tab.scale_to_fit {
                    "fit".into()
                } else {
                    format!("{:.0}%", tab.zoom * 100.0)
                }
            })
            .unwrap_or_else(|| "—".into());
        let displayed_operation = tab_id
            .or(self.active_tab)
            .and_then(|id| self.active_operations.get(&id));
        let displayed_status = displayed_operation
            .map(|operation| operation.message.as_str())
            .unwrap_or(&self.status);
        let progress = displayed_operation
            .map(|operation| operation.progress)
            .or(self.progress)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        div()
            .h(px(STATUS_HEIGHT))
            .w_full()
            .flex_none()
            .flex()
            .items_center()
            .justify_between()
            .bg(rgb(CHROME))
            .border_t_1()
            .border_color(rgb(CHROME_DARK))
            .pl_2()
            .text_size(px(14.0))
            .text_color(rgb(TEXT))
            .child(format!("{displayed_status}{pointer}"))
            .child(
                div()
                    .h_full()
                    .flex()
                    .items_center()
                    .gap_2()
                    .pr_2()
                    .child(format!("{} · {}", self.selected_tool.label(), zoom))
                    .child(
                        div()
                            .w(px(100.0))
                            .h(px(12.0))
                            .border_1()
                            .border_color(rgb(CHROME_DARK))
                            .rounded_full()
                            .bg(rgb(MUTED))
                            .child(
                                div()
                                    .h_full()
                                    .w(px(98.0 * progress))
                                    .rounded_full()
                                    .bg(rgb(ACCENT)),
                            ),
                    ),
            )
    }

    fn render_menu_window(
        &self,
        menu_index: usize,
        focus_handle: &FocusHandle,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let Some(top) = self.menus.get(menu_index).cloned() else {
            return div().into_any_element();
        };
        let mut items = top.items;
        if top._id == "window" && !self.tabs.is_empty() {
            items.push(MenuManifestItem {
                kind: "separator".into(),
                id: None,
                label: None,
                command: None,
                shortcut: None,
                enabled: None,
                items: None,
            });
            items.extend(self.tabs.iter().map(|tab| MenuManifestItem {
                kind: "item".into(),
                id: Some(format!("window.viewer.{}", tab.id)),
                label: Some(format!(
                    "{}{}",
                    if self.active_tab == Some(tab.id) {
                        "✓ "
                    } else {
                        ""
                    },
                    tab.title
                )),
                command: Some(format!("window.viewer.{}", tab.id)),
                shortcut: None,
                enabled: Some(true),
                items: None,
            }));
        }
        let max_height = f32::from(window.bounds().size.height).max(48.0);
        let primary = self.popup_panel("primary-menu", &items, 0.0, 0.0, max_height, cx);
        let mut overlay = div()
            .id("native-menu-popup")
            .track_focus(focus_handle)
            .key_context("ImageJ")
            .on_action(cx.listener(|this, _: &Escape, _, cx| {
                this.close_menu_popup(cx);
                cx.notify();
            }))
            .relative()
            .size_full()
            .bg(rgb(CHROME_LIGHT))
            .child(primary);

        if let Some(submenu_id) = self.open_submenu.as_deref()
            && let Some(submenu_items) = find_submenu(&items, submenu_id)
        {
            let mut secondary_items = submenu_items.to_vec();
            if submenu_id == "plugins.macros" {
                let installed = macros::list_installed_macro_files();
                if !installed.is_empty() {
                    secondary_items.push(MenuManifestItem {
                        kind: "separator".into(),
                        id: None,
                        label: None,
                        command: None,
                        shortcut: None,
                        enabled: None,
                        items: None,
                    });
                    secondary_items.extend(installed.into_iter().enumerate().map(
                        |(index, path)| {
                            MenuManifestItem {
                                kind: "item".into(),
                                id: Some(format!("plugins.macros.installed.{index}")),
                                label: Some(
                                    path.file_stem()
                                        .and_then(|name| name.to_str())
                                        .unwrap_or("Installed Macro")
                                        .replace('_', " "),
                                ),
                                command: Some(format!("plugins.macros.installed.{index}")),
                                shortcut: None,
                                enabled: Some(true),
                                items: None,
                            }
                        },
                    ));
                }
            }
            overlay = overlay.child(self.popup_panel(
                "secondary-menu",
                &secondary_items,
                POPUP_WIDTH - 5.0,
                0.0,
                max_height,
                cx,
            ));
        }
        overlay.into_any_element()
    }

    fn popup_panel(
        &self,
        id: &'static str,
        items: &[MenuManifestItem],
        left: f32,
        top: f32,
        max_height: f32,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let rows = items
            .iter()
            .enumerate()
            .map(|(row_index, item)| self.popup_row(item, row_index, cx))
            .collect::<Vec<_>>();
        div()
            .id(id)
            .absolute()
            .left(px(left))
            .top(px(top))
            .w(px(POPUP_WIDTH))
            .max_h(px(max_height))
            .overflow_y_scroll()
            .py_1()
            .bg(rgb(CHROME_LIGHT))
            .border_1()
            .border_color(rgb(CHROME_DARK))
            .rounded_lg()
            .shadow_lg()
            .children(rows)
            .into_any_element()
    }

    fn popup_row(
        &self,
        item: &MenuManifestItem,
        row_index: usize,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        if item.kind == "separator" {
            return div()
                .h(px(7.0))
                .mx_2()
                .border_b_1()
                .border_color(rgb(CHROME_DARK))
                .into_any_element();
        }
        let label = item.label.clone().unwrap_or_default();
        let command = item
            .command
            .clone()
            .or_else(|| item.id.clone())
            .unwrap_or_default();
        let shortcut = item.shortcut.clone().unwrap_or_default();
        let is_submenu = item.kind == "submenu";
        let requires_image = if is_submenu {
            false
        } else {
            command_registry::metadata(&command).requires_image
        };
        let enabled = item.enabled.unwrap_or(true)
            && (is_submenu || command_is_routed(&command))
            && (!requires_image || self.active_tab.is_some());
        let command_for_click = command.clone();
        let submenu_for_click = item.id.clone().unwrap_or_else(|| command.clone());
        let submenu_for_hover = submenu_for_click.clone();
        div()
            .id(("popup-row", row_index))
            .h(px(31.0))
            .w_full()
            .flex()
            .items_center()
            .justify_between()
            .px_3()
            .text_size(px(15.0))
            .text_color(if enabled { rgb(TEXT) } else { rgb(0x9b9b9b) })
            .when(enabled, |element| {
                element
                    .hover(|style| style.bg(rgb(MUTED)).text_color(rgb(TEXT)))
                    .cursor_pointer()
                    .when(is_submenu, |element| {
                        element.on_mouse_move(cx.listener(
                            move |this, _: &MouseMoveEvent, _, cx| {
                                if this.open_submenu.as_deref() != Some(submenu_for_hover.as_str())
                                {
                                    this.open_submenu = Some(submenu_for_hover.clone());
                                    cx.notify();
                                }
                            },
                        ))
                    })
                    .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                        if is_submenu {
                            this.open_submenu = Some(submenu_for_click.clone());
                            cx.notify();
                        } else {
                            this.dispatch_command(&command_for_click, window, cx);
                        }
                    }))
            })
            .child(label)
            .child(if is_submenu {
                "▸".to_string()
            } else {
                shortcut
            })
            .into_any_element()
    }

    fn render_dialog(&self, window: &Window, cx: &mut Context<Self>) -> Option<gpui::Div> {
        let dialog = self.dialog.clone()?;
        let width = 470.0;
        let left = ((f32::from(window.bounds().size.width) - width) * 0.5).max(8.0);
        let is_operation = matches!(dialog, DialogState::Operation { .. });
        let is_apply_lut = matches!(
            &dialog,
            DialogState::Operation { command_id, .. } if command_id == "__apply_lut"
        );
        let is_recorder = matches!(dialog, DialogState::MacroRecorder);
        let confirm_close = match &dialog {
            DialogState::ConfirmClose {
                tab_id,
                continuation,
                ..
            } => Some((*tab_id, *continuation)),
            _ => None,
        };
        let (title, body) = match dialog {
            DialogState::About => {
                let lines = [
                    "A GPUI-native, Rust-first ImageJ-compatible workspace.",
                    "Persistent launcher · native image viewer windows",
                    "Image IO, processing operations, and CLI share one core.",
                ];
                (
                    "About ImageJ — image-rs".to_string(),
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .p_4()
                        .text_size(px(16.0))
                        .children(lines)
                        .into_any_element(),
                )
            }
            DialogState::ImageInfo { title, lines } => (
                title,
                div()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .p_4()
                    .text_size(px(16.0))
                    .children(lines)
                    .into_any_element(),
            ),
            DialogState::Message { title, body } => (
                title,
                div()
                    .p_4()
                    .text_size(px(16.0))
                    .child(body)
                    .into_any_element(),
            ),
            DialogState::ConfirmClose { title, .. } => (
                "Unsaved changes".to_string(),
                div()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .p_4()
                    .text_size(px(16.0))
                    .child(format!("Save changes to “{title}” before closing?"))
                    .child(
                        div()
                            .text_size(px(14.0))
                            .text_color(rgb(TEXT_MUTED))
                            .child("Your edits will be lost if you choose Don’t Save."),
                    )
                    .into_any_element(),
            ),
            DialogState::MacroRecorder => {
                let lines = if self.macro_recorded.is_empty() {
                    vec!["// Commands you run will appear here.".to_string()]
                } else {
                    self.macro_recorded.lines().map(str::to_string).collect()
                };
                (
                    "Macro Recorder".to_string(),
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .p_4()
                        .child(div().text_size(px(14.0)).text_color(rgb(TEXT_MUTED)).child(
                            if self.macro_recording {
                                "Recording ImageJ-compatible run(...) commands"
                            } else {
                                "Recording paused"
                            },
                        ))
                        .child(
                            div()
                                .id("macro-recorder-source")
                                .h(px(260.0))
                                .w_full()
                                .overflow_y_scroll()
                                .p_3()
                                .rounded_md()
                                .border_1()
                                .border_color(rgb(CHROME_DARK))
                                .bg(rgb(0x18181b))
                                .text_color(rgb(0xe4e4e7))
                                .font_family("monospace")
                                .text_size(px(14.0))
                                .children(lines),
                        )
                        .into_any_element(),
                )
            }
            DialogState::Operation {
                command_id,
                title,
                fields,
                focused,
                ..
            } => {
                let is_apply_lut = command_id == "__apply_lut";
                let rows = fields
                    .into_iter()
                    .enumerate()
                    .map(|(index, field)| {
                        let active = focused == index;
                        let kind = field.kind;
                        let value = field.value.clone();
                        let control = if kind == ParameterKind::Boolean {
                            div()
                                .id(("parameter-bool", index))
                                .w(px(190.0))
                                .h(px(27.0))
                                .flex()
                                .items_center()
                                .gap_2()
                                .cursor_pointer()
                                .child(
                                    div()
                                        .size_5()
                                        .flex()
                                        .items_center()
                                        .justify_center()
                                        .border_1()
                                        .border_color(rgb(if active {
                                            ACCENT
                                        } else {
                                            CHROME_DARK
                                        }))
                                        .rounded_sm()
                                        .bg(rgb(CHROME_LIGHT))
                                        .child(if value == "true" { "✓" } else { "" }),
                                )
                                .child(if value == "true" {
                                    "Enabled"
                                } else {
                                    "Disabled"
                                })
                                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                    if let Some(DialogState::Operation {
                                        fields, focused, ..
                                    }) = this.dialog.as_mut()
                                    {
                                        *focused = index;
                                        if let Some(field) = fields.get_mut(index) {
                                            field.value = if field.value == "true" {
                                                "false".into()
                                            } else {
                                                "true".into()
                                            };
                                        }
                                    }
                                    cx.notify();
                                }))
                                .into_any_element()
                        } else {
                            div()
                                .id(("parameter-value", index))
                                .w(px(190.0))
                                .h(px(28.0))
                                .flex()
                                .items_center()
                                .px_2()
                                .rounded_md()
                                .bg(rgb(CHROME_LIGHT))
                                .border_1()
                                .border_color(rgb(if active { ACCENT } else { CHROME_DARK }))
                                .cursor_text()
                                .child(if active { format!("{value}│") } else { value })
                                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                    if let Some(DialogState::Operation { focused, .. }) =
                                        this.dialog.as_mut()
                                    {
                                        *focused = index;
                                    }
                                    cx.notify();
                                }))
                                .into_any_element()
                        };
                        div()
                            .h(px(34.0))
                            .w_full()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(field.label)
                            .child(control)
                    })
                    .collect::<Vec<_>>();
                (
                    title,
                    div()
                        .flex()
                        .flex_col()
                        .gap_3()
                        .p_4()
                        .text_size(px(15.0))
                        .when(is_apply_lut, |body| {
                            body.child(
                                div()
                                    .p_3()
                                    .rounded_md()
                                    .border_1()
                                    .border_color(rgb(0xf59e0b))
                                    .bg(rgb(0xfffbeb))
                                    .text_color(rgb(0x92400e))
                                    .child(
                                        "This converts the current display mapping into pixel values. The change is destructive, but can be undone.",
                                    ),
                            )
                        })
                        .children(rows)
                        .into_any_element(),
                )
            }
        };
        let footer = div().w_full().flex().justify_end().gap_2().px_4().pb_4();
        let footer = if let Some((tab_id, continuation)) = confirm_close {
            footer
                .child(
                    div()
                        .id("dialog-cancel")
                        .px_4()
                        .py_1()
                        .border_1()
                        .border_color(rgb(CHROME_DARK))
                        .rounded_md()
                        .cursor_pointer()
                        .hover(|style| style.bg(rgb(MUTED)))
                        .child("Cancel")
                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                            this.cancel_dialog(cx);
                            cx.notify();
                        })),
                )
                .child(
                    div()
                        .id("dialog-discard")
                        .px_4()
                        .py_1()
                        .border_1()
                        .border_color(rgb(CHROME_DARK))
                        .rounded_md()
                        .cursor_pointer()
                        .hover(|style| style.bg(rgb(MUTED)))
                        .child("Don’t Save")
                        .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                            this.confirm_close(tab_id, continuation, false, cx);
                            cx.notify();
                        })),
                )
                .child(
                    div()
                        .id("dialog-save")
                        .px_4()
                        .py_1()
                        .bg(rgb(ACCENT))
                        .border_1()
                        .border_color(rgb(ACCENT))
                        .rounded_md()
                        .text_color(rgb(0xffffff))
                        .cursor_pointer()
                        .hover(|style| style.opacity(0.88))
                        .child("Save")
                        .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                            this.confirm_close(tab_id, continuation, true, cx);
                            cx.notify();
                        })),
                )
        } else if is_recorder {
            footer
                .child(
                    div()
                        .id("recorder-clear")
                        .px_3()
                        .py_1()
                        .border_1()
                        .border_color(rgb(CHROME_DARK))
                        .rounded_md()
                        .cursor_pointer()
                        .hover(|style| style.bg(rgb(MUTED)))
                        .child("Clear")
                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                            this.macro_recorded.clear();
                            this.status = "Macro Recorder cleared".into();
                            cx.notify();
                        })),
                )
                .child(
                    div()
                        .id("recorder-toggle")
                        .px_3()
                        .py_1()
                        .border_1()
                        .border_color(rgb(CHROME_DARK))
                        .rounded_md()
                        .cursor_pointer()
                        .hover(|style| style.bg(rgb(MUTED)))
                        .child(if self.macro_recording {
                            "Pause"
                        } else {
                            "Resume"
                        })
                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                            this.macro_recording = !this.macro_recording;
                            this.status = if this.macro_recording {
                                "Macro Recorder resumed".into()
                            } else {
                                "Macro Recorder paused".into()
                            };
                            cx.notify();
                        })),
                )
                .child(
                    div()
                        .id("recorder-save")
                        .px_3()
                        .py_1()
                        .bg(rgb(ACCENT))
                        .text_color(rgb(0xffffff))
                        .rounded_md()
                        .cursor_pointer()
                        .hover(|style| style.opacity(0.88))
                        .child("Save…")
                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                            this.save_macro_recording();
                            cx.notify();
                        })),
                )
                .child(
                    div()
                        .id("recorder-close")
                        .px_3()
                        .py_1()
                        .border_1()
                        .border_color(rgb(CHROME_DARK))
                        .rounded_md()
                        .cursor_pointer()
                        .hover(|style| style.bg(rgb(MUTED)))
                        .child("Close")
                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                            this.cancel_dialog(cx);
                            cx.notify();
                        })),
                )
        } else {
            footer
                .when(is_operation, |footer| {
                    footer.child(
                        div()
                            .id("dialog-cancel")
                            .px_5()
                            .py_1()
                            .border_1()
                            .border_color(rgb(CHROME_DARK))
                            .rounded_md()
                            .cursor_pointer()
                            .hover(|style| style.bg(rgb(MUTED)))
                            .child("Cancel")
                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                this.cancel_dialog(cx);
                                cx.notify();
                            })),
                    )
                })
                .child(
                    div()
                        .id("dialog-ok")
                        .px_5()
                        .py_1()
                        .bg(rgb(if is_operation { ACCENT } else { MUTED }))
                        .border_1()
                        .border_color(rgb(if is_operation { ACCENT } else { CHROME_DARK }))
                        .rounded_md()
                        .text_color(if is_operation {
                            rgb(0xffffff)
                        } else {
                            rgb(TEXT)
                        })
                        .cursor_pointer()
                        .hover(|style| style.opacity(0.88))
                        .child(if is_apply_lut { "Apply" } else { "OK" })
                        .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                            if is_operation {
                                this.apply_operation_dialog(cx);
                            } else {
                                this.dismiss_dialog_window(cx);
                            }
                            cx.notify();
                        })),
                )
        };
        Some(
            div()
                .absolute()
                .top_0()
                .left_0()
                .size_full()
                .flex()
                .justify_center()
                .bg(rgb(CHROME))
                .child(
                    div()
                        .absolute()
                        .left(px(left))
                        .top(px(24.0))
                        .w(px(width))
                        .bg(rgb(CHROME_LIGHT))
                        .border_1()
                        .border_color(rgb(CHROME_DARK))
                        .rounded_lg()
                        .shadow_lg()
                        .child(
                            div()
                                .h(px(36.0))
                                .flex()
                                .items_center()
                                .justify_between()
                                .px_3()
                                .bg(rgb(CHROME))
                                .border_b_1()
                                .border_color(rgb(CHROME_DARK))
                                .font_weight(FontWeight::SEMIBOLD)
                                .child(title)
                                .child(
                                    div()
                                        .id("close-dialog")
                                        .size_6()
                                        .flex()
                                        .items_center()
                                        .justify_center()
                                        .rounded_sm()
                                        .hover(|style| {
                                            style.bg(rgb(0xc85d5d)).text_color(gpui::white())
                                        })
                                        .cursor_pointer()
                                        .child("×")
                                        .on_click(cx.listener(
                                            move |this, _: &ClickEvent, _, cx| {
                                                this.cancel_dialog(cx);
                                                cx.notify();
                                            },
                                        )),
                                ),
                        )
                        .child(body)
                        .child(footer),
                ),
        )
    }

    fn handle_key_down(
        &mut self,
        event: &KeyDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.activate_session_for_window(window);
        let key = event.keystroke.key.to_ascii_lowercase();
        let mut apply_dialog = false;
        if let Some(DialogState::Operation {
            fields, focused, ..
        }) = self.dialog.as_mut()
        {
            match key.as_str() {
                "enter" => apply_dialog = true,
                "tab" => {
                    if !fields.is_empty() {
                        let direction = if event.keystroke.modifiers.shift {
                            -1
                        } else {
                            1
                        };
                        *focused = (*focused as isize + direction).rem_euclid(fields.len() as isize)
                            as usize;
                    }
                }
                "backspace" => {
                    if let Some(field) = fields.get_mut(*focused)
                        && field.kind != ParameterKind::Boolean
                    {
                        field.value.pop();
                    }
                }
                "space" => {
                    if let Some(field) = fields.get_mut(*focused)
                        && field.kind == ParameterKind::Boolean
                    {
                        field.value = if field.value == "true" {
                            "false".into()
                        } else {
                            "true".into()
                        };
                    }
                }
                _ => {
                    if !event.keystroke.modifiers.control
                        && !event.keystroke.modifiers.platform
                        && !event.keystroke.modifiers.alt
                        && let Some(text) = event.keystroke.key_char.as_deref()
                        && let Some(field) = fields.get_mut(*focused)
                        && field.kind != ParameterKind::Boolean
                        && (field.kind != ParameterKind::Number
                            || text.chars().all(|character| {
                                character.is_ascii_digit() || ".-+eE".contains(character)
                            }))
                    {
                        field.value.push_str(text);
                    }
                }
            }
            if apply_dialog {
                self.apply_operation_dialog(cx);
            }
            cx.notify();
            return;
        }
        if event.keystroke.modifiers.platform || event.keystroke.modifiers.control {
            return;
        }
        if let Some(tool) = tool_from_shortcut(&key) {
            self.select_tool(tool);
            cx.notify();
        }
    }
}

impl ImageJApp {
    fn render_window(
        &mut self,
        viewer_id: Option<u64>,
        focus_handle: &FocusHandle,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let window_id = window.window_handle().window_id();
        let is_launcher = viewer_id.is_none();
        let title = viewer_id
            .and_then(|id| self.tab(id))
            .map(|tab| {
                let dirty = if tab.dirty { " *" } else { "" };
                format!("{}{dirty} — ImageJ / image-rs", tab.title)
            })
            .unwrap_or_else(|| APP_TITLE.to_string());
        window.set_window_title(&title);
        if let Some(tab) = viewer_id.and_then(|id| self.tab(id)) {
            window.set_window_edited(tab.dirty);
        }

        let mut root = div()
            .id("imagej-app")
            .track_focus(focus_handle)
            .key_context("ImageJ")
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, window, cx| {
                this.handle_key_down(event, window, cx)
            }))
            .on_action(cx.listener(|this, _: &NewImage, _, cx| {
                this.open_new_image_dialog();
                this.open_dialog_window(cx);
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &OpenImage, _, cx| {
                this.pick_and_open(cx);
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &Save, window, cx| {
                this.activate_session_for_window(window);
                this.save_active();
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &SaveAs, window, cx| {
                this.activate_session_for_window(window);
                this.save_active_as();
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &CloseImage, window, cx| {
                this.activate_session_for_window(window);
                this.close_active(cx);
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &Cut, window, cx| {
                this.activate_session_for_window(window);
                this.cut_active();
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &Copy, window, cx| {
                this.activate_session_for_window(window);
                this.copy_active();
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &Paste, _, cx| {
                this.paste_clipboard(cx);
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &Undo, window, cx| {
                this.activate_session_for_window(window);
                this.undo();
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &Redo, window, cx| {
                this.activate_session_for_window(window);
                this.redo();
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &ZoomIn, window, cx| {
                this.activate_session_for_window(window);
                this.zoom(1.5);
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &ZoomOut, window, cx| {
                this.activate_session_for_window(window);
                this.zoom(1.0 / 1.5);
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &ZoomActual, window, cx| {
                this.activate_session_for_window(window);
                this.zoom_actual();
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &ZoomFit, window, cx| {
                this.activate_session_for_window(window);
                this.zoom_fit();
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &NextTab, _, cx| {
                this.cycle_tab(1, cx);
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &PreviousTab, _, cx| {
                this.cycle_tab(-1, cx);
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &Escape, window, cx| {
                this.close_menu_popup(cx);
                if this.dialog.is_some() {
                    this.cancel_dialog(cx);
                } else {
                    this.activate_session_for_window(window);
                    this.cancel_active_operation();
                }
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &Quit, _, cx| {
                this.request_quit(cx);
                cx.notify();
            }))
            .relative()
            .size_full()
            .flex()
            .flex_col()
            .overflow_hidden()
            .bg(rgb(CHROME))
            .font_family("Sans Serif")
            .text_color(rgb(TEXT));

        if is_launcher {
            root = root
                .on_drop(cx.listener(|this, paths: &ExternalPaths, _, cx| {
                    this.open_paths(paths.paths().iter().cloned(), cx);
                    cx.notify();
                }))
                .child(self.render_menu_bar(cx))
                .child(self.render_toolbar(cx))
                .child(self.render_status(None));
        } else if let Some(tab_id) = viewer_id {
            root = root
                .child(self.render_viewer(tab_id, window, cx))
                .child(self.render_status(Some(tab_id)));
        }

        let dialog_host = self
            .active_tab
            .and_then(|id| self.viewer_handles.get(&id))
            .map(AnyWindowHandle::window_id)
            .unwrap_or(self.launcher_window.window_id());
        if self.dialog_window.is_none()
            && window_id == dialog_host
            && let Some(dialog) = self.render_dialog(window, cx)
        {
            root = root.child(dialog);
        }
        root.into_any_element()
    }
}

impl Render for ImageJApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let focus_handle = self.focus_handle.clone();
        self.render_window(None, &focus_handle, window, cx)
    }
}

impl Render for ImageViewerWindow {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if !self.ready {
            return div().size_full().bg(rgb(CANVAS)).into_any_element();
        }
        let tab_id = self.tab_id;
        let focus_handle = self.focus_handle.clone();
        self.app.update(cx, |app, cx| {
            app.render_window(Some(tab_id), &focus_handle, window, cx)
        })
    }
}

impl Render for MenuPopupWindow {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if !self.ready {
            return div().size_full().bg(rgb(CHROME_LIGHT)).into_any_element();
        }
        let wide = self.app.read(cx).open_submenu.is_some();
        if wide != self.wide {
            self.wide = wide;
            let width = if wide {
                POPUP_WIDTH * 2.0 - 5.0
            } else {
                POPUP_WIDTH
            };
            window.resize(size(px(width), px(self.height)));
        }
        let menu_index = self.menu_index;
        let focus_handle = self.focus_handle.clone();
        self.app.update(cx, |app, cx| {
            app.render_menu_window(menu_index, &focus_handle, window, cx)
        })
    }
}

impl Render for AppDialogWindow {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if !self.ready {
            return div().size_full().bg(rgb(CHROME)).into_any_element();
        }
        if self.app.read(cx).dialog.is_none() {
            window.remove_window();
            return div().size_full().bg(rgb(CHROME)).into_any_element();
        }
        if let Some(dialog) = self.app.read(cx).dialog.as_ref() {
            let (title, height) = dialog_window_spec(dialog);
            window.set_window_title(&title);
            let current = window.bounds().size;
            if (f32::from(current.height) - height).abs() > 1.0 {
                window.resize(size(current.width, px(height)));
            }
        }
        let focus_handle = self.focus_handle.clone();
        self.app.update(cx, |app, cx| {
            let dialog = app
                .render_dialog(window, cx)
                .map(IntoElement::into_any_element)
                .unwrap_or_else(|| div().into_any_element());
            div()
                .id("dialog-window-root")
                .track_focus(&focus_handle)
                .key_context("ImageJ")
                .on_key_down(cx.listener(|this, event: &KeyDownEvent, window, cx| {
                    this.handle_key_down(event, window, cx)
                }))
                .on_action(cx.listener(|this, _: &Escape, _, cx| {
                    this.cancel_dialog(cx);
                    cx.notify();
                }))
                .relative()
                .size_full()
                .bg(rgb(CHROME))
                .child(dialog)
                .into_any_element()
        })
    }
}

impl Render for ResultsWindow {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if !self.ready {
            return div().size_full().bg(rgb(CHROME)).into_any_element();
        }
        let focus_handle = self.focus_handle.clone();
        self.app
            .update(cx, |app, cx| app.render_results_window(&focus_handle, cx))
    }
}

impl Render for RoiManagerWindow {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if !self.ready {
            return div().size_full().bg(rgb(CHROME)).into_any_element();
        }
        let focus_handle = self.focus_handle.clone();
        self.app.update(cx, |app, cx| {
            app.render_roi_manager_window(&focus_handle, cx)
        })
    }
}

impl Render for DisplayAdjustWindow {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if !self.ready {
            return div().size_full().bg(rgb(CHROME)).into_any_element();
        }
        let mode = self.app.read(cx).display_adjust_mode;
        window.set_window_title(match mode {
            DisplayAdjustMode::BrightnessContrast => "Brightness & Contrast — ImageJ / image-rs",
            DisplayAdjustMode::WindowLevel => "Window & Level — ImageJ / image-rs",
        });
        let focus_handle = self.focus_handle.clone();
        self.app.update(cx, |app, cx| {
            app.render_display_adjust_window(&focus_handle, cx)
        })
    }
}

fn axis_len(dataset: &DatasetF32, axis: AxisKind) -> usize {
    dataset
        .axis_index(axis)
        .and_then(|index| dataset.shape().get(index).copied())
        .unwrap_or(1)
        .max(1)
}

fn dataset_is_true_rgb(dataset: &DatasetF32) -> bool {
    if dataset.metadata.pixel_type != PixelType::U8 || axis_len(dataset, AxisKind::Channel) != 3 {
        return false;
    }
    let [red, green, blue] = dataset.metadata.channel_names.as_slice() else {
        return false;
    };
    let is_named = |actual: &str, short: &str, long: &str| {
        let actual = actual.trim();
        actual.eq_ignore_ascii_case(short) || actual.eq_ignore_ascii_case(long)
    };
    is_named(red, "r", "red") && is_named(green, "g", "green") && is_named(blue, "b", "blue")
}

fn render_dataset_plane(
    dataset: &DatasetF32,
    z: usize,
    t: usize,
    channel: usize,
    lut: LookupTable,
    lut_inverted: bool,
    display_min: f32,
    display_max: f32,
) -> Result<RenderedPlane, String> {
    if dataset.ndim() < 2 {
        return Err("dataset must have at least two dimensions".into());
    }
    let y_axis = dataset.axis_index(AxisKind::Y).unwrap_or(0);
    let x_axis = dataset
        .axis_index(AxisKind::X)
        .unwrap_or(if y_axis == 0 { 1 } else { 0 });
    if x_axis >= dataset.ndim() || y_axis >= dataset.ndim() || x_axis == y_axis {
        return Err("could not infer distinct X/Y axes".into());
    }
    let width = dataset.shape()[x_axis];
    let height = dataset.shape()[y_axis];
    let channel_axis = dataset.axis_index(AxisKind::Channel);
    let channel_count = channel_axis.map(|axis| dataset.shape()[axis]).unwrap_or(1);
    let true_rgb = dataset_is_true_rgb(dataset);
    let mut rgba = Vec::with_capacity(width * height * 4);
    let mut index = vec![0usize; dataset.ndim()];
    if let Some(axis) = dataset.axis_index(AxisKind::Z) {
        index[axis] = z.min(dataset.shape()[axis].saturating_sub(1));
    }
    if let Some(axis) = dataset.axis_index(AxisKind::Time) {
        index[axis] = t.min(dataset.shape()[axis].saturating_sub(1));
    }
    for y in 0..height {
        index[y_axis] = y;
        for x in 0..width {
            index[x_axis] = x;
            let (red, green, blue) = if true_rgb {
                let axis = channel_axis.expect("true RGB data has a channel axis");
                index[axis] = 0;
                let red = display_sample(dataset.data[IxDyn(&index)], display_min, display_max);
                index[axis] = 1;
                let green = display_sample(dataset.data[IxDyn(&index)], display_min, display_max);
                index[axis] = 2;
                let blue = display_sample(dataset.data[IxDyn(&index)], display_min, display_max);
                (red, green, blue)
            } else {
                if let Some(axis) = channel_axis {
                    index[axis] = channel.min(channel_count.saturating_sub(1));
                }
                let mut gray =
                    display_sample(dataset.data[IxDyn(&index)], display_min, display_max);
                if lut_inverted {
                    gray = 255 - gray;
                }
                lut_color(lut, gray)
            };
            // GPUI's RenderImage texture bytes are BGRA.
            rgba.extend_from_slice(&[blue, green, red, 255]);
        }
    }
    let buffer = RgbaImage::from_raw(width as u32, height as u32, rgba)
        .ok_or_else(|| "could not construct image buffer".to_string())?;
    let image = Arc::new(RenderImage::new(smallvec![Frame::new(buffer)]));
    Ok(RenderedPlane {
        width,
        height,
        image,
    })
}

fn display_sample(value: f32, display_min: f32, display_max: f32) -> u8 {
    let normalized = if display_max > display_min {
        (value - display_min) / (display_max - display_min)
    } else if display_max == display_min {
        if value <= display_min { 0.0 } else { 1.0 }
    } else {
        0.0
    };
    (normalized.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn apply_lut_sample(value: f32, minimum: f32, maximum: f32, output_maximum: f32) -> f32 {
    if value <= minimum {
        0.0
    } else if value >= maximum {
        output_maximum
    } else {
        (((value - minimum) / (maximum - minimum)) * (output_maximum + 1.0))
            .floor()
            .min(output_maximum)
    }
}

fn display_range_count(dataset: &DatasetF32) -> usize {
    if dataset_is_true_rgb(dataset) {
        1
    } else {
        axis_len(dataset, AxisKind::Channel)
    }
}

fn default_display_ranges(dataset: &DatasetF32, z: usize, t: usize) -> Vec<(f32, f32)> {
    (0..display_range_count(dataset))
        .map(|channel| default_display_range(dataset, z, t, channel))
        .collect()
}

fn default_display_range(dataset: &DatasetF32, z: usize, t: usize, channel: usize) -> (f32, f32) {
    match dataset.metadata.pixel_type {
        PixelType::U8 => (0.0, 255.0),
        PixelType::U16 => {
            displayed_plane_min_max(dataset, z, t, channel).unwrap_or((0.0, 65_535.0))
        }
        PixelType::F32 => displayed_plane_min_max(dataset, z, t, channel).unwrap_or((0.0, 1.0)),
    }
}

fn displayed_plane_min_max(
    dataset: &DatasetF32,
    z: usize,
    t: usize,
    channel: usize,
) -> Option<(f32, f32)> {
    if dataset.ndim() < 2 {
        return None;
    }
    let y_axis = dataset.axis_index(AxisKind::Y).unwrap_or(0);
    let x_axis = dataset
        .axis_index(AxisKind::X)
        .unwrap_or(if y_axis == 0 { 1 } else { 0 });
    if x_axis >= dataset.ndim() || y_axis >= dataset.ndim() || x_axis == y_axis {
        return None;
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

    let mut range: Option<(f32, f32)> = None;
    for y in 0..dataset.shape()[y_axis] {
        index[y_axis] = y;
        for x in 0..dataset.shape()[x_axis] {
            index[x_axis] = x;
            let value = dataset.data[IxDyn(&index)];
            if !value.is_finite() {
                continue;
            }
            range = Some(match range {
                Some((minimum, maximum)) => (minimum.min(value), maximum.max(value)),
                None => (value, value),
            });
        }
    }
    range
}

fn display_control_domain(tab: &ImageTab) -> (f32, f32) {
    let (minimum, maximum) = default_display_range(tab.dataset.as_ref(), tab.z, tab.t, tab.channel);
    if maximum > minimum {
        return (minimum, maximum);
    }
    match tab.dataset.metadata.pixel_type {
        PixelType::U8 => (0.0, 255.0),
        PixelType::U16 => (0.0, 65_535.0),
        PixelType::F32 => {
            let padding = minimum.abs().max(1.0);
            (minimum - padding, maximum + padding)
        }
    }
}

fn contrast_window_from_fraction(domain_span: f32, fraction: f32) -> f32 {
    let domain_span = domain_span.max(f32::EPSILON);
    let fraction = fraction.clamp(0.0, 1.0);
    if fraction <= 0.5 {
        // ImageJ's contrast slider is centered at the default data span. Values to
        // the left progressively widen the window; cap the endpoint at 512 spans.
        domain_span / (2.0 * fraction.max(1.0 / 1024.0))
    } else {
        domain_span * 2.0 * (1.0 - fraction)
    }
}

fn contrast_fraction_from_window(domain_span: f32, window_width: f32) -> f32 {
    let domain_span = domain_span.max(f32::EPSILON);
    if window_width <= 0.0 {
        return 1.0;
    }
    let slope = domain_span / window_width;
    if slope <= 1.0 {
        (slope * 0.5).clamp(0.0, 0.5)
    } else {
        (1.0 - 0.5 / slope).clamp(0.5, 1.0)
    }
}

fn displayed_plane_values(tab: &ImageTab) -> Vec<f32> {
    let sample_budget = 120_000usize;
    let pixel_count = tab.width.saturating_mul(tab.height);
    let mut values = Vec::with_capacity(pixel_count.min(sample_budget));
    let true_rgb = dataset_is_true_rgb(tab.dataset.as_ref());
    let mut push_sample = |x: usize, y: usize| {
        let value = if true_rgb {
            let red = sample_dataset(tab.dataset.as_ref(), x, y, tab.z, tab.t, 0);
            let green = sample_dataset(tab.dataset.as_ref(), x, y, tab.z, tab.t, 1);
            let blue = sample_dataset(tab.dataset.as_ref(), x, y, tab.z, tab.t, 2);
            match (red, green, blue) {
                (Some(red), Some(green), Some(blue))
                    if red.is_finite() && green.is_finite() && blue.is_finite() =>
                {
                    // ImageJ-compatible weighted RGB-to-luminance conversion.
                    Some(red * 0.299 + green * 0.587 + blue * 0.114)
                }
                _ => None,
            }
        } else {
            sample_dataset(tab.dataset.as_ref(), x, y, tab.z, tab.t, tab.channel)
                .filter(|value| value.is_finite())
        };
        if let Some(value) = value {
            values.push(value);
        }
    };

    if let Some(selection) = tab.roi.as_ref() {
        let pixels = roi_sample_pixels(selection, tab.width, tab.height);
        let stride = pixels
            .len()
            .saturating_add(sample_budget.saturating_sub(1))
            .checked_div(sample_budget)
            .unwrap_or(1)
            .max(1);
        for (x, y) in pixels.into_iter().step_by(stride) {
            push_sample(x, y);
        }
    } else {
        let stride = ((pixel_count as f64 / sample_budget as f64).sqrt().ceil() as usize).max(1);
        for y in (0..tab.height).step_by(stride) {
            for x in (0..tab.width).step_by(stride) {
                push_sample(x, y);
            }
        }
    }
    values
}

fn display_histogram(tab: &ImageTab, bins: usize) -> Vec<usize> {
    let bins = bins.max(1);
    let mut histogram = vec![0usize; bins];
    let (minimum, maximum) = display_control_domain(tab);
    let span = (maximum - minimum).max(f32::EPSILON);
    for value in displayed_plane_values(tab) {
        let bin = (((value - minimum) / span) * (bins.saturating_sub(1) as f32))
            .floor()
            .clamp(0.0, bins.saturating_sub(1) as f32) as usize;
        histogram[bin] = histogram[bin].saturating_add(1);
    }
    histogram
}

fn auto_display_range(tab: &ImageTab, threshold_divisor: usize) -> Option<(f32, f32)> {
    let values = displayed_plane_values(tab);
    if values.is_empty() {
        return None;
    }
    let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if minimum == maximum {
        return Some((minimum, maximum));
    }

    const BINS: usize = 256;
    let span = maximum - minimum;
    let mut histogram = [0usize; BINS];
    for value in values.iter().copied() {
        let bin = (((value - minimum) / span) * (BINS - 1) as f32)
            .floor()
            .clamp(0.0, (BINS - 1) as f32) as usize;
        histogram[bin] = histogram[bin].saturating_add(1);
    }
    let dominant_bin_limit = values.len() / 10;
    if dominant_bin_limit == 0 {
        return Some((minimum, maximum));
    }
    let threshold = values.len() / threshold_divisor.max(1);
    let is_signal = |count: usize| count <= dominant_bin_limit && count > threshold;
    let Some(low_bin) = histogram.iter().position(|count| is_signal(*count)) else {
        return Some((minimum, maximum));
    };
    let Some(high_bin) = histogram.iter().rposition(|count| is_signal(*count)) else {
        return Some((minimum, maximum));
    };
    if high_bin <= low_bin {
        return Some((minimum, maximum));
    }
    let bin_size = span / BINS as f32;
    Some((
        minimum + low_bin as f32 * bin_size,
        minimum + high_bin as f32 * bin_size,
    ))
}

fn lut_color(lut: LookupTable, gray: u8) -> (u8, u8, u8) {
    let t = gray as f32 / 255.0;
    let channel = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    match lut {
        LookupTable::Grays => (gray, gray, gray),
        LookupTable::Fire => (
            channel(t * 3.0),
            channel(t * 3.0 - 1.0),
            channel(t * 3.0 - 2.0),
        ),
        LookupTable::Ice => (
            channel(t * 0.45),
            channel(t * 0.95),
            channel(0.18 + t * 0.82),
        ),
        LookupTable::Spectrum => hsv_to_rgb(t * 300.0, 0.9, 1.0),
        LookupTable::Rgb332 => (gray & 0xe0, (gray << 3) & 0xe0, (gray << 6) & 0xc0),
        LookupTable::Red => (gray, 0, 0),
        LookupTable::Green => (0, gray, 0),
        LookupTable::Blue => (0, 0, gray),
        LookupTable::Cyan => (0, gray, gray),
        LookupTable::Magenta => (gray, 0, gray),
        LookupTable::Yellow => (gray, gray, 0),
        LookupTable::RedGreen if gray < 128 => (gray.saturating_mul(2), 0, 0),
        LookupTable::RedGreen => (0, gray.wrapping_mul(2), 0),
    }
}

fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> (u8, u8, u8) {
    let chroma = value * saturation;
    let sector = (hue / 60.0).rem_euclid(6.0);
    let x = chroma * (1.0 - (sector.rem_euclid(2.0) - 1.0).abs());
    let (red, green, blue) = match sector as usize {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    let offset = value - chroma;
    let channel = |component: f32| ((component + offset) * 255.0).round() as u8;
    (channel(red), channel(green), channel(blue))
}

fn sample_dataset(
    dataset: &DatasetF32,
    x: usize,
    y: usize,
    z: usize,
    t: usize,
    channel: usize,
) -> Option<f32> {
    let y_axis = dataset.axis_index(AxisKind::Y).unwrap_or(0);
    let x_axis = dataset.axis_index(AxisKind::X).unwrap_or(1);
    let mut index = vec![0usize; dataset.ndim()];
    *index.get_mut(y_axis)? = y;
    *index.get_mut(x_axis)? = x;
    if let Some(axis) = dataset.axis_index(AxisKind::Z) {
        index[axis] = z.min(dataset.shape()[axis].saturating_sub(1));
    }
    if let Some(axis) = dataset.axis_index(AxisKind::Time) {
        index[axis] = t.min(dataset.shape()[axis].saturating_sub(1));
    }
    if let Some(axis) = dataset.axis_index(AxisKind::Channel) {
        index[axis] = channel.min(dataset.shape()[axis].saturating_sub(1));
    }
    dataset.data.get(IxDyn(&index)).copied()
}

fn set_dataset_sample(
    dataset: &mut DatasetF32,
    x: usize,
    y: usize,
    z: usize,
    t: usize,
    channel: usize,
    value: f32,
) -> bool {
    let y_axis = dataset.axis_index(AxisKind::Y).unwrap_or(0);
    let x_axis = dataset.axis_index(AxisKind::X).unwrap_or(1);
    let mut index = vec![0usize; dataset.ndim()];
    if y >= dataset.shape()[y_axis] || x >= dataset.shape()[x_axis] {
        return false;
    }
    index[y_axis] = y;
    index[x_axis] = x;
    if let Some(axis) = dataset.axis_index(AxisKind::Z) {
        index[axis] = z.min(dataset.shape()[axis].saturating_sub(1));
    }
    if let Some(axis) = dataset.axis_index(AxisKind::Time) {
        index[axis] = t.min(dataset.shape()[axis].saturating_sub(1));
    }
    if let Some(axis) = dataset.axis_index(AxisKind::Channel) {
        index[axis] = channel.min(dataset.shape()[axis].saturating_sub(1));
    }
    let Some(pixel) = dataset.data.get_mut(IxDyn(&index)) else {
        return false;
    };
    *pixel = value;
    true
}

fn selection_bounds(tab: &ImageTab) -> (usize, usize, usize, usize) {
    let Some(selection) = tab.roi.as_ref().filter(|roi| !roi.points.is_empty()) else {
        return (0, 0, tab.width, tab.height);
    };
    roi_bounds(selection, tab.width, tab.height)
}

fn rasterize_processing_area(
    selection: Option<&RoiSelection>,
    image_width: usize,
    image_height: usize,
) -> Result<Option<RasterizedAreaMask>, String> {
    let Some(selection) = selection else {
        return Ok(None);
    };
    let minimum_points = match selection.tool {
        ToolId::Rect | ToolId::Oval => 2,
        ToolId::Poly | ToolId::Free => 3,
        ToolId::Line
        | ToolId::Angle
        | ToolId::Point
        | ToolId::Wand
        | ToolId::Text
        | ToolId::Zoom
        | ToolId::Hand
        | ToolId::Dropper
        | ToolId::More => {
            return Err(format!(
                "{} is not an area selection and cannot mask image processing",
                selection.tool.label()
            ));
        }
    };
    if image_width == 0 || image_height == 0 || selection.points.len() < minimum_points {
        return Err("The active area selection is incomplete".into());
    }

    let (left, top, width, height) = roi_bounds(selection, image_width, image_height);
    let mut members = Vec::with_capacity(width.saturating_mul(height));
    for y in top..top.saturating_add(height) {
        for x in left..left.saturating_add(width) {
            members.push(u8::from(roi_contains_pixel(selection, x, y)));
        }
    }
    if !members.contains(&1) {
        return Err("The active area selection contains no image pixels".into());
    }
    Ok(Some(RasterizedAreaMask {
        left,
        top,
        width,
        height,
        members,
    }))
}

fn roi_bounds(
    selection: &RoiSelection,
    image_width: usize,
    image_height: usize,
) -> (usize, usize, usize, usize) {
    let min_x = selection
        .points
        .iter()
        .map(|point| point.0)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, image_width.saturating_sub(1) as f32) as usize;
    let min_y = selection
        .points
        .iter()
        .map(|point| point.1)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, image_height.saturating_sub(1) as f32) as usize;
    let max_x = selection
        .points
        .iter()
        .map(|point| point.0)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .max((min_x + 1) as f32)
        .clamp(1.0, image_width as f32) as usize;
    let max_y = selection
        .points
        .iter()
        .map(|point| point.1)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .max((min_y + 1) as f32)
        .clamp(1.0, image_height as f32) as usize;
    (min_x, min_y, max_x - min_x, max_y - min_y)
}

fn active_measurement_selection(tab: &ImageTab) -> RoiSelection {
    tab.roi.clone().unwrap_or_else(|| RoiSelection {
        tool: ToolId::Rect,
        points: vec![(0.0, 0.0), (tab.width as f32, tab.height as f32)],
    })
}

fn measure_stack_rows(
    tab: &ImageTab,
    selection: &RoiSelection,
    settings: &MeasurementSettings,
    first_roi_number: usize,
) -> Result<Vec<BTreeMap<String, Value>>, String> {
    if tab.slices <= 1 {
        return Err("Measure Stack requires more than one Z slice".into());
    }
    // ImageJ's MeasureStack macro temporarily enables Stack position even
    // when that global setting is off, so every emitted row remains
    // identifiable. This native milestone iterates Z at the active C/T.
    let mut stack_settings = *settings;
    stack_settings.stack_position = true;
    (0..tab.slices)
        .map(|z| {
            measure_roi_on_tab(
                tab,
                selection,
                &format!("{}: slice {}", tab.title, z + 1),
                "Measure Stack",
                first_roi_number.saturating_add(z),
                RoiPosition {
                    channel: tab.channel,
                    z,
                    t: tab.t,
                },
                &stack_settings,
            )
        })
        .collect()
}

#[derive(Debug, Clone)]
struct MeasurementAxisCalibration {
    spacing: f64,
    origin: f64,
    direction: f64,
    unit: String,
}

impl MeasurementAxisCalibration {
    fn coordinate(&self, pixel: f64) -> f64 {
        self.origin + self.direction * self.spacing * pixel
    }
}

fn measurement_axis_calibration(
    dataset: &DatasetF32,
    axis_kind: AxisKind,
    label: &str,
) -> Result<MeasurementAxisCalibration, String> {
    let axis = dataset
        .axis_index(axis_kind)
        .ok_or_else(|| format!("Measurement requires an {label}-axis"))?;
    let dim = &dataset.metadata.dims[axis];
    let spacing = f64::from(dim.spacing.unwrap_or(1.0));
    if !spacing.is_finite() || spacing <= 0.0 {
        return Err("Measurement requires finite positive X/Y spacing".into());
    }
    let unit = dim
        .unit
        .as_deref()
        .map(str::trim)
        .filter(|unit| !unit.is_empty())
        .unwrap_or("pixel")
        .to_string();
    let origin_key = format!("{label}_origin_coordinate");
    let origin = match dataset.metadata.extras.get(&origin_key) {
        None => 0.0,
        Some(value) => value
            .as_f64()
            .filter(|origin| origin.is_finite())
            .ok_or_else(|| format!("Measurement requires a finite `{origin_key}`"))?,
    };
    let inverted_key = format!("{label}_coordinate_inverted");
    let inverted = match dataset.metadata.extras.get(&inverted_key) {
        None => false,
        Some(value) => value
            .as_bool()
            .ok_or_else(|| format!("Measurement requires `{inverted_key}` to be a boolean"))?,
    };
    Ok(MeasurementAxisCalibration {
        spacing,
        origin,
        direction: if inverted { -1.0 } else { 1.0 },
        unit,
    })
}

fn measure_roi_on_tab(
    tab: &ImageTab,
    selection: &RoiSelection,
    label: &str,
    command: &str,
    roi_number: usize,
    position: RoiPosition,
    settings: &MeasurementSettings,
) -> Result<BTreeMap<String, Value>, String> {
    let channel = position.channel.min(tab.channels.saturating_sub(1));
    let z = position.z.min(tab.slices.saturating_sub(1));
    let t = position.t.min(tab.frames.saturating_sub(1));
    let bounds = if selection.points.is_empty() || tab.width == 0 || tab.height == 0 {
        (0, 0, 0, 0)
    } else {
        roi_bounds(selection, tab.width, tab.height)
    };
    let pixels = match selection.tool {
        // ImageJ measures a point's intensity at its nearest pixel, while
        // retaining the original floating coordinate in the Results table.
        ToolId::Point => selection
            .points
            .first()
            .and_then(|point| nearest_pixel_at_point(*point, tab.width, tab.height))
            .into_iter()
            .collect(),
        // Analyzer.measureAngle deliberately uses empty statistics and adds
        // only the geometric angle after the selected measurement columns.
        ToolId::Line | ToolId::Angle => Vec::new(),
        _ => roi_sample_pixels(selection, tab.width, tab.height),
    };
    let samples = pixels
        .iter()
        .filter_map(|&(x, y)| {
            sample_dataset(tab.dataset.as_ref(), x, y, z, t, channel).map(|value| (x, y, value))
        })
        .filter(|(_, _, value)| value.is_finite())
        .collect::<Vec<_>>();
    let values = if selection.tool == ToolId::Line {
        line_profile_values(tab, selection, z, t, channel)
    } else {
        samples
            .iter()
            .map(|(_, _, value)| *value)
            .collect::<Vec<_>>()
    };
    let values = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let statistics = measurement_statistics(&values);
    let x_calibration = measurement_axis_calibration(tab.dataset.as_ref(), AxisKind::X, "x")?;
    let y_calibration = measurement_axis_calibration(tab.dataset.as_ref(), AxisKind::Y, "y")?;
    let common_length_unit = x_calibration.unit == y_calibration.unit;
    let (length_x_scale, length_y_scale, length_unit) = if common_length_unit {
        (
            x_calibration.spacing,
            y_calibration.spacing,
            x_calibration.unit.clone(),
        )
    } else {
        (1.0, 1.0, "pixel".to_string())
    };
    let area_unit = if x_calibration.unit == y_calibration.unit {
        format!("{}^2", x_calibration.unit)
    } else {
        format!("{}·{}", x_calibration.unit, y_calibration.unit)
    };
    let sampled_area = values.len() as f64 * x_calibration.spacing * y_calibration.spacing;
    let reported_area = if selection.tool == ToolId::Point {
        0.0
    } else {
        sampled_area
    };
    let y_coordinate = |pixel: f64| {
        if settings.invert_y_coordinates && y_calibration.direction > 0.0 {
            y_calibration.origin + y_calibration.spacing * (tab.height as f64 - pixel - 1.0)
        } else {
            y_calibration.coordinate(pixel)
        }
    };

    let mut row = BTreeMap::new();
    if settings.display_label {
        row.insert("Label".into(), json!(label));
    }
    row.insert("Image".into(), json!(tab.title));
    row.insert("Command".into(), json!(command));
    row.insert("ROI".into(), json!(roi_number));
    row.insert("Type".into(), json!(selection.tool.label()));
    let point_stack_position =
        selection.tool == ToolId::Point && (tab.channels > 1 || tab.slices > 1 || tab.frames > 1);
    if settings.stack_position || point_stack_position {
        let hyperstack = tab.channels > 1 || tab.frames > 1;
        if tab.channels > 1 {
            row.insert("Ch".into(), json!(channel + 1));
        }
        // ImageJ's hyperstack branch includes only dimensions whose size is
        // greater than one. Ordinary images/stacks use the legacy Slice
        // column even for a single plane.
        if tab.slices > 1 || !hyperstack {
            row.insert("Slice".into(), json!(z + 1));
        }
        if tab.frames > 1 {
            row.insert("Frame".into(), json!(t + 1));
        }
    }
    if settings.area {
        row.insert("Area".into(), json!(reported_area));
        row.insert("area_unit".into(), json!(area_unit));
    }
    if settings.mean {
        row.insert("Mean".into(), json!(statistics.mean));
    }
    if settings.standard_deviation {
        row.insert("StdDev".into(), json!(statistics.standard_deviation));
    }
    if settings.min_max {
        row.insert("Min".into(), json!(statistics.minimum));
        row.insert("Max".into(), json!(statistics.maximum));
    }
    if settings.median {
        row.insert("Median".into(), json!(statistics.median));
    }
    if settings.integrated_density {
        row.insert("IntDen".into(), json!(sampled_area * statistics.mean));
        row.insert("RawIntDen".into(), json!(statistics.sum));
        row.entry("area_unit".into())
            .or_insert_with(|| json!(area_unit));
    }
    if selection.tool == ToolId::Point
        && let Some(&(x, y)) = selection.points.first()
    {
        row.insert("X".into(), json!(x_calibration.coordinate(f64::from(x))));
        row.insert("Y".into(), json!(y_coordinate(f64::from(y))));
    } else if settings.centroid && selection.tool == ToolId::Angle {
        row.insert("X".into(), json!(0.0));
        row.insert("Y".into(), json!(0.0));
    } else if settings.centroid && selection.tool == ToolId::Line && selection.points.len() >= 2 {
        let first = selection.points[0];
        let last = *selection
            .points
            .last()
            .expect("line has at least two points");
        let midpoint_x = f64::from(first.0 + last.0) * 0.5;
        let midpoint_y = f64::from(first.1 + last.1) * 0.5;
        row.insert("X".into(), json!(x_calibration.coordinate(midpoint_x)));
        row.insert("Y".into(), json!(y_coordinate(midpoint_y)));
    } else if settings.centroid && !samples.is_empty() {
        let count = samples.len() as f64;
        let centroid_x = samples.iter().map(|(x, _, _)| *x as f64 + 0.5).sum::<f64>() / count;
        let centroid_y = samples.iter().map(|(_, y, _)| *y as f64 + 0.5).sum::<f64>() / count;
        row.insert("X".into(), json!(x_calibration.coordinate(centroid_x)));
        row.insert("Y".into(), json!(y_coordinate(centroid_y)));
    }
    if settings.bounding_rectangle {
        row.insert(
            "BX".into(),
            json!(x_calibration.coordinate(bounds.0 as f64)),
        );
        row.insert("BY".into(), json!(y_coordinate(bounds.1 as f64)));
        row.insert(
            "Width".into(),
            json!(bounds.2 as f64 * x_calibration.spacing),
        );
        row.insert(
            "Height".into(),
            json!(bounds.3 as f64 * y_calibration.spacing),
        );
    }

    if settings.perimeter
        && let Some(perimeter) = roi_perimeter(selection, length_x_scale, length_y_scale)
    {
        row.insert("Perim.".into(), json!(perimeter));
        row.insert("length_unit".into(), json!(length_unit.clone()));
    }

    if selection.tool == ToolId::Line {
        row.insert(
            "Length".into(),
            json!(polyline_length(
                &selection.points,
                length_x_scale,
                length_y_scale
            )),
        );
        row.insert("length_unit".into(), json!(length_unit));
        if selection.points.len() == 2
            && let Some(angle) = roi_angle(selection, length_x_scale, length_y_scale)
        {
            row.insert("Angle".into(), json!(angle));
        }
    } else if selection.tool == ToolId::Angle
        && let Some(angle) = roi_angle(selection, length_x_scale, length_y_scale)
    {
        row.insert("Angle".into(), json!(angle));
    }
    Ok(row)
}

fn roi_sample_pixels(
    selection: &RoiSelection,
    image_width: usize,
    image_height: usize,
) -> Vec<(usize, usize)> {
    if image_width == 0 || image_height == 0 || selection.points.is_empty() {
        return Vec::new();
    }
    match selection.tool {
        ToolId::Line | ToolId::Angle => {
            polyline_sample_pixels(&selection.points, image_width, image_height)
        }
        ToolId::Point | ToolId::Wand | ToolId::Text => selection
            .points
            .first()
            .and_then(|point| pixel_at_point(*point, image_width, image_height))
            .into_iter()
            .collect(),
        ToolId::Poly | ToolId::Free if selection.points.len() < 3 => {
            polyline_sample_pixels(&selection.points, image_width, image_height)
        }
        ToolId::Rect | ToolId::Oval | ToolId::Poly | ToolId::Free => {
            let (left, top, width, height) = roi_bounds(selection, image_width, image_height);
            let mut pixels = Vec::with_capacity(width.saturating_mul(height));
            for y in top..top.saturating_add(height) {
                for x in left..left.saturating_add(width) {
                    if roi_contains_pixel(selection, x, y) {
                        pixels.push((x, y));
                    }
                }
            }
            pixels
        }
        ToolId::Zoom | ToolId::Hand | ToolId::Dropper | ToolId::More => Vec::new(),
    }
}

fn roi_contains_pixel(selection: &RoiSelection, x: usize, y: usize) -> bool {
    let sample = (x as f32 + 0.5, y as f32 + 0.5);
    match selection.tool {
        ToolId::Rect if selection.points.len() >= 2 => {
            let (minimum, maximum) = selection_axis_bounds(selection);
            let right = if maximum.0 > minimum.0 {
                maximum.0
            } else {
                minimum.0 + 1.0
            };
            let bottom = if maximum.1 > minimum.1 {
                maximum.1
            } else {
                minimum.1 + 1.0
            };
            sample.0 >= minimum.0 && sample.0 < right && sample.1 >= minimum.1 && sample.1 < bottom
        }
        ToolId::Oval if selection.points.len() >= 2 => {
            let (minimum, maximum) = selection_axis_bounds(selection);
            let radius_x = ((maximum.0 - minimum.0) * 0.5).max(0.5);
            let radius_y = ((maximum.1 - minimum.1) * 0.5).max(0.5);
            let center_x = minimum.0 + radius_x;
            let center_y = minimum.1 + radius_y;
            let dx = (sample.0 - center_x) / radius_x;
            let dy = (sample.1 - center_y) / radius_y;
            dx * dx + dy * dy <= 1.0 + f32::EPSILON
        }
        ToolId::Poly | ToolId::Free if selection.points.len() >= 3 => {
            point_in_polygon(sample, &selection.points)
        }
        ToolId::Point | ToolId::Wand | ToolId::Text => {
            selection
                .points
                .first()
                .and_then(|point| pixel_at_point(*point, usize::MAX, usize::MAX))
                == Some((x, y))
        }
        _ => false,
    }
}

fn selection_axis_bounds(selection: &RoiSelection) -> ((f32, f32), (f32, f32)) {
    let minimum = selection.points.iter().fold(
        (f32::INFINITY, f32::INFINITY),
        |(minimum_x, minimum_y), &(x, y)| (minimum_x.min(x), minimum_y.min(y)),
    );
    let maximum = selection.points.iter().fold(
        (f32::NEG_INFINITY, f32::NEG_INFINITY),
        |(maximum_x, maximum_y), &(x, y)| (maximum_x.max(x), maximum_y.max(y)),
    );
    (minimum, maximum)
}

fn point_in_polygon(point: (f32, f32), vertices: &[(f32, f32)]) -> bool {
    if vertices.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut previous = *vertices
        .last()
        .expect("polygon has at least three vertices");
    for &current in vertices {
        if point_on_segment(point, previous, current) {
            return true;
        }
        let crosses_scanline = (current.1 > point.1) != (previous.1 > point.1);
        if crosses_scanline {
            let intersection_x = (previous.0 - current.0) * (point.1 - current.1)
                / (previous.1 - current.1)
                + current.0;
            if point.0 < intersection_x {
                inside = !inside;
            }
        }
        previous = current;
    }
    inside
}

fn point_on_segment(point: (f32, f32), start: (f32, f32), end: (f32, f32)) -> bool {
    let segment = (end.0 - start.0, end.1 - start.1);
    let offset = (point.0 - start.0, point.1 - start.1);
    let length_squared = segment.0 * segment.0 + segment.1 * segment.1;
    if length_squared <= 1.0e-8 {
        return offset.0 * offset.0 + offset.1 * offset.1 <= 1.0e-8;
    }
    let cross = segment.0 * offset.1 - segment.1 * offset.0;
    if cross.abs() > 1.0e-4 {
        return false;
    }
    let dot = offset.0 * segment.0 + offset.1 * segment.1;
    dot >= -1.0e-4 && dot <= length_squared + 1.0e-4
}

fn pixel_at_point(
    point: (f32, f32),
    image_width: usize,
    image_height: usize,
) -> Option<(usize, usize)> {
    if !point.0.is_finite()
        || !point.1.is_finite()
        || point.0 < 0.0
        || point.1 < 0.0
        || point.0 >= image_width as f32
        || point.1 >= image_height as f32
    {
        return None;
    }
    Some((point.0.floor() as usize, point.1.floor() as usize))
}

fn nearest_pixel_at_point(
    point: (f32, f32),
    image_width: usize,
    image_height: usize,
) -> Option<(usize, usize)> {
    if !point.0.is_finite() || !point.1.is_finite() {
        return None;
    }
    let x = point.0.round();
    let y = point.1.round();
    if x < 0.0 || y < 0.0 || x >= image_width as f32 || y >= image_height as f32 {
        return None;
    }
    Some((x as usize, y as usize))
}

/// ImageJ's default line profile uses approximately one bilinearly
/// interpolated value per Euclidean pixel. Keep a shared vertex only once
/// when the ROI is a multi-segment polyline.
fn line_profile_values(
    tab: &ImageTab,
    selection: &RoiSelection,
    z: usize,
    t: usize,
    channel: usize,
) -> Vec<f32> {
    if selection.points.len() < 2 {
        return Vec::new();
    }
    let mut values = Vec::new();
    for segment in selection.points.windows(2) {
        let start = segment[0];
        let end = segment[1];
        let dx = f64::from(end.0 - start.0);
        let dy = f64::from(end.1 - start.1);
        let intervals = dx.hypot(dy).round().max(0.0) as usize;
        let first_step = usize::from(!values.is_empty());
        if intervals == 0 {
            if values.is_empty() {
                values.push(interpolated_dataset_sample(
                    tab,
                    f64::from(start.0),
                    f64::from(start.1),
                    z,
                    t,
                    channel,
                ));
            }
            continue;
        }
        values.extend((first_step..=intervals).map(|step| {
            let fraction = step as f64 / intervals as f64;
            let x = f64::from(start.0) + dx * fraction;
            let y = f64::from(start.1) + dy * fraction;
            interpolated_dataset_sample(tab, x, y, z, t, channel)
        }));
    }
    values
}

fn interpolated_dataset_sample(
    tab: &ImageTab,
    x: f64,
    y: f64,
    z: usize,
    t: usize,
    channel: usize,
) -> f32 {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= tab.width as f64
        || y >= tab.height as f64
    {
        return 0.0;
    }
    if tab.width <= 1 || tab.height <= 1 {
        return nearest_pixel_at_point((x as f32, y as f32), tab.width, tab.height)
            .and_then(|(x, y)| sample_dataset(tab.dataset.as_ref(), x, y, z, t, channel))
            .unwrap_or(0.0);
    }
    let left = x.floor().clamp(0.0, tab.width.saturating_sub(2) as f64) as usize;
    let top = y.floor().clamp(0.0, tab.height.saturating_sub(2) as f64) as usize;
    let x_fraction = (x - left as f64).clamp(0.0, 1.0);
    let y_fraction = (y - top as f64).clamp(0.0, 1.0);
    let sample =
        |x, y| sample_dataset(tab.dataset.as_ref(), x, y, z, t, channel).unwrap_or(0.0) as f64;
    let lower_left = sample(left, top);
    let lower_right = sample(left + 1, top);
    let upper_left = sample(left, top + 1);
    let upper_right = sample(left + 1, top + 1);
    let lower = lower_left + x_fraction * (lower_right - lower_left);
    let upper = upper_left + x_fraction * (upper_right - upper_left);
    (lower + y_fraction * (upper - lower)) as f32
}

fn polyline_sample_pixels(
    points: &[(f32, f32)],
    image_width: usize,
    image_height: usize,
) -> Vec<(usize, usize)> {
    let mut pixels = Vec::new();
    let mut seen = std::collections::HashSet::new();
    if points.len() == 1 {
        if let Some(pixel) = pixel_at_point(points[0], image_width, image_height) {
            pixels.push(pixel);
        }
        return pixels;
    }
    for segment in points.windows(2) {
        let start = segment[0];
        let end = segment[1];
        let steps = (end.0 - start.0)
            .abs()
            .max((end.1 - start.1).abs())
            .ceil()
            .max(1.0) as usize;
        for step in 0..=steps {
            let fraction = step as f32 / steps as f32;
            let point = (
                start.0 + (end.0 - start.0) * fraction,
                start.1 + (end.1 - start.1) * fraction,
            );
            if let Some(pixel) = pixel_at_point(point, image_width, image_height)
                && seen.insert(pixel)
            {
                pixels.push(pixel);
            }
        }
    }
    pixels
}

fn polyline_length(points: &[(f32, f32)], x_scale: f64, y_scale: f64) -> f64 {
    points
        .windows(2)
        .map(|segment| {
            let dx = f64::from(segment[1].0 - segment[0].0) * x_scale;
            let dy = f64::from(segment[1].1 - segment[0].1) * y_scale;
            dx.hypot(dy)
        })
        .sum()
}

fn roi_angle(selection: &RoiSelection, x_scale: f64, y_scale: f64) -> Option<f64> {
    if selection.points.len() >= 3 {
        let first = selection.points[0];
        let vertex = selection.points[1];
        let last = selection.points[2];
        let first_vector = (
            f64::from(first.0 - vertex.0) * x_scale,
            f64::from(first.1 - vertex.1) * y_scale,
        );
        let second_vector = (
            f64::from(last.0 - vertex.0) * x_scale,
            f64::from(last.1 - vertex.1) * y_scale,
        );
        let denominator =
            first_vector.0.hypot(first_vector.1) * second_vector.0.hypot(second_vector.1);
        if denominator <= f64::EPSILON {
            return None;
        }
        let cosine = ((first_vector.0 * second_vector.0 + first_vector.1 * second_vector.1)
            / denominator)
            .clamp(-1.0, 1.0);
        return Some(cosine.acos().to_degrees());
    }
    let [first, last] = selection.points.as_slice() else {
        return None;
    };
    Some(
        (-f64::from(last.1 - first.1) * y_scale)
            .atan2(f64::from(last.0 - first.0) * x_scale)
            .to_degrees(),
    )
}

fn roi_perimeter(selection: &RoiSelection, x_scale: f64, y_scale: f64) -> Option<f64> {
    match selection.tool {
        ToolId::Line | ToolId::Angle if selection.points.len() >= 2 => {
            Some(polyline_length(&selection.points, x_scale, y_scale))
        }
        ToolId::Rect if selection.points.len() >= 2 => {
            let (minimum, maximum) = selection_axis_bounds(selection);
            let width = f64::from((maximum.0 - minimum.0).abs()) * x_scale;
            let height = f64::from((maximum.1 - minimum.1).abs()) * y_scale;
            Some(2.0 * (width + height))
        }
        ToolId::Oval if selection.points.len() >= 2 => {
            let (minimum, maximum) = selection_axis_bounds(selection);
            let a = f64::from((maximum.0 - minimum.0).abs()) * x_scale * 0.5;
            let b = f64::from((maximum.1 - minimum.1).abs()) * y_scale * 0.5;
            if a <= f64::EPSILON || b <= f64::EPSILON {
                return Some(0.0);
            }
            let h = ((a - b) * (a - b)) / ((a + b) * (a + b));
            Some(std::f64::consts::PI * (a + b) * (1.0 + 3.0 * h / (10.0 + (4.0 - 3.0 * h).sqrt())))
        }
        ToolId::Poly | ToolId::Free if selection.points.len() >= 3 => {
            let mut perimeter = polyline_length(&selection.points, x_scale, y_scale);
            let first = selection.points[0];
            let last = *selection
                .points
                .last()
                .expect("area ROI has at least three points");
            perimeter += (f64::from(last.0 - first.0) * x_scale)
                .hypot(f64::from(last.1 - first.1) * y_scale);
            Some(perimeter)
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
struct MeasurementStatistics {
    minimum: f32,
    maximum: f32,
    mean: f64,
    sum: f64,
    standard_deviation: f64,
    median: f64,
}

fn measurement_statistics(values: &[f32]) -> MeasurementStatistics {
    if values.is_empty() {
        return MeasurementStatistics {
            minimum: 0.0,
            maximum: 0.0,
            mean: 0.0,
            sum: 0.0,
            standard_deviation: 0.0,
            median: 0.0,
        };
    }
    let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = values.iter().map(|value| f64::from(*value)).sum::<f64>();
    let mean = sum / values.len() as f64;
    let standard_deviation = if values.len() > 1 {
        let squared_error = values
            .iter()
            .map(|value| (f64::from(*value) - mean).powi(2))
            .sum::<f64>();
        (squared_error / (values.len() - 1) as f64).sqrt()
    } else {
        0.0
    };
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let middle = sorted.len() / 2;
    let median = if sorted.len().is_multiple_of(2) {
        (f64::from(sorted[middle - 1]) + f64::from(sorted[middle])) * 0.5
    } else {
        f64::from(sorted[middle])
    };
    MeasurementStatistics {
        minimum,
        maximum,
        mean,
        sum,
        standard_deviation,
        median,
    }
}

fn clipboard_patch(tab: &ImageTab) -> ClipboardPatch {
    let (left, top, width, height) = selection_bounds(tab);
    let mut pixels = Vec::with_capacity(width.saturating_mul(height));
    for y in top..top.saturating_add(height) {
        for x in left..left.saturating_add(width) {
            pixels.push(
                sample_dataset(tab.dataset.as_ref(), x, y, tab.z, tab.t, tab.channel)
                    .unwrap_or(0.0),
            );
        }
    }
    ClipboardPatch {
        width,
        height,
        pixels,
        pixel_type: tab.dataset.metadata.pixel_type,
    }
}

fn clipboard_dataset(patch: &ClipboardPatch) -> Result<DatasetF32, String> {
    let data = Array::from_shape_vec((patch.height, patch.width), patch.pixels.clone())
        .map_err(|error| format!("Clipboard image is invalid: {error}"))?
        .into_dyn();
    Dataset::new(
        data,
        Metadata {
            dims: vec![
                Dim::new(AxisKind::Y, patch.height),
                Dim::new(AxisKind::X, patch.width),
            ],
            pixel_type: patch.pixel_type,
            ..Metadata::default()
        },
    )
    .map_err(|error| format!("Clipboard image is invalid: {error}"))
}

fn active_plane_dataset(tab: &ImageTab) -> Result<DatasetF32, String> {
    let mut pixels = Vec::with_capacity(tab.width.saturating_mul(tab.height));
    for y in 0..tab.height {
        for x in 0..tab.width {
            pixels.push(
                sample_dataset(tab.dataset.as_ref(), x, y, tab.z, tab.t, tab.channel)
                    .unwrap_or(0.0),
            );
        }
    }
    let data = Array::from_shape_vec((tab.height, tab.width), pixels)
        .map_err(|error| format!("Could not extract the active image plane: {error}"))?
        .into_dyn();
    let y_axis = tab.dataset.axis_index(AxisKind::Y).unwrap_or(0);
    let x_axis = tab.dataset.axis_index(AxisKind::X).unwrap_or(1);
    Dataset::new(
        data,
        Metadata {
            dims: vec![
                tab.dataset.metadata.dims[y_axis].clone(),
                tab.dataset.metadata.dims[x_axis].clone(),
            ],
            pixel_type: tab.dataset.metadata.pixel_type,
            source: tab.dataset.metadata.source.clone(),
            extras: tab.dataset.metadata.extras.clone(),
            ..Metadata::default()
        },
    )
    .map_err(|error| format!("Could not extract the active image plane: {error}"))
}

fn tool_for_command(command: &str) -> Option<ToolId> {
    TOOLBAR_ITEMS
        .iter()
        .find(|item| item.tool.command_id() == command)
        .map(|item| item.tool)
}

fn command_label(command_id: &str) -> String {
    menu::manifest_commands()
        .iter()
        .find(|entry| entry.id == command_id)
        .map(|entry| entry.label.clone())
        .unwrap_or_else(|| command_id.to_string())
}

pub(super) fn command_is_routed(command: &str) -> bool {
    if operation_for_command(command).is_some() {
        return true;
    }
    if command.starts_with("window.viewer.") || command.starts_with("plugins.macros.installed.") {
        return true;
    }
    if command.starts_with("launcher.tool.") {
        return tool_for_command(command).is_some();
    }
    matches!(
        command,
        "file.new"
            | "file.open"
            | "file.close"
            | "file.close_all"
            | "file.save"
            | "file.save_as"
            | "file.export.results"
            | "file.revert"
            | "file.quit"
            | "edit.undo"
            | "edit.redo"
            | "edit.cut"
            | "edit.copy"
            | "edit.paste"
            | "edit.internal_clipboard"
            | "edit.selection.all"
            | "edit.clear"
            | "edit.fill"
            | "edit.selection.none"
            | "image.show_info"
            | "image.properties"
            | "image.adjust.brightness"
            | "image.adjust.window_level"
            | "image.duplicate"
            | "image.rename"
            | "image.crop"
            | "image.zoom.in"
            | "image.zoom.out"
            | "image.zoom.reset"
            | "image.zoom.original"
            | "image.zoom.view100"
            | "image.zoom.scale_to_fit"
            | "image.zoom.maximize"
            | "image.lookup.invert_lut"
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
            | "image.lookup.red_green"
            | "image.lookup.apply_lut"
            | "image.overlay.add_selection"
            | "image.overlay.hide"
            | "image.overlay.show"
            | "image.overlay.toggle"
            | "image.overlay.remove"
            | "image.overlay.list"
            | "image.overlay.measure"
            | "image.overlay.from_roi_manager"
            | "image.overlay.to_roi_manager"
            | "analyze.measure"
            | "analyze.set_measurements"
            | "analyze.tools.roi_manager"
            | "analyze.tools.results"
            | "analyze.clear_results"
            | "analyze.summarize"
            | "window.next"
            | "window.previous"
            | "window.put_behind"
            | "window.main"
            | "window.show_all"
            | "process.repeat_command"
            | "image.stacks.next"
            | "image.stacks.previous"
            | "image.stacks.measure_stack"
            | "help.about"
            | "help.docs"
            | "help.shortcuts"
            | "plugins.macros.run"
            | "plugins.macros.record"
            | "plugins.macros.install"
            | "plugins.utilities.startup"
    )
}

fn operation_for_command(command: &str) -> Option<(&'static str, Value)> {
    let operation = match command {
        "edit.invert" => ("intensity.invert", json!({})),
        "image.type.8bit" => ("image.convert", json!({ "target": "u8" })),
        "image.type.16bit" => ("image.convert", json!({ "target": "u16" })),
        "image.type.32bit" => ("image.convert", json!({ "target": "f32" })),
        "image.type.rgb" | "image.color.stack_to_rgb" => {
            ("image.convert", json!({ "target": "rgb" }))
        }
        "image.adjust.threshold" => ("threshold.fixed", json!({})),
        "image.adjust.color_threshold" => ("image.color_threshold", json!({})),
        "image.adjust.size" => ("image.resize", json!({})),
        "image.adjust.canvas" => ("image.canvas_resize", json!({})),
        "image.adjust.coordinates" => ("image.coordinates", json!({})),
        "image.scale" => ("image.scale", json!({})),
        "image.crop" => ("image.crop", json!({})),
        "image.stacks.add_slice" => ("image.stack.add_slice", json!({})),
        "image.stacks.delete_slice" => ("image.stack.delete_slice", json!({})),
        "image.stacks.make_substack" => ("image.stack.substack", json!({})),
        "image.stacks.reslice" => ("image.stack.reslice", json!({})),
        "image.stacks.z_project" => ("image.stack.z_project", json!({})),
        "image.stacks.make_montage" => ("image.stack.montage", json!({})),
        "image.stacks.montage_to_stack" => ("image.stack.montage_to_stack", json!({})),
        "image.stacks.grouped_z_project" => ("image.stack.grouped_z_project", json!({})),
        "image.stacks.reduce" => ("image.stack.reduce", json!({})),
        "image.stacks.plot_z_profile" => ("image.stack.z_profile", json!({})),
        "image.hyperstacks.stack_to_hyperstack" => ("image.stack.to_hyperstack", json!({})),
        "image.hyperstacks.hyperstack_to_stack" => ("image.hyperstack.to_stack", json!({})),
        "image.hyperstacks.reduce_dimensionality" => {
            ("image.hyperstack.reduce_dimensionality", json!({}))
        }
        "image.hyperstacks.make_subset" => ("image.hyperstack.subset", json!({})),
        "image.transform.rotate" => ("image.rotate", json!({})),
        "image.transform.translate" => ("image.translate", json!({})),
        "image.transform.bin" => ("image.bin", json!({})),
        "process.smooth" | "process.gaussian" | "process.filters.gaussian" => {
            ("gaussian.blur", json!({}))
        }
        "process.sharpen" => ("image.sharpen", json!({})),
        "process.find_edges" => ("image.find_edges", json!({})),
        "process.find_maxima" => ("image.find_maxima", json!({})),
        "process.enhance_contrast" => ("intensity.enhance_contrast", json!({})),
        "process.subtract_background" => ("image.subtract_background", json!({})),
        "process.noise.add" | "process.noise.specified" => ("noise.gaussian", json!({})),
        "process.noise.salt_pepper" => ("noise.salt_and_pepper", json!({})),
        "process.noise.despeckle" | "process.filters.median" => {
            ("image.median_filter", json!({ "radius": 1 }))
        }
        "process.noise.remove_nans" => ("image.remove_nans", json!({})),
        "process.noise.remove_outliers" => ("image.remove_outliers", json!({})),
        "process.binary.make" | "process.binary.convert_mask" => {
            ("threshold.make_binary", json!({}))
        }
        "process.binary.erode" => ("morphology.erode", json!({})),
        "process.binary.dilate" => ("morphology.dilate", json!({})),
        "process.binary.open" => ("morphology.open", json!({})),
        "process.binary.close" => ("morphology.close", json!({})),
        "process.binary.median" => ("morphology.binary_median", json!({})),
        "process.binary.outline" => ("morphology.outline", json!({})),
        "process.binary.fill_holes" => ("morphology.fill_holes", json!({})),
        "process.binary.skeletonize" => ("morphology.skeletonize", json!({})),
        "process.binary.distance_map" => ("morphology.distance_map", json!({})),
        "process.binary.ultimate_points" => ("morphology.ultimate_points", json!({})),
        "process.binary.watershed" => ("morphology.watershed", json!({})),
        "process.binary.voronoi" => ("morphology.voronoi", json!({})),
        "process.fft.fft" => ("image.fft_power_spectrum", json!({})),
        "process.fft.bandpass" => ("image.fft_bandpass", json!({})),
        "process.fft.swap_quadrants" => ("image.swap_quadrants", json!({})),
        "process.filters.convolve" => ("image.convolve", json!({})),
        "process.filters.unsharp_mask" => ("image.unsharp_mask", json!({})),
        command
            if matches!(
                command,
                "process.filters.median_3d"
                    | "process.filters.mean_3d"
                    | "process.filters.minimum_3d"
                    | "process.filters.maximum_3d"
                    | "process.filters.variance_3d"
            ) =>
        {
            let filter = command
                .trim_start_matches("process.filters.")
                .trim_end_matches("_3d");
            ("image.rank_filter_3d", json!({ "filter": filter }))
        }
        "process.filters.gaussian_3d" | "process.filters.show_circular_masks" => return None,
        "image.transform.flip_horizontal" => ("image.flip", json!({ "axis": "horizontal" })),
        "image.transform.flip_vertical" => ("image.flip", json!({ "axis": "vertical" })),
        "image.transform.flip_z" => ("image.flip", json!({ "axis": "z" })),
        "image.transform.rotate_right" => ("image.rotate_90", json!({ "direction": "right" })),
        "image.transform.rotate_left" => ("image.rotate_90", json!({ "direction": "left" })),
        "image.stacks.statistics" => ("image.stack.statistics", json!({})),
        "analyze.analyze_particles" => ("measurements.particles", json!({})),
        "analyze.histogram" => ("measurements.histogram", json!({})),
        "analyze.plot_profile" => ("measurements.profile", json!({})),
        "analyze.surface_plot" => ("image.surface_plot", json!({})),
        "analyze.set_scale" => ("image.set_scale", json!({})),
        "analyze.calibrate" => ("image.calibrate", json!({})),
        command if command.starts_with("process.shadows.") && command != "process.shadows.demo" => {
            let direction = command.trim_start_matches("process.shadows.");
            ("image.shadow", json!({ "direction": direction }))
        }
        "process.shadows.demo" => ("image.shadow_demo", json!({})),
        "process.math.nan_background" => ("intensity.nan_background", json!({})),
        command if command.starts_with("process.math.") => {
            let kind = command.trim_start_matches("process.math.");
            ("intensity.math", json!({ "operation": kind }))
        }
        command if command.starts_with("process.filters.") => {
            let filter = command.trim_start_matches("process.filters.");
            ("image.rank_filter", json!({ "filter": filter }))
        }
        _ => return None,
    };
    Some(operation)
}

fn merge_json_objects(target: &mut Value, source: Value) {
    let (Some(target), Some(source)) = (target.as_object_mut(), source.as_object()) else {
        return;
    };
    for (key, value) in source {
        target.entry(key.clone()).or_insert_with(|| value.clone());
    }
}

fn overlay_json_objects(target: &mut Value, source: Value) {
    let (Some(target), Some(source)) = (target.as_object_mut(), source.as_object()) else {
        return;
    };
    for (key, value) in source {
        target.insert(key.clone(), value.clone());
    }
}

fn measurement_settings_from_params(
    params: &Value,
    current: MeasurementSettings,
    clear_unspecified: bool,
) -> MeasurementSettings {
    let missing = |value: bool| if clear_unspecified { false } else { value };
    let option = |keys: &[&str], current: bool| {
        keys.iter()
            .find_map(|key| params.get(*key))
            .and_then(|value| match value {
                Value::Bool(value) => Some(*value),
                Value::Number(value) => value.as_i64().map(|value| value != 0),
                Value::String(value) if value.eq_ignore_ascii_case("true") => Some(true),
                Value::String(value) if value.eq_ignore_ascii_case("false") => Some(false),
                _ => None,
            })
            .unwrap_or_else(|| missing(current))
    };
    let decimal_places = ["decimal_places", "decimal", "precision"]
        .iter()
        .find_map(|key| params.get(*key))
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .map(|value| value.round().clamp(0.0, 9.0) as u8)
        .unwrap_or(current.decimal_places);
    MeasurementSettings {
        area: option(&["area"], current.area),
        mean: option(&["mean"], current.mean),
        standard_deviation: option(
            &["standard_deviation", "standard", "std_dev"],
            current.standard_deviation,
        ),
        min_max: option(&["min_max", "min", "minimum"], current.min_max),
        centroid: option(&["centroid"], current.centroid),
        perimeter: option(&["perimeter"], current.perimeter),
        bounding_rectangle: option(
            &["bounding_rectangle", "bounding", "rect"],
            current.bounding_rectangle,
        ),
        integrated_density: option(
            &["integrated_density", "integrated"],
            current.integrated_density,
        ),
        median: option(&["median"], current.median),
        stack_position: option(
            &["stack_position", "stack", "slice"],
            current.stack_position,
        ),
        display_label: option(
            &["display_label", "display", "labels"],
            current.display_label,
        ),
        invert_y_coordinates: option(
            &["invert_y_coordinates", "invert_y", "invert"],
            current.invert_y_coordinates,
        ),
        decimal_places,
    }
}

fn parameter_fields(defaults: &Value) -> Vec<ParameterField> {
    defaults
        .as_object()
        .into_iter()
        .flat_map(|values| values.iter())
        .filter(|(key, _)| !matches!(key.as_str(), "operation" | "direction" | "filter"))
        .map(|(key, value)| {
            let (kind, text) = match value {
                Value::Bool(value) => (ParameterKind::Boolean, value.to_string()),
                Value::Number(value) => (ParameterKind::Number, value.to_string()),
                Value::String(value) => (ParameterKind::Text, value.clone()),
                Value::Array(_) | Value::Object(_) => (ParameterKind::Json, value.to_string()),
                Value::Null => (ParameterKind::Text, String::new()),
            };
            ParameterField {
                key: key.clone(),
                label: humanize_key(key),
                value: text,
                kind,
            }
        })
        .collect()
}

fn humanize_key(key: &str) -> String {
    let text = key.replace('_', " ");
    let mut characters = text.chars();
    match characters.next() {
        Some(first) => first.to_uppercase().collect::<String>() + characters.as_str(),
        None => String::new(),
    }
}

fn format_compact_number(value: f32) -> String {
    if value.fract().abs() < f32::EPSILON {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

fn format_json_value(value: &Value, decimal_places: u8) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Number(value) if value.is_i64() || value.is_u64() => value.to_string(),
        Value::Number(value) => value
            .as_f64()
            .map(|value| {
                format!(
                    "{value:.precision$}",
                    precision = usize::from(decimal_places)
                )
            })
            .unwrap_or_else(|| value.to_string()),
        _ => value.to_string(),
    }
}

fn common_result_units(
    rows: &[BTreeMap<String, Value>],
) -> Result<BTreeMap<String, Value>, String> {
    let mut shared = BTreeMap::new();
    for (key, label) in [("area_unit", "area units"), ("length_unit", "length units")] {
        let units = rows
            .iter()
            .filter_map(|row| row.get(key).and_then(Value::as_str))
            .map(str::trim)
            .filter(|unit| !unit.is_empty())
            .map(str::to_string)
            .collect::<BTreeSet<_>>();
        if units.len() > 1 {
            return Err(format!(
                "Cannot summarize results with mixed {label}: {}",
                units.into_iter().collect::<Vec<_>>().join(", ")
            ));
        }
        if let Some(unit) = units.into_iter().next() {
            shared.insert(key.to_string(), Value::String(unit));
        }
    }
    Ok(shared)
}

fn result_column_width(rows: &[BTreeMap<String, Value>], column: &str, decimal_places: u8) -> f32 {
    let widest = rows
        .iter()
        .filter_map(|row| row.get(column))
        .map(|value| format_json_value(value, decimal_places))
        .map(|value| value.chars().count())
        .chain(std::iter::once(column.chars().count()))
        .max()
        .unwrap_or(column.len());
    (widest as f32 * 8.0 + 24.0).clamp(96.0, 520.0)
}

fn result_columns(rows: &[BTreeMap<String, Value>]) -> Vec<String> {
    const IMAGEJ_ORDER: &[&str] = &[
        "Label",
        "Area",
        "Mean",
        "StdDev",
        "Min",
        "Max",
        "Mode",
        "X",
        "Y",
        "XM",
        "YM",
        "BX",
        "BY",
        "Width",
        "Height",
        "Perim.",
        "Major",
        "Minor",
        "Circ.",
        "Feret",
        "IntDen",
        "Median",
        "Skew",
        "Kurt",
        "%Area",
        "RawIntDen",
        "Ch",
        "Slice",
        "Frame",
        "Pixels",
        "Length",
        "Angle",
        "C",
        "Z",
        "T",
        "ROI",
        "Type",
        "area_unit",
        "length_unit",
        "Command",
        "Image",
    ];
    let mut discovered = Vec::new();
    for row in rows {
        for key in row.keys() {
            if !discovered.contains(key) {
                discovered.push(key.clone());
            }
        }
    }
    let mut columns = IMAGEJ_ORDER
        .iter()
        .filter(|column| discovered.iter().any(|found| found == **column))
        .map(|column| (*column).to_string())
        .collect::<Vec<_>>();
    let remaining = discovered
        .into_iter()
        .filter(|column| !columns.contains(column))
        .collect::<Vec<_>>();
    columns.extend(remaining);
    columns
}

fn csv_cell(value: &str) -> String {
    if value
        .chars()
        .any(|character| matches!(character, ',' | '"' | '\n' | '\r'))
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn json_f32(params: &Value, key: &str) -> Result<f32, String> {
    let value = params
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing numeric macro parameter `{key}`"))? as f32;
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| format!("macro parameter `{key}` must be finite"))
}

fn json_usize(params: &Value, key: &str, fallback: usize) -> usize {
    params
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .map(|value| value.round().max(1.0) as usize)
        .unwrap_or(fallback)
}

fn parse_pixel_type(value: &str) -> PixelType {
    match value.trim().to_ascii_lowercase().as_str() {
        "u16" | "16" | "16-bit" => PixelType::U16,
        "f32" | "32" | "32-bit" | "float" => PixelType::F32,
        _ => PixelType::U8,
    }
}

fn macro_save_path(path: &str, format: Option<&str>) -> PathBuf {
    let mut path = PathBuf::from(path);
    if path.extension().is_none() {
        let extension = match format.unwrap_or_default().to_ascii_lowercase().as_str() {
            value if value.contains("tif") => "tif",
            value if value.contains("jpeg") || value.contains("jpg") => "jpg",
            _ => "png",
        };
        path.set_extension(extension);
    }
    path
}

fn has_stack_controls(tab: &ImageTab) -> bool {
    tab.slices > 1
        || tab.frames > 1
        || (tab.channels > 1 && !dataset_is_true_rgb(tab.dataset.as_ref()))
}

fn roi_status(selection: &RoiSelection) -> String {
    if selection.points.len() >= 2 {
        let first = selection.points[0];
        let last = *selection.points.last().unwrap_or(&first);
        let width = (last.0 - first.0).abs();
        let height = (last.1 - first.1).abs();
        if matches!(selection.tool, ToolId::Rect | ToolId::Oval) {
            return format!(
                "{}: x={:.0}, y={:.0}, width={width:.0}, height={height:.0}",
                selection.tool.label(),
                first.0.min(last.0),
                first.1.min(last.1),
            );
        }
        let length = (width * width + height * height).sqrt();
        return format!("{}: length={length:.2}", selection.tool.label());
    }
    selection
        .points
        .first()
        .map(|point| {
            format!(
                "{}: x={:.0}, y={:.0}",
                selection.tool.label(),
                point.0,
                point.1
            )
        })
        .unwrap_or_else(|| format!("{} selection", selection.tool.label()))
}

fn paint_roi(
    window: &mut Window,
    bounds: Bounds<Pixels>,
    selection: &RoiSelection,
    zoom: f32,
    display_width: f32,
    display_height: f32,
    color: u32,
) {
    if selection.points.is_empty() {
        return;
    }
    let left = f32::from(bounds.origin.x) + (f32::from(bounds.size.width) - display_width) * 0.5;
    let top = f32::from(bounds.origin.y) + (f32::from(bounds.size.height) - display_height) * 0.5;
    let map =
        |position: (f32, f32)| point(px(left + position.0 * zoom), px(top + position.1 * zoom));
    let mut builder = PathBuilder::stroke(px(1.5)).dash_array(&[px(5.0), px(3.0)]);
    match selection.tool {
        ToolId::Rect if selection.points.len() >= 2 => {
            let first = selection.points[0];
            let last = *selection.points.last().unwrap_or(&first);
            let left = first.0.min(last.0);
            let right = first.0.max(last.0);
            let top = first.1.min(last.1);
            let bottom = first.1.max(last.1);
            builder.move_to(map((left, top)));
            builder.line_to(map((right, top)));
            builder.line_to(map((right, bottom)));
            builder.line_to(map((left, bottom)));
            builder.close();
        }
        ToolId::Oval if selection.points.len() >= 2 => {
            let first = selection.points[0];
            let last = *selection.points.last().unwrap_or(&first);
            let center = ((first.0 + last.0) * 0.5, (first.1 + last.1) * 0.5);
            let radius = (
                (last.0 - first.0).abs() * 0.5,
                (last.1 - first.1).abs() * 0.5,
            );
            for step in 0..=48 {
                let angle = step as f32 / 48.0 * std::f32::consts::TAU;
                let position = (
                    center.0 + radius.0 * angle.cos(),
                    center.1 + radius.1 * angle.sin(),
                );
                if step == 0 {
                    builder.move_to(map(position));
                } else {
                    builder.line_to(map(position));
                }
            }
            builder.close();
        }
        ToolId::Point | ToolId::Wand | ToolId::Text => {
            let center = map(selection.points[0]);
            builder.move_to(point(center.x - px(6.0), center.y));
            builder.line_to(point(center.x + px(6.0), center.y));
            builder.move_to(point(center.x, center.y - px(6.0)));
            builder.line_to(point(center.x, center.y + px(6.0)));
        }
        _ => {
            builder.move_to(map(selection.points[0]));
            for position in selection.points.iter().copied().skip(1) {
                builder.line_to(map(position));
            }
            if selection.tool == ToolId::Poly && selection.points.len() > 2 {
                builder.close();
            }
        }
    }
    if let Ok(path) = builder.build() {
        window.paint_path(path, rgb(color));
    }
}

fn menu_left(menus: &[MenuManifestTopLevel], index: usize) -> f32 {
    menus
        .iter()
        .take(index)
        .map(|menu| menu.label.chars().count() as f32 * 12.5 + 40.0)
        .sum()
}

fn find_submenu<'a>(items: &'a [MenuManifestItem], id: &str) -> Option<&'a [MenuManifestItem]> {
    for item in items {
        if item.id.as_deref() == Some(id) && item.kind == "submenu" {
            return item.items.as_deref();
        }
        if let Some(children) = item.items.as_deref()
            && let Some(found) = find_submenu(children, id)
        {
            return Some(found);
        }
    }
    None
}

fn install_key_bindings(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("secondary-n", NewImage, Some("ImageJ")),
        KeyBinding::new("secondary-o", OpenImage, Some("ImageJ")),
        KeyBinding::new("secondary-s", Save, Some("ImageJ")),
        KeyBinding::new("secondary-shift-s", SaveAs, Some("ImageJ")),
        KeyBinding::new("secondary-w", CloseImage, Some("ImageJ")),
        KeyBinding::new("secondary-x", Cut, Some("ImageJ")),
        KeyBinding::new("secondary-c", Copy, Some("ImageJ")),
        KeyBinding::new("secondary-v", Paste, Some("ImageJ")),
        KeyBinding::new("secondary-z", Undo, Some("ImageJ")),
        KeyBinding::new("secondary-shift-z", Redo, Some("ImageJ")),
        KeyBinding::new("+", ZoomIn, Some("ImageJ")),
        KeyBinding::new("-", ZoomOut, Some("ImageJ")),
        KeyBinding::new("4", ZoomActual, Some("ImageJ")),
        KeyBinding::new("5", ZoomActual, Some("ImageJ")),
        KeyBinding::new("secondary-0", ZoomFit, Some("ImageJ")),
        KeyBinding::new("ctrl-tab", NextTab, Some("ImageJ")),
        KeyBinding::new("ctrl-shift-tab", PreviousTab, Some("ImageJ")),
        KeyBinding::new("escape", Escape, Some("ImageJ")),
        KeyBinding::new("secondary-q", Quit, Some("ImageJ")),
    ]);
}

pub fn run(startup_input: Option<PathBuf>) -> Result<(), String> {
    run_with_ops(startup_input, OpsService::default())
}

pub fn run_with_ops(startup_input: Option<PathBuf>, ops_service: OpsService) -> Result<(), String> {
    gpui_platform::application().run(move |cx: &mut App| {
        install_key_bindings(cx);
        cx.set_menus(vec![
            Menu {
                name: "File".into(),
                items: vec![
                    MenuItem::action("New", NewImage),
                    MenuItem::action("Open…", OpenImage),
                    MenuItem::action("Save", Save),
                    MenuItem::action("Save As…", SaveAs),
                    MenuItem::action("Close", CloseImage),
                    MenuItem::separator(),
                    MenuItem::action("Quit", Quit),
                ],
                disabled: false,
            },
            Menu {
                name: "Edit".into(),
                items: vec![
                    MenuItem::action("Undo", Undo),
                    MenuItem::action("Redo", Redo),
                ],
                disabled: false,
            },
        ]);
        let bounds = Bounds::new(point(px(80.0), px(80.0)), size(px(900.0), px(122.0)));
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            window_min_size: Some(size(px(700.0), px(122.0))),
            kind: WindowKind::Floating,
            is_resizable: false,
            titlebar: Some(TitlebarOptions {
                title: Some(SharedString::from(APP_TITLE)),
                appears_transparent: false,
                ..Default::default()
            }),
            app_id: Some("image-rs".into()),
            ..Default::default()
        };
        let launcher_ops = ops_service.clone();
        let launcher = cx
            .open_window(options, move |window, cx| {
                let ops_service = launcher_ops.clone();
                cx.new(|cx| ImageJApp::new(window, ops_service, cx))
            })
            .expect("failed to open GPUI ImageJ window");
        let app = launcher
            .entity(cx)
            .expect("launcher root should be the ImageJ application");
        let launcher_window_id = launcher.window_id();
        let weak_app = app.downgrade();
        let quit_app = weak_app.clone();
        cx.on_action(move |_: &Quit, cx| {
            if let Some(app) = quit_app.upgrade() {
                let _ = app.update(cx, |app, cx| app.request_quit(cx));
            } else {
                cx.quit();
            }
        });
        cx.on_window_closed(move |cx, window_id| {
            if window_id == launcher_window_id {
                cx.quit();
                return;
            }
            let weak_app = weak_app.clone();
            cx.defer(move |cx| {
                if let Some(app) = weak_app.upgrade() {
                    app.update(cx, |app, cx| app.handle_window_closed(window_id, cx));
                }
            });
        })
        .detach();
        let _ = launcher.update(cx, |app, _, cx| {
            if let Some(path) = startup_input.clone() {
                app.open_paths([path], cx);
            }
            let startup_macro = macros::startup_macro_path();
            if startup_macro.exists() {
                app.run_macro_path(&startup_macro, cx);
                if app.dialog.is_some() {
                    app.open_dialog_window(cx);
                }
            }
        });
        cx.activate(true);
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measurement_test_tab(width: usize, height: usize, pixels: Vec<f32>) -> ImageTab {
        let data = Array::from_shape_vec((height, width), pixels)
            .unwrap()
            .into_dyn();
        let dataset = Dataset::new(
            data,
            Metadata {
                dims: vec![Dim::new(AxisKind::Y, height), Dim::new(AxisKind::X, width)],
                pixel_type: PixelType::F32,
                ..Metadata::default()
            },
        )
        .unwrap();
        ImageTab::from_dataset(1, None, "Measurement test".into(), dataset).unwrap()
    }

    #[test]
    fn path_identity_normalizes_existing_and_not_yet_created_targets() {
        let temp = tempfile::tempdir().unwrap();
        let image_directory = temp.path().join("images");
        std::fs::create_dir(&image_directory).unwrap();
        let existing = image_directory.join("existing.tif");
        std::fs::write(&existing, b"image").unwrap();

        let existing_alias = image_directory
            .join("..")
            .join("images")
            .join("existing.tif");
        assert_eq!(
            normalized_path_identity(&existing_alias),
            existing.canonicalize().unwrap()
        );

        let new_alias = image_directory.join(".").join("new-image.tif");
        assert_eq!(
            normalized_path_identity(&new_alias),
            image_directory
                .canonicalize()
                .unwrap()
                .join("new-image.tif")
        );
    }

    #[test]
    fn path_owner_excludes_current_tab_but_detects_another_live_tab() {
        let temp = tempfile::tempdir().unwrap();
        let owned_path = temp.path().join("owned.tif");
        let requested_alias = temp.path().join(".").join("owned.tif");

        assert_eq!(
            other_path_owner([(7, owned_path.as_path())], 11, &requested_alias),
            Some(7)
        );
        assert_eq!(
            other_path_owner([(7, owned_path.as_path())], 7, &requested_alias),
            None
        );
    }

    #[test]
    fn processing_area_rasterizes_exact_area_membership() {
        let rectangle = RoiSelection {
            tool: ToolId::Rect,
            points: vec![(1.0, 1.0), (3.0, 3.0)],
        };
        let mask = rasterize_processing_area(Some(&rectangle), 5, 5)
            .unwrap()
            .unwrap();
        assert_eq!(
            mask,
            RasterizedAreaMask {
                left: 1,
                top: 1,
                width: 2,
                height: 2,
                members: vec![1, 1, 1, 1],
            }
        );

        let oval = RoiSelection {
            tool: ToolId::Oval,
            points: vec![(0.0, 0.0), (4.0, 4.0)],
        };
        let mask = rasterize_processing_area(Some(&oval), 4, 4)
            .unwrap()
            .unwrap();
        assert_eq!((mask.left, mask.top, mask.width, mask.height), (0, 0, 4, 4));
        assert_eq!(mask.members[0], 0);
        assert_eq!(mask.members[5], 1);
        assert_eq!(mask.members[15], 0);
    }

    #[test]
    fn processing_area_rejects_non_area_tools() {
        let line = RoiSelection {
            tool: ToolId::Line,
            points: vec![(0.0, 0.0), (3.0, 3.0)],
        };
        let error = rasterize_processing_area(Some(&line), 4, 4).unwrap_err();
        assert!(error.contains("not an area selection"));
        assert_eq!(rasterize_processing_area(None, 4, 4).unwrap(), None);
    }

    #[test]
    fn failed_viewer_rollback_removes_only_failed_session_and_does_not_reuse_its_id() {
        let mut first = measurement_test_tab(1, 1, vec![1.0]);
        first.id = 41;
        first.internal_label = "viewer-41".into();
        let mut failed = measurement_test_tab(1, 1, vec![2.0]);
        failed.id = 42;
        failed.internal_label = "viewer-42".into();
        let mut tabs = vec![first, failed];
        let mut activation_order = vec![41, 42];
        let mut active_tab = Some(42);
        let next_tab_id = 42_u64;

        let removed =
            rollback_failed_tab_state(&mut tabs, &mut activation_order, &mut active_tab, 42)
                .unwrap();

        assert_eq!(removed.id, 42);
        assert_eq!(tabs.iter().map(|tab| tab.id).collect::<Vec<_>>(), vec![41]);
        assert_eq!(activation_order, vec![41]);
        assert_eq!(active_tab, Some(41));
        assert_eq!(next_tab_id.saturating_add(1), 43);
    }

    fn measured_row(tab: &ImageTab, selection: RoiSelection) -> BTreeMap<String, Value> {
        measure_roi_on_tab(
            tab,
            &selection,
            "ROI-1",
            "ROI Manager Measure",
            1,
            RoiPosition {
                channel: 0,
                z: 0,
                t: 0,
            },
            &MeasurementSettings::all_supported(),
        )
        .unwrap()
    }

    #[test]
    fn renderer_preserves_rgb_channels() {
        let data = Array::from_shape_vec((1, 1, 3), vec![255.0, 64.0, 4.0])
            .unwrap()
            .into_dyn();
        let metadata = Metadata {
            dims: vec![
                Dim::new(AxisKind::Y, 1),
                Dim::new(AxisKind::X, 1),
                Dim::new(AxisKind::Channel, 3),
            ],
            pixel_type: PixelType::U8,
            channel_names: vec!["R".into(), "G".into(), "B".into()],
            ..Metadata::default()
        };
        let dataset = Dataset::new(data, metadata).unwrap();
        assert!(dataset_is_true_rgb(&dataset));
        let plane =
            render_dataset_plane(&dataset, 0, 0, 0, LookupTable::Grays, false, 0.0, 255.0).unwrap();
        assert_eq!(plane.width, 1);
        assert_eq!(plane.height, 1);
        assert_eq!(plane.image.as_bytes(0), Some([4, 64, 255, 255].as_slice()));

        let mut labeled_u16 = dataset.clone();
        labeled_u16.metadata.pixel_type = PixelType::U16;
        assert!(!dataset_is_true_rgb(&labeled_u16));
    }

    #[test]
    fn unlabeled_three_channel_data_is_a_channel_image_not_rgb() {
        let data = Array::from_shape_vec((1, 1, 3), vec![255.0, 64.0, 4.0])
            .unwrap()
            .into_dyn();
        let metadata = Metadata {
            dims: vec![
                Dim::new(AxisKind::Y, 1),
                Dim::new(AxisKind::X, 1),
                Dim::new(AxisKind::Channel, 3),
            ],
            pixel_type: PixelType::U8,
            ..Metadata::default()
        };
        let dataset = Dataset::new(data, metadata).unwrap();

        assert!(!dataset_is_true_rgb(&dataset));
        let plane =
            render_dataset_plane(&dataset, 0, 0, 1, LookupTable::Grays, false, 0.0, 255.0).unwrap();
        assert_eq!(plane.image.as_bytes(0), Some([64, 64, 64, 255].as_slice()));
    }

    #[test]
    fn integer_and_float_defaults_use_the_active_plane() {
        let values = vec![10.0, 100.0, 20.0, 200.0, 30.0, 300.0, 40.0, 400.0];
        for pixel_type in [PixelType::U16, PixelType::F32] {
            let data = Array::from_shape_vec((1, 2, 2, 2), values.clone())
                .unwrap()
                .into_dyn();
            let metadata = Metadata {
                dims: vec![
                    Dim::new(AxisKind::Y, 1),
                    Dim::new(AxisKind::X, 2),
                    Dim::new(AxisKind::Z, 2),
                    Dim::new(AxisKind::Channel, 2),
                ],
                pixel_type,
                ..Metadata::default()
            };
            let dataset = Dataset::new(data, metadata).unwrap();

            assert_eq!(default_display_range(&dataset, 0, 0, 0), (10.0, 30.0));
            assert_eq!(default_display_range(&dataset, 1, 0, 1), (200.0, 400.0));
        }
    }

    #[test]
    fn equal_display_bounds_render_as_a_step() {
        assert_eq!(display_sample(4.0, 5.0, 5.0), 0);
        assert_eq!(display_sample(5.0, 5.0, 5.0), 0);
        assert_eq!(display_sample(6.0, 5.0, 5.0), 255);
    }

    #[test]
    fn scientific_channels_keep_independent_display_ranges() {
        let data = Array::from_shape_vec((1, 2, 2), vec![0.0, 100.0, 10.0, 200.0])
            .unwrap()
            .into_dyn();
        let dataset = Dataset::new(
            data,
            Metadata {
                dims: vec![
                    Dim::new(AxisKind::Y, 1),
                    Dim::new(AxisKind::X, 2),
                    Dim::new(AxisKind::Channel, 2),
                ],
                pixel_type: PixelType::F32,
                channel_names: vec!["DAPI".into(), "FITC".into()],
                ..Metadata::default()
            },
        )
        .unwrap();
        let mut tab = ImageTab::from_dataset(1, None, "Composite".into(), dataset).unwrap();

        assert_eq!(tab.display_ranges, vec![(0.0, 10.0), (100.0, 200.0)]);
        tab.set_display_range(-5.0, 5.0);
        tab.channel = 1;
        assert_eq!(tab.display_range(), (100.0, 200.0));
        tab.set_display_range(120.0, 180.0);
        tab.channel = 0;
        assert_eq!(tab.display_range(), (-5.0, 5.0));

        tab.reset_display_range();
        assert_eq!(tab.display_range(), (0.0, 10.0));
        tab.channel = 1;
        assert_eq!(tab.display_range(), (120.0, 180.0));
    }

    #[test]
    fn true_rgb_uses_one_shared_display_range() {
        let data = Array::from_shape_vec((1, 1, 3), vec![255.0, 64.0, 4.0])
            .unwrap()
            .into_dyn();
        let dataset = Dataset::new(
            data,
            Metadata {
                dims: vec![
                    Dim::new(AxisKind::Y, 1),
                    Dim::new(AxisKind::X, 1),
                    Dim::new(AxisKind::Channel, 3),
                ],
                pixel_type: PixelType::U8,
                channel_names: vec!["R".into(), "G".into(), "B".into()],
                ..Metadata::default()
            },
        )
        .unwrap();
        let mut tab = ImageTab::from_dataset(1, None, "RGB".into(), dataset).unwrap();

        assert_eq!(tab.display_ranges, vec![(0.0, 255.0)]);
        tab.set_display_range(10.0, 200.0);
        tab.channel = 2;
        assert_eq!(tab.display_range(), (10.0, 200.0));
    }

    #[test]
    fn contrast_slider_supports_windows_wider_than_the_data_domain() {
        let span = 100.0;
        assert_eq!(contrast_window_from_fraction(span, 0.5), span);
        assert_eq!(contrast_window_from_fraction(span, 0.25), 200.0);
        assert_eq!(contrast_window_from_fraction(span, 0.75), 50.0);
        assert!(contrast_window_from_fraction(span, 0.0) > span);

        for width in [25.0, 50.0, 100.0, 200.0, 1_600.0] {
            let fraction = contrast_fraction_from_window(span, width);
            let round_trip = contrast_window_from_fraction(span, fraction);
            assert!((round_trip - width).abs() < 0.001, "width={width}");
        }
    }

    #[test]
    fn apply_lut_mapping_matches_imagej_integer_math() {
        assert_eq!(apply_lut_sample(49.0, 50.0, 150.0, 255.0), 0.0);
        assert_eq!(apply_lut_sample(50.0, 50.0, 150.0, 255.0), 0.0);
        assert_eq!(apply_lut_sample(100.0, 50.0, 150.0, 255.0), 128.0);
        assert_eq!(apply_lut_sample(150.0, 50.0, 150.0, 255.0), 255.0);
        assert_eq!(apply_lut_sample(7.0, 7.0, 7.0, 65_535.0), 0.0);
        assert_eq!(apply_lut_sample(8.0, 7.0, 7.0, 65_535.0), 65_535.0);
    }

    #[test]
    fn display_adjuster_histogram_and_auto_range_follow_the_active_plane() {
        let tab = measurement_test_tab(4, 2, vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]);
        let histogram = display_histogram(&tab, 8);
        assert_eq!(histogram.iter().sum::<usize>(), 8);
        assert_eq!(
            auto_display_range(&tab, AUTO_THRESHOLD_DIVISOR),
            Some((0.0, 13.0))
        );

        let constant = measurement_test_tab(2, 2, vec![7.0; 4]);
        let domain = display_control_domain(&constant);
        assert!(domain.0 < 7.0 && domain.1 > 7.0);
        assert_eq!(
            auto_display_range(&constant, AUTO_THRESHOLD_DIVISOR),
            Some((7.0, 7.0))
        );
    }

    #[test]
    fn display_histogram_and_auto_range_are_scoped_to_the_roi() {
        let mut tab = measurement_test_tab(4, 4, (0..16).map(|value| value as f32).collect());
        tab.roi = Some(RoiSelection {
            tool: ToolId::Rect,
            points: vec![(1.0, 1.0), (3.0, 3.0)],
        });

        assert_eq!(displayed_plane_values(&tab), vec![5.0, 6.0, 9.0, 10.0]);
        assert_eq!(display_histogram(&tab, 16).iter().sum::<usize>(), 4);
        assert_eq!(
            auto_display_range(&tab, AUTO_THRESHOLD_DIVISOR),
            Some((5.0, 10.0))
        );
    }

    #[test]
    fn repeated_auto_progressively_rejects_sparse_tail_bins() {
        let mut values = Vec::with_capacity(10_000);
        values.extend(std::iter::repeat_n(0.0, 3));
        values.extend(std::iter::repeat_n(10.0, 997));
        for value in [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0] {
            values.extend(std::iter::repeat_n(value, 1_000));
        }
        values.extend(std::iter::repeat_n(90.0, 997));
        values.extend(std::iter::repeat_n(100.0, 3));
        let tab = measurement_test_tab(100, 100, values);

        let first = auto_display_range(&tab, AUTO_THRESHOLD_DIVISOR).unwrap();
        let repeated = auto_display_range(&tab, AUTO_THRESHOLD_DIVISOR / 2).unwrap();
        assert!(repeated.0 > first.0);
        assert!(repeated.1 < first.1);
    }

    #[test]
    fn rgb_histogram_and_auto_use_weighted_luminance() {
        let data = Array::from_shape_vec((1, 2, 3), vec![255.0, 0.0, 0.0, 0.0, 255.0, 0.0])
            .unwrap()
            .into_dyn();
        let dataset = Dataset::new(
            data,
            Metadata {
                dims: vec![
                    Dim::new(AxisKind::Y, 1),
                    Dim::new(AxisKind::X, 2),
                    Dim::new(AxisKind::Channel, 3),
                ],
                pixel_type: PixelType::U8,
                channel_names: vec!["R".into(), "G".into(), "B".into()],
                ..Metadata::default()
            },
        )
        .unwrap();
        let mut tab = ImageTab::from_dataset(1, None, "RGB".into(), dataset).unwrap();

        let values = displayed_plane_values(&tab);
        assert_eq!(values.len(), 2);
        assert!((values[0] - 76.245).abs() < 0.001);
        assert!((values[1] - 149.685).abs() < 0.001);
        let auto = auto_display_range(&tab, AUTO_THRESHOLD_DIVISOR).unwrap();
        assert!((auto.0 - 76.245).abs() < 0.001);
        assert!((auto.1 - 149.685).abs() < 0.001);

        tab.roi = Some(RoiSelection {
            tool: ToolId::Rect,
            points: vec![(0.0, 0.0), (1.0, 1.0)],
        });
        let roi_values = displayed_plane_values(&tab);
        assert_eq!(roi_values.len(), 1);
        assert!((roi_values[0] - 76.245).abs() < 0.001);
    }

    #[test]
    fn imagej_process_commands_reuse_core_ops() {
        assert_eq!(
            operation_for_command("process.smooth").unwrap().0,
            "gaussian.blur"
        );
        assert_eq!(
            operation_for_command("process.binary.erode").unwrap().0,
            "morphology.erode"
        );
        assert!(operation_for_command("help.about").is_none());
    }

    #[test]
    fn operation_scope_selection_supports_restricted_plugin_capabilities() {
        let descriptor = |scopes| OperationDescriptor {
            schema: crate::commands::OpSchema {
                name: "test.scope".into(),
                description: "scope test".into(),
                params: Vec::new(),
            },
            scopes,
            area_mask: crate::runtime::AreaMaskSupport::Unsupported,
        };

        let active_and_stack = descriptor(vec![
            OperationScope::ActivePlane,
            OperationScope::ZStack,
            OperationScope::AllPlanes,
        ]);
        assert_eq!(
            choose_operation_scope(&active_and_stack, false),
            Some(OperationScope::ActivePlane)
        );
        assert_eq!(
            choose_operation_scope(&active_and_stack, true),
            Some(OperationScope::ZStack)
        );

        let stack_only = descriptor(vec![OperationScope::ZStack]);
        assert_eq!(
            choose_operation_scope(&stack_only, false),
            Some(OperationScope::ZStack)
        );
        assert_eq!(
            choose_operation_scope(&stack_only, true),
            Some(OperationScope::ZStack)
        );

        let all_planes_only = descriptor(vec![OperationScope::AllPlanes]);
        assert_eq!(
            choose_operation_scope(&all_planes_only, false),
            Some(OperationScope::AllPlanes)
        );
        assert_eq!(choose_operation_scope(&all_planes_only, true), None);
    }

    #[test]
    fn recorded_macro_scope_replays_as_active_plane_or_z_stack() {
        let catalog = command_registry::command_catalog();
        let descriptor = OpsService::default().describe("gaussian.blur").unwrap();

        let active = macros::parse_macro_source(r#"run("Smooth");"#, &catalog).unwrap();
        let mut active_params = active[0].params.clone().unwrap_or_else(|| json!({}));
        assert!(!take_process_stack_parameter(&mut active_params));
        assert_eq!(
            choose_operation_scope(&descriptor, false),
            Some(OperationScope::ActivePlane)
        );

        let recorded = macros::macro_record_line_for_command(
            "process.smooth",
            Some(&json!({ "__image_rs_process_stack": true })),
            &catalog,
        )
        .unwrap();
        let stack = macros::parse_macro_source(&recorded, &catalog).unwrap();
        let mut stack_params = stack[0].params.clone().unwrap();
        assert!(take_process_stack_parameter(&mut stack_params));
        assert_eq!(
            choose_operation_scope(&descriptor, true),
            Some(OperationScope::ZStack)
        );
    }

    #[test]
    fn operation_job_identity_rejects_aba_progress_updates() {
        let input = measurement_test_tab(1, 1, vec![1.0]).dataset;
        let operation = ActiveOperation {
            job_id: 12,
            revision: 4,
            input: input.clone(),
            cancellation: CancellationToken::default(),
            progress: 0.5,
            message: "running".into(),
        };

        assert!(active_operation_matches(&operation, 12, 4, &input));
        assert!(!active_operation_matches(&operation, 13, 4, &input));
        assert!(!active_operation_matches(&operation, 12, 5, &input));
        let equal_but_distinct = Arc::new(input.as_ref().clone());
        assert!(!active_operation_matches(
            &operation,
            12,
            4,
            &equal_but_distinct
        ));
    }

    #[test]
    fn every_operation_route_names_a_registered_core_operation() {
        let operations = crate::commands::list_operations()
            .into_iter()
            .map(|schema| schema.name)
            .collect::<std::collections::HashSet<_>>();

        for entry in menu::manifest_commands() {
            if let Some((operation, _)) = operation_for_command(&entry.id) {
                assert!(
                    operations.contains(operation),
                    "{} routes to missing core operation {operation}",
                    entry.id
                );
            }
        }
    }

    #[test]
    fn parameter_fields_preserve_structured_json_values() {
        let fields = parameter_fields(&json!({
            "kernel": [0.0, 1.0, 0.0],
            "settings": {"normalize": true}
        }));

        assert!(fields.iter().all(|field| field.kind == ParameterKind::Json));
        for field in fields {
            serde_json::from_str::<Value>(&field.value).expect("valid structured JSON field");
        }
    }

    #[test]
    fn parameter_fields_keep_optional_nulls_editable() {
        let fields = parameter_fields(&json!({
            "channels": null,
            "operation": "internal"
        }));
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].key, "channels");
        assert_eq!(fields[0].kind, ParameterKind::Text);
        assert!(fields[0].value.is_empty());
    }

    #[test]
    fn imagej_lookup_tables_use_their_real_channel_layouts() {
        assert_eq!(lut_color(LookupTable::Rgb332, 0b1110_0111), (224, 32, 192));
        assert_eq!(lut_color(LookupTable::RedGreen, 127), (254, 0, 0));
        assert_eq!(lut_color(LookupTable::RedGreen, 128), (0, 0, 0));
        assert_eq!(lut_color(LookupTable::RedGreen, 255), (0, 254, 0));
        assert!(command_is_routed("image.lookup.apply_lut"));
        assert!(!command_is_routed("image.overlay.flatten"));
    }

    #[test]
    fn result_columns_are_stable_and_csv_cells_are_escaped() {
        let rows = vec![
            BTreeMap::from([
                ("Area".to_string(), json!(12.0)),
                ("Label".to_string(), json!("first")),
            ]),
            BTreeMap::from([("Mean".to_string(), json!(4.5))]),
        ];
        assert_eq!(result_columns(&rows), vec!["Label", "Area", "Mean"]);
        let particle_rows = vec![BTreeMap::from([
            ("Image".to_string(), json!("long-image-name.tif")),
            ("Command".to_string(), json!("Analyze Particles")),
            ("Pixels".to_string(), json!(12)),
            ("Circ.".to_string(), json!(0.9)),
            ("BX".to_string(), json!(4.0)),
            ("X".to_string(), json!(5.0)),
            ("area_unit".to_string(), json!("um^2")),
        ])];
        assert_eq!(
            result_columns(&particle_rows),
            vec![
                "X",
                "BX",
                "Circ.",
                "Pixels",
                "area_unit",
                "Command",
                "Image"
            ]
        );
        assert!(
            result_column_width(&particle_rows, "Image", 3)
                > result_column_width(&particle_rows, "X", 3)
        );
        assert_eq!(csv_cell("plain"), "plain");
        assert_eq!(csv_cell("a,b"), "\"a,b\"");
        assert_eq!(csv_cell("a\"b"), "\"a\"\"b\"");
    }

    #[test]
    fn result_units_must_match_before_summary() {
        let compatible = vec![
            BTreeMap::from([
                ("Area".to_string(), json!(2.0)),
                ("area_unit".to_string(), json!("um^2")),
                ("length_unit".to_string(), json!("um")),
            ]),
            BTreeMap::from([
                ("Area".to_string(), json!(3.0)),
                ("area_unit".to_string(), json!("um^2")),
                ("length_unit".to_string(), json!("um")),
            ]),
        ];
        assert_eq!(
            common_result_units(&compatible).unwrap(),
            BTreeMap::from([
                ("area_unit".to_string(), json!("um^2")),
                ("length_unit".to_string(), json!("um")),
            ])
        );

        let mixed = vec![
            BTreeMap::from([("area_unit".to_string(), json!("px^2"))]),
            BTreeMap::from([("area_unit".to_string(), json!("um^2"))]),
        ];
        assert!(
            common_result_units(&mixed)
                .unwrap_err()
                .contains("mixed area units")
        );
    }

    #[test]
    fn submenu_lookup_finds_nested_imagej_entries() {
        let image = menu::manifest()
            .iter()
            .find(|menu| menu.label == "Image")
            .unwrap();
        assert!(find_submenu(&image.items, "image.stacks.tools").is_some());
    }

    #[test]
    fn measurement_defaults_match_imagej_and_precision_only_changes_formatting() {
        let settings = MeasurementSettings::default();
        assert!(settings.area);
        assert!(settings.mean);
        assert!(settings.min_max);
        assert!(!settings.standard_deviation);
        assert!(!settings.centroid);
        assert!(!settings.perimeter);
        assert!(!settings.bounding_rectangle);
        assert!(!settings.integrated_density);
        assert!(!settings.median);
        assert!(!settings.stack_position);
        assert!(!settings.display_label);
        assert!(!settings.invert_y_coordinates);
        assert_eq!(settings.decimal_places, 3);

        let tab = measurement_test_tab(2, 2, vec![0.0, 1.0, 2.0, 3.0]);
        let selection = active_measurement_selection(&tab);
        let row = measure_roi_on_tab(
            &tab,
            &selection,
            "default",
            "Measure",
            1,
            RoiPosition {
                channel: 0,
                z: 0,
                t: 0,
            },
            &settings,
        )
        .unwrap();
        let imagej_cells = row
            .keys()
            .filter(|key| {
                !matches!(
                    key.as_str(),
                    "Image" | "Command" | "ROI" | "Type" | "area_unit" | "length_unit"
                )
            })
            .cloned()
            .collect::<BTreeSet<_>>();
        assert_eq!(
            imagej_cells,
            BTreeSet::from([
                "Area".to_string(),
                "Max".to_string(),
                "Mean".to_string(),
                "Min".to_string(),
            ])
        );
        assert_eq!(format_json_value(&json!(1.23456), 3), "1.235");
        assert_eq!(format_json_value(&json!(2), 7), "2");
        assert_eq!(row.get("Mean").and_then(Value::as_f64), Some(1.5));
    }

    #[test]
    fn set_measurements_params_replace_the_application_wide_selection() {
        let settings = measurement_settings_from_params(
            &json!({
                "std_dev": true,
                "bounding": true,
                "integrated": true,
                "labels": true,
                "precision": 5
            }),
            MeasurementSettings::default(),
            true,
        );
        assert!(!settings.area);
        assert!(!settings.mean);
        assert!(!settings.min_max);
        assert!(settings.standard_deviation);
        assert!(settings.bounding_rectangle);
        assert!(settings.integrated_density);
        assert!(settings.display_label);
        assert_eq!(settings.decimal_places, 5);

        let updated = measurement_settings_from_params(&json!({"area": false}), settings, false);
        assert!(!updated.area);
        assert!(updated.standard_deviation);
        assert_eq!(updated.decimal_places, 5);
    }

    #[test]
    fn imagej_measurement_macros_route_to_the_native_workflow() {
        let invocations = macros::parse_macro_source(
            r#"
                run("Set Measurements...", "area standard min bounding integrated stack display decimal=4");
                run("Measure");
                run("Measure Stack");
            "#,
            &command_registry::command_catalog(),
        )
        .unwrap();
        assert_eq!(invocations[0].command_id, "analyze.set_measurements");
        assert_eq!(invocations[1].command_id, "analyze.measure");
        assert_eq!(invocations[2].command_id, "image.stacks.measure_stack");
        let settings = measurement_settings_from_params(
            invocations[0].params.as_ref().unwrap(),
            MeasurementSettings::default(),
            true,
        );
        assert!(settings.area);
        assert!(settings.standard_deviation);
        assert!(settings.min_max);
        assert!(settings.bounding_rectangle);
        assert!(settings.integrated_density);
        assert!(settings.stack_position);
        assert!(settings.display_label);
        assert_eq!(settings.decimal_places, 4);
    }

    #[test]
    fn measurement_is_scoped_to_the_active_channel_slice_time_and_exact_roi() {
        let mut pixels = Vec::new();
        for y in 0..2 {
            for x in 0..2 {
                for z in 0..2 {
                    for channel in 0..2 {
                        for time in 0..2 {
                            pixels.push((channel * 1_000 + z * 100 + time * 10 + y * 2 + x) as f32);
                        }
                    }
                }
            }
        }
        let dataset = Dataset::new(
            Array::from_shape_vec((2, 2, 2, 2, 2), pixels)
                .unwrap()
                .into_dyn(),
            Metadata {
                dims: vec![
                    Dim::new(AxisKind::Y, 2),
                    Dim::new(AxisKind::X, 2),
                    Dim::new(AxisKind::Z, 2),
                    Dim::new(AxisKind::Channel, 2),
                    Dim::new(AxisKind::Time, 2),
                ],
                pixel_type: PixelType::F32,
                ..Metadata::default()
            },
        )
        .unwrap();
        let mut tab = ImageTab::from_dataset(1, None, "Hyperstack".into(), dataset).unwrap();
        tab.channel = 1;
        tab.z = 1;
        tab.t = 1;
        let selection = RoiSelection {
            tool: ToolId::Rect,
            points: vec![(0.0, 0.0), (2.0, 2.0)],
        };
        let row = measure_roi_on_tab(
            &tab,
            &selection,
            "active",
            "Measure",
            1,
            RoiPosition {
                channel: tab.channel,
                z: tab.z,
                t: tab.t,
            },
            &MeasurementSettings::default(),
        )
        .unwrap();

        assert_eq!(row.get("Area"), Some(&json!(4.0)));
        assert_eq!(row.get("Mean"), Some(&json!(1_111.5)));
        assert_eq!((tab.channel, tab.z, tab.t), (1, 1, 1));
    }

    #[test]
    fn stack_position_omits_singleton_slice_for_channel_time_hyperstacks() {
        let dataset = Dataset::new(
            Array::from_shape_vec((1, 1, 2, 2), vec![1.0, 2.0, 3.0, 4.0])
                .unwrap()
                .into_dyn(),
            Metadata {
                dims: vec![
                    Dim::new(AxisKind::Y, 1),
                    Dim::new(AxisKind::X, 1),
                    Dim::new(AxisKind::Channel, 2),
                    Dim::new(AxisKind::Time, 2),
                ],
                pixel_type: PixelType::F32,
                ..Metadata::default()
            },
        )
        .unwrap();
        let tab = ImageTab::from_dataset(1, None, "C/T hyperstack".into(), dataset).unwrap();
        let row = measure_roi_on_tab(
            &tab,
            &active_measurement_selection(&tab),
            "active",
            "Measure",
            1,
            RoiPosition {
                channel: 1,
                z: 0,
                t: 1,
            },
            &MeasurementSettings::all_supported(),
        )
        .unwrap();

        assert_eq!(row.get("Ch"), Some(&json!(2)));
        assert_eq!(row.get("Frame"), Some(&json!(2)));
        assert!(!row.contains_key("Slice"));
    }

    #[test]
    fn zero_intensity_pixels_still_contribute_to_measured_area() {
        let tab = measurement_test_tab(2, 2, vec![0.0; 4]);
        let row = measure_roi_on_tab(
            &tab,
            &active_measurement_selection(&tab),
            "zeros",
            "Measure",
            1,
            RoiPosition {
                channel: 0,
                z: 0,
                t: 0,
            },
            &MeasurementSettings::default(),
        )
        .unwrap();
        assert_eq!(row.get("Area"), Some(&json!(4.0)));
        assert_eq!(row.get("Mean"), Some(&json!(0.0)));
    }

    #[test]
    fn rectangle_measurement_uses_only_pixels_inside_the_roi() {
        let tab = measurement_test_tab(4, 4, (1..=16).map(|value| value as f32).collect());
        let row = measured_row(
            &tab,
            RoiSelection {
                tool: ToolId::Rect,
                points: vec![(1.0, 1.0), (3.0, 3.0)],
            },
        );

        assert_eq!(row.get("Area"), Some(&json!(4.0)));
        assert_eq!(row.get("Mean"), Some(&json!(8.5)));
        assert_eq!(row.get("Min"), Some(&json!(6.0)));
        assert_eq!(row.get("Max"), Some(&json!(11.0)));
        assert_eq!(row.get("X"), Some(&json!(2.0)));
        assert_eq!(row.get("Y"), Some(&json!(2.0)));
        assert_eq!(row.get("BX"), Some(&json!(1.0)));
        assert_eq!(row.get("BY"), Some(&json!(1.0)));
        assert_eq!(row.get("Width"), Some(&json!(2.0)));
        assert_eq!(row.get("Height"), Some(&json!(2.0)));
        assert_eq!(row.get("Perim."), Some(&json!(8.0)));
        assert_eq!(row.get("Slice"), Some(&json!(1)));
        assert!(!row.contains_key("Ch"));
        assert!(!row.contains_key("Frame"));
    }

    #[test]
    fn oval_polygon_and_freehand_measurements_use_shape_masks() {
        let tab = measurement_test_tab(4, 4, vec![1.0; 16]);
        let oval = measured_row(
            &tab,
            RoiSelection {
                tool: ToolId::Oval,
                points: vec![(0.0, 0.0), (4.0, 4.0)],
            },
        );
        assert_eq!(oval.get("Area"), Some(&json!(12.0)));

        for tool in [ToolId::Poly, ToolId::Free] {
            let mut points = vec![(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)];
            if tool == ToolId::Free {
                // Interactive freehand ROIs start with two identical drag points.
                points.insert(0, (0.0, 0.0));
            }
            let triangle = measured_row(&tab, RoiSelection { tool, points });
            assert_eq!(triangle.get("Area"), Some(&json!(10.0)), "{tool:?}");
            assert_eq!(triangle.get("Mean"), Some(&json!(1.0)), "{tool:?}");
        }
    }

    #[test]
    fn point_measurement_samples_exactly_one_pixel() {
        let tab = measurement_test_tab(3, 3, (0..9).map(|value| value as f32).collect());
        let row = measured_row(
            &tab,
            RoiSelection {
                tool: ToolId::Point,
                points: vec![(1.6, 2.2)],
            },
        );

        assert_eq!(row.get("Area"), Some(&json!(0.0)));
        assert_eq!(row.get("Mean"), Some(&json!(8.0)));
        assert_eq!(row.get("Min"), Some(&json!(8.0)));
        assert_eq!(row.get("Max"), Some(&json!(8.0)));
        assert_eq!(row.get("IntDen"), Some(&json!(8.0)));
        assert_eq!(row.get("RawIntDen"), Some(&json!(8.0)));
        assert_eq!(row.get("X"), Some(&json!(1.6_f32)));
        assert_eq!(row.get("Y"), Some(&json!(2.2_f32)));
    }

    #[test]
    fn line_and_angle_measurements_sample_polylines_and_report_length() {
        let tab = measurement_test_tab(5, 5, (0..25).map(|value| (value % 5) as f32).collect());
        let line = measured_row(
            &tab,
            RoiSelection {
                tool: ToolId::Line,
                points: vec![(0.0, 1.0), (4.0, 1.0)],
            },
        );
        assert_eq!(line.get("Length"), Some(&json!(4.0)));
        assert_eq!(line.get("Mean"), Some(&json!(2.0)));
        assert_eq!(line.get("Min"), Some(&json!(0.0)));
        assert_eq!(line.get("Max"), Some(&json!(4.0)));
        assert_eq!(line.get("Area"), Some(&json!(5.0)));
        assert_eq!(line.get("Perim."), Some(&json!(4.0)));

        let angle = measured_row(
            &tab,
            RoiSelection {
                tool: ToolId::Angle,
                points: vec![(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)],
            },
        );
        assert_eq!(angle.get("Area"), Some(&json!(0.0)));
        assert_eq!(angle.get("Mean"), Some(&json!(0.0)));
        assert!(!angle.contains_key("Length"));
        assert_eq!(angle.get("Perim."), Some(&json!(7.0)));
        let measured_angle = angle.get("Angle").and_then(Value::as_f64).unwrap();
        assert!((measured_angle - 90.0).abs() < 1.0e-9);
    }

    #[test]
    fn multi_segment_line_measurements_profile_every_segment() {
        let pixels = (0..5)
            .flat_map(|y| (0..5).map(move |x| (x + 10 * y) as f32))
            .collect();
        let tab = measurement_test_tab(5, 5, pixels);
        let line = measured_row(
            &tab,
            RoiSelection {
                tool: ToolId::Line,
                points: vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)],
            },
        );

        assert_eq!(line.get("Area"), Some(&json!(5.0)));
        assert_eq!(line.get("Min"), Some(&json!(0.0)));
        assert_eq!(line.get("Max"), Some(&json!(22.0)));
        assert_eq!(line.get("Length"), Some(&json!(4.0)));
        assert!(!line.contains_key("Angle"));
        let mean = line.get("Mean").and_then(Value::as_f64).unwrap();
        assert!((mean - 7.4).abs() < 1.0e-6);
    }

    #[test]
    fn diagonal_line_statistics_use_imagej_profile_intervals_and_interpolation() {
        let pixels = (0..5)
            .flat_map(|y| (0..5).map(move |x| (x + 10 * y) as f32))
            .collect();
        let tab = measurement_test_tab(5, 5, pixels);
        let row = measure_roi_on_tab(
            &tab,
            &RoiSelection {
                tool: ToolId::Line,
                points: vec![(0.0, 0.0), (3.0, 4.0)],
            },
            "diagonal",
            "Measure",
            1,
            RoiPosition {
                channel: 0,
                z: 0,
                t: 0,
            },
            &MeasurementSettings::default(),
        )
        .unwrap();

        // Raw length 5 => five intervals and six profile values. A linear
        // gradient remains linear under ImageJ's default bilinear sampling.
        assert_eq!(row.get("Area"), Some(&json!(6.0)));
        let mean = row.get("Mean").and_then(Value::as_f64).unwrap();
        assert!((mean - 21.5).abs() < 1.0e-6);
        assert_eq!(row.get("Min"), Some(&json!(0.0)));
        assert_eq!(row.get("Max"), Some(&json!(43.0)));
    }

    #[test]
    fn calibrated_measurements_scale_area_bounds_perimeter_length_and_angle() {
        let data = Array::from_shape_vec((5, 5), vec![1.0; 25])
            .unwrap()
            .into_dyn();
        let mut metadata = Metadata {
            dims: vec![Dim::new(AxisKind::Y, 5), Dim::new(AxisKind::X, 5)],
            pixel_type: PixelType::F32,
            ..Metadata::default()
        };
        metadata.dims[0].spacing = Some(3.0);
        metadata.dims[0].unit = Some("um".into());
        metadata.dims[1].spacing = Some(2.0);
        metadata.dims[1].unit = Some("um".into());
        metadata
            .extras
            .insert("x_origin_coordinate".into(), json!(10.0));
        metadata
            .extras
            .insert("y_origin_coordinate".into(), json!(20.0));
        let dataset = Dataset::new(data, metadata).unwrap();
        let tab = ImageTab::from_dataset(1, None, "Calibrated".into(), dataset).unwrap();
        let settings = MeasurementSettings::all_supported();
        let rectangle = measure_roi_on_tab(
            &tab,
            &RoiSelection {
                tool: ToolId::Rect,
                points: vec![(1.0, 1.0), (3.0, 3.0)],
            },
            "rectangle",
            "Measure",
            1,
            RoiPosition {
                channel: 0,
                z: 0,
                t: 0,
            },
            &settings,
        )
        .unwrap();
        assert_eq!(rectangle.get("Area"), Some(&json!(24.0)));
        assert_eq!(rectangle.get("X"), Some(&json!(14.0)));
        assert_eq!(rectangle.get("Y"), Some(&json!(26.0)));
        assert_eq!(rectangle.get("BX"), Some(&json!(12.0)));
        assert_eq!(rectangle.get("BY"), Some(&json!(23.0)));
        assert_eq!(rectangle.get("Width"), Some(&json!(4.0)));
        assert_eq!(rectangle.get("Height"), Some(&json!(6.0)));
        assert_eq!(rectangle.get("Perim."), Some(&json!(20.0)));
        assert_eq!(rectangle.get("area_unit"), Some(&json!("um^2")));
        assert_eq!(rectangle.get("length_unit"), Some(&json!("um")));

        let line = measure_roi_on_tab(
            &tab,
            &RoiSelection {
                tool: ToolId::Line,
                points: vec![(0.0, 0.0), (3.0, 4.0)],
            },
            "line",
            "Measure",
            2,
            RoiPosition {
                channel: 0,
                z: 0,
                t: 0,
            },
            &settings,
        )
        .unwrap();
        let length = line.get("Length").and_then(Value::as_f64).unwrap();
        assert!((length - 180.0_f64.sqrt()).abs() < 1.0e-9);
        let angle = line.get("Angle").and_then(Value::as_f64).unwrap();
        assert!((angle - (-12.0_f64).atan2(6.0).to_degrees()).abs() < 1.0e-9);

        let inverted_settings = MeasurementSettings {
            invert_y_coordinates: true,
            ..settings
        };
        let inverted = measure_roi_on_tab(
            &tab,
            &RoiSelection {
                tool: ToolId::Rect,
                points: vec![(0.0, 1.0), (1.0, 2.0)],
            },
            "inverted",
            "Measure",
            3,
            RoiPosition {
                channel: 0,
                z: 0,
                t: 0,
            },
            &inverted_settings,
        )
        .unwrap();
        assert_eq!(inverted.get("Y"), Some(&json!(27.5)));
        assert_eq!(inverted.get("BY"), Some(&json!(29.0)));
    }

    #[test]
    fn measurement_rejects_invalid_spatial_calibration() {
        let data = Array::from_shape_vec((1, 1), vec![1.0]).unwrap().into_dyn();
        let mut metadata = Metadata {
            dims: vec![Dim::new(AxisKind::Y, 1), Dim::new(AxisKind::X, 1)],
            pixel_type: PixelType::F32,
            ..Metadata::default()
        };
        metadata.dims[1].spacing = Some(0.0);
        let dataset = Dataset::new(data, metadata).unwrap();
        let tab = ImageTab::from_dataset(1, None, "Invalid calibration".into(), dataset).unwrap();
        let error = measure_roi_on_tab(
            &tab,
            &active_measurement_selection(&tab),
            "invalid",
            "Measure",
            1,
            RoiPosition {
                channel: 0,
                z: 0,
                t: 0,
            },
            &MeasurementSettings::default(),
        )
        .unwrap_err();
        assert!(error.contains("finite positive X/Y spacing"));
    }

    #[test]
    fn measure_stack_holds_channel_time_and_roi_while_iterating_z() {
        let mut pixels = Vec::new();
        for z in 0..3 {
            for channel in 0..2 {
                for time in 0..2 {
                    pixels.push((z * 100 + channel * 10 + time) as f32);
                }
            }
        }
        let dataset = Dataset::new(
            Array::from_shape_vec((1, 1, 3, 2, 2), pixels)
                .unwrap()
                .into_dyn(),
            Metadata {
                dims: vec![
                    Dim::new(AxisKind::Y, 1),
                    Dim::new(AxisKind::X, 1),
                    Dim::new(AxisKind::Z, 3),
                    Dim::new(AxisKind::Channel, 2),
                    Dim::new(AxisKind::Time, 2),
                ],
                pixel_type: PixelType::F32,
                ..Metadata::default()
            },
        )
        .unwrap();
        let mut tab = ImageTab::from_dataset(1, None, "Stack".into(), dataset).unwrap();
        tab.channel = 1;
        tab.t = 1;
        tab.z = 2;
        let rows = measure_stack_rows(
            &tab,
            &active_measurement_selection(&tab),
            &MeasurementSettings::default(),
            10,
        )
        .unwrap();

        assert_eq!(rows.len(), 3);
        for (z, row) in rows.iter().enumerate() {
            assert_eq!(row.get("Ch"), Some(&json!(2)));
            assert_eq!(row.get("Slice"), Some(&json!(z + 1)));
            assert_eq!(row.get("Frame"), Some(&json!(2)));
            assert_eq!(row.get("ROI"), Some(&json!(10 + z)));
            assert_eq!(row.get("Mean"), Some(&json!((z * 100 + 11) as f64)));
        }
        assert_eq!((tab.channel, tab.z, tab.t), (1, 2, 1));
    }

    #[test]
    fn managed_roi_selection_uses_stable_ids_for_single_and_toggle_clicks() {
        let order = [10, 20, 30, 40];
        let mut selected = BTreeSet::new();
        let mut anchor = None;

        assert!(apply_managed_roi_selection(
            &order,
            &mut selected,
            &mut anchor,
            20,
            ManagedRoiSelectionGesture::Single,
        ));
        assert_eq!(selected, BTreeSet::from([20]));
        assert_eq!(anchor, Some(20));

        apply_managed_roi_selection(
            &order,
            &mut selected,
            &mut anchor,
            40,
            ManagedRoiSelectionGesture::Toggle,
        );
        assert_eq!(selected, BTreeSet::from([20, 40]));
        assert_eq!(anchor, Some(40));

        apply_managed_roi_selection(
            &order,
            &mut selected,
            &mut anchor,
            20,
            ManagedRoiSelectionGesture::Toggle,
        );
        assert_eq!(selected, BTreeSet::from([40]));
        assert_eq!(anchor, Some(20));
    }

    #[test]
    fn managed_roi_shift_selection_uses_anchor_and_visual_order() {
        let order = [101, 305, 207, 999];
        let mut selected = BTreeSet::new();
        let mut anchor = None;
        apply_managed_roi_selection(
            &order,
            &mut selected,
            &mut anchor,
            305,
            ManagedRoiSelectionGesture::Single,
        );
        apply_managed_roi_selection(
            &order,
            &mut selected,
            &mut anchor,
            999,
            ManagedRoiSelectionGesture::Range { additive: false },
        );

        assert_eq!(selected, BTreeSet::from([305, 207, 999]));
        assert_eq!(anchor, Some(305));

        apply_managed_roi_selection(
            &order,
            &mut selected,
            &mut anchor,
            101,
            ManagedRoiSelectionGesture::Range { additive: true },
        );
        assert_eq!(selected, BTreeSet::from([101, 305, 207, 999]));
        assert_eq!(anchor, Some(305));
    }

    #[test]
    fn managed_roi_commands_use_all_live_entries_when_nothing_is_selected() {
        let order = [7, 2, 9];
        assert_eq!(
            effective_managed_roi_selection(&order, &BTreeSet::new()),
            BTreeSet::from(order)
        );
        assert_eq!(
            effective_managed_roi_selection(&order, &BTreeSet::from([2, 404])),
            BTreeSet::from([2])
        );
        assert_eq!(
            effective_managed_roi_selection(&order, &BTreeSet::from([404])),
            BTreeSet::from(order)
        );
    }
}
