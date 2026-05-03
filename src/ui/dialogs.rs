use std::path::PathBuf;

use eframe::egui;

use crate::model::PixelType;

#[derive(Debug, Clone)]
pub(super) struct NewImageDialogState {
    pub(super) open: bool,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) slices: usize,
    pub(super) channels: usize,
    pub(super) frames: usize,
    pub(super) pixel_type: PixelType,
    pub(super) fill: f32,
}

impl Default for NewImageDialogState {
    fn default() -> Self {
        Self {
            open: false,
            width: 512,
            height: 512,
            slices: 1,
            channels: 1,
            frames: 1,
            pixel_type: PixelType::F32,
            fill: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct ResizeDialogState {
    pub(super) open: bool,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) fill: f32,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AdjustDialogKind {
    BrightnessContrast,
    WindowLevel,
    ColorBalance,
    Threshold,
    ColorThreshold,
    LineWidth,
    Coordinates,
}

impl AdjustDialogKind {
    pub(super) fn title(self) -> &'static str {
        match self {
            Self::BrightnessContrast => "Brightness/Contrast",
            Self::WindowLevel => "Window/Level",
            Self::ColorBalance => "Color Balance",
            Self::Threshold => "Threshold",
            Self::ColorThreshold => "Color Threshold",
            Self::LineWidth => "Line Width",
            Self::Coordinates => "Coordinates",
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct AdjustHistogram {
    pub(super) counts: Vec<u64>,
    pub(super) min: f32,
    pub(super) max: f32,
    pub(super) pixel_count: usize,
}

#[derive(Debug, Clone)]
pub(super) struct AdjustDialogState {
    pub(super) open: bool,
    pub(super) window_label: String,
    pub(super) kind: AdjustDialogKind,
    pub(super) histogram: Option<AdjustHistogram>,
    pub(super) default_min: f32,
    pub(super) default_max: f32,
    pub(super) min: f32,
    pub(super) max: f32,
    pub(super) brightness: f32,
    pub(super) contrast: f32,
    pub(super) threshold: f32,
    pub(super) dark_background: bool,
    pub(super) red: f32,
    pub(super) green: f32,
    pub(super) blue: f32,
    pub(super) line_width: f32,
    pub(super) left: f32,
    pub(super) right: f32,
    pub(super) top: f32,
    pub(super) bottom: f32,
    pub(super) front: f32,
    pub(super) back: f32,
    pub(super) x_unit: String,
    pub(super) y_unit: String,
    pub(super) z_unit: String,
}

impl Default for AdjustDialogState {
    fn default() -> Self {
        Self {
            open: false,
            window_label: String::new(),
            kind: AdjustDialogKind::BrightnessContrast,
            histogram: None,
            default_min: 0.0,
            default_max: 1.0,
            min: 0.0,
            max: 1.0,
            brightness: 0.5,
            contrast: 0.5,
            threshold: 0.5,
            dark_background: true,
            red: 1.0,
            green: 1.0,
            blue: 1.0,
            line_width: 1.0,
            left: 0.0,
            right: 512.0,
            top: 0.0,
            bottom: 512.0,
            front: 0.0,
            back: 1.0,
            x_unit: "pixel".to_string(),
            y_unit: "pixel".to_string(),
            z_unit: "pixel".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct StackPositionDialogState {
    pub(super) open: bool,
    pub(super) window_label: String,
    pub(super) channel: usize,
    pub(super) slice: usize,
    pub(super) frame: usize,
}

impl Default for StackPositionDialogState {
    fn default() -> Self {
        Self {
            open: false,
            window_label: String::new(),
            channel: 1,
            slice: 1,
            frame: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct StackLabelDialogState {
    pub(super) open: bool,
    pub(super) window_label: String,
    pub(super) slice: usize,
    pub(super) label: String,
}

impl Default for StackLabelDialogState {
    fn default() -> Self {
        Self {
            open: false,
            window_label: String::new(),
            slice: 1,
            label: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct ZoomSetDialogState {
    pub(super) open: bool,
    pub(super) window_label: String,
    pub(super) zoom_percent: f32,
    pub(super) x_center: f32,
    pub(super) y_center: f32,
}

impl Default for ZoomSetDialogState {
    fn default() -> Self {
        Self {
            open: false,
            window_label: String::new(),
            zoom_percent: 100.0,
            x_center: 0.0,
            y_center: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ColorDialogMode {
    Colors,
    Foreground,
    Background,
    Picker,
}

#[derive(Debug, Clone)]
pub(super) struct ColorDialogState {
    pub(super) open: bool,
    pub(super) mode: ColorDialogMode,
    pub(super) foreground: egui::Color32,
    pub(super) background: egui::Color32,
}

impl Default for ColorDialogState {
    fn default() -> Self {
        Self {
            open: false,
            mode: ColorDialogMode::Colors,
            foreground: egui::Color32::WHITE,
            background: egui::Color32::BLACK,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct RawImportDialogState {
    pub(super) open: bool,
    pub(super) path: Option<PathBuf>,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) slices: usize,
    pub(super) channels: usize,
    pub(super) pixel_type: PixelType,
    pub(super) little_endian: bool,
    pub(super) byte_offset: usize,
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
pub(super) struct UrlImportDialogState {
    pub(super) open: bool,
    pub(super) url: String,
}

#[derive(Debug, Clone, Default)]
pub(super) struct CommandFinderState {
    pub(super) query: String,
}

#[derive(Debug, Clone, Default)]
pub(super) struct RoiManagerState {
    pub(super) rename_buffer: String,
}
