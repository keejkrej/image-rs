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
    pub(super) original_width: usize,
    pub(super) original_height: usize,
    pub(super) x_scale: f32,
    pub(super) y_scale: f32,
    pub(super) z_scale: f32,
    pub(super) depth: usize,
    pub(super) frames: usize,
    pub(super) constrain_aspect: bool,
    pub(super) average_when_downsizing: bool,
    pub(super) interpolation: String,
    pub(super) fill_with_background: bool,
    pub(super) fill_with_background_available: bool,
    pub(super) create_new_window: bool,
    pub(super) title: String,
    pub(super) position: String,
    pub(super) zero_fill: bool,
    pub(super) fill: f32,
}

impl Default for ResizeDialogState {
    fn default() -> Self {
        Self {
            open: false,
            width: 512,
            height: 512,
            original_width: 512,
            original_height: 512,
            x_scale: 1.0,
            y_scale: 1.0,
            z_scale: 1.0,
            depth: 1,
            frames: 1,
            constrain_aspect: true,
            average_when_downsizing: true,
            interpolation: "Bilinear".to_string(),
            fill_with_background: false,
            fill_with_background_available: false,
            create_new_window: true,
            title: "Untitled".to_string(),
            position: "Center".to_string(),
            zero_fill: false,
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
            Self::BrightnessContrast => "B&C",
            Self::WindowLevel => "W&L",
            Self::ColorBalance => "Color",
            Self::Threshold => "Threshold",
            Self::ColorThreshold => "Threshold Color",
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
    pub(super) contrast_auto_threshold: u32,
    pub(super) log_histogram: bool,
    pub(super) threshold: f32,
    pub(super) threshold_method: String,
    pub(super) threshold_mode: String,
    pub(super) threshold_stack_histogram: bool,
    pub(super) threshold_no_reset: bool,
    pub(super) threshold_raw_values: bool,
    pub(super) threshold_sixteen_bit_histogram: bool,
    pub(super) dark_background: bool,
    pub(super) color_balance_channel: String,
    pub(super) color_balance_channel_labels: Vec<String>,
    pub(super) color_balance_lut_color: bool,
    pub(super) color_threshold_space: String,
    pub(super) color_threshold_method: String,
    pub(super) color_threshold_mode: String,
    pub(super) hue_min: f32,
    pub(super) hue_max: f32,
    pub(super) saturation_min: f32,
    pub(super) saturation_max: f32,
    pub(super) brightness_min: f32,
    pub(super) brightness_max: f32,
    pub(super) hue_pass: bool,
    pub(super) saturation_pass: bool,
    pub(super) brightness_pass: bool,
    pub(super) line_width: f32,
    pub(super) spline_fit: bool,
    pub(super) left: f32,
    pub(super) right: f32,
    pub(super) top: f32,
    pub(super) bottom: f32,
    pub(super) front: f32,
    pub(super) back: f32,
    pub(super) coordinates_mode: String,
    pub(super) coordinates_x_pixel: f32,
    pub(super) coordinates_y_pixel: f32,
    pub(super) coordinates_width: f32,
    pub(super) coordinates_height: f32,
    pub(super) coordinates_z_pixel: f32,
    pub(super) coordinates_depth: f32,
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
            contrast_auto_threshold: 0,
            log_histogram: false,
            threshold: 0.5,
            threshold_method: "Default".to_string(),
            threshold_mode: "Red".to_string(),
            threshold_stack_histogram: false,
            threshold_no_reset: true,
            threshold_raw_values: false,
            threshold_sixteen_bit_histogram: false,
            dark_background: true,
            color_balance_channel: "Red".to_string(),
            color_balance_channel_labels: vec![
                "Red".to_string(),
                "Green".to_string(),
                "Blue".to_string(),
                "Cyan".to_string(),
                "Magenta".to_string(),
                "Yellow".to_string(),
                "All".to_string(),
            ],
            color_balance_lut_color: false,
            color_threshold_space: "HSB".to_string(),
            color_threshold_method: "Default".to_string(),
            color_threshold_mode: "Red".to_string(),
            hue_min: 0.0,
            hue_max: 255.0,
            saturation_min: 0.0,
            saturation_max: 255.0,
            brightness_min: 0.0,
            brightness_max: 255.0,
            hue_pass: true,
            saturation_pass: true,
            brightness_pass: true,
            line_width: 1.0,
            spline_fit: false,
            left: 0.0,
            right: 512.0,
            top: 0.0,
            bottom: 512.0,
            front: 0.0,
            back: 1.0,
            coordinates_mode: "image".to_string(),
            coordinates_x_pixel: 0.0,
            coordinates_y_pixel: 0.0,
            coordinates_width: 512.0,
            coordinates_height: 512.0,
            coordinates_z_pixel: 0.0,
            coordinates_depth: 1.0,
            x_unit: "pixel".to_string(),
            y_unit: "pixel".to_string(),
            z_unit: "pixel".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct ThresholdApplyDialogState {
    pub(super) open: bool,
    pub(super) window_label: String,
    pub(super) min: f32,
    pub(super) max: f32,
    pub(super) dark_background: bool,
}

impl Default for ThresholdApplyDialogState {
    fn default() -> Self {
        Self {
            open: false,
            window_label: String::new(),
            min: 0.0,
            max: 1.0,
            dark_background: true,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct ThresholdSetDialogState {
    pub(super) open: bool,
    pub(super) window_label: String,
    pub(super) min: f32,
    pub(super) max: f32,
    pub(super) mode: String,
    pub(super) dark_background: bool,
}

impl Default for ThresholdSetDialogState {
    fn default() -> Self {
        Self {
            open: false,
            window_label: String::new(),
            min: 0.0,
            max: 255.0,
            mode: "Red".to_string(),
            dark_background: true,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct ApplyLutDialogState {
    pub(super) open: bool,
    pub(super) stack_prompt: bool,
    pub(super) window_label: String,
    pub(super) command_id: String,
    pub(super) params: serde_json::Value,
    pub(super) slice_params: serde_json::Value,
    pub(super) stack_slices: usize,
}

impl Default for ApplyLutDialogState {
    fn default() -> Self {
        Self {
            open: false,
            stack_prompt: false,
            window_label: String::new(),
            command_id: String::new(),
            params: serde_json::Value::Null,
            slice_params: serde_json::Value::Null,
            stack_slices: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct SetDisplayRangeDialogState {
    pub(super) open: bool,
    pub(super) window_label: String,
    pub(super) command_id: String,
    pub(super) low_key: String,
    pub(super) high_key: String,
    pub(super) minimum: f32,
    pub(super) maximum: f32,
    pub(super) unsigned_16bit_range: String,
    pub(super) propagate: bool,
    pub(super) all_channels: bool,
    pub(super) show_all_channels: bool,
    pub(super) channel: String,
    pub(super) channel_count: usize,
}

impl Default for SetDisplayRangeDialogState {
    fn default() -> Self {
        Self {
            open: false,
            window_label: String::new(),
            command_id: String::new(),
            low_key: "min".to_string(),
            high_key: "max".to_string(),
            minimum: 0.0,
            maximum: 1.0,
            unsigned_16bit_range: "Automatic".to_string(),
            propagate: false,
            all_channels: false,
            show_all_channels: false,
            channel: String::new(),
            channel_count: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct SetWindowLevelDialogState {
    pub(super) open: bool,
    pub(super) window_label: String,
    pub(super) level: f32,
    pub(super) window: f32,
    pub(super) propagate: bool,
}

impl Default for SetWindowLevelDialogState {
    fn default() -> Self {
        Self {
            open: false,
            window_label: String::new(),
            level: 0.5,
            window: 1.0,
            propagate: false,
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
