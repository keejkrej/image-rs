#![allow(dead_code)]

use eframe::egui;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolId {
    Rect,
    Oval,
    Poly,
    Free,
    Line,
    Angle,
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
    pub const fn command_id(self) -> &'static str {
        match self {
            Self::Rect => "launcher.tool.rect",
            Self::Oval => "launcher.tool.oval",
            Self::Poly => "launcher.tool.poly",
            Self::Free => "launcher.tool.free",
            Self::Line => "launcher.tool.line",
            Self::Angle => "launcher.tool.angle",
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

    pub const fn label(self) -> &'static str {
        match self {
            Self::Rect => "Rect",
            Self::Oval => "Oval",
            Self::Poly => "Poly",
            Self::Free => "Free",
            Self::Line => "Line",
            Self::Angle => "Angle",
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

    pub const fn has_behavior(self) -> bool {
        !matches!(
            self,
            Self::Custom1 | Self::Custom2 | Self::Custom3 | Self::More
        )
    }

    pub const fn is_shape_tool(self) -> bool {
        matches!(
            self,
            Self::Rect | Self::Oval | Self::Poly | Self::Free | Self::Line | Self::Angle
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RectMode {
    Rectangle,
    Rounded,
    Rotated,
}

impl Default for RectMode {
    fn default() -> Self {
        Self::Rectangle
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OvalMode {
    Oval,
    Ellipse,
    Brush,
}

impl Default for OvalMode {
    fn default() -> Self {
        Self::Oval
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineMode {
    Straight,
    Segmented,
    Freehand,
    Arrow,
}

impl Default for LineMode {
    fn default() -> Self {
        Self::Straight
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PointMode {
    Point,
    MultiPoint,
}

impl Default for PointMode {
    fn default() -> Self {
        Self::Point
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WandMode {
    Legacy,
    Connected,
    FourConnected,
}

impl Default for WandMode {
    fn default() -> Self {
        Self::Legacy
    }
}

#[derive(Debug, Clone)]
pub struct ToolOptionsState {
    pub rect_mode: RectMode,
    pub oval_mode: OvalMode,
    pub line_mode: LineMode,
    pub point_mode: PointMode,
    pub wand_tolerance: f32,
    pub wand_mode: WandMode,
    pub brush_size_px: u32,
    pub arc_size_px: u32,
    pub foreground_color: egui::Color32,
    pub background_color: egui::Color32,
}

impl Default for ToolOptionsState {
    fn default() -> Self {
        Self {
            rect_mode: RectMode::default(),
            oval_mode: OvalMode::default(),
            line_mode: LineMode::default(),
            point_mode: PointMode::default(),
            wand_tolerance: 0.0,
            wand_mode: WandMode::default(),
            brush_size_px: 15,
            arc_size_px: 20,
            foreground_color: egui::Color32::WHITE,
            background_color: egui::Color32::BLACK,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ToolState {
    pub selected: ToolId,
}

impl Default for ToolState {
    fn default() -> Self {
        Self {
            selected: ToolId::Rect,
        }
    }
}

pub fn tool_from_command_id(command_id: &str) -> Option<ToolId> {
    match command_id {
        "launcher.tool.rect" => Some(ToolId::Rect),
        "launcher.tool.oval" => Some(ToolId::Oval),
        "launcher.tool.poly" => Some(ToolId::Poly),
        "launcher.tool.free" => Some(ToolId::Free),
        "launcher.tool.line" => Some(ToolId::Line),
        "launcher.tool.angle" => Some(ToolId::Angle),
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

pub fn tool_shortcut_command(text: &str) -> Option<&'static str> {
    tool_shortcut_tool(text).map(ToolId::command_id)
}

pub fn tool_shortcut_tool(text: &str) -> Option<ToolId> {
    match text {
        "r" | "R" => Some(ToolId::Rect),
        "o" | "O" => Some(ToolId::Oval),
        "g" | "G" => Some(ToolId::Poly),
        "f" | "F" => Some(ToolId::Free),
        "l" | "L" => Some(ToolId::Line),
        "a" | "A" => Some(ToolId::Angle),
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
