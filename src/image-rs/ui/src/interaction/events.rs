#![allow(dead_code)]

use eframe::egui;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanvasButton {
    Primary,
    Secondary,
    Middle,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CanvasModifiers {
    pub shift: bool,
    pub alt: bool,
    pub ctrl: bool,
    pub command: bool,
    pub space: bool,
}

impl CanvasModifiers {
    pub fn from_egui(input: &egui::InputState) -> Self {
        Self {
            shift: input.modifiers.shift,
            alt: input.modifiers.alt,
            ctrl: input.modifiers.ctrl,
            command: input.modifiers.command,
            space: input.key_down(egui::Key::Space),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CanvasEventKind {
    Move,
    Press(CanvasButton),
    Drag(CanvasButton),
    Release(CanvasButton),
    Scroll { delta: egui::Vec2 },
    DoubleClick(CanvasButton),
}

#[derive(Debug, Clone, Copy)]
pub struct CanvasEvent {
    pub kind: CanvasEventKind,
    pub pointer_screen: Option<egui::Pos2>,
    pub pointer_image: Option<egui::Pos2>,
    pub modifiers: CanvasModifiers,
}
