#![allow(dead_code)]

use eframe::egui;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RoiPosition {
    pub channel: usize,
    pub z: usize,
    pub t: usize,
}

#[derive(Debug, Clone)]
pub enum RoiKind {
    Rect {
        start: egui::Pos2,
        end: egui::Pos2,
        rounded: bool,
        rotated: bool,
    },
    Oval {
        start: egui::Pos2,
        end: egui::Pos2,
        ellipse: bool,
        brush: bool,
    },
    Polygon {
        points: Vec<egui::Pos2>,
        closed: bool,
    },
    Freehand {
        points: Vec<egui::Pos2>,
    },
    Line {
        start: egui::Pos2,
        end: egui::Pos2,
        arrow: bool,
    },
    Angle {
        a: egui::Pos2,
        b: egui::Pos2,
        c: egui::Pos2,
    },
    Point {
        points: Vec<egui::Pos2>,
        multi: bool,
    },
    Text {
        at: egui::Pos2,
        text: String,
    },
    WandTrace {
        points: Vec<egui::Pos2>,
    },
}

#[derive(Debug, Clone)]
pub struct RoiModel {
    pub id: u64,
    pub kind: RoiKind,
    pub position: RoiPosition,
}

#[derive(Debug, Clone, Default)]
pub struct RoiStore {
    pub active_roi: Option<RoiModel>,
    pub overlay_rois: Vec<RoiModel>,
    pub selected_roi_id: Option<u64>,
    next_id: u64,
}

impl RoiStore {
    fn next_id(&mut self) -> u64 {
        self.next_id = self.next_id.saturating_add(1);
        self.next_id
    }

    pub fn begin_active(&mut self, kind: RoiKind, position: RoiPosition) {
        self.active_roi = Some(RoiModel {
            id: self.next_id(),
            kind,
            position,
        });
    }

    pub fn update_active(&mut self, kind: RoiKind) {
        if let Some(active) = &mut self.active_roi {
            active.kind = kind;
        }
    }

    pub fn commit_active(&mut self, keep_existing: bool) {
        if let Some(active) = self.active_roi.take() {
            if !keep_existing {
                self.overlay_rois.clear();
            }
            self.selected_roi_id = Some(active.id);
            self.overlay_rois.push(active);
        }
    }

    pub fn abort_active(&mut self) {
        self.active_roi = None;
    }

    pub fn clear_all(&mut self) {
        self.active_roi = None;
        self.overlay_rois.clear();
        self.selected_roi_id = None;
    }

    pub fn select_next(&mut self) -> Option<u64> {
        if self.overlay_rois.is_empty() {
            self.selected_roi_id = None;
            return None;
        }

        let current_idx = self
            .selected_roi_id
            .and_then(|id| self.overlay_rois.iter().position(|roi| roi.id == id))
            .unwrap_or(usize::MAX);

        let next_idx = if current_idx == usize::MAX {
            0
        } else {
            (current_idx + 1) % self.overlay_rois.len()
        };

        let id = self.overlay_rois[next_idx].id;
        self.selected_roi_id = Some(id);
        Some(id)
    }

    pub fn visible_rois<'a>(&'a self, position: RoiPosition) -> impl Iterator<Item = &'a RoiModel> {
        self.overlay_rois
            .iter()
            .filter(move |roi| roi.position == position)
            .chain(
                self.active_roi
                    .iter()
                    .filter(move |roi| roi.position == position),
            )
    }

    pub fn active_status_text(&self) -> Option<String> {
        let roi = self.active_roi.as_ref().or_else(|| {
            self.selected_roi_id
                .and_then(|id| self.overlay_rois.iter().find(|roi| roi.id == id))
        })?;

        let text = match &roi.kind {
            RoiKind::Rect { start, end, .. } => {
                let w = (end.x - start.x).abs();
                let h = (end.y - start.y).abs();
                format!("Rect {:.0}x{:.0}", w, h)
            }
            RoiKind::Oval { start, end, .. } => {
                let w = (end.x - start.x).abs();
                let h = (end.y - start.y).abs();
                format!("Oval {:.0}x{:.0}", w, h)
            }
            RoiKind::Polygon { points, .. } => format!("Polygon points: {}", points.len()),
            RoiKind::Freehand { points } => format!("Freehand points: {}", points.len()),
            RoiKind::Line { start, end, .. } => {
                let length = (*end - *start).length();
                format!("Line length {:.2}", length)
            }
            RoiKind::Angle { a, b, c } => {
                let ba = *a - *b;
                let bc = *c - *b;
                let denom = ba.length() * bc.length();
                let angle = if denom <= f32::EPSILON {
                    0.0
                } else {
                    (ba.dot(bc) / denom).clamp(-1.0, 1.0).acos().to_degrees()
                };
                format!("Angle {:.2} deg", angle)
            }
            RoiKind::Point { points, .. } => format!("Point count {}", points.len()),
            RoiKind::Text { text, .. } => format!("Text \"{}\"", text),
            RoiKind::WandTrace { points } => format!("Wand vertices {}", points.len()),
        };

        Some(text)
    }
}
