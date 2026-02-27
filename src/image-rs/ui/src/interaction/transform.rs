#![allow(dead_code)]

use eframe::egui;

pub const MIN_MAGNIFICATION: f32 = 1.0 / 72.0;
pub const MAX_MAGNIFICATION: f32 = 32.0;

pub const ZOOM_LEVELS: [f32; 23] = [
    1.0 / 72.0,
    1.0 / 48.0,
    1.0 / 32.0,
    1.0 / 24.0,
    1.0 / 16.0,
    1.0 / 12.0,
    1.0 / 8.0,
    1.0 / 6.0,
    1.0 / 4.0,
    1.0 / 3.0,
    1.0 / 2.0,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    12.0,
    16.0,
    24.0,
    32.0,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoomStep {
    In,
    Out,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl SourceRect {
    pub fn full(image_width: usize, image_height: usize) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: image_width.max(1) as f32,
            height: image_height.max(1) as f32,
        }
    }

    pub fn clamp_to_image(&mut self, image_width: usize, image_height: usize) {
        let image_w = image_width.max(1) as f32;
        let image_h = image_height.max(1) as f32;

        self.width = self.width.clamp(1.0, image_w);
        self.height = self.height.clamp(1.0, image_h);
        self.x = self.x.clamp(0.0, (image_w - self.width).max(0.0));
        self.y = self.y.clamp(0.0, (image_h - self.height).max(0.0));
    }

    pub fn uv_rect(&self, image_width: usize, image_height: usize) -> egui::Rect {
        let image_w = image_width.max(1) as f32;
        let image_h = image_height.max(1) as f32;
        egui::Rect::from_min_max(
            egui::pos2(self.x / image_w, self.y / image_h),
            egui::pos2(
                (self.x + self.width) / image_w,
                (self.y + self.height) / image_h,
            ),
        )
    }
}

#[derive(Debug, Clone)]
pub struct ViewerTransformState {
    pub magnification: f32,
    pub src_rect: SourceRect,
    pub scale_to_fit: bool,
    pub last_zoom_screen: Option<egui::Pos2>,
    pub zoom_target_offscreen: Option<egui::Pos2>,
}

impl Default for ViewerTransformState {
    fn default() -> Self {
        Self::new(1, 1)
    }
}

impl ViewerTransformState {
    pub fn new(image_width: usize, image_height: usize) -> Self {
        Self {
            magnification: 1.0,
            src_rect: SourceRect::full(image_width, image_height),
            scale_to_fit: false,
            last_zoom_screen: None,
            zoom_target_offscreen: None,
        }
    }

    pub fn fit_to_canvas(
        &mut self,
        canvas_rect: egui::Rect,
        image_width: usize,
        image_height: usize,
    ) {
        let image_w = image_width.max(1) as f32;
        let image_h = image_height.max(1) as f32;
        let fit_mag = (canvas_rect.width() / image_w)
            .min(canvas_rect.height() / image_h)
            .clamp(MIN_MAGNIFICATION, MAX_MAGNIFICATION);
        self.magnification = fit_mag;
        self.src_rect = SourceRect::full(image_width, image_height);
        self.scale_to_fit = true;
    }

    pub fn screen_to_image(
        &self,
        canvas_rect: egui::Rect,
        screen_pos: egui::Pos2,
    ) -> Option<egui::Pos2> {
        if !canvas_rect.contains(screen_pos) {
            return None;
        }

        let u = if canvas_rect.width() > 0.0 {
            (screen_pos.x - canvas_rect.min.x) / canvas_rect.width()
        } else {
            0.0
        }
        .clamp(0.0, 1.0);
        let v = if canvas_rect.height() > 0.0 {
            (screen_pos.y - canvas_rect.min.y) / canvas_rect.height()
        } else {
            0.0
        }
        .clamp(0.0, 1.0);

        Some(egui::pos2(
            self.src_rect.x + u * self.src_rect.width,
            self.src_rect.y + v * self.src_rect.height,
        ))
    }

    pub fn image_to_screen(&self, canvas_rect: egui::Rect, image_pos: egui::Pos2) -> egui::Pos2 {
        let u = if self.src_rect.width > 0.0 {
            (image_pos.x - self.src_rect.x) / self.src_rect.width
        } else {
            0.0
        };
        let v = if self.src_rect.height > 0.0 {
            (image_pos.y - self.src_rect.y) / self.src_rect.height
        } else {
            0.0
        };

        egui::pos2(
            canvas_rect.min.x + u * canvas_rect.width(),
            canvas_rect.min.y + v * canvas_rect.height(),
        )
    }

    pub fn zoom_in_at(
        &mut self,
        canvas_rect: egui::Rect,
        pointer_screen: egui::Pos2,
        image_width: usize,
        image_height: usize,
    ) {
        let next = zoom_level_up(self.magnification);
        self.set_magnification_at(canvas_rect, pointer_screen, next, image_width, image_height);
    }

    pub fn zoom_out_at(
        &mut self,
        canvas_rect: egui::Rect,
        pointer_screen: egui::Pos2,
        image_width: usize,
        image_height: usize,
    ) {
        let next = zoom_level_down(self.magnification);
        self.set_magnification_at(canvas_rect, pointer_screen, next, image_width, image_height);
    }

    pub fn zoom_step_at(
        &mut self,
        step: ZoomStep,
        canvas_rect: egui::Rect,
        pointer_screen: egui::Pos2,
        image_width: usize,
        image_height: usize,
    ) {
        match step {
            ZoomStep::In => self.zoom_in_at(canvas_rect, pointer_screen, image_width, image_height),
            ZoomStep::Out => {
                self.zoom_out_at(canvas_rect, pointer_screen, image_width, image_height)
            }
        }
    }

    pub fn zoom_original(
        &mut self,
        canvas_rect: egui::Rect,
        image_width: usize,
        image_height: usize,
    ) {
        self.fit_to_canvas(canvas_rect, image_width, image_height);
    }

    pub fn zoom_view_100(
        &mut self,
        canvas_rect: egui::Rect,
        image_width: usize,
        image_height: usize,
    ) {
        let center_screen = canvas_rect.center();
        self.set_magnification_at(canvas_rect, center_screen, 1.0, image_width, image_height);
    }

    pub fn set_magnification_at(
        &mut self,
        canvas_rect: egui::Rect,
        pointer_screen: egui::Pos2,
        next_magnification: f32,
        image_width: usize,
        image_height: usize,
    ) {
        let image_w = image_width.max(1) as f32;
        let image_h = image_height.max(1) as f32;
        let next_mag = next_magnification.clamp(MIN_MAGNIFICATION, MAX_MAGNIFICATION);

        let pointer_image =
            self.screen_to_image(canvas_rect, pointer_screen)
                .unwrap_or(egui::pos2(
                    self.src_rect.x + self.src_rect.width * 0.5,
                    self.src_rect.y + self.src_rect.height * 0.5,
                ));

        let nx = if canvas_rect.width() > 0.0 {
            (pointer_screen.x - canvas_rect.min.x) / canvas_rect.width()
        } else {
            0.5
        }
        .clamp(0.0, 1.0);
        let ny = if canvas_rect.height() > 0.0 {
            (pointer_screen.y - canvas_rect.min.y) / canvas_rect.height()
        } else {
            0.5
        }
        .clamp(0.0, 1.0);

        let mut new_width = (canvas_rect.width() / next_mag).max(1.0);
        let mut new_height = (canvas_rect.height() / next_mag).max(1.0);
        new_width = new_width.min(image_w);
        new_height = new_height.min(image_h);

        self.src_rect = SourceRect {
            x: pointer_image.x - nx * new_width,
            y: pointer_image.y - ny * new_height,
            width: new_width,
            height: new_height,
        };
        self.src_rect.clamp_to_image(image_width, image_height);
        self.magnification = next_mag;
        self.last_zoom_screen = Some(pointer_screen);
        self.zoom_target_offscreen = Some(pointer_image);
        self.scale_to_fit = false;
    }

    pub fn scroll_by_screen_delta(
        &mut self,
        delta: egui::Vec2,
        image_width: usize,
        image_height: usize,
    ) {
        if self.magnification <= 0.0 {
            return;
        }
        self.src_rect.x -= delta.x / self.magnification;
        self.src_rect.y -= delta.y / self.magnification;
        self.src_rect.clamp_to_image(image_width, image_height);
    }

    pub fn wheel_pan(
        &mut self,
        wheel_delta: f32,
        horizontal: bool,
        image_width: usize,
        image_height: usize,
    ) {
        let step = if horizontal {
            (image_width.max(1) as f32 / 200.0).max(1.0)
        } else {
            (image_height.max(1) as f32 / 200.0).max(1.0)
        };

        if horizontal {
            self.src_rect.x += wheel_delta * step;
        } else {
            self.src_rect.y += wheel_delta * step;
        }
        self.src_rect.clamp_to_image(image_width, image_height);
    }
}

pub fn zoom_level_down(current: f32) -> f32 {
    let mut next = ZOOM_LEVELS[0];
    for level in ZOOM_LEVELS {
        if level < current {
            next = level;
        } else {
            break;
        }
    }
    next
}

pub fn zoom_level_up(current: f32) -> f32 {
    let mut next = ZOOM_LEVELS[ZOOM_LEVELS.len() - 1];
    for level in ZOOM_LEVELS.iter().rev().copied() {
        if level > current {
            next = level;
        } else {
            break;
        }
    }
    next
}

#[cfg(test)]
mod tests {
    use super::{ViewerTransformState, zoom_level_down, zoom_level_up};

    #[test]
    fn zoom_levels_move_to_expected_neighbors() {
        assert!((zoom_level_up(1.0) - 1.5).abs() < f32::EPSILON);
        assert!((zoom_level_down(1.0) - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn set_magnification_preserves_pointer_anchor() {
        let mut state = ViewerTransformState::new(512, 512);
        let canvas = eframe::egui::Rect::from_min_size(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::vec2(256.0, 256.0),
        );
        state.set_magnification_at(canvas, eframe::egui::pos2(128.0, 128.0), 2.0, 512, 512);
        assert!((state.magnification - 2.0).abs() < f32::EPSILON);
        assert!(state.src_rect.width < 512.0);
        assert!(state.src_rect.height < 512.0);
    }
}
