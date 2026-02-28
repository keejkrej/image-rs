use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AxisKind {
    X,
    Y,
    Z,
    Channel,
    Time,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PixelType {
    U8,
    U16,
    #[default]
    F32,
}

pub fn default_axis_for_index(index: usize) -> AxisKind {
    match index {
        0 => AxisKind::Y,
        1 => AxisKind::X,
        2 => AxisKind::Z,
        3 => AxisKind::Channel,
        4 => AxisKind::Time,
        _ => AxisKind::Unknown,
    }
}
