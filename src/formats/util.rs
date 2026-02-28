use std::path::Path;

use crate::model::{Dim, Metadata, PixelType};

use super::{IoError, Result};

pub(crate) fn extension(path: &Path) -> Result<String> {
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .ok_or_else(|| IoError::UnsupportedFormat(path.to_string_lossy().to_string()))?;
    Ok(ext)
}

pub(crate) fn metadata_for_dims(path: &Path, dims: Vec<Dim>, pixel_type: PixelType) -> Metadata {
    Metadata {
        dims,
        pixel_type,
        source: Some(path.to_path_buf()),
        ..Metadata::default()
    }
}

pub(crate) fn to_u8_samples(values: &[f32]) -> Vec<u8> {
    let (min, max) = min_max(values);
    let use_unit_range = min >= 0.0 && max <= 1.0;
    values
        .iter()
        .map(|value| {
            let normalized = if use_unit_range {
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

pub(crate) fn to_u16_samples(values: &[f32]) -> Vec<u16> {
    let (min, max) = min_max(values);
    let use_unit_range = min >= 0.0 && max <= 1.0;
    values
        .iter()
        .map(|value| {
            let normalized = if use_unit_range {
                *value
            } else if (max - min).abs() < f32::EPSILON {
                0.0
            } else {
                (*value - min) / (max - min)
            };
            (normalized.clamp(0.0, 1.0) * 65_535.0).round() as u16
        })
        .collect()
}

pub(crate) fn scale_to_u8(values: &[f32]) -> Vec<u8> {
    to_u8_samples(values)
}

pub(crate) fn min_max(values: &[f32]) -> (f32, f32) {
    let mut iter = values.iter().copied();
    let first = iter.next().unwrap_or(0.0);
    let mut min = first;
    let mut max = first;
    for value in iter {
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
    }
    (min, max)
}
