use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{AxisKind, CoreError, PixelType, Result, default_axis_for_index};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dim {
    pub axis: AxisKind,
    pub size: usize,
    pub spacing: Option<f32>,
    pub unit: Option<String>,
}

impl Dim {
    pub fn new(axis: AxisKind, size: usize) -> Self {
        Self {
            axis,
            size,
            spacing: None,
            unit: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metadata {
    pub dims: Vec<Dim>,
    pub pixel_type: PixelType,
    pub channel_names: Vec<String>,
    pub source: Option<PathBuf>,
    pub extras: BTreeMap<String, serde_json::Value>,
}

impl Default for Metadata {
    fn default() -> Self {
        Self {
            dims: Vec::new(),
            pixel_type: PixelType::F32,
            channel_names: Vec::new(),
            source: None,
            extras: BTreeMap::new(),
        }
    }
}

impl Metadata {
    pub fn from_shape(shape: &[usize], pixel_type: PixelType) -> Self {
        let dims = shape
            .iter()
            .enumerate()
            .map(|(index, size)| Dim::new(default_axis_for_index(index), *size))
            .collect();
        Self {
            dims,
            pixel_type,
            ..Self::default()
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.dims.iter().map(|d| d.size).collect()
    }

    pub fn axis_index(&self, axis: AxisKind) -> Option<usize> {
        self.dims.iter().position(|d| d.axis == axis)
    }

    pub fn validate_shape(&self, shape: &[usize]) -> Result<()> {
        if self.dims.len() != shape.len() {
            return Err(CoreError::DimensionalityMismatch {
                data_ndim: shape.len(),
                meta_ndim: self.dims.len(),
            });
        }
        for (axis, (dim, actual)) in self.dims.iter().zip(shape).enumerate() {
            if dim.size == 0 {
                return Err(CoreError::ZeroSizedDimension { axis });
            }
            if dim.size != *actual {
                return Err(CoreError::DimensionSizeMismatch {
                    axis,
                    data_size: *actual,
                    meta_size: dim.size,
                });
            }
        }
        Ok(())
    }
}
