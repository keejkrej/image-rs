use std::collections::BTreeMap;
use std::path::PathBuf;

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, CoreError>;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error(
        "metadata dimensionality mismatch: data has {data_ndim} dimensions but metadata has {meta_ndim}"
    )]
    DimensionalityMismatch { data_ndim: usize, meta_ndim: usize },

    #[error(
        "dimension size mismatch at axis {axis}: data size {data_size} != metadata size {meta_size}"
    )]
    DimensionSizeMismatch {
        axis: usize,
        data_size: usize,
        meta_size: usize,
    },

    #[error("invalid dimension size 0 at axis {axis}")]
    ZeroSizedDimension { axis: usize },

    #[error("invalid metadata: {0}")]
    InvalidMetadata(String),
}

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

#[derive(Debug, Clone)]
pub struct Dataset<T> {
    pub data: ArrayD<T>,
    pub metadata: Metadata,
}

impl<T> Dataset<T> {
    pub fn new(data: ArrayD<T>, metadata: Metadata) -> Result<Self> {
        metadata.validate_shape(data.shape())?;
        Ok(Self { data, metadata })
    }

    pub fn from_data_with_default_metadata(data: ArrayD<T>, pixel_type: PixelType) -> Self {
        let metadata = Metadata::from_shape(data.shape(), pixel_type);
        Self { data, metadata }
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    pub fn axis_index(&self, axis: AxisKind) -> Option<usize> {
        self.metadata.axis_index(axis)
    }

    pub fn validate(&self) -> Result<()> {
        self.metadata.validate_shape(self.data.shape())
    }
}

impl Dataset<f32> {
    pub fn min_max(&self) -> Option<(f32, f32)> {
        let mut iter = self.data.iter().copied();
        let first = iter.next()?;
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
        Some((min, max))
    }
}

pub type DatasetF32 = Dataset<f32>;

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

#[cfg(test)]
mod tests {
    use ndarray::Array;

    use super::{AxisKind, Dataset, Dim, Metadata, PixelType};

    #[test]
    fn metadata_roundtrip_json() {
        let mut metadata = Metadata::from_shape(&[4, 5, 3], PixelType::U8);
        metadata.channel_names = vec!["R".into(), "G".into(), "B".into()];
        metadata
            .extras
            .insert("dataset".into(), serde_json::json!("synthetic"));
        let serialized = serde_json::to_string_pretty(&metadata).expect("serialize metadata");
        let restored: Metadata = serde_json::from_str(&serialized).expect("deserialize metadata");
        assert_eq!(restored, metadata);
    }

    #[test]
    fn dataset_rejects_invalid_metadata_shape() {
        let data = Array::from_shape_vec((2, 2), vec![0.0_f32, 1.0, 2.0, 3.0])
            .expect("shape")
            .into_dyn();
        let metadata = Metadata {
            dims: vec![Dim::new(AxisKind::X, 2)],
            pixel_type: PixelType::F32,
            ..Metadata::default()
        };
        assert!(Dataset::new(data, metadata).is_err());
    }

    #[test]
    fn dataset_validates_dimension_sizes() {
        let data = Array::from_shape_vec((2, 2), vec![0.0_f32, 1.0, 2.0, 3.0])
            .expect("shape")
            .into_dyn();
        let metadata = Metadata {
            dims: vec![Dim::new(AxisKind::Y, 2), Dim::new(AxisKind::X, 2)],
            pixel_type: PixelType::F32,
            ..Metadata::default()
        };
        let dataset = Dataset::new(data, metadata).expect("dataset");
        assert!(dataset.validate().is_ok());
        assert_eq!(dataset.axis_index(AxisKind::X), Some(1));
        assert_eq!(dataset.axis_index(AxisKind::Y), Some(0));
    }
}

