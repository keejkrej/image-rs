use ndarray::ArrayD;

use super::{AxisKind, Metadata, PixelType, Result};

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
