use std::path::Path;

use crate::formats::{IoError, read_dataset, read_dataset_bytes, write_dataset};
use crate::model::{AxisKind, Dataset, DatasetF32, Dim, Metadata, PixelType};
use ndarray::{ArrayD, IxDyn};

use super::Result;

#[derive(Debug, Default, Clone, Copy)]
pub struct IoService;

impl IoService {
    pub fn read(&self, path: impl AsRef<Path>) -> Result<DatasetF32> {
        Ok(read_dataset(path)?)
    }

    pub fn read_bytes(&self, bytes: &[u8], format_hint: &str) -> Result<DatasetF32> {
        Ok(read_dataset_bytes(bytes, format_hint)?)
    }

    pub fn write(&self, path: impl AsRef<Path>, dataset: &DatasetF32) -> Result<()> {
        write_dataset(path, dataset)?;
        Ok(())
    }

    pub fn read_raw(
        &self,
        bytes: &[u8],
        width: usize,
        height: usize,
        slices: usize,
        channels: usize,
        pixel_type: PixelType,
        little_endian: bool,
        byte_offset: usize,
    ) -> Result<DatasetF32> {
        let bytes = bytes.get(byte_offset..).ok_or_else(|| {
            IoError::Io(std::io::Error::other("raw offset exceeds buffer length"))
        })?;
        let voxel_count = width
            .saturating_mul(height)
            .saturating_mul(slices.max(1))
            .saturating_mul(channels.max(1));
        let values = match pixel_type {
            PixelType::U8 => bytes
                .iter()
                .copied()
                .take(voxel_count)
                .map(|value| f32::from(value) / 255.0)
                .collect::<Vec<_>>(),
            PixelType::U16 => bytes
                .chunks_exact(2)
                .take(voxel_count)
                .map(|chunk| {
                    let raw = if little_endian {
                        u16::from_le_bytes([chunk[0], chunk[1]])
                    } else {
                        u16::from_be_bytes([chunk[0], chunk[1]])
                    };
                    f32::from(raw) / 65_535.0
                })
                .collect::<Vec<_>>(),
            PixelType::F32 => bytes
                .chunks_exact(4)
                .take(voxel_count)
                .map(|chunk| {
                    if little_endian {
                        f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                    } else {
                        f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                    }
                })
                .collect::<Vec<_>>(),
        };
        if values.len() != voxel_count {
            return Err(
                IoError::Io(std::io::Error::other("raw data is smaller than expected")).into(),
            );
        }

        let mut shape = vec![height, width];
        let mut dims = vec![Dim::new(AxisKind::Y, height), Dim::new(AxisKind::X, width)];
        if slices > 1 {
            shape.push(slices);
            dims.push(Dim::new(AxisKind::Z, slices));
        }
        if channels > 1 {
            shape.push(channels);
            dims.push(Dim::new(AxisKind::Channel, channels));
        }
        let data = ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|error| {
            IoError::Io(std::io::Error::other(format!("raw shape error: {error}")))
        })?;
        let metadata = Metadata {
            dims,
            pixel_type,
            ..Metadata::default()
        };
        Ok(Dataset::new(data, metadata)?)
    }
}
