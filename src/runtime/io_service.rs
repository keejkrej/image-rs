use std::path::Path;

use crate::formats::{
    IoError, NativeRasterImage, read_dataset, read_dataset_bytes, read_native_image, write_dataset,
    write_native_image,
};
use crate::model::{AxisKind, Dataset, DatasetF32, Dim, Metadata, PixelType};
use ndarray::{ArrayD, IxDyn};

use super::Result;

#[derive(Debug, Default, Clone, Copy)]
pub struct IoService;

impl IoService {
    pub fn read(&self, path: impl AsRef<Path>) -> Result<DatasetF32> {
        Ok(read_dataset(path)?)
    }

    pub fn read_native(&self, path: impl AsRef<Path>) -> Result<Option<NativeRasterImage>> {
        Ok(read_native_image(path)?)
    }

    pub fn read_bytes(&self, bytes: &[u8], format_hint: &str) -> Result<DatasetF32> {
        Ok(read_dataset_bytes(bytes, format_hint)?)
    }

    pub fn write(&self, path: impl AsRef<Path>, dataset: &DatasetF32) -> Result<()> {
        write_dataset(path, dataset)?;
        Ok(())
    }

    pub fn write_native(&self, path: impl AsRef<Path>, image: &NativeRasterImage) -> Result<()> {
        write_native_image(path, image)?;
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
                .map(f32::from)
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
                    f32::from(raw)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_raw_preserves_integer_sample_values() {
        let service = IoService;
        let u8_dataset = service
            .read_raw(&[0, 10, 128, 255], 2, 2, 1, 1, PixelType::U8, true, 0)
            .expect("u8 raw");
        assert_eq!(u8_dataset.metadata.pixel_type, PixelType::U8);
        assert_eq!(
            u8_dataset.data.iter().copied().collect::<Vec<_>>(),
            vec![0.0, 10.0, 128.0, 255.0]
        );

        let mut bytes = Vec::new();
        for value in [0_u16, 10, 4096, 65_535] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let u16_dataset = service
            .read_raw(&bytes, 2, 2, 1, 1, PixelType::U16, true, 0)
            .expect("u16 raw");
        assert_eq!(u16_dataset.metadata.pixel_type, PixelType::U16);
        assert_eq!(
            u16_dataset.data.iter().copied().collect::<Vec<_>>(),
            vec![0.0, 10.0, 4096.0, 65_535.0]
        );
    }
}
