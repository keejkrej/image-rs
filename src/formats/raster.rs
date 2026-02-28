use std::path::Path;

use crate::model::{AxisKind, Dataset, DatasetF32, Dim, PixelType};
use image::{DynamicImage, ImageBuffer, Luma, Rgb};
use ndarray::{Array, IxDyn};

use super::util::{metadata_for_dims, scale_to_u8};
use super::{IoError, Result};

pub(crate) fn read_common_raster(path: &Path) -> Result<DatasetF32> {
    let image = image::open(path)?;
    match image {
        DynamicImage::ImageLuma8(buffer) => {
            let (width, height) = buffer.dimensions();
            let values = buffer
                .pixels()
                .map(|pixel| f32::from(pixel.0[0]) / 255.0)
                .collect::<Vec<_>>();
            let data = Array::from_shape_vec((height as usize, width as usize), values)
                .expect("shape checked")
                .into_dyn();
            let metadata = metadata_for_dims(
                path,
                vec![
                    Dim::new(AxisKind::Y, height as usize),
                    Dim::new(AxisKind::X, width as usize),
                ],
                PixelType::U8,
            );
            Ok(Dataset::new(data, metadata)?)
        }
        DynamicImage::ImageLuma16(buffer) => {
            let (width, height) = buffer.dimensions();
            let values = buffer
                .pixels()
                .map(|pixel| f32::from(pixel.0[0]) / 65_535.0)
                .collect::<Vec<_>>();
            let data = Array::from_shape_vec((height as usize, width as usize), values)
                .expect("shape checked")
                .into_dyn();
            let metadata = metadata_for_dims(
                path,
                vec![
                    Dim::new(AxisKind::Y, height as usize),
                    Dim::new(AxisKind::X, width as usize),
                ],
                PixelType::U16,
            );
            Ok(Dataset::new(data, metadata)?)
        }
        DynamicImage::ImageRgb16(buffer) => {
            let (width, height) = buffer.dimensions();
            let mut values = Vec::with_capacity(height as usize * width as usize * 3);
            for pixel in buffer.pixels() {
                values.push(f32::from(pixel.0[0]) / 65_535.0);
                values.push(f32::from(pixel.0[1]) / 65_535.0);
                values.push(f32::from(pixel.0[2]) / 65_535.0);
            }
            let data = Array::from_shape_vec((height as usize, width as usize, 3usize), values)
                .expect("shape checked")
                .into_dyn();
            let mut metadata = metadata_for_dims(
                path,
                vec![
                    Dim::new(AxisKind::Y, height as usize),
                    Dim::new(AxisKind::X, width as usize),
                    Dim::new(AxisKind::Channel, 3),
                ],
                PixelType::U16,
            );
            metadata.channel_names = vec!["R".into(), "G".into(), "B".into()];
            Ok(Dataset::new(data, metadata)?)
        }
        other => {
            let rgb = other.to_rgb8();
            let (width, height) = rgb.dimensions();
            let mut values = Vec::with_capacity(height as usize * width as usize * 3);
            for pixel in rgb.pixels() {
                values.push(f32::from(pixel.0[0]) / 255.0);
                values.push(f32::from(pixel.0[1]) / 255.0);
                values.push(f32::from(pixel.0[2]) / 255.0);
            }
            let data = Array::from_shape_vec((height as usize, width as usize, 3usize), values)
                .expect("shape checked")
                .into_dyn();
            let mut metadata = metadata_for_dims(
                path,
                vec![
                    Dim::new(AxisKind::Y, height as usize),
                    Dim::new(AxisKind::X, width as usize),
                    Dim::new(AxisKind::Channel, 3),
                ],
                PixelType::U8,
            );
            metadata.channel_names = vec!["R".into(), "G".into(), "B".into()];
            Ok(Dataset::new(data, metadata)?)
        }
    }
}

pub(crate) fn write_common_raster(path: &Path, dataset: &DatasetF32) -> Result<()> {
    let shape = dataset.shape();
    if shape.len() == 2 {
        let (height, width) = (shape[0], shape[1]);
        let values = dataset.data.iter().copied().collect::<Vec<_>>();
        let bytes = scale_to_u8(&values);
        let image = ImageBuffer::<Luma<u8>, _>::from_vec(width as u32, height as u32, bytes)
            .ok_or_else(|| IoError::UnsupportedLayout("failed to construct gray image".into()))?;
        image.save(path)?;
        return Ok(());
    }

    if shape.len() == 3 {
        if shape[2] == 1 {
            let (height, width) = (shape[0], shape[1]);
            let mut values = Vec::with_capacity(height * width);
            for y in 0..height {
                for x in 0..width {
                    values.push(dataset.data[IxDyn(&[y, x, 0])]);
                }
            }
            let bytes = scale_to_u8(&values);
            let image = ImageBuffer::<Luma<u8>, _>::from_vec(width as u32, height as u32, bytes)
                .ok_or_else(|| {
                    IoError::UnsupportedLayout("failed to construct gray image".into())
                })?;
            image.save(path)?;
            return Ok(());
        }

        if shape[2] == 3 {
            let (height, width) = (shape[0], shape[1]);
            let mut bytes = Vec::with_capacity(height * width * 3);
            for y in 0..height {
                for x in 0..width {
                    bytes.push(
                        (dataset.data[IxDyn(&[y, x, 0])].clamp(0.0, 1.0) * 255.0).round() as u8,
                    );
                    bytes.push(
                        (dataset.data[IxDyn(&[y, x, 1])].clamp(0.0, 1.0) * 255.0).round() as u8,
                    );
                    bytes.push(
                        (dataset.data[IxDyn(&[y, x, 2])].clamp(0.0, 1.0) * 255.0).round() as u8,
                    );
                }
            }
            let image = ImageBuffer::<Rgb<u8>, _>::from_vec(width as u32, height as u32, bytes)
                .ok_or_else(|| {
                    IoError::UnsupportedLayout("failed to construct RGB image".into())
                })?;
            image.save(path)?;
            return Ok(());
        }
    }

    Err(IoError::UnsupportedLayout(format!(
        "raster write expects [Y, X] or [Y, X, C], found shape {shape:?}"
    )))
}
