use std::path::Path;

use crate::model::{AxisKind, Dataset, DatasetF32, Dim, Metadata, PixelType};
use image::{DynamicImage, ImageBuffer, Luma, Rgb};
use ndarray::{Array, IxDyn};

use super::util::{metadata_for_dims, scale_to_u8};
use super::{IoError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeRasterImage {
    Gray8 {
        width: usize,
        height: usize,
        pixels: Vec<u8>,
        source: Option<std::path::PathBuf>,
    },
    Gray16 {
        width: usize,
        height: usize,
        pixels: Vec<u16>,
        source: Option<std::path::PathBuf>,
    },
    Rgb8 {
        width: usize,
        height: usize,
        pixels: Vec<u8>,
        source: Option<std::path::PathBuf>,
    },
}

impl NativeRasterImage {
    pub fn width(&self) -> usize {
        match self {
            Self::Gray8 { width, .. } | Self::Gray16 { width, .. } | Self::Rgb8 { width, .. } => {
                *width
            }
        }
    }

    pub fn height(&self) -> usize {
        match self {
            Self::Gray8 { height, .. }
            | Self::Gray16 { height, .. }
            | Self::Rgb8 { height, .. } => *height,
        }
    }

    pub fn channel_count(&self) -> usize {
        match self {
            Self::Rgb8 { .. } => 3,
            Self::Gray8 { .. } | Self::Gray16 { .. } => 1,
        }
    }

    pub fn pixel_type(&self) -> PixelType {
        match self {
            Self::Gray8 { .. } | Self::Rgb8 { .. } => PixelType::U8,
            Self::Gray16 { .. } => PixelType::U16,
        }
    }

    pub fn source(&self) -> Option<&Path> {
        match self {
            Self::Gray8 { source, .. }
            | Self::Gray16 { source, .. }
            | Self::Rgb8 { source, .. } => source.as_deref(),
        }
    }

    pub fn metadata(&self) -> Metadata {
        let mut dims = vec![
            Dim::new(AxisKind::Y, self.height()),
            Dim::new(AxisKind::X, self.width()),
        ];
        if self.channel_count() > 1 {
            dims.push(Dim::new(AxisKind::Channel, self.channel_count()));
        }
        let mut metadata = metadata_for_dims(
            self.source().unwrap_or_else(|| Path::new("<memory>")),
            dims,
            self.pixel_type(),
        );
        if self.channel_count() == 3 {
            metadata.channel_names = vec!["R".into(), "G".into(), "B".into()];
        }
        metadata
    }

    pub fn min_max(&self) -> (f32, f32) {
        match self {
            Self::Gray8 { pixels, .. } => min_max_u8(pixels),
            Self::Gray16 { pixels, .. } => min_max_u16(pixels),
            Self::Rgb8 { pixels, .. } => min_max_u8(pixels),
        }
    }

    pub fn to_dataset(&self) -> Result<DatasetF32> {
        match self {
            Self::Gray8 {
                width,
                height,
                pixels,
                ..
            } => {
                let values = pixels
                    .iter()
                    .map(|pixel| f32::from(*pixel))
                    .collect::<Vec<_>>();
                let data = Array::from_shape_vec((*height, *width), values)
                    .expect("shape checked")
                    .into_dyn();
                Ok(Dataset::new(data, self.metadata())?)
            }
            Self::Gray16 {
                width,
                height,
                pixels,
                ..
            } => {
                let values = pixels
                    .iter()
                    .map(|pixel| f32::from(*pixel))
                    .collect::<Vec<_>>();
                let data = Array::from_shape_vec((*height, *width), values)
                    .expect("shape checked")
                    .into_dyn();
                Ok(Dataset::new(data, self.metadata())?)
            }
            Self::Rgb8 {
                width,
                height,
                pixels,
                ..
            } => {
                let values = pixels
                    .iter()
                    .map(|pixel| f32::from(*pixel))
                    .collect::<Vec<_>>();
                let data = Array::from_shape_vec((*height, *width, 3usize), values)
                    .expect("shape checked")
                    .into_dyn();
                Ok(Dataset::new(data, self.metadata())?)
            }
        }
    }
}

pub(crate) fn read_common_raster(path: &Path) -> Result<DatasetF32> {
    let image = image::open(path)?;
    dataset_from_dynamic_image(image, Some(path))
}

pub(crate) fn read_native_raster(path: &Path) -> Result<Option<NativeRasterImage>> {
    let image = image::open(path)?;
    Ok(native_raster_from_dynamic_image(image, Some(path)))
}

pub(crate) fn read_common_raster_bytes(bytes: &[u8], format_hint: &str) -> Result<DatasetF32> {
    let image = image::load_from_memory(bytes)?;
    let pseudo_path = Path::new(format_hint);
    dataset_from_dynamic_image(image, Some(pseudo_path))
}

pub(crate) fn read_native_raster_bytes(
    bytes: &[u8],
    format_hint: &str,
) -> Result<Option<NativeRasterImage>> {
    let image = image::load_from_memory(bytes)?;
    Ok(native_raster_from_dynamic_image(
        image,
        Some(Path::new(format_hint)),
    ))
}

fn dataset_from_dynamic_image(image: DynamicImage, path: Option<&Path>) -> Result<DatasetF32> {
    match image {
        DynamicImage::ImageLuma8(buffer) => {
            let (width, height) = buffer.dimensions();
            let values = buffer
                .pixels()
                .map(|pixel| f32::from(pixel.0[0]))
                .collect::<Vec<_>>();
            let data = Array::from_shape_vec((height as usize, width as usize), values)
                .expect("shape checked")
                .into_dyn();
            let metadata = metadata_for_dims(
                path.unwrap_or_else(|| Path::new("<memory>")),
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
                .map(|pixel| f32::from(pixel.0[0]))
                .collect::<Vec<_>>();
            let data = Array::from_shape_vec((height as usize, width as usize), values)
                .expect("shape checked")
                .into_dyn();
            let metadata = metadata_for_dims(
                path.unwrap_or_else(|| Path::new("<memory>")),
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
                values.push(f32::from(pixel.0[0]));
                values.push(f32::from(pixel.0[1]));
                values.push(f32::from(pixel.0[2]));
            }
            let data = Array::from_shape_vec((height as usize, width as usize, 3usize), values)
                .expect("shape checked")
                .into_dyn();
            let mut metadata = metadata_for_dims(
                path.unwrap_or_else(|| Path::new("<memory>")),
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
                values.push(f32::from(pixel.0[0]));
                values.push(f32::from(pixel.0[1]));
                values.push(f32::from(pixel.0[2]));
            }
            let data = Array::from_shape_vec((height as usize, width as usize, 3usize), values)
                .expect("shape checked")
                .into_dyn();
            let mut metadata = metadata_for_dims(
                path.unwrap_or_else(|| Path::new("<memory>")),
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

fn native_raster_from_dynamic_image(
    image: DynamicImage,
    path: Option<&Path>,
) -> Option<NativeRasterImage> {
    let source = path.map(Path::to_path_buf);
    match image {
        DynamicImage::ImageLuma8(buffer) => {
            let (width, height) = buffer.dimensions();
            Some(NativeRasterImage::Gray8 {
                width: width as usize,
                height: height as usize,
                pixels: buffer.into_raw(),
                source,
            })
        }
        DynamicImage::ImageLuma16(buffer) => {
            let (width, height) = buffer.dimensions();
            Some(NativeRasterImage::Gray16 {
                width: width as usize,
                height: height as usize,
                pixels: buffer.into_raw(),
                source,
            })
        }
        DynamicImage::ImageRgb8(buffer) => {
            let (width, height) = buffer.dimensions();
            Some(NativeRasterImage::Rgb8 {
                width: width as usize,
                height: height as usize,
                pixels: buffer.into_raw(),
                source,
            })
        }
        DynamicImage::ImageRgba8(buffer) => {
            let (width, height) = buffer.dimensions();
            Some(NativeRasterImage::Rgb8 {
                width: width as usize,
                height: height as usize,
                pixels: flatten_rgba_to_rgb(buffer.into_raw()),
                source,
            })
        }
        _ => None,
    }
}

pub(crate) fn write_common_raster(path: &Path, dataset: &DatasetF32) -> Result<()> {
    let shape = dataset.shape();
    if shape.len() == 2 {
        let (height, width) = (shape[0], shape[1]);
        let values = dataset.data.iter().copied().collect::<Vec<_>>();
        let bytes = samples_to_u8_for_pixel_type(&values, dataset.metadata.pixel_type);
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
            let bytes = samples_to_u8_for_pixel_type(&values, dataset.metadata.pixel_type);
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
                    bytes.push(sample_to_u8_for_pixel_type(
                        dataset.data[IxDyn(&[y, x, 0])],
                        dataset.metadata.pixel_type,
                    ));
                    bytes.push(sample_to_u8_for_pixel_type(
                        dataset.data[IxDyn(&[y, x, 1])],
                        dataset.metadata.pixel_type,
                    ));
                    bytes.push(sample_to_u8_for_pixel_type(
                        dataset.data[IxDyn(&[y, x, 2])],
                        dataset.metadata.pixel_type,
                    ));
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

pub(crate) fn write_native_raster(path: &Path, raster: &NativeRasterImage) -> Result<()> {
    match raster {
        NativeRasterImage::Gray8 {
            width,
            height,
            pixels,
            ..
        } => {
            let image =
                ImageBuffer::<Luma<u8>, _>::from_vec(*width as u32, *height as u32, pixels.clone())
                    .ok_or_else(|| {
                        IoError::UnsupportedLayout("failed to construct gray image".into())
                    })?;
            image.save(path)?;
            Ok(())
        }
        NativeRasterImage::Gray16 {
            width,
            height,
            pixels,
            ..
        } => {
            let image = ImageBuffer::<Luma<u16>, _>::from_vec(
                *width as u32,
                *height as u32,
                pixels.clone(),
            )
            .ok_or_else(|| IoError::UnsupportedLayout("failed to construct gray image".into()))?;
            image.save(path)?;
            Ok(())
        }
        NativeRasterImage::Rgb8 {
            width,
            height,
            pixels,
            ..
        } => {
            let image =
                ImageBuffer::<Rgb<u8>, _>::from_vec(*width as u32, *height as u32, pixels.clone())
                    .ok_or_else(|| {
                        IoError::UnsupportedLayout("failed to construct RGB image".into())
                    })?;
            image.save(path)?;
            Ok(())
        }
    }
}

fn flatten_rgba_to_rgb(pixels: Vec<u8>) -> Vec<u8> {
    pixels
        .chunks_exact(4)
        .flat_map(|chunk| flatten_rgba_pixel(chunk).into_iter())
        .collect()
}

fn flatten_rgba_pixel(chunk: &[u8]) -> [u8; 3] {
    let alpha = f32::from(chunk[3]) / 255.0;
    [
        blend_channel(chunk[0], alpha),
        blend_channel(chunk[1], alpha),
        blend_channel(chunk[2], alpha),
    ]
}

fn blend_channel(channel: u8, alpha: f32) -> u8 {
    (f32::from(channel) * alpha + 255.0 * (1.0 - alpha)).round() as u8
}

fn samples_to_u8_for_pixel_type(values: &[f32], pixel_type: PixelType) -> Vec<u8> {
    match pixel_type {
        PixelType::U8 | PixelType::U16 => values
            .iter()
            .map(|value| sample_to_u8_for_pixel_type(*value, pixel_type))
            .collect(),
        PixelType::F32 => scale_to_u8(values),
    }
}

fn sample_to_u8_for_pixel_type(value: f32, pixel_type: PixelType) -> u8 {
    match pixel_type {
        PixelType::U8 => value.clamp(0.0, 255.0).round() as u8,
        PixelType::U16 => (value.clamp(0.0, 65_535.0) / 257.0).round() as u8,
        PixelType::F32 => (value.clamp(0.0, 1.0) * 255.0).round() as u8,
    }
}

fn min_max_u8(values: &[u8]) -> (f32, f32) {
    let Some(first) = values.first().copied() else {
        return (0.0, 0.0);
    };
    let mut min = first;
    let mut max = first;
    for value in values.iter().copied().skip(1) {
        min = min.min(value);
        max = max.max(value);
    }
    (f32::from(min), f32::from(max))
}

fn min_max_u16(values: &[u16]) -> (f32, f32) {
    let Some(first) = values.first().copied() else {
        return (0.0, 0.0);
    };
    let mut min = first;
    let mut max = first;
    for value in values.iter().copied().skip(1) {
        min = min.min(value);
        max = max.max(value);
    }
    (f32::from(min), f32::from(max))
}
