use std::fs::File;
use std::path::{Path, PathBuf};

use image_model::{AxisKind, CoreError, Dataset, DatasetF32, Dim, Metadata, PixelType};
use image::{DynamicImage, ImageBuffer, Luma, Rgb};
use ndarray::{Array, IxDyn};
use thiserror::Error;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::{TiffEncoder, colortype};

pub type Result<T> = std::result::Result<T, IoError>;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("unsupported image format: {0}")]
    UnsupportedFormat(String),

    #[error("unsupported dataset layout for this format: {0}")]
    UnsupportedLayout(String),

    #[error("I/O failure: {0}")]
    Io(#[from] std::io::Error),

    #[error("image decode/encode failure: {0}")]
    Image(#[from] image::ImageError),

    #[error("TIFF decode/encode failure: {0}")]
    Tiff(#[from] tiff::TiffError),

    #[error("core dataset/metadata failure: {0}")]
    Core(#[from] CoreError),
}

pub trait ImageReader {
    fn supports_extension(&self, extension: &str) -> bool;
    fn read(&self, path: &Path) -> Result<DatasetF32>;
}

pub trait ImageWriter {
    fn supports_extension(&self, extension: &str) -> bool;
    fn write(&self, path: &Path, dataset: &DatasetF32) -> Result<()>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultImageCodec;

impl ImageReader for DefaultImageCodec {
    fn supports_extension(&self, extension: &str) -> bool {
        matches!(extension, "png" | "jpg" | "jpeg" | "tif" | "tiff")
    }

    fn read(&self, path: &Path) -> Result<DatasetF32> {
        read_dataset(path)
    }
}

impl ImageWriter for DefaultImageCodec {
    fn supports_extension(&self, extension: &str) -> bool {
        matches!(extension, "png" | "jpg" | "jpeg" | "tif" | "tiff")
    }

    fn write(&self, path: &Path, dataset: &DatasetF32) -> Result<()> {
        write_dataset(path, dataset)
    }
}

pub fn read_dataset(path: impl AsRef<Path>) -> Result<DatasetF32> {
    let path = path.as_ref();
    let extension = extension(path)?;
    match extension.as_str() {
        "png" | "jpg" | "jpeg" => read_common_raster(path),
        "tif" | "tiff" => read_tiff(path),
        other => Err(IoError::UnsupportedFormat(other.to_string())),
    }
}

pub fn write_dataset(path: impl AsRef<Path>, dataset: &DatasetF32) -> Result<()> {
    let path = path.as_ref();
    let extension = extension(path)?;
    match extension.as_str() {
        "png" | "jpg" | "jpeg" => write_common_raster(path, dataset),
        "tif" | "tiff" => write_tiff(path, dataset),
        other => Err(IoError::UnsupportedFormat(other.to_string())),
    }
}

pub fn supported_formats() -> &'static [&'static str] {
    &["png", "jpg", "jpeg", "tif", "tiff"]
}

fn extension(path: &Path) -> Result<String> {
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .ok_or_else(|| IoError::UnsupportedFormat(path.to_string_lossy().to_string()))?;
    Ok(ext)
}

fn metadata_for_dims(path: &Path, dims: Vec<Dim>, pixel_type: PixelType) -> Metadata {
    Metadata {
        dims,
        pixel_type,
        source: Some(path.to_path_buf()),
        ..Metadata::default()
    }
}

fn read_common_raster(path: &Path) -> Result<DatasetF32> {
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

fn write_common_raster(path: &Path, dataset: &DatasetF32) -> Result<()> {
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
                    IoError::UnsupportedLayout("failed to construct single-channel image".into())
                })?;
            image.save(path)?;
            return Ok(());
        }
        if shape[2] >= 3 {
            let (height, width, channels) = (shape[0], shape[1], shape[2]);
            let mut values = Vec::with_capacity(height * width * 3);
            for y in 0..height {
                for x in 0..width {
                    values.push(dataset.data[IxDyn(&[y, x, 0])]);
                    values.push(dataset.data[IxDyn(&[y, x, 1])]);
                    values.push(dataset.data[IxDyn(&[y, x, if channels > 2 { 2 } else { 0 }])]);
                }
            }
            let bytes = scale_to_u8(&values);
            let image = ImageBuffer::<Rgb<u8>, _>::from_vec(width as u32, height as u32, bytes)
                .ok_or_else(|| {
                    IoError::UnsupportedLayout("failed to construct RGB image".into())
                })?;
            image.save(path)?;
            return Ok(());
        }
    }

    Err(IoError::UnsupportedLayout(format!(
        "PNG/JPEG expects [Y, X] or [Y, X, C], found shape {shape:?}"
    )))
}

fn read_tiff(path: &Path) -> Result<DatasetF32> {
    let file = File::open(path)?;
    let mut decoder = Decoder::new(file)?;
    let (width, height) = decoder.dimensions()?;
    let mut pages = Vec::new();
    let mut pixel_type = PixelType::F32;

    loop {
        let page = decode_tiff_page(&mut decoder, &mut pixel_type, width, height)?;
        pages.push(page);
        if !decoder.more_images() {
            break;
        }
        decoder.next_image()?;
        let (other_width, other_height) = decoder.dimensions()?;
        if other_width != width || other_height != height {
            return Err(IoError::UnsupportedLayout(
                "TIFF pages must have identical dimensions".into(),
            ));
        }
    }

    let dims = if pages.len() == 1 {
        vec![
            Dim::new(AxisKind::Y, height as usize),
            Dim::new(AxisKind::X, width as usize),
        ]
    } else {
        vec![
            Dim::new(AxisKind::Y, height as usize),
            Dim::new(AxisKind::X, width as usize),
            Dim::new(AxisKind::Z, pages.len()),
        ]
    };

    let data = if pages.len() == 1 {
        Array::from_shape_vec((height as usize, width as usize), pages.remove(0))
            .expect("shape checked")
            .into_dyn()
    } else {
        let depth = pages.len();
        let mut values = vec![0.0_f32; height as usize * width as usize * depth];
        for (z, page) in pages.iter().enumerate() {
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let destination = z + depth * (x + width as usize * y);
                    values[destination] = page[x + width as usize * y];
                }
            }
        }
        Array::from_shape_vec((height as usize, width as usize, depth), values)
            .expect("shape checked")
            .into_dyn()
    };

    let metadata = metadata_for_dims(path, dims, pixel_type);
    Ok(Dataset::new(data, metadata)?)
}

fn decode_tiff_page(
    decoder: &mut Decoder<File>,
    pixel_type: &mut PixelType,
    width: u32,
    height: u32,
) -> Result<Vec<f32>> {
    let values = match decoder.read_image()? {
        DecodingResult::U8(buffer) => {
            *pixel_type = PixelType::U8;
            if buffer.len() != width as usize * height as usize {
                return Err(IoError::UnsupportedLayout(
                    "TIFF RGB/alpha pages are not yet supported".into(),
                ));
            }
            buffer
                .into_iter()
                .map(|value| f32::from(value) / 255.0)
                .collect::<Vec<_>>()
        }
        DecodingResult::U16(buffer) => {
            *pixel_type = PixelType::U16;
            if buffer.len() != width as usize * height as usize {
                return Err(IoError::UnsupportedLayout(
                    "TIFF RGB/alpha pages are not yet supported".into(),
                ));
            }
            buffer
                .into_iter()
                .map(|value| f32::from(value) / 65_535.0)
                .collect::<Vec<_>>()
        }
        DecodingResult::F32(buffer) => {
            *pixel_type = PixelType::F32;
            if buffer.len() != width as usize * height as usize {
                return Err(IoError::UnsupportedLayout(
                    "TIFF RGB/alpha pages are not yet supported".into(),
                ));
            }
            buffer
        }
        other => {
            return Err(IoError::UnsupportedLayout(format!(
                "unsupported TIFF sample type: {other:?}"
            )));
        }
    };
    Ok(values)
}

fn write_tiff(path: &Path, dataset: &DatasetF32) -> Result<()> {
    let shape = dataset.shape();
    if shape.len() != 2 && shape.len() != 3 {
        return Err(IoError::UnsupportedLayout(format!(
            "TIFF supports [Y, X] or [Y, X, Z], found shape {shape:?}"
        )));
    }
    let height = shape[0] as u32;
    let width = shape[1] as u32;
    let depth = if shape.len() == 2 { 1 } else { shape[2] };
    let file = File::create(path)?;
    let mut encoder = TiffEncoder::new(file)?;

    for z in 0..depth {
        match dataset.metadata.pixel_type {
            PixelType::U8 => {
                let page = if shape.len() == 2 {
                    dataset.data.iter().copied().collect::<Vec<_>>()
                } else {
                    extract_tiff_page(dataset, z)?
                };
                let page = to_u8_samples(&page);
                let image = encoder.new_image::<colortype::Gray8>(width, height)?;
                image.write_data(&page)?;
            }
            PixelType::U16 => {
                let page = if shape.len() == 2 {
                    dataset.data.iter().copied().collect::<Vec<_>>()
                } else {
                    extract_tiff_page(dataset, z)?
                };
                let page = to_u16_samples(&page);
                let image = encoder.new_image::<colortype::Gray16>(width, height)?;
                image.write_data(&page)?;
            }
            PixelType::F32 => {
                let page = if shape.len() == 2 {
                    dataset.data.iter().copied().collect::<Vec<_>>()
                } else {
                    extract_tiff_page(dataset, z)?
                };
                let image = encoder.new_image::<colortype::Gray32Float>(width, height)?;
                image.write_data(&page)?;
            }
        }
    }
    Ok(())
}

fn extract_tiff_page(dataset: &DatasetF32, z: usize) -> Result<Vec<f32>> {
    let shape = dataset.shape();
    if shape.len() != 3 {
        return Err(IoError::UnsupportedLayout("expected [Y, X, Z]".into()));
    }
    let mut page = Vec::with_capacity(shape[0] * shape[1]);
    for y in 0..shape[0] {
        for x in 0..shape[1] {
            page.push(dataset.data[IxDyn(&[y, x, z])]);
        }
    }
    Ok(page)
}

fn to_u8_samples(values: &[f32]) -> Vec<u8> {
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

fn to_u16_samples(values: &[f32]) -> Vec<u16> {
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

fn scale_to_u8(values: &[f32]) -> Vec<u8> {
    to_u8_samples(values)
}

fn min_max(values: &[f32]) -> (f32, f32) {
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

pub fn save_slice_png(dataset: &DatasetF32, path: &Path) -> Result<()> {
    write_common_raster(path, dataset)
}

pub fn source_path(dataset: &DatasetF32) -> Option<PathBuf> {
    dataset.metadata.source.clone()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use image::{ImageBuffer, Luma, Rgb};
    use ndarray::Array;
    use tempfile::tempdir;

    use super::{read_dataset, write_dataset};
    use image_model::{AxisKind, Dataset, Dim, Metadata, PixelType};

    #[test]
    fn tiff_roundtrip_preserves_shape_and_type() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("roundtrip.tiff");
        let values = vec![0.0_f32, 0.25, 0.5, 1.0];
        let data = Array::from_shape_vec((2, 2), values)
            .expect("shape")
            .into_dyn();
        let metadata = Metadata {
            dims: vec![Dim::new(AxisKind::Y, 2), Dim::new(AxisKind::X, 2)],
            pixel_type: PixelType::F32,
            ..Metadata::default()
        };
        let dataset = Dataset::new(data, metadata).expect("dataset");
        write_dataset(&path, &dataset).expect("write tiff");
        let restored = read_dataset(&path).expect("read tiff");
        assert_eq!(restored.shape(), &[2, 2]);
        assert_eq!(restored.metadata.pixel_type, PixelType::F32);
    }

    #[test]
    fn png_and_jpeg_decode_channel_mapping() {
        let dir = tempdir().expect("tempdir");
        let png_path = dir.path().join("color.png");
        let jpg_path = dir.path().join("color.jpg");
        let mut image = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(2, 1);
        image.put_pixel(0, 0, Rgb([255, 0, 0]));
        image.put_pixel(1, 0, Rgb([0, 255, 0]));
        image.save(&png_path).expect("save png");
        image.save(&jpg_path).expect("save jpg");

        let png = read_dataset(&png_path).expect("read png");
        let jpg = read_dataset(&jpg_path).expect("read jpg");
        assert_eq!(png.shape(), &[1, 2, 3]);
        assert_eq!(jpg.shape(), &[1, 2, 3]);
        assert_eq!(png.metadata.axis_index(AxisKind::Channel), Some(2));
        assert_eq!(jpg.metadata.axis_index(AxisKind::Channel), Some(2));
    }

    #[test]
    fn conversion_between_formats() {
        let dir = tempdir().expect("tempdir");
        let input = dir.path().join("input.png");
        let output = dir.path().join("output.tiff");
        let image =
            ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(2, 2, vec![0, 50, 100, 255]).expect("image");
        image.save(&input).expect("save input");
        let dataset = read_dataset(&input).expect("read input");
        write_dataset(&output, &dataset).expect("write output");
        let restored = read_dataset(&output).expect("read output");
        assert_eq!(restored.shape(), &[2, 2]);
    }

    #[test]
    fn unsupported_layout_errors() {
        let dir = tempdir().expect("tempdir");
        let output = dir.path().join("bad.png");
        let data = Array::zeros((2, 2, 2, 2)).into_dyn();
        let metadata = Metadata::from_shape(&[2, 2, 2, 2], PixelType::F32);
        let dataset = Dataset::new(data, metadata).expect("dataset");
        let err = write_dataset(Path::new(&output), &dataset).expect_err("must fail");
        let message = err.to_string();
        assert!(message.contains("expects [Y, X] or [Y, X, C]"));
    }
}
