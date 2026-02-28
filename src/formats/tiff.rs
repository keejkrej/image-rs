use std::fs::File;
use std::path::Path;

use crate::model::{AxisKind, Dataset, DatasetF32, Dim, PixelType};
use ndarray::{Array, IxDyn};
use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::{TiffEncoder, colortype};

use super::util::{metadata_for_dims, to_u8_samples, to_u16_samples};
use super::{IoError, Result};

pub(crate) fn read_tiff(path: &Path) -> Result<DatasetF32> {
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

pub(crate) fn write_tiff(path: &Path, dataset: &DatasetF32) -> Result<()> {
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
