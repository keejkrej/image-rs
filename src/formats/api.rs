use std::path::{Path, PathBuf};

use crate::model::DatasetF32;

use super::raster::{
    NativeRasterImage, read_common_raster, read_common_raster_bytes, read_native_raster,
    read_native_raster_bytes, write_common_raster, write_native_raster,
};
use super::tiff::{read_tiff, read_tiff_bytes, write_tiff};
use super::util::extension;
use super::{IoError, Result};

pub fn read_dataset(path: impl AsRef<Path>) -> Result<DatasetF32> {
    let path = path.as_ref();
    let extension = extension(path)?;
    match extension.as_str() {
        "png" | "jpg" | "jpeg" => read_common_raster(path),
        "tif" | "tiff" => read_tiff(path),
        other => Err(IoError::UnsupportedFormat(other.to_string())),
    }
}

pub fn read_dataset_bytes(bytes: &[u8], format_hint: &str) -> Result<DatasetF32> {
    match format_hint.to_ascii_lowercase().as_str() {
        "png" | "jpg" | "jpeg" => read_common_raster_bytes(bytes, format_hint),
        "tif" | "tiff" => read_tiff_bytes(bytes, format_hint),
        other => Err(IoError::UnsupportedFormat(other.to_string())),
    }
}

pub fn read_native_image(path: impl AsRef<Path>) -> Result<Option<NativeRasterImage>> {
    let path = path.as_ref();
    let extension = extension(path)?;
    match extension.as_str() {
        "png" | "jpg" | "jpeg" => read_native_raster(path),
        _ => Ok(None),
    }
}

pub fn read_native_image_bytes(
    bytes: &[u8],
    format_hint: &str,
) -> Result<Option<NativeRasterImage>> {
    match format_hint.to_ascii_lowercase().as_str() {
        "png" | "jpg" | "jpeg" => read_native_raster_bytes(bytes, format_hint),
        _ => Ok(None),
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

pub fn write_native_image(path: impl AsRef<Path>, image: &NativeRasterImage) -> Result<()> {
    let path = path.as_ref();
    let extension = extension(path)?;
    match extension.as_str() {
        "png" | "jpg" | "jpeg" => write_native_raster(path, image),
        other => Err(IoError::UnsupportedFormat(other.to_string())),
    }
}

pub fn supported_formats() -> &'static [&'static str] {
    &["png", "jpg", "jpeg", "tif", "tiff"]
}

pub fn save_slice_png(dataset: &DatasetF32, path: &Path) -> Result<()> {
    write_common_raster(path, dataset)
}

pub fn source_path(dataset: &DatasetF32) -> Option<PathBuf> {
    dataset.metadata.source.clone()
}
