use std::path::{Path, PathBuf};

use crate::model::DatasetF32;

use super::raster::{read_common_raster, write_common_raster};
use super::tiff::{read_tiff, write_tiff};
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

pub fn save_slice_png(dataset: &DatasetF32, path: &Path) -> Result<()> {
    write_common_raster(path, dataset)
}

pub fn source_path(dataset: &DatasetF32) -> Option<PathBuf> {
    dataset.metadata.source.clone()
}
