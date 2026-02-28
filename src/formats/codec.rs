use std::path::Path;

use crate::model::DatasetF32;

use super::{Result, read_dataset, write_dataset};

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
