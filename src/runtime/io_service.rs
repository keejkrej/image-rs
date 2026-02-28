use std::path::Path;

use crate::formats::{read_dataset, write_dataset};
use crate::model::DatasetF32;

use super::Result;

#[derive(Debug, Default, Clone, Copy)]
pub struct IoService;

impl IoService {
    pub fn read(&self, path: impl AsRef<Path>) -> Result<DatasetF32> {
        Ok(read_dataset(path)?)
    }

    pub fn write(&self, path: impl AsRef<Path>, dataset: &DatasetF32) -> Result<()> {
        write_dataset(path, dataset)?;
        Ok(())
    }
}
