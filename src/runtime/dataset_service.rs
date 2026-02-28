use crate::model::DatasetF32;

use super::Result;

#[derive(Debug, Default, Clone, Copy)]
pub struct DatasetService;

impl DatasetService {
    pub fn validate(&self, dataset: &DatasetF32) -> Result<()> {
        dataset.validate()?;
        Ok(())
    }
}
