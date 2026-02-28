use std::path::Path;

use crate::model::DatasetF32;
use crate::workflow::{PipelineReport, PipelineSpec, load_spec, run_pipeline, save_report};

use super::{OpsService, Result};

#[derive(Debug, Clone, Default)]
pub struct PipelineService {
    ops: OpsService,
}

impl PipelineService {
    pub fn load_spec(&self, path: impl AsRef<Path>) -> Result<PipelineSpec> {
        Ok(load_spec(path)?)
    }

    pub fn run(
        &self,
        spec: &PipelineSpec,
        input: &DatasetF32,
    ) -> Result<(DatasetF32, PipelineReport)> {
        Ok(run_pipeline(spec, input, self.ops.registry())?)
    }

    pub fn save_report(&self, path: impl AsRef<Path>, report: &PipelineReport) -> Result<()> {
        save_report(path, report)?;
        Ok(())
    }
}
