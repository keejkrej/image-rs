use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use ijrs_core::{CoreError, DatasetF32};
use ijrs_io::{IoError, read_dataset, write_dataset};
use ijrs_ops::{
    OpOutput, OpSchema, Operation, OpsError, default_registry, execute_operation_with_registry,
};
use ijrs_pipeline::{
    PipelineError, PipelineReport, PipelineSpec, load_spec, run_pipeline, save_report,
};
use serde_json::Value;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, AppError>;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("dataset service error: {0}")]
    Dataset(#[from] CoreError),

    #[error("I/O service error: {0}")]
    Io(#[from] IoError),

    #[error("operation service error: {0}")]
    Ops(#[from] OpsError),

    #[error("pipeline service error: {0}")]
    Pipeline(#[from] PipelineError),
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DatasetService;

impl DatasetService {
    pub fn validate(&self, dataset: &DatasetF32) -> Result<()> {
        dataset.validate()?;
        Ok(())
    }
}

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

#[derive(Clone)]
pub struct OpsService {
    registry: HashMap<&'static str, Arc<dyn Operation>>,
}

impl std::fmt::Debug for OpsService {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpsService")
            .field("registered_ops", &self.registry.len())
            .finish()
    }
}

impl Default for OpsService {
    fn default() -> Self {
        Self {
            registry: default_registry(),
        }
    }
}

impl OpsService {
    pub fn list(&self) -> Vec<OpSchema> {
        let mut schemas = self
            .registry
            .values()
            .map(|operation| operation.schema())
            .collect::<Vec<_>>();
        schemas.sort_by(|left, right| left.name.cmp(&right.name));
        schemas
    }

    pub fn execute(&self, op: &str, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        Ok(execute_operation_with_registry(
            &self.registry,
            op,
            dataset,
            params,
        )?)
    }

    pub fn registry(&self) -> &HashMap<&'static str, Arc<dyn Operation>> {
        &self.registry
    }
}

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

#[derive(Debug, Clone, Default)]
pub struct AppContext {
    dataset_service: DatasetService,
    io_service: IoService,
    ops_service: OpsService,
    pipeline_service: PipelineService,
}

impl AppContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dataset_service(&self) -> &DatasetService {
        &self.dataset_service
    }

    pub fn io_service(&self) -> &IoService {
        &self.io_service
    }

    pub fn ops_service(&self) -> &OpsService {
        &self.ops_service
    }

    pub fn pipeline_service(&self) -> &PipelineService {
        &self.pipeline_service
    }
}
