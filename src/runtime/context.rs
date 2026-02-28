use super::{DatasetService, IoService, OpsService, PipelineService};

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
