use crate::commands::OpsError;
use crate::model::CoreError;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, PipelineError>;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("pipeline specification parse failure: {0}")]
    Parse(String),

    #[error("pipeline I/O failure: {0}")]
    Io(#[from] std::io::Error),

    #[error("pipeline serialization failure: {0}")]
    SerdeJson(#[from] serde_json::Error),

    #[error("pipeline YAML serialization failure: {0}")]
    SerdeYaml(#[from] serde_yaml::Error),

    #[error("operation execution failed: {0}")]
    Operation(#[from] OpsError),

    #[error("dataset validation failed: {0}")]
    Core(#[from] CoreError),
}
