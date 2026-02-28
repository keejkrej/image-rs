use crate::commands::OpsError;
use crate::formats::IoError;
use crate::model::CoreError;
use crate::workflow::PipelineError;
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
