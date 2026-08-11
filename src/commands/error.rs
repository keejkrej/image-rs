use crate::model::CoreError;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, OpsError>;

#[derive(Debug, Error)]
pub enum OpsError {
    #[error("unknown operation: {0}")]
    UnknownOperation(String),

    #[error("duplicate operation: {0}")]
    DuplicateOperation(String),

    #[error("invalid operation parameters: {0}")]
    InvalidParams(String),

    #[error("unsupported dataset layout: {0}")]
    UnsupportedLayout(String),

    #[error("plugin operation `{operation}` failed: {message}")]
    PluginExecution { operation: String, message: String },

    #[error("core dataset error: {0}")]
    Core(#[from] CoreError),
}
