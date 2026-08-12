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

    #[error("operation `{operation}` does not support scope {scope:?}")]
    UnsupportedScope {
        operation: String,
        scope: super::OperationScope,
    },

    #[error("active plane position is out of bounds: {0}")]
    ActivePosition(String),

    #[error("invalid area mask: {0}")]
    InvalidAreaMask(String),

    #[error("operation execution was cancelled")]
    Cancelled,

    #[error("operation `{operation}` returned an invalid result: {message}")]
    InvalidOperationOutput { operation: String, message: String },

    #[error("plugin operation `{operation}` failed: {message}")]
    PluginExecution { operation: String, message: String },

    #[error("core dataset error: {0}")]
    Core(#[from] CoreError),
}
