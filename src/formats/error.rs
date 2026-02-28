use crate::model::CoreError;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, IoError>;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("unsupported image format: {0}")]
    UnsupportedFormat(String),

    #[error("unsupported dataset layout for this format: {0}")]
    UnsupportedLayout(String),

    #[error("I/O failure: {0}")]
    Io(#[from] std::io::Error),

    #[error("image decode/encode failure: {0}")]
    Image(#[from] image::ImageError),

    #[error("TIFF decode/encode failure: {0}")]
    Tiff(#[from] tiff::TiffError),

    #[error("core dataset/metadata failure: {0}")]
    Core(#[from] CoreError),
}
