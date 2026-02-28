use thiserror::Error;

pub type Result<T> = std::result::Result<T, CoreError>;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error(
        "metadata dimensionality mismatch: data has {data_ndim} dimensions but metadata has {meta_ndim}"
    )]
    DimensionalityMismatch { data_ndim: usize, meta_ndim: usize },

    #[error(
        "dimension size mismatch at axis {axis}: data size {data_size} != metadata size {meta_size}"
    )]
    DimensionSizeMismatch {
        axis: usize,
        data_size: usize,
        meta_size: usize,
    },

    #[error("invalid dimension size 0 at axis {axis}")]
    ZeroSizedDimension { axis: usize },

    #[error("invalid metadata: {0}")]
    InvalidMetadata(String),
}
