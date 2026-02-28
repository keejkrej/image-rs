mod axis;
mod dataset;
mod error;
mod metadata;

#[cfg(test)]
mod tests;

pub use axis::{AxisKind, PixelType, default_axis_for_index};
pub use dataset::{Dataset, DatasetF32};
pub use error::{CoreError, Result};
pub use metadata::{Dim, Metadata};
