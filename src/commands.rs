mod axes;
mod components;
mod error;
mod gaussian;
mod intensity;
mod measurements;
#[cfg(feature = "morpholib")]
mod morpholibj;
mod morphology;
mod params;
mod registry;
mod schema;
mod threshold;
#[cfg(feature = "thunderstorm")]
mod thunderstorm;
mod util;

#[cfg(test)]
mod tests;

pub use components::ComponentsLabelOp;
pub use error::{OpsError, Result};
pub use gaussian::GaussianBlurOp;
pub use intensity::{IntensityNormalizeOp, IntensityWindowOp};
pub use measurements::MeasurementsSummaryOp;
#[cfg(feature = "morpholib")]
pub use morpholibj::{
    MorpholibjChamferDistanceOp, MorpholibjReconstructByDilationOp,
    MorpholibjReconstructByErosionOp,
};
pub use morphology::{MorphologyCloseOp, MorphologyDilateOp, MorphologyErodeOp, MorphologyOpenOp};
pub use registry::{
    default_registry, execute_operation, execute_operation_with_registry, list_operations,
};
pub use schema::{MeasurementTable, OpOutput, OpSchema, Operation, ParamSpec};
pub use threshold::{ThresholdFixedOp, ThresholdOtsuOp};
#[cfg(feature = "thunderstorm")]
pub use thunderstorm::{
    ThunderstormGaussianFilterOp, ThunderstormLsqGaussianFitOp, ThunderstormNonMaxSuppressionOp,
    ThunderstormPipelineLocalizeOp,
};

pub(crate) use axes::spatial_axes;
#[cfg(feature = "morpholib")]
pub(crate) use params::get_optional_bool;
pub(crate) use params::{get_optional_f32, get_optional_usize, get_required_f32};
