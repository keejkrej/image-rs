use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::model::DatasetF32;
use serde_json::Value;

use super::{
    ComponentsLabelOp, GaussianBlurOp, ImageBinOp, ImageCalibrateOp, ImageCanvasResizeOp,
    ImageColorThresholdOp, ImageConvertOp, ImageConvolveOp, ImageCoordinatesOp, ImageCropOp,
    ImageFftBandpassOp, ImageFftPowerSpectrumOp, ImageFindEdgesOp, ImageFindMaximaOp, ImageFlipOp,
    ImageHyperstackReduceDimensionalityOp, ImageHyperstackSubsetOp, ImageHyperstackToStackOp,
    ImageMedianFilterOp, ImageRankFilter3dOp, ImageRankFilterOp, ImageRemoveNaNsOp,
    ImageRemoveOutliersOp, ImageResizeOp, ImageRotate90Op, ImageRotateOp, ImageScaleOp,
    ImageSetScaleOp, ImageShadowDemoOp, ImageShadowOp, ImageSharpenOp, ImageStackAddSliceOp,
    ImageStackDeleteSliceOp, ImageStackGroupedZProjectOp, ImageStackMontageOp,
    ImageStackMontageToStackOp, ImageStackReduceOp, ImageStackResliceOp, ImageStackStatisticsOp,
    ImageStackSubstackOp, ImageStackToHyperstackOp, ImageStackZProfileOp, ImageStackZProjectOp,
    ImageSubtractBackgroundOp, ImageSurfacePlotOp, ImageSwapQuadrantsOp, ImageTranslateOp,
    ImageUnsharpMaskOp, IntensityEnhanceContrastOp, IntensityInvertOp, IntensityMathOp,
    IntensityNaNBackgroundOp, IntensityNormalizeOp, IntensityWindowOp, MeasurementsHistogramOp,
    MeasurementsProfileOp, MeasurementsSummaryOp, MorphologyBinaryMedianOp, MorphologyCloseOp,
    MorphologyDilateOp, MorphologyDistanceMapOp, MorphologyErodeOp, MorphologyFillHolesOp,
    MorphologyOpenOp, MorphologyOutlineOp, MorphologySkeletonizeOp, MorphologyUltimatePointsOp,
    MorphologyVoronoiOp, MorphologyWatershedOp, NoiseGaussianOp, NoiseSaltAndPepperOp, OpOutput,
    OpSchema, Operation, OpsError, Result, ThresholdFixedOp, ThresholdMakeBinaryOp,
    ThresholdOtsuOp,
};
#[cfg(feature = "morpholib")]
use super::{
    MorpholibjChamferDistanceOp, MorpholibjReconstructByDilationOp,
    MorpholibjReconstructByErosionOp,
};
#[cfg(feature = "thunderstorm")]
use super::{
    ThunderstormGaussianFilterOp, ThunderstormLsqGaussianFitOp, ThunderstormNonMaxSuppressionOp,
    ThunderstormPipelineLocalizeOp,
};

type Registry = HashMap<&'static str, Arc<dyn Operation>>;

fn register<O: Operation + 'static>(map: &mut Registry, operation: O) {
    map.insert(operation.name(), Arc::new(operation));
}

fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut map: Registry = HashMap::new();
        register(&mut map, IntensityNormalizeOp);
        register(&mut map, IntensityEnhanceContrastOp);
        register(&mut map, IntensityInvertOp);
        register(&mut map, IntensityMathOp);
        register(&mut map, IntensityNaNBackgroundOp);
        register(&mut map, IntensityWindowOp);
        register(&mut map, GaussianBlurOp);
        register(&mut map, ImageConvertOp);
        register(&mut map, ImageColorThresholdOp);
        register(&mut map, ImageResizeOp);
        register(&mut map, ImageScaleOp);
        register(&mut map, ImageCanvasResizeOp);
        register(&mut map, ImageCropOp);
        register(&mut map, ImageCoordinatesOp);
        register(&mut map, ImageSetScaleOp);
        register(&mut map, ImageCalibrateOp);
        register(&mut map, ImageStackAddSliceOp);
        register(&mut map, ImageStackDeleteSliceOp);
        register(&mut map, ImageStackZProjectOp);
        register(&mut map, ImageStackMontageOp);
        register(&mut map, ImageStackMontageToStackOp);
        register(&mut map, ImageStackGroupedZProjectOp);
        register(&mut map, ImageStackReduceOp);
        register(&mut map, ImageStackResliceOp);
        register(&mut map, ImageStackSubstackOp);
        register(&mut map, ImageStackToHyperstackOp);
        register(&mut map, ImageHyperstackToStackOp);
        register(&mut map, ImageHyperstackReduceDimensionalityOp);
        register(&mut map, ImageHyperstackSubsetOp);
        register(&mut map, ImageStackZProfileOp);
        register(&mut map, ImageStackStatisticsOp);
        register(&mut map, ImageBinOp);
        register(&mut map, ImageFlipOp);
        register(&mut map, ImageRotate90Op);
        register(&mut map, ImageRotateOp);
        register(&mut map, ImageTranslateOp);
        register(&mut map, ImageRankFilterOp);
        register(&mut map, ImageRankFilter3dOp);
        register(&mut map, ImageMedianFilterOp);
        register(&mut map, ImageRemoveNaNsOp);
        register(&mut map, ImageRemoveOutliersOp);
        register(&mut map, ImageSharpenOp);
        register(&mut map, ImageFindEdgesOp);
        register(&mut map, ImageFindMaximaOp);
        register(&mut map, ImageShadowOp);
        register(&mut map, ImageShadowDemoOp);
        register(&mut map, ImageSubtractBackgroundOp);
        register(&mut map, ImageUnsharpMaskOp);
        register(&mut map, ImageConvolveOp);
        register(&mut map, ImageSwapQuadrantsOp);
        register(&mut map, ImageFftPowerSpectrumOp);
        register(&mut map, ImageFftBandpassOp);
        register(&mut map, ImageSurfacePlotOp);
        register(&mut map, ThresholdFixedOp);
        register(&mut map, ThresholdMakeBinaryOp);
        register(&mut map, ThresholdOtsuOp);
        register(&mut map, MeasurementsHistogramOp);
        register(&mut map, MeasurementsProfileOp);
        register(&mut map, MorphologyErodeOp);
        register(&mut map, MorphologyDilateOp);
        register(&mut map, MorphologyOpenOp);
        register(&mut map, MorphologyCloseOp);
        register(&mut map, MorphologyBinaryMedianOp);
        register(&mut map, MorphologyDistanceMapOp);
        register(&mut map, MorphologyUltimatePointsOp);
        register(&mut map, MorphologyWatershedOp);
        register(&mut map, MorphologyVoronoiOp);
        register(&mut map, MorphologyFillHolesOp);
        register(&mut map, MorphologyOutlineOp);
        register(&mut map, MorphologySkeletonizeOp);
        register(&mut map, NoiseGaussianOp);
        register(&mut map, NoiseSaltAndPepperOp);
        register(&mut map, ComponentsLabelOp);
        #[cfg(feature = "morpholib")]
        register(&mut map, MorpholibjChamferDistanceOp);
        #[cfg(feature = "morpholib")]
        register(&mut map, MorpholibjReconstructByDilationOp);
        #[cfg(feature = "morpholib")]
        register(&mut map, MorpholibjReconstructByErosionOp);
        #[cfg(feature = "thunderstorm")]
        register(&mut map, ThunderstormGaussianFilterOp);
        #[cfg(feature = "thunderstorm")]
        register(&mut map, ThunderstormNonMaxSuppressionOp);
        #[cfg(feature = "thunderstorm")]
        register(&mut map, ThunderstormLsqGaussianFitOp);
        #[cfg(feature = "thunderstorm")]
        register(&mut map, ThunderstormPipelineLocalizeOp);
        register(&mut map, MeasurementsSummaryOp);
        map
    })
}

pub fn default_registry() -> HashMap<&'static str, Arc<dyn Operation>> {
    registry()
        .iter()
        .map(|(name, op)| (*name, Arc::clone(op)))
        .collect()
}

pub fn list_operations() -> Vec<OpSchema> {
    let mut schemas = registry()
        .values()
        .map(|op| op.schema())
        .collect::<Vec<_>>();
    schemas.sort_by(|left, right| left.name.cmp(&right.name));
    schemas
}

pub fn execute_operation(name: &str, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
    let op = registry()
        .get(name)
        .ok_or_else(|| OpsError::UnknownOperation(name.to_string()))?;
    op.execute(dataset, params)
}

pub fn execute_operation_with_registry(
    registry: &HashMap<&'static str, Arc<dyn Operation>>,
    name: &str,
    dataset: &DatasetF32,
    params: &Value,
) -> Result<OpOutput> {
    let op = registry
        .get(name)
        .ok_or_else(|| OpsError::UnknownOperation(name.to_string()))?;
    op.execute(dataset, params)
}
