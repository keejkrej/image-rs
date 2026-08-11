use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::model::DatasetF32;
use serde_json::Value;

use super::{
    AnalyzeParticlesOp, ComponentsLabelOp, GaussianBlurOp, ImageBinOp, ImageCalibrateOp,
    ImageCanvasResizeOp, ImageColorThresholdOp, ImageConvertOp, ImageConvolveOp,
    ImageCoordinatesOp, ImageCropOp, ImageFftBandpassOp, ImageFftPowerSpectrumOp, ImageFindEdgesOp,
    ImageFindMaximaOp, ImageFlipOp, ImageHyperstackReduceDimensionalityOp, ImageHyperstackSubsetOp,
    ImageHyperstackToStackOp, ImageMedianFilterOp, ImageRankFilter3dOp, ImageRankFilterOp,
    ImageRemoveNaNsOp, ImageRemoveOutliersOp, ImageResizeOp, ImageRotate90Op, ImageRotateOp,
    ImageScaleOp, ImageSetScaleOp, ImageShadowDemoOp, ImageShadowOp, ImageSharpenOp,
    ImageStackAddSliceOp, ImageStackDeleteSliceOp, ImageStackGroupedZProjectOp,
    ImageStackMontageOp, ImageStackMontageToStackOp, ImageStackReduceOp, ImageStackResliceOp,
    ImageStackStatisticsOp, ImageStackSubstackOp, ImageStackToHyperstackOp, ImageStackZProfileOp,
    ImageStackZProjectOp, ImageSubtractBackgroundOp, ImageSurfacePlotOp, ImageSwapQuadrantsOp,
    ImageTranslateOp, ImageUnsharpMaskOp, IntensityEnhanceContrastOp, IntensityInvertOp,
    IntensityMathOp, IntensityNaNBackgroundOp, IntensityNormalizeOp, IntensityWindowOp,
    MeasurementsHistogramOp, MeasurementsProfileOp, MeasurementsSummaryOp,
    MorphologyBinaryMedianOp, MorphologyCloseOp, MorphologyDilateOp, MorphologyDistanceMapOp,
    MorphologyErodeOp, MorphologyFillHolesOp, MorphologyOpenOp, MorphologyOutlineOp,
    MorphologySkeletonizeOp, MorphologyUltimatePointsOp, MorphologyVoronoiOp,
    MorphologyWatershedOp, NoiseGaussianOp, NoiseSaltAndPepperOp, OpOutput, OpSchema, Operation,
    OpsError, Result, ThresholdFixedOp, ThresholdMakeBinaryOp, ThresholdOtsuOp,
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

/// Application-owned operation registry with runtime-safe identifiers.
///
/// Built-ins and future sandboxed plugin adapters cross the same interface.
/// Registration rejects collisions instead of silently replacing an existing
/// implementation.
#[derive(Clone, Default)]
pub struct OperationRegistry {
    operations: HashMap<String, Arc<dyn Operation>>,
}

impl std::fmt::Debug for OperationRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OperationRegistry")
            .field("registered_ops", &self.operations.len())
            .finish()
    }
}

impl OperationRegistry {
    pub fn register(&mut self, operation: Arc<dyn Operation>) -> Result<()> {
        let declared_name = operation.name();
        if declared_name.is_empty() || declared_name.trim() != declared_name {
            return Err(OpsError::InvalidParams(
                "operation identifiers must be non-empty and have no surrounding whitespace".into(),
            ));
        }
        let name = declared_name.to_string();
        if operation.schema().name != name {
            return Err(OpsError::InvalidParams(format!(
                "operation schema name must match `{name}`"
            )));
        }
        if self.operations.contains_key(&name) {
            return Err(OpsError::DuplicateOperation(name));
        }
        self.operations.insert(name, operation);
        Ok(())
    }

    pub fn list(&self) -> Vec<OpSchema> {
        let mut schemas = self
            .operations
            .values()
            .map(|operation| operation.schema())
            .collect::<Vec<_>>();
        schemas.sort_by(|left, right| left.name.cmp(&right.name));
        schemas
    }

    pub fn execute(&self, name: &str, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let operation = self
            .operations
            .get(name)
            .ok_or_else(|| OpsError::UnknownOperation(name.to_string()))?;
        operation.execute(dataset, params)
    }

    pub fn len(&self) -> usize {
        self.operations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

fn register<O: Operation + 'static>(registry: &mut OperationRegistry, operation: O) {
    registry
        .register(Arc::new(operation))
        .expect("built-in operation identifiers are unique");
}

fn registry() -> &'static OperationRegistry {
    static REGISTRY: OnceLock<OperationRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut map = OperationRegistry::default();
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
        register(&mut map, AnalyzeParticlesOp);
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

pub fn default_registry() -> OperationRegistry {
    registry().clone()
}

pub fn list_operations() -> Vec<OpSchema> {
    registry().list()
}

pub fn execute_operation(name: &str, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
    registry().execute(name, dataset, params)
}

pub fn execute_operation_with_registry(
    registry: &OperationRegistry,
    name: &str,
    dataset: &DatasetF32,
    params: &Value,
) -> Result<OpOutput> {
    registry.execute(name, dataset, params)
}

#[cfg(test)]
mod tests {
    use ndarray::Array;
    use serde_json::Value;

    use crate::model::{AxisKind, Dataset, Dim, Metadata, PixelType};

    use super::*;

    struct RuntimeNamedOperation {
        name: String,
    }

    impl Operation for RuntimeNamedOperation {
        fn name(&self) -> &str {
            &self.name
        }

        fn schema(&self) -> OpSchema {
            OpSchema {
                name: self.name.clone(),
                description: "runtime-owned test operation".into(),
                params: Vec::new(),
            }
        }

        fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
            Ok(OpOutput::dataset_only(dataset.clone()))
        }
    }

    fn dataset() -> DatasetF32 {
        Dataset::new(
            Array::from_shape_vec((1, 1), vec![7.0]).unwrap().into_dyn(),
            Metadata {
                dims: vec![Dim::new(AxisKind::Y, 1), Dim::new(AxisKind::X, 1)],
                pixel_type: PixelType::F32,
                ..Metadata::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn registry_accepts_runtime_owned_names_and_rejects_collisions() {
        let mut registry = OperationRegistry::default();
        let name = String::from("org.example.runtime.filter");
        registry
            .register(Arc::new(RuntimeNamedOperation { name: name.clone() }))
            .unwrap();
        let output = registry.execute(&name, &dataset(), &Value::Null).unwrap();
        assert_eq!(output.dataset.data[[0, 0]], 7.0);

        let error = registry
            .register(Arc::new(RuntimeNamedOperation { name: name.clone() }))
            .unwrap_err();
        assert!(matches!(error, OpsError::DuplicateOperation(id) if id == name));
    }
}
