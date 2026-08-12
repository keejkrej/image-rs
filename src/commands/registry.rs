use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::model::DatasetF32;
use serde_json::Value;

use super::invocation::{datasets_identical, validate_plane_result_changes};

use super::{
    AnalyzeParticlesOp, AreaMaskSupport, ComponentsLabelOp, DatasetEffect, ExecutionControl,
    GaussianBlurOp, ImageBinOp, ImageCalibrateOp, ImageCanvasResizeOp, ImageColorThresholdOp,
    ImageConvertOp, ImageConvolveOp, ImageCoordinatesOp, ImageCropOp, ImageFftBandpassOp,
    ImageFftPowerSpectrumOp, ImageFindEdgesOp, ImageFindMaximaOp, ImageFlipOp,
    ImageHyperstackReduceDimensionalityOp, ImageHyperstackSubsetOp, ImageHyperstackToStackOp,
    ImageMedianFilterOp, ImageRankFilter3dOp, ImageRankFilterOp, ImageRemoveNaNsOp,
    ImageRemoveOutliersOp, ImageResizeOp, ImageRotate90Op, ImageRotateOp, ImageScaleOp,
    ImageSetScaleOp, ImageShadowDemoOp, ImageShadowOp, ImageSharpenOp, ImageStackAddSliceOp,
    ImageStackDeleteSliceOp, ImageStackGroupedZProjectOp, ImageStackMontageOp,
    ImageStackMontageToStackOp, ImageStackReduceOp, ImageStackResliceOp, ImageStackStatisticsOp,
    ImageStackSubstackOp, ImageStackToHyperstackOp, ImageStackZProfileOp, ImageStackZProjectOp,
    ImageSubtractBackgroundOp, ImageSurfacePlotOp, ImageSwapQuadrantsOp, ImageTranslateOp,
    ImageUnsharpMaskOp, IntensityEnhanceContrastOp, IntensityInvertOp, IntensityMathOp,
    IntensityNaNBackgroundOp, IntensityNormalizeOp, IntensityWindowOp, InvocationRequest,
    InvocationResult, MeasurementsHistogramOp, MeasurementsProfileOp, MeasurementsSummaryOp,
    MorphologyBinaryMedianOp, MorphologyCloseOp, MorphologyDilateOp, MorphologyDistanceMapOp,
    MorphologyErodeOp, MorphologyFillHolesOp, MorphologyOpenOp, MorphologyOutlineOp,
    MorphologySkeletonizeOp, MorphologyUltimatePointsOp, MorphologyVoronoiOp,
    MorphologyWatershedOp, NativePlaneAdapter, NoiseGaussianOp, NoiseSaltAndPepperOp, OpOutput,
    OpSchema, Operation, OperationDescriptor, OperationScope, OpsError, Result, ScopedOperation,
    ThresholdFixedOp, ThresholdMakeBinaryOp, ThresholdOtsuOp,
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
    operations: HashMap<String, RegisteredOperation>,
}

#[derive(Clone)]
enum RegisteredOperation {
    Dataset {
        operation: Arc<dyn Operation>,
        changes_dataset: bool,
    },
    Scoped {
        operation: Arc<dyn ScopedOperation>,
        legacy_dataset: Option<Arc<dyn Operation>>,
    },
}

impl RegisteredOperation {
    fn descriptor(&self) -> OperationDescriptor {
        match self {
            Self::Dataset { operation, .. } => {
                OperationDescriptor::whole_dataset(operation.schema())
            }
            Self::Scoped { operation, .. } => operation.descriptor(),
        }
    }
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
        self.register_dataset(operation, true)
    }

    /// Register a whole-dataset operation that promises not to change the input.
    pub fn register_read_only(&mut self, operation: Arc<dyn Operation>) -> Result<()> {
        self.register_dataset(operation, false)
    }

    fn register_dataset(
        &mut self,
        operation: Arc<dyn Operation>,
        changes_dataset: bool,
    ) -> Result<()> {
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
        self.operations.insert(
            name,
            RegisteredOperation::Dataset {
                operation,
                changes_dataset,
            },
        );
        Ok(())
    }

    pub(crate) fn register_scoped(&mut self, operation: Arc<dyn ScopedOperation>) -> Result<()> {
        self.register_scoped_with_legacy(operation, None)
    }

    fn register_native_plane(&mut self, operation: Arc<dyn Operation>) -> Result<()> {
        let adapter = Arc::new(NativePlaneAdapter::new(operation.clone()));
        self.register_scoped_with_legacy(adapter, Some(operation))
    }

    fn register_scoped_with_legacy(
        &mut self,
        operation: Arc<dyn ScopedOperation>,
        legacy_dataset: Option<Arc<dyn Operation>>,
    ) -> Result<()> {
        let declared_name = operation.name();
        if declared_name.is_empty() || declared_name.trim() != declared_name {
            return Err(OpsError::InvalidParams(
                "operation identifiers must be non-empty and have no surrounding whitespace".into(),
            ));
        }
        let name = declared_name.to_string();
        let descriptor = operation.descriptor();
        validate_descriptor(&name, &descriptor)?;
        if self.operations.contains_key(&name) {
            return Err(OpsError::DuplicateOperation(name));
        }
        self.operations.insert(
            name,
            RegisteredOperation::Scoped {
                operation,
                legacy_dataset,
            },
        );
        Ok(())
    }

    pub fn list(&self) -> Vec<OpSchema> {
        let mut schemas = self
            .operations
            .values()
            .map(|operation| operation.descriptor().schema)
            .collect::<Vec<_>>();
        schemas.sort_by(|left, right| left.name.cmp(&right.name));
        schemas
    }

    pub fn execute(&self, name: &str, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let registered = self
            .operations
            .get(name)
            .ok_or_else(|| OpsError::UnknownOperation(name.to_string()))?;
        if let RegisteredOperation::Scoped {
            legacy_dataset: Some(operation),
            ..
        } = registered
        {
            let output = operation.execute(dataset, params)?;
            output.dataset.validate()?;
            return Ok(output);
        }
        let descriptor = registered.descriptor();
        let scope = [
            OperationScope::AllPlanes,
            OperationScope::WholeDataset,
            OperationScope::ActivePlane,
            OperationScope::ZStack,
        ]
        .into_iter()
        .find(|scope| descriptor.supports(*scope))
        .ok_or_else(|| {
            OpsError::InvalidParams(format!(
                "operation `{name}` does not expose a legacy-compatible execution scope"
            ))
        })?;
        let input = Arc::new(dataset.clone());
        let result = self.invoke(
            InvocationRequest {
                operation: name.to_string(),
                input: input.clone(),
                parameters: params.clone(),
                scope,
                active: Default::default(),
                area_mask: None,
            },
            &ExecutionControl::default(),
        )?;
        let output_dataset = result.dataset_effect.dataset(&input).as_ref().clone();
        Ok(OpOutput {
            dataset: output_dataset,
            measurements: result.measurements,
            status: result.status,
        })
    }

    pub fn describe(&self, name: &str) -> Option<OperationDescriptor> {
        self.operations
            .get(name)
            .map(RegisteredOperation::descriptor)
    }

    pub fn invoke(
        &self,
        request: InvocationRequest,
        control: &ExecutionControl,
    ) -> Result<InvocationResult> {
        let registered = self
            .operations
            .get(&request.operation)
            .ok_or_else(|| OpsError::UnknownOperation(request.operation.clone()))?;
        request.input.validate()?;
        let descriptor = registered.descriptor();
        validate_invocation(&request, &descriptor)?;
        if control.is_cancelled() {
            return Err(OpsError::Cancelled);
        }

        let result = match registered {
            RegisteredOperation::Dataset {
                operation,
                changes_dataset,
            } => {
                let output = operation.execute(&request.input, &request.parameters)?;
                output.dataset.validate()?;
                if *changes_dataset && !datasets_identical(&request.input, &output.dataset) {
                    InvocationResult::replaced(
                        request.input.clone(),
                        output.dataset,
                        output.measurements,
                        output.status,
                    )
                } else {
                    if !*changes_dataset && !datasets_identical(&request.input, &output.dataset) {
                        return Err(OpsError::InvalidOperationOutput {
                            operation: request.operation.clone(),
                            message: "a read-only operation changed its dataset".into(),
                        });
                    }
                    InvocationResult::unchanged(output.measurements, output.status)
                }
            }
            RegisteredOperation::Scoped { operation, .. } => {
                let result = operation.invoke(&request, control)?;
                validate_result(&request, &result)?;
                normalize_result(&request, result)
            }
        };
        if control.is_cancelled() {
            return Err(OpsError::Cancelled);
        }
        Ok(result)
    }

    pub fn len(&self) -> usize {
        self.operations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

fn validate_descriptor(name: &str, descriptor: &OperationDescriptor) -> Result<()> {
    if descriptor.schema.name != name {
        return Err(OpsError::InvalidParams(format!(
            "operation schema name must match `{name}`"
        )));
    }
    if descriptor.scopes.is_empty() {
        return Err(OpsError::InvalidParams(format!(
            "operation `{name}` must support at least one invocation scope"
        )));
    }
    let mut scopes = std::collections::HashSet::new();
    if descriptor.scopes.iter().any(|scope| !scopes.insert(*scope)) {
        return Err(OpsError::InvalidParams(format!(
            "operation `{name}` repeats an invocation scope"
        )));
    }
    if descriptor.area_mask != AreaMaskSupport::Unsupported
        && (descriptor.scopes.contains(&OperationScope::WholeDataset)
            || !descriptor.scopes.iter().any(|scope| scope.is_plane_wise()))
    {
        return Err(OpsError::InvalidParams(format!(
            "operation `{name}` advertises an area mask outside exclusively plane-wise scopes"
        )));
    }
    Ok(())
}

fn validate_invocation(
    request: &InvocationRequest,
    descriptor: &OperationDescriptor,
) -> Result<()> {
    if !descriptor.supports(request.scope) {
        return Err(OpsError::UnsupportedScope {
            operation: request.operation.clone(),
            scope: request.scope,
        });
    }
    if !request.scope.is_plane_wise() && request.area_mask.is_some() {
        return Err(OpsError::InvalidAreaMask(
            "area masks are defined only for plane-wise invocation scopes".into(),
        ));
    }
    match (descriptor.area_mask, request.area_mask.as_ref()) {
        (AreaMaskSupport::Unsupported, Some(_)) => {
            return Err(OpsError::InvalidAreaMask(format!(
                "operation `{}` does not accept an area mask",
                request.operation
            )));
        }
        (AreaMaskSupport::Required, None) => {
            return Err(OpsError::InvalidAreaMask(format!(
                "operation `{}` requires an area mask",
                request.operation
            )));
        }
        _ => {}
    }
    if request.scope.is_plane_wise() {
        super::invocation::validate_plane_request(request)?;
    }
    Ok(())
}

fn validate_result(request: &InvocationRequest, result: &InvocationResult) -> Result<()> {
    match &result.dataset_effect {
        DatasetEffect::Unchanged => Ok(()),
        DatasetEffect::Replaced { before, after } => {
            if !Arc::ptr_eq(before, &request.input) {
                return Err(OpsError::InvalidOperationOutput {
                    operation: request.operation.clone(),
                    message: "replacement result does not reference the request input".into(),
                });
            }
            after.validate().map_err(OpsError::from)?;
            if request.scope.is_plane_wise() {
                validate_plane_result_layout(request, after)?;
            }
            Ok(())
        }
    }
}

fn normalize_result(request: &InvocationRequest, result: InvocationResult) -> InvocationResult {
    let InvocationResult {
        dataset_effect,
        measurements,
        status,
    } = result;
    match dataset_effect {
        DatasetEffect::Replaced { after, .. }
            if datasets_identical(request.input.as_ref(), after.as_ref()) =>
        {
            InvocationResult::unchanged(measurements, status)
        }
        dataset_effect => InvocationResult {
            dataset_effect,
            measurements,
            status,
        },
    }
}

fn validate_plane_result_layout(request: &InvocationRequest, after: &DatasetF32) -> Result<()> {
    let before = request.input.as_ref();
    let same_axes_and_sizes = before.metadata.dims.len() == after.metadata.dims.len()
        && before
            .metadata
            .dims
            .iter()
            .zip(&after.metadata.dims)
            .all(|(before, after)| before.axis == after.axis && before.size == after.size);
    if before.shape() != after.shape() || !same_axes_and_sizes {
        return Err(OpsError::InvalidOperationOutput {
            operation: request.operation.clone(),
            message: "plane-wise operation changed dataset axes or sizes".into(),
        });
    }
    if before.metadata.pixel_type != after.metadata.pixel_type {
        return Err(OpsError::InvalidOperationOutput {
            operation: request.operation.clone(),
            message: "plane-wise operation changed the host-owned pixel type".into(),
        });
    }
    if before.metadata.source != after.metadata.source {
        return Err(OpsError::InvalidOperationOutput {
            operation: request.operation.clone(),
            message: "plane-wise operation changed the host-owned source path".into(),
        });
    }
    validate_plane_result_changes(request, after)?;
    Ok(())
}

fn register<O: Operation + 'static>(registry: &mut OperationRegistry, operation: O) {
    registry
        .register(Arc::new(operation))
        .expect("built-in operation identifiers are unique");
}

fn register_read_only<O: Operation + 'static>(registry: &mut OperationRegistry, operation: O) {
    registry
        .register_read_only(Arc::new(operation))
        .expect("built-in operation identifiers are unique");
}

fn register_plane<O: Operation + 'static>(registry: &mut OperationRegistry, operation: O) {
    registry
        .register_native_plane(Arc::new(operation))
        .expect("built-in operation identifiers are unique");
}

fn registry() -> &'static OperationRegistry {
    static REGISTRY: OnceLock<OperationRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut map = OperationRegistry::default();
        register(&mut map, IntensityNormalizeOp);
        register(&mut map, IntensityEnhanceContrastOp);
        register_plane(&mut map, IntensityInvertOp);
        register_plane(&mut map, IntensityMathOp);
        register(&mut map, IntensityNaNBackgroundOp);
        register(&mut map, IntensityWindowOp);
        register_plane(&mut map, GaussianBlurOp);
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
        register_read_only(&mut map, ImageStackZProfileOp);
        register_read_only(&mut map, ImageStackStatisticsOp);
        register(&mut map, ImageBinOp);
        register(&mut map, ImageFlipOp);
        register(&mut map, ImageRotate90Op);
        register(&mut map, ImageRotateOp);
        register(&mut map, ImageTranslateOp);
        register_plane(&mut map, ImageRankFilterOp);
        register(&mut map, ImageRankFilter3dOp);
        register_plane(&mut map, ImageMedianFilterOp);
        register(&mut map, ImageRemoveNaNsOp);
        register(&mut map, ImageRemoveOutliersOp);
        register_plane(&mut map, ImageSharpenOp);
        register_plane(&mut map, ImageFindEdgesOp);
        register(&mut map, ImageFindMaximaOp);
        register(&mut map, ImageShadowOp);
        register(&mut map, ImageShadowDemoOp);
        register(&mut map, ImageSubtractBackgroundOp);
        register_plane(&mut map, ImageUnsharpMaskOp);
        register_plane(&mut map, ImageConvolveOp);
        register(&mut map, ImageSwapQuadrantsOp);
        register(&mut map, ImageFftPowerSpectrumOp);
        register(&mut map, ImageFftBandpassOp);
        register(&mut map, ImageSurfacePlotOp);
        register_plane(&mut map, ThresholdFixedOp);
        register(&mut map, ThresholdMakeBinaryOp);
        register(&mut map, ThresholdOtsuOp);
        register_read_only(&mut map, MeasurementsHistogramOp);
        register_read_only(&mut map, MeasurementsProfileOp);
        register_plane(&mut map, MorphologyErodeOp);
        register_plane(&mut map, MorphologyDilateOp);
        register_plane(&mut map, MorphologyOpenOp);
        register_plane(&mut map, MorphologyCloseOp);
        register_plane(&mut map, MorphologyBinaryMedianOp);
        // These distance-derived commands change the dataset's pixel representation to F32,
        // which cannot be expressed as an active-plane replacement in a single-typed dataset.
        register(&mut map, MorphologyDistanceMapOp);
        register(&mut map, MorphologyUltimatePointsOp);
        register_plane(&mut map, MorphologyWatershedOp);
        register(&mut map, MorphologyVoronoiOp);
        register_plane(&mut map, MorphologyFillHolesOp);
        register_plane(&mut map, MorphologyOutlineOp);
        register_plane(&mut map, MorphologySkeletonizeOp);
        register(&mut map, NoiseGaussianOp);
        register(&mut map, NoiseSaltAndPepperOp);
        register_read_only(&mut map, AnalyzeParticlesOp);
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
        register_read_only(&mut map, MeasurementsSummaryOp);
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
        let result = registry
            .invoke(
                InvocationRequest::whole_dataset(&name, Arc::new(dataset()), Value::Null),
                &ExecutionControl::default(),
            )
            .unwrap();
        assert!(matches!(result.dataset_effect, DatasetEffect::Unchanged));

        let error = registry
            .register(Arc::new(RuntimeNamedOperation { name: name.clone() }))
            .unwrap_err();
        assert!(matches!(error, OpsError::DuplicateOperation(id) if id == name));
    }
}
