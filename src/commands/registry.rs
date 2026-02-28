use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::model::DatasetF32;
use serde_json::Value;

use super::{
    ComponentsLabelOp, GaussianBlurOp, IntensityNormalizeOp, IntensityWindowOp,
    MeasurementsSummaryOp, MorphologyCloseOp, MorphologyDilateOp, MorphologyErodeOp,
    MorphologyOpenOp, OpOutput, OpSchema, Operation, OpsError, Result, ThresholdFixedOp,
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
        register(&mut map, IntensityWindowOp);
        register(&mut map, GaussianBlurOp);
        register(&mut map, ThresholdFixedOp);
        register(&mut map, ThresholdOtsuOp);
        register(&mut map, MorphologyErodeOp);
        register(&mut map, MorphologyDilateOp);
        register(&mut map, MorphologyOpenOp);
        register(&mut map, MorphologyCloseOp);
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
