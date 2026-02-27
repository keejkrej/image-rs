mod components;
mod gaussian;
mod intensity;
mod measurements;
#[cfg(feature = "morpholib")]
mod morpholibj;
mod morphology;
mod threshold;
#[cfg(feature = "thunderstorm")]
mod thunderstorm;
mod util;

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::model::{AxisKind, CoreError, DatasetF32};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub use components::ComponentsLabelOp;
pub use gaussian::GaussianBlurOp;
pub use intensity::{IntensityNormalizeOp, IntensityWindowOp};
pub use measurements::MeasurementsSummaryOp;
#[cfg(feature = "morpholib")]
pub use morpholibj::{
    MorpholibjChamferDistanceOp, MorpholibjReconstructByDilationOp,
    MorpholibjReconstructByErosionOp,
};
pub use morphology::{MorphologyCloseOp, MorphologyDilateOp, MorphologyErodeOp, MorphologyOpenOp};
pub use threshold::{ThresholdFixedOp, ThresholdOtsuOp};
#[cfg(feature = "thunderstorm")]
pub use thunderstorm::{
    ThunderstormGaussianFilterOp, ThunderstormLsqGaussianFitOp, ThunderstormNonMaxSuppressionOp,
    ThunderstormPipelineLocalizeOp,
};

pub type Result<T> = std::result::Result<T, OpsError>;

#[derive(Debug, Error)]
pub enum OpsError {
    #[error("unknown operation: {0}")]
    UnknownOperation(String),

    #[error("invalid operation parameters: {0}")]
    InvalidParams(String),

    #[error("unsupported dataset layout: {0}")]
    UnsupportedLayout(String),

    #[error("core dataset error: {0}")]
    Core(#[from] CoreError),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParamSpec {
    pub name: String,
    pub description: String,
    pub required: bool,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpSchema {
    pub name: String,
    pub description: String,
    pub params: Vec<ParamSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MeasurementTable {
    pub values: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone)]
pub struct OpOutput {
    pub dataset: DatasetF32,
    pub measurements: Option<MeasurementTable>,
}

impl OpOutput {
    pub fn dataset_only(dataset: DatasetF32) -> Self {
        Self {
            dataset,
            measurements: None,
        }
    }
}

pub trait Operation: Send + Sync {
    fn name(&self) -> &'static str;
    fn schema(&self) -> OpSchema;
    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput>;
}

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

pub fn default_registry() -> Registry {
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
    registry: &Registry,
    name: &str,
    dataset: &DatasetF32,
    params: &Value,
) -> Result<OpOutput> {
    let op = registry
        .get(name)
        .ok_or_else(|| OpsError::UnknownOperation(name.to_string()))?;
    op.execute(dataset, params)
}

pub(crate) fn get_required_f32(params: &Value, key: &str) -> Result<f32> {
    params
        .get(key)
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .ok_or_else(|| OpsError::InvalidParams(format!("missing float parameter `{key}`")))
}

pub(crate) fn get_optional_f32(params: &Value, key: &str, default: f32) -> f32 {
    params
        .get(key)
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(default)
}

pub(crate) fn get_optional_usize(params: &Value, key: &str, default: usize) -> usize {
    params
        .get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(default)
}

#[cfg(feature = "morpholib")]
pub(crate) fn get_optional_bool(params: &Value, key: &str, default: bool) -> bool {
    params.get(key).and_then(Value::as_bool).unwrap_or(default)
}

pub(crate) fn spatial_axes(dataset: &DatasetF32) -> Vec<usize> {
    dataset
        .metadata
        .dims
        .iter()
        .enumerate()
        .filter_map(|(index, dim)| match dim.axis {
            AxisKind::X | AxisKind::Y | AxisKind::Z | AxisKind::Unknown => Some(index),
            AxisKind::Channel | AxisKind::Time => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use ndarray::{Array, IxDyn};
    use serde_json::json;

    use crate::model::{AxisKind, Dataset, Dim, Metadata, PixelType};

    use super::{execute_operation, list_operations};

    fn test_dataset(values: Vec<f32>, shape: (usize, usize)) -> Dataset<f32> {
        let data = Array::from_shape_vec(shape, values)
            .expect("shape")
            .into_dyn();
        let metadata = Metadata {
            dims: vec![
                Dim::new(AxisKind::Y, shape.0),
                Dim::new(AxisKind::X, shape.1),
            ],
            pixel_type: PixelType::F32,
            ..Metadata::default()
        };
        Dataset::new(data, metadata).expect("dataset")
    }

    #[test]
    fn contains_required_operations() {
        let names = list_operations()
            .into_iter()
            .map(|schema| schema.name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"gaussian.blur".to_string()));
        assert!(names.contains(&"components.label".to_string()));
        assert!(names.contains(&"measurements.summary".to_string()));
        #[cfg(feature = "morpholib")]
        assert!(names.contains(&"morpholibj.distance.chamfer".to_string()));
        #[cfg(feature = "thunderstorm")]
        assert!(names.contains(&"thunderstorm.pipeline.localize".to_string()));
    }

    #[test]
    fn gaussian_blur_smooths_spike() {
        let dataset = test_dataset(vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], (3, 3));
        let output =
            execute_operation("gaussian.blur", &dataset, &json!({"sigma": 1.0})).expect("blur");
        let center = output.dataset.data[IxDyn(&[1, 1])];
        let corner = output.dataset.data[IxDyn(&[0, 0])];
        assert!(center < 1.0);
        assert!(center > corner);
    }

    #[test]
    fn connected_components_reports_regions() {
        let dataset = test_dataset(
            vec![
                1.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.0, 1.0, 1.0, //
            ],
            (3, 3),
        );
        let output =
            execute_operation("components.label", &dataset, &json!({})).expect("components");
        let count = output
            .measurements
            .as_ref()
            .and_then(|table| table.values.get("component_count"))
            .and_then(|value| value.as_u64())
            .expect("count");
        assert_eq!(count, 2);
    }

    #[test]
    fn window_operation_validates_bounds() {
        let dataset = test_dataset(vec![0.0, 0.5, 0.75, 1.0], (2, 2));
        let error = execute_operation(
            "intensity.window",
            &dataset,
            &json!({
                "low": 1.0,
                "high": 0.1
            }),
        )
        .expect_err("invalid bounds");
        assert!(error.to_string().contains("high"));
    }

    #[test]
    fn otsu_threshold_returns_measurement() {
        let dataset = test_dataset(vec![0.05, 0.1, 0.2, 0.8, 0.9, 0.95], (2, 3));
        let output = execute_operation("threshold.otsu", &dataset, &json!({})).expect("otsu");
        let threshold = output
            .measurements
            .as_ref()
            .and_then(|table| table.values.get("threshold"))
            .and_then(|value| value.as_f64())
            .expect("threshold");
        assert!(threshold > 0.0);
    }

    #[test]
    fn measurements_include_bbox() {
        let dataset = test_dataset(
            vec![
                0.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, //
            ],
            (3, 3),
        );
        let output =
            execute_operation("measurements.summary", &dataset, &json!({})).expect("measure");
        let table = output.measurements.expect("measurements");
        assert_eq!(
            table
                .values
                .get("area")
                .and_then(|value| value.as_u64())
                .expect("area"),
            2
        );
        assert!(table.values.contains_key("bbox_min"));
        assert!(table.values.contains_key("bbox_max"));
    }
}

