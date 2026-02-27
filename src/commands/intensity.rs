use crate::model::{Dataset, DatasetF32};
use ndarray::{Array, IxDyn};
use rayon::prelude::*;
use serde_json::Value;

use super::{OpOutput, OpSchema, Operation, OpsError, ParamSpec, Result, get_required_f32};

#[derive(Debug, Clone, Copy)]
pub struct IntensityNormalizeOp;

impl Operation for IntensityNormalizeOp {
    fn name(&self) -> &'static str {
        "intensity.normalize"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Normalize values into [0, 1] using optional min/max.".to_string(),
            params: vec![
                ParamSpec {
                    name: "min".to_string(),
                    description: "Optional lower bound. Uses dataset minimum when omitted."
                        .to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "max".to_string(),
                    description: "Optional upper bound. Uses dataset maximum when omitted."
                        .to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let (source_min, source_max) = dataset.min_max().unwrap_or((0.0, 1.0));
        let min = params
            .get("min")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(source_min);
        let max = params
            .get("max")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(source_max);
        let scale = if (max - min).abs() < f32::EPSILON {
            1.0
        } else {
            max - min
        };
        let mut values = dataset.data.iter().copied().collect::<Vec<_>>();
        values.par_iter_mut().for_each(|value| {
            *value = ((*value - min) / scale).clamp(0.0, 1.0);
        });
        let normalized = Array::from_shape_vec(IxDyn(dataset.shape()), values)
            .expect("shape is unchanged and valid");
        let output_dataset = Dataset::new(normalized, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IntensityWindowOp;

impl Operation for IntensityWindowOp {
    fn name(&self) -> &'static str {
        "intensity.window"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Apply a [low, high] window and clamp to [0, 1].".to_string(),
            params: vec![
                ParamSpec {
                    name: "low".to_string(),
                    description: "Lower window bound.".to_string(),
                    required: true,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "high".to_string(),
                    description: "Upper window bound.".to_string(),
                    required: true,
                    kind: "float".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let low = get_required_f32(params, "low")?;
        let high = get_required_f32(params, "high")?;
        if high <= low {
            return Err(OpsError::InvalidParams(
                "`high` must be greater than `low`".to_string(),
            ));
        }
        let mut values = dataset.data.iter().copied().collect::<Vec<_>>();
        values.par_iter_mut().for_each(|value| {
            *value = ((*value - low) / (high - low)).clamp(0.0, 1.0);
        });
        let windowed = Array::from_shape_vec(IxDyn(dataset.shape()), values)
            .expect("shape is unchanged and valid");
        let output_dataset = Dataset::new(windowed, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

