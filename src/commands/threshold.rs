use crate::model::{Dataset, DatasetF32};
use ndarray::{Array, IxDyn};
use rayon::prelude::*;
use serde_json::{Value, json};

use super::{
    MeasurementTable, OpOutput, OpSchema, Operation, ParamSpec, Result, get_optional_f32,
    util::min_max,
};

#[derive(Debug, Clone, Copy)]
pub struct ThresholdFixedOp;

impl Operation for ThresholdFixedOp {
    fn name(&self) -> &'static str {
        "threshold.fixed"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Binary threshold using a fixed value.".to_string(),
            params: vec![ParamSpec {
                name: "threshold".to_string(),
                description: "Values >= threshold become 1; otherwise 0.".to_string(),
                required: false,
                kind: "float".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let threshold = get_optional_f32(params, "threshold", 0.5);
        let mut values = dataset.data.iter().copied().collect::<Vec<_>>();
        values.par_iter_mut().for_each(|value| {
            *value = if *value >= threshold { 1.0 } else { 0.0 };
        });
        let binary = Array::from_shape_vec(IxDyn(dataset.shape()), values)
            .expect("shape is unchanged and valid");
        let output_dataset = Dataset::new(binary, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ThresholdOtsuOp;

impl Operation for ThresholdOtsuOp {
    fn name(&self) -> &'static str {
        "threshold.otsu"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Binary threshold using Otsu's method.".to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        let values = dataset.data.iter().copied().collect::<Vec<_>>();
        let threshold = otsu_threshold(&values);
        let mut binary = values;
        binary.par_iter_mut().for_each(|value| {
            *value = if *value >= threshold { 1.0 } else { 0.0 };
        });
        let output = Array::from_shape_vec(IxDyn(dataset.shape()), binary)
            .expect("shape is unchanged and valid");
        let dataset = Dataset::new(output, dataset.metadata.clone())?;
        let mut measurements = MeasurementTable::default();
        measurements
            .values
            .insert("threshold".to_string(), json!(threshold));
        Ok(OpOutput {
            dataset,
            measurements: Some(measurements),
        })
    }
}

fn otsu_threshold(values: &[f32]) -> f32 {
    let mut histogram = [0_u64; 256];
    let (min, max) = min_max(values);
    let span = (max - min).max(f32::EPSILON);

    for value in values {
        let normalized = ((*value - min) / span).clamp(0.0, 1.0);
        let bin = (normalized * 255.0).round() as usize;
        histogram[bin] += 1;
    }

    let total = values.len() as f64;
    let mut weighted_sum = 0.0_f64;
    for (index, count) in histogram.iter().enumerate() {
        weighted_sum += (index as f64) * (*count as f64);
    }

    let mut sum_background = 0.0_f64;
    let mut weight_background = 0.0_f64;
    let mut best_variance = -1.0_f64;
    let mut best_threshold = 0usize;

    for (index, count) in histogram.iter().enumerate() {
        weight_background += *count as f64;
        if weight_background == 0.0 {
            continue;
        }
        let weight_foreground = total - weight_background;
        if weight_foreground == 0.0 {
            break;
        }

        sum_background += (index as f64) * (*count as f64);
        let mean_background = sum_background / weight_background;
        let mean_foreground = (weighted_sum - sum_background) / weight_foreground;
        let between =
            weight_background * weight_foreground * (mean_background - mean_foreground).powi(2);
        if between > best_variance {
            best_variance = between;
            best_threshold = index;
        }
    }

    (best_threshold as f32 / 255.0) * span + min
}
