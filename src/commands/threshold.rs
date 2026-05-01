use crate::model::{Dataset, DatasetF32, PixelType};
use ndarray::{Array, IxDyn};
use rayon::prelude::*;
use serde_json::{Value, json};

use super::{
    MeasurementTable, OpOutput, OpSchema, Operation, OpsError, ParamSpec, Result, get_optional_f32,
    util::min_max,
};

#[derive(Debug, Clone, Copy)]
pub struct ThresholdFixedOp;

#[derive(Debug, Clone, Copy)]
pub struct ThresholdMakeBinaryOp;

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

impl Operation for ThresholdMakeBinaryOp {
    fn name(&self) -> &'static str {
        "threshold.make_binary"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Convert to an ImageJ-style binary mask.".to_string(),
            params: vec![
                ParamSpec {
                    name: "method".to_string(),
                    description: "Auto-threshold method: default, ij_isodata, otsu, or fixed."
                        .to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "background".to_string(),
                    description: "Foreground polarity: default, dark, or light.".to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "threshold".to_string(),
                    description: "Fixed threshold used when method is fixed.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "min".to_string(),
                    description: "Explicit lower threshold bound.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "max".to_string(),
                    description: "Explicit upper threshold bound.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let (lower, upper, threshold) = binary_threshold_range(dataset, params)?;
        let mut values = dataset.data.iter().copied().collect::<Vec<_>>();
        values.par_iter_mut().for_each(|value| {
            *value = if value.is_finite() && *value >= lower && *value <= upper {
                1.0
            } else {
                0.0
            };
        });

        let mask = Array::from_shape_vec(IxDyn(dataset.shape()), values)
            .expect("shape is unchanged and valid");
        let mut metadata = dataset.metadata.clone();
        metadata.pixel_type = PixelType::U8;
        metadata
            .extras
            .insert("threshold_min".to_string(), json!(lower));
        metadata
            .extras
            .insert("threshold_max".to_string(), json!(upper));
        let dataset = Dataset::new(mask, metadata)?;
        let mut measurements = MeasurementTable::default();
        measurements
            .values
            .insert("threshold".to_string(), json!(threshold));
        measurements
            .values
            .insert("threshold_min".to_string(), json!(lower));
        measurements
            .values
            .insert("threshold_max".to_string(), json!(upper));
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

fn binary_threshold_range(dataset: &DatasetF32, params: &Value) -> Result<(f32, f32, f32)> {
    if let (Some(lower), Some(upper)) = (
        params.get("min").and_then(Value::as_f64),
        params.get("max").and_then(Value::as_f64),
    ) {
        let lower = lower as f32;
        let upper = upper as f32;
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(OpsError::InvalidParams(
                "threshold min/max must be finite and ordered".to_string(),
            ));
        }
        return Ok((lower, upper, (lower + upper) * 0.5));
    }

    let method = params
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("default")
        .to_ascii_lowercase();
    let threshold = match method.as_str() {
        "default" | "isodata" => default_isodata_threshold(dataset),
        "ij_isodata" | "ij-isodata" => ij_isodata_threshold(dataset),
        "otsu" => finite_otsu_threshold(dataset),
        "fixed" => get_optional_f32(params, "threshold", 0.5),
        other => {
            return Err(OpsError::InvalidParams(format!(
                "unsupported threshold method `{other}`"
            )));
        }
    };

    if !threshold.is_finite() {
        return Err(OpsError::InvalidParams(
            "threshold could not be computed from finite pixels".to_string(),
        ));
    }

    let background = params
        .get("background")
        .and_then(Value::as_str)
        .unwrap_or("default")
        .to_ascii_lowercase();
    let values = finite_values(dataset);
    let (min, max) = min_max(&values);
    let mode = histogram_mode_value(&values);
    let foreground_is_high = match background.as_str() {
        "dark" => true,
        "light" => false,
        "default" => mode <= threshold,
        other => {
            return Err(OpsError::InvalidParams(format!(
                "unsupported threshold background `{other}`"
            )));
        }
    };
    if foreground_is_high {
        Ok((threshold, max, threshold))
    } else {
        Ok((min, threshold, threshold))
    }
}

fn finite_values(dataset: &DatasetF32) -> Vec<f32> {
    dataset
        .data
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect()
}

fn finite_otsu_threshold(dataset: &DatasetF32) -> f32 {
    let values = finite_values(dataset);
    otsu_threshold(&values)
}

fn default_isodata_threshold(dataset: &DatasetF32) -> f32 {
    let mut histogram = histogram_256(dataset);
    let Some(mode) = histogram_mode(&histogram) else {
        return f32::NAN;
    };
    let max_count = histogram[mode];
    let max_count2 = histogram
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != mode)
        .map(|(_, count)| *count)
        .max()
        .unwrap_or(0);
    if max_count > max_count2.saturating_mul(2) && max_count2 != 0 {
        histogram[mode] = ((max_count2 as f32) * 1.5) as u64;
    }
    threshold_bin_to_value(dataset, ij_isodata_bin(&histogram))
}

fn ij_isodata_threshold(dataset: &DatasetF32) -> f32 {
    let histogram = histogram_256(dataset);
    threshold_bin_to_value(dataset, ij_isodata_bin(&histogram))
}

fn histogram_256(dataset: &DatasetF32) -> [u64; 256] {
    let values = finite_values(dataset);
    let mut histogram = [0_u64; 256];
    if values.is_empty() {
        return histogram;
    }
    let (min, max) = min_max(&values);
    let span = (max - min).max(f32::EPSILON);
    for value in values {
        let normalized = ((value - min) / span).clamp(0.0, 1.0);
        histogram[(normalized * 255.0).round() as usize] += 1;
    }
    histogram
}

fn histogram_mode(histogram: &[u64; 256]) -> Option<usize> {
    let mut mode = 0;
    let mut max_count = 0;
    for (index, count) in histogram.iter().enumerate() {
        if *count > max_count {
            max_count = *count;
            mode = index;
        }
    }
    if max_count == 0 { None } else { Some(mode) }
}

fn histogram_mode_value(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    let (min, max) = min_max(values);
    let span = (max - min).max(f32::EPSILON);
    let mut histogram = [0_u64; 256];
    for value in values {
        let normalized = ((*value - min) / span).clamp(0.0, 1.0);
        histogram[(normalized * 255.0).round() as usize] += 1;
    }
    let mode = histogram_mode(&histogram).unwrap_or(0);
    min + (mode as f32 / 255.0) * span
}

fn ij_isodata_bin(histogram: &[u64; 256]) -> usize {
    let mut data = *histogram;
    let max_value = data.len() - 1;
    data[0] = 0;
    data[max_value] = 0;
    let min = data
        .iter()
        .position(|count| *count != 0)
        .unwrap_or(max_value);
    let max = data.iter().rposition(|count| *count != 0).unwrap_or(0);
    if min >= max {
        return data.len() / 2;
    }

    let mut moving_index = min;
    let result = loop {
        let (mut sum1, mut sum2, mut sum3, mut sum4) = (0.0, 0.0, 0.0, 0.0);
        for (index, count) in data.iter().enumerate().take(moving_index + 1).skip(min) {
            sum1 += index as f64 * *count as f64;
            sum2 += *count as f64;
        }
        for (index, count) in data.iter().enumerate().take(max + 1).skip(moving_index + 1) {
            sum3 += index as f64 * *count as f64;
            sum4 += *count as f64;
        }
        let result = (sum1 / sum2 + sum3 / sum4) / 2.0;
        moving_index += 1;
        if (moving_index + 1) as f64 > result || moving_index >= max - 1 {
            break result;
        }
    };
    result.round().clamp(0.0, 255.0) as usize
}

fn threshold_bin_to_value(dataset: &DatasetF32, bin: usize) -> f32 {
    let values = finite_values(dataset);
    if values.is_empty() {
        return f32::NAN;
    }
    let (min, max) = min_max(&values);
    min + (bin.min(255) as f32 / 255.0) * (max - min).max(f32::EPSILON)
}
