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
                    description:
                        "Auto-threshold method: default, huang, intermodes, isodata, ij_isodata, li, max_entropy, mean, min_error, minimum, moments, otsu, percentile, renyi_entropy, shanbhag, triangle, yen, or fixed."
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
                255.0
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
        .to_string();
    let method_key = threshold_method_key(&method);
    let sixteen_bit_histogram = params
        .get("sixteen_bit")
        .or_else(|| params.get("16-bit"))
        .or_else(|| params.get("hist16"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
        && dataset.metadata.pixel_type == PixelType::U16;
    let threshold = match method_key.as_str() {
        "default" | "isodata" if sixteen_bit_histogram => {
            default_isodata_threshold_u16_raw(dataset)
        }
        "default" | "isodata" => default_isodata_threshold(dataset),
        "ijisodata" if sixteen_bit_histogram => ij_isodata_threshold_u16_raw(dataset),
        "ijisodata" => ij_isodata_threshold(dataset),
        "huang" if sixteen_bit_histogram => huang_threshold_u16_raw(dataset),
        "huang" => huang_threshold(dataset),
        "li" if sixteen_bit_histogram => li_threshold_u16_raw(dataset),
        "li" => li_threshold(dataset),
        "maxentropy" if sixteen_bit_histogram => max_entropy_threshold_u16_raw(dataset),
        "maxentropy" => max_entropy_threshold(dataset),
        "otsu" if sixteen_bit_histogram => otsu_threshold_u16_raw(dataset),
        "otsu" => finite_otsu_threshold(dataset),
        "mean" if sixteen_bit_histogram => mean_threshold_u16_raw(dataset),
        "mean" => mean_threshold(dataset),
        "percentile" if sixteen_bit_histogram => percentile_threshold_u16_raw(dataset, 0.5),
        "percentile" => percentile_threshold(dataset, 0.5),
        "triangle" if sixteen_bit_histogram => triangle_threshold_u16_raw(dataset),
        "triangle" => triangle_threshold(dataset),
        "minimum" if sixteen_bit_histogram => minimum_threshold_u16_raw(dataset),
        "minimum" => minimum_threshold(dataset),
        "intermodes" if sixteen_bit_histogram => intermodes_threshold_u16_raw(dataset),
        "intermodes" => intermodes_threshold(dataset),
        "minerror" if sixteen_bit_histogram => min_error_threshold_u16_raw(dataset),
        "minerror" => min_error_threshold(dataset),
        "moments" if sixteen_bit_histogram => moments_threshold_u16_raw(dataset),
        "moments" => moments_threshold(dataset),
        "renyientropy" if sixteen_bit_histogram => renyi_entropy_threshold_u16_raw(dataset),
        "renyientropy" => renyi_entropy_threshold(dataset),
        "shanbhag" if sixteen_bit_histogram => shanbhag_threshold_u16_raw(dataset),
        "shanbhag" => shanbhag_threshold(dataset),
        "yen" if sixteen_bit_histogram => yen_threshold_u16_raw(dataset),
        "yen" => yen_threshold(dataset),
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

fn threshold_method_key(method: &str) -> String {
    method
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
}

fn finite_values(dataset: &DatasetF32) -> Vec<f32> {
    dataset
        .data
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect()
}

fn imagej_auto_threshold_u16_raw(
    dataset: &DatasetF32,
    threshold_fn: impl FnOnce(&[u64]) -> Option<usize>,
) -> Option<usize> {
    let histogram = histogram_u16_raw(dataset);
    imagej_auto_threshold_bin_slice(&histogram, threshold_fn)
}

fn imagej_auto_threshold_bin_slice(
    histogram: &[u64],
    threshold_fn: impl FnOnce(&[u64]) -> Option<usize>,
) -> Option<usize> {
    if histogram.is_empty() {
        return None;
    }
    if let Some(threshold) = imagej_bilevel_bin(histogram) {
        return Some(threshold);
    }

    if histogram.len() <= 256 {
        return Some(threshold_fn(histogram).unwrap_or(0));
    }

    let min_bin = first_nonzero_bin(histogram)?;
    let max_bin = last_nonzero_bin(histogram)?;
    let cropped = &histogram[min_bin..=max_bin];
    let threshold = threshold_fn(cropped).unwrap_or(0);
    Some(min_bin + threshold)
}

fn imagej_bilevel_bin(histogram: &[u64]) -> Option<usize> {
    let mut first_nonzero = None;
    let mut second_nonzero = None;
    let mut nonzero_count = 0;
    for (index, count) in histogram.iter().enumerate() {
        if *count == 0 {
            continue;
        }
        nonzero_count += 1;
        if nonzero_count > 2 {
            return None;
        }
        if first_nonzero.is_none() {
            first_nonzero = Some(index);
        } else {
            second_nonzero = Some(index);
        }
    }

    match (nonzero_count, first_nonzero, second_nonzero) {
        (1, Some(first), _) if first > 0 => Some(first - 1),
        (2, _, Some(second)) if second > 0 => Some(second - 1),
        _ => None,
    }
}

fn finite_otsu_threshold(dataset: &DatasetF32) -> f32 {
    let values = finite_values(dataset);
    otsu_threshold(&values)
}

fn otsu_threshold_u16_raw(dataset: &DatasetF32) -> f32 {
    let Some(bin) = imagej_auto_threshold_u16_raw(dataset, otsu_bin_slice) else {
        return f32::NAN;
    };
    bin as f32
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

fn default_isodata_threshold_u16_raw(dataset: &DatasetF32) -> f32 {
    let Some(bin) = imagej_auto_threshold_u16_raw(dataset, default_isodata_bin_slice) else {
        return f32::NAN;
    };
    bin as f32
}

fn ij_isodata_threshold_u16_raw(dataset: &DatasetF32) -> f32 {
    let Some(bin) =
        imagej_auto_threshold_u16_raw(dataset, |histogram| Some(ij_isodata_bin_slice(histogram)))
    else {
        return f32::NAN;
    };
    bin as f32
}

fn mean_threshold(dataset: &DatasetF32) -> f32 {
    let histogram = histogram_256(dataset);
    let total = histogram.iter().sum::<u64>();
    if total == 0 {
        return f32::NAN;
    }
    let weighted_sum = histogram
        .iter()
        .enumerate()
        .map(|(index, count)| index as f64 * *count as f64)
        .sum::<f64>();
    threshold_bin_to_value(dataset, (weighted_sum / total as f64).floor() as usize)
}

fn mean_threshold_u16_raw(dataset: &DatasetF32) -> f32 {
    let Some(bin) = imagej_auto_threshold_u16_raw(dataset, mean_bin) else {
        return f32::NAN;
    };
    bin as f32
}

fn percentile_threshold(dataset: &DatasetF32, percentile: f64) -> f32 {
    let histogram = histogram_256(dataset);
    let Some(best_bin) = percentile_bin(&histogram, percentile) else {
        return f32::NAN;
    };
    threshold_bin_to_value(dataset, best_bin)
}

fn percentile_bin(histogram: &[u64], percentile: f64) -> Option<usize> {
    let total = histogram.iter().sum::<u64>();
    if total == 0 {
        return None;
    }
    let target = percentile.clamp(0.0, 1.0);
    let mut best_bin = 0;
    let mut best_distance = f64::INFINITY;
    let mut cumulative = 0_u64;
    for (index, count) in histogram.iter().enumerate() {
        cumulative += *count;
        let distance = ((cumulative as f64 / total as f64) - target).abs();
        if distance < best_distance {
            best_distance = distance;
            best_bin = index;
        }
    }
    Some(best_bin)
}

fn percentile_threshold_u16_raw(dataset: &DatasetF32, percentile: f64) -> f32 {
    let Some(bin) =
        imagej_auto_threshold_u16_raw(dataset, |histogram| percentile_bin(histogram, percentile))
    else {
        return f32::NAN;
    };
    bin as f32
}

fn triangle_threshold(dataset: &DatasetF32) -> f32 {
    let mut histogram = histogram_256(dataset);
    let Some(bin) = triangle_bin(&mut histogram) else {
        return f32::NAN;
    };
    threshold_bin_to_value(dataset, bin)
}

fn triangle_threshold_u16_raw(dataset: &DatasetF32) -> f32 {
    let Some(bin) = imagej_auto_threshold_u16_raw(dataset, |histogram| {
        let mut histogram = histogram.to_vec();
        triangle_bin(&mut histogram)
    }) else {
        return f32::NAN;
    };
    bin as f32
}

fn minimum_threshold(dataset: &DatasetF32) -> f32 {
    let histogram = histogram_256(dataset);
    let Some(bin) = minimum_bin(&histogram) else {
        return f32::NAN;
    };
    threshold_bin_to_value(dataset, bin)
}

fn minimum_threshold_u16_raw(dataset: &DatasetF32) -> f32 {
    let Some(bin) = imagej_auto_threshold_u16_raw(dataset, minimum_bin) else {
        return f32::NAN;
    };
    bin as f32
}

fn intermodes_threshold(dataset: &DatasetF32) -> f32 {
    let histogram = histogram_256(dataset);
    let Some(bin) = intermodes_bin(&histogram) else {
        return f32::NAN;
    };
    threshold_bin_to_value(dataset, bin)
}

fn intermodes_threshold_u16_raw(dataset: &DatasetF32) -> f32 {
    let Some(bin) = imagej_auto_threshold_u16_raw(dataset, intermodes_bin) else {
        return f32::NAN;
    };
    bin as f32
}

fn huang_threshold(dataset: &DatasetF32) -> f32 {
    let histogram = histogram_256(dataset);
    let Some(bin) = huang_bin(&histogram) else {
        return f32::NAN;
    };
    threshold_bin_to_value(dataset, bin)
}

fn huang_threshold_u16_raw(dataset: &DatasetF32) -> f32 {
    let Some(bin) = imagej_auto_threshold_u16_raw(dataset, huang_bin) else {
        return f32::NAN;
    };
    bin as f32
}

fn li_threshold(dataset: &DatasetF32) -> f32 {
    let histogram = histogram_256(dataset);
    let Some(bin) = li_bin(&histogram) else {
        return f32::NAN;
    };
    threshold_bin_to_value(dataset, bin)
}

fn li_threshold_u16_raw(dataset: &DatasetF32) -> f32 {
    let Some(bin) = imagej_auto_threshold_u16_raw(dataset, li_bin) else {
        return f32::NAN;
    };
    bin as f32
}

fn max_entropy_threshold(dataset: &DatasetF32) -> f32 {
    let histogram = histogram_256(dataset);
    let Some(bin) = max_entropy_bin(&histogram) else {
        return f32::NAN;
    };
    threshold_bin_to_value(dataset, bin)
}

fn max_entropy_threshold_u16_raw(dataset: &DatasetF32) -> f32 {
    let Some(bin) = imagej_auto_threshold_u16_raw(dataset, max_entropy_bin) else {
        return f32::NAN;
    };
    bin as f32
}

fn min_error_threshold(dataset: &DatasetF32) -> f32 {
    let histogram = histogram_256(dataset);
    let Some(bin) = min_error_bin(&histogram) else {
        return f32::NAN;
    };
    threshold_bin_to_value(dataset, bin)
}

fn min_error_threshold_u16_raw(dataset: &DatasetF32) -> f32 {
    let Some(bin) = imagej_auto_threshold_u16_raw(dataset, min_error_bin) else {
        return f32::NAN;
    };
    bin as f32
}

fn moments_threshold(dataset: &DatasetF32) -> f32 {
    let histogram = histogram_256(dataset);
    let Some(bin) = moments_bin(&histogram) else {
        return f32::NAN;
    };
    threshold_bin_to_value(dataset, bin)
}

fn moments_threshold_u16_raw(dataset: &DatasetF32) -> f32 {
    let Some(bin) = imagej_auto_threshold_u16_raw(dataset, moments_bin) else {
        return f32::NAN;
    };
    bin as f32
}

fn renyi_entropy_threshold(dataset: &DatasetF32) -> f32 {
    let histogram = histogram_256(dataset);
    let Some(bin) = renyi_entropy_bin(&histogram) else {
        return f32::NAN;
    };
    threshold_bin_to_value(dataset, bin)
}

fn renyi_entropy_threshold_u16_raw(dataset: &DatasetF32) -> f32 {
    let Some(bin) = imagej_auto_threshold_u16_raw(dataset, renyi_entropy_bin) else {
        return f32::NAN;
    };
    bin as f32
}

fn shanbhag_threshold(dataset: &DatasetF32) -> f32 {
    let histogram = histogram_256(dataset);
    let Some(bin) = shanbhag_bin(&histogram) else {
        return f32::NAN;
    };
    threshold_bin_to_value(dataset, bin)
}

fn shanbhag_threshold_u16_raw(dataset: &DatasetF32) -> f32 {
    let Some(bin) = imagej_auto_threshold_u16_raw(dataset, shanbhag_bin) else {
        return f32::NAN;
    };
    bin as f32
}

fn yen_threshold(dataset: &DatasetF32) -> f32 {
    let histogram = histogram_256(dataset);
    let Some(bin) = yen_bin(&histogram) else {
        return f32::NAN;
    };
    threshold_bin_to_value(dataset, bin)
}

fn yen_threshold_u16_raw(dataset: &DatasetF32) -> f32 {
    let Some(bin) = imagej_auto_threshold_u16_raw(dataset, yen_bin) else {
        return f32::NAN;
    };
    bin as f32
}

#[derive(Debug, Clone, Copy)]
enum SmoothEndpoint {
    ImageJIntermodes,
    ZeroOutside,
}

fn smooth_until_bimodal(histogram: &[u64], endpoint: SmoothEndpoint) -> Option<Vec<f64>> {
    if histogram.len() < 2 {
        return Some(histogram.iter().map(|count| *count as f64).collect());
    }
    let mut values = histogram
        .iter()
        .map(|count| *count as f64)
        .collect::<Vec<_>>();
    for _ in 0..10_000 {
        let maxima = local_maxima_f64(&values);
        if maxima.len() == 2 {
            return Some(values);
        }
        let mut smoothed = vec![0.0_f64; values.len()];
        let last = values.len() - 1;
        match endpoint {
            SmoothEndpoint::ImageJIntermodes => {
                let mut current = 0.0;
                let mut next = values[0];
                for index in 0..last {
                    let previous = current;
                    current = next;
                    next = values[index + 1];
                    smoothed[index] = (previous + current + next) / 3.0;
                }
                smoothed[last] = (current + next) / 3.0;
            }
            SmoothEndpoint::ZeroOutside => {
                for index in 1..last {
                    smoothed[index] = (values[index - 1] + values[index] + values[index + 1]) / 3.0;
                }
                smoothed[0] = (values[0] + values[1]) / 3.0;
                smoothed[last] = (values[last - 1] + values[last]) / 3.0;
            }
        }
        values = smoothed;
    }
    None
}

fn local_maxima_f64(histogram: &[f64]) -> Vec<usize> {
    if histogram.len() < 3 {
        return Vec::new();
    }
    (1..histogram.len() - 1)
        .filter(|index| {
            histogram[index - 1] < histogram[*index] && histogram[index + 1] < histogram[*index]
        })
        .collect()
}

fn histogram_total(histogram: &[u64]) -> u64 {
    histogram.iter().sum()
}

fn normalized_histogram(histogram: &[u64]) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let total = histogram_total(histogram);
    if total == 0 {
        return None;
    }
    let mut normalized = vec![0.0_f64; histogram.len()];
    let mut p1 = vec![0.0_f64; histogram.len()];
    let mut p2 = vec![0.0_f64; histogram.len()];
    for (index, count) in histogram.iter().enumerate() {
        normalized[index] = *count as f64 / total as f64;
        p1[index] = normalized[index] + if index == 0 { 0.0 } else { p1[index - 1] };
        p2[index] = 1.0 - p1[index];
    }
    Some((normalized, p1, p2))
}

fn first_nonzero_bin(histogram: &[u64]) -> Option<usize> {
    histogram.iter().position(|count| *count > 0)
}

fn last_nonzero_bin(histogram: &[u64]) -> Option<usize> {
    histogram.iter().rposition(|count| *count > 0)
}

fn first_positive_probability(p1: &[f64]) -> usize {
    p1.iter()
        .position(|value| value.abs() >= f64::EPSILON)
        .unwrap_or(0)
}

fn last_positive_tail_probability(p2: &[f64], first_bin: usize) -> usize {
    (first_bin..p2.len())
        .rev()
        .find(|index| p2[*index].abs() >= f64::EPSILON)
        .unwrap_or(first_bin)
}

fn minimum_bin(histogram: &[u64]) -> Option<usize> {
    let smoothed = smooth_until_bimodal(histogram, SmoothEndpoint::ZeroOutside)?;
    let max = histogram
        .iter()
        .rposition(|count| *count > 0)
        .unwrap_or(smoothed.len().saturating_sub(1));
    for index in 1..max {
        if smoothed[index - 1] > smoothed[index] && smoothed[index + 1] >= smoothed[index] {
            return Some(index);
        }
    }
    Some(0)
}

fn intermodes_bin(histogram: &[u64]) -> Option<usize> {
    let smoothed = smooth_until_bimodal(histogram, SmoothEndpoint::ImageJIntermodes)?;
    let maxima = local_maxima_f64(&smoothed);
    if maxima.len() != 2 {
        return None;
    }
    Some((maxima[0] + maxima[1]) / 2)
}

fn huang_bin(histogram: &[u64]) -> Option<usize> {
    let len = histogram.len();
    let first = first_nonzero_bin(histogram)?;
    let last = last_nonzero_bin(histogram)?;
    if first >= last {
        return Some(first);
    }
    let term = 1.0 / (last - first) as f64;
    let mut mu_0 = vec![0.0_f64; len];
    let mut sum_pix = 0.0_f64;
    let mut num_pix = 0.0_f64;
    for index in first..len {
        sum_pix += index as f64 * histogram[index] as f64;
        num_pix += histogram[index] as f64;
        if num_pix > 0.0 {
            mu_0[index] = sum_pix / num_pix;
        }
    }

    let mut mu_1 = vec![0.0_f64; len];
    sum_pix = 0.0;
    num_pix = 0.0;
    for index in (1..=last).rev() {
        sum_pix += index as f64 * histogram[index] as f64;
        num_pix += histogram[index] as f64;
        if num_pix > 0.0 {
            mu_1[index - 1] = sum_pix / num_pix;
        }
    }

    let mut threshold = first;
    let mut min_entropy = f64::MAX;
    for split in 0..len {
        let mut entropy = 0.0;
        for index in 0..=split {
            let mu_x = 1.0 / (1.0 + term * (index as f64 - mu_0[split]).abs());
            if (1.0e-6..=0.999999).contains(&mu_x) {
                entropy += histogram[index] as f64
                    * (-mu_x * mu_x.ln() - (1.0 - mu_x) * (1.0 - mu_x).ln());
            }
        }
        for index in (split + 1)..len {
            let mu_x = 1.0 / (1.0 + term * (index as f64 - mu_1[split]).abs());
            if (1.0e-6..=0.999999).contains(&mu_x) {
                entropy += histogram[index] as f64
                    * (-mu_x * mu_x.ln() - (1.0 - mu_x) * (1.0 - mu_x).ln());
            }
        }
        if entropy < min_entropy {
            min_entropy = entropy;
            threshold = split;
        }
    }
    Some(threshold)
}

fn li_bin(histogram: &[u64]) -> Option<usize> {
    let last_bin = histogram.len().checked_sub(1)?;
    let total = histogram_total(histogram);
    if total == 0 {
        return None;
    }
    let mean = histogram
        .iter()
        .enumerate()
        .map(|(index, count)| index as f64 * *count as f64)
        .sum::<f64>()
        / total as f64;
    let mut new_threshold = mean;
    let threshold = loop {
        let old_threshold = new_threshold;
        let threshold = (old_threshold + 0.5).floor().clamp(0.0, last_bin as f64) as usize;
        let mut sum_back = 0.0_f64;
        let mut num_back = 0.0_f64;
        for (index, count) in histogram.iter().enumerate().take(threshold + 1) {
            sum_back += index as f64 * *count as f64;
            num_back += *count as f64;
        }
        let mut sum_obj = 0.0_f64;
        let mut num_obj = 0.0_f64;
        for (index, count) in histogram.iter().enumerate().skip(threshold + 1) {
            sum_obj += index as f64 * *count as f64;
            num_obj += *count as f64;
        }
        let mean_back = if num_back == 0.0 {
            0.0
        } else {
            sum_back / num_back
        };
        let mean_obj = if num_obj == 0.0 {
            0.0
        } else {
            sum_obj / num_obj
        };
        let denominator = mean_back.ln() - mean_obj.ln();
        if denominator.abs() < f64::EPSILON {
            break threshold;
        }
        let temp = (mean_back - mean_obj) / denominator;
        if !temp.is_finite() {
            break threshold;
        }
        new_threshold = if temp < -f64::EPSILON {
            (temp - 0.5) as i64 as f64
        } else {
            (temp + 0.5) as i64 as f64
        };
        if (new_threshold - old_threshold).abs() <= 0.5 {
            break threshold;
        }
    };
    Some(threshold.min(last_bin))
}

fn max_entropy_bin(histogram: &[u64]) -> Option<usize> {
    let len = histogram.len();
    let (normalized, p1, p2) = normalized_histogram(histogram)?;
    let first = first_positive_probability(&p1);
    let last = last_positive_tail_probability(&p2, first);
    let mut threshold = first;
    let mut max_entropy = f64::MIN;
    for split in first..=last {
        if p1[split] <= 0.0 || p2[split] <= 0.0 {
            continue;
        }
        let mut background_entropy = 0.0;
        for index in 0..=split {
            if histogram[index] != 0 {
                let probability = normalized[index] / p1[split];
                background_entropy -= probability * probability.ln();
            }
        }
        let mut object_entropy = 0.0;
        for index in (split + 1)..len {
            if histogram[index] != 0 {
                let probability = normalized[index] / p2[split];
                object_entropy -= probability * probability.ln();
            }
        }
        let entropy = background_entropy + object_entropy;
        if max_entropy < entropy {
            max_entropy = entropy;
            threshold = split;
        }
    }
    Some(threshold)
}

fn min_error_bin(histogram: &[u64]) -> Option<usize> {
    let last_bin = histogram.len().checked_sub(1)?;
    let total = histogram_total(histogram) as f64;
    if total == 0.0 {
        return None;
    }
    let mut threshold = mean_bin(histogram)?;
    let mut previous = usize::MAX;
    while threshold != previous {
        let a_threshold = histogram_partial_count(histogram, threshold);
        let b_threshold = histogram_partial_weighted_sum(histogram, threshold);
        let c_threshold = histogram_partial_square_sum(histogram, threshold);
        let a_total = total;
        if a_threshold <= 0.0 || a_threshold >= a_total {
            break;
        }
        let mu = b_threshold / a_threshold;
        let nu = (histogram_partial_weighted_sum(histogram, last_bin) - b_threshold)
            / (a_total - a_threshold);
        let p = a_threshold / a_total;
        let q = (a_total - a_threshold) / a_total;
        let sigma2 = c_threshold / a_threshold - mu * mu;
        let tau2 = (histogram_partial_square_sum(histogram, last_bin) - c_threshold)
            / (a_total - a_threshold)
            - nu * nu;
        if sigma2 <= 0.0 || tau2 <= 0.0 {
            break;
        }
        let w0 = 1.0 / sigma2 - 1.0 / tau2;
        let w1 = mu / sigma2 - nu / tau2;
        let w2 =
            (mu * mu) / sigma2 - (nu * nu) / tau2 + ((sigma2 * q * q) / (tau2 * p * p)).log10();
        let square_term = w1 * w1 - w0 * w2;
        if square_term < 0.0 || w0.abs() <= f64::EPSILON {
            break;
        }
        previous = threshold;
        let temp = (w1 + square_term.sqrt()) / w0;
        if !temp.is_finite() {
            threshold = previous;
            break;
        }
        threshold = temp.floor().clamp(0.0, last_bin as f64) as usize;
    }
    Some(threshold.min(last_bin))
}

fn moments_bin(histogram: &[u64]) -> Option<usize> {
    let last_bin = histogram.len().checked_sub(1)?;
    let total = histogram_total(histogram) as f64;
    if total == 0.0 {
        return None;
    }
    let mut histo = vec![0.0_f64; histogram.len()];
    for (index, count) in histogram.iter().enumerate() {
        histo[index] = *count as f64 / total;
    }
    let mut m1 = 0.0;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for (index, probability) in histo.iter().enumerate() {
        let x = index as f64;
        m1 += x * probability;
        m2 += x * x * probability;
        m3 += x * x * x * probability;
    }
    let cd = m2 - m1 * m1;
    if cd.abs() <= f64::EPSILON {
        return Some(0);
    }
    let c0 = (-m2 * m2 + m1 * m3) / cd;
    let c1 = (-m3 + m2 * m1) / cd;
    let discriminant = c1 * c1 - 4.0 * c0;
    if discriminant < 0.0 {
        return Some(0);
    }
    let z0 = 0.5 * (-c1 - discriminant.sqrt());
    let z1 = 0.5 * (-c1 + discriminant.sqrt());
    if (z1 - z0).abs() <= f64::EPSILON {
        return Some(0);
    }
    let p0 = (z1 - m1) / (z1 - z0);
    let mut sum = 0.0;
    for (index, probability) in histo.iter().enumerate() {
        sum += probability;
        if sum > p0 {
            return Some(index);
        }
    }
    Some(last_bin)
}

fn renyi_entropy_bin(histogram: &[u64]) -> Option<usize> {
    let (normalized, p1, p2) = normalized_histogram(histogram)?;
    let first = first_positive_probability(&p1);
    let last = last_positive_tail_probability(&p2, first);

    let t_star2 = entropy_threshold(histogram, &normalized, &p1, &p2, first, last)?;
    let t_star1 = renyi_alpha_threshold(&normalized, &p1, &p2, first, last, 0.5)?;
    let t_star3 = renyi_alpha_threshold(&normalized, &p1, &p2, first, last, 2.0)?;

    let mut stars = [t_star1, t_star2, t_star3];
    stars.sort();
    let [t1, t2, t3] = stars;
    let (beta1, beta2, beta3) = if t2.abs_diff(t1) <= 5 {
        if t3.abs_diff(t2) <= 5 {
            (1.0, 2.0, 1.0)
        } else {
            (0.0, 1.0, 3.0)
        }
    } else if t3.abs_diff(t2) <= 5 {
        (3.0, 1.0, 0.0)
    } else {
        (1.0, 2.0, 1.0)
    };
    let omega = p1[t3] - p1[t1];
    let threshold = t1 as f64 * (p1[t1] + 0.25 * omega * beta1)
        + 0.25 * t2 as f64 * omega * beta2
        + t3 as f64 * (p2[t3] + 0.25 * omega * beta3);
    Some(threshold.clamp(0.0, histogram.len().saturating_sub(1) as f64) as usize)
}

fn entropy_threshold(
    histogram: &[u64],
    normalized: &[f64],
    p1: &[f64],
    p2: &[f64],
    first: usize,
    last: usize,
) -> Option<usize> {
    let len = histogram.len();
    let mut threshold = first;
    let mut max_entropy = 0.0;
    for split in first..=last {
        if p1[split] <= 0.0 || p2[split] <= 0.0 {
            continue;
        }
        let mut background_entropy = 0.0;
        for index in 0..=split {
            if histogram[index] != 0 {
                let probability = normalized[index] / p1[split];
                background_entropy -= probability * probability.ln();
            }
        }
        let mut object_entropy = 0.0;
        for index in (split + 1)..len {
            if histogram[index] != 0 {
                let probability = normalized[index] / p2[split];
                object_entropy -= probability * probability.ln();
            }
        }
        let entropy = background_entropy + object_entropy;
        if entropy > max_entropy {
            max_entropy = entropy;
            threshold = split;
        }
    }
    Some(threshold)
}

fn renyi_alpha_threshold(
    normalized: &[f64],
    p1: &[f64],
    p2: &[f64],
    first: usize,
    last: usize,
    alpha: f64,
) -> Option<usize> {
    let len = normalized.len();
    let term = 1.0 / (1.0 - alpha);
    let mut threshold = first;
    let mut max_entropy = 0.0;
    for split in first..=last {
        if p1[split] <= 0.0 || p2[split] <= 0.0 {
            continue;
        }
        let (mut background_entropy, mut object_entropy) = (0.0, 0.0);
        if (alpha - 0.5).abs() <= f64::EPSILON {
            for index in 0..=split {
                background_entropy += (normalized[index] / p1[split]).sqrt();
            }
            for index in (split + 1)..len {
                object_entropy += (normalized[index] / p2[split]).sqrt();
            }
        } else {
            for index in 0..=split {
                background_entropy +=
                    normalized[index] * normalized[index] / (p1[split] * p1[split]);
            }
            for index in (split + 1)..len {
                object_entropy += normalized[index] * normalized[index] / (p2[split] * p2[split]);
            }
        }
        let entropy = if background_entropy * object_entropy > 0.0 {
            term * (background_entropy * object_entropy).ln()
        } else {
            0.0
        };
        if entropy > max_entropy {
            max_entropy = entropy;
            threshold = split;
        }
    }
    Some(threshold)
}

fn shanbhag_bin(histogram: &[u64]) -> Option<usize> {
    let len = histogram.len();
    let (normalized, p1, p2) = normalized_histogram(histogram)?;
    let first = first_positive_probability(&p1);
    let last = last_positive_tail_probability(&p2, first);
    let mut threshold = first;
    let mut min_entropy = f64::MAX;
    for split in first..=last {
        if p1[split] <= 0.0 || p2[split] <= 0.0 {
            continue;
        }
        let mut background_entropy = 0.0;
        let term = 0.5 / p1[split];
        for index in 1..=split {
            let argument = 1.0 - term * p1[index - 1];
            if argument > 0.0 {
                background_entropy -= normalized[index] * argument.ln();
            }
        }
        background_entropy *= term;

        let mut object_entropy = 0.0;
        let term = 0.5 / p2[split];
        for index in (split + 1)..len {
            let argument = 1.0 - term * p2[index];
            if argument > 0.0 {
                object_entropy -= normalized[index] * argument.ln();
            }
        }
        object_entropy *= term;
        let entropy = (background_entropy - object_entropy).abs();
        if entropy < min_entropy {
            min_entropy = entropy;
            threshold = split;
        }
    }
    Some(threshold)
}

fn yen_bin(histogram: &[u64]) -> Option<usize> {
    let (normalized, p1, _) = normalized_histogram(histogram)?;
    let len = histogram.len();
    if len == 0 {
        return None;
    }
    let mut p1_sq = vec![0.0_f64; len];
    let mut p2_sq = vec![0.0_f64; len];
    for index in 0..len {
        let squared = normalized[index] * normalized[index];
        p1_sq[index] = squared + if index == 0 { 0.0 } else { p1_sq[index - 1] };
    }
    for index in (0..len.saturating_sub(1)).rev() {
        p2_sq[index] = p2_sq[index + 1] + normalized[index + 1] * normalized[index + 1];
    }
    let mut threshold = 0;
    let mut max_criterion = f64::MIN;
    for split in 0..len {
        let first = if p1_sq[split] * p2_sq[split] > 0.0 {
            -(p1_sq[split] * p2_sq[split]).ln()
        } else {
            0.0
        };
        let second = if p1[split] * (1.0 - p1[split]) > 0.0 {
            2.0 * (p1[split] * (1.0 - p1[split])).ln()
        } else {
            0.0
        };
        let criterion = first + second;
        if criterion > max_criterion {
            max_criterion = criterion;
            threshold = split;
        }
    }
    Some(threshold)
}

fn triangle_bin(histogram: &mut [u64]) -> Option<usize> {
    let last_bin = histogram.len().checked_sub(1)?;
    let mut min = first_nonzero_bin(histogram)?;
    if min > 0 {
        min -= 1;
    }
    let mut min2 = last_nonzero_bin(histogram)?;
    if min2 < last_bin {
        min2 += 1;
    }
    let (mut max, _) = histogram
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| *count)
        .unwrap_or((0, &0));
    let inverted = (max - min) < (min2 - max);
    if inverted {
        histogram.reverse();
        min = last_bin - min2;
        max = last_bin - max;
    }
    if min == max {
        return Some(min);
    }
    let mut nx = histogram[max] as f64;
    let mut ny = min as f64 - max as f64;
    let norm = (nx * nx + ny * ny).sqrt();
    if norm <= f64::EPSILON {
        if inverted {
            histogram.reverse();
        }
        return Some(max);
    }
    nx /= norm;
    ny /= norm;
    let d = nx * min as f64 + ny * histogram[min] as f64;
    let mut split = min;
    let mut split_distance = 0.0;
    for index in (min + 1)..=max {
        let distance = nx * index as f64 + ny * histogram[index] as f64 - d;
        if distance > split_distance {
            split = index;
            split_distance = distance;
        }
    }
    split = split.saturating_sub(1);
    if inverted {
        histogram.reverse();
        Some(last_bin - split)
    } else {
        Some(split)
    }
}

fn mean_bin(histogram: &[u64]) -> Option<usize> {
    let last_bin = histogram.len().checked_sub(1)?;
    let total = histogram_total(histogram);
    if total == 0 {
        return None;
    }
    let weighted_sum = histogram
        .iter()
        .enumerate()
        .map(|(index, count)| index as u64 * *count)
        .sum::<u64>();
    Some((weighted_sum / total).min(last_bin as u64) as usize)
}

fn histogram_partial_count(histogram: &[u64], index: usize) -> f64 {
    let last_bin = histogram.len().saturating_sub(1);
    histogram
        .iter()
        .take(index.min(last_bin) + 1)
        .map(|count| *count as f64)
        .sum()
}

fn histogram_partial_weighted_sum(histogram: &[u64], index: usize) -> f64 {
    let last_bin = histogram.len().saturating_sub(1);
    histogram
        .iter()
        .enumerate()
        .take(index.min(last_bin) + 1)
        .map(|(bin, count)| bin as f64 * *count as f64)
        .sum()
}

fn histogram_partial_square_sum(histogram: &[u64], index: usize) -> f64 {
    let last_bin = histogram.len().saturating_sub(1);
    histogram
        .iter()
        .enumerate()
        .take(index.min(last_bin) + 1)
        .map(|(bin, count)| bin as f64 * bin as f64 * *count as f64)
        .sum()
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

fn histogram_u16_raw(dataset: &DatasetF32) -> Vec<u64> {
    let mut histogram = vec![0_u64; 65_536];
    for value in dataset
        .data
        .iter()
        .copied()
        .filter(|value| value.is_finite())
    {
        let bin = value.round().clamp(0.0, 65_535.0) as usize;
        histogram[bin] += 1;
    }
    histogram
}

fn histogram_mode(histogram: &[u64; 256]) -> Option<usize> {
    histogram_mode_slice(histogram)
}

fn histogram_mode_slice(histogram: &[u64]) -> Option<usize> {
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

fn otsu_bin_slice(histogram: &[u64]) -> Option<usize> {
    let total = histogram.iter().sum::<u64>();
    if total == 0 {
        return None;
    }
    let weighted_sum = histogram
        .iter()
        .enumerate()
        .map(|(index, count)| index as f64 * *count as f64)
        .sum::<f64>();

    let mut sum_background = 0.0_f64;
    let mut weight_background = 0.0_f64;
    let mut best_variance = -1.0_f64;
    let mut best_threshold = 0usize;

    for (index, count) in histogram.iter().enumerate() {
        weight_background += *count as f64;
        if weight_background == 0.0 {
            continue;
        }
        let weight_foreground = total as f64 - weight_background;
        if weight_foreground == 0.0 {
            break;
        }

        sum_background += index as f64 * *count as f64;
        let mean_background = sum_background / weight_background;
        let mean_foreground = (weighted_sum - sum_background) / weight_foreground;
        let between =
            weight_background * weight_foreground * (mean_background - mean_foreground).powi(2);
        if between > best_variance {
            best_variance = between;
            best_threshold = index;
        }
    }

    Some(best_threshold)
}

fn default_isodata_bin_slice(histogram: &[u64]) -> Option<usize> {
    let mut histogram = histogram.to_vec();
    let mode = histogram_mode_slice(&histogram)?;
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
    Some(ij_isodata_bin_slice(&histogram))
}

fn ij_isodata_bin(histogram: &[u64; 256]) -> usize {
    ij_isodata_bin_slice(histogram)
}

fn ij_isodata_bin_slice(histogram: &[u64]) -> usize {
    let mut data = histogram.to_vec();
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
    result.round().clamp(0.0, max_value as f64) as usize
}

fn threshold_bin_to_value(dataset: &DatasetF32, bin: usize) -> f32 {
    let values = finite_values(dataset);
    if values.is_empty() {
        return f32::NAN;
    }
    let (min, max) = min_max(&values);
    min + (bin.min(255) as f32 / 255.0) * (max - min).max(f32::EPSILON)
}
