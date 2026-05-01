use crate::model::{AxisKind, DatasetF32};
use ndarray::IxDyn;
use serde_json::{Value, json};

use super::{MeasurementTable, OpOutput, OpSchema, Operation, OpsError, ParamSpec, Result};

#[derive(Debug, Clone, Copy)]
pub struct MeasurementsSummaryOp;

#[derive(Debug, Clone, Copy)]
pub struct MeasurementsHistogramOp;

#[derive(Debug, Clone, Copy)]
pub struct MeasurementsProfileOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageStackStatisticsOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageStackZProfileOp;

impl Operation for MeasurementsSummaryOp {
    fn name(&self) -> &'static str {
        "measurements.summary"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Compute area/volume, min/max/mean, centroid and bounding box."
                .to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0_f64;
        let mut count = 0usize;

        let rank = dataset.ndim();
        let mut nonzero_count = 0usize;
        let mut centroid_accumulator = vec![0.0_f64; rank];
        let mut bbox_min = vec![usize::MAX; rank];
        let mut bbox_max = vec![0usize; rank];

        for (index, value) in dataset.data.indexed_iter() {
            min = min.min(*value);
            max = max.max(*value);
            sum += f64::from(*value);
            count += 1;

            if *value > 0.0 {
                nonzero_count += 1;
                for axis in 0..rank {
                    let coordinate = index[axis];
                    centroid_accumulator[axis] += coordinate as f64;
                    bbox_min[axis] = bbox_min[axis].min(coordinate);
                    bbox_max[axis] = bbox_max[axis].max(coordinate);
                }
            }
        }

        let centroid = if nonzero_count > 0 {
            centroid_accumulator
                .iter()
                .map(|value| value / nonzero_count as f64)
                .collect::<Vec<_>>()
        } else {
            vec![0.0; rank]
        };

        let mut measurements = MeasurementTable::default();
        measurements.values.insert("min".to_string(), json!(min));
        measurements.values.insert("max".to_string(), json!(max));
        measurements
            .values
            .insert("mean".to_string(), json!(sum / count.max(1) as f64));
        measurements
            .values
            .insert("area".to_string(), json!(nonzero_count));
        measurements
            .values
            .insert("volume".to_string(), json!(nonzero_count));
        measurements
            .values
            .insert("centroid".to_string(), json!(centroid));
        measurements.values.insert(
            "bbox_min".to_string(),
            json!(
                bbox_min
                    .iter()
                    .map(|value| if *value == usize::MAX { 0 } else { *value })
                    .collect::<Vec<_>>()
            ),
        );
        measurements
            .values
            .insert("bbox_max".to_string(), json!(bbox_max));

        Ok(OpOutput {
            dataset: dataset.clone(),
            measurements: Some(measurements),
        })
    }
}

impl Operation for MeasurementsHistogramOp {
    fn name(&self) -> &'static str {
        "measurements.histogram"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Compute an ImageJ-style histogram for an image slice or stack."
                .to_string(),
            params: vec![
                ParamSpec {
                    name: "bins".to_string(),
                    description: "Number of histogram bins.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "min".to_string(),
                    description: "Optional histogram minimum value.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "max".to_string(),
                    description: "Optional histogram maximum value.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "stack".to_string(),
                    description: "When true, include all Z slices.".to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
                ParamSpec {
                    name: "z".to_string(),
                    description: "Zero-based Z slice for non-stack histograms.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let measurements = histogram(dataset, params)?;
        Ok(OpOutput {
            dataset: dataset.clone(),
            measurements: Some(measurements),
        })
    }
}

impl Operation for MeasurementsProfileOp {
    fn name(&self) -> &'static str {
        "measurements.profile"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Compute an ImageJ-style line or rectangular profile plot.".to_string(),
            params: vec![
                ParamSpec {
                    name: "left".to_string(),
                    description: "Rectangle ROI left coordinate in pixels.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "top".to_string(),
                    description: "Rectangle ROI top coordinate in pixels.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "width".to_string(),
                    description: "Rectangle ROI width in pixels.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "height".to_string(),
                    description: "Rectangle ROI height in pixels.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "x0".to_string(),
                    description: "Line ROI start X coordinate.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "y0".to_string(),
                    description: "Line ROI start Y coordinate.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "x1".to_string(),
                    description: "Line ROI end X coordinate.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "y1".to_string(),
                    description: "Line ROI end Y coordinate.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "vertical".to_string(),
                    description:
                        "For rectangular ROIs, average columns when false and rows when true."
                            .to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let measurements = profile(dataset, params)?;
        Ok(OpOutput {
            dataset: dataset.clone(),
            measurements: Some(measurements),
        })
    }
}

impl Operation for ImageStackZProfileOp {
    fn name(&self) -> &'static str {
        "image.stack.z_profile"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Measure the ImageJ-style mean gray-value profile along Z.".to_string(),
            params: vec![
                ParamSpec {
                    name: "left".to_string(),
                    description: "Optional ROI left coordinate in pixels.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "top".to_string(),
                    description: "Optional ROI top coordinate in pixels.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "width".to_string(),
                    description: "Optional ROI width in pixels.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "height".to_string(),
                    description: "Optional ROI height in pixels.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "min_threshold".to_string(),
                    description: "Optional inclusive lower threshold.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "max_threshold".to_string(),
                    description: "Optional inclusive upper threshold.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let measurements = z_axis_profile(dataset, params)?;
        Ok(OpOutput {
            dataset: dataset.clone(),
            measurements: Some(measurements),
        })
    }
}

impl Operation for ImageStackStatisticsOp {
    fn name(&self) -> &'static str {
        "image.stack.statistics"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Compute ImageJ-style stack statistics for the active Z stack."
                .to_string(),
            params: vec![
                ParamSpec {
                    name: "left".to_string(),
                    description: "Optional ROI left coordinate in pixels.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "top".to_string(),
                    description: "Optional ROI top coordinate in pixels.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "width".to_string(),
                    description: "Optional ROI width in pixels.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "height".to_string(),
                    description: "Optional ROI height in pixels.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "min_threshold".to_string(),
                    description: "Optional inclusive lower threshold.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "max_threshold".to_string(),
                    description: "Optional inclusive upper threshold.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let measurements = stack_statistics(dataset, params)?;
        Ok(OpOutput {
            dataset: dataset.clone(),
            measurements: Some(measurements),
        })
    }
}

fn z_axis_profile(dataset: &DatasetF32, params: &Value) -> Result<MeasurementTable> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let z_axis = axis_index(dataset, AxisKind::Z)?;
    let shape = dataset.shape();
    let slices = shape[z_axis];
    if slices < 2 {
        return Err(OpsError::InvalidParams(
            "Plot Z-axis Profile requires a stack".to_string(),
        ));
    }

    let left = optional_usize_param(params, "left")?.unwrap_or(0);
    let top = optional_usize_param(params, "top")?.unwrap_or(0);
    let width =
        optional_usize_param(params, "width")?.unwrap_or(shape[x_axis].saturating_sub(left));
    let height =
        optional_usize_param(params, "height")?.unwrap_or(shape[y_axis].saturating_sub(top));
    validate_roi(left, top, width, height, shape[x_axis], shape[y_axis])?;

    let min_threshold = optional_f32_param(params, "min_threshold")?;
    let max_threshold = optional_f32_param(params, "max_threshold")?;
    if let (Some(min), Some(max)) = (min_threshold, max_threshold) {
        if min > max {
            return Err(OpsError::InvalidParams(
                "`min_threshold` must be <= `max_threshold`".to_string(),
            ));
        }
    }

    let channel_axis = dataset.axis_index(AxisKind::Channel);
    let time_axis = dataset.axis_index(AxisKind::Time);
    let channel = fixed_axis_index(dataset, params, channel_axis, "channel")?;
    let time = fixed_axis_index(dataset, params, time_axis, "time")?;
    reject_unsupported_axes(
        dataset,
        &[
            Some(x_axis),
            Some(y_axis),
            Some(z_axis),
            channel_axis,
            time_axis,
        ],
    )?;

    let mut means = Vec::with_capacity(slices);
    let mut counts = Vec::with_capacity(slices);
    let mut positions = Vec::with_capacity(slices);
    let mut rows = Vec::with_capacity(slices);

    for z in 0..slices {
        let mut coord = vec![0usize; shape.len()];
        coord[z_axis] = z;
        if let Some((axis, index)) = channel {
            coord[axis] = index;
        }
        if let Some((axis, index)) = time {
            coord[axis] = index;
        }

        let mut sum = 0.0_f64;
        let mut count = 0usize;
        for y in top..top + height {
            coord[y_axis] = y;
            for x in left..left + width {
                coord[x_axis] = x;
                let value = dataset.data[IxDyn(&coord)];
                if min_threshold.is_some_and(|min| value < min)
                    || max_threshold.is_some_and(|max| value > max)
                {
                    continue;
                }
                sum += f64::from(value);
                count += 1;
            }
        }

        let mean = if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        };
        let position = calibrated_z_position(dataset, z_axis, z);
        means.push(mean);
        counts.push(count);
        positions.push(position);
        rows.push(json!({
            "Slice": z + 1,
            "Z": position,
            "Mean": mean,
            "Count": count
        }));
    }

    let mut measurements = MeasurementTable::default();
    measurements
        .values
        .insert("profile_axis".to_string(), json!("z"));
    measurements
        .values
        .insert("z_profile".to_string(), json!(means));
    measurements
        .values
        .insert("z_positions".to_string(), json!(positions));
    measurements
        .values
        .insert("counts".to_string(), json!(counts));
    measurements.values.insert("rows".to_string(), json!(rows));
    if let Some(unit) = &dataset.metadata.dims[z_axis].unit {
        measurements
            .values
            .insert("z_unit".to_string(), json!(unit));
    }
    Ok(measurements)
}

fn stack_statistics(dataset: &DatasetF32, params: &Value) -> Result<MeasurementTable> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let z_axis = axis_index(dataset, AxisKind::Z)?;
    let shape = dataset.shape();
    let slices = shape[z_axis];
    if slices < 2 {
        return Err(OpsError::InvalidParams(
            "Stack Statistics requires a stack".to_string(),
        ));
    }

    let left = optional_usize_param(params, "left")?.unwrap_or(0);
    let top = optional_usize_param(params, "top")?.unwrap_or(0);
    let width =
        optional_usize_param(params, "width")?.unwrap_or(shape[x_axis].saturating_sub(left));
    let height =
        optional_usize_param(params, "height")?.unwrap_or(shape[y_axis].saturating_sub(top));
    validate_roi(left, top, width, height, shape[x_axis], shape[y_axis])?;

    let min_threshold = optional_f32_param(params, "min_threshold")?;
    let max_threshold = optional_f32_param(params, "max_threshold")?;
    if let (Some(min), Some(max)) = (min_threshold, max_threshold) {
        if min > max {
            return Err(OpsError::InvalidParams(
                "`min_threshold` must be <= `max_threshold`".to_string(),
            ));
        }
    }

    let channel_axis = dataset.axis_index(AxisKind::Channel);
    let time_axis = dataset.axis_index(AxisKind::Time);
    let channel = fixed_axis_index(dataset, params, channel_axis, "channel")?;
    let time = fixed_axis_index(dataset, params, time_axis, "time")?;
    reject_unsupported_axes(
        dataset,
        &[
            Some(x_axis),
            Some(y_axis),
            Some(z_axis),
            channel_axis,
            time_axis,
        ],
    )?;

    let mut values = Vec::with_capacity(width * height * slices);
    let mut sum = 0.0_f64;
    let mut sum2 = 0.0_f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut coord = vec![0usize; shape.len()];
    if let Some((axis, index)) = channel {
        coord[axis] = index;
    }
    if let Some((axis, index)) = time {
        coord[axis] = index;
    }

    for z in 0..slices {
        coord[z_axis] = z;
        for y in top..top + height {
            coord[y_axis] = y;
            for x in left..left + width {
                coord[x_axis] = x;
                let value = dataset.data[IxDyn(&coord)];
                if !value.is_finite()
                    || min_threshold.is_some_and(|min| value < min)
                    || max_threshold.is_some_and(|max| value > max)
                {
                    continue;
                }
                min = min.min(value);
                max = max.max(value);
                sum += f64::from(value);
                sum2 += f64::from(value) * f64::from(value);
                values.push(value);
            }
        }
    }

    if values.is_empty() {
        return Err(OpsError::InvalidParams(
            "Stack Statistics found no finite pixels in range".to_string(),
        ));
    }

    let voxels = values.len();
    let stack_voxels = width * height * slices;
    let mean = sum / voxels as f64;
    let std_dev = sample_std_dev(voxels, sum, sum2);
    let median = median(&mut values);
    let mode = mode(&values);
    let voxel_volume = voxel_volume(dataset, x_axis, y_axis, z_axis);
    let volume = voxels as f64 * voxel_volume;
    let percent_volume = voxels as f64 * 100.0 / stack_voxels as f64;

    let mut row = serde_json::Map::new();
    row.insert("Voxels".to_string(), json!(voxels));
    row.insert("Volume".to_string(), json!(volume));
    row.insert("%Volume".to_string(), json!(percent_volume));
    row.insert("Mean".to_string(), json!(mean));
    row.insert("StdDev".to_string(), json!(std_dev));
    row.insert("Min".to_string(), json!(min));
    row.insert("Max".to_string(), json!(max));
    row.insert("Mode".to_string(), json!(mode));
    row.insert("Median".to_string(), json!(median));

    let mut measurements = MeasurementTable::default();
    measurements
        .values
        .insert("voxels".to_string(), json!(voxels));
    measurements
        .values
        .insert("volume".to_string(), json!(volume));
    measurements
        .values
        .insert("percent_volume".to_string(), json!(percent_volume));
    measurements.values.insert("mean".to_string(), json!(mean));
    measurements
        .values
        .insert("std_dev".to_string(), json!(std_dev));
    measurements.values.insert("min".to_string(), json!(min));
    measurements.values.insert("max".to_string(), json!(max));
    measurements.values.insert("mode".to_string(), json!(mode));
    measurements
        .values
        .insert("median".to_string(), json!(median));
    measurements.values.insert("rows".to_string(), json!([row]));
    if let Some(unit) = common_spatial_unit(dataset, x_axis, y_axis, z_axis) {
        measurements
            .values
            .insert("volume_unit".to_string(), json!(format!("{unit}^3")));
    }
    Ok(measurements)
}

fn histogram(dataset: &DatasetF32, params: &Value) -> Result<MeasurementTable> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let bins = optional_usize_param(params, "bins")?.unwrap_or(256);
    if bins == 0 {
        return Err(OpsError::InvalidParams(
            "histogram bins must be > 0".to_string(),
        ));
    }

    let include_stack = params
        .get("stack")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let z_axis = dataset.axis_index(AxisKind::Z);
    let channel_axis = dataset.axis_index(AxisKind::Channel);
    let time_axis = dataset.axis_index(AxisKind::Time);
    let z = if include_stack {
        None
    } else {
        fixed_axis_index(dataset, params, z_axis, "z")?
    };
    let channel = fixed_axis_index(dataset, params, channel_axis, "channel")?;
    let time = fixed_axis_index(dataset, params, time_axis, "time")?;
    reject_unsupported_axes(
        dataset,
        &[Some(x_axis), Some(y_axis), z_axis, channel_axis, time_axis],
    )?;

    let values = histogram_values(dataset, x_axis, y_axis, z_axis, z, channel, time);
    let (auto_min, auto_max) = min_max_finite(&values)?;
    let min = optional_f32_param(params, "min")?.unwrap_or(auto_min);
    let max = optional_f32_param(params, "max")?.unwrap_or(auto_max);
    if min > max {
        return Err(OpsError::InvalidParams(
            "`min` must be <= `max`".to_string(),
        ));
    }

    let mut counts = vec![0_u64; bins];
    let mut included = 0usize;
    let span = max - min;
    for value in values {
        if !value.is_finite() || value < min || value > max {
            continue;
        }
        let bin = if span == 0.0 {
            0
        } else {
            let scaled = ((value - min) / span) * bins as f32;
            (scaled.floor() as usize).min(bins - 1)
        };
        counts[bin] += 1;
        included += 1;
    }

    let bin_width = if bins == 0 {
        0.0
    } else {
        f64::from(max - min) / bins as f64
    };
    let rows = counts
        .iter()
        .enumerate()
        .map(|(index, count)| {
            let start = f64::from(min) + bin_width * index as f64;
            let end = if index + 1 == bins {
                f64::from(max)
            } else {
                start + bin_width
            };
            json!({
                "Bin": index,
                "Start": start,
                "End": end,
                "Count": count
            })
        })
        .collect::<Vec<_>>();

    let mut measurements = MeasurementTable::default();
    measurements.values.insert("bins".to_string(), json!(bins));
    measurements
        .values
        .insert("histogram".to_string(), json!(counts));
    measurements.values.insert("min".to_string(), json!(min));
    measurements.values.insert("max".to_string(), json!(max));
    measurements
        .values
        .insert("pixel_count".to_string(), json!(included));
    measurements
        .values
        .insert("stack_histogram".to_string(), json!(include_stack));
    measurements.values.insert("rows".to_string(), json!(rows));
    Ok(measurements)
}

fn profile(dataset: &DatasetF32, params: &Value) -> Result<MeasurementTable> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape();
    let z_axis = dataset.axis_index(AxisKind::Z);
    let channel_axis = dataset.axis_index(AxisKind::Channel);
    let time_axis = dataset.axis_index(AxisKind::Time);
    let z = fixed_axis_index(dataset, params, z_axis, "z")?;
    let channel = fixed_axis_index(dataset, params, channel_axis, "channel")?;
    let time = fixed_axis_index(dataset, params, time_axis, "time")?;
    reject_unsupported_axes(
        dataset,
        &[Some(x_axis), Some(y_axis), z_axis, channel_axis, time_axis],
    )?;

    let mut coord = vec![0usize; shape.len()];
    if let Some((axis, index)) = z {
        coord[axis] = index;
    }
    if let Some((axis, index)) = channel {
        coord[axis] = index;
    }
    if let Some((axis, index)) = time {
        coord[axis] = index;
    }

    let (profile_axis, values) = if let Some(line) = line_profile_params(params)? {
        (
            "line",
            line_profile(dataset, x_axis, y_axis, &mut coord, line),
        )
    } else if let Some(rect) = rect_profile_params(params, shape[x_axis], shape[y_axis])? {
        let vertical = params
            .get("vertical")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let axis = if vertical { "y" } else { "x" };
        (
            axis,
            rect_profile(dataset, x_axis, y_axis, &mut coord, rect, vertical),
        )
    } else {
        return Err(OpsError::InvalidParams(
            "Plot Profile requires a line or rectangular selection".to_string(),
        ));
    };

    let rows = values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            json!({
                "Distance": index,
                "Value": value
            })
        })
        .collect::<Vec<_>>();

    let mut measurements = MeasurementTable::default();
    measurements
        .values
        .insert("profile_axis".to_string(), json!(profile_axis));
    measurements
        .values
        .insert("profile".to_string(), json!(values));
    measurements
        .values
        .insert("sample_count".to_string(), json!(values.len()));
    measurements.values.insert("rows".to_string(), json!(rows));
    Ok(measurements)
}

fn rect_profile_params(
    params: &Value,
    image_width: usize,
    image_height: usize,
) -> Result<Option<(usize, usize, usize, usize)>> {
    let has_any_rect_param = ["left", "top", "width", "height"]
        .iter()
        .any(|key| params.get(*key).is_some_and(|value| !value.is_null()));
    if !has_any_rect_param {
        return Ok(None);
    }

    let left = optional_usize_param(params, "left")?.unwrap_or(0);
    let top = optional_usize_param(params, "top")?.unwrap_or(0);
    let width = optional_usize_param(params, "width")?.unwrap_or(image_width.saturating_sub(left));
    let height =
        optional_usize_param(params, "height")?.unwrap_or(image_height.saturating_sub(top));
    validate_roi(left, top, width, height, image_width, image_height)?;
    Ok(Some((left, top, width, height)))
}

fn line_profile_params(params: &Value) -> Result<Option<(f32, f32, f32, f32)>> {
    let values = ["x0", "y0", "x1", "y1"]
        .iter()
        .map(|key| optional_f32_param(params, key))
        .collect::<Result<Vec<_>>>()?;
    let has_any_line_param = values.iter().any(Option::is_some);
    if !has_any_line_param {
        return Ok(None);
    }
    let [x0, y0, x1, y1]: [Option<f32>; 4] = values
        .try_into()
        .map_err(|_| OpsError::InvalidParams("line profile parameters are invalid".to_string()))?;
    match (x0, y0, x1, y1) {
        (Some(x0), Some(y0), Some(x1), Some(y1)) => Ok(Some((x0, y0, x1, y1))),
        _ => Err(OpsError::InvalidParams(
            "line profiles require `x0`, `y0`, `x1` and `y1`".to_string(),
        )),
    }
}

fn rect_profile(
    dataset: &DatasetF32,
    x_axis: usize,
    y_axis: usize,
    coord: &mut [usize],
    rect: (usize, usize, usize, usize),
    vertical: bool,
) -> Vec<f64> {
    let (left, top, width, height) = rect;
    let mut values = Vec::with_capacity(if vertical { height } else { width });
    if vertical {
        for y in top..top + height {
            coord[y_axis] = y;
            let mut sum = 0.0_f64;
            for x in left..left + width {
                coord[x_axis] = x;
                sum += f64::from(dataset.data[IxDyn(coord)]);
            }
            values.push(sum / width as f64);
        }
    } else {
        for x in left..left + width {
            coord[x_axis] = x;
            let mut sum = 0.0_f64;
            for y in top..top + height {
                coord[y_axis] = y;
                sum += f64::from(dataset.data[IxDyn(coord)]);
            }
            values.push(sum / height as f64);
        }
    }
    values
}

fn line_profile(
    dataset: &DatasetF32,
    x_axis: usize,
    y_axis: usize,
    coord: &mut [usize],
    line: (f32, f32, f32, f32),
) -> Vec<f64> {
    let shape = dataset.shape();
    let (x0, y0, x1, y1) = line;
    let steps = ((x1 - x0).hypot(y1 - y0)).round().max(1.0) as usize;
    let mut values = Vec::with_capacity(steps + 1);
    for step in 0..=steps {
        let t = step as f32 / steps.max(1) as f32;
        let x = x0 + (x1 - x0) * t;
        let y = y0 + (y1 - y0) * t;
        coord[x_axis] = x.clamp(0.0, shape[x_axis].saturating_sub(1) as f32).floor() as usize;
        coord[y_axis] = y.clamp(0.0, shape[y_axis].saturating_sub(1) as f32).floor() as usize;
        values.push(f64::from(dataset.data[IxDyn(coord)]));
    }
    values
}

fn histogram_values(
    dataset: &DatasetF32,
    x_axis: usize,
    y_axis: usize,
    z_axis: Option<usize>,
    z: Option<(usize, usize)>,
    channel: Option<(usize, usize)>,
    time: Option<(usize, usize)>,
) -> Vec<f32> {
    let shape = dataset.shape();
    let z_range = if let Some((_, index)) = z {
        index..index + 1
    } else if let Some(axis) = z_axis {
        0..shape[axis]
    } else {
        0..1
    };

    let mut values = Vec::new();
    let mut coord = vec![0usize; shape.len()];
    if let Some((axis, index)) = channel {
        coord[axis] = index;
    }
    if let Some((axis, index)) = time {
        coord[axis] = index;
    }
    for z in z_range {
        if let Some(axis) = z_axis {
            coord[axis] = z;
        }
        for y in 0..shape[y_axis] {
            coord[y_axis] = y;
            for x in 0..shape[x_axis] {
                coord[x_axis] = x;
                values.push(dataset.data[IxDyn(&coord)]);
            }
        }
    }
    values
}

fn min_max_finite(values: &[f32]) -> Result<(f32, f32)> {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in values {
        if !value.is_finite() {
            continue;
        }
        min = min.min(*value);
        max = max.max(*value);
    }
    if min.is_finite() && max.is_finite() {
        Ok((min, max))
    } else {
        Err(OpsError::InvalidParams(
            "histogram found no finite pixels".to_string(),
        ))
    }
}

fn axis_index(dataset: &DatasetF32, axis: AxisKind) -> Result<usize> {
    dataset
        .axis_index(axis)
        .ok_or_else(|| OpsError::UnsupportedLayout(format!("dataset has no {axis:?} axis")))
}

fn optional_usize_param(params: &Value, key: &str) -> Result<Option<usize>> {
    let Some(value) = params.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_u64()
        .map(|value| Some(value as usize))
        .ok_or_else(|| OpsError::InvalidParams(format!("`{key}` must be a non-negative integer")))
}

fn optional_f32_param(params: &Value, key: &str) -> Result<Option<f32>> {
    let Some(value) = params.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let value = value
        .as_f64()
        .ok_or_else(|| OpsError::InvalidParams(format!("`{key}` must be a number")))?;
    if !value.is_finite() {
        return Err(OpsError::InvalidParams(format!("`{key}` must be finite")));
    }
    Ok(Some(value as f32))
}

fn validate_roi(
    left: usize,
    top: usize,
    width: usize,
    height: usize,
    image_width: usize,
    image_height: usize,
) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(OpsError::InvalidParams(
            "profile ROI width and height must be positive".to_string(),
        ));
    }
    if left >= image_width
        || top >= image_height
        || left + width > image_width
        || top + height > image_height
    {
        return Err(OpsError::InvalidParams(
            "profile ROI is outside image bounds".to_string(),
        ));
    }
    Ok(())
}

fn fixed_axis_index(
    dataset: &DatasetF32,
    params: &Value,
    axis: Option<usize>,
    key: &str,
) -> Result<Option<(usize, usize)>> {
    let Some(axis) = axis else {
        return Ok(None);
    };
    let index = optional_usize_param(params, key)?.unwrap_or(0);
    if index >= dataset.shape()[axis] {
        return Err(OpsError::InvalidParams(format!(
            "`{key}` index is outside axis bounds"
        )));
    }
    Ok(Some((axis, index)))
}

fn reject_unsupported_axes(dataset: &DatasetF32, supported: &[Option<usize>]) -> Result<()> {
    for (axis, dim) in dataset.metadata.dims.iter().enumerate() {
        if supported.iter().any(|supported| *supported == Some(axis)) {
            continue;
        }
        if !matches!(dim.axis, AxisKind::X | AxisKind::Y | AxisKind::Z) {
            return Err(OpsError::UnsupportedLayout(format!(
                "Plot Z-axis Profile does not support unfixed {:?} axes",
                dim.axis
            )));
        }
    }
    Ok(())
}

fn calibrated_z_position(dataset: &DatasetF32, z_axis: usize, z: usize) -> f64 {
    let dim = &dataset.metadata.dims[z_axis];
    let spacing = dim.spacing.unwrap_or(1.0) as f64;
    let Some(origin) = dataset
        .metadata
        .extras
        .get("z_origin_coordinate")
        .and_then(Value::as_f64)
    else {
        return (z + 1) as f64 * spacing;
    };
    let inverted = dataset
        .metadata
        .extras
        .get("z_coordinate_inverted")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let direction = if inverted { -1.0 } else { 1.0 };
    origin + direction * spacing * z as f64
}

fn sample_std_dev(n: usize, sum: f64, sum2: f64) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let variance = (n as f64 * sum2 - sum * sum) / n as f64;
    if variance > 0.0 {
        (variance / (n as f64 - 1.0)).sqrt()
    } else {
        0.0
    }
}

fn median(values: &mut [f32]) -> f64 {
    values.sort_by(|left, right| left.total_cmp(right));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (f64::from(values[middle - 1]) + f64::from(values[middle])) / 2.0
    } else {
        f64::from(values[middle])
    }
}

fn mode(values: &[f32]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let mut best_value = sorted[0];
    let mut best_count = 1usize;
    let mut current_value = sorted[0];
    let mut current_count = 1usize;
    for value in sorted.iter().copied().skip(1) {
        if value.to_bits() == current_value.to_bits() {
            current_count += 1;
        } else {
            if current_count > best_count {
                best_value = current_value;
                best_count = current_count;
            }
            current_value = value;
            current_count = 1;
        }
    }
    if current_count > best_count {
        best_value = current_value;
    }
    f64::from(best_value)
}

fn voxel_volume(dataset: &DatasetF32, x_axis: usize, y_axis: usize, z_axis: usize) -> f64 {
    [x_axis, y_axis, z_axis]
        .iter()
        .map(|axis| dataset.metadata.dims[*axis].spacing.unwrap_or(1.0) as f64)
        .product()
}

fn common_spatial_unit(
    dataset: &DatasetF32,
    x_axis: usize,
    y_axis: usize,
    z_axis: usize,
) -> Option<String> {
    let x_unit = dataset.metadata.dims[x_axis].unit.as_deref()?;
    let y_unit = dataset.metadata.dims[y_axis].unit.as_deref()?;
    let z_unit = dataset.metadata.dims[z_axis].unit.as_deref()?;
    if x_unit == y_unit && x_unit == z_unit && x_unit != "pixel" {
        Some(x_unit.to_string())
    } else {
        None
    }
}
