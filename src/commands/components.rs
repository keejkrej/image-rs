use std::collections::VecDeque;
use std::f64::consts::PI;

use crate::model::{AxisKind, Dataset, DatasetF32};
use ndarray::{Array, ArrayD, Dimension, IxDyn};
use serde_json::{Value, json};

use super::{
    MeasurementTable, OpOutput, OpSchema, Operation, OpsError, Result, spatial_axes,
    util::neighborhood_offsets,
};

#[derive(Debug, Clone, Copy)]
pub struct ComponentsLabelOp;

/// Measure connected foreground regions in a single X/Y image plane.
///
/// This intentionally leaves presentation concerns (outlines, masks, ROI Manager
/// integration) to the UI layer. The command returns one ImageJ-style result row
/// per accepted particle while preserving the source dataset.
#[derive(Debug, Clone, Copy)]
pub struct AnalyzeParticlesOp;

impl Operation for ComponentsLabelOp {
    fn name(&self) -> &'static str {
        "components.label"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Connected-component labeling on binary input.".to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        let axes = spatial_axes(dataset);
        if axes.is_empty() {
            return Err(OpsError::UnsupportedLayout(
                "dataset has no spatial axes".to_string(),
            ));
        }
        let offsets = neighborhood_offsets(axes.len(), 1, false);
        let mut labels = ArrayD::<u32>::zeros(IxDyn(dataset.shape()));
        let mut next_label = 1_u32;

        for (index, value) in dataset.data.indexed_iter() {
            if *value <= 0.5 || labels[index.clone()] != 0 {
                continue;
            }

            let mut queue = VecDeque::new();
            queue.push_back(index.slice().to_vec());
            labels[IxDyn(index.slice())] = next_label;

            while let Some(point) = queue.pop_front() {
                for offset in &offsets {
                    let mut candidate = point.clone();
                    let mut out_of_bounds = false;
                    for (offset_axis, data_axis) in axes.iter().enumerate() {
                        let size = dataset.shape()[*data_axis] as isize;
                        let next = candidate[*data_axis] as isize + offset[offset_axis];
                        if next < 0 || next >= size {
                            out_of_bounds = true;
                            break;
                        }
                        candidate[*data_axis] = next as usize;
                    }
                    if out_of_bounds {
                        continue;
                    }

                    let candidate_idx = IxDyn(&candidate);
                    if dataset.data[candidate_idx.clone()] <= 0.5
                        || labels[candidate_idx.clone()] != 0
                    {
                        continue;
                    }
                    labels[candidate_idx] = next_label;
                    queue.push_back(candidate);
                }
            }

            next_label += 1;
        }

        let labeled_values = labels.iter().map(|value| *value as f32).collect::<Vec<_>>();
        let labeled_array = Array::from_shape_vec(IxDyn(dataset.shape()), labeled_values)
            .expect("shape is unchanged and valid");
        let output_dataset = Dataset::new(labeled_array, dataset.metadata.clone())?;

        let mut measurements = MeasurementTable::default();
        measurements.values.insert(
            "component_count".to_string(),
            json!(next_label.saturating_sub(1)),
        );
        Ok(OpOutput {
            dataset: output_dataset,
            measurements: Some(measurements),
        })
    }
}

impl Operation for AnalyzeParticlesOp {
    fn name(&self) -> &'static str {
        "measurements.particles"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description:
                "Measure ImageJ-style connected particles in a thresholded or binary X/Y plane."
                    .to_string(),
            params: vec![
                super::ParamSpec {
                    name: "threshold".to_string(),
                    description:
                        "Inclusive foreground threshold; aliases `min_threshold` and defaults to 0.5."
                            .to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                super::ParamSpec {
                    name: "max_threshold".to_string(),
                    description: "Optional inclusive upper foreground threshold.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                super::ParamSpec {
                    name: "min_size".to_string(),
                    description: "Minimum calibrated particle area, inclusive.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                super::ParamSpec {
                    name: "max_size".to_string(),
                    description: "Optional maximum calibrated particle area, inclusive."
                        .to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                super::ParamSpec {
                    name: "min_circularity".to_string(),
                    description: "Minimum circularity in the ImageJ range 0..1, inclusive."
                        .to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                super::ParamSpec {
                    name: "max_circularity".to_string(),
                    description: "Maximum circularity in the ImageJ range 0..1, inclusive."
                        .to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                super::ParamSpec {
                    name: "connectivity".to_string(),
                    description: "Pixel connectivity: 4 or 8 (ImageJ default).".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                super::ParamSpec {
                    name: "exclude_edges".to_string(),
                    description: "Exclude particles touching an image edge.".to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let measurements = analyze_particles(dataset, params)?;
        Ok(OpOutput {
            dataset: dataset.clone(),
            measurements: Some(measurements),
        })
    }
}

#[derive(Debug)]
struct Particle {
    pixels: Vec<(usize, usize, f32)>,
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    touches_edge: bool,
}

#[derive(Debug, Clone, Copy)]
struct ParticleOptions {
    min_threshold: f32,
    max_threshold: f32,
    min_size: f64,
    max_size: f64,
    min_circularity: f64,
    max_circularity: f64,
    connectivity: usize,
    exclude_edges: bool,
}

#[derive(Debug, Clone, Copy)]
struct AxisCalibration {
    spacing: f64,
    origin: f64,
    direction: f64,
}

impl AxisCalibration {
    fn coordinate(self, pixel: f64) -> f64 {
        self.origin + self.direction * self.spacing * pixel
    }
}

fn analyze_particles(dataset: &DatasetF32, params: &Value) -> Result<MeasurementTable> {
    let (y_axis, x_axis) = xy_plane_axes(dataset)?;
    let options = particle_options(params)?;
    let width = dataset.shape()[x_axis];
    let height = dataset.shape()[y_axis];
    let x_calibration = axis_calibration(dataset, x_axis, "x")?;
    let y_calibration = axis_calibration(dataset, y_axis, "y")?;
    let area_scale = x_calibration.spacing * y_calibration.spacing;

    let values = xy_plane_values(dataset, y_axis, x_axis, height, width);
    let mut visited = vec![false; width * height];
    let mut rows = Vec::new();
    let mut total_area = 0.0_f64;
    let offsets: &[(isize, isize)] = if options.connectivity == 4 {
        &[(0, -1), (-1, 0), (1, 0), (0, 1)]
    } else {
        &[
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]
    };

    for y in 0..height {
        for x in 0..width {
            let flat = y * width + x;
            if visited[flat] || !is_foreground(values[flat], options) {
                continue;
            }

            let mut queue = VecDeque::from([(x, y)]);
            visited[flat] = true;
            let mut particle = Particle {
                pixels: Vec::new(),
                min_x: x,
                min_y: y,
                max_x: x,
                max_y: y,
                touches_edge: false,
            };

            while let Some((particle_x, particle_y)) = queue.pop_front() {
                let value = values[particle_y * width + particle_x];
                particle.pixels.push((particle_x, particle_y, value));
                particle.min_x = particle.min_x.min(particle_x);
                particle.min_y = particle.min_y.min(particle_y);
                particle.max_x = particle.max_x.max(particle_x);
                particle.max_y = particle.max_y.max(particle_y);
                particle.touches_edge |= particle_x == 0
                    || particle_y == 0
                    || particle_x + 1 == width
                    || particle_y + 1 == height;

                for (dx, dy) in offsets {
                    let candidate_x = particle_x as isize + dx;
                    let candidate_y = particle_y as isize + dy;
                    if candidate_x < 0
                        || candidate_y < 0
                        || candidate_x >= width as isize
                        || candidate_y >= height as isize
                    {
                        continue;
                    }
                    let candidate_x = candidate_x as usize;
                    let candidate_y = candidate_y as usize;
                    let candidate_flat = candidate_y * width + candidate_x;
                    if !visited[candidate_flat] && is_foreground(values[candidate_flat], options) {
                        visited[candidate_flat] = true;
                        queue.push_back((candidate_x, candidate_y));
                    }
                }
            }

            let pixel_count = particle.pixels.len();
            let area = pixel_count as f64 * area_scale;
            let perimeter = particle_perimeter(
                &particle,
                &values,
                width,
                height,
                options,
                x_calibration.spacing,
                y_calibration.spacing,
            );
            let circularity = if perimeter > 0.0 {
                (4.0 * PI * area / perimeter.powi(2)).min(1.0)
            } else {
                0.0
            };
            if (options.exclude_edges && particle.touches_edge)
                || area < options.min_size
                || area > options.max_size
                || circularity < options.min_circularity
                || circularity > options.max_circularity
            {
                continue;
            }

            total_area += area;
            rows.push(particle_row(
                rows.len() + 1,
                &particle,
                area,
                perimeter,
                circularity,
                x_calibration,
                y_calibration,
            ));
        }
    }

    let unit = common_xy_unit(dataset, x_axis, y_axis);
    let mut measurements = MeasurementTable::default();
    measurements
        .values
        .insert("particle_count".to_string(), json!(rows.len()));
    measurements
        .values
        .insert("total_area".to_string(), json!(total_area));
    measurements
        .values
        .insert("rows".to_string(), Value::Array(rows));
    measurements
        .values
        .insert("threshold_min".to_string(), json!(options.min_threshold));
    measurements.values.insert(
        "threshold_max".to_string(),
        if options.max_threshold.is_finite() {
            json!(options.max_threshold)
        } else {
            Value::Null
        },
    );
    measurements
        .values
        .insert("connectivity".to_string(), json!(options.connectivity));
    measurements
        .values
        .insert("area_unit".to_string(), json!(format!("{unit}^2")));
    measurements
        .values
        .insert("length_unit".to_string(), json!(unit));
    Ok(measurements)
}

fn xy_plane_axes(dataset: &DatasetF32) -> Result<(usize, usize)> {
    let y_axis = dataset
        .metadata
        .axis_index(AxisKind::Y)
        .ok_or_else(|| OpsError::UnsupportedLayout("particle analysis requires a Y axis".into()))?;
    let x_axis = dataset.metadata.axis_index(AxisKind::X).ok_or_else(|| {
        OpsError::UnsupportedLayout("particle analysis requires an X axis".into())
    })?;
    if dataset
        .shape()
        .iter()
        .enumerate()
        .any(|(axis, size)| axis != x_axis && axis != y_axis && *size != 1)
    {
        return Err(OpsError::UnsupportedLayout(
            "particle analysis accepts one X/Y plane; select a single channel, slice, and frame"
                .into(),
        ));
    }
    Ok((y_axis, x_axis))
}

fn particle_options(params: &Value) -> Result<ParticleOptions> {
    if !params.is_object() {
        return Err(OpsError::InvalidParams(
            "particle parameters must be a JSON object".into(),
        ));
    }
    let min_threshold = optional_finite_f32(params, "threshold")?
        .or(optional_finite_f32(params, "min_threshold")?)
        .unwrap_or(0.5);
    let max_threshold = optional_finite_f32(params, "max_threshold")?.unwrap_or(f32::INFINITY);
    if min_threshold > max_threshold {
        return Err(OpsError::InvalidParams(
            "`threshold`/`min_threshold` must be <= `max_threshold`".into(),
        ));
    }

    let min_size = optional_finite_f64(params, "min_size")?.unwrap_or(0.0);
    let max_size = optional_finite_f64(params, "max_size")?.unwrap_or(f64::MAX);
    if min_size < 0.0 || max_size < min_size {
        return Err(OpsError::InvalidParams(
            "particle size range must be non-negative and ordered".into(),
        ));
    }

    let min_circularity = optional_finite_f64(params, "min_circularity")?.unwrap_or(0.0);
    let max_circularity = optional_finite_f64(params, "max_circularity")?.unwrap_or(1.0);
    if !(0.0..=1.0).contains(&min_circularity)
        || !(0.0..=1.0).contains(&max_circularity)
        || min_circularity > max_circularity
    {
        return Err(OpsError::InvalidParams(
            "particle circularity range must be ordered within 0..=1".into(),
        ));
    }

    let connectivity = match params.get("connectivity") {
        None => 8,
        Some(value) => value.as_u64().ok_or_else(|| {
            OpsError::InvalidParams("`connectivity` must be the integer 4 or 8".into())
        })? as usize,
    };
    if !matches!(connectivity, 4 | 8) {
        return Err(OpsError::InvalidParams(
            "`connectivity` must be the integer 4 or 8".into(),
        ));
    }
    let exclude_edges = match params.get("exclude_edges") {
        None => false,
        Some(value) => value
            .as_bool()
            .ok_or_else(|| OpsError::InvalidParams("`exclude_edges` must be a boolean".into()))?,
    };

    Ok(ParticleOptions {
        min_threshold,
        max_threshold,
        min_size,
        max_size,
        min_circularity,
        max_circularity,
        connectivity,
        exclude_edges,
    })
}

fn optional_finite_f32(params: &Value, key: &str) -> Result<Option<f32>> {
    let Some(value) = params.get(key) else {
        return Ok(None);
    };
    let number = value
        .as_f64()
        .ok_or_else(|| OpsError::InvalidParams(format!("`{key}` must be a finite number")))?;
    if !number.is_finite() || number < f32::MIN as f64 || number > f32::MAX as f64 {
        return Err(OpsError::InvalidParams(format!(
            "`{key}` must be a finite number"
        )));
    }
    Ok(Some(number as f32))
}

fn optional_finite_f64(params: &Value, key: &str) -> Result<Option<f64>> {
    let Some(value) = params.get(key) else {
        return Ok(None);
    };
    let number = value
        .as_f64()
        .ok_or_else(|| OpsError::InvalidParams(format!("`{key}` must be a finite number")))?;
    if !number.is_finite() {
        return Err(OpsError::InvalidParams(format!(
            "`{key}` must be a finite number"
        )));
    }
    Ok(Some(number))
}

fn axis_calibration(dataset: &DatasetF32, axis: usize, label: &str) -> Result<AxisCalibration> {
    let spacing = dataset.metadata.dims[axis].spacing.unwrap_or(1.0) as f64;
    if !spacing.is_finite() || spacing <= 0.0 {
        return Err(OpsError::UnsupportedLayout(
            "particle analysis requires finite positive X/Y spacing".into(),
        ));
    }
    let origin_key = format!("{label}_origin_coordinate");
    let origin = match dataset.metadata.extras.get(&origin_key) {
        None => 0.0,
        Some(value) => value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| {
                OpsError::UnsupportedLayout(format!(
                    "particle analysis requires a finite `{origin_key}`"
                ))
            })?,
    };
    let inverted_key = format!("{label}_coordinate_inverted");
    let inverted = match dataset.metadata.extras.get(&inverted_key) {
        None => false,
        Some(value) => value.as_bool().ok_or_else(|| {
            OpsError::UnsupportedLayout(format!(
                "particle analysis requires `{inverted_key}` to be a boolean"
            ))
        })?,
    };
    Ok(AxisCalibration {
        spacing,
        origin,
        direction: if inverted { -1.0 } else { 1.0 },
    })
}

fn xy_plane_values(
    dataset: &DatasetF32,
    y_axis: usize,
    x_axis: usize,
    height: usize,
    width: usize,
) -> Vec<f32> {
    let mut index = vec![0; dataset.ndim()];
    let mut values = Vec::with_capacity(width * height);
    for y in 0..height {
        index[y_axis] = y;
        for x in 0..width {
            index[x_axis] = x;
            values.push(dataset.data[IxDyn(&index)]);
        }
    }
    values
}

fn is_foreground(value: f32, options: ParticleOptions) -> bool {
    value.is_finite() && value >= options.min_threshold && value <= options.max_threshold
}

fn particle_perimeter(
    particle: &Particle,
    values: &[f32],
    width: usize,
    height: usize,
    options: ParticleOptions,
    x_spacing: f64,
    y_spacing: f64,
) -> f64 {
    let mut perimeter = 0.0;
    for (x, y, _) in &particle.pixels {
        for (dx, dy, edge_length) in [
            (-1_isize, 0_isize, y_spacing),
            (1, 0, y_spacing),
            (0, -1, x_spacing),
            (0, 1, x_spacing),
        ] {
            let adjacent_x = *x as isize + dx;
            let adjacent_y = *y as isize + dy;
            if adjacent_x < 0
                || adjacent_y < 0
                || adjacent_x >= width as isize
                || adjacent_y >= height as isize
                || !is_foreground(
                    values[adjacent_y as usize * width + adjacent_x as usize],
                    options,
                )
            {
                perimeter += edge_length;
            }
        }
    }
    perimeter
}

fn particle_row(
    label: usize,
    particle: &Particle,
    area: f64,
    perimeter: f64,
    circularity: f64,
    x_calibration: AxisCalibration,
    y_calibration: AxisCalibration,
) -> Value {
    let pixel_count = particle.pixels.len();
    let mut sum = 0.0_f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut centroid_x = 0.0_f64;
    let mut centroid_y = 0.0_f64;
    for (x, y, value) in &particle.pixels {
        sum += f64::from(*value);
        min = min.min(*value);
        max = max.max(*value);
        centroid_x += x_calibration.coordinate(*x as f64 + 0.5);
        centroid_y += y_calibration.coordinate(*y as f64 + 0.5);
    }
    centroid_x /= pixel_count as f64;
    centroid_y /= pixel_count as f64;

    json!({
        "Label": label,
        "Area": area,
        "Mean": sum / pixel_count as f64,
        "Min": min,
        "Max": max,
        "X": centroid_x,
        "Y": centroid_y,
        "BX": x_calibration.coordinate(particle.min_x as f64),
        "BY": y_calibration.coordinate(particle.min_y as f64),
        "Width": (particle.max_x - particle.min_x + 1) as f64 * x_calibration.spacing,
        "Height": (particle.max_y - particle.min_y + 1) as f64 * y_calibration.spacing,
        "Perim.": perimeter,
        "Circ.": circularity,
        "Pixels": pixel_count,
    })
}

fn common_xy_unit(dataset: &DatasetF32, x_axis: usize, y_axis: usize) -> String {
    match (
        dataset.metadata.dims[x_axis].unit.as_deref(),
        dataset.metadata.dims[y_axis].unit.as_deref(),
    ) {
        (Some(x), Some(y)) if x == y && !x.is_empty() => x.to_string(),
        _ => "pixel".to_string(),
    }
}
