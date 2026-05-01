use crate::model::{AxisKind, Dataset, DatasetF32, PixelType};
use ndarray::{Array, IxDyn};
use serde_json::Value;

use super::{OpOutput, OpSchema, Operation, OpsError, ParamSpec, Result, get_optional_f32};

#[derive(Debug, Clone, Copy)]
pub struct NoiseGaussianOp;

impl Operation for NoiseGaussianOp {
    fn name(&self) -> &'static str {
        "noise.gaussian"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Add ImageJ-style Gaussian noise to every pixel.".to_string(),
            params: vec![
                ParamSpec {
                    name: "sigma".to_string(),
                    description: "Gaussian standard deviation; ImageJ's Add Noise default is 25."
                        .to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "seed".to_string(),
                    description: "Optional deterministic random seed.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let sigma = get_optional_f32(params, "sigma", 25.0);
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(OpsError::InvalidParams(
                "`sigma` must be a finite non-negative value".to_string(),
            ));
        }
        let seed = params.get("seed").and_then(Value::as_u64).unwrap_or(0);
        Ok(OpOutput::dataset_only(add_gaussian_noise(
            dataset, sigma, seed,
        )?))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NoiseSaltAndPepperOp;

impl Operation for NoiseSaltAndPepperOp {
    fn name(&self) -> &'static str {
        "noise.salt_and_pepper"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Apply ImageJ-style salt-and-pepper noise to U8 X/Y planes.".to_string(),
            params: vec![
                ParamSpec {
                    name: "percent".to_string(),
                    description: "Fraction of pixels to affect; ImageJ default is 0.05."
                        .to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "seed".to_string(),
                    description: "Optional deterministic random seed.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        if dataset.metadata.pixel_type != PixelType::U8 {
            return Err(OpsError::UnsupportedLayout(
                "salt-and-pepper noise requires u8 pixel metadata".to_string(),
            ));
        }
        let percent = get_optional_f32(params, "percent", 0.05);
        if !(0.0..=1.0).contains(&percent) {
            return Err(OpsError::InvalidParams(
                "`percent` must be between 0 and 1".to_string(),
            ));
        }
        let seed = params.get("seed").and_then(Value::as_u64).unwrap_or(0);
        Ok(OpOutput::dataset_only(salt_and_pepper(
            dataset, percent, seed,
        )?))
    }
}

fn add_gaussian_noise(dataset: &DatasetF32, sigma: f32, seed: u64) -> Result<DatasetF32> {
    let mut rng = Lcg::new(seed);
    let output = match dataset.metadata.pixel_type {
        PixelType::U8 => add_integer_gaussian_noise(dataset, sigma, 255, &mut rng),
        PixelType::U16 => add_integer_gaussian_noise(dataset, sigma, 65_535, &mut rng),
        PixelType::F32 => dataset
            .data
            .iter()
            .map(|value| value + rng.next_gaussian() as f32 * sigma)
            .collect::<Vec<_>>(),
    };
    let data = Array::from_shape_vec(IxDyn(dataset.shape()), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build Gaussian noise output".to_string())
    })?;
    Ok(Dataset::new(data, dataset.metadata.clone())?)
}

fn add_integer_gaussian_noise(
    dataset: &DatasetF32,
    sigma: f32,
    sample_max: i32,
    rng: &mut Lcg,
) -> Vec<f32> {
    dataset
        .data
        .iter()
        .map(|value| {
            let raw = (value.clamp(0.0, 1.0) * sample_max as f32).round() as i32;
            if sigma == 0.0 {
                return raw as f32 / sample_max as f32;
            }
            let noisy = redraw_integer_noise(raw, sigma, sample_max, rng);
            noisy as f32 / sample_max as f32
        })
        .collect()
}

fn redraw_integer_noise(raw: i32, sigma: f32, sample_max: i32, rng: &mut Lcg) -> i32 {
    for _ in 0..10_000 {
        let offset = (rng.next_gaussian() as f32 * sigma).round() as i32;
        let noisy = raw + offset;
        if (0..=sample_max).contains(&noisy) {
            return noisy;
        }
    }
    (raw + (rng.next_gaussian() as f32 * sigma).round() as i32).clamp(0, sample_max)
}

fn salt_and_pepper(dataset: &DatasetF32, percent: f32, seed: u64) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let width = shape[x_axis];
    let height = shape[y_axis];
    let pixels_per_plane = width * height;
    let affected = (percent * pixels_per_plane as f32) as usize;
    let half = affected / 2;
    let mut output = dataset.data.iter().copied().collect::<Vec<_>>();
    let mut rng = Lcg::new(seed);

    let plane_shape = shape
        .iter()
        .enumerate()
        .filter_map(|(axis, size)| (axis != x_axis && axis != y_axis).then_some(*size))
        .collect::<Vec<_>>();
    iterate_indices(&plane_shape, |plane_coord| {
        let mut base_coord = vec![0usize; shape.len()];
        let mut plane_at = 0usize;
        for axis in 0..shape.len() {
            if axis != x_axis && axis != y_axis {
                base_coord[axis] = plane_coord[plane_at];
                plane_at += 1;
            }
        }
        for _ in 0..half {
            set_random_xy(
                &mut output,
                &shape,
                &base_coord,
                x_axis,
                y_axis,
                &mut rng,
                1.0,
            );
            set_random_xy(
                &mut output,
                &shape,
                &base_coord,
                x_axis,
                y_axis,
                &mut rng,
                0.0,
            );
        }
    });

    let data = Array::from_shape_vec(IxDyn(&shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build salt-and-pepper output".to_string())
    })?;
    Ok(Dataset::new(data, dataset.metadata.clone())?)
}

fn set_random_xy(
    output: &mut [f32],
    shape: &[usize],
    base_coord: &[usize],
    x_axis: usize,
    y_axis: usize,
    rng: &mut Lcg,
    value: f32,
) {
    let mut coord = base_coord.to_vec();
    coord[x_axis] = rng.next_usize(shape[x_axis]);
    coord[y_axis] = rng.next_usize(shape[y_axis]);
    let index = row_major_offset(shape, &coord);
    output[index] = value;
}

#[derive(Debug, Clone, Copy)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        (self.next_u32() as usize) % upper
    }

    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_unit_f64();
        let u2 = self.next_unit_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn next_unit_f64(&mut self) -> f64 {
        (self.next_u32() as f64 + 1.0) / (u32::MAX as f64 + 2.0)
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.state >> 32) as u32
    }
}

fn row_major_offset(shape: &[usize], coord: &[usize]) -> usize {
    let mut offset = 0usize;
    for axis in 0..shape.len() {
        offset *= shape[axis];
        offset += coord[axis];
    }
    offset
}

fn axis_index(dataset: &DatasetF32, axis: AxisKind) -> Result<usize> {
    dataset
        .axis_index(axis)
        .ok_or_else(|| OpsError::UnsupportedLayout(format!("dataset has no {axis:?} axis")))
}

fn iterate_indices(shape: &[usize], mut callback: impl FnMut(&[usize])) {
    if shape.is_empty() {
        callback(&[]);
        return;
    }
    let mut index = vec![0usize; shape.len()];
    loop {
        callback(&index);
        let mut axis = shape.len();
        while axis > 0 {
            axis -= 1;
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
            if axis == 0 {
                return;
            }
        }
    }
}
