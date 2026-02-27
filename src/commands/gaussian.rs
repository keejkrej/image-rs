use crate::model::{Dataset, DatasetF32};
use ndarray::{Array, IxDyn};
use rayon::prelude::*;
use serde_json::Value;

use super::{
    OpOutput, OpSchema, Operation, OpsError, ParamSpec, Result, get_optional_f32, spatial_axes,
    util::gaussian_kernel,
};

#[derive(Debug, Clone, Copy)]
pub struct GaussianBlurOp;

impl Operation for GaussianBlurOp {
    fn name(&self) -> &'static str {
        "gaussian.blur"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Gaussian blur across spatial axes (X/Y/Z).".to_string(),
            params: vec![ParamSpec {
                name: "sigma".to_string(),
                description: "Standard deviation for Gaussian kernel.".to_string(),
                required: false,
                kind: "float".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let sigma = get_optional_f32(params, "sigma", 1.0);
        if sigma < 0.0 {
            return Err(OpsError::InvalidParams("`sigma` must be >= 0".to_string()));
        }
        if sigma <= f32::EPSILON {
            return Ok(OpOutput::dataset_only(dataset.clone()));
        }
        let axes = spatial_axes(dataset);
        if axes.is_empty() {
            return Err(OpsError::UnsupportedLayout(
                "dataset has no spatial axes".to_string(),
            ));
        }

        let kernel = gaussian_kernel(sigma);
        let radius = (kernel.len() / 2) as isize;
        let shape = dataset.shape().to_vec();
        let mut current = dataset
            .data
            .as_slice_memory_order()
            .map(|slice| slice.to_vec())
            .unwrap_or_else(|| dataset.data.iter().copied().collect::<Vec<_>>());
        let strides = row_major_strides(&shape);

        for axis in axes {
            if shape[axis] <= 1 {
                continue;
            }
            current = blur_axis(&current, &shape, &strides, axis, &kernel, radius);
        }

        let output_data = Array::from_shape_vec(IxDyn(&shape), current).map_err(|_| {
            OpsError::UnsupportedLayout("failed to rebuild blurred output array".to_string())
        })?;
        let output_dataset = Dataset::new(output_data, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

fn blur_axis(
    input: &[f32],
    shape: &[usize],
    strides: &[usize],
    axis: usize,
    kernel: &[f32],
    radius: isize,
) -> Vec<f32> {
    let axis_len = shape[axis];
    let axis_stride = strides[axis];
    let lane_count = input.len() / axis_len;
    let lane_bases = (0..lane_count)
        .map(|lane| lane_base_offset(lane, axis, shape, strides))
        .collect::<Vec<_>>();

    let lane_results = lane_bases
        .par_iter()
        .map(|base| {
            let mut lane_output = vec![0.0_f32; axis_len];
            for (coord, output) in lane_output.iter_mut().enumerate() {
                let mut sum = 0.0_f32;
                for (kernel_index, weight) in kernel.iter().enumerate() {
                    let offset = kernel_index as isize - radius;
                    let candidate = coord as isize + offset;
                    let clamped = candidate.clamp(0, axis_len as isize - 1) as usize;
                    let index = *base + clamped * axis_stride;
                    sum += input[index] * *weight;
                }
                *output = sum;
            }
            lane_output
        })
        .collect::<Vec<_>>();

    let mut output = vec![0.0_f32; input.len()];
    for (lane, lane_output) in lane_results.into_iter().enumerate() {
        let base = lane_bases[lane];
        for (coord, value) in lane_output.into_iter().enumerate() {
            output[base + coord * axis_stride] = value;
        }
    }

    output
}

fn lane_base_offset(lane_index: usize, axis: usize, shape: &[usize], strides: &[usize]) -> usize {
    let mut remainder = lane_index;
    let mut base = 0usize;
    for dimension in 0..shape.len() {
        if dimension == axis {
            continue;
        }
        let size = shape[dimension];
        let coord = remainder % size;
        remainder /= size;
        base += coord * strides[dimension];
    }
    base
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    if shape.len() < 2 {
        return strides;
    }

    for index in (0..(shape.len() - 1)).rev() {
        strides[index] = strides[index + 1] * shape[index + 1];
    }
    strides
}

