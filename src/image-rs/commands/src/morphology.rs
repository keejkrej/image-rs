use image_model::{Dataset, DatasetF32};
use ndarray::{Dimension, IxDyn};
use serde_json::Value;

use crate::{
    OpOutput, OpSchema, Operation, OpsError, ParamSpec, Result, get_optional_usize, spatial_axes,
    util::neighborhood_offsets,
};

#[derive(Debug, Clone, Copy)]
enum MorphologyKind {
    Erode,
    Dilate,
}

fn apply_morphology(
    dataset: &DatasetF32,
    radius: usize,
    kind: MorphologyKind,
) -> Result<ndarray::ArrayD<f32>> {
    let axes = spatial_axes(dataset);
    if axes.is_empty() {
        return Err(OpsError::UnsupportedLayout(
            "dataset has no spatial axes".to_string(),
        ));
    }
    let offsets = neighborhood_offsets(axes.len(), radius, true);
    let shape = dataset.shape().to_vec();
    let mut output = dataset.data.clone();

    for (index, value) in output.indexed_iter_mut() {
        let mut aggregate = match kind {
            MorphologyKind::Erode => 1.0_f32,
            MorphologyKind::Dilate => 0.0_f32,
        };

        for offset in &offsets {
            let mut coord = index.slice().to_vec();
            for (offset_axis, data_axis) in axes.iter().enumerate() {
                let axis_size = shape[*data_axis] as isize;
                let candidate = coord[*data_axis] as isize + offset[offset_axis];
                let clamped = candidate.clamp(0, axis_size - 1) as usize;
                coord[*data_axis] = clamped;
            }
            let binary = if dataset.data[IxDyn(&coord)] > 0.5 {
                1.0_f32
            } else {
                0.0_f32
            };
            match kind {
                MorphologyKind::Erode => aggregate = aggregate.min(binary),
                MorphologyKind::Dilate => aggregate = aggregate.max(binary),
            }
        }
        *value = aggregate;
    }

    Ok(output)
}

#[derive(Debug, Clone, Copy)]
pub struct MorphologyErodeOp;

impl Operation for MorphologyErodeOp {
    fn name(&self) -> &'static str {
        "morphology.erode"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Binary erosion over spatial axes.".to_string(),
            params: vec![ParamSpec {
                name: "radius".to_string(),
                description: "Neighborhood radius.".to_string(),
                required: false,
                kind: "int".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let radius = get_optional_usize(params, "radius", 1);
        let eroded = apply_morphology(dataset, radius, MorphologyKind::Erode)?;
        let output_dataset = Dataset::new(eroded, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MorphologyDilateOp;

impl Operation for MorphologyDilateOp {
    fn name(&self) -> &'static str {
        "morphology.dilate"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Binary dilation over spatial axes.".to_string(),
            params: vec![ParamSpec {
                name: "radius".to_string(),
                description: "Neighborhood radius.".to_string(),
                required: false,
                kind: "int".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let radius = get_optional_usize(params, "radius", 1);
        let dilated = apply_morphology(dataset, radius, MorphologyKind::Dilate)?;
        let output_dataset = Dataset::new(dilated, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MorphologyOpenOp;

impl Operation for MorphologyOpenOp {
    fn name(&self) -> &'static str {
        "morphology.open"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Binary opening (erode then dilate).".to_string(),
            params: vec![ParamSpec {
                name: "radius".to_string(),
                description: "Neighborhood radius.".to_string(),
                required: false,
                kind: "int".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let radius = get_optional_usize(params, "radius", 1);
        let eroded = apply_morphology(dataset, radius, MorphologyKind::Erode)?;
        let intermediate = Dataset::new(eroded, dataset.metadata.clone())?;
        let opened = apply_morphology(&intermediate, radius, MorphologyKind::Dilate)?;
        let output_dataset = Dataset::new(opened, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MorphologyCloseOp;

impl Operation for MorphologyCloseOp {
    fn name(&self) -> &'static str {
        "morphology.close"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Binary closing (dilate then erode).".to_string(),
            params: vec![ParamSpec {
                name: "radius".to_string(),
                description: "Neighborhood radius.".to_string(),
                required: false,
                kind: "int".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let radius = get_optional_usize(params, "radius", 1);
        let dilated = apply_morphology(dataset, radius, MorphologyKind::Dilate)?;
        let intermediate = Dataset::new(dilated, dataset.metadata.clone())?;
        let closed = apply_morphology(&intermediate, radius, MorphologyKind::Erode)?;
        let output_dataset = Dataset::new(closed, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}
