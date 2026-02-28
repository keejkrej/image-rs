use crate::model::{Dataset, DatasetF32};
use morpholib_rs::{
    Connectivity2D, chamfer_distance_map, reconstruct_by_dilation, reconstruct_by_erosion,
};
use ndarray::{Array, IxDyn};
use serde_json::{Value, json};

use super::{
    MeasurementTable, OpOutput, OpSchema, Operation, OpsError, ParamSpec, Result,
    get_optional_bool, get_optional_f32, get_optional_usize,
};

fn ensure_2d_shape(dataset: &DatasetF32) -> Result<(usize, usize)> {
    if dataset.ndim() != 2 {
        return Err(OpsError::UnsupportedLayout(format!(
            "morpholibj operations currently support only 2D datasets; found {}D",
            dataset.ndim()
        )));
    }
    let shape = dataset.shape();
    Ok((shape[0], shape[1]))
}

fn connectivity_from_params(params: &Value) -> Result<Connectivity2D> {
    let value = get_optional_usize(params, "connectivity", 8);
    let value_u8 = u8::try_from(value)
        .map_err(|_| OpsError::InvalidParams(format!("unsupported connectivity: {value}")))?;
    Connectivity2D::from_u8(value_u8)
        .ok_or_else(|| OpsError::InvalidParams(format!("unsupported connectivity: {value}")))
}

#[derive(Debug, Clone, Copy)]
pub struct MorpholibjChamferDistanceOp;

impl Operation for MorpholibjChamferDistanceOp {
    fn name(&self) -> &'static str {
        "morpholibj.distance.chamfer"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Compute a 2D chamfer distance map from binary foreground.".to_string(),
            params: vec![
                ParamSpec {
                    name: "threshold".to_string(),
                    description: "Values > threshold are treated as foreground.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "connectivity".to_string(),
                    description: "Neighborhood connectivity: 4 or 8 (default 8).".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "normalize".to_string(),
                    description: "Normalize by first chamfer weight when true (default true)."
                        .to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let (height, width) = ensure_2d_shape(dataset)?;
        let threshold = get_optional_f32(params, "threshold", 0.5);
        let connectivity = connectivity_from_params(params)?;
        let normalize = get_optional_bool(params, "normalize", true);

        let foreground = dataset
            .data
            .iter()
            .map(|value| *value > threshold)
            .collect::<Vec<_>>();

        let distance = chamfer_distance_map(&foreground, width, height, connectivity, normalize)
            .map_err(|error| {
                OpsError::InvalidParams(format!("chamfer distance failed: {error}"))
            })?;

        let max_distance = distance
            .iter()
            .copied()
            .fold(0.0_f32, |acc, value| acc.max(value));

        let output = Array::from_shape_vec(IxDyn(dataset.shape()), distance)
            .expect("shape is unchanged and valid");
        let output_dataset = Dataset::new(output, dataset.metadata.clone())?;

        let mut measurements = MeasurementTable::default();
        measurements
            .values
            .insert("max_distance".to_string(), json!(max_distance));

        Ok(OpOutput {
            dataset: output_dataset,
            measurements: Some(measurements),
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MorpholibjReconstructByDilationOp;

impl Operation for MorpholibjReconstructByDilationOp {
    fn name(&self) -> &'static str {
        "morpholibj.reconstruct.by_dilation"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description:
                "2D grayscale reconstruction by dilation using marker=(image-h), mask=image."
                    .to_string(),
            params: vec![
                ParamSpec {
                    name: "h".to_string(),
                    description: "Marker offset; marker = image - h (default 1.0).".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "connectivity".to_string(),
                    description: "Neighborhood connectivity: 4 or 8 (default 8).".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let (height, width) = ensure_2d_shape(dataset)?;
        let h = get_optional_f32(params, "h", 1.0).max(0.0);
        let connectivity = connectivity_from_params(params)?;

        let mask = dataset.data.iter().copied().collect::<Vec<_>>();
        let marker = mask.iter().map(|value| *value - h).collect::<Vec<_>>();

        let reconstructed = reconstruct_by_dilation(&marker, &mask, width, height, connectivity)
            .map_err(|error| {
                OpsError::InvalidParams(format!("reconstruction by dilation failed: {error}"))
            })?;

        let output = Array::from_shape_vec(IxDyn(dataset.shape()), reconstructed)
            .expect("shape is unchanged and valid");
        let output_dataset = Dataset::new(output, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MorpholibjReconstructByErosionOp;

impl Operation for MorpholibjReconstructByErosionOp {
    fn name(&self) -> &'static str {
        "morpholibj.reconstruct.by_erosion"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description:
                "2D grayscale reconstruction by erosion using marker=(image+h), mask=image."
                    .to_string(),
            params: vec![
                ParamSpec {
                    name: "h".to_string(),
                    description: "Marker offset; marker = image + h (default 1.0).".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "connectivity".to_string(),
                    description: "Neighborhood connectivity: 4 or 8 (default 8).".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let (height, width) = ensure_2d_shape(dataset)?;
        let h = get_optional_f32(params, "h", 1.0).max(0.0);
        let connectivity = connectivity_from_params(params)?;

        let mask = dataset.data.iter().copied().collect::<Vec<_>>();
        let marker = mask.iter().map(|value| *value + h).collect::<Vec<_>>();

        let reconstructed = reconstruct_by_erosion(&marker, &mask, width, height, connectivity)
            .map_err(|error| {
                OpsError::InvalidParams(format!("reconstruction by erosion failed: {error}"))
            })?;

        let output = Array::from_shape_vec(IxDyn(dataset.shape()), reconstructed)
            .expect("shape is unchanged and valid");
        let output_dataset = Dataset::new(output, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{AxisKind, Dataset, Dim, Metadata, PixelType};
    use ndarray::{Array, IxDyn};
    use serde_json::json;

    use crate::commands::execute_operation;

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
    fn chamfer_distance_increases_toward_center() {
        let dataset = test_dataset(
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 1.0, 1.0, 0.0, //
                0.0, 1.0, 1.0, 1.0, 0.0, //
                0.0, 1.0, 1.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            (5, 5),
        );

        let output = execute_operation(
            "morpholibj.distance.chamfer",
            &dataset,
            &json!({"threshold": 0.5, "connectivity": 8, "normalize": true}),
        )
        .expect("distance");

        let center = output.dataset.data[IxDyn(&[2, 2])];
        let edge = output.dataset.data[IxDyn(&[1, 2])];
        assert!(center > edge);
    }

    #[test]
    fn reconstruction_by_dilation_preserves_peaks_with_h() {
        let dataset = test_dataset(
            vec![
                0.0, 0.0, 0.0, //
                0.0, 2.0, 0.0, //
                0.0, 0.0, 0.0,
            ],
            (3, 3),
        );

        let output = execute_operation(
            "morpholibj.reconstruct.by_dilation",
            &dataset,
            &json!({"h": 1.0, "connectivity": 4}),
        )
        .expect("reconstruction");

        assert!(output.dataset.data[IxDyn(&[1, 1])] >= 1.0);
    }
}
