use image_model::{Dataset, DatasetF32, Metadata};
use ndarray::{Array, IxDyn};
use serde_json::{Value, json};

use crate::{MeasurementTable, OpOutput, OpSchema, Operation, OpsError, ParamSpec, Result};

fn map_ts_error(error: thunderstorm_rs::TsError) -> OpsError {
    match error {
        thunderstorm_rs::TsError::UnknownOperation(message) => OpsError::UnknownOperation(message),
        thunderstorm_rs::TsError::UnsupportedLayout(message) => {
            OpsError::UnsupportedLayout(message)
        }
        other => OpsError::InvalidParams(other.to_string()),
    }
}

fn dataset_to_frame(dataset: &DatasetF32) -> Result<thunderstorm_rs::Frame2D> {
    if dataset.ndim() != 2 {
        return Err(OpsError::UnsupportedLayout(format!(
            "thunderstorm operations currently support only 2D datasets; found {}D",
            dataset.ndim()
        )));
    }
    let shape = dataset.shape();
    let height = shape[0];
    let width = shape[1];
    thunderstorm_rs::Frame2D::new(width, height, dataset.data.iter().copied().collect())
        .map_err(map_ts_error)
}

fn frame_to_dataset(frame: &thunderstorm_rs::Frame2D, metadata: &Metadata) -> Result<DatasetF32> {
    let data = Array::from_shape_vec(IxDyn(&[frame.height, frame.width]), frame.pixels.clone())
        .map_err(|error| {
            OpsError::InvalidParams(format!("thunderstorm output shape is invalid: {error}"))
        })?;
    Ok(Dataset::new(data, metadata.clone())?)
}

fn to_measurements(output: &thunderstorm_rs::OpOutput) -> MeasurementTable {
    let mut table = MeasurementTable::default();
    for (key, value) in &output.measurements {
        table.values.insert(format!("thunderstorm.{key}"), value.clone());
    }
    table.values.insert(
        "thunderstorm.detection_count".to_string(),
        json!(output.detections.len()),
    );
    table.values.insert(
        "thunderstorm.localization_count".to_string(),
        json!(output.molecules.len()),
    );
    table
        .values
        .insert("thunderstorm.detections".to_string(), json!(&output.detections));
    table.values.insert(
        "thunderstorm.localizations".to_string(),
        json!(&output.molecules),
    );
    table
}

fn execute_thunderstorm(op_name: &str, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
    let frame = dataset_to_frame(dataset)?;
    let output = thunderstorm_rs::execute_operation(op_name, &frame, params).map_err(map_ts_error)?;
    let dataset = frame_to_dataset(&output.frame, &dataset.metadata)?;
    let measurements = to_measurements(&output);
    Ok(OpOutput {
        dataset,
        measurements: Some(measurements),
    })
}

#[derive(Debug, Clone, Copy)]
pub struct ThunderstormGaussianFilterOp;

impl Operation for ThunderstormGaussianFilterOp {
    fn name(&self) -> &'static str {
        "thunderstorm.filter.gaussian"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "ThunderSTORM-style gaussian filtering.".to_string(),
            params: vec![
                ParamSpec {
                    name: "sigma".to_string(),
                    description: "Gaussian sigma in pixels (default 1.2).".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "radius".to_string(),
                    description: "Kernel radius in pixels (default ceil(3*sigma)).".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        execute_thunderstorm(self.name(), dataset, params)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ThunderstormNonMaxSuppressionOp;

impl Operation for ThunderstormNonMaxSuppressionOp {
    fn name(&self) -> &'static str {
        "thunderstorm.detect.non_max_suppression"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description:
                "ThunderSTORM-style local maxima detector producing candidate localizations."
                    .to_string(),
            params: vec![
                ParamSpec {
                    name: "threshold".to_string(),
                    description: "Minimum response threshold for candidate acceptance.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "radius".to_string(),
                    description: "Neighborhood radius for maxima suppression.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        execute_thunderstorm(self.name(), dataset, params)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ThunderstormLsqGaussianFitOp;

impl Operation for ThunderstormLsqGaussianFitOp {
    fn name(&self) -> &'static str {
        "thunderstorm.fit.lsq_gaussian_2d"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "ThunderSTORM-style LSQ Gaussian fitting for detected candidates."
                .to_string(),
            params: vec![ParamSpec {
                name: "window_radius".to_string(),
                description: "Half-window radius around each candidate (default 3).".to_string(),
                required: false,
                kind: "int".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        execute_thunderstorm(self.name(), dataset, params)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ThunderstormPipelineLocalizeOp;

impl Operation for ThunderstormPipelineLocalizeOp {
    fn name(&self) -> &'static str {
        "thunderstorm.pipeline.localize"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description:
                "Run ThunderSTORM filter+detect+fit sequence; configurable via nested filter/detector/fitter blocks."
                    .to_string(),
            params: vec![
                ParamSpec {
                    name: "filter".to_string(),
                    description:
                        "Object with `op` and `params`, default op thunderstorm.filter.gaussian."
                            .to_string(),
                    required: false,
                    kind: "object".to_string(),
                },
                ParamSpec {
                    name: "detector".to_string(),
                    description:
                        "Object with `op` and `params`, default op thunderstorm.detect.non_max_suppression."
                            .to_string(),
                    required: false,
                    kind: "object".to_string(),
                },
                ParamSpec {
                    name: "fitter".to_string(),
                    description:
                        "Object with `op` and `params`, default op thunderstorm.fit.lsq_gaussian_2d."
                            .to_string(),
                    required: false,
                    kind: "object".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        execute_thunderstorm(self.name(), dataset, params)
    }
}
