use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use ijrs_core::{CoreError, DatasetF32, Metadata};
use ijrs_ops::{MeasurementTable, Operation, OpsError, execute_operation_with_registry};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, PipelineError>;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("pipeline specification parse failure: {0}")]
    Parse(String),

    #[error("pipeline I/O failure: {0}")]
    Io(#[from] std::io::Error),

    #[error("pipeline serialization failure: {0}")]
    SerdeJson(#[from] serde_json::Error),

    #[error("pipeline YAML serialization failure: {0}")]
    SerdeYaml(#[from] serde_yaml::Error),

    #[error("operation execution failed: {0}")]
    Operation(#[from] OpsError),

    #[error("dataset validation failed: {0}")]
    Core(#[from] CoreError),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineSpec {
    pub name: Option<String>,
    #[serde(default)]
    pub operations: Vec<OpInvocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpInvocation {
    pub op: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StepReport {
    pub op: String,
    pub duration_ms: u128,
    pub measurements: Option<MeasurementTable>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineReport {
    pub pipeline_name: Option<String>,
    pub steps: Vec<StepReport>,
    pub final_measurements: BTreeMap<String, Value>,
    pub output_metadata: Metadata,
}

impl PipelineSpec {
    pub fn validate(&self) -> Result<()> {
        if self.operations.is_empty() {
            return Err(PipelineError::Parse(
                "pipeline must include at least one operation".to_string(),
            ));
        }
        for (index, step) in self.operations.iter().enumerate() {
            if step.op.trim().is_empty() {
                return Err(PipelineError::Parse(format!(
                    "operation at index {index} has an empty name"
                )));
            }
            if !step.params.is_object() && !step.params.is_null() {
                return Err(PipelineError::Parse(format!(
                    "operation `{}` parameters must be a JSON object",
                    step.op
                )));
            }
        }
        Ok(())
    }
}

pub fn load_spec(path: impl AsRef<Path>) -> Result<PipelineSpec> {
    let path = path.as_ref();
    let raw = fs::read_to_string(path)?;
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let spec = if matches!(extension.as_str(), "yaml" | "yml") {
        serde_yaml::from_str::<PipelineSpec>(&raw)?
    } else {
        serde_json::from_str::<PipelineSpec>(&raw)?
    };
    spec.validate()?;
    Ok(spec)
}

pub fn save_report(path: impl AsRef<Path>, report: &PipelineReport) -> Result<()> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let serialized = if matches!(extension.as_str(), "yaml" | "yml") {
        serde_yaml::to_string(report)?
    } else {
        serde_json::to_string_pretty(report)?
    };
    fs::write(path, serialized)?;
    Ok(())
}

pub fn run_pipeline(
    spec: &PipelineSpec,
    dataset: &DatasetF32,
    registry: &HashMap<&'static str, Arc<dyn Operation>>,
) -> Result<(DatasetF32, PipelineReport)> {
    spec.validate()?;
    dataset.validate()?;

    let mut current = dataset.clone();
    let mut steps = Vec::with_capacity(spec.operations.len());
    let mut final_measurements = BTreeMap::new();

    for invocation in &spec.operations {
        let started = Instant::now();
        let output = execute_operation_with_registry(
            registry,
            &invocation.op,
            &current,
            &invocation.params,
        )?;
        let duration_ms = started.elapsed().as_millis();
        if let Some(measurements) = &output.measurements {
            for (key, value) in &measurements.values {
                final_measurements.insert(key.clone(), value.clone());
            }
        }
        steps.push(StepReport {
            op: invocation.op.clone(),
            duration_ms,
            measurements: output.measurements.clone(),
        });
        current = output.dataset;
    }

    let report = PipelineReport {
        pipeline_name: spec.name.clone(),
        steps,
        final_measurements,
        output_metadata: current.metadata.clone(),
    };
    Ok((current, report))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use ijrs_core::{AxisKind, Dataset, Dim, Metadata, PixelType};
    use ijrs_ops::{Operation, default_registry};
    use ndarray::Array;
    use serde_json::json;

    use crate::{OpInvocation, PipelineSpec, run_pipeline};

    fn test_dataset() -> Dataset<f32> {
        let data = Array::from_shape_vec((2, 2), vec![0.1_f32, 0.3, 0.8, 0.9])
            .expect("shape")
            .into_dyn();
        let metadata = Metadata {
            dims: vec![Dim::new(AxisKind::Y, 2), Dim::new(AxisKind::X, 2)],
            pixel_type: PixelType::F32,
            ..Metadata::default()
        };
        Dataset::new(data, metadata).expect("dataset")
    }

    #[test]
    fn pipeline_executes_in_order() {
        let spec = PipelineSpec {
            name: Some("test".to_string()),
            operations: vec![
                OpInvocation {
                    op: "intensity.normalize".to_string(),
                    params: json!({}),
                },
                OpInvocation {
                    op: "threshold.fixed".to_string(),
                    params: json!({"threshold": 0.5}),
                },
            ],
        };
        let dataset = test_dataset();
        let registry: HashMap<&'static str, Arc<dyn Operation>> = default_registry();
        let (result, report) = run_pipeline(&spec, &dataset, &registry).expect("pipeline");
        assert_eq!(report.steps.len(), 2);
        assert!(
            result
                .data
                .iter()
                .all(|value| *value == 0.0 || *value == 1.0)
        );
    }

    #[test]
    fn invalid_pipeline_is_rejected() {
        let spec = PipelineSpec {
            name: None,
            operations: vec![],
        };
        let dataset = test_dataset();
        let registry: HashMap<&'static str, Arc<dyn Operation>> = default_registry();
        assert!(run_pipeline(&spec, &dataset, &registry).is_err());
    }
}
