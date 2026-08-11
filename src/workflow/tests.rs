use std::sync::Arc;

use crate::commands::{
    OpOutput, OpSchema, Operation, OperationRegistry, Result as OperationResult, default_registry,
};
use crate::model::{AxisKind, Dataset, Dim, Metadata, PixelType};
use ndarray::Array;
use serde_json::{Value, json};

use super::{OpInvocation, PipelineSpec, run_pipeline};

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

struct StatusOperation;

impl Operation for StatusOperation {
    fn name(&self) -> &str {
        "test.status"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Return a test status".to_string(),
            params: Vec::new(),
        }
    }

    fn execute(&self, dataset: &Dataset<f32>, _params: &Value) -> OperationResult<OpOutput> {
        Ok(OpOutput {
            dataset: dataset.clone(),
            measurements: None,
            status: Some("plugin completed".to_string()),
        })
    }
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
    let registry: OperationRegistry = default_registry();
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
    let registry: OperationRegistry = default_registry();
    assert!(run_pipeline(&spec, &dataset, &registry).is_err());
}

#[test]
fn pipeline_preserves_operation_status_in_step_report() {
    let spec = PipelineSpec {
        name: None,
        operations: vec![OpInvocation {
            op: "test.status".to_string(),
            params: json!({}),
        }],
    };
    let dataset = test_dataset();
    let mut registry = OperationRegistry::default();
    registry.register(Arc::new(StatusOperation)).unwrap();

    let (_, report) = run_pipeline(&spec, &dataset, &registry).expect("pipeline");

    assert_eq!(report.steps[0].status.as_deref(), Some("plugin completed"));
}
