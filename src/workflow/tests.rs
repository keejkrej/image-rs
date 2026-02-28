use std::collections::HashMap;
use std::sync::Arc;

use crate::commands::{Operation, default_registry};
use crate::model::{AxisKind, Dataset, Dim, Metadata, PixelType};
use ndarray::Array;
use serde_json::json;

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
