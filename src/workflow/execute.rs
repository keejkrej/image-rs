use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use crate::commands::{Operation, execute_operation_with_registry};
use crate::model::DatasetF32;

use super::{PipelineReport, PipelineSpec, Result, StepReport};

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
