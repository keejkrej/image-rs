use std::collections::BTreeMap;

use crate::commands::MeasurementTable;
use crate::model::Metadata;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
