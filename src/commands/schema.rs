use crate::model::DatasetF32;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::Result;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParamSpec {
    pub name: String,
    pub description: String,
    pub required: bool,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpSchema {
    pub name: String,
    pub description: String,
    pub params: Vec<ParamSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MeasurementTable {
    pub values: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone)]
pub struct OpOutput {
    pub dataset: DatasetF32,
    pub measurements: Option<MeasurementTable>,
}

impl OpOutput {
    pub fn dataset_only(dataset: DatasetF32) -> Self {
        Self {
            dataset,
            measurements: None,
        }
    }
}

pub trait Operation: Send + Sync {
    fn name(&self) -> &'static str;
    fn schema(&self) -> OpSchema;
    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput>;
}
