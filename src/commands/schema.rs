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
    pub status: Option<String>,
}

impl OpOutput {
    pub fn dataset_only(dataset: DatasetF32) -> Self {
        Self {
            dataset,
            measurements: None,
            status: None,
        }
    }
}

pub trait Operation: Send + Sync {
    /// Stable operation identifier.
    ///
    /// The identifier may be owned by the operation implementation. This is
    /// what lets validated plugin operations join the same registry as the
    /// built-in operations without leaking dynamically allocated strings.
    fn name(&self) -> &str;
    fn schema(&self) -> OpSchema;
    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput>;
}
