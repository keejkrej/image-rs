use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{PipelineError, Result};

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
