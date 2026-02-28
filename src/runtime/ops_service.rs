use std::collections::HashMap;
use std::sync::Arc;

use crate::commands::{
    OpOutput, OpSchema, Operation, default_registry, execute_operation_with_registry,
};
use crate::model::DatasetF32;
use serde_json::Value;

use super::Result;

#[derive(Clone)]
pub struct OpsService {
    registry: HashMap<&'static str, Arc<dyn Operation>>,
}

impl std::fmt::Debug for OpsService {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpsService")
            .field("registered_ops", &self.registry.len())
            .finish()
    }
}

impl Default for OpsService {
    fn default() -> Self {
        Self {
            registry: default_registry(),
        }
    }
}

impl OpsService {
    pub fn list(&self) -> Vec<OpSchema> {
        let mut schemas = self
            .registry
            .values()
            .map(|operation| operation.schema())
            .collect::<Vec<_>>();
        schemas.sort_by(|left, right| left.name.cmp(&right.name));
        schemas
    }

    pub fn execute(&self, op: &str, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        Ok(execute_operation_with_registry(
            &self.registry,
            op,
            dataset,
            params,
        )?)
    }

    pub fn registry(&self) -> &HashMap<&'static str, Arc<dyn Operation>> {
        &self.registry
    }
}
