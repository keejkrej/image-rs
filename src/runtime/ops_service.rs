use std::sync::Arc;

use crate::commands::{
    OpOutput, OpSchema, OperationRegistry, default_registry, execute_operation_with_registry,
};
use crate::model::DatasetF32;
use serde_json::Value;

use super::Result;

#[derive(Clone)]
pub struct OpsService {
    registry: Arc<OperationRegistry>,
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
            registry: Arc::new(default_registry()),
        }
    }
}

impl OpsService {
    pub fn from_registry(registry: OperationRegistry) -> Self {
        Self {
            registry: Arc::new(registry),
        }
    }

    pub fn list(&self) -> Vec<OpSchema> {
        self.registry.list()
    }

    pub fn execute(&self, op: &str, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        Ok(execute_operation_with_registry(
            &self.registry,
            op,
            dataset,
            params,
        )?)
    }

    pub fn registry(&self) -> &OperationRegistry {
        self.registry.as_ref()
    }
}
