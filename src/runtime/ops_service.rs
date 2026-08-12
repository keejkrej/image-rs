use std::sync::Arc;

use crate::commands::{
    ExecutionControl, InvocationRequest, InvocationResult, OpOutput, OpSchema, OperationDescriptor,
    OperationRegistry, default_registry, execute_operation_with_registry,
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

    pub fn describe(&self, op: &str) -> Option<OperationDescriptor> {
        self.registry.describe(op)
    }

    pub fn invoke(
        &self,
        request: InvocationRequest,
        control: &ExecutionControl,
    ) -> Result<InvocationResult> {
        Ok(self.registry.invoke(request, control)?)
    }

    pub fn registry(&self) -> &OperationRegistry {
        self.registry.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array;
    use serde_json::json;

    use crate::commands::{DatasetEffect, OperationScope, PlanePosition};
    use crate::model::{AxisKind, Dataset, Dim, Metadata, PixelType};

    use super::*;

    fn two_plane_dataset() -> Arc<DatasetF32> {
        Arc::new(
            Dataset::new(
                Array::from_shape_vec((2, 1, 2), vec![1.0, 2.0, 10.0, 20.0])
                    .unwrap()
                    .into_dyn(),
                Metadata {
                    dims: vec![
                        Dim::new(AxisKind::Z, 2),
                        Dim::new(AxisKind::Y, 1),
                        Dim::new(AxisKind::X, 2),
                    ],
                    pixel_type: PixelType::F32,
                    ..Metadata::default()
                },
            )
            .unwrap(),
        )
    }

    fn three_slice_impulse() -> Arc<DatasetF32> {
        Arc::new(
            Dataset::new(
                Array::from_shape_vec((3, 1, 1), vec![0.0, 10.0, 0.0])
                    .unwrap()
                    .into_dyn(),
                Metadata {
                    dims: vec![
                        Dim::new(AxisKind::Z, 3),
                        Dim::new(AxisKind::Y, 1),
                        Dim::new(AxisKind::X, 1),
                    ],
                    pixel_type: PixelType::F32,
                    ..Metadata::default()
                },
            )
            .unwrap(),
        )
    }

    #[test]
    fn service_describes_and_invokes_active_plane_operations() {
        let service = OpsService::default();
        let descriptor = service.describe("intensity.math").unwrap();
        assert!(descriptor.supports(OperationScope::ActivePlane));
        assert!(!descriptor.supports(OperationScope::WholeDataset));

        let input = two_plane_dataset();
        let result = service
            .invoke(
                InvocationRequest {
                    operation: "intensity.math".into(),
                    input: input.clone(),
                    parameters: json!({"operation": "add", "value": 5.0}),
                    scope: OperationScope::ActivePlane,
                    active: PlanePosition {
                        channel: 0,
                        z: 1,
                        time: 0,
                    },
                    area_mask: None,
                },
                &ExecutionControl::default(),
            )
            .unwrap();
        let DatasetEffect::Replaced { before, after } = result.dataset_effect else {
            panic!("expected replacement");
        };
        assert!(Arc::ptr_eq(&before, &input));
        assert_eq!(after.data[[0, 0, 0]], 1.0);
        assert_eq!(after.data[[0, 0, 1]], 2.0);
        assert_eq!(after.data[[1, 0, 0]], 15.0);
        assert_eq!(after.data[[1, 0, 1]], 25.0);
    }

    #[test]
    fn legacy_execute_keeps_all_plane_behavior_for_plane_adapters() {
        let service = OpsService::default();
        let input = two_plane_dataset();
        let output = service
            .execute(
                "intensity.math",
                &input,
                &json!({"operation": "add", "value": 5.0}),
            )
            .unwrap();
        assert_eq!(
            output.dataset.data.iter().copied().collect::<Vec<_>>(),
            vec![6.0, 7.0, 15.0, 25.0]
        );
    }

    #[test]
    fn legacy_execute_preserves_native_volume_semantics_while_invoke_is_scoped() {
        let service = OpsService::default();
        let input = three_slice_impulse();

        let legacy = service
            .execute("gaussian.blur", &input, &json!({"sigma": 1.0}))
            .unwrap();
        assert!(legacy.dataset.data[[0, 0, 0]] > 0.0);
        assert!(legacy.dataset.data[[1, 0, 0]] < 10.0);

        let scoped = service
            .invoke(
                InvocationRequest {
                    operation: "gaussian.blur".into(),
                    input: input.clone(),
                    parameters: json!({"sigma": 1.0}),
                    scope: OperationScope::ActivePlane,
                    active: PlanePosition {
                        channel: 0,
                        z: 1,
                        time: 0,
                    },
                    area_mask: None,
                },
                &ExecutionControl::default(),
            )
            .unwrap();
        assert!(matches!(scoped.dataset_effect, DatasetEffect::Unchanged));
    }
}
