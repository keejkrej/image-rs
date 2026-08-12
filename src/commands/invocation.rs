use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use serde_json::Value;

use ndarray::{Array, IxDyn};

use crate::model::{AxisKind, Dataset, DatasetF32, Metadata};

use super::{MeasurementTable, OpSchema, Operation, OpsError, Result};

/// How an operation should consume the owned input dataset.
///
/// `WholeDataset` is intentionally distinct from `AllPlanes`: the former invokes
/// an n-dimensional operation once, while the latter invokes a 2D operation for
/// each C/Z/T plane in C-fastest, then Z, then T order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationScope {
    WholeDataset,
    ActivePlane,
    ZStack,
    AllPlanes,
}

impl OperationScope {
    pub fn is_plane_wise(self) -> bool {
        !matches!(self, Self::WholeDataset)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PlanePosition {
    pub channel: usize,
    pub z: usize,
    pub time: usize,
}

/// Exact area selection in image pixel coordinates.
///
/// Construction validates shape and membership values. Dataset bounds are
/// validated when the request is invoked because the mask deliberately does not
/// own or borrow a dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AreaMask {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    members: Arc<[u8]>,
}

impl AreaMask {
    pub fn new(
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        members: impl Into<Arc<[u8]>>,
    ) -> Result<Self> {
        let members = members.into();
        if width == 0 || height == 0 {
            return Err(super::OpsError::InvalidAreaMask(
                "area mask width and height must be greater than zero".into(),
            ));
        }
        let expected = width.checked_mul(height).ok_or_else(|| {
            super::OpsError::InvalidAreaMask("area mask dimensions overflowed".into())
        })?;
        if members.len() != expected {
            return Err(super::OpsError::InvalidAreaMask(format!(
                "area mask contains {} members but its bounds require {expected}",
                members.len()
            )));
        }
        if members.iter().any(|member| *member > 1) {
            return Err(super::OpsError::InvalidAreaMask(
                "area mask members must be exactly 0 or 1".into(),
            ));
        }
        x.checked_add(width).ok_or_else(|| {
            super::OpsError::InvalidAreaMask("area mask horizontal bounds overflowed".into())
        })?;
        y.checked_add(height).ok_or_else(|| {
            super::OpsError::InvalidAreaMask("area mask vertical bounds overflowed".into())
        })?;
        Ok(Self {
            x,
            y,
            width,
            height,
            members,
        })
    }

    pub fn x(&self) -> usize {
        self.x
    }

    pub fn y(&self) -> usize {
        self.y
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn members(&self) -> &[u8] {
        &self.members
    }

    pub fn contains(&self, x: usize, y: usize) -> bool {
        if x < self.x || y < self.y || x >= self.x + self.width || y >= self.y + self.height {
            return false;
        }
        let local_x = x - self.x;
        let local_y = y - self.y;
        self.members[local_y * self.width + local_x] == 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AreaMaskSupport {
    Unsupported,
    Optional,
    Required,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OperationDescriptor {
    pub schema: OpSchema,
    pub scopes: Vec<OperationScope>,
    pub area_mask: AreaMaskSupport,
}

impl OperationDescriptor {
    pub fn whole_dataset(schema: OpSchema) -> Self {
        Self {
            schema,
            scopes: vec![OperationScope::WholeDataset],
            area_mask: AreaMaskSupport::Unsupported,
        }
    }

    pub fn plane_wise(schema: OpSchema, area_mask: AreaMaskSupport) -> Self {
        Self {
            schema,
            scopes: vec![
                OperationScope::ActivePlane,
                OperationScope::ZStack,
                OperationScope::AllPlanes,
            ],
            area_mask,
        }
    }

    pub fn supports(&self, scope: OperationScope) -> bool {
        self.scopes.contains(&scope)
    }
}

#[derive(Debug, Clone)]
pub struct InvocationRequest {
    pub operation: String,
    pub input: Arc<DatasetF32>,
    pub parameters: Value,
    pub scope: OperationScope,
    pub active: PlanePosition,
    pub area_mask: Option<AreaMask>,
}

impl InvocationRequest {
    pub fn whole_dataset(
        operation: impl Into<String>,
        input: Arc<DatasetF32>,
        parameters: Value,
    ) -> Self {
        Self {
            operation: operation.into(),
            input,
            parameters,
            scope: OperationScope::WholeDataset,
            active: PlanePosition::default(),
            area_mask: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum DatasetEffect {
    Unchanged,
    Replaced {
        before: Arc<DatasetF32>,
        after: Arc<DatasetF32>,
    },
}

impl DatasetEffect {
    pub fn dataset<'a>(&'a self, unchanged: &'a Arc<DatasetF32>) -> &'a Arc<DatasetF32> {
        match self {
            Self::Unchanged => unchanged,
            Self::Replaced { after, .. } => after,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InvocationResult {
    pub dataset_effect: DatasetEffect,
    pub measurements: Option<MeasurementTable>,
    pub status: Option<String>,
}

impl InvocationResult {
    pub fn unchanged(measurements: Option<MeasurementTable>, status: Option<String>) -> Self {
        Self {
            dataset_effect: DatasetEffect::Unchanged,
            measurements,
            status,
        }
    }

    pub fn replaced(
        before: Arc<DatasetF32>,
        after: DatasetF32,
        measurements: Option<MeasurementTable>,
        status: Option<String>,
    ) -> Self {
        Self {
            dataset_effect: DatasetEffect::Replaced {
                before,
                after: Arc::new(after),
            },
            measurements,
            status,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkProgress {
    pub completed: u64,
    pub total: Option<u64>,
    pub message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressEvent {
    pub completed_planes: usize,
    pub total_planes: usize,
    pub current_plane: Option<PlanePosition>,
    pub detail: Option<WorkProgress>,
}

pub trait ProgressSink: Send + Sync {
    fn report(&self, event: ProgressEvent);
}

#[derive(Debug, Default)]
struct NoopProgressSink;

impl ProgressSink for NoopProgressSink {
    fn report(&self, _event: ProgressEvent) {}
}

#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Clone)]
pub struct ExecutionControl {
    cancellation: CancellationToken,
    progress: Arc<dyn ProgressSink>,
}

impl std::fmt::Debug for ExecutionControl {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExecutionControl")
            .field("cancelled", &self.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl Default for ExecutionControl {
    fn default() -> Self {
        Self {
            cancellation: CancellationToken::default(),
            progress: Arc::new(NoopProgressSink),
        }
    }
}

impl ExecutionControl {
    pub fn new(cancellation: CancellationToken, progress: Arc<dyn ProgressSink>) -> Self {
        Self {
            cancellation,
            progress,
        }
    }

    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    pub fn report(&self, event: ProgressEvent) {
        self.progress.report(event);
    }
}

/// A full-invocation adapter registered at the command execution seam.
///
/// Native plane wrapping and sandboxed plugins are separate adapters. Each may
/// retain private per-invocation state while callers use the same owned request,
/// cancellation/progress control, and atomic result interface.
pub(crate) trait ScopedOperation: Send + Sync {
    fn name(&self) -> &str;
    fn descriptor(&self) -> OperationDescriptor;
    fn invoke(
        &self,
        request: &InvocationRequest,
        control: &ExecutionControl,
    ) -> Result<InvocationResult>;
}

/// Adapts a legacy same-shape 2D native operation to the scoped invocation seam.
///
/// Every kernel receives a full calibrated X/Y plane. Masking is applied only
/// while scattering the replacement, so neighborhood filters retain pixels
/// outside the selection as context without modifying them.
pub struct NativePlaneAdapter {
    operation: Arc<dyn Operation>,
}

impl NativePlaneAdapter {
    pub fn new(operation: Arc<dyn Operation>) -> Self {
        Self { operation }
    }
}

impl std::fmt::Debug for NativePlaneAdapter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativePlaneAdapter")
            .field("operation", &self.operation.name())
            .finish()
    }
}

impl ScopedOperation for NativePlaneAdapter {
    fn name(&self) -> &str {
        self.operation.name()
    }

    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor::plane_wise(self.operation.schema(), AreaMaskSupport::Optional)
    }

    fn invoke(
        &self,
        request: &InvocationRequest,
        control: &ExecutionControl,
    ) -> Result<InvocationResult> {
        if !request.scope.is_plane_wise() {
            return Err(OpsError::UnsupportedScope {
                operation: request.operation.clone(),
                scope: request.scope,
            });
        }
        let layout = PlaneLayout::from_request(request)?;
        let positions = layout.schedule(request.scope, request.active);
        let total_planes = positions.len();
        let mut staged = request.input.as_ref().clone();
        let mut measurement_rows = Vec::new();
        let mut status = None;

        control.report(ProgressEvent {
            completed_planes: 0,
            total_planes,
            current_plane: None,
            detail: None,
        });
        for (index, position) in positions.into_iter().enumerate() {
            check_cancelled(control)?;
            control.report(ProgressEvent {
                completed_planes: index,
                total_planes,
                current_plane: Some(position),
                detail: None,
            });
            let plane = layout.gather(request.input.as_ref(), position)?;
            let output = self.operation.execute(&plane, &request.parameters)?;
            layout.validate_native_output(&output.dataset, &plane, self.name())?;
            layout.scatter(
                &mut staged,
                &output.dataset,
                position,
                request.area_mask.as_ref(),
            );

            if let Some(measurements) = output.measurements {
                measurement_rows.push(serde_json::json!({
                    "channel": position.channel,
                    "z": position.z,
                    "time": position.time,
                    "values": measurements.values,
                }));
            }
            if output.status.is_some() {
                status = output.status;
            }
            check_cancelled(control)?;
            control.report(ProgressEvent {
                completed_planes: index + 1,
                total_planes,
                current_plane: Some(position),
                detail: None,
            });
        }
        check_cancelled(control)?;
        staged.validate()?;
        let measurements = (!measurement_rows.is_empty()).then(|| MeasurementTable {
            values: std::collections::BTreeMap::from([(
                "planes".to_string(),
                Value::Array(measurement_rows),
            )]),
        });
        if datasets_identical(request.input.as_ref(), &staged) {
            Ok(InvocationResult::unchanged(measurements, status))
        } else {
            Ok(InvocationResult::replaced(
                request.input.clone(),
                staged,
                measurements,
                status,
            ))
        }
    }
}

fn check_cancelled(control: &ExecutionControl) -> Result<()> {
    if control.is_cancelled() {
        Err(OpsError::Cancelled)
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct PlaneLayout {
    x_axis: usize,
    y_axis: usize,
    channel_axis: Option<usize>,
    z_axis: Option<usize>,
    time_axis: Option<usize>,
    width: usize,
    height: usize,
    channels: usize,
    depth: usize,
    times: usize,
}

impl PlaneLayout {
    fn from_request(request: &InvocationRequest) -> Result<Self> {
        let dataset = request.input.as_ref();
        let mut x_axis = None;
        let mut y_axis = None;
        let mut channel_axis = None;
        let mut z_axis = None;
        let mut time_axis = None;
        for (index, dimension) in dataset.metadata.dims.iter().enumerate() {
            let slot = match dimension.axis {
                AxisKind::X => Some(&mut x_axis),
                AxisKind::Y => Some(&mut y_axis),
                AxisKind::Channel => Some(&mut channel_axis),
                AxisKind::Z => Some(&mut z_axis),
                AxisKind::Time => Some(&mut time_axis),
                AxisKind::Unknown => {
                    if dimension.size != 1 {
                        return Err(OpsError::UnsupportedLayout(format!(
                            "plane-wise operations cannot schedule unknown axis {index} of size {}",
                            dimension.size
                        )));
                    }
                    None
                }
            };
            if let Some(slot) = slot
                && slot.replace(index).is_some()
            {
                return Err(OpsError::UnsupportedLayout(format!(
                    "plane-wise operations require unique {:?} axes",
                    dimension.axis
                )));
            }
        }
        let x_axis = x_axis.ok_or_else(|| {
            OpsError::UnsupportedLayout("plane-wise operations require an X axis".into())
        })?;
        let y_axis = y_axis.ok_or_else(|| {
            OpsError::UnsupportedLayout("plane-wise operations require a Y axis".into())
        })?;
        let width = dataset.shape()[x_axis];
        let height = dataset.shape()[y_axis];
        let channels = channel_axis.map_or(1, |axis| dataset.shape()[axis]);
        let depth = z_axis.map_or(1, |axis| dataset.shape()[axis]);
        let times = time_axis.map_or(1, |axis| dataset.shape()[axis]);
        validate_active(request.active, channels, depth, times)?;
        if let Some(mask) = request.area_mask.as_ref()
            && (mask.x() + mask.width() > width || mask.y() + mask.height() > height)
        {
            return Err(OpsError::InvalidAreaMask(format!(
                "area mask bounds ({}, {}) {}x{} exceed the {width}x{height} image plane",
                mask.x(),
                mask.y(),
                mask.width(),
                mask.height()
            )));
        }
        Ok(Self {
            x_axis,
            y_axis,
            channel_axis,
            z_axis,
            time_axis,
            width,
            height,
            channels,
            depth,
            times,
        })
    }

    fn schedule(&self, scope: OperationScope, active: PlanePosition) -> Vec<PlanePosition> {
        match scope {
            OperationScope::WholeDataset => Vec::new(),
            OperationScope::ActivePlane => vec![active],
            OperationScope::ZStack => (0..self.depth)
                .map(|z| PlanePosition {
                    channel: active.channel,
                    z,
                    time: active.time,
                })
                .collect(),
            OperationScope::AllPlanes => {
                let mut positions = Vec::with_capacity(self.channels * self.depth * self.times);
                for time in 0..self.times {
                    for z in 0..self.depth {
                        for channel in 0..self.channels {
                            positions.push(PlanePosition { channel, z, time });
                        }
                    }
                }
                positions
            }
        }
    }

    fn gather(&self, dataset: &DatasetF32, position: PlanePosition) -> Result<DatasetF32> {
        let mut coordinate = vec![0; dataset.ndim()];
        self.set_position(&mut coordinate, position);
        let mut values = Vec::with_capacity(self.width * self.height);
        for y in 0..self.height {
            coordinate[self.y_axis] = y;
            for x in 0..self.width {
                coordinate[self.x_axis] = x;
                values.push(dataset.data[IxDyn(&coordinate)]);
            }
        }
        let data = Array::from_shape_vec((self.height, self.width), values)
            .map_err(|error| OpsError::UnsupportedLayout(error.to_string()))?
            .into_dyn();
        let y_dimension = dataset.metadata.dims[self.y_axis].clone();
        let x_dimension = dataset.metadata.dims[self.x_axis].clone();
        let metadata = Metadata {
            dims: vec![y_dimension, x_dimension],
            pixel_type: dataset.metadata.pixel_type,
            channel_names: Vec::new(),
            source: dataset.metadata.source.clone(),
            extras: dataset.metadata.extras.clone(),
        };
        Ok(Dataset::new(data, metadata)?)
    }

    fn scatter(
        &self,
        staged: &mut DatasetF32,
        replacement: &DatasetF32,
        position: PlanePosition,
        mask: Option<&AreaMask>,
    ) {
        let mut coordinate = vec![0; staged.ndim()];
        self.set_position(&mut coordinate, position);
        for y in 0..self.height {
            coordinate[self.y_axis] = y;
            for x in 0..self.width {
                if mask.is_some_and(|mask| !mask.contains(x, y)) {
                    continue;
                }
                coordinate[self.x_axis] = x;
                staged.data[IxDyn(&coordinate)] = replacement.data[[y, x]];
            }
        }
    }

    fn set_position(&self, coordinate: &mut [usize], position: PlanePosition) {
        if let Some(axis) = self.channel_axis {
            coordinate[axis] = position.channel;
        }
        if let Some(axis) = self.z_axis {
            coordinate[axis] = position.z;
        }
        if let Some(axis) = self.time_axis {
            coordinate[axis] = position.time;
        }
    }

    fn validate_native_output(
        &self,
        output: &DatasetF32,
        input: &DatasetF32,
        operation: &str,
    ) -> Result<()> {
        output.validate()?;
        if output.shape() != [self.height, self.width] {
            return Err(OpsError::InvalidOperationOutput {
                operation: operation.to_string(),
                message: format!(
                    "plane replacement shape {:?} does not match [{}, {}]",
                    output.shape(),
                    self.height,
                    self.width
                ),
            });
        }
        if output.metadata.pixel_type != input.metadata.pixel_type {
            return Err(OpsError::InvalidOperationOutput {
                operation: operation.to_string(),
                message: "plane replacement changed the pixel type".into(),
            });
        }
        if !metadata_identical(&output.metadata, &input.metadata) {
            return Err(OpsError::InvalidOperationOutput {
                operation: operation.to_string(),
                message: "plane replacement changed metadata".into(),
            });
        }
        Ok(())
    }
}

pub(super) fn datasets_identical(left: &DatasetF32, right: &DatasetF32) -> bool {
    metadata_identical(&left.metadata, &right.metadata)
        && left.shape() == right.shape()
        && left
            .data
            .iter()
            .zip(right.data.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn metadata_identical(left: &Metadata, right: &Metadata) -> bool {
    left.pixel_type == right.pixel_type
        && left.channel_names == right.channel_names
        && left.source == right.source
        && left.extras == right.extras
        && left.dims.len() == right.dims.len()
        && left.dims.iter().zip(&right.dims).all(|(left, right)| {
            left.axis == right.axis
                && left.size == right.size
                && match (left.spacing, right.spacing) {
                    (Some(left), Some(right)) => left.to_bits() == right.to_bits(),
                    (None, None) => true,
                    _ => false,
                }
                && left.unit == right.unit
        })
}

fn validate_active(
    active: PlanePosition,
    channels: usize,
    depth: usize,
    times: usize,
) -> Result<()> {
    if active.channel >= channels || active.z >= depth || active.time >= times {
        return Err(OpsError::ActivePosition(format!(
            "C/Z/T ({}, {}, {}) is outside logical shape ({channels}, {depth}, {times})",
            active.channel, active.z, active.time
        )));
    }
    Ok(())
}

pub(super) fn validate_plane_request(request: &InvocationRequest) -> Result<()> {
    PlaneLayout::from_request(request).map(|_| ())
}

/// Defensively verify that a scoped adapter changed only host-scheduled mask members.
///
/// Native and Wasm adapters already scatter through the host layout, but the registry accepts
/// other `ScopedOperation` implementations as well. Keeping this check at that public seam makes
/// scope and exact-mask authority an invariant rather than an adapter convention.
pub(super) fn validate_plane_result_changes(
    request: &InvocationRequest,
    after: &DatasetF32,
) -> Result<()> {
    let layout = PlaneLayout::from_request(request)?;
    let scheduled = layout
        .schedule(request.scope, request.active)
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
    for (coordinate, before) in request.input.data.indexed_iter() {
        let replacement = after.data[coordinate.clone()];
        if before.to_bits() == replacement.to_bits() {
            continue;
        }
        let position = PlanePosition {
            channel: layout.channel_axis.map_or(0, |axis| coordinate[axis]),
            z: layout.z_axis.map_or(0, |axis| coordinate[axis]),
            time: layout.time_axis.map_or(0, |axis| coordinate[axis]),
        };
        let x = coordinate[layout.x_axis];
        let y = coordinate[layout.y_axis];
        if !scheduled.contains(&position)
            || request
                .area_mask
                .as_ref()
                .is_some_and(|mask| !mask.contains(x, y))
        {
            return Err(OpsError::InvalidOperationOutput {
                operation: request.operation.clone(),
                message: format!(
                    "plane-wise operation changed pixel ({x}, {y}) outside its scheduled scope or area mask"
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use ndarray::{ArrayD, IxDyn};
    use serde_json::json;

    use crate::commands::{OperationRegistry, default_registry};
    use crate::model::{Dim, PixelType};

    use super::*;

    struct AddOneOp {
        observed_first_pixels: Arc<Mutex<Vec<f32>>>,
    }

    struct MetadataChangingOp;

    impl Operation for AddOneOp {
        fn name(&self) -> &str {
            "test.add_one_plane"
        }

        fn schema(&self) -> OpSchema {
            OpSchema {
                name: self.name().into(),
                description: "test plane operation".into(),
                params: Vec::new(),
            }
        }

        fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<super::super::OpOutput> {
            self.observed_first_pixels
                .lock()
                .unwrap()
                .push(dataset.data[[0, 0]]);
            let mut output = dataset.clone();
            output.data.mapv_inplace(|value| value + 1.0);
            Ok(super::super::OpOutput::dataset_only(output))
        }
    }

    impl Operation for MetadataChangingOp {
        fn name(&self) -> &str {
            "test.metadata_changing_plane"
        }

        fn schema(&self) -> OpSchema {
            OpSchema {
                name: self.name().into(),
                description: "invalid test operation".into(),
                params: Vec::new(),
            }
        }

        fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<super::super::OpOutput> {
            let mut output = dataset.clone();
            output
                .metadata
                .extras
                .insert("unexpected".into(), Value::Bool(true));
            Ok(super::super::OpOutput::dataset_only(output))
        }
    }

    #[derive(Default)]
    struct RecordingProgress(Mutex<Vec<ProgressEvent>>);

    impl ProgressSink for RecordingProgress {
        fn report(&self, event: ProgressEvent) {
            self.0.lock().unwrap().push(event);
        }
    }

    struct CancelAfterFirst {
        cancellation: CancellationToken,
    }

    impl ProgressSink for CancelAfterFirst {
        fn report(&self, event: ProgressEvent) {
            if event.completed_planes == 1 {
                self.cancellation.cancel();
            }
        }
    }

    fn arbitrary_axis_dataset() -> Arc<DatasetF32> {
        // C, Y, T, X, Z deliberately differs from conventional storage order.
        let shape = [2, 2, 2, 3, 2];
        let data = ArrayD::from_shape_fn(IxDyn(&shape), |index| {
            (index[2] * 10_000 + index[4] * 1_000 + index[0] * 100 + index[1] * 10 + index[3])
                as f32
        });
        Arc::new(
            Dataset::new(
                data,
                Metadata {
                    dims: vec![
                        Dim::new(AxisKind::Channel, shape[0]),
                        Dim::new(AxisKind::Y, shape[1]),
                        Dim::new(AxisKind::Time, shape[2]),
                        Dim::new(AxisKind::X, shape[3]),
                        Dim::new(AxisKind::Z, shape[4]),
                    ],
                    pixel_type: PixelType::F32,
                    ..Metadata::default()
                },
            )
            .unwrap(),
        )
    }

    fn registry_with_add_one(observed: Arc<Mutex<Vec<f32>>>) -> OperationRegistry {
        let mut registry = OperationRegistry::default();
        registry
            .register_scoped(Arc::new(NativePlaneAdapter::new(Arc::new(AddOneOp {
                observed_first_pixels: observed,
            }))))
            .unwrap();
        registry
    }

    fn request(
        input: Arc<DatasetF32>,
        scope: OperationScope,
        active: PlanePosition,
        area_mask: Option<AreaMask>,
    ) -> InvocationRequest {
        InvocationRequest {
            operation: "test.add_one_plane".into(),
            input,
            parameters: Value::Null,
            scope,
            active,
            area_mask,
        }
    }

    fn replacement(result: &InvocationResult) -> &Arc<DatasetF32> {
        match &result.dataset_effect {
            DatasetEffect::Replaced { after, .. } => after,
            DatasetEffect::Unchanged => panic!("expected a replacement"),
        }
    }

    #[test]
    fn area_mask_constructor_enforces_exact_shape_and_binary_membership() {
        assert!(AreaMask::new(0, 0, 0, 1, Vec::<u8>::new()).is_err());
        assert!(AreaMask::new(0, 0, 2, 2, vec![1, 0, 1]).is_err());
        assert!(AreaMask::new(0, 0, 1, 1, vec![2]).is_err());
        assert!(AreaMask::new(usize::MAX, 0, 1, 1, vec![1]).is_err());

        let mask = AreaMask::new(3, 4, 2, 2, vec![1, 0, 0, 1]).unwrap();
        assert_eq!(
            (mask.x(), mask.y(), mask.width(), mask.height()),
            (3, 4, 2, 2)
        );
        assert!(mask.contains(3, 4));
        assert!(!mask.contains(4, 4));
        assert!(mask.contains(4, 5));
        assert!(!mask.contains(2, 4));
    }

    #[test]
    fn all_planes_use_c_fast_z_then_t_schedule_for_arbitrary_axis_order() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let registry = registry_with_add_one(observed.clone());
        let input = arbitrary_axis_dataset();
        let progress = Arc::new(RecordingProgress::default());
        let control = ExecutionControl::new(CancellationToken::default(), progress.clone());
        let result = registry
            .invoke(
                request(
                    input.clone(),
                    OperationScope::AllPlanes,
                    PlanePosition::default(),
                    None,
                ),
                &control,
            )
            .unwrap();

        assert_eq!(
            *observed.lock().unwrap(),
            vec![
                0.0, 100.0, 1_000.0, 1_100.0, 10_000.0, 10_100.0, 11_000.0, 11_100.0
            ]
        );
        assert!(Arc::ptr_eq(
            match &result.dataset_effect {
                DatasetEffect::Replaced { before, .. } => before,
                DatasetEffect::Unchanged => panic!("expected replacement"),
            },
            &input
        ));
        assert_eq!(replacement(&result).data[[1, 1, 1, 2, 1]], 11_113.0);
        assert_eq!(input.data[[1, 1, 1, 2, 1]], 11_112.0);

        let events = progress.0.lock().unwrap();
        assert_eq!(events.first().unwrap().total_planes, 8);
        assert_eq!(events.last().unwrap().completed_planes, 8);
        assert_eq!(
            events.last().unwrap().current_plane,
            Some(PlanePosition {
                channel: 1,
                z: 1,
                time: 1
            })
        );
    }

    #[test]
    fn active_plane_gathers_full_xy_but_scatters_only_exact_mask_members() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let registry = registry_with_add_one(observed.clone());
        let input = arbitrary_axis_dataset();
        let active = PlanePosition {
            channel: 1,
            z: 1,
            time: 0,
        };
        let mask = AreaMask::new(1, 0, 2, 2, vec![1, 0, 0, 1]).unwrap();
        let result = registry
            .invoke(
                request(
                    input.clone(),
                    OperationScope::ActivePlane,
                    active,
                    Some(mask),
                ),
                &ExecutionControl::default(),
            )
            .unwrap();
        let after = replacement(&result);

        // The kernel observed the full plane, not the mask bounding box.
        assert_eq!(*observed.lock().unwrap(), vec![1_100.0]);
        assert_eq!(after.data[[1, 0, 0, 1, 1]], 1_102.0);
        assert_eq!(after.data[[1, 1, 0, 2, 1]], 1_113.0);
        assert_eq!(after.data[[1, 0, 0, 2, 1]], input.data[[1, 0, 0, 2, 1]]);
        assert_eq!(after.data[[1, 1, 0, 1, 1]], input.data[[1, 1, 0, 1, 1]]);
        assert_eq!(after.data[[0, 0, 0, 1, 1]], input.data[[0, 0, 0, 1, 1]]);
    }

    #[test]
    fn registry_guard_rejects_changes_outside_scope_or_exact_mask() {
        let input = arbitrary_axis_dataset();
        let request = request(
            input.clone(),
            OperationScope::ActivePlane,
            PlanePosition {
                channel: 1,
                z: 1,
                time: 0,
            },
            Some(AreaMask::new(1, 0, 1, 1, vec![1]).unwrap()),
        );

        let mut allowed = input.as_ref().clone();
        allowed.data[[1, 0, 0, 1, 1]] += 1.0;
        validate_plane_result_changes(&request, &allowed).unwrap();

        let mut outside_mask = input.as_ref().clone();
        outside_mask.data[[1, 0, 0, 2, 1]] += 1.0;
        assert!(matches!(
            validate_plane_result_changes(&request, &outside_mask),
            Err(OpsError::InvalidOperationOutput { .. })
        ));

        let mut outside_scope = input.as_ref().clone();
        outside_scope.data[[0, 0, 0, 1, 1]] += 1.0;
        assert!(matches!(
            validate_plane_result_changes(&request, &outside_scope),
            Err(OpsError::InvalidOperationOutput { .. })
        ));
    }

    #[test]
    fn z_stack_holds_active_channel_and_time() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let registry = registry_with_add_one(observed.clone());
        registry
            .invoke(
                request(
                    arbitrary_axis_dataset(),
                    OperationScope::ZStack,
                    PlanePosition {
                        channel: 1,
                        z: 1,
                        time: 1,
                    },
                    None,
                ),
                &ExecutionControl::default(),
            )
            .unwrap();
        assert_eq!(*observed.lock().unwrap(), vec![10_100.0, 11_100.0]);
    }

    #[test]
    fn invalid_active_position_and_out_of_bounds_mask_are_rejected_before_kernel() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let registry = registry_with_add_one(observed.clone());
        let input = arbitrary_axis_dataset();
        let error = registry
            .invoke(
                request(
                    input.clone(),
                    OperationScope::ActivePlane,
                    PlanePosition {
                        channel: 2,
                        z: 0,
                        time: 0,
                    },
                    None,
                ),
                &ExecutionControl::default(),
            )
            .unwrap_err();
        assert!(matches!(error, OpsError::ActivePosition(_)));

        let error = registry
            .invoke(
                request(
                    input,
                    OperationScope::ActivePlane,
                    PlanePosition::default(),
                    Some(AreaMask::new(2, 0, 2, 1, vec![1, 1]).unwrap()),
                ),
                &ExecutionControl::default(),
            )
            .unwrap_err();
        assert!(matches!(error, OpsError::InvalidAreaMask(_)));
        assert!(observed.lock().unwrap().is_empty());
    }

    #[test]
    fn missing_logical_axes_have_size_one_and_require_zero_active_index() {
        let input = Arc::new(
            Dataset::new(
                ArrayD::zeros(IxDyn(&[2, 3])),
                Metadata {
                    dims: vec![Dim::new(AxisKind::Y, 2), Dim::new(AxisKind::X, 3)],
                    pixel_type: PixelType::F32,
                    ..Metadata::default()
                },
            )
            .unwrap(),
        );
        let registry = registry_with_add_one(Arc::new(Mutex::new(Vec::new())));
        let error = registry
            .invoke(
                request(
                    input,
                    OperationScope::ActivePlane,
                    PlanePosition {
                        channel: 0,
                        z: 1,
                        time: 0,
                    },
                    None,
                ),
                &ExecutionControl::default(),
            )
            .unwrap_err();
        assert!(matches!(error, OpsError::ActivePosition(_)));
    }

    #[test]
    fn singleton_unknown_axis_is_pinned_and_non_singleton_unknown_axis_is_rejected() {
        let build = |unknown_size| {
            Arc::new(
                Dataset::new(
                    ArrayD::from_shape_fn(IxDyn(&[2, unknown_size, 3]), |index| {
                        (index[0] * 10 + index[2]) as f32
                    }),
                    Metadata {
                        dims: vec![
                            Dim::new(AxisKind::Y, 2),
                            Dim::new(AxisKind::Unknown, unknown_size),
                            Dim::new(AxisKind::X, 3),
                        ],
                        pixel_type: PixelType::F32,
                        ..Metadata::default()
                    },
                )
                .unwrap(),
            )
        };
        let registry = registry_with_add_one(Arc::new(Mutex::new(Vec::new())));
        let singleton = build(1);
        let result = registry
            .invoke(
                request(
                    singleton,
                    OperationScope::AllPlanes,
                    PlanePosition::default(),
                    None,
                ),
                &ExecutionControl::default(),
            )
            .unwrap();
        assert_eq!(replacement(&result).data[[1, 0, 2]], 13.0);

        let error = registry
            .invoke(
                request(
                    build(2),
                    OperationScope::AllPlanes,
                    PlanePosition::default(),
                    None,
                ),
                &ExecutionControl::default(),
            )
            .unwrap_err();
        assert!(matches!(error, OpsError::UnsupportedLayout(_)));
    }

    #[test]
    fn cancellation_after_a_plane_discards_the_staged_replacement() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let registry = registry_with_add_one(observed.clone());
        let input = arbitrary_axis_dataset();
        let original_bits = input
            .data
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let cancellation = CancellationToken::default();
        let control = ExecutionControl::new(
            cancellation.clone(),
            Arc::new(CancelAfterFirst {
                cancellation: cancellation.clone(),
            }),
        );

        let error = registry
            .invoke(
                request(
                    input.clone(),
                    OperationScope::AllPlanes,
                    PlanePosition::default(),
                    None,
                ),
                &control,
            )
            .unwrap_err();
        assert!(matches!(error, OpsError::Cancelled));
        assert_eq!(observed.lock().unwrap().len(), 1);
        assert_eq!(
            input
                .data
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            original_bits
        );
    }

    #[test]
    fn descriptors_distinguish_plane_wise_and_whole_dataset_operations() {
        let registry = default_registry();
        let gaussian = registry.describe("gaussian.blur").unwrap();
        assert_eq!(gaussian.area_mask, AreaMaskSupport::Optional);
        assert!(!gaussian.supports(OperationScope::WholeDataset));
        assert!(gaussian.supports(OperationScope::ActivePlane));
        assert!(gaussian.supports(OperationScope::ZStack));
        assert!(gaussian.supports(OperationScope::AllPlanes));

        let rank_3d = registry.describe("image.rank_filter_3d").unwrap();
        assert_eq!(rank_3d.scopes, vec![OperationScope::WholeDataset]);
        assert_eq!(rank_3d.area_mask, AreaMaskSupport::Unsupported);
    }

    #[test]
    fn native_adapter_rejects_unrepresentable_plane_metadata_changes() {
        let mut registry = OperationRegistry::default();
        registry
            .register_scoped(Arc::new(NativePlaneAdapter::new(Arc::new(
                MetadataChangingOp,
            ))))
            .unwrap();
        let input = arbitrary_axis_dataset();
        let error = registry
            .invoke(
                InvocationRequest {
                    operation: "test.metadata_changing_plane".into(),
                    input,
                    parameters: Value::Null,
                    scope: OperationScope::ActivePlane,
                    active: PlanePosition::default(),
                    area_mask: None,
                },
                &ExecutionControl::default(),
            )
            .unwrap_err();
        assert!(matches!(error, OpsError::InvalidOperationOutput { .. }));
    }

    #[test]
    fn bit_identical_plane_output_is_explicitly_unchanged() {
        let input = Arc::new(
            Dataset::new(
                ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 1.0, 1.0, 0.0]).unwrap(),
                Metadata {
                    dims: vec![Dim::new(AxisKind::Y, 2), Dim::new(AxisKind::X, 2)],
                    pixel_type: PixelType::F32,
                    ..Metadata::default()
                },
            )
            .unwrap(),
        );
        let result = default_registry()
            .invoke(
                InvocationRequest {
                    operation: "threshold.fixed".into(),
                    input,
                    parameters: json!({"threshold": 0.5}),
                    scope: OperationScope::AllPlanes,
                    active: PlanePosition::default(),
                    area_mask: None,
                },
                &ExecutionControl::default(),
            )
            .unwrap();
        assert!(matches!(result.dataset_effect, DatasetEffect::Unchanged));
    }

    #[test]
    fn read_only_measurement_has_explicit_unchanged_effect() {
        let registry = default_registry();
        let input = arbitrary_axis_dataset();
        let result = registry
            .invoke(
                InvocationRequest::whole_dataset("measurements.summary", input, json!({})),
                &ExecutionControl::default(),
            )
            .unwrap();
        assert!(matches!(result.dataset_effect, DatasetEffect::Unchanged));
        assert!(result.measurements.is_some());
    }
}
