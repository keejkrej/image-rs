//! Host-side invariants for the versioned WebAssembly Component interface.
//!
//! The WIT file is the guest-facing interface. This module keeps the safety-critical checks that
//! must happen before a future runtime adapter copies host-owned data into a component or commits
//! component output back into application state.

use std::collections::BTreeSet;

use serde_json::Value;
use thiserror::Error;

use crate::model::PixelType;

/// The canonical WIT package shipped with this version of the host.
pub const PLUGIN_WIT_SOURCE: &str = include_str!("../../wit/image-rs-plugin.wit");

pub const PLUGIN_WIT_NAMESPACE: &str = "image-rs";
pub const PLUGIN_WIT_PACKAGE: &str = "plugin";
pub const IMAGE_OPERATION_WORLD: &str = "image-operation-plugin";
pub const COMMAND_HANDLER_WORLD: &str = "command-handler-plugin";
pub const COMBINED_PLUGIN_WORLD: &str = "combined-plugin";

/// Maximum bytes copied across the component seam for one full plane or ROI mask.
pub const MAX_PLUGIN_BUFFER_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_PLUGIN_TEXT_BYTES: usize = 64 * 1024;
pub const MAX_PLUGIN_JSON_BYTES: usize = 256 * 1024;
pub const MAX_PLUGIN_PAYLOAD_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_PLUGIN_PAYLOAD_NODES: usize = 64 * 1024;
pub const MAX_PLUGIN_JSON_DEPTH: usize = 16;
pub const MAX_PLUGIN_DIMENSIONS: usize = 16;
pub const MAX_PLUGIN_METADATA_ENTRIES: usize = 1024;
pub const MAX_PLUGIN_CHANNEL_NAMES: usize = 4096;
pub const MAX_PLUGIN_MEASUREMENT_ROWS: usize = 10_000;
pub const MAX_PLUGIN_MEASUREMENTS_PER_ROW: usize = 1024;
pub const MAX_PLUGIN_PLANES_PER_INVOCATION: usize = 1_000_000;

/// Pixel encodings supported by the v0.1 plugin contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PluginPixelType {
    U8,
    U16,
    F32,
}

impl PluginPixelType {
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::F32 => 4,
        }
    }
}

impl From<PixelType> for PluginPixelType {
    fn from(value: PixelType) -> Self {
        match value {
            PixelType::U8 => Self::U8,
            PixelType::U16 => Self::U16,
            PixelType::F32 => Self::F32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PluginPlaneScope {
    ActivePlane,
    ZStack,
    AllPlanes,
}

/// Zero-based position of a plane within C/Z/T axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PluginPlanePosition {
    pub channel: u32,
    pub z: u32,
    pub time: u32,
}

/// Non-zero C/Z/T bounds used to validate positions and derive a deterministic schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PluginPlaneBounds {
    channels: u32,
    z: u32,
    time: u32,
}

impl PluginPlaneBounds {
    pub fn new(channels: u32, z: u32, time: u32) -> Result<Self, PluginContractError> {
        if channels == 0 || z == 0 || time == 0 {
            return Err(PluginContractError::EmptyPlaneAxis);
        }
        Ok(Self { channels, z, time })
    }

    pub fn validate_position(
        &self,
        position: PluginPlanePosition,
    ) -> Result<(), PluginContractError> {
        if position.channel >= self.channels || position.z >= self.z || position.time >= self.time {
            return Err(PluginContractError::PlanePositionOutOfBounds { position });
        }
        Ok(())
    }

    pub fn channels(&self) -> u32 {
        self.channels
    }

    pub fn z(&self) -> u32 {
        self.z
    }

    pub fn time(&self) -> u32 {
        self.time
    }
}

/// Host-owned plane order for one invocation.
///
/// Z-stack order increases Z at the active C/T. All-plane order is C-fastest, then Z, then T.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginPlaneSchedule {
    expected: Vec<PluginPlanePosition>,
    next: usize,
}

impl PluginPlaneSchedule {
    pub fn new(
        bounds: PluginPlaneBounds,
        active: PluginPlanePosition,
        scope: PluginPlaneScope,
    ) -> Result<Self, PluginContractError> {
        bounds.validate_position(active)?;
        let count = match scope {
            PluginPlaneScope::ActivePlane => 1_u64,
            PluginPlaneScope::ZStack => u64::from(bounds.z),
            PluginPlaneScope::AllPlanes => u64::from(bounds.channels)
                .checked_mul(u64::from(bounds.z))
                .and_then(|count| count.checked_mul(u64::from(bounds.time)))
                .ok_or(PluginContractError::PlaneCountOverflow)?,
        };
        let count = usize::try_from(count).map_err(|_| PluginContractError::PlaneCountOverflow)?;
        if count > MAX_PLUGIN_PLANES_PER_INVOCATION {
            return Err(PluginContractError::PlaneCountLimit {
                actual: count,
                limit: MAX_PLUGIN_PLANES_PER_INVOCATION,
            });
        }

        let mut expected = Vec::with_capacity(count);
        match scope {
            PluginPlaneScope::ActivePlane => expected.push(active),
            PluginPlaneScope::ZStack => {
                for z in 0..bounds.z {
                    expected.push(PluginPlanePosition { z, ..active });
                }
            }
            PluginPlaneScope::AllPlanes => {
                for time in 0..bounds.time {
                    for z in 0..bounds.z {
                        for channel in 0..bounds.channels {
                            expected.push(PluginPlanePosition { channel, z, time });
                        }
                    }
                }
            }
        }
        Ok(Self { expected, next: 0 })
    }

    pub fn len(&self) -> usize {
        self.expected.len()
    }

    pub fn is_empty(&self) -> bool {
        self.expected.is_empty()
    }

    /// Record one processed plane, rejecting duplicates, omissions, reordering, and unscheduled
    /// positions before any guest output can be committed.
    pub fn record(&mut self, position: PluginPlanePosition) -> Result<(), PluginContractError> {
        let Some(expected) = self.expected.get(self.next).copied() else {
            return Err(PluginContractError::TooManyPlanes {
                expected: self.expected.len(),
            });
        };
        if position != expected {
            return Err(PluginContractError::UnexpectedPlane {
                index: self.next,
                expected,
                actual: position,
            });
        }
        self.next += 1;
        Ok(())
    }

    pub fn finish(&self) -> Result<(), PluginContractError> {
        if self.next != self.expected.len() {
            return Err(PluginContractError::MissingPlanes {
                expected: self.expected.len(),
                processed: self.next,
            });
        }
        Ok(())
    }
}

/// Named capabilities returned by the image-operation dispatcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginOperationCapabilities {
    pixel_types: BTreeSet<PluginPixelType>,
    scopes: BTreeSet<PluginPlaneScope>,
    requires_area_roi: bool,
    accepts_area_mask: bool,
    modifies_pixels: bool,
}

impl PluginOperationCapabilities {
    pub fn new(
        pixel_types: impl IntoIterator<Item = PluginPixelType>,
        scopes: impl IntoIterator<Item = PluginPlaneScope>,
        requires_area_roi: bool,
        accepts_area_mask: bool,
        modifies_pixels: bool,
    ) -> Result<Self, PluginContractError> {
        let mut pixel_type_count = 0;
        let mut validated_pixel_types = BTreeSet::new();
        for pixel_type in pixel_types {
            pixel_type_count += 1;
            validate_count("supported pixel types", pixel_type_count, 3)?;
            if !validated_pixel_types.insert(pixel_type) {
                return Err(PluginContractError::DuplicateCapability);
            }
        }
        let mut scope_count = 0;
        let mut validated_scopes = BTreeSet::new();
        for scope in scopes {
            scope_count += 1;
            validate_count("supported plane scopes", scope_count, 3)?;
            if !validated_scopes.insert(scope) {
                return Err(PluginContractError::DuplicateCapability);
            }
        }
        if validated_pixel_types.is_empty() || validated_scopes.is_empty() {
            return Err(PluginContractError::EmptyCapabilities);
        }
        if requires_area_roi && !accepts_area_mask {
            return Err(PluginContractError::IncoherentRoiCapabilities);
        }
        Ok(Self {
            pixel_types: validated_pixel_types,
            scopes: validated_scopes,
            requires_area_roi,
            accepts_area_mask,
            modifies_pixels,
        })
    }

    pub fn modifies_pixels(&self) -> bool {
        self.modifies_pixels
    }

    pub fn scopes(&self) -> impl ExactSizeIterator<Item = PluginPlaneScope> + '_ {
        self.scopes.iter().copied()
    }

    pub fn requires_area_roi(&self) -> bool {
        self.requires_area_roi
    }

    pub fn accepts_area_mask(&self) -> bool {
        self.accepts_area_mask
    }

    pub fn validate_invocation(
        &self,
        pixel_type: PluginPixelType,
        selected_scope: PluginPlaneScope,
        area_roi_present: bool,
    ) -> Result<(), PluginContractError> {
        if !self.pixel_types.contains(&pixel_type) {
            return Err(PluginContractError::UnsupportedPixelType);
        }
        if !self.scopes.contains(&selected_scope) {
            return Err(PluginContractError::UnsupportedPlaneScope);
        }
        if self.requires_area_roi && !area_roi_present {
            return Err(PluginContractError::RequiredRoiMissing);
        }
        if !self.accepts_area_mask && area_roi_present {
            return Err(PluginContractError::UnexpectedRoiMask);
        }
        Ok(())
    }
}

/// A non-empty rectangular region in full-plane coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PluginRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Validated layout of one tightly packed full-plane buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PluginPlaneLayout {
    width: u32,
    height: u32,
    position: PluginPlanePosition,
    pixel_type: PluginPixelType,
    pixel_bytes: usize,
}

impl PluginPlaneLayout {
    pub fn new(
        width: u32,
        height: u32,
        bounds: PluginPlaneBounds,
        position: PluginPlanePosition,
        pixel_type: PluginPixelType,
        pixel_bytes: usize,
    ) -> Result<Self, PluginContractError> {
        if width == 0 || height == 0 {
            return Err(PluginContractError::EmptyPlane);
        }
        bounds.validate_position(position)?;
        let expected = checked_buffer_len(width, height, pixel_type.bytes_per_pixel())?;
        if pixel_bytes != expected {
            return Err(PluginContractError::PixelLength {
                expected,
                actual: pixel_bytes,
            });
        }
        Ok(Self {
            width,
            height,
            position,
            pixel_type,
            pixel_bytes,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn position(&self) -> PluginPlanePosition {
        self.position
    }

    pub fn pixel_type(&self) -> PluginPixelType {
        self.pixel_type
    }

    pub fn pixel_bytes(&self) -> usize {
        self.pixel_bytes
    }

    /// Enforce the v0.1 same-layout rule and the operation's mutation capability.
    pub fn validate_replacement(
        &self,
        replacement: Option<&Self>,
        capabilities: &PluginOperationCapabilities,
    ) -> Result<(), PluginContractError> {
        let Some(replacement) = replacement else {
            return Ok(());
        };
        if !capabilities.modifies_pixels() {
            return Err(PluginContractError::ReplacementNotAllowed);
        }
        if self != replacement {
            return Err(PluginContractError::ReplacementLayout);
        }
        Ok(())
    }
}

/// Validate the one-byte-per-pixel exact area mask used by the WIT contract.
pub fn validate_roi_mask(
    plane: &PluginPlaneLayout,
    bounds: PluginRegion,
    members: &[u8],
) -> Result<(), PluginContractError> {
    validate_region(plane.width, plane.height, bounds)?;
    let expected = checked_buffer_len(bounds.width, bounds.height, 1)?;
    if members.len() != expected {
        return Err(PluginContractError::MaskLength {
            expected,
            actual: members.len(),
        });
    }
    if let Some((index, value)) = members
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| *value > 1)
    {
        return Err(PluginContractError::InvalidMaskValue { index, value });
    }
    Ok(())
}

fn validate_region(
    image_width: u32,
    image_height: u32,
    region: PluginRegion,
) -> Result<(), PluginContractError> {
    if region.width == 0 || region.height == 0 {
        return Err(PluginContractError::EmptyRegion);
    }
    let right = region
        .x
        .checked_add(region.width)
        .ok_or(PluginContractError::RegionOverflow)?;
    let bottom = region
        .y
        .checked_add(region.height)
        .ok_or(PluginContractError::RegionOverflow)?;
    if right > image_width || bottom > image_height {
        return Err(PluginContractError::RegionOutOfBounds);
    }
    Ok(())
}

fn checked_buffer_len(
    width: u32,
    height: u32,
    bytes_per_pixel: usize,
) -> Result<usize, PluginContractError> {
    let width = usize::try_from(width).map_err(|_| PluginContractError::BufferOverflow)?;
    let height = usize::try_from(height).map_err(|_| PluginContractError::BufferOverflow)?;
    let byte_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(bytes_per_pixel))
        .ok_or(PluginContractError::BufferOverflow)?;
    if byte_len > MAX_PLUGIN_BUFFER_BYTES {
        return Err(PluginContractError::BufferLimit {
            actual: byte_len,
            limit: MAX_PLUGIN_BUFFER_BYTES,
        });
    }
    Ok(byte_len)
}

/// Cumulative budget for all strings, JSON values, collections, and result cells on one side of
/// an invocation.
///
/// A future adapter must retain one output budget across every `process-plane` result and the
/// final `finish` result so staged output cannot multiply this limit by the plane count. Input and
/// output may use separate budgets, and a command-handler invocation uses one budget per side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PluginPayloadBudget {
    bytes: usize,
    nodes: usize,
}

impl PluginPayloadBudget {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn validate_text(
        &mut self,
        field: &'static str,
        text: &str,
    ) -> Result<(), PluginContractError> {
        if text.len() > MAX_PLUGIN_TEXT_BYTES {
            return Err(PluginContractError::TextLimit {
                field,
                actual: text.len(),
                limit: MAX_PLUGIN_TEXT_BYTES,
            });
        }
        self.charge(1, text.len())
    }

    pub fn validate_json_value(
        &mut self,
        field: &'static str,
        source: &str,
    ) -> Result<(), PluginContractError> {
        self.validate_json(field, source, false)
    }

    pub fn validate_json_object(
        &mut self,
        field: &'static str,
        source: &str,
    ) -> Result<(), PluginContractError> {
        self.validate_json(field, source, true)
    }

    fn validate_json(
        &mut self,
        field: &'static str,
        source: &str,
        require_object: bool,
    ) -> Result<(), PluginContractError> {
        if source.len() > MAX_PLUGIN_JSON_BYTES {
            return Err(PluginContractError::JsonLimit {
                field,
                actual: source.len(),
                limit: MAX_PLUGIN_JSON_BYTES,
            });
        }
        let value = serde_json::from_str::<Value>(source)
            .map_err(|_| PluginContractError::InvalidJson { field })?;
        if require_object && !value.is_object() {
            return Err(PluginContractError::JsonObjectRequired { field });
        }
        self.charge(1, source.len())?;

        let mut stack = vec![(&value, 1_usize)];
        while let Some((value, depth)) = stack.pop() {
            if depth > MAX_PLUGIN_JSON_DEPTH {
                return Err(PluginContractError::JsonDepth {
                    field,
                    actual: depth,
                    limit: MAX_PLUGIN_JSON_DEPTH,
                });
            }
            self.charge(1, 0)?;
            match value {
                Value::Array(values) => {
                    for value in values {
                        stack.push((value, depth + 1));
                    }
                }
                Value::Object(values) => {
                    for value in values.values() {
                        stack.push((value, depth + 1));
                    }
                }
                Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
            }
        }
        Ok(())
    }

    pub(super) fn charge(&mut self, nodes: usize, bytes: usize) -> Result<(), PluginContractError> {
        let next_nodes = self
            .nodes
            .checked_add(nodes)
            .ok_or(PluginContractError::PayloadOverflow)?;
        let next_bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or(PluginContractError::PayloadOverflow)?;
        if next_nodes > MAX_PLUGIN_PAYLOAD_NODES {
            return Err(PluginContractError::PayloadNodeLimit {
                actual: next_nodes,
                limit: MAX_PLUGIN_PAYLOAD_NODES,
            });
        }
        if next_bytes > MAX_PLUGIN_PAYLOAD_BYTES {
            return Err(PluginContractError::PayloadByteLimit {
                actual: next_bytes,
                limit: MAX_PLUGIN_PAYLOAD_BYTES,
            });
        }
        self.nodes = next_nodes;
        self.bytes = next_bytes;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PluginAxisKind {
    X,
    Y,
    Z,
    Channel,
    Time,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PluginDimension {
    pub axis: PluginAxisKind,
    pub size: u64,
    pub spacing: Option<f64>,
    pub unit: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginMetadataEntry {
    pub name: String,
    pub value_json: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PluginImageMetadata {
    pub dimensions: Vec<PluginDimension>,
    pub channel_names: Vec<String>,
    pub properties: Vec<PluginMetadataEntry>,
}

impl PluginImageMetadata {
    pub fn validate(&self, budget: &mut PluginPayloadBudget) -> Result<(), PluginContractError> {
        validate_count(
            "metadata dimensions",
            self.dimensions.len(),
            MAX_PLUGIN_DIMENSIONS,
        )?;
        if self.dimensions.is_empty() {
            return Err(PluginContractError::EmptyMetadataDimensions);
        }
        budget.charge(self.dimensions.len(), 0)?;

        let mut known_axes = BTreeSet::new();
        let mut channel_size = None;
        for dimension in &self.dimensions {
            if dimension.size == 0 {
                return Err(PluginContractError::ZeroDimension {
                    axis: dimension.axis,
                });
            }
            if dimension.axis != PluginAxisKind::Unknown && !known_axes.insert(dimension.axis) {
                return Err(PluginContractError::DuplicateAxis {
                    axis: dimension.axis,
                });
            }
            if dimension.axis == PluginAxisKind::Channel {
                channel_size = Some(dimension.size);
            }
            if let Some(spacing) = dimension.spacing {
                let host_spacing = spacing as f32;
                if !spacing.is_finite()
                    || spacing <= 0.0
                    || !host_spacing.is_finite()
                    || host_spacing <= 0.0
                {
                    return Err(PluginContractError::InvalidSpacing {
                        axis: dimension.axis,
                    });
                }
            }
            if let Some(unit) = &dimension.unit {
                budget.validate_text("dimension unit", unit)?;
            }
        }

        validate_count(
            "channel names",
            self.channel_names.len(),
            MAX_PLUGIN_CHANNEL_NAMES,
        )?;
        if !self.channel_names.is_empty() {
            let expected = channel_size.ok_or(PluginContractError::ChannelNamesWithoutAxis)?;
            let expected = usize::try_from(expected)
                .map_err(|_| PluginContractError::ChannelNameCardinality)?;
            if self.channel_names.len() != expected {
                return Err(PluginContractError::ChannelNameCardinality);
            }
        }
        for name in &self.channel_names {
            budget.validate_text("channel name", name)?;
        }

        validate_count(
            "metadata properties",
            self.properties.len(),
            MAX_PLUGIN_METADATA_ENTRIES,
        )?;
        let mut property_names = BTreeSet::new();
        for property in &self.properties {
            validate_name("metadata property", &property.name)?;
            if !property_names.insert(property.name.as_str()) {
                return Err(PluginContractError::DuplicateName {
                    field: "metadata property",
                });
            }
            budget.validate_text("metadata property name", &property.name)?;
            budget.validate_json_value("metadata property value", &property.value_json)?;
        }
        Ok(())
    }

    /// Cross-check a validated plane against this metadata after [`Self::validate`] succeeds.
    pub fn validate_plane_layout(
        &self,
        plane: &PluginPlaneLayout,
    ) -> Result<(), PluginContractError> {
        let mut unknown_axes = self
            .dimensions
            .iter()
            .filter(|dimension| dimension.axis == PluginAxisKind::Unknown);
        if let Some(dimension) = unknown_axes.next() {
            if dimension.size > 1 {
                return Err(PluginContractError::UnsupportedOperationAxis {
                    size: dimension.size,
                });
            }
            if unknown_axes.next().is_some() {
                return Err(PluginContractError::DuplicateUnknownAxis);
            }
        }
        let width = self.required_axis_size(PluginAxisKind::X)?;
        let height = self.required_axis_size(PluginAxisKind::Y)?;
        let channels = self.optional_axis_size(PluginAxisKind::Channel)?;
        let z = self.optional_axis_size(PluginAxisKind::Z)?;
        let time = self.optional_axis_size(PluginAxisKind::Time)?;
        let bounds = PluginPlaneBounds::new(channels, z, time)?;
        bounds.validate_position(plane.position())?;
        if plane.width() != width || plane.height() != height {
            return Err(PluginContractError::PlaneMetadataMismatch);
        }
        Ok(())
    }

    fn required_axis_size(&self, axis: PluginAxisKind) -> Result<u32, PluginContractError> {
        let size = self
            .dimensions
            .iter()
            .find(|dimension| dimension.axis == axis)
            .ok_or(PluginContractError::MissingSpatialAxis { axis })?
            .size;
        u32::try_from(size).map_err(|_| PluginContractError::DimensionTooLarge { axis, size })
    }

    fn optional_axis_size(&self, axis: PluginAxisKind) -> Result<u32, PluginContractError> {
        let Some(size) = self
            .dimensions
            .iter()
            .find(|dimension| dimension.axis == axis)
            .map(|dimension| dimension.size)
        else {
            return Ok(1);
        };
        u32::try_from(size).map_err(|_| PluginContractError::DimensionTooLarge { axis, size })
    }

    /// Validate a finish-time metadata replacement against the invocation input.
    pub fn validate_replacement(
        &self,
        replacement: &Self,
        budget: &mut PluginPayloadBudget,
    ) -> Result<(), PluginContractError> {
        replacement.validate(budget)?;
        let same_shape = self.dimensions.len() == replacement.dimensions.len()
            && self
                .dimensions
                .iter()
                .zip(&replacement.dimensions)
                .all(|(left, right)| left.axis == right.axis && left.size == right.size);
        if !same_shape {
            return Err(PluginContractError::MetadataShapeChanged);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PluginMeasurementValue {
    Number(f64),
    Integer(i64),
    Boolean(bool),
    Text(String),
    Missing,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PluginMeasurement {
    pub column: String,
    pub value: PluginMeasurementValue,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PluginMeasurementRow {
    pub label: Option<String>,
    pub values: Vec<PluginMeasurement>,
}

pub fn validate_measurement_rows(
    rows: &[PluginMeasurementRow],
    budget: &mut PluginPayloadBudget,
) -> Result<(), PluginContractError> {
    validate_count("measurement rows", rows.len(), MAX_PLUGIN_MEASUREMENT_ROWS)?;
    budget.charge(rows.len(), 0)?;
    for row in rows {
        if let Some(label) = &row.label {
            budget.validate_text("measurement label", label)?;
        }
        validate_count(
            "measurements per row",
            row.values.len(),
            MAX_PLUGIN_MEASUREMENTS_PER_ROW,
        )?;
        let mut columns = BTreeSet::new();
        for measurement in &row.values {
            validate_name("measurement column", &measurement.column)?;
            if !columns.insert(measurement.column.as_str()) {
                return Err(PluginContractError::DuplicateName {
                    field: "measurement column",
                });
            }
            budget.validate_text("measurement column", &measurement.column)?;
            budget.charge(1, 0)?;
            match &measurement.value {
                PluginMeasurementValue::Number(value) if !value.is_finite() => {
                    return Err(PluginContractError::NonFiniteMeasurement);
                }
                PluginMeasurementValue::Text(value) => {
                    budget.validate_text("measurement text", value)?;
                }
                PluginMeasurementValue::Number(_)
                | PluginMeasurementValue::Integer(_)
                | PluginMeasurementValue::Boolean(_)
                | PluginMeasurementValue::Missing => {}
            }
        }
    }
    Ok(())
}

fn validate_count(
    field: &'static str,
    actual: usize,
    limit: usize,
) -> Result<(), PluginContractError> {
    if actual > limit {
        return Err(PluginContractError::CollectionLimit {
            field,
            actual,
            limit,
        });
    }
    Ok(())
}

pub(super) fn validate_name(field: &'static str, value: &str) -> Result<(), PluginContractError> {
    if value.is_empty() || value.trim() != value {
        return Err(PluginContractError::InvalidName { field });
    }
    Ok(())
}

/// Progress state owned by one plugin invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PluginProgress {
    last: Option<ProgressSnapshot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProgressSnapshot {
    completed: u64,
    total: Option<u64>,
}

impl PluginProgress {
    /// Accept a monotonic update. A total may first appear after indeterminate progress starts,
    /// but once present it cannot change or disappear.
    pub fn update(
        &mut self,
        completed: u64,
        total: Option<u64>,
        message: Option<&str>,
        budget: &mut PluginPayloadBudget,
    ) -> Result<(), PluginContractError> {
        if let Some(message) = message {
            budget.validate_text("progress message", message)?;
        }
        if total == Some(0) {
            return Err(PluginContractError::ZeroProgressTotal);
        }
        if let Some(total) = total
            && completed > total
        {
            return Err(PluginContractError::ProgressBeyondTotal { completed, total });
        }
        if let Some(previous) = self.last {
            if completed < previous.completed {
                return Err(PluginContractError::ProgressRegressed {
                    previous: previous.completed,
                    next: completed,
                });
            }
            match (previous.total, total) {
                (Some(expected), Some(actual)) if expected != actual => {
                    return Err(PluginContractError::ProgressTotalChanged { expected, actual });
                }
                (Some(_), None) => return Err(PluginContractError::ProgressTotalRemoved),
                _ => {}
            }
        }
        self.last = Some(ProgressSnapshot { completed, total });
        Ok(())
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum PluginContractError {
    #[error("plugin plane axes must all be non-zero")]
    EmptyPlaneAxis,
    #[error("plugin plane position {position:?} lies outside the dataset")]
    PlanePositionOutOfBounds { position: PluginPlanePosition },
    #[error("plugin plane count overflowed the host address space")]
    PlaneCountOverflow,
    #[error("plugin invocation contains {actual} planes, above the {limit}-plane limit")]
    PlaneCountLimit { actual: usize, limit: usize },
    #[error("plugin produced more than the {expected} scheduled planes")]
    TooManyPlanes { expected: usize },
    #[error("plugin plane {index} was {actual:?}; expected scheduled position {expected:?}")]
    UnexpectedPlane {
        index: usize,
        expected: PluginPlanePosition,
        actual: PluginPlanePosition,
    },
    #[error("plugin processed {processed} of {expected} scheduled planes")]
    MissingPlanes { expected: usize, processed: usize },
    #[error("plugin operation must support at least one pixel type and plane scope")]
    EmptyCapabilities,
    #[error("plugin operation capabilities contain a duplicate value")]
    DuplicateCapability,
    #[error("an operation that requires an area ROI must accept an area mask")]
    IncoherentRoiCapabilities,
    #[error("plugin operation does not support the input pixel type")]
    UnsupportedPixelType,
    #[error("plugin operation does not support the selected plane scope")]
    UnsupportedPlaneScope,
    #[error("plugin operation requires an area ROI")]
    RequiredRoiMissing,
    #[error("plugin operation does not accept an area ROI mask")]
    UnexpectedRoiMask,
    #[error("plugin full planes require non-zero dimensions")]
    EmptyPlane,
    #[error("plugin ROI requires non-zero dimensions")]
    EmptyRegion,
    #[error("plugin ROI coordinates overflow")]
    RegionOverflow,
    #[error("plugin ROI lies outside the supplied full plane")]
    RegionOutOfBounds,
    #[error("plugin buffer byte length overflowed the host address space")]
    BufferOverflow,
    #[error("plugin buffer contains {actual} bytes, above the {limit}-byte limit")]
    BufferLimit { actual: usize, limit: usize },
    #[error("plugin pixel buffer has {actual} bytes; expected exactly {expected}")]
    PixelLength { expected: usize, actual: usize },
    #[error("plugin ROI mask has {actual} bytes; expected exactly {expected}")]
    MaskLength { expected: usize, actual: usize },
    #[error("plugin ROI mask byte {index} has invalid value {value}; expected zero or one")]
    InvalidMaskValue { index: usize, value: u8 },
    #[error("plugin returned replacement pixels despite declaring a non-modifying operation")]
    ReplacementNotAllowed,
    #[error("plugin replacement must preserve width, height, position, pixel type, and byte count")]
    ReplacementLayout,
    #[error("plugin text field `{field}` contains {actual} bytes, above the {limit}-byte limit")]
    TextLimit {
        field: &'static str,
        actual: usize,
        limit: usize,
    },
    #[error("plugin JSON field `{field}` contains {actual} bytes, above the {limit}-byte limit")]
    JsonLimit {
        field: &'static str,
        actual: usize,
        limit: usize,
    },
    #[error("plugin field `{field}` is not valid JSON")]
    InvalidJson { field: &'static str },
    #[error("plugin field `{field}` must be a JSON object")]
    JsonObjectRequired { field: &'static str },
    #[error("plugin JSON field `{field}` has depth {actual}, above the {limit}-level limit")]
    JsonDepth {
        field: &'static str,
        actual: usize,
        limit: usize,
    },
    #[error("plugin payload accounting overflowed the host address space")]
    PayloadOverflow,
    #[error("plugin payload has {actual} bytes, above the {limit}-byte limit")]
    PayloadByteLimit { actual: usize, limit: usize },
    #[error("plugin payload has {actual} nodes, above the {limit}-node limit")]
    PayloadNodeLimit { actual: usize, limit: usize },
    #[error("plugin field `{field}` has {actual} entries, above the {limit}-entry limit")]
    CollectionLimit {
        field: &'static str,
        actual: usize,
        limit: usize,
    },
    #[error("plugin metadata requires at least one dimension")]
    EmptyMetadataDimensions,
    #[error("plugin metadata axis {axis:?} has size zero")]
    ZeroDimension { axis: PluginAxisKind },
    #[error("plugin metadata repeats axis {axis:?}")]
    DuplicateAxis { axis: PluginAxisKind },
    #[error("plugin metadata axis {axis:?} has invalid spacing")]
    InvalidSpacing { axis: PluginAxisKind },
    #[error("plugin metadata has channel names without a channel axis")]
    ChannelNamesWithoutAxis,
    #[error("plugin channel-name count does not match the channel dimension")]
    ChannelNameCardinality,
    #[error("plugin field `{field}` must be non-empty and have no surrounding whitespace")]
    InvalidName { field: &'static str },
    #[error("plugin field `{field}` contains duplicate names")]
    DuplicateName { field: &'static str },
    #[error("plugin finish metadata changed dimension axes or sizes")]
    MetadataShapeChanged,
    #[error("plugin metadata is missing required spatial axis {axis:?}")]
    MissingSpatialAxis { axis: PluginAxisKind },
    #[error("plugin metadata axis {axis:?} size {size} exceeds the v0.1 coordinate range")]
    DimensionTooLarge { axis: PluginAxisKind, size: u64 },
    #[error("plugin plane dimensions do not match the invocation metadata")]
    PlaneMetadataMismatch,
    #[error("plugin image operations cannot schedule an unknown axis of size {size}")]
    UnsupportedOperationAxis { size: u64 },
    #[error("plugin image operations accept at most one singleton unknown axis")]
    DuplicateUnknownAxis,
    #[error("plugin measurement numbers must be finite")]
    NonFiniteMeasurement,
    #[error("plugin progress total must be greater than zero")]
    ZeroProgressTotal,
    #[error("plugin progress {completed} exceeds total {total}")]
    ProgressBeyondTotal { completed: u64, total: u64 },
    #[error("plugin progress regressed from {previous} to {next}")]
    ProgressRegressed { previous: u64, next: u64 },
    #[error("plugin progress total changed from {expected} to {actual}")]
    ProgressTotalChanged { expected: u64, actual: u64 },
    #[error("plugin progress removed a previously declared total")]
    ProgressTotalRemoved,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::Path;

    use wit_parser::{Resolve, WorldItem};

    use super::*;
    use crate::plugins::PLUGIN_API_VERSION;

    const ACTIVE: PluginPlanePosition = PluginPlanePosition {
        channel: 1,
        z: 1,
        time: 0,
    };

    fn bounds() -> PluginPlaneBounds {
        PluginPlaneBounds::new(2, 3, 2).unwrap()
    }

    fn capabilities(modifies_pixels: bool) -> PluginOperationCapabilities {
        PluginOperationCapabilities::new(
            [
                PluginPixelType::U8,
                PluginPixelType::U16,
                PluginPixelType::F32,
            ],
            [PluginPlaneScope::ActivePlane, PluginPlaneScope::ZStack],
            false,
            true,
            modifies_pixels,
        )
        .unwrap()
    }

    #[test]
    fn wit_contract_parses_and_matches_the_manifest_api_version() {
        let mut resolve = Resolve::default();
        let (package_id, _) = resolve
            .push_path(Path::new(env!("CARGO_MANIFEST_DIR")).join("wit"))
            .expect("plugin WIT must resolve");
        let package = &resolve.packages[package_id];

        assert_eq!(package.name.namespace, PLUGIN_WIT_NAMESPACE);
        assert_eq!(package.name.name, PLUGIN_WIT_PACKAGE);
        assert_eq!(
            package.name.version.as_ref().map(ToString::to_string),
            Some(PLUGIN_API_VERSION.to_string())
        );
        assert_eq!(
            package
                .interfaces
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["types", "host", "image-operation", "command-handler"]
        );
        assert_eq!(
            package
                .worlds
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            [
                IMAGE_OPERATION_WORLD,
                COMMAND_HANDLER_WORLD,
                COMBINED_PLUGIN_WORLD,
            ]
        );
    }

    #[test]
    fn wit_contract_semantics_match_the_versioned_snapshot() {
        let mut resolve = Resolve::default();
        let (package_id, _) = resolve
            .push_path(Path::new(env!("CARGO_MANIFEST_DIR")).join("wit"))
            .expect("plugin WIT must resolve");
        let mut printer = wit_component::WitPrinter::default();
        printer.emit_docs(false);
        printer
            .print(&resolve, package_id, &[])
            .expect("plugin WIT must normalize");
        let normalized = printer.output.to_string();

        // This stable FNV-1a snapshot covers record fields/order, variants, function signatures,
        // resource ownership, imports, and exports without making doc-only edits break the ABI.
        // A deliberate semantic change requires a compatibility decision and snapshot update.
        assert_eq!(fnv1a64(normalized.as_bytes()), 0x7869_7a70_cb93_4c7c);
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    #[test]
    fn worlds_import_only_local_contract_interfaces() {
        let mut resolve = Resolve::default();
        let (package_id, _) = resolve
            .push_path(Path::new(env!("CARGO_MANIFEST_DIR")).join("wit"))
            .expect("plugin WIT must resolve");
        let package = &resolve.packages[package_id];

        for world_id in package.worlds.values() {
            let world = &resolve.worlds[*world_id];
            let imported_names = world
                .imports
                .values()
                .map(|item| {
                    let id = interface_id(item).expect("world imports must be interfaces only");
                    assert_eq!(resolve.interfaces[id].package, Some(package_id));
                    resolve.interfaces[id].name.as_deref().unwrap()
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(imported_names, BTreeSet::from(["host", "types"]));

            let exported_names = world
                .exports
                .values()
                .map(|item| {
                    let id = interface_id(item).expect("world exports must be interfaces only");
                    assert_eq!(resolve.interfaces[id].package, Some(package_id));
                    resolve.interfaces[id].name.as_deref().unwrap()
                })
                .collect::<BTreeSet<_>>();
            let expected_exports = match world.name.as_str() {
                IMAGE_OPERATION_WORLD => BTreeSet::from(["image-operation"]),
                COMMAND_HANDLER_WORLD => BTreeSet::from(["command-handler"]),
                COMBINED_PLUGIN_WORLD => BTreeSet::from(["command-handler", "image-operation"]),
                name => panic!("unexpected plugin world `{name}`"),
            };
            assert_eq!(exported_names, expected_exports);
        }
    }

    #[test]
    fn operation_contract_has_stateful_plane_lifecycle_and_handler_dispatch() {
        let mut resolve = Resolve::default();
        let (package_id, _) = resolve
            .push_path(Path::new(env!("CARGO_MANIFEST_DIR")).join("wit"))
            .expect("plugin WIT must resolve");
        let package = &resolve.packages[package_id];
        let operation = &resolve.interfaces[package.interfaces["image-operation"]];
        let handler = &resolve.interfaces[package.interfaces["command-handler"]];
        let host = &resolve.interfaces[package.interfaces["host"]];

        assert!(operation.types.contains_key("operation-invocation"));
        assert_eq!(
            operation
                .functions
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            [
                "[method]operation-invocation.process-plane",
                "capabilities",
                "begin",
                "finish"
            ]
        );
        assert_eq!(
            handler
                .functions
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["invoke"]
        );
        assert_eq!(
            host.functions
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["report-progress", "is-cancelled"]
        );
    }

    fn interface_id(item: &WorldItem) -> Option<wit_parser::InterfaceId> {
        match item {
            WorldItem::Interface { id, .. } => Some(*id),
            WorldItem::Function(_) | WorldItem::Type(_) => None,
        }
    }

    #[test]
    fn host_selects_scope_and_schedule_rejects_wrong_or_missing_planes() {
        let mut stack =
            PluginPlaneSchedule::new(bounds(), ACTIVE, PluginPlaneScope::ZStack).unwrap();
        assert_eq!(stack.len(), 3);
        stack
            .record(PluginPlanePosition { z: 0, ..ACTIVE })
            .unwrap();
        assert!(matches!(
            stack.record(PluginPlanePosition { z: 2, ..ACTIVE }),
            Err(PluginContractError::UnexpectedPlane { .. })
        ));
        assert!(matches!(
            stack.finish(),
            Err(PluginContractError::MissingPlanes { .. })
        ));

        let policy = capabilities(true);
        policy
            .validate_invocation(PluginPixelType::F32, PluginPlaneScope::ZStack, true)
            .unwrap();
        assert_eq!(
            policy.validate_invocation(PluginPixelType::F32, PluginPlaneScope::AllPlanes, false,),
            Err(PluginContractError::UnsupportedPlaneScope)
        );
    }

    #[test]
    fn plane_layout_tracks_position_and_rejects_forbidden_or_changed_replacements() {
        let input =
            PluginPlaneLayout::new(10, 8, bounds(), ACTIVE, PluginPixelType::U16, 160).unwrap();
        let moved = PluginPlaneLayout::new(
            10,
            8,
            bounds(),
            PluginPlanePosition { z: 2, ..ACTIVE },
            PluginPixelType::U16,
            160,
        )
        .unwrap();
        assert_eq!(
            input.validate_replacement(Some(&moved), &capabilities(true)),
            Err(PluginContractError::ReplacementLayout)
        );
        assert_eq!(
            input.validate_replacement(Some(&input), &capabilities(false)),
            Err(PluginContractError::ReplacementNotAllowed)
        );
    }

    #[test]
    fn plane_layout_accepts_each_encoding_and_rejects_bad_lengths_and_limits() {
        for (pixel_type, bytes) in [
            (PluginPixelType::U8, 80),
            (PluginPixelType::U16, 160),
            (PluginPixelType::F32, 320),
        ] {
            PluginPlaneLayout::new(10, 8, bounds(), ACTIVE, pixel_type, bytes).unwrap();
        }
        assert_eq!(
            PluginPlaneLayout::new(2, 2, bounds(), ACTIVE, PluginPixelType::F32, 15).unwrap_err(),
            PluginContractError::PixelLength {
                expected: 16,
                actual: 15,
            }
        );
        assert!(matches!(
            PluginPlaneLayout::new(
                u32::MAX,
                u32::MAX,
                bounds(),
                ACTIVE,
                PluginPixelType::F32,
                usize::MAX,
            ),
            Err(PluginContractError::BufferOverflow | PluginContractError::BufferLimit { .. })
        ));
    }

    #[test]
    fn roi_masks_are_plane_bounded_exact_and_binary() {
        let plane =
            PluginPlaneLayout::new(10, 10, bounds(), ACTIVE, PluginPixelType::U8, 100).unwrap();
        let roi = PluginRegion {
            x: 2,
            y: 3,
            width: 2,
            height: 2,
        };
        validate_roi_mask(&plane, roi, &[0, 1, 1, 0]).unwrap();
        assert!(matches!(
            validate_roi_mask(&plane, roi, &[0, 2, 1, 0]),
            Err(PluginContractError::InvalidMaskValue { .. })
        ));
        assert_eq!(
            validate_roi_mask(&plane, PluginRegion { x: 9, ..roi }, &[0; 4]),
            Err(PluginContractError::RegionOutOfBounds)
        );
    }

    #[test]
    fn metadata_json_measurements_and_strings_are_structurally_bounded() {
        let metadata = PluginImageMetadata {
            dimensions: vec![
                PluginDimension {
                    axis: PluginAxisKind::Y,
                    size: 8,
                    spacing: Some(0.5),
                    unit: Some("µm".into()),
                },
                PluginDimension {
                    axis: PluginAxisKind::X,
                    size: 10,
                    spacing: Some(0.5),
                    unit: Some("µm".into()),
                },
                PluginDimension {
                    axis: PluginAxisKind::Channel,
                    size: 2,
                    spacing: None,
                    unit: None,
                },
                PluginDimension {
                    axis: PluginAxisKind::Z,
                    size: 3,
                    spacing: Some(1.0),
                    unit: Some("µm".into()),
                },
                PluginDimension {
                    axis: PluginAxisKind::Time,
                    size: 2,
                    spacing: Some(1.0),
                    unit: Some("s".into()),
                },
            ],
            channel_names: vec!["DAPI".into(), "GFP".into()],
            properties: vec![PluginMetadataEntry {
                name: "display".into(),
                value_json: r#"{"minimum":0,"maximum":255}"#.into(),
            }],
        };
        let mut budget = PluginPayloadBudget::new();
        metadata.validate(&mut budget).unwrap();
        let plane =
            PluginPlaneLayout::new(10, 8, bounds(), ACTIVE, PluginPixelType::F32, 320).unwrap();
        metadata.validate_plane_layout(&plane).unwrap();
        let mismatched_plane =
            PluginPlaneLayout::new(9, 8, bounds(), ACTIVE, PluginPixelType::F32, 288).unwrap();
        assert_eq!(
            metadata.validate_plane_layout(&mismatched_plane),
            Err(PluginContractError::PlaneMetadataMismatch)
        );
        budget
            .validate_json_object("parameters", r#"{"sigma":2.0}"#)
            .unwrap();
        validate_measurement_rows(
            &[PluginMeasurementRow {
                label: Some("plane-1".into()),
                values: vec![PluginMeasurement {
                    column: "Mean".into(),
                    value: PluginMeasurementValue::Number(42.0),
                }],
            }],
            &mut budget,
        )
        .unwrap();

        let mut changed = metadata.clone();
        changed.dimensions[0].size = 9;
        assert_eq!(
            metadata.validate_replacement(&changed, &mut PluginPayloadBudget::new()),
            Err(PluginContractError::MetadataShapeChanged)
        );
        for spacing in [f64::MIN_POSITIVE, 1e300] {
            let mut invalid_calibration = metadata.clone();
            invalid_calibration.dimensions[0].spacing = Some(spacing);
            assert_eq!(
                invalid_calibration.validate(&mut PluginPayloadBudget::new()),
                Err(PluginContractError::InvalidSpacing {
                    axis: PluginAxisKind::Y,
                })
            );
        }
        assert!(matches!(
            PluginPayloadBudget::new()
                .validate_text("status", &"x".repeat(MAX_PLUGIN_TEXT_BYTES + 1),),
            Err(PluginContractError::TextLimit { .. })
        ));
        assert_eq!(
            PluginPayloadBudget::new().validate_json_value("details", "{bad"),
            Err(PluginContractError::InvalidJson { field: "details" })
        );
        assert_eq!(
            validate_measurement_rows(
                &[PluginMeasurementRow {
                    label: None,
                    values: vec![PluginMeasurement {
                        column: "Mean".into(),
                        value: PluginMeasurementValue::Number(f64::NAN),
                    }],
                }],
                &mut PluginPayloadBudget::new(),
            ),
            Err(PluginContractError::NonFiniteMeasurement)
        );

        let mut unschedulable = metadata.clone();
        unschedulable.dimensions.push(PluginDimension {
            axis: PluginAxisKind::Unknown,
            size: 2,
            spacing: None,
            unit: None,
        });
        unschedulable
            .validate(&mut PluginPayloadBudget::new())
            .unwrap();
        assert_eq!(
            unschedulable.validate_plane_layout(&plane),
            Err(PluginContractError::UnsupportedOperationAxis { size: 2 })
        );
    }

    #[test]
    fn output_budget_accumulates_across_plane_and_finish_results() {
        let text = "x".repeat(MAX_PLUGIN_TEXT_BYTES);
        let validate_one_response = |budget: &mut PluginPayloadBudget| {
            for _ in 0..33 {
                budget.validate_text("staged output", &text)?;
            }
            Ok::<_, PluginContractError>(())
        };

        validate_one_response(&mut PluginPayloadBudget::new()).unwrap();
        validate_one_response(&mut PluginPayloadBudget::new()).unwrap();

        let mut invocation_output = PluginPayloadBudget::new();
        validate_one_response(&mut invocation_output).unwrap();
        assert!(matches!(
            validate_one_response(&mut invocation_output),
            Err(PluginContractError::PayloadByteLimit { .. })
        ));
    }

    #[test]
    fn progress_is_monotonic_and_messages_share_the_payload_budget() {
        let mut progress = PluginProgress::default();
        let mut budget = PluginPayloadBudget::new();
        progress
            .update(1, None, Some("starting"), &mut budget)
            .unwrap();
        progress.update(2, Some(4), None, &mut budget).unwrap();
        progress.update(4, Some(4), None, &mut budget).unwrap();

        assert!(matches!(
            progress.update(3, Some(4), None, &mut budget),
            Err(PluginContractError::ProgressRegressed { .. })
        ));

        let mut changed_total = PluginProgress::default();
        changed_total.update(1, Some(3), None, &mut budget).unwrap();
        assert_eq!(
            changed_total.update(2, Some(4), None, &mut budget),
            Err(PluginContractError::ProgressTotalChanged {
                expected: 3,
                actual: 4,
            })
        );
    }
}

// Generating every guest world in ordinary test builds catches WIT constructs that parse but are
// not consumable by the Rust Component Model toolchain. Runtime/host bindings remain milestone 2.
#[cfg(test)]
mod image_operation_guest_bindings {
    wit_bindgen::generate!({
        path: "wit",
        world: "image-operation-plugin",
    });
}

#[cfg(test)]
mod command_handler_guest_bindings {
    wit_bindgen::generate!({
        path: "wit",
        world: "command-handler-plugin",
    });
}

#[cfg(test)]
mod combined_guest_bindings {
    wit_bindgen::generate!({
        path: "wit",
        world: "combined-plugin",
    });
}
