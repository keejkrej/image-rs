//! Dataset conversion at the WebAssembly Component boundary.
//!
//! The adapter keeps the generated Component types inside the private runtime module and uses
//! the public contract validators as the authority for every layout and metadata invariant.

use std::path::PathBuf;

use ndarray::IxDyn;
use thiserror::Error;

use super::bindings::image_rs::plugin::types as wit;
use crate::model::{AxisKind, CoreError, DatasetF32, Dim, Metadata, PixelType};
use crate::plugins::contract::{
    MAX_PLUGIN_CHANNEL_NAMES, MAX_PLUGIN_DIMENSIONS, MAX_PLUGIN_METADATA_ENTRIES, PluginAxisKind,
    PluginContractError, PluginDimension, PluginImageMetadata, PluginMetadataEntry,
    PluginPayloadBudget, PluginPixelType, PluginPlaneBounds, PluginPlaneLayout,
    PluginPlanePosition, PluginPlaneSchedule, PluginPlaneScope, PluginRegion, validate_roi_mask,
};

/// Errors produced while adapting the host dataset to the component contract.
#[derive(Debug, Error)]
pub(super) enum DatasetAdapterError {
    #[error(transparent)]
    Core(#[from] CoreError),

    #[error(transparent)]
    Contract(#[from] PluginContractError),

    #[error(
        "dataset dimension {axis_index} with size {size} cannot be represented in plugin metadata"
    )]
    DimensionSizeOverflow { axis_index: usize, size: usize },

    #[error("the staged dataset shape no longer matches the invocation input")]
    StagedShapeChanged,

    #[error(
        "non-finite value {value} cannot be encoded as {pixel_type:?} at plane {position:?}, pixel ({x}, {y})"
    )]
    NonFiniteIntegerPixel {
        pixel_type: PluginPixelType,
        position: PluginPlanePosition,
        x: u32,
        y: u32,
        value: f32,
    },

    #[error("validated dataset index unexpectedly fell outside the ndarray")]
    DatasetIndexOutOfBounds,

    #[error("metadata property `{name}` could not be serialized as JSON")]
    MetadataSerialization {
        name: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("plugin metadata property `{name}` is not valid JSON")]
    MetadataDeserialization {
        name: String,
        #[source]
        source: serde_json::Error,
    },
}

#[derive(Debug, Clone, Copy)]
struct AxisIndices {
    x: usize,
    y: usize,
    channel: Option<usize>,
    z: Option<usize>,
    time: Option<usize>,
}

/// The validated mapping between arbitrary ndarray axes and C/Z/T full planes.
#[derive(Debug, Clone)]
pub(super) struct DatasetLayout {
    axes: AxisIndices,
    shape: Vec<usize>,
    width: u32,
    height: u32,
    bounds: PluginPlaneBounds,
    pixel_type: PluginPixelType,
    plane_bytes: usize,
    all_plane_positions: Vec<PluginPlanePosition>,
}

impl DatasetLayout {
    pub(super) fn bounds(&self) -> PluginPlaneBounds {
        self.bounds
    }

    pub(super) fn pixel_type(&self) -> PluginPixelType {
        self.pixel_type
    }

    /// Return a fresh contract schedule for the caller-selected scope and active position.
    pub(super) fn schedule(
        &self,
        active: PluginPlanePosition,
        scope: PluginPlaneScope,
    ) -> Result<PluginPlaneSchedule, DatasetAdapterError> {
        Ok(PluginPlaneSchedule::new(self.bounds, active, scope)?)
    }

    /// Derive the exact deterministic plane order for a scoped invocation.
    pub(super) fn plane_positions(
        &self,
        active: PluginPlanePosition,
        scope: PluginPlaneScope,
    ) -> Result<Vec<PluginPlanePosition>, DatasetAdapterError> {
        // Build the authoritative contract schedule first so all bounds and count limits are
        // applied before allocating the position vector.
        let schedule = self.schedule(active, scope)?;
        let mut positions = Vec::with_capacity(schedule.len());
        match scope {
            PluginPlaneScope::ActivePlane => positions.push(active),
            PluginPlaneScope::ZStack => {
                for z in 0..self.bounds.z() {
                    positions.push(PluginPlanePosition { z, ..active });
                }
            }
            PluginPlaneScope::AllPlanes => {
                positions.extend_from_slice(&self.all_plane_positions);
            }
        }
        debug_assert_eq!(positions.len(), schedule.len());
        Ok(positions)
    }

    pub(super) fn plane_layout(
        &self,
        position: PluginPlanePosition,
    ) -> Result<PluginPlaneLayout, DatasetAdapterError> {
        Ok(PluginPlaneLayout::new(
            self.width,
            self.height,
            self.bounds,
            position,
            self.pixel_type,
            self.plane_bytes,
        )?)
    }

    fn validate_staged(&self, staged: &DatasetF32) -> Result<(), DatasetAdapterError> {
        staged.validate()?;
        if staged.shape() != self.shape {
            return Err(DatasetAdapterError::StagedShapeChanged);
        }
        Ok(())
    }

    fn coordinates(&self, position: PluginPlanePosition, x: usize, y: usize) -> Vec<usize> {
        let mut coordinates = vec![0; self.shape.len()];
        coordinates[self.axes.x] = x;
        coordinates[self.axes.y] = y;
        if let Some(axis) = self.axes.channel {
            coordinates[axis] = position.channel as usize;
        }
        if let Some(axis) = self.axes.z {
            coordinates[axis] = position.z as usize;
        }
        if let Some(axis) = self.axes.time {
            coordinates[axis] = position.time as usize;
        }
        coordinates
    }
}

/// A validated, invocation-local view of one host dataset.
pub(super) struct DatasetAdapter<'a> {
    source: &'a DatasetF32,
    layout: DatasetLayout,
    metadata: PluginImageMetadata,
    source_path: Option<PathBuf>,
    source_pixel_type: PixelType,
}

impl<'a> DatasetAdapter<'a> {
    /// Validate a dataset using a fresh input budget.
    #[cfg(test)]
    pub(super) fn new(source: &'a DatasetF32) -> Result<Self, DatasetAdapterError> {
        let mut budget = PluginPayloadBudget::new();
        Self::with_input_budget(source, &mut budget)
    }

    /// Validate a dataset while charging metadata against the invocation's input budget.
    pub(super) fn with_input_budget(
        source: &'a DatasetF32,
        budget: &mut PluginPayloadBudget,
    ) -> Result<Self, DatasetAdapterError> {
        source.validate()?;

        let metadata = host_metadata_to_contract(&source.metadata)?;
        metadata.validate(budget)?;

        let axes = required_axis_indices(&source.metadata)?;
        let width = axis_size_u32(&metadata, PluginAxisKind::X)?;
        let height = axis_size_u32(&metadata, PluginAxisKind::Y)?;
        let channels = optional_axis_size_u32(&metadata, PluginAxisKind::Channel)?;
        let z = optional_axis_size_u32(&metadata, PluginAxisKind::Z)?;
        let time = optional_axis_size_u32(&metadata, PluginAxisKind::Time)?;
        let bounds = PluginPlaneBounds::new(channels, z, time)?;
        let pixel_type = PluginPixelType::from(source.metadata.pixel_type);
        let plane_bytes = checked_plane_bytes(width, height, pixel_type)?;

        let first_layout = PluginPlaneLayout::new(
            width,
            height,
            bounds,
            PluginPlanePosition {
                channel: 0,
                z: 0,
                time: 0,
            },
            pixel_type,
            plane_bytes,
        )?;
        // This also applies the operation-specific singleton-unknown-axis rule.
        metadata.validate_plane_layout(&first_layout)?;

        let all_plane_positions = derive_all_plane_positions(bounds)?;
        let layout = DatasetLayout {
            axes,
            shape: source.shape().to_vec(),
            width,
            height,
            bounds,
            pixel_type,
            plane_bytes,
            all_plane_positions,
        };

        Ok(Self {
            source,
            layout,
            metadata,
            source_path: source.metadata.source.clone(),
            source_pixel_type: source.metadata.pixel_type,
        })
    }

    pub(super) fn layout(&self) -> &DatasetLayout {
        &self.layout
    }

    /// Metadata passed to the component. Host source paths are deliberately absent.
    pub(super) fn image_metadata(&self) -> wit::ImageMetadata {
        contract_metadata_to_wit(&self.metadata)
    }

    /// Create the isolated dataset on which all guest output is staged.
    pub(super) fn staged_dataset(&self) -> DatasetF32 {
        self.source.clone()
    }

    /// Gather one arbitrary-axis ndarray plane into row-major, X-fastest component bytes.
    pub(super) fn encode_plane(
        &self,
        position: PluginPlanePosition,
    ) -> Result<wit::PlaneBuffer, DatasetAdapterError> {
        let validated_layout = self.layout.plane_layout(position)?;
        let mut pixels = Vec::with_capacity(validated_layout.pixel_bytes());
        let width = self.layout.width as usize;
        let height = self.layout.height as usize;

        for y in 0..height {
            for x in 0..width {
                let coordinates = self.layout.coordinates(position, x, y);
                let value = *self
                    .source
                    .data
                    .get(IxDyn(&coordinates))
                    .ok_or(DatasetAdapterError::DatasetIndexOutOfBounds)?;
                encode_sample(
                    value,
                    self.layout.pixel_type,
                    position,
                    x as u32,
                    y as u32,
                    &mut pixels,
                )?;
            }
        }

        // Guard the encoder itself as well as the source metadata.
        PluginPlaneLayout::new(
            self.layout.width,
            self.layout.height,
            self.layout.bounds,
            position,
            self.layout.pixel_type,
            pixels.len(),
        )?;

        Ok(wit::PlaneBuffer {
            width: self.layout.width,
            height: self.layout.height,
            position: plugin_position_to_wit(position),
            sample_type: plugin_pixel_type_to_wit(self.layout.pixel_type),
            pixels,
        })
    }

    /// Validate and scatter one exact-layout replacement into the isolated staged dataset.
    pub(super) fn scatter_replacement(
        &self,
        staged: &mut DatasetF32,
        replacement: &wit::PlaneBuffer,
        area_roi: Option<(PluginRegion, &[u8])>,
    ) -> Result<PluginPlanePosition, DatasetAdapterError> {
        self.layout.validate_staged(staged)?;

        let position = wit_position_to_plugin(replacement.position);
        let actual_layout = PluginPlaneLayout::new(
            replacement.width,
            replacement.height,
            self.layout.bounds,
            position,
            wit_pixel_type_to_plugin(replacement.sample_type),
            replacement.pixels.len(),
        )?;
        let expected_layout = self.layout.plane_layout(position)?;
        if actual_layout != expected_layout {
            return Err(PluginContractError::ReplacementLayout.into());
        }
        if let Some((bounds, members)) = area_roi {
            validate_roi_mask(&expected_layout, bounds, members)?;
        }

        let width = self.layout.width as usize;
        let sample_count = width * self.layout.height as usize;
        for index in 0..sample_count {
            let x = index % width;
            let y = index / width;
            if !area_roi.is_none_or(|(bounds, members)| {
                let x = x as u32;
                let y = y as u32;
                if x < bounds.x
                    || y < bounds.y
                    || x >= bounds.x + bounds.width
                    || y >= bounds.y + bounds.height
                {
                    return false;
                }
                let local_x = (x - bounds.x) as usize;
                let local_y = (y - bounds.y) as usize;
                members[local_y * bounds.width as usize + local_x] == 1
            }) {
                // The staged dataset started as an immutable-source clone. Skipping every zero
                // member and every pixel outside the ROI rectangle defensively restores source
                // pixels even when an untrusted guest changed its full replacement plane.
                continue;
            }
            let value = decode_sample(&replacement.pixels, self.layout.pixel_type, index);
            let coordinates = self.layout.coordinates(position, x, y);
            *staged
                .data
                .get_mut(IxDyn(&coordinates))
                .ok_or(DatasetAdapterError::DatasetIndexOutOfBounds)? = value;
        }
        Ok(position)
    }

    /// Validate and apply finish-time metadata without exposing or replacing host capabilities.
    pub(super) fn apply_finish_metadata(
        &self,
        staged: &mut DatasetF32,
        replacement: &wit::ImageMetadata,
        output_budget: &mut PluginPayloadBudget,
    ) -> Result<(), DatasetAdapterError> {
        self.layout.validate_staged(staged)?;
        preflight_wit_metadata(replacement, output_budget)?;
        let replacement = wit_metadata_to_contract(replacement);
        self.metadata
            .validate_replacement(&replacement, output_budget)?;

        let metadata = contract_metadata_to_host(
            &replacement,
            self.source_path.clone(),
            self.source_pixel_type,
        )?;
        metadata.validate_shape(staged.shape())?;
        staged.metadata = metadata;
        Ok(())
    }
}

/// Bound lifted guest values before cloning them into contract records. The authoritative
/// validator still runs below; this probe prevents a second large host-side copy first.
fn preflight_wit_metadata(
    metadata: &wit::ImageMetadata,
    output_budget: &PluginPayloadBudget,
) -> Result<(), DatasetAdapterError> {
    for (field, actual, limit) in [
        (
            "metadata dimensions",
            metadata.dimensions.len(),
            MAX_PLUGIN_DIMENSIONS,
        ),
        (
            "channel names",
            metadata.channel_names.len(),
            MAX_PLUGIN_CHANNEL_NAMES,
        ),
        (
            "metadata properties",
            metadata.properties.len(),
            MAX_PLUGIN_METADATA_ENTRIES,
        ),
    ] {
        if actual > limit {
            return Err(PluginContractError::CollectionLimit {
                field,
                actual,
                limit,
            }
            .into());
        }
    }

    let mut probe = *output_budget;
    for dimension in &metadata.dimensions {
        if let Some(unit) = &dimension.unit {
            probe.validate_text("dimension unit", unit)?;
        }
    }
    for name in &metadata.channel_names {
        probe.validate_text("channel name", name)?;
    }
    for property in &metadata.properties {
        probe.validate_text("metadata property name", &property.name)?;
        probe.validate_json_value("metadata property value", &property.value_json)?;
    }
    Ok(())
}

fn required_axis_indices(metadata: &Metadata) -> Result<AxisIndices, DatasetAdapterError> {
    // Duplicate known axes have already been rejected by the contract metadata validator.
    // Operation-specific unknown-axis checks run when the first plane layout is cross-checked.
    let x = metadata
        .axis_index(AxisKind::X)
        .ok_or(PluginContractError::MissingSpatialAxis {
            axis: PluginAxisKind::X,
        })?;
    let y = metadata
        .axis_index(AxisKind::Y)
        .ok_or(PluginContractError::MissingSpatialAxis {
            axis: PluginAxisKind::Y,
        })?;
    Ok(AxisIndices {
        x,
        y,
        channel: metadata.axis_index(AxisKind::Channel),
        z: metadata.axis_index(AxisKind::Z),
        time: metadata.axis_index(AxisKind::Time),
    })
}

fn axis_size_u32(
    metadata: &PluginImageMetadata,
    axis: PluginAxisKind,
) -> Result<u32, DatasetAdapterError> {
    let size = metadata
        .dimensions
        .iter()
        .find(|dimension| dimension.axis == axis)
        .ok_or(PluginContractError::MissingSpatialAxis { axis })?
        .size;
    u32::try_from(size).map_err(|_| PluginContractError::DimensionTooLarge { axis, size }.into())
}

fn optional_axis_size_u32(
    metadata: &PluginImageMetadata,
    axis: PluginAxisKind,
) -> Result<u32, DatasetAdapterError> {
    let Some(size) = metadata
        .dimensions
        .iter()
        .find(|dimension| dimension.axis == axis)
        .map(|dimension| dimension.size)
    else {
        return Ok(1);
    };
    u32::try_from(size).map_err(|_| PluginContractError::DimensionTooLarge { axis, size }.into())
}

fn checked_plane_bytes(
    width: u32,
    height: u32,
    pixel_type: PluginPixelType,
) -> Result<usize, DatasetAdapterError> {
    let byte_len = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(pixel_type.bytes_per_pixel()))
        .ok_or(PluginContractError::BufferOverflow)?;
    // The contract constructor owns the byte-limit check.
    PluginPlaneLayout::new(
        width,
        height,
        PluginPlaneBounds::new(1, 1, 1)?,
        PluginPlanePosition {
            channel: 0,
            z: 0,
            time: 0,
        },
        pixel_type,
        byte_len,
    )?;
    Ok(byte_len)
}

fn derive_all_plane_positions(
    bounds: PluginPlaneBounds,
) -> Result<Vec<PluginPlanePosition>, DatasetAdapterError> {
    let active = PluginPlanePosition {
        channel: 0,
        z: 0,
        time: 0,
    };
    let mut schedule = PluginPlaneSchedule::new(bounds, active, PluginPlaneScope::AllPlanes)?;
    let mut positions = Vec::with_capacity(schedule.len());
    for time in 0..bounds.time() {
        for z in 0..bounds.z() {
            for channel in 0..bounds.channels() {
                let position = PluginPlanePosition { channel, z, time };
                schedule.record(position)?;
                positions.push(position);
            }
        }
    }
    schedule.finish()?;
    Ok(positions)
}

fn encode_sample(
    value: f32,
    pixel_type: PluginPixelType,
    position: PluginPlanePosition,
    x: u32,
    y: u32,
    output: &mut Vec<u8>,
) -> Result<(), DatasetAdapterError> {
    match pixel_type {
        PluginPixelType::U8 => {
            let value = quantize_integer_sample(value, u8::MAX as f32, pixel_type, position, x, y)?;
            output.push(value as u8);
        }
        PluginPixelType::U16 => {
            let value =
                quantize_integer_sample(value, u16::MAX as f32, pixel_type, position, x, y)?;
            let encoded = value as u16;
            output.extend_from_slice(&encoded.to_le_bytes());
        }
        PluginPixelType::F32 => output.extend_from_slice(&value.to_bits().to_le_bytes()),
    }
    Ok(())
}

fn quantize_integer_sample(
    value: f32,
    maximum: f32,
    pixel_type: PluginPixelType,
    position: PluginPlanePosition,
    x: u32,
    y: u32,
) -> Result<f32, DatasetAdapterError> {
    if value.is_finite() {
        return Ok(value.clamp(0.0, maximum).round());
    }
    Err(DatasetAdapterError::NonFiniteIntegerPixel {
        pixel_type,
        position,
        x,
        y,
        value,
    })
}

fn decode_sample(bytes: &[u8], pixel_type: PluginPixelType, index: usize) -> f32 {
    match pixel_type {
        PluginPixelType::U8 => f32::from(bytes[index]),
        PluginPixelType::U16 => {
            let offset = index * 2;
            f32::from(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
        }
        PluginPixelType::F32 => {
            let offset = index * 4;
            f32::from_bits(u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]))
        }
    }
}

#[cfg(test)]
fn decode_plane(bytes: &[u8], pixel_type: PluginPixelType) -> Vec<f32> {
    let samples = bytes.len() / pixel_type.bytes_per_pixel();
    (0..samples)
        .map(|index| decode_sample(bytes, pixel_type, index))
        .collect()
}

fn host_metadata_to_contract(
    metadata: &Metadata,
) -> Result<PluginImageMetadata, DatasetAdapterError> {
    let dimensions = metadata
        .dims
        .iter()
        .enumerate()
        .map(|(axis_index, dimension)| {
            Ok(PluginDimension {
                axis: host_axis_to_plugin(dimension.axis),
                size: u64::try_from(dimension.size).map_err(|_| {
                    DatasetAdapterError::DimensionSizeOverflow {
                        axis_index,
                        size: dimension.size,
                    }
                })?,
                spacing: dimension.spacing.map(f64::from),
                unit: dimension.unit.clone(),
            })
        })
        .collect::<Result<Vec<_>, DatasetAdapterError>>()?;
    let properties = metadata
        .extras
        .iter()
        .map(|(name, value)| {
            let value_json = serde_json::to_string(value).map_err(|source| {
                DatasetAdapterError::MetadataSerialization {
                    name: name.clone(),
                    source,
                }
            })?;
            Ok(PluginMetadataEntry {
                name: name.clone(),
                value_json,
            })
        })
        .collect::<Result<Vec<_>, DatasetAdapterError>>()?;
    Ok(PluginImageMetadata {
        dimensions,
        channel_names: metadata.channel_names.clone(),
        properties,
    })
}

fn contract_metadata_to_host(
    metadata: &PluginImageMetadata,
    source: Option<PathBuf>,
    pixel_type: PixelType,
) -> Result<Metadata, DatasetAdapterError> {
    let dims = metadata
        .dimensions
        .iter()
        .map(|dimension| {
            Ok(Dim {
                axis: plugin_axis_to_host(dimension.axis),
                size: usize::try_from(dimension.size).map_err(|_| {
                    PluginContractError::DimensionTooLarge {
                        axis: dimension.axis,
                        size: dimension.size,
                    }
                })?,
                spacing: dimension.spacing.map(|spacing| spacing as f32),
                unit: dimension.unit.clone(),
            })
        })
        .collect::<Result<Vec<_>, DatasetAdapterError>>()?;
    let extras = metadata
        .properties
        .iter()
        .map(|property| {
            let value = serde_json::from_str(&property.value_json).map_err(|source| {
                DatasetAdapterError::MetadataDeserialization {
                    name: property.name.clone(),
                    source,
                }
            })?;
            Ok((property.name.clone(), value))
        })
        .collect::<Result<_, DatasetAdapterError>>()?;
    Ok(Metadata {
        dims,
        pixel_type,
        channel_names: metadata.channel_names.clone(),
        source,
        extras,
    })
}

fn contract_metadata_to_wit(metadata: &PluginImageMetadata) -> wit::ImageMetadata {
    wit::ImageMetadata {
        dimensions: metadata
            .dimensions
            .iter()
            .map(|dimension| wit::Dimension {
                axis: plugin_axis_to_wit(dimension.axis),
                size: dimension.size,
                spacing: dimension.spacing,
                unit: dimension.unit.clone(),
            })
            .collect(),
        channel_names: metadata.channel_names.clone(),
        properties: metadata
            .properties
            .iter()
            .map(|property| wit::MetadataEntry {
                name: property.name.clone(),
                value_json: property.value_json.clone(),
            })
            .collect(),
    }
}

fn wit_metadata_to_contract(metadata: &wit::ImageMetadata) -> PluginImageMetadata {
    PluginImageMetadata {
        dimensions: metadata
            .dimensions
            .iter()
            .map(|dimension| PluginDimension {
                axis: wit_axis_to_plugin(dimension.axis),
                size: dimension.size,
                spacing: dimension.spacing,
                unit: dimension.unit.clone(),
            })
            .collect(),
        channel_names: metadata.channel_names.clone(),
        properties: metadata
            .properties
            .iter()
            .map(|property| PluginMetadataEntry {
                name: property.name.clone(),
                value_json: property.value_json.clone(),
            })
            .collect(),
    }
}

fn host_axis_to_plugin(axis: AxisKind) -> PluginAxisKind {
    match axis {
        AxisKind::X => PluginAxisKind::X,
        AxisKind::Y => PluginAxisKind::Y,
        AxisKind::Z => PluginAxisKind::Z,
        AxisKind::Channel => PluginAxisKind::Channel,
        AxisKind::Time => PluginAxisKind::Time,
        AxisKind::Unknown => PluginAxisKind::Unknown,
    }
}

fn plugin_axis_to_host(axis: PluginAxisKind) -> AxisKind {
    match axis {
        PluginAxisKind::X => AxisKind::X,
        PluginAxisKind::Y => AxisKind::Y,
        PluginAxisKind::Z => AxisKind::Z,
        PluginAxisKind::Channel => AxisKind::Channel,
        PluginAxisKind::Time => AxisKind::Time,
        PluginAxisKind::Unknown => AxisKind::Unknown,
    }
}

fn plugin_axis_to_wit(axis: PluginAxisKind) -> wit::AxisKind {
    match axis {
        PluginAxisKind::X => wit::AxisKind::X,
        PluginAxisKind::Y => wit::AxisKind::Y,
        PluginAxisKind::Z => wit::AxisKind::Z,
        PluginAxisKind::Channel => wit::AxisKind::Channel,
        PluginAxisKind::Time => wit::AxisKind::Time,
        PluginAxisKind::Unknown => wit::AxisKind::Unknown,
    }
}

fn wit_axis_to_plugin(axis: wit::AxisKind) -> PluginAxisKind {
    match axis {
        wit::AxisKind::X => PluginAxisKind::X,
        wit::AxisKind::Y => PluginAxisKind::Y,
        wit::AxisKind::Z => PluginAxisKind::Z,
        wit::AxisKind::Channel => PluginAxisKind::Channel,
        wit::AxisKind::Time => PluginAxisKind::Time,
        wit::AxisKind::Unknown => PluginAxisKind::Unknown,
    }
}

pub(super) fn plugin_position_to_wit(position: PluginPlanePosition) -> wit::PlanePosition {
    wit::PlanePosition {
        channel: position.channel,
        z: position.z,
        time: position.time,
    }
}

pub(super) fn wit_position_to_plugin(position: wit::PlanePosition) -> PluginPlanePosition {
    PluginPlanePosition {
        channel: position.channel,
        z: position.z,
        time: position.time,
    }
}

pub(super) fn plugin_pixel_type_to_wit(pixel_type: PluginPixelType) -> wit::PixelType {
    match pixel_type {
        PluginPixelType::U8 => wit::PixelType::Uint8,
        PluginPixelType::U16 => wit::PixelType::Uint16,
        PluginPixelType::F32 => wit::PixelType::Float32,
    }
}

pub(super) fn wit_pixel_type_to_plugin(pixel_type: wit::PixelType) -> PluginPixelType {
    match pixel_type {
        wit::PixelType::Uint8 => PluginPixelType::U8,
        wit::PixelType::Uint16 => PluginPixelType::U16,
        wit::PixelType::Float32 => PluginPixelType::F32,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ndarray::{ArrayD, IxDyn};
    use serde_json::json;

    use super::*;

    fn arbitrary_axis_dataset(pixel_type: PixelType) -> DatasetF32 {
        // C, Y, T, X, Z: the adapter must not rely on conventional axis order.
        let shape = [2, 2, 2, 3, 2];
        let mut data = ArrayD::zeros(IxDyn(&shape));
        for channel in 0..shape[0] {
            for y in 0..shape[1] {
                for time in 0..shape[2] {
                    for x in 0..shape[3] {
                        for z in 0..shape[4] {
                            data[IxDyn(&[channel, y, time, x, z])] =
                                (channel * 10_000 + time * 1_000 + z * 100 + y * 10 + x) as f32;
                        }
                    }
                }
            }
        }
        let mut metadata = Metadata {
            dims: vec![
                Dim::new(AxisKind::Channel, shape[0]),
                Dim::new(AxisKind::Y, shape[1]),
                Dim::new(AxisKind::Time, shape[2]),
                Dim::new(AxisKind::X, shape[3]),
                Dim::new(AxisKind::Z, shape[4]),
            ],
            pixel_type,
            channel_names: vec!["first".into(), "second".into()],
            source: Some(PathBuf::from("/secret/source.tif")),
            extras: Default::default(),
        };
        metadata.dims[1].spacing = Some(0.5);
        metadata.dims[1].unit = Some("um".into());
        DatasetF32::new(data, metadata).unwrap()
    }

    #[test]
    fn gathers_arbitrary_axes_in_contract_plane_order_and_hides_source() {
        let dataset = arbitrary_axis_dataset(PixelType::F32);
        let adapter = DatasetAdapter::new(&dataset).unwrap();
        let expected_positions = (0..2)
            .flat_map(|time| {
                (0..2).flat_map(move |z| {
                    (0..2).map(move |channel| PluginPlanePosition { channel, z, time })
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(adapter.layout().all_plane_positions, expected_positions);
        assert_eq!(adapter.image_metadata().dimensions.len(), 5);

        let position = PluginPlanePosition {
            channel: 1,
            z: 1,
            time: 1,
        };
        let mut plane = adapter.encode_plane(position).unwrap();
        let values = decode_plane(&plane.pixels, PluginPixelType::F32);
        assert_eq!(
            values,
            vec![11_100.0, 11_101.0, 11_102.0, 11_110.0, 11_111.0, 11_112.0]
        );

        plane.pixels = (0..6)
            .flat_map(|value| (20_000.0_f32 + value as f32).to_bits().to_le_bytes())
            .collect();
        let mut staged = adapter.staged_dataset();
        adapter
            .scatter_replacement(&mut staged, &plane, None)
            .unwrap();
        for y in 0..2 {
            for x in 0..3 {
                assert_eq!(
                    staged.data[IxDyn(&[1, y, 1, x, 1])],
                    20_000.0 + (y * 3 + x) as f32
                );
            }
        }
        assert_eq!(staged.data[IxDyn(&[0, 0, 1, 0, 1])], 1_100.0);
    }

    #[test]
    fn scoped_schedules_and_exact_masks_are_host_authoritative() {
        let dataset = arbitrary_axis_dataset(PixelType::F32);
        let adapter = DatasetAdapter::new(&dataset).unwrap();
        let active = PluginPlanePosition {
            channel: 1,
            z: 1,
            time: 1,
        };
        assert_eq!(
            adapter
                .layout()
                .plane_positions(active, PluginPlaneScope::ActivePlane)
                .unwrap(),
            vec![active]
        );
        assert_eq!(
            adapter
                .layout()
                .plane_positions(active, PluginPlaneScope::ZStack)
                .unwrap(),
            vec![
                PluginPlanePosition { z: 0, ..active },
                PluginPlanePosition { z: 1, ..active },
            ]
        );

        let mut replacement = adapter.encode_plane(active).unwrap();
        replacement.pixels = [20_000.0_f32; 6]
            .into_iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        let bounds = PluginRegion {
            x: 1,
            y: 0,
            width: 2,
            height: 2,
        };
        // Members are (1,0) and (2,1); the two zero members and every pixel outside the
        // rectangle must remain source-owned even though the replacement changed all six.
        let members = [1, 0, 0, 1];
        let mut staged = adapter.staged_dataset();
        adapter
            .scatter_replacement(&mut staged, &replacement, Some((bounds, &members)))
            .unwrap();
        assert_eq!(staged.data[IxDyn(&[1, 0, 1, 0, 1])], 11_100.0);
        assert_eq!(staged.data[IxDyn(&[1, 0, 1, 1, 1])], 20_000.0);
        assert_eq!(staged.data[IxDyn(&[1, 0, 1, 2, 1])], 11_102.0);
        assert_eq!(staged.data[IxDyn(&[1, 1, 1, 0, 1])], 11_110.0);
        assert_eq!(staged.data[IxDyn(&[1, 1, 1, 1, 1])], 11_111.0);
        assert_eq!(staged.data[IxDyn(&[1, 1, 1, 2, 1])], 20_000.0);
    }

    #[test]
    fn integer_planes_quantize_finite_samples_and_reject_non_finite_values() {
        let data = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0, 1.0, 128.0, 255.0]).unwrap();
        let metadata = Metadata {
            dims: vec![Dim::new(AxisKind::Y, 1), Dim::new(AxisKind::X, 4)],
            pixel_type: PixelType::U8,
            ..Metadata::default()
        };
        let dataset = DatasetF32::new(data, metadata).unwrap();
        let adapter = DatasetAdapter::new(&dataset).unwrap();
        assert_eq!(
            adapter
                .encode_plane(PluginPlanePosition {
                    channel: 0,
                    z: 0,
                    time: 0,
                })
                .unwrap()
                .pixels,
            vec![0, 1, 128, 255]
        );

        let mut invalid = dataset;
        invalid.data[IxDyn(&[0, 2])] = f32::NAN;
        let adapter = DatasetAdapter::new(&invalid).unwrap();
        assert!(matches!(
            adapter.encode_plane(PluginPlanePosition {
                channel: 0,
                z: 0,
                time: 0,
            }),
            Err(DatasetAdapterError::NonFiniteIntegerPixel { x: 2, .. })
        ));

        invalid.data[IxDyn(&[0, 2])] = 1.5;
        let adapter = DatasetAdapter::new(&invalid).unwrap();
        assert_eq!(
            adapter
                .encode_plane(PluginPlanePosition {
                    channel: 0,
                    z: 0,
                    time: 0,
                })
                .unwrap()
                .pixels,
            vec![0, 1, 2, 255]
        );

        invalid.data[IxDyn(&[0, 2])] = 256.0;
        invalid.data[IxDyn(&[0, 1])] = -4.0;
        let adapter = DatasetAdapter::new(&invalid).unwrap();
        assert_eq!(
            adapter
                .encode_plane(PluginPlanePosition {
                    channel: 0,
                    z: 0,
                    time: 0,
                })
                .unwrap()
                .pixels,
            vec![0, 0, 255, 255]
        );

        invalid.data[IxDyn(&[0, 2])] = f32::INFINITY;
        let adapter = DatasetAdapter::new(&invalid).unwrap();
        assert!(matches!(
            adapter.encode_plane(PluginPlanePosition {
                channel: 0,
                z: 0,
                time: 0,
            }),
            Err(DatasetAdapterError::NonFiniteIntegerPixel { x: 2, .. })
        ));

        let data =
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0x1234 as f32, u16::MAX as f32]).unwrap();
        let metadata = Metadata {
            dims: vec![Dim::new(AxisKind::Y, 1), Dim::new(AxisKind::X, 2)],
            pixel_type: PixelType::U16,
            ..Metadata::default()
        };
        let dataset = DatasetF32::new(data, metadata).unwrap();
        let adapter = DatasetAdapter::new(&dataset).unwrap();
        assert_eq!(
            adapter
                .encode_plane(PluginPlanePosition {
                    channel: 0,
                    z: 0,
                    time: 0,
                })
                .unwrap()
                .pixels,
            vec![0x34, 0x12, 0xff, 0xff]
        );
    }

    #[test]
    fn float_planes_preserve_bits_and_exact_replacements_scatter() {
        let special_nan = f32::from_bits(0x7fc0_1234);
        let data = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![special_nan, -0.0]).unwrap();
        let metadata = Metadata {
            dims: vec![Dim::new(AxisKind::Y, 1), Dim::new(AxisKind::X, 2)],
            pixel_type: PixelType::F32,
            ..Metadata::default()
        };
        let dataset = DatasetF32::new(data, metadata).unwrap();
        let adapter = DatasetAdapter::new(&dataset).unwrap();
        let position = PluginPlanePosition {
            channel: 0,
            z: 0,
            time: 0,
        };
        let encoded = adapter.encode_plane(position).unwrap();
        assert_eq!(&encoded.pixels[..4], &special_nan.to_bits().to_le_bytes());
        assert_eq!(&encoded.pixels[4..], &(-0.0_f32).to_bits().to_le_bytes());

        let mut replacement = encoded;
        replacement.pixels = [3.25_f32, -7.5_f32]
            .into_iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        let mut staged = adapter.staged_dataset();
        adapter
            .scatter_replacement(&mut staged, &replacement, None)
            .unwrap();
        assert_eq!(staged.data.as_slice().unwrap(), &[3.25, -7.5]);

        replacement.width += 1;
        assert!(matches!(
            adapter.scatter_replacement(&mut staged, &replacement, None),
            Err(DatasetAdapterError::Contract(
                PluginContractError::PixelLength { .. } | PluginContractError::ReplacementLayout
            ))
        ));
    }

    #[test]
    fn finish_metadata_is_validated_and_preserves_host_only_fields() {
        let mut dataset = arbitrary_axis_dataset(PixelType::U16);
        dataset
            .metadata
            .extras
            .insert("original".into(), json!({ "value": 1 }));
        let adapter = DatasetAdapter::new(&dataset).unwrap();
        let mut replacement = adapter.image_metadata();
        replacement.dimensions[1].spacing = Some(2.5);
        replacement.channel_names = vec!["red".into(), "green".into()];
        replacement.properties = vec![wit::MetadataEntry {
            name: "guest".into(),
            value_json: "{\"ok\":true}".into(),
        }];

        let mut staged = adapter.staged_dataset();
        let mut budget = PluginPayloadBudget::new();
        adapter
            .apply_finish_metadata(&mut staged, &replacement, &mut budget)
            .unwrap();
        assert_eq!(staged.metadata.source, dataset.metadata.source);
        assert_eq!(staged.metadata.pixel_type, PixelType::U16);
        assert_eq!(staged.metadata.dims[1].spacing, Some(2.5));
        assert_eq!(staged.metadata.extras["guest"], json!({ "ok": true }));

        replacement.dimensions[3].size += 1;
        assert!(matches!(
            adapter.apply_finish_metadata(&mut staged, &replacement, &mut budget),
            Err(DatasetAdapterError::Contract(
                PluginContractError::MetadataShapeChanged
            ))
        ));
    }

    #[test]
    fn rejects_missing_spatial_axes_and_non_singleton_unknown_axes() {
        let data = ArrayD::zeros(IxDyn(&[2, 2]));
        let missing_x = Metadata {
            dims: vec![Dim::new(AxisKind::Y, 2), Dim::new(AxisKind::Z, 2)],
            ..Metadata::default()
        };
        let dataset = DatasetF32::new(data.clone(), missing_x).unwrap();
        assert!(matches!(
            DatasetAdapter::new(&dataset),
            Err(DatasetAdapterError::Contract(
                PluginContractError::MissingSpatialAxis {
                    axis: PluginAxisKind::X
                }
            ))
        ));

        let unknown = Metadata {
            dims: vec![
                Dim::new(AxisKind::Y, 2),
                Dim::new(AxisKind::X, 1),
                Dim::new(AxisKind::Unknown, 2),
            ],
            ..Metadata::default()
        };
        let dataset = DatasetF32::new(ArrayD::zeros(IxDyn(&[2, 1, 2])), unknown).unwrap();
        assert!(matches!(
            DatasetAdapter::new(&dataset),
            Err(DatasetAdapterError::Contract(
                PluginContractError::UnsupportedOperationAxis { size: 2 }
            ))
        ));
    }
}
