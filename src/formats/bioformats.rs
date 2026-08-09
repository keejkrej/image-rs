use std::error::Error;
use std::sync::Arc;

pub use bioformats_rs::{
    BioFormatsError, Dataset as BioformatsDataset, FormatId as BioformatsFormatId,
    ImageMetadata as BioformatsImageMetadata, PixelLayout, PixelType as BioformatsPixelType, Plane,
    PlaneCoordinates, PlaneInfo, ReadRequest, Rect, Region, Resolution as BioformatsResolution,
    Series as BioformatsSeries,
};
use bioformats_rs::{
    CompanionReference, CompanionResolver, RandomAccessSource, SourceId, SourceInfo, SourceInput,
};
use ndarray::{ArrayD, IxDyn};
use serde_json::json;

use super::{IoError, Result};
use crate::model::{AxisKind, Dataset, DatasetF32, Dim, Metadata, PixelType};

/// Error returned by application-owned range storage.
pub type StorageError = Box<dyn Error + Send + Sync + 'static>;

/// Result returned by application-owned range storage.
pub type StorageResult<T> = std::result::Result<T, StorageError>;

/// Immutable description of one versioned asset in application-owned storage.
///
/// The identity, logical name, length, and bytes must remain stable while an
/// opened dataset retains this snapshot. The logical name is a format and
/// companion-resolution hint; it is not interpreted as a filesystem path.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AssetSnapshot {
    identity: Arc<str>,
    logical_name: Arc<str>,
    len: u64,
}

impl AssetSnapshot {
    pub fn new(
        identity: impl Into<Arc<str>>,
        logical_name: impl Into<Arc<str>>,
        len: u64,
    ) -> Result<Self> {
        let identity = identity.into();
        let logical_name = logical_name.into();
        if identity.is_empty() {
            return Err(IoError::InvalidAsset("identity must not be empty".into()));
        }
        if logical_name.is_empty() {
            return Err(IoError::InvalidAsset(
                "logical name must not be empty".into(),
            ));
        }
        Ok(Self {
            identity,
            logical_name,
            len,
        })
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn logical_name(&self) -> &str {
        &self.logical_name
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn from_source_info(info: &SourceInfo) -> Self {
        Self {
            identity: Arc::from(info.identity().as_str()),
            logical_name: Arc::from(info.name()),
            len: info.len(),
        }
    }
}

/// Application-owned, immutable exact-range storage.
///
/// `read_exact_at` must fill the complete destination or return an error.
/// Calls may arrive concurrently and in any order.
pub trait RangeStorage: Send + Sync + 'static {
    fn read_exact_at(
        &self,
        asset: &AssetSnapshot,
        offset: u64,
        destination: &mut [u8],
    ) -> StorageResult<()>;

    /// Resolve one exact metadata-declared companion in the asset namespace.
    fn resolve_named(
        &self,
        _from: &AssetSnapshot,
        _logical_name: &str,
    ) -> StorageResult<Option<AssetSnapshot>> {
        Ok(None)
    }

    /// Return the complete candidate set for convention-based split assets.
    fn siblings(&self, _from: &AssetSnapshot) -> StorageResult<Vec<AssetSnapshot>> {
        Ok(Vec::new())
    }
}

struct StorageSource {
    storage: Arc<dyn RangeStorage>,
    asset: AssetSnapshot,
    info: SourceInfo,
}

impl StorageSource {
    fn new(storage: Arc<dyn RangeStorage>, asset: AssetSnapshot) -> Self {
        let info = SourceInfo::new(
            SourceId::new(Arc::clone(&asset.identity)),
            Arc::clone(&asset.logical_name),
            asset.len,
        );
        Self {
            storage,
            asset,
            info,
        }
    }
}

impl RandomAccessSource for StorageSource {
    fn info(&self) -> &SourceInfo {
        &self.info
    }

    fn read_at(&self, offset: u64, destination: &mut [u8]) -> StorageResult<()> {
        self.storage.read_exact_at(&self.asset, offset, destination)
    }
}

struct StorageResolver {
    storage: Arc<dyn RangeStorage>,
}

impl StorageResolver {
    fn source(&self, asset: AssetSnapshot) -> Arc<dyn RandomAccessSource> {
        Arc::new(StorageSource::new(Arc::clone(&self.storage), asset))
    }
}

impl CompanionResolver for StorageResolver {
    fn resolve(
        &self,
        from: &SourceInfo,
        reference: CompanionReference<'_>,
    ) -> StorageResult<Vec<Arc<dyn RandomAccessSource>>> {
        let from = AssetSnapshot::from_source_info(from);
        match reference {
            CompanionReference::Named(logical_name) => Ok(self
                .storage
                .resolve_named(&from, logical_name)?
                .into_iter()
                .map(|asset| self.source(asset))
                .collect()),
            CompanionReference::Siblings => Ok(self
                .storage
                .siblings(&from)?
                .into_iter()
                .map(|asset| self.source(asset))
                .collect()),
            _ => Ok(Vec::new()),
        }
    }
}

/// Open a lazy native Bio-Formats dataset over application-owned range storage.
///
/// Opening reads format metadata and indexes but does not materialize pixel
/// planes. Native `PixelLayout` values and explicit series/resolution/ZCT/region
/// requests remain the interface for subsequent reads.
pub fn open_bioformats_asset(
    storage: Arc<dyn RangeStorage>,
    primary: AssetSnapshot,
) -> Result<BioformatsDataset> {
    let primary: Arc<dyn RandomAccessSource> =
        Arc::new(StorageSource::new(Arc::clone(&storage), primary));
    let resolver: Arc<dyn CompanionResolver> = Arc::new(StorageResolver { storage });
    Ok(bioformats_rs::open_source(
        SourceInput::new(primary).with_companion_resolver(resolver),
    )?)
}

/// Eagerly convert one explicit native plane or region into image-rs's f32 model.
///
/// Values are converted without normalization. Wider integer and floating-point
/// samples may lose precision in image-rs's `f32` model. This allocates only the
/// selected plane/region; opening and native reads remain lazy.
pub fn materialize_bioformats_plane(
    dataset: &BioformatsDataset,
    request: ReadRequest,
) -> Result<DatasetF32> {
    let plane = dataset.read_plane(request)?;
    let info = *plane.info();
    let width = usize::try_from(info.region.width)
        .map_err(|_| IoError::UnsupportedLayout("plane width does not fit usize".into()))?;
    let height = usize::try_from(info.region.height)
        .map_err(|_| IoError::UnsupportedLayout("plane height does not fit usize".into()))?;
    let samples = usize::try_from(info.layout.samples_per_pixel)
        .map_err(|_| IoError::UnsupportedLayout("sample count does not fit usize".into()))?;
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| IoError::UnsupportedLayout("plane pixel count overflows usize".into()))?;
    let sample_count = pixel_count
        .checked_mul(samples)
        .ok_or_else(|| IoError::UnsupportedLayout("plane sample count overflows usize".into()))?;

    let mut values = Vec::new();
    values.try_reserve_exact(sample_count).map_err(|error| {
        IoError::UnsupportedLayout(format!(
            "cannot allocate {sample_count}-sample image-rs plane: {error}"
        ))
    })?;
    for pixel in 0..pixel_count {
        for sample in 0..samples {
            let source_index = if info.layout.interleaved || samples == 1 {
                pixel
                    .checked_mul(samples)
                    .and_then(|index| index.checked_add(sample))
            } else {
                sample
                    .checked_mul(pixel_count)
                    .and_then(|index| index.checked_add(pixel))
            }
            .ok_or_else(|| {
                IoError::UnsupportedLayout("native plane sample index overflows usize".into())
            })?;
            values.push(decode_sample(plane.bytes(), source_index, info.layout)?);
        }
    }

    let mut shape = vec![height, width];
    let mut dims = vec![Dim::new(AxisKind::Y, height), Dim::new(AxisKind::X, width)];
    if samples > 1 {
        shape.push(samples);
        dims.push(Dim::new(AxisKind::Channel, samples));
    }

    let source_metadata =
        dataset.series()[request.series].resolutions()[request.resolution].metadata();
    dims[0].spacing = source_metadata.physical_size_y_um.map(|value| value as f32);
    dims[0].unit = dims[0].spacing.map(|_| "µm".to_owned());
    dims[1].spacing = source_metadata.physical_size_x_um.map(|value| value as f32);
    dims[1].unit = dims[1].spacing.map(|_| "µm".to_owned());

    let pixel_type = match info.layout.pixel_type {
        BioformatsPixelType::Uint8 => PixelType::U8,
        BioformatsPixelType::Uint16 => PixelType::U16,
        _ => PixelType::F32,
    };
    let mut metadata = Metadata {
        dims,
        pixel_type,
        ..Metadata::default()
    };
    if source_metadata.is_rgb && samples == 3 {
        metadata.channel_names = vec!["R".into(), "G".into(), "B".into()];
    }
    metadata
        .extras
        .insert("bioformats_series".into(), json!(request.series));
    metadata
        .extras
        .insert("bioformats_resolution".into(), json!(request.resolution));
    metadata.extras.insert(
        "bioformats_plane_coordinates".into(),
        json!({"z": request.plane.z, "c": request.plane.c, "t": request.plane.t}),
    );
    metadata.extras.insert(
        "bioformats_region".into(),
        json!({
            "x": info.region.x,
            "y": info.region.y,
            "width": info.region.width,
            "height": info.region.height,
        }),
    );
    metadata.extras.insert(
        "bioformats_native_layout".into(),
        json!({
            "pixel_type": format!("{:?}", info.layout.pixel_type),
            "significant_bits": info.layout.significant_bits,
            "samples_per_pixel": info.layout.samples_per_pixel,
            "interleaved": info.layout.interleaved,
            "little_endian": info.layout.little_endian,
        }),
    );
    metadata
        .extras
        .insert("bioformats_format".into(), json!(dataset.format().as_str()));
    metadata.extras.insert(
        "bioformats_dimension_order".into(),
        json!(source_metadata.dimension_order.as_str()),
    );
    metadata
        .extras
        .insert("bioformats_is_rgb".into(), json!(source_metadata.is_rgb));
    metadata.extras.insert(
        "bioformats_is_indexed".into(),
        json!(source_metadata.is_indexed),
    );
    metadata.extras.insert(
        "bioformats_is_false_color".into(),
        json!(source_metadata.is_false_color),
    );
    metadata.extras.insert(
        "bioformats_lookup_table".into(),
        json!(&source_metadata.lookup_table),
    );
    metadata.extras.insert(
        "bioformats_channel_metadata".into(),
        json!(&source_metadata.channel_metadata),
    );
    metadata.extras.insert(
        "bioformats_selected_channel_metadata".into(),
        json!(
            source_metadata
                .channel_metadata
                .get(request.plane.c as usize)
        ),
    );
    metadata.extras.insert(
        "bioformats_source_identities".into(),
        json!(
            dataset
                .used_sources()
                .iter()
                .map(|source| source.identity().as_str())
                .collect::<Vec<_>>()
        ),
    );

    let data = ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|error| {
        IoError::UnsupportedLayout(format!(
            "cannot shape materialized Bio-Formats plane: {error}"
        ))
    })?;
    Ok(Dataset::new(data, metadata)?)
}

fn decode_sample(bytes: &[u8], index: usize, layout: PixelLayout) -> Result<f32> {
    let sample_size = layout.pixel_type.bytes_per_sample();
    if sample_size == 0 {
        return Err(IoError::UnsupportedLayout(
            "packed bit samples cannot be materialized into image-rs".into(),
        ));
    }
    let start = index
        .checked_mul(sample_size)
        .ok_or_else(|| IoError::UnsupportedLayout("sample byte offset overflows usize".into()))?;
    let end = start
        .checked_add(sample_size)
        .ok_or_else(|| IoError::UnsupportedLayout("sample byte range overflows usize".into()))?;
    let sample = bytes.get(start..end).ok_or_else(|| {
        IoError::UnsupportedLayout("native plane is shorter than its PixelLayout".into())
    })?;
    let little_endian = layout.little_endian;

    Ok(match layout.pixel_type {
        BioformatsPixelType::Int8 => f32::from(sample[0] as i8),
        BioformatsPixelType::Uint8 => f32::from(sample[0]),
        BioformatsPixelType::Int16 => f32::from(if little_endian {
            i16::from_le_bytes([sample[0], sample[1]])
        } else {
            i16::from_be_bytes([sample[0], sample[1]])
        }),
        BioformatsPixelType::Uint16 => f32::from(if little_endian {
            u16::from_le_bytes([sample[0], sample[1]])
        } else {
            u16::from_be_bytes([sample[0], sample[1]])
        }),
        BioformatsPixelType::Int32 => {
            (if little_endian {
                i32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]])
            } else {
                i32::from_be_bytes([sample[0], sample[1], sample[2], sample[3]])
            }) as f32
        }
        BioformatsPixelType::Uint32 => {
            (if little_endian {
                u32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]])
            } else {
                u32::from_be_bytes([sample[0], sample[1], sample[2], sample[3]])
            }) as f32
        }
        BioformatsPixelType::Float32 => {
            if little_endian {
                f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]])
            } else {
                f32::from_be_bytes([sample[0], sample[1], sample[2], sample[3]])
            }
        }
        BioformatsPixelType::Float64 => {
            (if little_endian {
                f64::from_le_bytes([
                    sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6],
                    sample[7],
                ])
            } else {
                f64::from_be_bytes([
                    sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6],
                    sample[7],
                ])
            }) as f32
        }
        BioformatsPixelType::Bit => unreachable!("zero-sized packed samples returned above"),
    })
}
