#![cfg(feature = "bioformats")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use image_rs::formats::{
    AssetSnapshot, BioFormatsError, BioformatsPixelType, IoError, PlaneCoordinates, RangeStorage,
    ReadRequest, Rect, Region, StorageResult, materialize_bioformats_plane, open_bioformats_asset,
};
use image_rs::model::{AxisKind, PixelType};

#[derive(Default)]
struct MemoryRangeStorage {
    bytes: HashMap<String, Arc<[u8]>>,
    fail_at_or_after: HashMap<String, u64>,
    named: HashMap<(String, String), AssetSnapshot>,
    siblings: HashMap<String, Vec<AssetSnapshot>>,
    ranges: Mutex<Vec<(String, u64, usize)>>,
}

impl MemoryRangeStorage {
    fn with_asset(mut self, identity: &str, bytes: Vec<u8>) -> Self {
        self.bytes.insert(identity.to_owned(), bytes.into());
        self
    }

    fn failing_at_or_after(mut self, identity: &str, offset: u64) -> Self {
        self.fail_at_or_after.insert(identity.to_owned(), offset);
        self
    }

    fn with_named_asset(
        mut self,
        from_identity: &str,
        identity: &str,
        logical_name: &str,
        bytes: Vec<u8>,
    ) -> Self {
        let asset = AssetSnapshot::new(identity, logical_name, bytes.len() as u64)
            .expect("valid named asset");
        self.bytes.insert(identity.to_owned(), bytes.into());
        self.named
            .insert((from_identity.to_owned(), logical_name.to_owned()), asset);
        self
    }

    fn with_sibling_asset(
        mut self,
        from_identity: &str,
        identity: &str,
        logical_name: &str,
        bytes: Vec<u8>,
    ) -> Self {
        let asset = AssetSnapshot::new(identity, logical_name, bytes.len() as u64)
            .expect("valid sibling asset");
        self.bytes.insert(identity.to_owned(), bytes.into());
        self.siblings
            .entry(from_identity.to_owned())
            .or_default()
            .push(asset);
        self
    }

    fn ranges(&self) -> Vec<(String, u64, usize)> {
        self.ranges.lock().expect("range recorder lock").clone()
    }
}

impl RangeStorage for MemoryRangeStorage {
    fn read_exact_at(
        &self,
        asset: &AssetSnapshot,
        offset: u64,
        destination: &mut [u8],
    ) -> StorageResult<()> {
        self.ranges.lock().expect("range recorder lock").push((
            asset.identity().to_owned(),
            offset,
            destination.len(),
        ));
        if self
            .fail_at_or_after
            .get(asset.identity())
            .is_some_and(|first_failing_offset| offset >= *first_failing_offset)
        {
            return Err(std::io::Error::other("injected range failure").into());
        }
        let bytes = self
            .bytes
            .get(asset.identity())
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "asset not found"))?;
        let start = usize::try_from(offset)?;
        let end = start
            .checked_add(destination.len())
            .ok_or_else(|| std::io::Error::other("range overflow"))?;
        let source = bytes
            .get(start..end)
            .ok_or_else(|| std::io::Error::other("range out of bounds"))?;
        destination.copy_from_slice(source);
        Ok(())
    }

    fn resolve_named(
        &self,
        from: &AssetSnapshot,
        logical_name: &str,
    ) -> StorageResult<Option<AssetSnapshot>> {
        Ok(self
            .named
            .get(&(from.identity().to_owned(), logical_name.to_owned()))
            .cloned())
    }

    fn siblings(&self, from: &AssetSnapshot) -> StorageResult<Vec<AssetSnapshot>> {
        Ok(self
            .siblings
            .get(from.identity())
            .cloned()
            .unwrap_or_default())
    }
}

#[test]
fn range_storage_opens_tiff_lazily_and_keeps_native_region_requests() {
    const PIXEL_OFFSET: usize = 64 * 1024;
    let bytes = padded_tiff(PIXEL_OFFSET);
    let source_len = bytes.len();
    let storage =
        Arc::new(MemoryRangeStorage::default().with_asset("asset:recorded-tiff@v1", bytes));
    let primary = AssetSnapshot::new("asset:recorded-tiff@v1", "recorded.tif", source_len as u64)
        .expect("valid asset snapshot");

    let dataset = open_bioformats_asset(storage.clone(), primary).expect("open range TIFF");

    assert!(dataset.used_files().is_empty());
    assert_eq!(
        dataset.used_sources()[0].identity().as_str(),
        "asset:recorded-tiff@v1"
    );
    let opening_ranges = storage.ranges();
    assert!(!opening_ranges.is_empty());
    assert!(opening_ranges.iter().all(|(_, offset, length)| {
        (*offset as usize) + length <= source_len && *length < source_len
    }));
    assert!(
        opening_ranges
            .iter()
            .all(|(_, offset, _)| *offset < PIXEL_OFFSET as u64)
    );

    let request = ReadRequest::new(0, PlaneCoordinates::new(0, 0, 0))
        .with_resolution(0)
        .with_region(Region::Rect(Rect::new(1, 0, 2, 2).expect("valid region")));
    let info = dataset.plane_info(request).expect("native plane info");
    assert_eq!(info.series, 0);
    assert_eq!(info.resolution, 0);
    assert_eq!(info.coordinates, PlaneCoordinates::new(0, 0, 0));
    assert_eq!(info.layout.pixel_type, BioformatsPixelType::Uint8);
    assert_eq!(info.layout.samples_per_pixel, 1);
    assert_eq!(info.byte_len, 4);

    let plane = dataset.read_plane(request).expect("read native region");
    assert_eq!(plane.bytes(), &[2, 3, 5, 6]);
    assert!(storage.ranges().iter().any(|(_, offset, length)| {
        *offset <= PIXEL_OFFSET as u64 && *offset + *length as u64 >= PIXEL_OFFSET as u64 + 6
    }));
}

#[test]
fn range_storage_resolves_detached_nrrd_by_logical_name() {
    let header =
        b"NRRD0004\ntype: uint8\ndimension: 2\nsizes: 3 2\nencoding: raw\ndata file: pixels.raw\n"
            .to_vec();
    let header_len = header.len();
    let storage = Arc::new(
        MemoryRangeStorage::default()
            .with_asset("asset:nrrd-header@v1", header)
            .with_named_asset(
                "asset:nrrd-header@v1",
                "asset:nrrd-pixels@v1",
                "pixels.raw",
                vec![1, 2, 3, 4, 5, 6],
            ),
    );
    let primary = AssetSnapshot::new("asset:nrrd-header@v1", "dataset.nhdr", header_len as u64)
        .expect("valid NRRD header asset");

    let dataset = open_bioformats_asset(storage, primary).expect("open detached NRRD");

    assert!(dataset.used_files().is_empty());
    assert_eq!(
        dataset
            .used_sources()
            .iter()
            .map(|source| source.identity().as_str())
            .collect::<Vec<_>>(),
        ["asset:nrrd-header@v1", "asset:nrrd-pixels@v1"]
    );
    let plane = dataset
        .read_plane(ReadRequest::new(0, PlaneCoordinates::new(0, 0, 0)))
        .expect("read detached NRRD pixels");
    assert_eq!(plane.bytes(), &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn range_storage_resolves_multi_file_ome_tiff_members_by_logical_name() {
    let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint8"
            SizeX="3" SizeY="2" SizeZ="2" SizeC="1" SizeT="1">
      <Channel ID="Channel:0:0" SamplesPerPixel="1"/>
      <TiffData FirstZ="0" FirstC="0" FirstT="0" IFD="0" PlaneCount="1"
                FileName="plane-0.tif"/>
      <TiffData FirstZ="1" FirstC="0" FirstT="0" IFD="0" PlaneCount="1"
                FileName="plane-1.tif"/>
    </Pixels>
  </Image>
</OME>"#;
    let storage = Arc::new(
        MemoryRangeStorage::default()
            .with_asset("asset:ome-metadata@v1", xml.as_bytes().to_vec())
            .with_named_asset(
                "asset:ome-metadata@v1",
                "asset:ome-plane-0@v1",
                "plane-0.tif",
                basic_tiff_with_pixels([1, 2, 3, 4, 5, 6]),
            )
            .with_named_asset(
                "asset:ome-metadata@v1",
                "asset:ome-plane-1@v1",
                "plane-1.tif",
                basic_tiff_with_pixels([11, 12, 13, 14, 15, 16]),
            ),
    );
    let primary = AssetSnapshot::new(
        "asset:ome-metadata@v1",
        "dataset.companion.ome",
        xml.len() as u64,
    )
    .expect("valid OME metadata asset");

    let dataset = open_bioformats_asset(storage, primary).expect("open multi-file OME-TIFF");

    assert_eq!(
        dataset
            .used_sources()
            .iter()
            .map(|source| source.identity().as_str())
            .collect::<Vec<_>>(),
        [
            "asset:ome-metadata@v1",
            "asset:ome-plane-0@v1",
            "asset:ome-plane-1@v1",
        ]
    );
    let first = dataset
        .read_plane(ReadRequest::new(0, PlaneCoordinates::new(0, 0, 0)))
        .expect("read first OME member");
    let second = dataset
        .read_plane(ReadRequest::new(0, PlaneCoordinates::new(1, 0, 0)))
        .expect("read second OME member");
    assert_eq!(first.bytes(), &[1, 2, 3, 4, 5, 6]);
    assert_eq!(second.bytes(), &[11, 12, 13, 14, 15, 16]);
}

#[test]
fn range_storage_keeps_nonzero_series_requests_explicit() {
    let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint8"
            SizeX="3" SizeY="2" SizeZ="1" SizeC="1" SizeT="1">
      <Channel ID="Channel:0:0" SamplesPerPixel="1"/>
      <TiffData IFD="0" PlaneCount="1" FileName="series-0.tif"/>
    </Pixels>
  </Image>
  <Image ID="Image:1">
    <Pixels ID="Pixels:1" DimensionOrder="XYZCT" Type="uint8"
            SizeX="3" SizeY="2" SizeZ="1" SizeC="1" SizeT="1">
      <Channel ID="Channel:1:0" SamplesPerPixel="1"/>
      <TiffData IFD="0" PlaneCount="1" FileName="series-1.tif"/>
    </Pixels>
  </Image>
</OME>"#;
    let storage = Arc::new(
        MemoryRangeStorage::default()
            .with_asset("asset:ome-series@v1", xml.as_bytes().to_vec())
            .with_named_asset(
                "asset:ome-series@v1",
                "asset:series-0@v1",
                "series-0.tif",
                basic_tiff_with_pixels([1, 2, 3, 4, 5, 6]),
            )
            .with_named_asset(
                "asset:ome-series@v1",
                "asset:series-1@v1",
                "series-1.tif",
                basic_tiff_with_pixels([21, 22, 23, 24, 25, 26]),
            ),
    );
    let primary = AssetSnapshot::new(
        "asset:ome-series@v1",
        "series.companion.ome",
        xml.len() as u64,
    )
    .expect("valid multi-series OME asset");

    let dataset = open_bioformats_asset(storage, primary).expect("open multi-series OME-TIFF");

    assert_eq!(dataset.series().len(), 2);
    let second_series = dataset
        .read_plane(ReadRequest::new(1, PlaneCoordinates::new(0, 0, 0)))
        .expect("read explicit second series");
    assert_eq!(second_series.info().series, 1);
    assert_eq!(second_series.bytes(), &[21, 22, 23, 24, 25, 26]);
}

#[test]
fn range_storage_supplies_complete_split_czi_sibling_set() {
    let master = minimal_czi(0, [1, 2, 3, 4, 5, 6]);
    let master_len = master.len();
    let storage = Arc::new(
        MemoryRangeStorage::default()
            // The set is intentionally out of dataset order and includes the
            // primary; bioformats-rs owns format-specific ordering/de-duplication.
            .with_sibling_asset(
                "asset:czi-master@v1",
                "asset:czi-part-1@v1",
                "sample (1).czi",
                minimal_czi(1, [11, 12, 13, 14, 15, 16]),
            )
            .with_sibling_asset(
                "asset:czi-master@v1",
                "asset:czi-master@v1",
                "sample.czi",
                master,
            ),
    );
    let primary = AssetSnapshot::new("asset:czi-master@v1", "sample.czi", master_len as u64)
        .expect("valid CZI primary asset");

    let dataset = open_bioformats_asset(storage, primary).expect("open split CZI");

    assert_eq!(
        dataset
            .used_sources()
            .iter()
            .map(|source| source.identity().as_str())
            .collect::<Vec<_>>(),
        ["asset:czi-master@v1", "asset:czi-part-1@v1"]
    );
    let first = dataset
        .read_plane(ReadRequest::new(0, PlaneCoordinates::new(0, 0, 0)))
        .expect("read CZI master plane");
    let second = dataset
        .read_plane(ReadRequest::new(0, PlaneCoordinates::new(1, 0, 0)))
        .expect("read CZI part plane");
    assert_eq!(first.bytes(), &[1, 2, 3, 4, 5, 6]);
    assert_eq!(second.bytes(), &[11, 12, 13, 14, 15, 16]);
}

#[test]
fn malformed_declared_range_is_a_structured_recoverable_error() {
    let storage = Arc::new(MemoryRangeStorage::default().with_asset(
        "asset:lying-length@v1",
        basic_tiff_with_pixels([1, 2, 3, 4, 5, 6]),
    ));
    let primary = AssetSnapshot::new("asset:lying-length@v1", "lying.tif", 4096)
        .expect("valid descriptor with backend-inconsistent length");

    let error = open_bioformats_asset(storage, primary)
        .err()
        .expect("lying storage length must fail");

    assert!(matches!(
        error,
        IoError::BioFormats(BioFormatsError::SourceRead { .. })
    ));
}

#[test]
fn lazy_pixel_range_failure_remains_a_structured_source_error() {
    const PIXEL_OFFSET: usize = 64 * 1024;
    let bytes = padded_tiff(PIXEL_OFFSET);
    let len = bytes.len();
    let storage = Arc::new(
        MemoryRangeStorage::default()
            .with_asset("asset:failing-pixels@v1", bytes)
            .failing_at_or_after("asset:failing-pixels@v1", PIXEL_OFFSET as u64),
    );
    let primary = AssetSnapshot::new("asset:failing-pixels@v1", "failing-pixels.tif", len as u64)
        .expect("valid failing TIFF asset");
    let dataset = open_bioformats_asset(storage, primary).expect("metadata open before failure");

    let error = dataset
        .read_plane(ReadRequest::new(0, PlaneCoordinates::new(0, 0, 0)))
        .expect_err("lazy pixel range must fail");

    assert!(matches!(error, BioFormatsError::SourceRead { .. }));
}

#[test]
fn shared_dataset_reads_concurrently_and_preserves_caller_buffer_suffixes() {
    let bytes = basic_tiff_with_pixels([1, 2, 3, 4, 5, 6]);
    let len = bytes.len();
    let storage =
        Arc::new(MemoryRangeStorage::default().with_asset("asset:concurrent-tiff@v1", bytes));
    let primary = AssetSnapshot::new("asset:concurrent-tiff@v1", "concurrent.tif", len as u64)
        .expect("valid concurrent TIFF asset");
    let dataset = Arc::new(open_bioformats_asset(storage, primary).expect("open shared TIFF"));

    let workers = (0..4)
        .map(|_| {
            let dataset = Arc::clone(&dataset);
            std::thread::spawn(move || {
                let mut destination = [0xaa_u8; 8];
                dataset
                    .read_plane_into(
                        ReadRequest::new(0, PlaneCoordinates::new(0, 0, 0)),
                        &mut destination,
                    )
                    .expect("concurrent caller-buffer read");
                destination
            })
        })
        .collect::<Vec<_>>();

    for worker in workers {
        assert_eq!(
            worker.join().expect("range worker"),
            [1, 2, 3, 4, 5, 6, 0xaa, 0xaa]
        );
    }
}

#[test]
fn explicitly_eager_plane_materialization_stays_in_image_rs() {
    let bytes = basic_tiff_with_pixels([1, 2, 3, 4, 5, 6]);
    let len = bytes.len();
    let storage =
        Arc::new(MemoryRangeStorage::default().with_asset("asset:materialized-tiff@v1", bytes));
    let primary = AssetSnapshot::new("asset:materialized-tiff@v1", "materialized.tif", len as u64)
        .expect("valid materialized TIFF asset");
    let dataset = open_bioformats_asset(storage, primary).expect("open materialized TIFF");
    let request = ReadRequest::new(0, PlaneCoordinates::new(0, 0, 0))
        .with_resolution(0)
        .with_region(Region::Rect(Rect::new(1, 0, 2, 2).expect("valid region")));

    let materialized =
        materialize_bioformats_plane(&dataset, request).expect("materialize selected region");

    assert_eq!(materialized.shape(), &[2, 2]);
    assert_eq!(materialized.metadata.pixel_type, PixelType::U8);
    assert_eq!(materialized.metadata.axis_index(AxisKind::Y), Some(0));
    assert_eq!(materialized.metadata.axis_index(AxisKind::X), Some(1));
    assert!(materialized.metadata.source.is_none());
    assert_eq!(
        materialized.data.iter().copied().collect::<Vec<_>>(),
        [2.0, 3.0, 5.0, 6.0]
    );
    assert_eq!(
        materialized.metadata.extras["bioformats_series"],
        serde_json::json!(0)
    );
    assert_eq!(
        materialized.metadata.extras["bioformats_source_identities"],
        serde_json::json!(["asset:materialized-tiff@v1"])
    );
    assert_eq!(
        materialized.metadata.extras["bioformats_format"],
        serde_json::json!("tiff")
    );
    assert_eq!(
        materialized.metadata.extras["bioformats_is_indexed"],
        serde_json::json!(false)
    );
    assert!(
        materialized
            .metadata
            .extras
            .contains_key("bioformats_lookup_table")
    );
    assert!(
        materialized
            .metadata
            .extras
            .contains_key("bioformats_channel_metadata")
    );
}

#[test]
fn materialization_interprets_big_endian_uint16_without_normalization() {
    let mut bytes = b"NRRD0005\n\
type: uint16\n\
dimension: 2\n\
sizes: 2 1\n\
endian: big\n\
encoding: raw\n\
\n"
    .to_vec();
    bytes.extend_from_slice(&[0x12, 0x34, 0xab, 0xcd]);
    let len = bytes.len();
    let storage =
        Arc::new(MemoryRangeStorage::default().with_asset("asset:big-endian-nrrd@v1", bytes));
    let primary = AssetSnapshot::new("asset:big-endian-nrrd@v1", "big-endian.nrrd", len as u64)
        .expect("valid big-endian NRRD asset");
    let dataset = open_bioformats_asset(storage, primary).expect("open big-endian NRRD");
    let request = ReadRequest::new(0, PlaneCoordinates::new(0, 0, 0));

    let info = dataset.plane_info(request).expect("big-endian layout");
    assert_eq!(info.layout.pixel_type, BioformatsPixelType::Uint16);
    assert!(!info.layout.little_endian);
    let materialized =
        materialize_bioformats_plane(&dataset, request).expect("materialize big-endian NRRD");

    assert_eq!(materialized.metadata.pixel_type, PixelType::U16);
    assert_eq!(materialized.shape(), &[1, 2]);
    assert_eq!(
        materialized.data.iter().copied().collect::<Vec<_>>(),
        [4660.0, 43981.0]
    );
}

#[test]
fn materialization_preserves_interleaved_sample_layout_as_channel_axis() {
    let mut bytes = b"NRRD0005\n\
type: uint8\n\
dimension: 3\n\
sizes: 2 2 1\n\
encoding: raw\n\
\n"
    .to_vec();
    bytes.extend_from_slice(&[1, 10, 2, 20]);
    let len = bytes.len();
    let storage = Arc::new(MemoryRangeStorage::default().with_asset("asset:vector-nrrd@v1", bytes));
    let primary = AssetSnapshot::new("asset:vector-nrrd@v1", "vector.nrrd", len as u64)
        .expect("valid vector NRRD asset");
    let dataset = open_bioformats_asset(storage, primary).expect("open vector NRRD");
    let request = ReadRequest::new(0, PlaneCoordinates::new(0, 0, 0));

    let info = dataset.plane_info(request).expect("interleaved layout");
    assert_eq!(info.layout.samples_per_pixel, 2);
    assert!(info.layout.interleaved);
    let materialized =
        materialize_bioformats_plane(&dataset, request).expect("materialize vector NRRD");

    assert_eq!(materialized.shape(), &[1, 2, 2]);
    assert_eq!(materialized.metadata.axis_index(AxisKind::Channel), Some(2));
    assert_eq!(
        materialized.data.iter().copied().collect::<Vec<_>>(),
        [1.0, 10.0, 2.0, 20.0]
    );
}

fn padded_tiff(pixel_offset: usize) -> Vec<u8> {
    tiff_with_pixels_at(pixel_offset, [1, 2, 3, 4, 5, 6])
}

fn basic_tiff_with_pixels(pixels: [u8; 6]) -> Vec<u8> {
    const IFD_OFFSET: usize = 8;
    const TAG_COUNT: usize = 9;
    let pixel_offset = IFD_OFFSET + 2 + TAG_COUNT * 12 + 4;
    tiff_with_pixels_at(pixel_offset, pixels)
}

fn tiff_with_pixels_at(pixel_offset: usize, pixels: [u8; 6]) -> Vec<u8> {
    let width = 3_u32;
    let height = 2_u32;
    let ifd_offset = 8_u32;
    let tag_count = 9_u16;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"II");
    bytes.extend_from_slice(&42_u16.to_le_bytes());
    bytes.extend_from_slice(&ifd_offset.to_le_bytes());
    bytes.extend_from_slice(&tag_count.to_le_bytes());
    push_tag(&mut bytes, 256, 4, 1, width);
    push_tag(&mut bytes, 257, 4, 1, height);
    push_tag(&mut bytes, 258, 3, 1, 8);
    push_tag(&mut bytes, 259, 3, 1, 1);
    push_tag(&mut bytes, 262, 3, 1, 1);
    push_tag(&mut bytes, 273, 4, 1, pixel_offset as u32);
    push_tag(&mut bytes, 277, 3, 1, 1);
    push_tag(&mut bytes, 278, 4, 1, height);
    push_tag(&mut bytes, 279, 4, 1, pixels.len() as u32);
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    bytes.resize(pixel_offset, 0);
    bytes.extend_from_slice(&pixels);
    bytes
}

fn push_tag(bytes: &mut Vec<u8>, tag: u16, field_type: u16, count: u32, value: u32) {
    bytes.extend_from_slice(&tag.to_le_bytes());
    bytes.extend_from_slice(&field_type.to_le_bytes());
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn minimal_czi(z: i32, pixels: [u8; 6]) -> Vec<u8> {
    const SEGMENT_HEADER: usize = 32;
    const FILE_HEADER_BODY: usize = 80;
    const DIRECTORY_HEADER: usize = 128;
    const DIMENSION_COUNT: usize = 5;
    const ENTRY_SIZE: usize = 32 + DIMENSION_COUNT * 20;
    const SUBBLOCK_HEADER: usize = 256;

    let directory_position = SEGMENT_HEADER + FILE_HEADER_BODY;
    let directory_used = DIRECTORY_HEADER + ENTRY_SIZE;
    let subblock_position = directory_position + SEGMENT_HEADER + directory_used;
    let subblock_used = SUBBLOCK_HEADER + pixels.len();
    let mut bytes = vec![0_u8; subblock_position + SEGMENT_HEADER + subblock_used];

    write_czi_segment_header(&mut bytes, 0, b"ZISRAWFILE", FILE_HEADER_BODY as u64);
    bytes[SEGMENT_HEADER + 52..SEGMENT_HEADER + 60]
        .copy_from_slice(&(directory_position as u64).to_le_bytes());

    write_czi_segment_header(
        &mut bytes,
        directory_position,
        b"ZISRAWDIRECTORY",
        directory_used as u64,
    );
    let directory_body = directory_position + SEGMENT_HEADER;
    bytes[directory_body..directory_body + 4].copy_from_slice(&1_i32.to_le_bytes());
    let entry = directory_body + DIRECTORY_HEADER;
    bytes[entry + 2..entry + 6].copy_from_slice(&0_i32.to_le_bytes());
    bytes[entry + 6..entry + 14].copy_from_slice(&(subblock_position as i64).to_le_bytes());
    bytes[entry + 18..entry + 22].copy_from_slice(&0_i32.to_le_bytes());
    bytes[entry + 28..entry + 32].copy_from_slice(&(DIMENSION_COUNT as i32).to_le_bytes());
    for (index, (name, start, size)) in [
        (b"X\0\0\0", 0_i32, 3_i32),
        (b"Y\0\0\0", 0, 2),
        (b"Z\0\0\0", z, 1),
        (b"C\0\0\0", 0, 1),
        (b"T\0\0\0", 0, 1),
    ]
    .into_iter()
    .enumerate()
    {
        let dimension = entry + 32 + index * 20;
        bytes[dimension..dimension + 4].copy_from_slice(name);
        bytes[dimension + 4..dimension + 8].copy_from_slice(&start.to_le_bytes());
        bytes[dimension + 8..dimension + 12].copy_from_slice(&size.to_le_bytes());
        bytes[dimension + 16..dimension + 20].copy_from_slice(&size.to_le_bytes());
    }

    write_czi_segment_header(
        &mut bytes,
        subblock_position,
        b"ZISRAWSUBBLOCK",
        subblock_used as u64,
    );
    let subblock_body = subblock_position + SEGMENT_HEADER;
    bytes[subblock_body + 8..subblock_body + 16]
        .copy_from_slice(&(pixels.len() as u64).to_le_bytes());
    bytes[subblock_body + SUBBLOCK_HEADER..subblock_body + SUBBLOCK_HEADER + pixels.len()]
        .copy_from_slice(&pixels);
    bytes
}

fn write_czi_segment_header(bytes: &mut [u8], offset: usize, kind: &[u8], used: u64) {
    bytes[offset..offset + kind.len()].copy_from_slice(kind);
    bytes[offset + 16..offset + 24].copy_from_slice(&used.to_le_bytes());
    bytes[offset + 24..offset + 32].copy_from_slice(&used.to_le_bytes());
}
