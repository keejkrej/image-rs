use ndarray::Array;

use super::{AxisKind, Dataset, Dim, Metadata, PixelType};

#[test]
fn metadata_roundtrip_json() {
    let mut metadata = Metadata::from_shape(&[4, 5, 3], PixelType::U8);
    metadata.channel_names = vec!["R".into(), "G".into(), "B".into()];
    metadata
        .extras
        .insert("dataset".into(), serde_json::json!("synthetic"));
    let serialized = serde_json::to_string_pretty(&metadata).expect("serialize metadata");
    let restored: Metadata = serde_json::from_str(&serialized).expect("deserialize metadata");
    assert_eq!(restored, metadata);
}

#[test]
fn dataset_rejects_invalid_metadata_shape() {
    let data = Array::from_shape_vec((2, 2), vec![0.0_f32, 1.0, 2.0, 3.0])
        .expect("shape")
        .into_dyn();
    let metadata = Metadata {
        dims: vec![Dim::new(AxisKind::X, 2)],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    assert!(Dataset::new(data, metadata).is_err());
}

#[test]
fn dataset_validates_dimension_sizes() {
    let data = Array::from_shape_vec((2, 2), vec![0.0_f32, 1.0, 2.0, 3.0])
        .expect("shape")
        .into_dyn();
    let metadata = Metadata {
        dims: vec![Dim::new(AxisKind::Y, 2), Dim::new(AxisKind::X, 2)],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");
    assert!(dataset.validate().is_ok());
    assert_eq!(dataset.axis_index(AxisKind::X), Some(1));
    assert_eq!(dataset.axis_index(AxisKind::Y), Some(0));
}
