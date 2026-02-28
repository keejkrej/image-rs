use std::path::Path;

use image::{ImageBuffer, Luma, Rgb};
use ndarray::Array;
use tempfile::tempdir;

use super::{read_dataset, write_dataset};
use crate::model::{AxisKind, Dataset, Dim, Metadata, PixelType};

#[test]
fn tiff_roundtrip_preserves_shape_and_type() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("roundtrip.tiff");
    let values = vec![0.0_f32, 0.25, 0.5, 1.0];
    let data = Array::from_shape_vec((2, 2), values)
        .expect("shape")
        .into_dyn();
    let metadata = Metadata {
        dims: vec![Dim::new(AxisKind::Y, 2), Dim::new(AxisKind::X, 2)],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");
    write_dataset(&path, &dataset).expect("write tiff");
    let restored = read_dataset(&path).expect("read tiff");
    assert_eq!(restored.shape(), &[2, 2]);
    assert_eq!(restored.metadata.pixel_type, PixelType::F32);
}

#[test]
fn png_and_jpeg_decode_channel_mapping() {
    let dir = tempdir().expect("tempdir");
    let png_path = dir.path().join("color.png");
    let jpg_path = dir.path().join("color.jpg");
    let mut image = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(2, 1);
    image.put_pixel(0, 0, Rgb([255, 0, 0]));
    image.put_pixel(1, 0, Rgb([0, 255, 0]));
    image.save(&png_path).expect("save png");
    image.save(&jpg_path).expect("save jpg");

    let png = read_dataset(&png_path).expect("read png");
    let jpg = read_dataset(&jpg_path).expect("read jpg");
    assert_eq!(png.shape(), &[1, 2, 3]);
    assert_eq!(jpg.shape(), &[1, 2, 3]);
    assert_eq!(png.metadata.axis_index(AxisKind::Channel), Some(2));
    assert_eq!(jpg.metadata.axis_index(AxisKind::Channel), Some(2));
}

#[test]
fn conversion_between_formats() {
    let dir = tempdir().expect("tempdir");
    let input = dir.path().join("input.png");
    let output = dir.path().join("output.tiff");
    let image =
        ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(2, 2, vec![0, 50, 100, 255]).expect("image");
    image.save(&input).expect("save input");
    let dataset = read_dataset(&input).expect("read input");
    write_dataset(&output, &dataset).expect("write output");
    let restored = read_dataset(&output).expect("read output");
    assert_eq!(restored.shape(), &[2, 2]);
}

#[test]
fn unsupported_layout_errors() {
    let dir = tempdir().expect("tempdir");
    let output = dir.path().join("bad.png");
    let data = Array::zeros((2, 2, 2, 2)).into_dyn();
    let metadata = Metadata::from_shape(&[2, 2, 2, 2], PixelType::F32);
    let dataset = Dataset::new(data, metadata).expect("dataset");
    let err = write_dataset(Path::new(&output), &dataset).expect_err("must fail");
    let message = err.to_string();
    assert!(message.contains("expects [Y, X] or [Y, X, C]"));
}
