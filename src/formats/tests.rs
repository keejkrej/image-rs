use std::path::Path;
use std::time::Instant;

use image::{ImageBuffer, Luma, Rgb, Rgba};
use ndarray::Array;
use tempfile::tempdir;

use super::{NativeRasterImage, read_dataset, read_native_image, write_dataset};
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
fn png_decode_preserves_integer_sample_values() {
    let dir = tempdir().expect("tempdir");
    let gray8_path = dir.path().join("gray8.png");
    let gray16_path = dir.path().join("gray16.png");

    ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(2, 2, vec![0, 10, 128, 255])
        .expect("gray8")
        .save(&gray8_path)
        .expect("save gray8");
    ImageBuffer::<Luma<u16>, Vec<u16>>::from_vec(2, 2, vec![0, 10, 4096, 65_535])
        .expect("gray16")
        .save(&gray16_path)
        .expect("save gray16");

    let gray8 = read_dataset(&gray8_path).expect("read gray8");
    let gray16 = read_dataset(&gray16_path).expect("read gray16");

    assert_eq!(gray8.metadata.pixel_type, PixelType::U8);
    assert_eq!(
        gray8.data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 10.0, 128.0, 255.0]
    );
    assert_eq!(gray16.metadata.pixel_type, PixelType::U16);
    assert_eq!(
        gray16.data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 10.0, 4096.0, 65_535.0]
    );
}

#[test]
fn tiff_roundtrip_preserves_integer_sample_values() {
    let dir = tempdir().expect("tempdir");
    let u8_path = dir.path().join("u8.tiff");
    let u16_path = dir.path().join("u16.tiff");

    let u8_dataset = Dataset::new(
        Array::from_shape_vec((2, 2), vec![0.0, 10.0, 128.0, 255.0])
            .expect("u8 shape")
            .into_dyn(),
        Metadata::from_shape(&[2, 2], PixelType::U8),
    )
    .expect("u8 dataset");
    let u16_dataset = Dataset::new(
        Array::from_shape_vec((2, 2), vec![0.0, 10.0, 4096.0, 65_535.0])
            .expect("u16 shape")
            .into_dyn(),
        Metadata::from_shape(&[2, 2], PixelType::U16),
    )
    .expect("u16 dataset");

    write_dataset(&u8_path, &u8_dataset).expect("write u8");
    write_dataset(&u16_path, &u16_dataset).expect("write u16");

    let restored_u8 = read_dataset(&u8_path).expect("read u8");
    let restored_u16 = read_dataset(&u16_path).expect("read u16");

    assert_eq!(restored_u8.metadata.pixel_type, PixelType::U8);
    assert_eq!(
        restored_u8.data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 10.0, 128.0, 255.0]
    );
    assert_eq!(restored_u16.metadata.pixel_type, PixelType::U16);
    assert_eq!(
        restored_u16.data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 10.0, 4096.0, 65_535.0]
    );
}

#[test]
fn read_native_image_fast_path_recognizes_common_rasters() {
    let dir = tempdir().expect("tempdir");
    let gray8_path = dir.path().join("gray8.png");
    let gray16_path = dir.path().join("gray16.png");
    let rgba_path = dir.path().join("rgba.png");

    ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(2, 1, vec![0, 255])
        .expect("gray8")
        .save(&gray8_path)
        .expect("save gray8");
    ImageBuffer::<Luma<u16>, Vec<u16>>::from_vec(2, 1, vec![0, 65_535])
        .expect("gray16")
        .save(&gray16_path)
        .expect("save gray16");
    ImageBuffer::<Rgba<u8>, Vec<u8>>::from_vec(1, 1, vec![0, 0, 255, 128])
        .expect("rgba")
        .save(&rgba_path)
        .expect("save rgba");

    let gray8 = read_native_image(&gray8_path).expect("read gray8");
    let gray16 = read_native_image(&gray16_path).expect("read gray16");
    let rgba = read_native_image(&rgba_path).expect("read rgba");

    assert!(matches!(
        gray8,
        Some(NativeRasterImage::Gray8 {
            width: 2,
            height: 1,
            ..
        })
    ));
    assert!(matches!(
        gray16,
        Some(NativeRasterImage::Gray16 {
            width: 2,
            height: 1,
            ..
        })
    ));
    assert!(matches!(
        rgba,
        Some(NativeRasterImage::Rgb8 {
            width: 1,
            height: 1,
            ..
        })
    ));
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

#[test]
#[ignore = "timing harness"]
fn benchmark_native_png_read_path() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bench.png");
    let image =
        ImageBuffer::<Luma<u8>, Vec<u8>>::from_fn(512, 512, |x, y| Luma([((x ^ y) & 0xff) as u8]));
    image.save(&path).expect("save png");

    let start = Instant::now();
    for _ in 0..50 {
        let native = read_native_image(&path).expect("native read");
        assert!(native.is_some());
    }
    let native_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..50 {
        let dataset = read_dataset(&path).expect("dataset read");
        assert_eq!(dataset.shape(), &[512, 512]);
    }
    let dataset_elapsed = start.elapsed();

    eprintln!(
        "native fast path: {:?}, eager dataset path: {:?}",
        native_elapsed, dataset_elapsed
    );
}
