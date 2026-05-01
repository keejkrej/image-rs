use ndarray::{Array, IxDyn};
use serde_json::json;

use crate::model::{AxisKind, Dataset, Dim, Metadata, PixelType};

use super::{execute_operation, list_operations};

fn test_dataset(values: Vec<f32>, shape: (usize, usize)) -> Dataset<f32> {
    let data = Array::from_shape_vec(shape, values)
        .expect("shape")
        .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, shape.0),
            Dim::new(AxisKind::X, shape.1),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    Dataset::new(data, metadata).expect("dataset")
}

#[test]
fn contains_required_operations() {
    let names = list_operations()
        .into_iter()
        .map(|schema| schema.name)
        .collect::<Vec<_>>();
    assert!(names.contains(&"gaussian.blur".to_string()));
    assert!(names.contains(&"intensity.invert".to_string()));
    assert!(names.contains(&"intensity.math".to_string()));
    assert!(names.contains(&"intensity.nan_background".to_string()));
    assert!(names.contains(&"image.convert".to_string()));
    assert!(names.contains(&"image.resize".to_string()));
    assert!(names.contains(&"image.canvas_resize".to_string()));
    assert!(names.contains(&"image.coordinates".to_string()));
    assert!(names.contains(&"image.bin".to_string()));
    assert!(names.contains(&"image.flip".to_string()));
    assert!(names.contains(&"image.median_filter".to_string()));
    assert!(names.contains(&"image.remove_nans".to_string()));
    assert!(names.contains(&"image.remove_outliers".to_string()));
    assert!(names.contains(&"image.rotate_90".to_string()));
    assert!(names.contains(&"image.rotate".to_string()));
    assert!(names.contains(&"image.translate".to_string()));
    assert!(names.contains(&"image.rank_filter".to_string()));
    assert!(names.contains(&"image.rank_filter_3d".to_string()));
    assert!(names.contains(&"image.sharpen".to_string()));
    assert!(names.contains(&"image.swap_quadrants".to_string()));
    assert!(names.contains(&"image.fft_power_spectrum".to_string()));
    assert!(names.contains(&"image.fft_bandpass".to_string()));
    assert!(names.contains(&"image.convolve".to_string()));
    assert!(names.contains(&"image.unsharp_mask".to_string()));
    assert!(names.contains(&"image.find_edges".to_string()));
    assert!(names.contains(&"image.shadow".to_string()));
    assert!(names.contains(&"image.shadow_demo".to_string()));
    assert!(names.contains(&"components.label".to_string()));
    assert!(names.contains(&"noise.gaussian".to_string()));
    assert!(names.contains(&"noise.salt_and_pepper".to_string()));
    assert!(names.contains(&"measurements.summary".to_string()));
    assert!(names.contains(&"morphology.binary_median".to_string()));
    assert!(names.contains(&"morphology.distance_map".to_string()));
    assert!(names.contains(&"morphology.ultimate_points".to_string()));
    assert!(names.contains(&"morphology.watershed".to_string()));
    assert!(names.contains(&"morphology.voronoi".to_string()));
    assert!(names.contains(&"morphology.fill_holes".to_string()));
    assert!(names.contains(&"morphology.outline".to_string()));
    assert!(names.contains(&"morphology.skeletonize".to_string()));
    #[cfg(feature = "morpholib")]
    assert!(names.contains(&"morpholibj.distance.chamfer".to_string()));
    #[cfg(feature = "thunderstorm")]
    assert!(names.contains(&"thunderstorm.pipeline.localize".to_string()));
}

#[test]
fn intensity_invert_uses_integer_normalized_range() {
    let mut dataset = test_dataset(vec![0.0, 0.25, 0.75, 1.0], (2, 2));
    dataset.metadata.pixel_type = PixelType::U8;

    let output = execute_operation("intensity.invert", &dataset, &json!({})).expect("invert");
    let values = output.dataset.data.iter().copied().collect::<Vec<_>>();
    assert_eq!(values, vec![1.0, 0.75, 0.25, 0.0]);
    assert_eq!(output.dataset.metadata.pixel_type, PixelType::U8);
}

#[test]
fn intensity_invert_uses_float_display_range() {
    let dataset = test_dataset(vec![2.0, 4.0, 6.0, 8.0], (2, 2));

    let output = execute_operation(
        "intensity.invert",
        &dataset,
        &json!({
            "min": 0.0,
            "max": 10.0
        }),
    )
    .expect("invert");

    let values = output.dataset.data.iter().copied().collect::<Vec<_>>();
    assert_eq!(values, vec![8.0, 6.0, 4.0, 2.0]);
}

#[test]
fn intensity_math_applies_binary_operations() {
    let dataset = test_dataset(vec![0.2, 0.4, 0.6, 0.8], (2, 2));

    let added = execute_operation(
        "intensity.math",
        &dataset,
        &json!({
            "operation": "add",
            "value": 0.1
        }),
    )
    .expect("add");
    let values = added.dataset.data.iter().copied().collect::<Vec<_>>();
    for (actual, expected) in values.iter().zip([0.3, 0.5, 0.7, 0.9]) {
        assert!((actual - expected).abs() < 1.0e-6);
    }

    let clipped_max = execute_operation(
        "intensity.math",
        &dataset,
        &json!({
            "operation": "max",
            "value": 0.5
        }),
    )
    .expect("max");
    assert_eq!(
        clipped_max.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![0.2, 0.4, 0.5, 0.5]
    );
}

#[test]
fn intensity_math_applies_unary_operations() {
    let dataset = test_dataset(vec![-2.0, 0.0, 4.0, 9.0], (2, 2));

    let abs =
        execute_operation("intensity.math", &dataset, &json!({"operation": "abs"})).expect("abs");
    assert_eq!(
        abs.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![2.0, 0.0, 4.0, 9.0]
    );

    let sqrt =
        execute_operation("intensity.math", &dataset, &json!({"operation": "sqrt"})).expect("sqrt");
    assert_eq!(
        sqrt.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 0.0, 2.0, 3.0]
    );
}

#[test]
fn intensity_math_applies_bitwise_operations_to_integer_pixels() {
    let mut dataset = test_dataset(
        vec![
            0b1111_0000 as f32 / 255.0,
            0b0000_1111 as f32 / 255.0,
            0b1010_1010 as f32 / 255.0,
            0b0101_0101 as f32 / 255.0,
        ],
        (2, 2),
    );
    dataset.metadata.pixel_type = PixelType::U8;

    let and_output = execute_operation(
        "intensity.math",
        &dataset,
        &json!({
            "operation": "and",
            "value": "11110000"
        }),
    )
    .expect("and");
    let samples = and_output
        .dataset
        .data
        .iter()
        .map(|value| (value * 255.0).round() as u8)
        .collect::<Vec<_>>();
    assert_eq!(samples, vec![0b1111_0000, 0, 0b1010_0000, 0b0101_0000]);

    let xor_output = execute_operation(
        "intensity.math",
        &dataset,
        &json!({
            "operation": "xor",
            "value": 0b1111_1111
        }),
    )
    .expect("xor");
    let samples = xor_output
        .dataset
        .data
        .iter()
        .map(|value| (value * 255.0).round() as u8)
        .collect::<Vec<_>>();
    assert_eq!(
        samples,
        vec![0b0000_1111, 0b1111_0000, 0b0101_0101, 0b1010_1010]
    );
}

#[test]
fn intensity_math_validates_parameters() {
    let dataset = test_dataset(vec![0.2, 0.4, 0.6, 0.8], (2, 2));

    let missing_value = execute_operation(
        "intensity.math",
        &dataset,
        &json!({"operation": "multiply"}),
    )
    .expect_err("missing value");
    assert!(missing_value.to_string().contains("value"));

    let bad_gamma = execute_operation(
        "intensity.math",
        &dataset,
        &json!({
            "operation": "gamma",
            "value": 10.0
        }),
    )
    .expect_err("bad gamma");
    assert!(bad_gamma.to_string().contains("gamma"));

    let unsupported_bitwise = execute_operation(
        "intensity.math",
        &dataset,
        &json!({"operation": "or", "value": 1}),
    )
    .expect_err("bitwise on f32");
    assert!(unsupported_bitwise.to_string().contains("u8 or u16"));
}

#[test]
fn intensity_nan_background_sets_outside_threshold_to_nan() {
    let dataset = test_dataset(vec![0.1, 0.3, 0.6, 0.9], (2, 2));

    let output = execute_operation(
        "intensity.nan_background",
        &dataset,
        &json!({
            "lower": 0.25,
            "upper": 0.75
        }),
    )
    .expect("nan background");

    assert!(output.dataset.data[IxDyn(&[0, 0])].is_nan());
    assert_eq!(output.dataset.data[IxDyn(&[0, 1])], 0.3);
    assert_eq!(output.dataset.data[IxDyn(&[1, 0])], 0.6);
    assert!(output.dataset.data[IxDyn(&[1, 1])].is_nan());
}

#[test]
fn intensity_nan_background_requires_f32_and_valid_thresholds() {
    let mut dataset = test_dataset(vec![0.1, 0.3, 0.6, 0.9], (2, 2));
    dataset.metadata.pixel_type = PixelType::U8;
    let error = execute_operation(
        "intensity.nan_background",
        &dataset,
        &json!({
            "lower": 0.25,
            "upper": 0.75
        }),
    )
    .expect_err("u8 rejected");
    assert!(error.to_string().contains("f32"));

    let dataset = test_dataset(vec![0.1, 0.3, 0.6, 0.9], (2, 2));
    let error = execute_operation(
        "intensity.nan_background",
        &dataset,
        &json!({
            "lower": 0.75,
            "upper": 0.25
        }),
    )
    .expect_err("invalid threshold");
    assert!(error.to_string().contains("upper"));
}

#[test]
fn image_resize_changes_xy_shape() {
    let dataset = test_dataset(vec![0.0, 1.0, 0.5, 0.25], (2, 2));
    let output = execute_operation("image.resize", &dataset, &json!({"width": 4, "height": 3}))
        .expect("resize");
    assert_eq!(output.dataset.shape(), &[3, 4]);
}

#[test]
fn image_coordinates_updates_spacing_units_and_origin_metadata() {
    let dataset = test_dataset(
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0,
        ],
        (2, 4),
    );

    let output = execute_operation(
        "image.coordinates",
        &dataset,
        &json!({
            "left": 10.0,
            "right": 18.0,
            "top": 20.0,
            "bottom": 24.0,
            "x_unit": "um",
            "y_unit": "<same as x unit>"
        }),
    )
    .expect("coordinates");

    assert_eq!(output.dataset.shape(), &[2, 4]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        dataset.data.iter().copied().collect::<Vec<_>>()
    );
    assert_eq!(output.dataset.metadata.dims[1].spacing, Some(2.0));
    assert_eq!(output.dataset.metadata.dims[0].spacing, Some(2.0));
    assert_eq!(output.dataset.metadata.dims[1].unit.as_deref(), Some("um"));
    assert_eq!(output.dataset.metadata.dims[0].unit.as_deref(), Some("um"));
    assert_eq!(
        output.dataset.metadata.extras.get("x_origin_coordinate"),
        Some(&json!(10.0))
    );
    assert_eq!(
        output.dataset.metadata.extras.get("y_origin_coordinate"),
        Some(&json!(20.0))
    );
}

#[test]
fn image_coordinates_rejects_incomplete_or_degenerate_bounds() {
    let dataset = test_dataset(vec![1.0, 2.0, 3.0, 4.0], (2, 2));

    let incomplete = execute_operation("image.coordinates", &dataset, &json!({"left": 1.0}))
        .expect_err("incomplete bounds");
    assert!(incomplete.to_string().contains("left"));

    let degenerate = execute_operation(
        "image.coordinates",
        &dataset,
        &json!({"left": 1.0, "right": 1.0}),
    )
    .expect_err("degenerate bounds");
    assert!(degenerate.to_string().contains("right"));
}

#[test]
fn image_convert_rgb_adds_channel_axis() {
    let dataset = test_dataset(vec![0.0, 1.0, 0.5, 0.25], (2, 2));
    let output =
        execute_operation("image.convert", &dataset, &json!({"target": "rgb"})).expect("convert");
    assert_eq!(output.dataset.shape(), &[2, 2, 3]);
}

#[test]
fn image_bin_reduces_xy_by_average() {
    let dataset = test_dataset(
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ],
        (4, 4),
    );

    let output = execute_operation(
        "image.bin",
        &dataset,
        &json!({
            "x": 2,
            "y": 2,
            "method": "average"
        }),
    )
    .expect("bin");

    assert_eq!(output.dataset.shape(), &[2, 2]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![3.5, 5.5, 11.5, 13.5]
    );
}

#[test]
fn image_bin_reduces_xyz_and_scales_spacing() {
    let data = Array::from_shape_vec(
        (2, 2, 4),
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let mut metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 2),
            Dim::new(AxisKind::X, 2),
            Dim::new(AxisKind::Z, 4),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    metadata.dims[0].spacing = Some(2.0);
    metadata.dims[1].spacing = Some(3.0);
    metadata.dims[2].spacing = Some(4.0);
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.bin",
        &dataset,
        &json!({
            "x": 2,
            "y": 1,
            "z": 2,
            "method": "sum"
        }),
    )
    .expect("bin");

    assert_eq!(output.dataset.shape(), &[2, 1, 2]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![14.0, 22.0, 46.0, 54.0]
    );
    assert_eq!(output.dataset.metadata.dims[0].spacing, Some(2.0));
    assert_eq!(output.dataset.metadata.dims[1].spacing, Some(6.0));
    assert_eq!(output.dataset.metadata.dims[2].spacing, Some(8.0));
}

#[test]
fn image_flip_mirrors_xy_planes() {
    let dataset = test_dataset(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));

    let horizontal = execute_operation(
        "image.flip",
        &dataset,
        &json!({
            "axis": "horizontal"
        }),
    )
    .expect("horizontal flip");
    assert_eq!(horizontal.dataset.shape(), &[2, 3]);
    assert_eq!(
        horizontal.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]
    );

    let vertical = execute_operation(
        "image.flip",
        &dataset,
        &json!({
            "axis": "vertical"
        }),
    )
    .expect("vertical flip");
    assert_eq!(
        vertical.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn image_flip_reverses_z_stack() {
    let data = Array::from_shape_vec(
        (2, 2, 3),
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
            10.0, 11.0, 12.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 2),
            Dim::new(AxisKind::X, 2),
            Dim::new(AxisKind::Z, 3),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.flip",
        &dataset,
        &json!({
            "axis": "z"
        }),
    )
    .expect("z flip");

    assert_eq!(output.dataset.shape(), &[2, 2, 3]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![
            3.0, 2.0, 1.0, //
            6.0, 5.0, 4.0, //
            9.0, 8.0, 7.0, //
            12.0, 11.0, 10.0,
        ]
    );
}

#[test]
fn image_rotate_90_swaps_xy_shape_and_values() {
    let mut dataset = test_dataset(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    dataset.metadata.dims[0].spacing = Some(2.0);
    dataset.metadata.dims[0].unit = Some("row".to_string());
    dataset.metadata.dims[1].spacing = Some(0.5);
    dataset.metadata.dims[1].unit = Some("col".to_string());

    let right = execute_operation(
        "image.rotate_90",
        &dataset,
        &json!({
            "direction": "right"
        }),
    )
    .expect("rotate right");
    assert_eq!(right.dataset.shape(), &[3, 2]);
    assert_eq!(
        right.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]
    );
    assert_eq!(right.dataset.metadata.dims[0].spacing, Some(0.5));
    assert_eq!(right.dataset.metadata.dims[0].unit.as_deref(), Some("col"));
    assert_eq!(right.dataset.metadata.dims[1].spacing, Some(2.0));
    assert_eq!(right.dataset.metadata.dims[1].unit.as_deref(), Some("row"));

    let left = execute_operation(
        "image.rotate_90",
        &dataset,
        &json!({
            "direction": "left"
        }),
    )
    .expect("rotate left");
    assert_eq!(left.dataset.shape(), &[3, 2]);
    assert_eq!(
        left.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![3.0, 6.0, 2.0, 5.0, 1.0, 4.0]
    );
}

#[test]
fn image_rotate_arbitrary_matches_clockwise_right_angle() {
    let dataset = test_dataset(
        vec![
            0.0, 1.0, 2.0, //
            3.0, 4.0, 5.0, //
            6.0, 7.0, 8.0,
        ],
        (3, 3),
    );

    let output = execute_operation(
        "image.rotate",
        &dataset,
        &json!({
            "angle": 90.0,
            "interpolation": "nearest",
            "fill": -1.0
        }),
    )
    .expect("rotate");

    assert_eq!(output.dataset.shape(), &[3, 3]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![
            6.0, 3.0, 0.0, //
            7.0, 4.0, 1.0, //
            8.0, 5.0, 2.0,
        ]
    );
}

#[test]
fn image_rotate_arbitrary_supports_bilinear_interpolation() {
    let dataset = test_dataset(
        vec![
            0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
        (3, 3),
    );

    let output = execute_operation(
        "image.rotate",
        &dataset,
        &json!({
            "angle": 45.0,
            "interpolation": "bilinear",
            "fill": 0.0
        }),
    )
    .expect("rotate");

    assert_eq!(output.dataset.shape(), &[3, 3]);
    let values = output.dataset.data.iter().copied().collect::<Vec<_>>();
    assert!((values[4] - 1.0).abs() < 1.0e-6);
}

#[test]
fn image_rotate_arbitrary_can_enlarge_canvas() {
    let dataset = test_dataset(
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0,
        ],
        (2, 4),
    );

    let output = execute_operation(
        "image.rotate",
        &dataset,
        &json!({
            "angle": 90.0,
            "interpolation": "nearest",
            "enlarge": true
        }),
    )
    .expect("rotate");

    assert_eq!(output.dataset.shape(), &[4, 2]);
    assert_eq!(output.dataset.metadata.dims[0].size, 4);
    assert_eq!(output.dataset.metadata.dims[1].size, 2);
}

#[test]
fn image_rotate_arbitrary_rejects_unknown_interpolation() {
    let dataset = test_dataset(vec![1.0], (1, 1));

    let error = execute_operation(
        "image.rotate",
        &dataset,
        &json!({
            "interpolation": "cubic"
        }),
    )
    .expect_err("invalid interpolation");

    assert!(error.to_string().contains("unsupported interpolation"));
}

#[test]
fn image_translate_offsets_xy_with_fill() {
    let dataset = test_dataset(
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ],
        (3, 3),
    );

    let output = execute_operation(
        "image.translate",
        &dataset,
        &json!({
            "x": 1.0,
            "y": 1.0,
            "fill": -1.0
        }),
    )
    .expect("translate");

    assert_eq!(output.dataset.shape(), &[3, 3]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![
            -1.0, -1.0, -1.0, //
            -1.0, 1.0, 2.0, //
            -1.0, 4.0, 5.0,
        ]
    );
}

#[test]
fn image_translate_supports_bilinear_interpolation() {
    let dataset = test_dataset(vec![1.0, 3.0, 5.0, 7.0], (2, 2));

    let output = execute_operation(
        "image.translate",
        &dataset,
        &json!({
            "x": 0.5,
            "y": 0.0,
            "interpolation": "bilinear",
            "fill": 0.0
        }),
    )
    .expect("translate");

    let values = output.dataset.data.iter().copied().collect::<Vec<_>>();
    assert!((values[1] - 2.0).abs() < 1.0e-6);
    assert!((values[3] - 6.0).abs() < 1.0e-6);
}

#[test]
fn image_median_filter_removes_despeckle_impulse() {
    let dataset = test_dataset(
        vec![
            0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
        (3, 3),
    );

    let output = execute_operation(
        "image.median_filter",
        &dataset,
        &json!({
            "radius": 1.0
        }),
    )
    .expect("median filter");

    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0,
        ]
    );
}

#[test]
fn image_median_filter_processes_xy_planes_independently() {
    let data = Array::from_shape_vec(
        (3, 3, 2),
        vec![
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, //
            0.0, 1.0, 1.0, 0.0, 0.0, 1.0, //
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 3),
            Dim::new(AxisKind::X, 3),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.median_filter",
        &dataset,
        &json!({
            "radius": 1.0
        }),
    )
    .expect("median filter");

    assert_eq!(output.dataset.shape(), &[3, 3, 2]);
    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 0])], 0.0);
    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 1])], 1.0);
}

#[test]
fn image_remove_nans_replaces_only_nan_pixels() {
    let dataset = test_dataset(
        vec![
            1.0,
            2.0,
            3.0, //
            4.0,
            f32::NAN,
            6.0, //
            7.0,
            8.0,
            9.0,
        ],
        (3, 3),
    );

    let output = execute_operation(
        "image.remove_nans",
        &dataset,
        &json!({
            "radius": 1.0
        }),
    )
    .expect("remove nans");

    let values = output.dataset.data.iter().copied().collect::<Vec<_>>();
    assert_eq!(values[0], 1.0);
    assert_eq!(values[4], 6.0);
    assert_eq!(values[8], 9.0);
}

#[test]
fn image_remove_nans_requires_f32_and_preserves_all_nan_neighborhoods() {
    let mut integer_dataset = test_dataset(vec![f32::NAN; 4], (2, 2));
    integer_dataset.metadata.pixel_type = PixelType::U8;
    let error = execute_operation("image.remove_nans", &integer_dataset, &json!({}))
        .expect_err("integer remove nans");
    assert!(error.to_string().contains("f32"));

    let output = execute_operation(
        "image.remove_nans",
        &test_dataset(vec![f32::NAN; 4], (2, 2)),
        &json!({
            "radius": 1.0
        }),
    )
    .expect("remove nans");
    assert!(output.dataset.data.iter().all(|value| value.is_nan()));
}

#[test]
fn image_remove_outliers_replaces_bright_and_dark_impulses() {
    let mut bright = test_dataset(
        vec![
            0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
        (3, 3),
    );
    bright.metadata.pixel_type = PixelType::U8;

    let output = execute_operation(
        "image.remove_outliers",
        &bright,
        &json!({
            "radius": 1.0,
            "threshold": 50.0,
            "which": "bright"
        }),
    )
    .expect("remove bright outliers");
    assert_eq!(output.dataset.data[IxDyn(&[1, 1])], 0.0);

    let dark = test_dataset(
        vec![
            1.0, 1.0, 1.0, //
            1.0, 0.0, 1.0, //
            1.0, 1.0, 1.0,
        ],
        (3, 3),
    );
    let output = execute_operation(
        "image.remove_outliers",
        &dark,
        &json!({
            "radius": 1.0,
            "threshold": 0.25,
            "which": "dark"
        }),
    )
    .expect("remove dark outliers");
    assert_eq!(output.dataset.data[IxDyn(&[1, 1])], 1.0);
}

#[test]
fn image_remove_outliers_preserves_pixels_within_threshold() {
    let dataset = test_dataset(
        vec![
            0.4, 0.4, 0.4, //
            0.4, 0.5, 0.4, //
            0.4, 0.4, 0.4,
        ],
        (3, 3),
    );

    let output = execute_operation(
        "image.remove_outliers",
        &dataset,
        &json!({
            "radius": 1.0,
            "threshold": 0.2,
            "which": "bright"
        }),
    )
    .expect("remove outliers");
    assert_eq!(output.dataset.data[IxDyn(&[1, 1])], 0.5);

    let bad_kind = execute_operation(
        "image.remove_outliers",
        &dataset,
        &json!({
            "which": "middle"
        }),
    )
    .expect_err("invalid kind");
    assert!(bad_kind.to_string().contains("outlier kind"));
}

#[test]
fn salt_and_pepper_noise_is_seeded_and_u8_only() {
    let mut dataset = test_dataset(vec![0.5; 100], (10, 10));
    dataset.metadata.pixel_type = PixelType::U8;

    let first = execute_operation(
        "noise.salt_and_pepper",
        &dataset,
        &json!({
            "percent": 0.2,
            "seed": 7
        }),
    )
    .expect("salt and pepper");
    let second = execute_operation(
        "noise.salt_and_pepper",
        &dataset,
        &json!({
            "percent": 0.2,
            "seed": 7
        }),
    )
    .expect("salt and pepper");
    assert_eq!(first.dataset.data, second.dataset.data);

    let values = first.dataset.data.iter().copied().collect::<Vec<_>>();
    assert!(values.iter().any(|value| *value == 0.0));
    assert!(values.iter().any(|value| *value == 1.0));
    assert!(values.iter().any(|value| *value == 0.5));

    let f32_error = execute_operation(
        "noise.salt_and_pepper",
        &test_dataset(vec![0.5; 100], (10, 10)),
        &json!({}),
    )
    .expect_err("f32 unsupported");
    assert!(f32_error.to_string().contains("u8"));
}

#[test]
fn gaussian_noise_is_seeded_and_keeps_integer_samples_in_range() {
    let mut dataset = test_dataset(vec![0.5; 100], (10, 10));
    dataset.metadata.pixel_type = PixelType::U8;

    let first = execute_operation(
        "noise.gaussian",
        &dataset,
        &json!({
            "sigma": 25.0,
            "seed": 11
        }),
    )
    .expect("gaussian noise");
    let second = execute_operation(
        "noise.gaussian",
        &dataset,
        &json!({
            "sigma": 25.0,
            "seed": 11
        }),
    )
    .expect("gaussian noise");

    assert_eq!(first.dataset.data, second.dataset.data);
    assert_eq!(first.dataset.metadata.pixel_type, PixelType::U8);
    let values = first.dataset.data.iter().copied().collect::<Vec<_>>();
    assert!(values.iter().any(|value| (*value - 0.5).abs() > 1.0e-6));
    assert!(values.iter().all(|value| (0.0..=1.0).contains(value)));
}

#[test]
fn gaussian_noise_adds_to_float_pixels_without_clamping() {
    let dataset = test_dataset(vec![0.0; 16], (4, 4));

    let output = execute_operation(
        "noise.gaussian",
        &dataset,
        &json!({
            "sigma": 5.0,
            "seed": 3
        }),
    )
    .expect("gaussian noise");

    let values = output.dataset.data.iter().copied().collect::<Vec<_>>();
    assert!(values.iter().any(|value| *value < 0.0));
    assert!(values.iter().any(|value| *value > 1.0));
}

#[test]
fn gaussian_noise_rejects_invalid_sigma() {
    let dataset = test_dataset(vec![0.0; 4], (2, 2));
    let error = execute_operation(
        "noise.gaussian",
        &dataset,
        &json!({
            "sigma": -1.0
        }),
    )
    .expect_err("negative sigma");

    assert!(error.to_string().contains("sigma"));
}

#[test]
fn find_edges_highlights_gradient() {
    let dataset = test_dataset(
        vec![
            0.0, 0.0, 0.0, //
            0.0, 1.0, 1.0, //
            0.0, 1.0, 1.0, //
        ],
        (3, 3),
    );
    let output = execute_operation("image.find_edges", &dataset, &json!({})).expect("edges");
    assert!(output.dataset.data[IxDyn(&[1, 1])] > 0.0);
}

#[test]
fn image_shadow_applies_imagej_directional_kernel() {
    let dataset = test_dataset(
        vec![
            0.0, 1.0, 2.0, //
            3.0, 4.0, 5.0, //
            6.0, 7.0, 8.0, //
        ],
        (3, 3),
    );

    let north = execute_operation(
        "image.shadow",
        &dataset,
        &json!({
            "direction": "north"
        }),
    )
    .expect("north shadow");
    let east = execute_operation(
        "image.shadow",
        &dataset,
        &json!({
            "direction": "east"
        }),
    )
    .expect("east shadow");

    assert_eq!(north.dataset.data[IxDyn(&[1, 1])], -20.0);
    assert_eq!(east.dataset.data[IxDyn(&[1, 1])], 12.0);
}

#[test]
fn image_shadow_processes_xy_planes_independently() {
    let data = Array::from_shape_vec(
        (3, 3, 2),
        vec![
            0.0, 10.0, 1.0, 10.0, 2.0, 10.0, //
            3.0, 10.0, 4.0, 10.0, 5.0, 10.0, //
            6.0, 10.0, 7.0, 10.0, 8.0, 10.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 3),
            Dim::new(AxisKind::X, 3),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.shadow",
        &dataset,
        &json!({
            "direction": "north"
        }),
    )
    .expect("shadow");

    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 0])], -20.0);
    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 1])], 10.0);
}

#[test]
fn image_shadow_demo_creates_eight_direction_stack() {
    let dataset = test_dataset(
        vec![
            0.0, 1.0, 2.0, //
            3.0, 4.0, 5.0, //
            6.0, 7.0, 8.0, //
        ],
        (3, 3),
    );

    let demo = execute_operation(
        "image.shadow_demo",
        &dataset,
        &json!({
            "iterations": 2
        }),
    )
    .expect("shadows demo");
    let north =
        execute_operation("image.shadow", &dataset, &json!({"direction": "north"})).expect("north");
    let northeast = execute_operation("image.shadow", &dataset, &json!({"direction": "northeast"}))
        .expect("northeast");

    assert_eq!(demo.dataset.shape(), &[3, 3, 16]);
    assert_eq!(demo.dataset.metadata.dims[2].axis, AxisKind::Z);
    assert_eq!(
        demo.dataset.data[IxDyn(&[1, 1, 0])],
        north.dataset.data[IxDyn(&[1, 1])]
    );
    assert_eq!(
        demo.dataset.data[IxDyn(&[1, 1, 1])],
        northeast.dataset.data[IxDyn(&[1, 1])]
    );
    assert_eq!(
        demo.dataset.data[IxDyn(&[1, 1, 8])],
        north.dataset.data[IxDyn(&[1, 1])]
    );
}

#[test]
fn image_shadow_demo_rejects_stacks_and_zero_iterations() {
    let data = Array::from_shape_vec((1, 1, 2), vec![0.0, 1.0])
        .expect("shape")
        .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 1),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let stack = Dataset::new(data, metadata).expect("dataset");
    let stack_error =
        execute_operation("image.shadow_demo", &stack, &json!({})).expect_err("stacks rejected");
    assert!(stack_error.to_string().contains("stacks"));

    let dataset = test_dataset(vec![0.0], (1, 1));
    let iterations_error = execute_operation(
        "image.shadow_demo",
        &dataset,
        &json!({
            "iterations": 0
        }),
    )
    .expect_err("zero iterations rejected");
    assert!(iterations_error.to_string().contains("iterations"));
}

#[test]
fn image_shadow_rejects_unknown_direction() {
    let dataset = test_dataset(vec![0.0; 9], (3, 3));
    let error = execute_operation(
        "image.shadow",
        &dataset,
        &json!({
            "direction": "up"
        }),
    )
    .expect_err("invalid direction");

    assert!(error.to_string().contains("direction"));
}

#[test]
fn image_rank_filter_applies_mean_min_max_and_variance() {
    let dataset = test_dataset(
        vec![
            0.0, 1.0, 2.0, //
            3.0, 4.0, 5.0, //
            6.0, 7.0, 8.0, //
        ],
        (3, 3),
    );

    let mean = execute_operation(
        "image.rank_filter",
        &dataset,
        &json!({"filter": "mean", "radius": 1.0}),
    )
    .expect("mean");
    let minimum = execute_operation(
        "image.rank_filter",
        &dataset,
        &json!({"filter": "minimum", "radius": 1.0}),
    )
    .expect("minimum");
    let maximum = execute_operation(
        "image.rank_filter",
        &dataset,
        &json!({"filter": "maximum", "radius": 1.0}),
    )
    .expect("maximum");
    let variance = execute_operation(
        "image.rank_filter",
        &dataset,
        &json!({"filter": "variance", "radius": 1.0}),
    )
    .expect("variance");

    assert_eq!(mean.dataset.data[IxDyn(&[1, 1])], 4.0);
    assert_eq!(minimum.dataset.data[IxDyn(&[1, 1])], 0.0);
    assert_eq!(maximum.dataset.data[IxDyn(&[1, 1])], 8.0);
    assert!((variance.dataset.data[IxDyn(&[1, 1])] - 6.666_666_5).abs() < 1.0e-5);
}

#[test]
fn image_rank_filter_processes_xy_planes_independently() {
    let data = Array::from_shape_vec(
        (3, 3, 2),
        vec![
            0.0, 10.0, 1.0, 10.0, 2.0, 10.0, //
            3.0, 10.0, 4.0, 10.0, 5.0, 10.0, //
            6.0, 10.0, 7.0, 10.0, 8.0, 10.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 3),
            Dim::new(AxisKind::X, 3),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.rank_filter",
        &dataset,
        &json!({"filter": "mean", "radius": 1.0}),
    )
    .expect("rank filter");

    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 0])], 4.0);
    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 1])], 10.0);
}

#[test]
fn image_rank_filter_top_hat_subtracts_grayscale_open() {
    let dataset = test_dataset(
        vec![
            1.0, 1.0, 1.0, //
            1.0, 10.0, 1.0, //
            1.0, 1.0, 1.0, //
        ],
        (3, 3),
    );

    let output = execute_operation(
        "image.rank_filter",
        &dataset,
        &json!({"filter": "top_hat", "radius": 1.0}),
    )
    .expect("top hat");

    assert_eq!(output.dataset.data[IxDyn(&[1, 1])], 9.0);
    assert_eq!(output.dataset.data[IxDyn(&[0, 0])], 0.0);
}

#[test]
fn image_rank_filter_top_hat_can_return_grayscale_close() {
    let dataset = test_dataset(
        vec![
            10.0, 10.0, 10.0, //
            10.0, 0.0, 10.0, //
            10.0, 10.0, 10.0, //
        ],
        (3, 3),
    );

    let output = execute_operation(
        "image.rank_filter",
        &dataset,
        &json!({
            "filter": "top_hat",
            "radius": 1.0,
            "light_background": true,
            "dont_subtract": true
        }),
    )
    .expect("top hat");

    assert_eq!(output.dataset.data[IxDyn(&[1, 1])], 10.0);
}

#[test]
fn image_rank_filter_rejects_invalid_params() {
    let dataset = test_dataset(vec![0.0; 9], (3, 3));
    let error = execute_operation(
        "image.rank_filter",
        &dataset,
        &json!({"filter": "range", "radius": 1.0}),
    )
    .expect_err("invalid filter");
    assert!(error.to_string().contains("rank filter"));

    let error = execute_operation(
        "image.rank_filter",
        &dataset,
        &json!({"filter": "mean", "radius": -1.0}),
    )
    .expect_err("invalid radius");
    assert!(error.to_string().contains("radius"));
}

#[test]
fn image_rank_filter_3d_uses_ellipsoid_neighborhood() {
    let mut values = vec![0.0; 27];
    values[1 * 9 + 1 * 3 + 1] = 1.0;
    values[1 * 9 + 1 * 3] = 2.0;
    values[1 * 9 + 1 * 3 + 2] = 3.0;
    values[1 * 9 + 1] = 4.0;
    values[1 * 9 + 2 * 3 + 1] = 5.0;
    values[1 * 3 + 1] = 6.0;
    values[2 * 9 + 1 * 3 + 1] = 7.0;
    values[0] = 99.0;
    let data = Array::from_shape_vec((3, 3, 3), values)
        .expect("shape")
        .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Z, 3),
            Dim::new(AxisKind::Y, 3),
            Dim::new(AxisKind::X, 3),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.rank_filter_3d",
        &dataset,
        &json!({"filter": "mean", "x_radius": 1.0, "y_radius": 1.0, "z_radius": 1.0}),
    )
    .expect("3D mean");

    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 1])], 4.0);
}

#[test]
fn image_rank_filter_3d_skips_out_of_bounds_neighbors() {
    let values = (0..27).map(|value| value as f32).collect::<Vec<_>>();
    let data = Array::from_shape_vec((3, 3, 3), values)
        .expect("shape")
        .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Z, 3),
            Dim::new(AxisKind::Y, 3),
            Dim::new(AxisKind::X, 3),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.rank_filter_3d",
        &dataset,
        &json!({"filter": "mean", "radius": 1.0}),
    )
    .expect("3D mean");

    assert_eq!(output.dataset.data[IxDyn(&[0, 0, 0])], 3.25);
}

#[test]
fn image_rank_filter_3d_rejects_invalid_params() {
    let dataset = test_dataset(vec![0.0; 9], (3, 3));
    let error = execute_operation(
        "image.rank_filter_3d",
        &dataset,
        &json!({"filter": "mode", "radius": 1.0}),
    )
    .expect_err("invalid filter");
    assert!(error.to_string().contains("3D rank filter"));

    let error = execute_operation(
        "image.rank_filter_3d",
        &dataset,
        &json!({"filter": "mean", "radius": -1.0}),
    )
    .expect_err("invalid radius");
    assert!(error.to_string().contains("radius"));
}

#[test]
fn image_swap_quadrants_swaps_xy_corner_blocks() {
    let dataset = test_dataset(
        vec![
            0.0, 1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, 7.0, //
            8.0, 9.0, 10.0, 11.0, //
            12.0, 13.0, 14.0, 15.0,
        ],
        (4, 4),
    );

    let output =
        execute_operation("image.swap_quadrants", &dataset, &json!({})).expect("swap quadrants");

    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![
            10.0, 11.0, 8.0, 9.0, //
            14.0, 15.0, 12.0, 13.0, //
            2.0, 3.0, 0.0, 1.0, //
            6.0, 7.0, 4.0, 5.0,
        ]
    );
}

#[test]
fn image_swap_quadrants_processes_xy_planes_independently() {
    let data = Array::from_shape_vec(
        (2, 2, 2),
        vec![
            1.0, 10.0, 2.0, 20.0, //
            3.0, 30.0, 4.0, 40.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 2),
            Dim::new(AxisKind::X, 2),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output =
        execute_operation("image.swap_quadrants", &dataset, &json!({})).expect("swap quadrants");

    assert_eq!(output.dataset.data[IxDyn(&[0, 0, 0])], 4.0);
    assert_eq!(output.dataset.data[IxDyn(&[0, 0, 1])], 40.0);
    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 0])], 1.0);
    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 1])], 10.0);
}

#[test]
fn image_swap_quadrants_rejects_non_square_xy() {
    let dataset = test_dataset(vec![0.0; 6], (2, 3));
    let error = execute_operation("image.swap_quadrants", &dataset, &json!({}))
        .expect_err("non-square image");

    assert!(error.to_string().contains("square"));
}

#[test]
fn image_fft_power_spectrum_centers_dc_component() {
    let dataset = test_dataset(vec![1.0; 16], (4, 4));

    let output = execute_operation("image.fft_power_spectrum", &dataset, &json!({}))
        .expect("FFT power spectrum");

    assert_eq!(output.dataset.metadata.pixel_type, PixelType::U8);
    assert!(output.dataset.data[IxDyn(&[2, 2])] > output.dataset.data[IxDyn(&[0, 0])]);
}

#[test]
fn image_fft_power_spectrum_processes_xy_planes_independently() {
    let data = Array::from_shape_vec(
        (2, 2, 2),
        vec![
            1.0, 1.0, 1.0, 2.0, //
            1.0, 3.0, 1.0, 4.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 2),
            Dim::new(AxisKind::X, 2),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation("image.fft_power_spectrum", &dataset, &json!({}))
        .expect("FFT power spectrum");

    assert!(output.dataset.data[IxDyn(&[1, 1, 0])] > 0.0);
    assert!(output.dataset.data[IxDyn(&[1, 1, 1])] > 0.0);
    let constant_plane = [
        output.dataset.data[IxDyn(&[0, 0, 0])],
        output.dataset.data[IxDyn(&[0, 1, 0])],
        output.dataset.data[IxDyn(&[1, 0, 0])],
    ];
    assert!(constant_plane.iter().all(|value| *value <= 1.0 / 255.0));
    assert!(output.dataset.data[IxDyn(&[0, 1, 1])] > output.dataset.data[IxDyn(&[0, 0, 1])]);
}

#[test]
fn image_fft_bandpass_suppresses_constant_background() {
    let dataset = test_dataset(vec![5.0; 16], (4, 4));

    let output = execute_operation(
        "image.fft_bandpass",
        &dataset,
        &json!({"filter_large": 40.0, "filter_small": 3.0, "autoscale": false}),
    )
    .expect("FFT bandpass");

    assert!(output.dataset.data.iter().all(|value| value.abs() < 1.0e-5));
}

#[test]
fn image_fft_bandpass_preserves_planes_and_validates_params() {
    let data = Array::from_shape_vec(
        (2, 2, 2),
        vec![
            1.0, 10.0, 2.0, 20.0, //
            3.0, 30.0, 4.0, 40.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 2),
            Dim::new(AxisKind::X, 2),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.fft_bandpass",
        &dataset,
        &json!({"filter_large": 2.0, "filter_small": 0.0, "autoscale": false}),
    )
    .expect("FFT bandpass");

    assert_ne!(
        output.dataset.data[IxDyn(&[0, 0, 0])],
        output.dataset.data[IxDyn(&[0, 0, 1])]
    );

    let error = execute_operation(
        "image.fft_bandpass",
        &dataset,
        &json!({"suppress_stripes": "diagonal"}),
    )
    .expect_err("invalid stripe suppression");
    assert!(error.to_string().contains("stripe"));
}

#[test]
fn image_convolve_applies_custom_kernel() {
    let dataset = test_dataset(
        vec![
            0.0, 1.0, 2.0, //
            3.0, 4.0, 5.0, //
            6.0, 7.0, 8.0, //
        ],
        (3, 3),
    );

    let output = execute_operation(
        "image.convolve",
        &dataset,
        &json!({
            "width": 3,
            "height": 3,
            "kernel": [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            "normalize": false
        }),
    )
    .expect("convolve");

    assert_eq!(output.dataset.data[IxDyn(&[1, 1])], 8.0);
}

#[test]
fn image_convolve_normalizes_nonzero_kernel_by_default() {
    let dataset = test_dataset(
        vec![
            0.0, 1.0, 2.0, //
            3.0, 4.0, 5.0, //
            6.0, 7.0, 8.0, //
        ],
        (3, 3),
    );

    let output = execute_operation(
        "image.convolve",
        &dataset,
        &json!({
            "width": 3,
            "height": 3,
            "kernel": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }),
    )
    .expect("convolve");

    assert_eq!(output.dataset.data[IxDyn(&[1, 1])], 4.0);
}

#[test]
fn image_convolve_processes_xy_planes_independently() {
    let data = Array::from_shape_vec(
        (3, 3, 2),
        vec![
            0.0, 10.0, 1.0, 10.0, 2.0, 10.0, //
            3.0, 10.0, 4.0, 10.0, 5.0, 10.0, //
            6.0, 10.0, 7.0, 10.0, 8.0, 10.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 3),
            Dim::new(AxisKind::X, 3),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.convolve",
        &dataset,
        &json!({
            "width": 3,
            "height": 3,
            "kernel": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }),
    )
    .expect("convolve");

    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 0])], 4.0);
    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 1])], 10.0);
}

#[test]
fn image_convolve_rejects_invalid_params() {
    let dataset = test_dataset(vec![0.0; 9], (3, 3));

    let error = execute_operation(
        "image.convolve",
        &dataset,
        &json!({
            "width": 2,
            "height": 3,
            "kernel": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }),
    )
    .expect_err("even width");
    assert!(error.to_string().contains("width"));

    let error = execute_operation(
        "image.convolve",
        &dataset,
        &json!({
            "width": 3,
            "height": 3,
            "kernel": [1.0, 1.0, 1.0]
        }),
    )
    .expect_err("kernel length");
    assert!(error.to_string().contains("kernel"));
}

#[test]
fn image_unsharp_mask_enhances_high_frequency_detail() {
    let dataset = test_dataset(vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], (3, 3));

    let output = execute_operation(
        "image.unsharp_mask",
        &dataset,
        &json!({"sigma": 1.0, "weight": 0.6}),
    )
    .expect("unsharp mask");

    assert!(output.dataset.data[IxDyn(&[1, 1])] > 1.0);
    assert!(output.dataset.data[IxDyn(&[0, 0])] < 0.0);
}

#[test]
fn image_unsharp_mask_processes_xy_planes_independently() {
    let data = Array::from_shape_vec(
        (3, 3, 2),
        vec![
            0.0, 10.0, 0.0, 10.0, 0.0, 10.0, //
            0.0, 10.0, 1.0, 10.0, 0.0, 10.0, //
            0.0, 10.0, 0.0, 10.0, 0.0, 10.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 3),
            Dim::new(AxisKind::X, 3),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.unsharp_mask",
        &dataset,
        &json!({"sigma": 1.0, "weight": 0.6}),
    )
    .expect("unsharp mask");

    assert!(output.dataset.data[IxDyn(&[1, 1, 0])] > 1.0);
    assert!((output.dataset.data[IxDyn(&[1, 1, 1])] - 10.0).abs() < 1.0e-4);
}

#[test]
fn image_unsharp_mask_rejects_invalid_params() {
    let dataset = test_dataset(vec![0.0; 9], (3, 3));
    let error = execute_operation(
        "image.unsharp_mask",
        &dataset,
        &json!({"sigma": -1.0, "weight": 0.6}),
    )
    .expect_err("invalid sigma");
    assert!(error.to_string().contains("sigma"));

    let error = execute_operation(
        "image.unsharp_mask",
        &dataset,
        &json!({"sigma": 1.0, "weight": 1.0}),
    )
    .expect_err("invalid weight");
    assert!(error.to_string().contains("weight"));
}

#[test]
fn gaussian_blur_smooths_spike() {
    let dataset = test_dataset(vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], (3, 3));
    let output =
        execute_operation("gaussian.blur", &dataset, &json!({"sigma": 1.0})).expect("blur");
    let center = output.dataset.data[IxDyn(&[1, 1])];
    let corner = output.dataset.data[IxDyn(&[0, 0])];
    assert!(center < 1.0);
    assert!(center > corner);
}

#[test]
fn connected_components_reports_regions() {
    let dataset = test_dataset(
        vec![
            1.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, //
            0.0, 1.0, 1.0, //
        ],
        (3, 3),
    );
    let output = execute_operation("components.label", &dataset, &json!({})).expect("components");
    let count = output
        .measurements
        .as_ref()
        .and_then(|table| table.values.get("component_count"))
        .and_then(|value| value.as_u64())
        .expect("count");
    assert_eq!(count, 2);
}

#[test]
fn morphology_fill_holes_fills_enclosed_background() {
    let dataset = test_dataset(
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, 1.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        (5, 5),
    );

    let output =
        execute_operation("morphology.fill_holes", &dataset, &json!({})).expect("fill holes");

    assert_eq!(output.dataset.data[IxDyn(&[2, 2])], 1.0);
    assert_eq!(output.dataset.data[IxDyn(&[0, 0])], 0.0);
    assert_eq!(output.dataset.data[IxDyn(&[4, 4])], 0.0);
}

#[test]
fn morphology_fill_holes_processes_xy_planes_independently() {
    let data = Array::from_shape_vec(
        (3, 3, 2),
        vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
            1.0, 0.0, 0.0, 0.0, 1.0, 1.0, //
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 3),
            Dim::new(AxisKind::X, 3),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output =
        execute_operation("morphology.fill_holes", &dataset, &json!({})).expect("fill holes");

    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 0])], 1.0);
    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 1])], 0.0);
}

#[test]
fn morphology_outline_keeps_boundary_foreground() {
    let dataset = test_dataset(vec![1.0; 9], (3, 3));

    let output = execute_operation("morphology.outline", &dataset, &json!({})).expect("outline");

    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![
            1.0, 1.0, 1.0, //
            1.0, 0.0, 1.0, //
            1.0, 1.0, 1.0,
        ]
    );
}

#[test]
fn morphology_outline_processes_xy_planes_independently() {
    let data = Array::from_shape_vec(
        (3, 3, 2),
        vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
            1.0, 1.0, 1.0, 0.0, 1.0, 1.0, //
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 3),
            Dim::new(AxisKind::X, 3),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation("morphology.outline", &dataset, &json!({})).expect("outline");

    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 0])], 0.0);
    assert_eq!(output.dataset.data[IxDyn(&[1, 0, 1])], 1.0);
    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 1])], 0.0);
}

#[test]
fn morphology_skeletonize_thins_filled_regions() {
    let dataset = test_dataset(
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        (5, 5),
    );

    let output =
        execute_operation("morphology.skeletonize", &dataset, &json!({})).expect("skeletonize");
    let values = output.dataset.data.iter().copied().collect::<Vec<_>>();

    assert!(values.iter().filter(|value| **value > 0.5).count() < 9);
    assert_eq!(output.dataset.data[IxDyn(&[2, 2])], 1.0);
}

#[test]
fn morphology_skeletonize_preserves_single_pixel_lines() {
    let dataset = test_dataset(
        vec![
            0.0, 0.0, 0.0, //
            1.0, 1.0, 1.0, //
            0.0, 0.0, 0.0,
        ],
        (3, 3),
    );

    let output =
        execute_operation("morphology.skeletonize", &dataset, &json!({})).expect("skeletonize");

    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![
            0.0, 0.0, 0.0, //
            1.0, 1.0, 1.0, //
            0.0, 0.0, 0.0,
        ]
    );
}

#[test]
fn morphology_binary_median_smooths_binary_regions() {
    let dataset = test_dataset(
        vec![
            0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
        (3, 3),
    );

    let output = execute_operation(
        "morphology.binary_median",
        &dataset,
        &json!({
            "radius": 1
        }),
    )
    .expect("binary median");
    assert!(output.dataset.data.iter().all(|value| *value == 0.0));

    let dataset = test_dataset(
        vec![
            1.0, 1.0, 1.0, //
            1.0, 0.0, 1.0, //
            1.0, 1.0, 1.0,
        ],
        (3, 3),
    );
    let output = execute_operation(
        "morphology.binary_median",
        &dataset,
        &json!({
            "radius": 1
        }),
    )
    .expect("binary median");
    assert_eq!(output.dataset.data[IxDyn(&[1, 1])], 1.0);
}

#[test]
fn morphology_distance_map_measures_nearest_background() {
    let mut dataset = test_dataset(
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        (5, 5),
    );
    dataset.metadata.pixel_type = PixelType::U8;

    let output =
        execute_operation("morphology.distance_map", &dataset, &json!({})).expect("distance map");

    assert_eq!(output.dataset.metadata.pixel_type, PixelType::F32);
    assert_eq!(output.dataset.data[IxDyn(&[0, 0])], 0.0);
    assert_eq!(output.dataset.data[IxDyn(&[1, 1])], 1.0);
    assert_eq!(output.dataset.data[IxDyn(&[2, 2])], 2.0);
}

#[test]
fn morphology_distance_map_processes_xy_planes_independently() {
    let data = Array::from_shape_vec(
        (3, 3, 2),
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 1.0, 0.0, 1.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 3),
            Dim::new(AxisKind::X, 3),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output =
        execute_operation("morphology.distance_map", &dataset, &json!({})).expect("distance map");

    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 0])], 1.0);
    assert_eq!(output.dataset.data[IxDyn(&[1, 1, 1])], 1.0);
}

#[test]
fn morphology_ultimate_points_keeps_edm_maxima() {
    let mut dataset = test_dataset(
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        (5, 5),
    );
    dataset.metadata.pixel_type = PixelType::U8;

    let output = execute_operation("morphology.ultimate_points", &dataset, &json!({}))
        .expect("ultimate points");

    assert_eq!(output.dataset.metadata.pixel_type, PixelType::F32);
    assert_eq!(output.dataset.data[IxDyn(&[2, 2])], 2.0);
    assert_eq!(
        output
            .dataset
            .data
            .iter()
            .filter(|value| **value > 0.0)
            .count(),
        1
    );
}

#[test]
fn morphology_ultimate_points_collapses_flat_maximum_plateau() {
    let dataset = test_dataset(
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        (6, 6),
    );

    let output = execute_operation("morphology.ultimate_points", &dataset, &json!({}))
        .expect("ultimate points");
    let points = output
        .dataset
        .data
        .indexed_iter()
        .filter(|(_, value)| **value > 0.0)
        .collect::<Vec<_>>();

    assert_eq!(points.len(), 1);
    assert_eq!(*points[0].1, 2.0);
}

#[test]
fn morphology_ultimate_points_processes_xy_planes_independently() {
    let data = Array::from_shape_vec(
        (5, 5, 2),
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 5),
            Dim::new(AxisKind::X, 5),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation("morphology.ultimate_points", &dataset, &json!({}))
        .expect("ultimate points");

    assert_eq!(output.dataset.data[IxDyn(&[2, 2, 0])], 2.0);
    assert_eq!(output.dataset.data[IxDyn(&[2, 2, 1])], 1.0);
    assert_eq!(
        output
            .dataset
            .data
            .indexed_iter()
            .filter(|(_, value)| **value > 0.0)
            .count(),
        2
    );
}

#[test]
fn morphology_watershed_splits_touching_binary_regions() {
    let dataset = test_dataset(
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        (5, 9),
    );

    let output =
        execute_operation("morphology.watershed", &dataset, &json!({})).expect("watershed");

    assert_eq!(output.dataset.data[IxDyn(&[2, 2])], 1.0);
    assert_eq!(output.dataset.data[IxDyn(&[2, 6])], 1.0);
    assert_eq!(output.dataset.data[IxDyn(&[2, 4])], 0.0);
    assert!(output.dataset.data.iter().sum::<f32>() < dataset.data.iter().sum::<f32>());
}

#[test]
fn morphology_watershed_keeps_single_seed_region() {
    let dataset = test_dataset(
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 1.0, 1.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        (5, 5),
    );

    let output =
        execute_operation("morphology.watershed", &dataset, &json!({})).expect("watershed");

    assert_eq!(output.dataset.data, dataset.data);
}

#[test]
fn morphology_watershed_processes_xy_planes_independently() {
    let data = Array::from_shape_vec(
        (5, 9, 2),
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, //
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.0, //
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.0, //
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 5),
            Dim::new(AxisKind::X, 9),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output =
        execute_operation("morphology.watershed", &dataset, &json!({})).expect("watershed");

    assert_eq!(output.dataset.data[IxDyn(&[2, 4, 0])], 0.0);
    assert_eq!(output.dataset.data[IxDyn(&[2, 4, 1])], 1.0);
}

#[test]
fn morphology_voronoi_marks_divider_between_particles() {
    let mut dataset = test_dataset(
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        (5, 7),
    );
    dataset.metadata.pixel_type = PixelType::U8;

    let output = execute_operation("morphology.voronoi", &dataset, &json!({})).expect("voronoi");

    assert_eq!(output.dataset.metadata.pixel_type, PixelType::F32);
    assert_eq!(output.dataset.data[IxDyn(&[2, 1])], 0.0);
    assert_eq!(output.dataset.data[IxDyn(&[2, 5])], 0.0);
    assert_eq!(output.dataset.data[IxDyn(&[2, 3])], 2.0);
    assert_eq!(output.dataset.data[IxDyn(&[2, 2])], 0.0);
}

#[test]
fn morphology_voronoi_requires_multiple_particles() {
    let dataset = test_dataset(
        vec![
            0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
        (3, 3),
    );

    let output = execute_operation("morphology.voronoi", &dataset, &json!({})).expect("voronoi");

    assert!(output.dataset.data.iter().all(|value| *value == 0.0));
}

#[test]
fn morphology_voronoi_processes_xy_planes_independently() {
    let mut values = Vec::new();
    for y in 0..5 {
        for x in 0..7 {
            values.push(if y == 2 && (x == 1 || x == 5) {
                1.0
            } else {
                0.0
            });
            values.push(if y == 2 && x == 3 { 1.0 } else { 0.0 });
        }
    }
    let data = Array::from_shape_vec((5, 7, 2), values)
        .expect("shape")
        .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 5),
            Dim::new(AxisKind::X, 7),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation("morphology.voronoi", &dataset, &json!({})).expect("voronoi");

    assert_eq!(output.dataset.data[IxDyn(&[2, 3, 0])], 2.0);
    assert_eq!(output.dataset.data[IxDyn(&[2, 3, 1])], 0.0);
    assert_eq!(
        output
            .dataset
            .data
            .indexed_iter()
            .filter(|(index, value)| index[2] == 1 && **value > 0.0)
            .count(),
        0
    );
}

#[test]
fn window_operation_validates_bounds() {
    let dataset = test_dataset(vec![0.0, 0.5, 0.75, 1.0], (2, 2));
    let output = execute_operation(
        "intensity.window",
        &dataset,
        &json!({
            "low": 0.25,
            "high": 0.75
        }),
    )
    .expect("window");
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 0.5, 1.0, 1.0]
    );

    let error = execute_operation(
        "intensity.window",
        &dataset,
        &json!({
            "low": 1.0,
            "high": 0.1
        }),
    )
    .expect_err("invalid bounds");
    assert!(error.to_string().contains("high"));
}

#[test]
fn otsu_threshold_returns_measurement() {
    let dataset = test_dataset(vec![0.05, 0.1, 0.2, 0.8, 0.9, 0.95], (2, 3));
    let output = execute_operation("threshold.otsu", &dataset, &json!({})).expect("otsu");
    let threshold = output
        .measurements
        .as_ref()
        .and_then(|table| table.values.get("threshold"))
        .and_then(|value| value.as_f64())
        .expect("threshold");
    assert!(threshold > 0.0);
}

#[test]
fn measurements_include_bbox() {
    let dataset = test_dataset(
        vec![
            0.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, //
        ],
        (3, 3),
    );
    let output = execute_operation("measurements.summary", &dataset, &json!({})).expect("measure");
    let table = output.measurements.expect("measurements");
    assert_eq!(
        table
            .values
            .get("area")
            .and_then(|value| value.as_u64())
            .expect("area"),
        2
    );
    assert!(table.values.contains_key("bbox_min"));
    assert!(table.values.contains_key("bbox_max"));
}
