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
    assert!(names.contains(&"intensity.enhance_contrast".to_string()));
    assert!(names.contains(&"intensity.invert".to_string()));
    assert!(names.contains(&"intensity.math".to_string()));
    assert!(names.contains(&"intensity.nan_background".to_string()));
    assert!(names.contains(&"image.convert".to_string()));
    assert!(names.contains(&"image.resize".to_string()));
    assert!(names.contains(&"image.canvas_resize".to_string()));
    assert!(names.contains(&"image.crop".to_string()));
    assert!(names.contains(&"image.coordinates".to_string()));
    assert!(names.contains(&"image.set_scale".to_string()));
    assert!(names.contains(&"image.calibrate".to_string()));
    assert!(names.contains(&"image.bin".to_string()));
    assert!(names.contains(&"image.flip".to_string()));
    assert!(names.contains(&"image.median_filter".to_string()));
    assert!(names.contains(&"image.remove_nans".to_string()));
    assert!(names.contains(&"image.remove_outliers".to_string()));
    assert!(names.contains(&"image.scale".to_string()));
    assert!(names.contains(&"image.stack.add_slice".to_string()));
    assert!(names.contains(&"image.stack.delete_slice".to_string()));
    assert!(names.contains(&"image.stack.grouped_z_project".to_string()));
    assert!(names.contains(&"image.stack.montage".to_string()));
    assert!(names.contains(&"image.stack.montage_to_stack".to_string()));
    assert!(names.contains(&"image.stack.reduce".to_string()));
    assert!(names.contains(&"image.stack.reslice".to_string()));
    assert!(names.contains(&"image.stack.statistics".to_string()));
    assert!(names.contains(&"image.stack.substack".to_string()));
    assert!(names.contains(&"image.stack.to_hyperstack".to_string()));
    assert!(names.contains(&"image.hyperstack.to_stack".to_string()));
    assert!(names.contains(&"image.hyperstack.reduce_dimensionality".to_string()));
    assert!(names.contains(&"image.stack.z_profile".to_string()));
    assert!(names.contains(&"image.stack.z_project".to_string()));
    assert!(names.contains(&"image.rotate_90".to_string()));
    assert!(names.contains(&"image.rotate".to_string()));
    assert!(names.contains(&"image.translate".to_string()));
    assert!(names.contains(&"image.rank_filter".to_string()));
    assert!(names.contains(&"image.rank_filter_3d".to_string()));
    assert!(names.contains(&"image.sharpen".to_string()));
    assert!(names.contains(&"image.swap_quadrants".to_string()));
    assert!(names.contains(&"image.fft_power_spectrum".to_string()));
    assert!(names.contains(&"image.fft_bandpass".to_string()));
    assert!(names.contains(&"image.surface_plot".to_string()));
    assert!(names.contains(&"image.convolve".to_string()));
    assert!(names.contains(&"image.unsharp_mask".to_string()));
    assert!(names.contains(&"image.find_edges".to_string()));
    assert!(names.contains(&"image.find_maxima".to_string()));
    assert!(names.contains(&"image.shadow".to_string()));
    assert!(names.contains(&"image.shadow_demo".to_string()));
    assert!(names.contains(&"image.subtract_background".to_string()));
    assert!(names.contains(&"threshold.make_binary".to_string()));
    assert!(names.contains(&"threshold.otsu".to_string()));
    assert!(names.contains(&"measurements.histogram".to_string()));
    assert!(names.contains(&"measurements.profile".to_string()));
    assert!(names.contains(&"components.label".to_string()));
    assert!(names.contains(&"noise.gaussian".to_string()));
    assert!(names.contains(&"noise.salt_and_pepper".to_string()));
    assert!(names.contains(&"measurements.summary".to_string()));
    assert!(names.contains(&"morphology.binary_median".to_string()));
    assert!(names.contains(&"morphology.erode".to_string()));
    assert!(names.contains(&"morphology.dilate".to_string()));
    assert!(names.contains(&"morphology.open".to_string()));
    assert!(names.contains(&"morphology.close".to_string()));
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
fn intensity_invert_uses_integer_pixel_type_range() {
    let mut dataset = test_dataset(vec![0.0, 64.0, 128.0, 255.0], (2, 2));
    dataset.metadata.pixel_type = PixelType::U8;

    let output = execute_operation("intensity.invert", &dataset, &json!({})).expect("invert");
    let values = output.dataset.data.iter().copied().collect::<Vec<_>>();
    assert_eq!(values, vec![255.0, 191.0, 127.0, 0.0]);
    assert_eq!(output.dataset.metadata.pixel_type, PixelType::U8);
}

#[test]
fn intensity_normalize_uses_integer_pixel_type_range() {
    let mut dataset = test_dataset(vec![0.0, 5.0, 10.0, 20.0], (2, 2));
    dataset.metadata.pixel_type = PixelType::U8;

    let output = execute_operation(
        "intensity.normalize",
        &dataset,
        &json!({
            "min": 0.0,
            "max": 10.0
        }),
    )
    .expect("normalize");

    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 128.0, 255.0, 255.0]
    );
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
fn intensity_enhance_contrast_clips_saturated_tails_and_normalizes() {
    let dataset = test_dataset(
        vec![
            -10.0, 0.0, 1.0, 2.0, //
            3.0, 4.0, 5.0, 100.0,
        ],
        (2, 4),
    );

    let output = execute_operation(
        "intensity.enhance_contrast",
        &dataset,
        &json!({
            "saturated_percent": 25.0,
            "normalize": true
        }),
    )
    .expect("enhance contrast");

    let values = output.dataset.data.iter().copied().collect::<Vec<_>>();
    assert_eq!(values[0], 0.0);
    assert_eq!(values[7], 1.0);
    assert!((values[1] - 0.0).abs() < 1.0e-6);
    assert!((values[6] - 1.0).abs() < 1.0e-6);
    assert!((values[3] - 0.4).abs() < 1.0e-6);
}

#[test]
fn intensity_enhance_contrast_uses_integer_pixel_type_range() {
    let mut dataset = test_dataset(vec![0.0, 5.0, 10.0, 20.0], (2, 2));
    dataset.metadata.pixel_type = PixelType::U8;

    let output = execute_operation(
        "intensity.enhance_contrast",
        &dataset,
        &json!({
            "saturated_percent": 0.0,
            "normalize": true
        }),
    )
    .expect("enhance contrast");

    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 64.0, 128.0, 255.0]
    );
    assert_eq!(output.dataset.metadata.pixel_type, PixelType::U8);
}

#[test]
fn intensity_enhance_contrast_can_record_display_range_without_changing_pixels() {
    let dataset = test_dataset(vec![0.0, 1.0, 2.0, 3.0], (2, 2));

    let output = execute_operation(
        "intensity.enhance_contrast",
        &dataset,
        &json!({
            "saturated_percent": 0.0,
            "normalize": false
        }),
    )
    .expect("enhance contrast display range");

    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 1.0, 2.0, 3.0]
    );
    assert_eq!(
        output.dataset.metadata.extras.get("display_min"),
        Some(&json!(0.0))
    );
    assert_eq!(
        output.dataset.metadata.extras.get("display_max"),
        Some(&json!(3.0))
    );
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
            0b1111_0000 as f32,
            0b0000_1111 as f32,
            0b1010_1010 as f32,
            0b0101_0101 as f32,
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
        .map(|value| value.round() as u8)
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
        .map(|value| value.round() as u8)
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
fn image_crop_extracts_xy_bounds_and_preserves_other_axes() {
    let data = Array::from_shape_vec(
        IxDyn(&[2, 3, 2]),
        (0..12).map(|v| v as f32).collect::<Vec<_>>(),
    )
    .expect("shape");
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 2),
            Dim::new(AxisKind::X, 3),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.crop",
        &dataset,
        &json!({"x": 1, "y": 0, "width": 2, "height": 2}),
    )
    .expect("crop");

    assert_eq!(output.dataset.shape(), &[2, 2, 2]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![2.0, 3.0, 4.0, 5.0, 8.0, 9.0, 10.0, 11.0]
    );
    assert_eq!(output.dataset.metadata.dims[2].axis, AxisKind::Z);
    assert_eq!(output.dataset.metadata.dims[2].size, 2);
}

#[test]
fn image_crop_updates_calibrated_origin_and_rejects_out_of_bounds() {
    let mut dataset = test_dataset(
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0,
        ],
        (2, 4),
    );
    dataset.metadata.dims[1].spacing = Some(2.0);
    dataset.metadata.dims[0].spacing = Some(5.0);
    dataset
        .metadata
        .extras
        .insert("x_origin_coordinate".to_string(), json!(10.0));
    dataset
        .metadata
        .extras
        .insert("y_origin_coordinate".to_string(), json!(20.0));

    let output = execute_operation(
        "image.crop",
        &dataset,
        &json!({"x": 1, "y": 1, "width": 2, "height": 1}),
    )
    .expect("crop");

    assert_eq!(output.dataset.shape(), &[1, 2]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![6.0, 7.0]
    );
    assert_eq!(
        output.dataset.metadata.extras.get("x_origin_coordinate"),
        Some(&json!(12.0))
    );
    assert_eq!(
        output.dataset.metadata.extras.get("y_origin_coordinate"),
        Some(&json!(25.0))
    );

    let error = execute_operation(
        "image.crop",
        &dataset,
        &json!({"x": 3, "y": 0, "width": 2, "height": 1}),
    )
    .expect_err("out of bounds");
    assert!(error.to_string().contains("crop bounds"));
}

#[test]
fn image_scale_resizes_by_factor_and_updates_spacing() {
    let mut dataset = test_dataset(
        vec![
            0.0, 1.0, 2.0, //
            3.0, 4.0, 5.0,
        ],
        (2, 3),
    );
    dataset.metadata.dims[1].spacing = Some(2.0);
    dataset.metadata.dims[0].spacing = Some(4.0);

    let output = execute_operation(
        "image.scale",
        &dataset,
        &json!({"x_scale": 2.0, "y_scale": 0.5}),
    )
    .expect("scale");

    assert_eq!(output.dataset.shape(), &[1, 6]);
    assert_eq!(output.dataset.metadata.dims[1].spacing, Some(1.0));
    assert_eq!(output.dataset.metadata.dims[0].spacing, Some(8.0));
    let values = output.dataset.data.iter().copied().collect::<Vec<_>>();
    assert_eq!(values.first().copied(), Some(0.0));
    assert_eq!(values.last().copied(), Some(2.0));
}

#[test]
fn image_scale_rejects_non_positive_factors() {
    let dataset = test_dataset(vec![0.0, 1.0, 2.0, 3.0], (2, 2));

    let error = execute_operation("image.scale", &dataset, &json!({"x_scale": 0.0}))
        .expect_err("invalid scale");
    assert!(error.to_string().contains("x_scale"));
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
fn image_set_scale_updates_xy_calibration_metadata() {
    let dataset = test_dataset(vec![1.0, 2.0, 3.0, 4.0], (2, 2));

    let output = execute_operation(
        "image.set_scale",
        &dataset,
        &json!({
            "distance_pixels": 25.0,
            "known_distance": 10.0,
            "pixel_aspect_ratio": 1.5,
            "unit": "um",
            "global": true
        }),
    )
    .expect("set scale");

    assert_eq!(output.dataset.shape(), dataset.shape());
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        dataset.data.iter().copied().collect::<Vec<_>>()
    );
    assert_eq!(output.dataset.metadata.dims[1].spacing, Some(0.4));
    assert_eq!(output.dataset.metadata.dims[0].spacing, Some(0.6));
    assert_eq!(output.dataset.metadata.dims[1].unit.as_deref(), Some("um"));
    assert_eq!(output.dataset.metadata.dims[0].unit.as_deref(), Some("um"));
    assert_eq!(
        output.dataset.metadata.extras.get("global_calibration"),
        Some(&json!(true))
    );
}

#[test]
fn image_set_scale_resets_pixel_units_and_updates_default_z_spacing() {
    let data = Array::from_shape_vec(IxDyn(&[2, 2, 2]), vec![0.0; 8]).expect("shape");
    let mut metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 2),
            Dim::new(AxisKind::X, 2),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    metadata.dims[2].spacing = Some(1.0);
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let calibrated = execute_operation(
        "image.set_scale",
        &dataset,
        &json!({
            "distance_pixels": 4.0,
            "known_distance": 2.0,
            "unit": "mm"
        }),
    )
    .expect("calibrated set scale");
    assert_eq!(calibrated.dataset.metadata.dims[1].spacing, Some(0.5));
    assert_eq!(calibrated.dataset.metadata.dims[0].spacing, Some(0.5));
    assert_eq!(calibrated.dataset.metadata.dims[2].spacing, Some(0.5));
    assert_eq!(
        calibrated.dataset.metadata.dims[2].unit.as_deref(),
        Some("mm")
    );

    let reset = execute_operation(
        "image.set_scale",
        &calibrated.dataset,
        &json!({
            "distance_pixels": 4.0,
            "known_distance": 2.0,
            "unit": "pixel"
        }),
    )
    .expect("reset set scale");
    assert_eq!(reset.dataset.metadata.dims[1].spacing, Some(1.0));
    assert_eq!(reset.dataset.metadata.dims[0].spacing, Some(1.0));
    assert_eq!(reset.dataset.metadata.dims[2].spacing, Some(1.0));
    assert_eq!(
        reset.dataset.metadata.dims[1].unit.as_deref(),
        Some("pixel")
    );
    assert_eq!(
        reset.dataset.metadata.dims[0].unit.as_deref(),
        Some("pixel")
    );
    assert_eq!(
        reset.dataset.metadata.dims[2].unit.as_deref(),
        Some("pixel")
    );
}

#[test]
fn image_calibrate_updates_value_unit_metadata() {
    let dataset = test_dataset(vec![1.0, 2.0, 3.0, 4.0], (2, 2));

    let output = execute_operation(
        "image.calibrate",
        &dataset,
        &json!({
            "function": "None",
            "unit": "OD",
            "global": true
        }),
    )
    .expect("calibrate");

    assert_eq!(output.dataset.shape(), dataset.shape());
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        dataset.data.iter().copied().collect::<Vec<_>>()
    );
    assert_eq!(
        output.dataset.metadata.extras.get("value_unit"),
        Some(&json!("OD"))
    );
    assert_eq!(
        output
            .dataset
            .metadata
            .extras
            .get("density_calibration_function"),
        Some(&json!("none"))
    );
    assert_eq!(
        output
            .dataset
            .metadata
            .extras
            .get("global_density_calibration"),
        Some(&json!(true))
    );
}

#[test]
fn image_calibrate_rejects_unsupported_curve_functions() {
    let dataset = test_dataset(vec![1.0, 2.0, 3.0, 4.0], (2, 2));

    let error = execute_operation(
        "image.calibrate",
        &dataset,
        &json!({"function": "Straight Line"}),
    )
    .expect_err("unsupported curve function");

    assert!(
        error
            .to_string()
            .contains("unsupported calibration function")
    );
}

#[test]
fn image_surface_plot_renders_grayscale_height_map() {
    let dataset = test_dataset(vec![0.0, 1.0, 0.5, 0.25], (2, 2));

    let output = execute_operation(
        "image.surface_plot",
        &dataset,
        &json!({"plot_width": 8, "polygon_multiplier": 100}),
    )
    .expect("surface plot");

    assert_eq!(output.dataset.ndim(), 2);
    assert!(output.dataset.shape()[0] > 2);
    assert!(output.dataset.shape()[1] >= 8);
    assert!(output.dataset.data.iter().any(|value| *value < 1.0));
    assert_eq!(
        output
            .dataset
            .metadata
            .extras
            .get("surface_plot_source_shape"),
        Some(&json!([2, 2]))
    );
    assert_eq!(
        output
            .dataset
            .metadata
            .extras
            .get("surface_plot_polygon_multiplier"),
        Some(&json!(100))
    );
}

#[test]
fn image_convert_rgb_adds_channel_axis() {
    let dataset = test_dataset(vec![0.0, 1.0, 0.5, 0.25], (2, 2));
    let output =
        execute_operation("image.convert", &dataset, &json!({"target": "rgb"})).expect("convert");
    assert_eq!(output.dataset.shape(), &[2, 2, 3]);
    assert_eq!(output.dataset.metadata.pixel_type, PixelType::U8);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![
            0.0, 0.0, 0.0, //
            255.0, 255.0, 255.0, //
            128.0, 128.0, 128.0, //
            64.0, 64.0, 64.0,
        ]
    );
}

#[test]
fn image_convert_pixel_types_use_raw_integer_ranges() {
    let dataset = test_dataset(vec![0.0, 1.0, 0.5, 0.25], (2, 2));
    let u8_output =
        execute_operation("image.convert", &dataset, &json!({"target": "u8"})).expect("u8");
    assert_eq!(u8_output.dataset.metadata.pixel_type, PixelType::U8);
    assert_eq!(
        u8_output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 255.0, 128.0, 64.0]
    );

    let u16_output = execute_operation(
        "image.convert",
        &u8_output.dataset,
        &json!({"target": "u16"}),
    )
    .expect("u16");
    assert_eq!(u16_output.dataset.metadata.pixel_type, PixelType::U16);
    assert_eq!(
        u16_output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 65_535.0, 32_896.0, 16_448.0]
    );
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
fn image_stack_add_slice_promotes_2d_and_inserts_into_stack() {
    let dataset = test_dataset(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    let promoted = execute_operation(
        "image.stack.add_slice",
        &dataset,
        &json!({
            "index": 1,
            "fill": 9.0
        }),
    )
    .expect("add slice to 2D");
    assert_eq!(promoted.dataset.shape(), &[2, 2, 2]);
    assert_eq!(promoted.dataset.metadata.dims[2].axis, AxisKind::Z);
    assert_eq!(
        promoted.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 4.0, 9.0]
    );

    let inserted = execute_operation(
        "image.stack.add_slice",
        &promoted.dataset,
        &json!({
            "index": 1,
            "fill": 5.0
        }),
    )
    .expect("insert slice");
    assert_eq!(inserted.dataset.shape(), &[2, 2, 3]);
    assert_eq!(inserted.dataset.data[IxDyn(&[0, 0, 1])], 5.0);
    assert_eq!(inserted.dataset.data[IxDyn(&[0, 0, 2])], 9.0);
}

#[test]
fn image_stack_delete_slice_removes_z_plane_and_validates_bounds() {
    let data = Array::from_shape_vec((1, 2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("shape")
        .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 2),
            Dim::new(AxisKind::Z, 3),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.stack.delete_slice",
        &dataset,
        &json!({
            "index": 1
        }),
    )
    .expect("delete slice");
    assert_eq!(output.dataset.shape(), &[1, 2, 2]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 3.0, 4.0, 6.0]
    );

    let error = execute_operation(
        "image.stack.delete_slice",
        &dataset,
        &json!({
            "index": 3
        }),
    )
    .expect_err("invalid delete index");
    assert!(error.to_string().contains("index"));
}

#[test]
fn image_stack_to_hyperstack_maps_linear_stack_to_czt_axes() {
    let data = Array::from_shape_vec(
        (1, 1, 6),
        vec![
            1.0, // c0 z0 t0
            2.0, // c1 z0 t0
            3.0, // c0 z1 t0
            4.0, // c1 z1 t0
            5.0, // c0 z2 t0
            6.0, // c1 z2 t0
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 1),
            Dim::new(AxisKind::Z, 6),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.stack.to_hyperstack",
        &dataset,
        &json!({
            "channels": 2,
            "slices": 3,
            "frames": 1,
            "order": "czt"
        }),
    )
    .expect("stack to hyperstack");

    assert_eq!(output.dataset.shape(), &[1, 1, 3, 2]);
    assert_eq!(
        output
            .dataset
            .metadata
            .dims
            .iter()
            .map(|dim| dim.axis)
            .collect::<Vec<_>>(),
        vec![AxisKind::Y, AxisKind::X, AxisKind::Z, AxisKind::Channel]
    );
    assert_eq!(output.dataset.data[IxDyn(&[0, 0, 0, 0])], 1.0);
    assert_eq!(output.dataset.data[IxDyn(&[0, 0, 0, 1])], 2.0);
    assert_eq!(output.dataset.data[IxDyn(&[0, 0, 2, 0])], 5.0);
    assert_eq!(output.dataset.data[IxDyn(&[0, 0, 2, 1])], 6.0);
    assert_eq!(
        output.dataset.metadata.extras.get("hyperstack_dimensions"),
        Some(&json!({"channels": 2, "slices": 3, "frames": 1}))
    );
}

#[test]
fn image_stack_to_hyperstack_honors_non_default_source_order() {
    let data = Array::from_shape_vec(
        (1, 1, 6),
        vec![
            1.0, // z0 t0 c0
            2.0, // z1 t0 c0
            3.0, // z2 t0 c0
            4.0, // z0 t0 c1
            5.0, // z1 t0 c1
            6.0, // z2 t0 c1
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 1),
            Dim::new(AxisKind::Z, 6),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.stack.to_hyperstack",
        &dataset,
        &json!({
            "channels": 2,
            "slices": 3,
            "frames": 1,
            "order": "ztc"
        }),
    )
    .expect("stack to hyperstack");

    assert_eq!(output.dataset.data[IxDyn(&[0, 0, 0, 0])], 1.0);
    assert_eq!(output.dataset.data[IxDyn(&[0, 0, 1, 0])], 2.0);
    assert_eq!(output.dataset.data[IxDyn(&[0, 0, 2, 0])], 3.0);
    assert_eq!(output.dataset.data[IxDyn(&[0, 0, 0, 1])], 4.0);
    assert_eq!(output.dataset.data[IxDyn(&[0, 0, 1, 1])], 5.0);
    assert_eq!(output.dataset.data[IxDyn(&[0, 0, 2, 1])], 6.0);
}

#[test]
fn image_hyperstack_to_stack_flattens_czt_order() {
    let data = Array::from_shape_vec(
        (1, 1, 2, 2, 2),
        vec![
            1.0, 5.0, // z0 c0 t0,t1
            2.0, 6.0, // z0 c1 t0,t1
            3.0, 7.0, // z1 c0 t0,t1
            4.0, 8.0, // z1 c1 t0,t1
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 1),
            Dim::new(AxisKind::Z, 2),
            Dim::new(AxisKind::Channel, 2),
            Dim::new(AxisKind::Time, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation("image.hyperstack.to_stack", &dataset, &json!({}))
        .expect("hyperstack to stack");

    assert_eq!(output.dataset.shape(), &[1, 1, 8]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
    assert_eq!(
        output
            .dataset
            .metadata
            .dims
            .iter()
            .map(|dim| dim.axis)
            .collect::<Vec<_>>(),
        vec![AxisKind::Y, AxisKind::X, AxisKind::Z]
    );
}

#[test]
fn image_stack_to_hyperstack_validates_dimensions() {
    let data = Array::from_shape_vec((1, 1, 5), vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("shape")
        .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 1),
            Dim::new(AxisKind::Z, 5),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let error = execute_operation(
        "image.stack.to_hyperstack",
        &dataset,
        &json!({"channels": 2, "slices": 3, "frames": 1}),
    )
    .expect_err("invalid hyperstack dimensions");

    assert!(error.to_string().contains("stack size"));
}

#[test]
fn image_hyperstack_reduce_dimensionality_keeps_selected_axes() {
    let data = Array::from_shape_vec(
        (1, 1, 2, 2, 2),
        vec![
            1.0, 5.0, // z0 c0 t0,t1
            2.0, 6.0, // z0 c1 t0,t1
            3.0, 7.0, // z1 c0 t0,t1
            4.0, 8.0, // z1 c1 t0,t1
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 1),
            Dim::new(AxisKind::Z, 2),
            Dim::new(AxisKind::Channel, 2),
            Dim::new(AxisKind::Time, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.hyperstack.reduce_dimensionality",
        &dataset,
        &json!({
            "keep_channels": true,
            "keep_slices": false,
            "keep_frames": true,
            "z": 1,
            "channel": 0,
            "time": 0
        }),
    )
    .expect("reduce dimensionality");

    assert_eq!(output.dataset.shape(), &[1, 1, 2, 2]);
    assert_eq!(
        output
            .dataset
            .metadata
            .dims
            .iter()
            .map(|dim| dim.axis)
            .collect::<Vec<_>>(),
        vec![AxisKind::Y, AxisKind::X, AxisKind::Channel, AxisKind::Time]
    );
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![3.0, 7.0, 4.0, 8.0]
    );
    assert_eq!(
        output
            .dataset
            .metadata
            .extras
            .get("reduced_hyperstack_source_position"),
        Some(&json!({"channel": 0, "z": 1, "time": 0}))
    );
}

#[test]
fn image_hyperstack_reduce_dimensionality_can_create_single_plane() {
    let data = Array::from_shape_vec(
        (1, 2, 2, 2),
        vec![
            1.0, 5.0, // x0 z0 c0,c1
            2.0, 6.0, // x0 z1 c0,c1
            3.0, 7.0, // x1 z0 c0,c1
            4.0, 8.0, // x1 z1 c0,c1
        ],
    )
    .expect("shape")
    .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 2),
            Dim::new(AxisKind::Z, 2),
            Dim::new(AxisKind::Channel, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.hyperstack.reduce_dimensionality",
        &dataset,
        &json!({
            "keep_channels": false,
            "keep_slices": false,
            "keep_frames": false,
            "channel": 1,
            "z": 0
        }),
    )
    .expect("reduce to plane");

    assert_eq!(output.dataset.shape(), &[1, 2]);
    assert_eq!(
        output
            .dataset
            .metadata
            .dims
            .iter()
            .map(|dim| dim.axis)
            .collect::<Vec<_>>(),
        vec![AxisKind::Y, AxisKind::X]
    );
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![5.0, 7.0]
    );
}

#[test]
fn image_hyperstack_reduce_dimensionality_validates_inputs() {
    let stack_data = Array::from_shape_vec((1, 1, 2), vec![1.0, 2.0])
        .expect("shape")
        .into_dyn();
    let stack_metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 1),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let stack = Dataset::new(stack_data, stack_metadata).expect("stack");

    let not_hyperstack =
        execute_operation("image.hyperstack.reduce_dimensionality", &stack, &json!({}))
            .expect_err("not a hyperstack");
    assert!(not_hyperstack.to_string().contains("hyperstack"));

    let hyper_data = Array::from_shape_vec((1, 1, 2), vec![1.0, 2.0])
        .expect("shape")
        .into_dyn();
    let hyper_metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 1),
            Dim::new(AxisKind::Channel, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let hyperstack = Dataset::new(hyper_data, hyper_metadata).expect("hyperstack");

    let out_of_bounds = execute_operation(
        "image.hyperstack.reduce_dimensionality",
        &hyperstack,
        &json!({"keep_channels": false, "channel": 2}),
    )
    .expect_err("channel out of bounds");
    assert!(out_of_bounds.to_string().contains("channel"));
}

#[test]
fn image_stack_z_profile_measures_slice_means_and_rows() {
    let data = Array::from_shape_vec(
        IxDyn(&[2, 2, 3]),
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
            10.0, 11.0, 12.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let mut metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 2),
            Dim::new(AxisKind::X, 2),
            Dim::new(AxisKind::Z, 3),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    metadata.dims[2].spacing = Some(0.5);
    metadata.dims[2].unit = Some("um".to_string());
    metadata
        .extras
        .insert("z_origin_coordinate".to_string(), json!(10.0));
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output =
        execute_operation("image.stack.z_profile", &dataset, &json!({})).expect("z profile");
    assert_eq!(output.dataset.shape(), dataset.shape());
    let measurements = output.measurements.expect("measurements");
    assert_eq!(measurements.values.get("profile_axis"), Some(&json!("z")));
    assert_eq!(
        measurements.values.get("z_profile"),
        Some(&json!([5.5, 6.5, 7.5]))
    );
    assert_eq!(
        measurements.values.get("z_positions"),
        Some(&json!([10.0, 10.5, 11.0]))
    );
    assert_eq!(measurements.values.get("z_unit"), Some(&json!("um")));
    assert_eq!(
        measurements
            .values
            .get("rows")
            .and_then(serde_json::Value::as_array)
            .and_then(|rows| rows.first())
            .and_then(|row| row.get("Mean")),
        Some(&json!(5.5))
    );
}

#[test]
fn image_stack_z_profile_supports_roi_threshold_and_rejects_non_stacks() {
    let data = Array::from_shape_vec(
        IxDyn(&[2, 2, 3]),
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
        "image.stack.z_profile",
        &dataset,
        &json!({
            "left": 1,
            "top": 0,
            "width": 1,
            "height": 2,
            "min_threshold": 6.0
        }),
    )
    .expect("z profile");
    let measurements = output.measurements.expect("measurements");
    assert_eq!(
        measurements.values.get("z_profile"),
        Some(&json!([10.0, 11.0, 9.0]))
    );
    assert_eq!(measurements.values.get("counts"), Some(&json!([1, 1, 2])));

    let single = test_dataset(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    let error = execute_operation("image.stack.z_profile", &single, &json!({}))
        .expect_err("single image rejected");
    assert!(error.to_string().contains("Z"));
}

#[test]
fn image_stack_statistics_reports_imagej_style_results_row() {
    let data = Array::from_shape_vec(
        IxDyn(&[2, 2, 3]),
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
            10.0, 11.0, 12.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let mut metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 2),
            Dim::new(AxisKind::X, 2),
            Dim::new(AxisKind::Z, 3),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    metadata.dims[0].spacing = Some(2.0);
    metadata.dims[1].spacing = Some(3.0);
    metadata.dims[2].spacing = Some(4.0);
    metadata.dims[0].unit = Some("um".to_string());
    metadata.dims[1].unit = Some("um".to_string());
    metadata.dims[2].unit = Some("um".to_string());
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation("image.stack.statistics", &dataset, &json!({}))
        .expect("stack statistics");
    assert_eq!(output.dataset.shape(), dataset.shape());
    let measurements = output.measurements.expect("measurements");
    assert_eq!(measurements.values.get("voxels"), Some(&json!(12)));
    assert_eq!(measurements.values.get("volume"), Some(&json!(288.0)));
    assert_eq!(
        measurements.values.get("percent_volume"),
        Some(&json!(100.0))
    );
    assert_eq!(measurements.values.get("mean"), Some(&json!(6.5)));
    let std_dev = measurements
        .values
        .get("std_dev")
        .and_then(serde_json::Value::as_f64)
        .expect("std dev");
    assert!((std_dev - 13.0_f64.sqrt()).abs() < 1e-6);
    assert_eq!(measurements.values.get("min"), Some(&json!(1.0)));
    assert_eq!(measurements.values.get("max"), Some(&json!(12.0)));
    assert_eq!(measurements.values.get("mode"), Some(&json!(1.0)));
    assert_eq!(measurements.values.get("median"), Some(&json!(6.5)));
    assert_eq!(measurements.values.get("volume_unit"), Some(&json!("um^3")));
    assert_eq!(
        measurements
            .values
            .get("rows")
            .and_then(serde_json::Value::as_array)
            .and_then(|rows| rows.first())
            .and_then(|row| row.get("Voxels")),
        Some(&json!(12))
    );
}

#[test]
fn image_stack_statistics_supports_roi_threshold_and_rejects_non_stacks() {
    let data = Array::from_shape_vec(
        IxDyn(&[2, 2, 3]),
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
        "image.stack.statistics",
        &dataset,
        &json!({
            "left": 1,
            "top": 0,
            "width": 1,
            "height": 2,
            "min_threshold": 6.0,
            "max_threshold": 11.0
        }),
    )
    .expect("stack statistics");
    let measurements = output.measurements.expect("measurements");
    assert_eq!(measurements.values.get("voxels"), Some(&json!(3)));
    assert_eq!(
        measurements.values.get("percent_volume"),
        Some(&json!(50.0))
    );
    assert_eq!(measurements.values.get("mean"), Some(&json!(9.0)));
    assert_eq!(measurements.values.get("min"), Some(&json!(6.0)));
    assert_eq!(measurements.values.get("max"), Some(&json!(11.0)));
    assert_eq!(measurements.values.get("median"), Some(&json!(10.0)));

    let single = test_dataset(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    let error = execute_operation("image.stack.statistics", &single, &json!({}))
        .expect_err("single image rejected");
    assert!(error.to_string().contains("Z"));
}

#[test]
fn image_stack_substack_extracts_imagej_style_range_and_list() {
    let data = Array::from_shape_vec(
        IxDyn(&[2, 2, 4]),
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
    metadata.dims[2].spacing = Some(0.25);
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let range = execute_operation(
        "image.stack.substack",
        &dataset,
        &json!({"slices": "2-4-2"}),
    )
    .expect("range substack");
    assert_eq!(range.dataset.shape(), &[2, 2, 2]);
    assert_eq!(range.dataset.metadata.dims[2].spacing, Some(0.25));
    assert_eq!(
        range.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    );
    assert_eq!(
        range.dataset.metadata.extras.get("substack_slices"),
        Some(&json!([2, 4]))
    );

    let list = execute_operation(
        "image.stack.substack",
        &dataset,
        &json!({"slices": "4,1,2"}),
    )
    .expect("list substack");
    assert_eq!(list.dataset.shape(), &[2, 2, 3]);
    assert_eq!(
        list.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![
            4.0, 1.0, 2.0, //
            8.0, 5.0, 6.0, //
            12.0, 9.0, 10.0, //
            16.0, 13.0, 14.0,
        ]
    );
}

#[test]
fn image_stack_substack_supports_indices_and_validates_selection() {
    let data = Array::from_shape_vec(IxDyn(&[1, 2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("shape")
        .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 2),
            Dim::new(AxisKind::Z, 3),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.stack.substack",
        &dataset,
        &json!({"indices": [2, 0, 2]}),
    )
    .expect("index substack");
    assert_eq!(output.dataset.shape(), &[1, 2, 3]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![3.0, 1.0, 3.0, 6.0, 4.0, 6.0]
    );

    let invalid = execute_operation("image.stack.substack", &dataset, &json!({"slices": "3-1"}))
        .expect_err("reverse range rejected");
    assert!(invalid.to_string().contains("start"));

    let single = test_dataset(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    let error = execute_operation("image.stack.substack", &single, &json!({"slices": "1"}))
        .expect_err("single image rejected");
    assert!(error.to_string().contains("Z"));
}

#[test]
fn image_stack_reslice_creates_top_and_left_orthogonal_stacks() {
    let data = Array::from_shape_vec(
        IxDyn(&[2, 3, 2]),
        vec![
            0.0, 1.0, 10.0, 11.0, 20.0, 21.0, //
            100.0, 101.0, 110.0, 111.0, 120.0, 121.0,
        ],
    )
    .expect("shape")
    .into_dyn();
    let mut metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 2),
            Dim::new(AxisKind::X, 3),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    metadata.dims[0].spacing = Some(2.0);
    metadata.dims[1].spacing = Some(3.0);
    metadata.dims[2].spacing = Some(4.0);
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let top = execute_operation("image.stack.reslice", &dataset, &json!({"start": "top"}))
        .expect("top reslice");
    assert_eq!(top.dataset.shape(), &[2, 3, 2]);
    assert_eq!(
        top.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![
            0.0, 100.0, 10.0, 110.0, 20.0, 120.0, //
            1.0, 101.0, 11.0, 111.0, 21.0, 121.0,
        ]
    );
    assert_eq!(top.dataset.metadata.dims[0].axis, AxisKind::Y);
    assert_eq!(top.dataset.metadata.dims[0].spacing, Some(4.0));
    assert_eq!(top.dataset.metadata.dims[1].axis, AxisKind::X);
    assert_eq!(top.dataset.metadata.dims[1].spacing, Some(3.0));
    assert_eq!(top.dataset.metadata.dims[2].axis, AxisKind::Z);
    assert_eq!(top.dataset.metadata.dims[2].spacing, Some(2.0));

    let left = execute_operation("image.stack.reslice", &dataset, &json!({"start": "left"}))
        .expect("left reslice");
    assert_eq!(left.dataset.shape(), &[2, 2, 3]);
    assert_eq!(
        left.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![
            0.0, 10.0, 20.0, 100.0, 110.0, 120.0, //
            1.0, 11.0, 21.0, 101.0, 111.0, 121.0,
        ]
    );
    assert_eq!(left.dataset.metadata.dims[0].spacing, Some(4.0));
    assert_eq!(left.dataset.metadata.dims[1].spacing, Some(2.0));
    assert_eq!(left.dataset.metadata.dims[2].spacing, Some(3.0));
}

#[test]
fn image_stack_reslice_supports_reverse_starts_and_validates_layout() {
    let data = Array::from_shape_vec(
        IxDyn(&[2, 2, 2]),
        vec![0.0, 1.0, 10.0, 11.0, 100.0, 101.0, 110.0, 111.0],
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

    let bottom = execute_operation("image.stack.reslice", &dataset, &json!({"start": "bottom"}))
        .expect("bottom reslice");
    assert_eq!(
        bottom.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![100.0, 0.0, 110.0, 10.0, 101.0, 1.0, 111.0, 11.0]
    );

    let right = execute_operation("image.stack.reslice", &dataset, &json!({"start": "right"}))
        .expect("right reslice");
    assert_eq!(
        right.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![10.0, 0.0, 110.0, 100.0, 11.0, 1.0, 111.0, 101.0]
    );

    let single = test_dataset(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    let error = execute_operation("image.stack.reslice", &single, &json!({}))
        .expect_err("single image rejected");
    assert!(error.to_string().contains("Z"));
}

#[test]
fn image_stack_z_project_averages_and_removes_z_axis() {
    let data = Array::from_shape_vec(
        IxDyn(&[2, 2, 3]),
        vec![
            1.0, 3.0, 5.0, //
            2.0, 4.0, 6.0, //
            10.0, 12.0, 14.0, //
            20.0, 22.0, 24.0,
        ],
    )
    .expect("shape");
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
        "image.stack.z_project",
        &dataset,
        &json!({"method": "average"}),
    )
    .expect("z project");

    assert_eq!(output.dataset.shape(), &[2, 2]);
    assert_eq!(
        output
            .dataset
            .metadata
            .dims
            .iter()
            .map(|dim| dim.axis)
            .collect::<Vec<_>>(),
        vec![AxisKind::Y, AxisKind::X]
    );
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![3.0, 4.0, 12.0, 22.0]
    );
    assert_eq!(
        output.dataset.metadata.extras.get("z_projection_method"),
        Some(&json!("average"))
    );
}

#[test]
fn image_stack_z_project_supports_range_median_sd_and_rejects_invalid_bounds() {
    let data = Array::from_shape_vec(
        IxDyn(&[1, 2, 4]),
        vec![1.0, 5.0, 9.0, 13.0, 2.0, 4.0, 8.0, 16.0],
    )
    .expect("shape");
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 2),
            Dim::new(AxisKind::Z, 4),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let median = execute_operation(
        "image.stack.z_project",
        &dataset,
        &json!({"method": "median", "start": 1, "stop": 3}),
    )
    .expect("median projection");
    assert_eq!(
        median.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![9.0, 8.0]
    );

    let sd = execute_operation(
        "image.stack.z_project",
        &dataset,
        &json!({"method": "sd", "start": 0, "stop": 1}),
    )
    .expect("sd projection");
    let values = sd.dataset.data.iter().copied().collect::<Vec<_>>();
    assert!((values[0] - 2.828427).abs() < 1.0e-5);
    assert!((values[1] - 1.4142135).abs() < 1.0e-5);

    let error = execute_operation(
        "image.stack.z_project",
        &dataset,
        &json!({"method": "average", "start": 3, "stop": 1}),
    )
    .expect_err("invalid z projection range");
    assert!(error.to_string().contains("start/stop"));
}

#[test]
fn image_stack_montage_tiles_z_slices_with_borders_and_fill() {
    let data = Array::from_shape_vec(
        IxDyn(&[2, 2, 3]),
        vec![
            0.0, 10.0, 20.0, //
            1.0, 11.0, 21.0, //
            2.0, 12.0, 22.0, //
            3.0, 13.0, 23.0,
        ],
    )
    .expect("shape");
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
        "image.stack.montage",
        &dataset,
        &json!({
            "columns": 2,
            "rows": 2,
            "border_width": 1,
            "fill": -1.0
        }),
    )
    .expect("montage");

    assert_eq!(output.dataset.shape(), &[5, 5]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![
            0.0, 1.0, -1.0, 10.0, 11.0, //
            2.0, 3.0, -1.0, 12.0, 13.0, //
            -1.0, -1.0, -1.0, -1.0, -1.0, //
            20.0, 21.0, -1.0, -1.0, -1.0, //
            22.0, 23.0, -1.0, -1.0, -1.0,
        ]
    );
    assert_eq!(
        output
            .dataset
            .metadata
            .dims
            .iter()
            .map(|dim| dim.axis)
            .collect::<Vec<_>>(),
        vec![AxisKind::Y, AxisKind::X]
    );
    assert_eq!(
        output.dataset.metadata.extras.get("montage_columns"),
        Some(&json!(2))
    );
}

#[test]
fn image_stack_montage_supports_range_scale_and_rejects_invalid_grid() {
    let data = Array::from_shape_vec(
        IxDyn(&[2, 4, 4]),
        (0..32).map(|value| value as f32).collect::<Vec<_>>(),
    )
    .expect("shape");
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 2),
            Dim::new(AxisKind::X, 4),
            Dim::new(AxisKind::Z, 4),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.stack.montage",
        &dataset,
        &json!({
            "columns": 2,
            "rows": 1,
            "first": 1,
            "last": 3,
            "increment": 2,
            "scale": 0.5
        }),
    )
    .expect("scaled montage");

    assert_eq!(output.dataset.shape(), &[1, 4]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 9.0, 3.0, 11.0]
    );

    let error = execute_operation(
        "image.stack.montage",
        &dataset,
        &json!({"columns": 0, "rows": 1}),
    )
    .expect_err("invalid montage grid");
    assert!(error.to_string().contains("columns"));
}

#[test]
fn image_stack_montage_to_stack_splits_montage_grid() {
    let data = Array::from_shape_vec(
        IxDyn(&[5, 5]),
        vec![
            0.0, 1.0, -1.0, 10.0, 11.0, //
            2.0, 3.0, -1.0, 12.0, 13.0, //
            -1.0, -1.0, -1.0, -1.0, -1.0, //
            20.0, 21.0, -1.0, 30.0, 31.0, //
            22.0, 23.0, -1.0, 32.0, 33.0,
        ],
    )
    .expect("shape");
    let metadata = Metadata {
        dims: vec![Dim::new(AxisKind::Y, 5), Dim::new(AxisKind::X, 5)],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.stack.montage_to_stack",
        &dataset,
        &json!({"columns": 2, "rows": 2, "border_width": 1}),
    )
    .expect("montage to stack");

    assert_eq!(output.dataset.shape(), &[2, 2, 4]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![
            0.0, 10.0, 20.0, 30.0, //
            1.0, 11.0, 21.0, 31.0, //
            2.0, 12.0, 22.0, 32.0, //
            3.0, 13.0, 23.0, 33.0,
        ]
    );
    assert_eq!(output.dataset.metadata.dims[2].axis, AxisKind::Z);
    assert_eq!(
        output
            .dataset
            .metadata
            .extras
            .get("montage_to_stack_columns"),
        Some(&json!(2))
    );
}

#[test]
fn image_stack_montage_to_stack_uses_montage_metadata_and_rejects_stacks() {
    let data = Array::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, 2.0, 3.0, 4.0]).expect("shape");
    let mut metadata = Metadata {
        dims: vec![Dim::new(AxisKind::Y, 1), Dim::new(AxisKind::X, 4)],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    metadata
        .extras
        .insert("montage_columns".to_string(), json!(2));
    metadata.extras.insert("montage_rows".to_string(), json!(1));
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation("image.stack.montage_to_stack", &dataset, &json!({}))
        .expect("metadata-driven montage to stack");
    assert_eq!(output.dataset.shape(), &[1, 2, 2]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 3.0, 2.0, 4.0]
    );

    let stack_data = Array::from_shape_vec(IxDyn(&[1, 1, 2]), vec![1.0, 2.0]).expect("shape");
    let stack_metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 1),
            Dim::new(AxisKind::Z, 2),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let stack = Dataset::new(stack_data, stack_metadata).expect("stack");
    let error = execute_operation("image.stack.montage_to_stack", &stack, &json!({}))
        .expect_err("stack input rejected");
    assert!(error.to_string().contains("montage"));
}

#[test]
fn image_stack_grouped_z_project_projects_adjacent_groups() {
    let data = Array::from_shape_vec(
        IxDyn(&[1, 2, 4]),
        vec![1.0, 3.0, 10.0, 30.0, 2.0, 4.0, 20.0, 40.0],
    )
    .expect("shape");
    let mut metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 2),
            Dim::new(AxisKind::Z, 4),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    metadata.dims[2].spacing = Some(0.5);
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.stack.grouped_z_project",
        &dataset,
        &json!({"method": "average", "group_size": 2}),
    )
    .expect("grouped projection");

    assert_eq!(output.dataset.shape(), &[1, 2, 2]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![2.0, 20.0, 3.0, 30.0]
    );
    assert_eq!(output.dataset.metadata.dims[2].spacing, Some(1.0));
    assert_eq!(
        output
            .dataset
            .metadata
            .extras
            .get("grouped_z_projection_group_size"),
        Some(&json!(2))
    );
}

#[test]
fn image_stack_grouped_z_project_supports_max_and_rejects_non_factor_group_size() {
    let data = Array::from_shape_vec(IxDyn(&[1, 1, 6]), vec![1.0, 4.0, 2.0, 8.0, 3.0, 6.0])
        .expect("shape");
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 1),
            Dim::new(AxisKind::Z, 6),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation(
        "image.stack.grouped_z_project",
        &dataset,
        &json!({"method": "max", "group_size": 3}),
    )
    .expect("grouped max");
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![4.0, 8.0]
    );

    let error = execute_operation(
        "image.stack.grouped_z_project",
        &dataset,
        &json!({"method": "average", "group_size": 4}),
    )
    .expect_err("invalid group size");
    assert!(error.to_string().contains("group_size"));
}

#[test]
fn image_stack_reduce_keeps_every_nth_slice_and_updates_spacing() {
    let data = Array::from_shape_vec(
        IxDyn(&[1, 2, 5]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
    )
    .expect("shape");
    let mut metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 2),
            Dim::new(AxisKind::Z, 5),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    metadata.dims[2].spacing = Some(0.25);
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let output = execute_operation("image.stack.reduce", &dataset, &json!({"factor": 2}))
        .expect("reduce stack");

    assert_eq!(output.dataset.shape(), &[1, 2, 3]);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 3.0, 5.0, 10.0, 30.0, 50.0]
    );
    assert_eq!(output.dataset.metadata.dims[2].spacing, Some(0.5));
    assert_eq!(
        output.dataset.metadata.extras.get("stack_reduce_factor"),
        Some(&json!(2))
    );
}

#[test]
fn image_stack_reduce_rejects_zero_factor_and_single_slice_images() {
    let data = Array::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0, 2.0, 3.0]).expect("shape");
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 1),
            Dim::new(AxisKind::Z, 3),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let zero = execute_operation("image.stack.reduce", &dataset, &json!({"factor": 0}))
        .expect_err("zero factor");
    assert!(zero.to_string().contains("factor"));

    let single = test_dataset(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    let error = execute_operation("image.stack.reduce", &single, &json!({"factor": 2}))
        .expect_err("single slice");
    assert!(error.to_string().contains("Z"));
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
            0.0, 255.0, 0.0, //
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
    let mut dataset = test_dataset(vec![128.0; 100], (10, 10));
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
    assert!(values.iter().any(|value| *value == 255.0));
    assert!(values.iter().any(|value| *value == 128.0));

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
    let mut dataset = test_dataset(vec![128.0; 100], (10, 10));
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
    assert!(values.iter().any(|value| (*value - 128.0).abs() > 1.0e-6));
    assert!(values.iter().all(|value| (0.0..=255.0).contains(value)));
    assert!(values.iter().all(|value| value.fract() == 0.0));
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
fn find_edges_preserves_integer_sample_range() {
    let mut dataset = test_dataset(
        vec![
            0.0, 0.0, 0.0, //
            0.0, 255.0, 255.0, //
            0.0, 255.0, 255.0, //
        ],
        (3, 3),
    );
    dataset.metadata.pixel_type = PixelType::U8;

    let output = execute_operation("image.find_edges", &dataset, &json!({})).expect("edges");

    assert_eq!(output.dataset.metadata.pixel_type, PixelType::U8);
    let center = output.dataset.data[IxDyn(&[1, 1])];
    assert!(center > 1.0);
    assert!(center <= 255.0);
}

#[test]
fn image_find_maxima_marks_prominent_local_peaks() {
    let dataset = test_dataset(
        vec![
            0.0, 0.0, 0.0, //
            0.0, 5.0, 1.0, //
            0.0, 1.0, 0.0, //
        ],
        (3, 3),
    );

    let output = execute_operation(
        "image.find_maxima",
        &dataset,
        &json!({
            "prominence": 3.0
        }),
    )
    .expect("find maxima");

    assert_eq!(output.dataset.metadata.pixel_type, PixelType::U8);
    assert_eq!(output.dataset.data[IxDyn(&[1, 1])], 255.0);
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
fn image_find_maxima_supports_edges_and_light_background_minima() {
    let edge_peak = test_dataset(
        vec![
            9.0, 1.0, 0.0, //
            1.0, 2.0, 0.0, //
            0.0, 0.0, 0.0, //
        ],
        (3, 3),
    );
    let included =
        execute_operation("image.find_maxima", &edge_peak, &json!({})).expect("include edges");
    assert_eq!(included.dataset.data[IxDyn(&[0, 0])], 255.0);
    let excluded = execute_operation(
        "image.find_maxima",
        &edge_peak,
        &json!({"exclude_edges": true}),
    )
    .expect("exclude edges");
    assert_eq!(excluded.dataset.data[IxDyn(&[0, 0])], 0.0);

    let valley = test_dataset(
        vec![
            5.0, 5.0, 5.0, //
            5.0, 0.0, 5.0, //
            5.0, 5.0, 5.0, //
        ],
        (3, 3),
    );
    let minima = execute_operation(
        "image.find_maxima",
        &valley,
        &json!({
            "prominence": 1.0,
            "light_background": true
        }),
    )
    .expect("find minima");
    assert_eq!(minima.dataset.data[IxDyn(&[1, 1])], 255.0);
}

#[test]
fn image_find_maxima_rejects_invalid_prominence() {
    let dataset = test_dataset(vec![0.0; 9], (3, 3));

    let error = execute_operation(
        "image.find_maxima",
        &dataset,
        &json!({
            "prominence": -1.0
        }),
    )
    .expect_err("invalid prominence");
    assert!(error.to_string().contains("prominence"));
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
fn image_rank_filter_light_top_hat_uses_integer_type_max() {
    let mut dataset = test_dataset(
        vec![
            255.0, 255.0, 255.0, //
            255.0, 0.0, 255.0, //
            255.0, 255.0, 255.0, //
        ],
        (3, 3),
    );
    dataset.metadata.pixel_type = PixelType::U8;

    let output = execute_operation(
        "image.rank_filter",
        &dataset,
        &json!({
            "filter": "top_hat",
            "radius": 1.0,
            "light_background": true
        }),
    )
    .expect("top hat");

    assert_eq!(output.dataset.data[IxDyn(&[1, 1])], 0.0);
    assert_eq!(output.dataset.data[IxDyn(&[0, 0])], 255.0);
    assert_eq!(output.dataset.metadata.pixel_type, PixelType::U8);
}

#[test]
fn image_subtract_background_subtracts_dark_background_estimate() {
    let dataset = test_dataset(
        vec![
            1.0, 1.0, 1.0, //
            1.0, 10.0, 1.0, //
            1.0, 1.0, 1.0, //
        ],
        (3, 3),
    );

    let output = execute_operation(
        "image.subtract_background",
        &dataset,
        &json!({
            "radius": 1.0,
            "light_background": false
        }),
    )
    .expect("subtract background");

    assert_eq!(output.dataset.data[IxDyn(&[1, 1])], 9.0);
    assert_eq!(output.dataset.data[IxDyn(&[0, 0])], 0.0);
}

#[test]
fn image_subtract_background_can_create_background_and_rejects_invalid_radius() {
    let dataset = test_dataset(
        vec![
            1.0, 1.0, 1.0, //
            1.0, 10.0, 1.0, //
            1.0, 1.0, 1.0, //
        ],
        (3, 3),
    );

    let background = execute_operation(
        "image.subtract_background",
        &dataset,
        &json!({
            "radius": 1.0,
            "light_background": false,
            "create_background": true
        }),
    )
    .expect("create background");
    assert_eq!(background.dataset.data[IxDyn(&[1, 1])], 1.0);

    let error = execute_operation(
        "image.subtract_background",
        &dataset,
        &json!({
            "radius": -1.0
        }),
    )
    .expect_err("invalid radius");
    assert!(error.to_string().contains("radius"));
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
    assert!(constant_plane.iter().all(|value| *value <= 1.0));
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
fn morphology_erode_honors_iterations() {
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

    let once = execute_operation(
        "morphology.erode",
        &dataset,
        &json!({
            "iterations": 1
        }),
    )
    .expect("erode once");
    assert_eq!(once.dataset.data[IxDyn(&[2, 2])], 1.0);

    let twice = execute_operation(
        "morphology.erode",
        &dataset,
        &json!({
            "iterations": 2
        }),
    )
    .expect("erode twice");
    assert_eq!(twice.dataset.data[IxDyn(&[2, 2])], 0.0);
}

#[test]
fn morphology_erode_rejects_zero_iterations() {
    let dataset = test_dataset(vec![1.0; 9], (3, 3));
    let error = execute_operation(
        "morphology.erode",
        &dataset,
        &json!({
            "iterations": 0
        }),
    )
    .expect_err("zero iterations should fail");

    assert!(error.to_string().contains("iterations"));
}

#[test]
fn morphology_erode_honors_neighbor_count() {
    let dataset = test_dataset(
        vec![
            0.0, 1.0, 1.0, //
            1.0, 1.0, 1.0, //
            1.0, 1.0, 1.0,
        ],
        (3, 3),
    );

    let count_one = execute_operation(
        "morphology.erode",
        &dataset,
        &json!({
            "count": 1
        }),
    )
    .expect("erode count one");
    assert_eq!(count_one.dataset.data[IxDyn(&[1, 1])], 0.0);

    let count_two = execute_operation(
        "morphology.erode",
        &dataset,
        &json!({
            "count": 2
        }),
    )
    .expect("erode count two");
    assert_eq!(count_two.dataset.data[IxDyn(&[1, 1])], 1.0);
}

#[test]
fn morphology_dilate_honors_neighbor_count() {
    let dataset = test_dataset(
        vec![
            0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
        (3, 3),
    );

    let count_one = execute_operation(
        "morphology.dilate",
        &dataset,
        &json!({
            "count": 1
        }),
    )
    .expect("dilate count one");
    assert_eq!(count_one.dataset.data[IxDyn(&[1, 1])], 1.0);

    let count_two = execute_operation(
        "morphology.dilate",
        &dataset,
        &json!({
            "count": 2
        }),
    )
    .expect("dilate count two");
    assert_eq!(count_two.dataset.data[IxDyn(&[1, 1])], 0.0);
}

#[test]
fn morphology_erode_rejects_zero_count() {
    let dataset = test_dataset(vec![1.0; 9], (3, 3));
    let error = execute_operation(
        "morphology.erode",
        &dataset,
        &json!({
            "count": 0
        }),
    )
    .expect_err("zero count should fail");

    assert!(error.to_string().contains("count"));
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
fn window_operation_uses_integer_pixel_type_range() {
    let mut dataset = test_dataset(vec![0.0, 5.0, 10.0, 20.0], (2, 2));
    dataset.metadata.pixel_type = PixelType::U8;

    let output = execute_operation(
        "intensity.window",
        &dataset,
        &json!({
            "low": 0.0,
            "high": 10.0
        }),
    )
    .expect("window");

    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 128.0, 255.0, 255.0]
    );
    assert_eq!(output.dataset.metadata.pixel_type, PixelType::U8);
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
fn make_binary_threshold_creates_imagej_style_u8_mask() {
    let dataset = test_dataset(vec![0.0, 0.1, 0.2, 0.8, 0.9, 1.0], (2, 3));

    let output = execute_operation(
        "threshold.make_binary",
        &dataset,
        &json!({"method": "fixed", "threshold": 0.5, "background": "dark"}),
    )
    .expect("make binary");

    assert_eq!(output.dataset.metadata.pixel_type, PixelType::U8);
    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 0.0, 0.0, 255.0, 255.0, 255.0]
    );
    assert_eq!(
        output.dataset.metadata.extras.get("threshold_min"),
        Some(&json!(0.5))
    );
    assert_eq!(
        output.dataset.metadata.extras.get("threshold_max"),
        Some(&json!(1.0))
    );
}

#[test]
fn make_binary_threshold_supports_light_background_and_explicit_range() {
    let dataset = test_dataset(vec![0.0, 0.25, 0.5, 0.75, 1.0, f32::NAN], (2, 3));

    let light = execute_operation(
        "threshold.make_binary",
        &dataset,
        &json!({"method": "fixed", "threshold": 0.5, "background": "light"}),
    )
    .expect("make binary light background");
    assert_eq!(
        light.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![255.0, 255.0, 255.0, 0.0, 0.0, 0.0]
    );

    let ranged = execute_operation(
        "threshold.make_binary",
        &dataset,
        &json!({"min": 0.25, "max": 0.75}),
    )
    .expect("make binary explicit range");
    assert_eq!(
        ranged.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 255.0, 255.0, 255.0, 0.0, 0.0]
    );
}

#[test]
fn make_binary_threshold_defaults_to_imagej_isodata_polarity() {
    let dataset = test_dataset(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], (2, 3));

    let output =
        execute_operation("threshold.make_binary", &dataset, &json!({})).expect("make binary");

    assert_eq!(
        output.dataset.data.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 0.0, 0.0, 255.0, 255.0, 255.0]
    );
    let threshold = output
        .measurements
        .as_ref()
        .and_then(|table| table.values.get("threshold"))
        .and_then(|value| value.as_f64())
        .expect("threshold");
    assert!((threshold - 0.501960813999176).abs() < 1.0e-12);
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

#[test]
fn measurements_histogram_reports_bins_and_rows() {
    let dataset = test_dataset(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], (2, 3));

    let output = execute_operation(
        "measurements.histogram",
        &dataset,
        &json!({"bins": 3, "min": 0.0, "max": 6.0}),
    )
    .expect("histogram");
    let measurements = output.measurements.expect("measurements");

    assert_eq!(measurements.values.get("bins"), Some(&json!(3)));
    assert_eq!(
        measurements.values.get("histogram"),
        Some(&json!([2, 2, 2]))
    );
    assert_eq!(measurements.values.get("pixel_count"), Some(&json!(6)));
    assert_eq!(measurements.values.get("min"), Some(&json!(0.0)));
    assert_eq!(measurements.values.get("max"), Some(&json!(6.0)));
    assert_eq!(
        measurements
            .values
            .get("rows")
            .and_then(serde_json::Value::as_array)
            .and_then(|rows| rows.get(1))
            .and_then(|row| row.get("Count")),
        Some(&json!(2))
    );
}

#[test]
fn measurements_histogram_supports_slice_and_stack_modes() {
    let data = Array::from_shape_vec(IxDyn(&[1, 2, 3]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("shape")
        .into_dyn();
    let metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, 1),
            Dim::new(AxisKind::X, 2),
            Dim::new(AxisKind::Z, 3),
        ],
        pixel_type: PixelType::F32,
        ..Metadata::default()
    };
    let dataset = Dataset::new(data, metadata).expect("dataset");

    let slice = execute_operation(
        "measurements.histogram",
        &dataset,
        &json!({"bins": 2, "z": 1, "min": 0.0, "max": 6.0}),
    )
    .expect("slice histogram");
    assert_eq!(
        slice
            .measurements
            .expect("measurements")
            .values
            .get("histogram"),
        Some(&json!([1, 1]))
    );

    let stack = execute_operation(
        "measurements.histogram",
        &dataset,
        &json!({"bins": 2, "stack": true, "min": 0.0, "max": 6.0}),
    )
    .expect("stack histogram");
    let measurements = stack.measurements.expect("measurements");
    assert_eq!(measurements.values.get("histogram"), Some(&json!([3, 3])));
    assert_eq!(
        measurements.values.get("stack_histogram"),
        Some(&json!(true))
    );

    let error = execute_operation("measurements.histogram", &dataset, &json!({"bins": 0}))
        .expect_err("zero bins rejected");
    assert!(error.to_string().contains("bins"));
}

#[test]
fn measurements_profile_reports_rectangular_column_means() {
    let dataset = test_dataset(
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
        ],
        (3, 3),
    );

    let output = execute_operation(
        "measurements.profile",
        &dataset,
        &json!({"left": 0, "top": 0, "width": 3, "height": 2}),
    )
    .expect("profile");
    let measurements = output.measurements.expect("measurements");

    assert_eq!(measurements.values.get("profile_axis"), Some(&json!("x")));
    assert_eq!(
        measurements.values.get("profile"),
        Some(&json!([2.5, 3.5, 4.5]))
    );
    assert_eq!(measurements.values.get("sample_count"), Some(&json!(3)));
    assert_eq!(
        measurements
            .values
            .get("rows")
            .and_then(serde_json::Value::as_array)
            .and_then(|rows| rows.first())
            .and_then(|row| row.get("Value")),
        Some(&json!(2.5))
    );
}

#[test]
fn measurements_profile_supports_vertical_rectangles_lines_and_selection_errors() {
    let dataset = test_dataset(
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
        ],
        (3, 3),
    );

    let vertical = execute_operation(
        "measurements.profile",
        &dataset,
        &json!({"left": 0, "top": 0, "width": 3, "height": 2, "vertical": true}),
    )
    .expect("vertical profile");
    assert_eq!(
        vertical
            .measurements
            .expect("measurements")
            .values
            .get("profile"),
        Some(&json!([2.0, 5.0]))
    );

    let line = execute_operation(
        "measurements.profile",
        &dataset,
        &json!({"x0": 0.0, "y0": 0.0, "x1": 2.0, "y1": 0.0}),
    )
    .expect("line profile");
    let measurements = line.measurements.expect("measurements");
    assert_eq!(
        measurements.values.get("profile_axis"),
        Some(&json!("line"))
    );
    assert_eq!(
        measurements.values.get("profile"),
        Some(&json!([1.0, 2.0, 3.0]))
    );

    let error = execute_operation("measurements.profile", &dataset, &json!({}))
        .expect_err("selection required");
    assert!(error.to_string().contains("line or rectangular selection"));
}
