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
    assert!(names.contains(&"components.label".to_string()));
    assert!(names.contains(&"measurements.summary".to_string()));
    #[cfg(feature = "morpholib")]
    assert!(names.contains(&"morpholibj.distance.chamfer".to_string()));
    #[cfg(feature = "thunderstorm")]
    assert!(names.contains(&"thunderstorm.pipeline.localize".to_string()));
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
fn window_operation_validates_bounds() {
    let dataset = test_dataset(vec![0.0, 0.5, 0.75, 1.0], (2, 2));
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
