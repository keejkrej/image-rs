use crate::model::DatasetF32;
use serde_json::{Value, json};

use super::{MeasurementTable, OpOutput, OpSchema, Operation, Result};

#[derive(Debug, Clone, Copy)]
pub struct MeasurementsSummaryOp;

impl Operation for MeasurementsSummaryOp {
    fn name(&self) -> &'static str {
        "measurements.summary"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Compute area/volume, min/max/mean, centroid and bounding box."
                .to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0_f64;
        let mut count = 0usize;

        let rank = dataset.ndim();
        let mut nonzero_count = 0usize;
        let mut centroid_accumulator = vec![0.0_f64; rank];
        let mut bbox_min = vec![usize::MAX; rank];
        let mut bbox_max = vec![0usize; rank];

        for (index, value) in dataset.data.indexed_iter() {
            min = min.min(*value);
            max = max.max(*value);
            sum += f64::from(*value);
            count += 1;

            if *value > 0.0 {
                nonzero_count += 1;
                for axis in 0..rank {
                    let coordinate = index[axis];
                    centroid_accumulator[axis] += coordinate as f64;
                    bbox_min[axis] = bbox_min[axis].min(coordinate);
                    bbox_max[axis] = bbox_max[axis].max(coordinate);
                }
            }
        }

        let centroid = if nonzero_count > 0 {
            centroid_accumulator
                .iter()
                .map(|value| value / nonzero_count as f64)
                .collect::<Vec<_>>()
        } else {
            vec![0.0; rank]
        };

        let mut measurements = MeasurementTable::default();
        measurements.values.insert("min".to_string(), json!(min));
        measurements.values.insert("max".to_string(), json!(max));
        measurements
            .values
            .insert("mean".to_string(), json!(sum / count.max(1) as f64));
        measurements
            .values
            .insert("area".to_string(), json!(nonzero_count));
        measurements
            .values
            .insert("volume".to_string(), json!(nonzero_count));
        measurements
            .values
            .insert("centroid".to_string(), json!(centroid));
        measurements.values.insert(
            "bbox_min".to_string(),
            json!(
                bbox_min
                    .iter()
                    .map(|value| if *value == usize::MAX { 0 } else { *value })
                    .collect::<Vec<_>>()
            ),
        );
        measurements
            .values
            .insert("bbox_max".to_string(), json!(bbox_max));

        Ok(OpOutput {
            dataset: dataset.clone(),
            measurements: Some(measurements),
        })
    }
}

