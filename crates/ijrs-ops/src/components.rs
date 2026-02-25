use std::collections::VecDeque;

use ijrs_core::{Dataset, DatasetF32};
use ndarray::{Array, ArrayD, Dimension, IxDyn};
use serde_json::{Value, json};

use crate::{
    MeasurementTable, OpOutput, OpSchema, Operation, OpsError, Result, spatial_axes,
    util::neighborhood_offsets,
};

#[derive(Debug, Clone, Copy)]
pub struct ComponentsLabelOp;

impl Operation for ComponentsLabelOp {
    fn name(&self) -> &'static str {
        "components.label"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Connected-component labeling on binary input.".to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        let axes = spatial_axes(dataset);
        if axes.is_empty() {
            return Err(OpsError::UnsupportedLayout(
                "dataset has no spatial axes".to_string(),
            ));
        }
        let offsets = neighborhood_offsets(axes.len(), 1, false);
        let mut labels = ArrayD::<u32>::zeros(IxDyn(dataset.shape()));
        let mut next_label = 1_u32;

        for (index, value) in dataset.data.indexed_iter() {
            if *value <= 0.5 || labels[index.clone()] != 0 {
                continue;
            }

            let mut queue = VecDeque::new();
            queue.push_back(index.slice().to_vec());
            labels[IxDyn(index.slice())] = next_label;

            while let Some(point) = queue.pop_front() {
                for offset in &offsets {
                    let mut candidate = point.clone();
                    let mut out_of_bounds = false;
                    for (offset_axis, data_axis) in axes.iter().enumerate() {
                        let size = dataset.shape()[*data_axis] as isize;
                        let next = candidate[*data_axis] as isize + offset[offset_axis];
                        if next < 0 || next >= size {
                            out_of_bounds = true;
                            break;
                        }
                        candidate[*data_axis] = next as usize;
                    }
                    if out_of_bounds {
                        continue;
                    }

                    let candidate_idx = IxDyn(&candidate);
                    if dataset.data[candidate_idx.clone()] <= 0.5
                        || labels[candidate_idx.clone()] != 0
                    {
                        continue;
                    }
                    labels[candidate_idx] = next_label;
                    queue.push_back(candidate);
                }
            }

            next_label += 1;
        }

        let labeled_values = labels.iter().map(|value| *value as f32).collect::<Vec<_>>();
        let labeled_array = Array::from_shape_vec(IxDyn(dataset.shape()), labeled_values)
            .expect("shape is unchanged and valid");
        let output_dataset = Dataset::new(labeled_array, dataset.metadata.clone())?;

        let mut measurements = MeasurementTable::default();
        measurements.values.insert(
            "component_count".to_string(),
            json!(next_label.saturating_sub(1)),
        );
        Ok(OpOutput {
            dataset: output_dataset,
            measurements: Some(measurements),
        })
    }
}
