use crate::model::{AxisKind, DatasetF32};

pub(crate) fn spatial_axes(dataset: &DatasetF32) -> Vec<usize> {
    dataset
        .metadata
        .dims
        .iter()
        .enumerate()
        .filter_map(|(index, dim)| match dim.axis {
            AxisKind::X | AxisKind::Y | AxisKind::Z | AxisKind::Unknown => Some(index),
            AxisKind::Channel | AxisKind::Time => None,
        })
        .collect()
}
