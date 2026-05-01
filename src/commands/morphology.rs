use crate::model::{AxisKind, Dataset, DatasetF32, PixelType};
use ndarray::{Dimension, IxDyn};
use serde_json::Value;

use super::{
    OpOutput, OpSchema, Operation, OpsError, ParamSpec, Result, get_optional_usize, spatial_axes,
    util::neighborhood_offsets,
};

#[derive(Debug, Clone, Copy)]
enum MorphologyKind {
    Erode,
    Dilate,
}

fn apply_morphology(
    dataset: &DatasetF32,
    radius: usize,
    kind: MorphologyKind,
) -> Result<ndarray::ArrayD<f32>> {
    let axes = spatial_axes(dataset);
    if axes.is_empty() {
        return Err(OpsError::UnsupportedLayout(
            "dataset has no spatial axes".to_string(),
        ));
    }
    let offsets = neighborhood_offsets(axes.len(), radius, true);
    let shape = dataset.shape().to_vec();
    let mut output = dataset.data.clone();

    for (index, value) in output.indexed_iter_mut() {
        let mut aggregate = match kind {
            MorphologyKind::Erode => 1.0_f32,
            MorphologyKind::Dilate => 0.0_f32,
        };

        for offset in &offsets {
            let mut coord = index.slice().to_vec();
            for (offset_axis, data_axis) in axes.iter().enumerate() {
                let axis_size = shape[*data_axis] as isize;
                let candidate = coord[*data_axis] as isize + offset[offset_axis];
                let clamped = candidate.clamp(0, axis_size - 1) as usize;
                coord[*data_axis] = clamped;
            }
            let binary = if dataset.data[IxDyn(&coord)] > 0.5 {
                1.0_f32
            } else {
                0.0_f32
            };
            match kind {
                MorphologyKind::Erode => aggregate = aggregate.min(binary),
                MorphologyKind::Dilate => aggregate = aggregate.max(binary),
            }
        }
        *value = aggregate;
    }

    Ok(output)
}

#[derive(Debug, Clone, Copy)]
pub struct MorphologyErodeOp;

impl Operation for MorphologyErodeOp {
    fn name(&self) -> &'static str {
        "morphology.erode"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Binary erosion over spatial axes.".to_string(),
            params: vec![ParamSpec {
                name: "radius".to_string(),
                description: "Neighborhood radius.".to_string(),
                required: false,
                kind: "int".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let radius = get_optional_usize(params, "radius", 1);
        let eroded = apply_morphology(dataset, radius, MorphologyKind::Erode)?;
        let output_dataset = Dataset::new(eroded, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MorphologyDilateOp;

impl Operation for MorphologyDilateOp {
    fn name(&self) -> &'static str {
        "morphology.dilate"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Binary dilation over spatial axes.".to_string(),
            params: vec![ParamSpec {
                name: "radius".to_string(),
                description: "Neighborhood radius.".to_string(),
                required: false,
                kind: "int".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let radius = get_optional_usize(params, "radius", 1);
        let dilated = apply_morphology(dataset, radius, MorphologyKind::Dilate)?;
        let output_dataset = Dataset::new(dilated, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MorphologyOpenOp;

impl Operation for MorphologyOpenOp {
    fn name(&self) -> &'static str {
        "morphology.open"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Binary opening (erode then dilate).".to_string(),
            params: vec![ParamSpec {
                name: "radius".to_string(),
                description: "Neighborhood radius.".to_string(),
                required: false,
                kind: "int".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let radius = get_optional_usize(params, "radius", 1);
        let eroded = apply_morphology(dataset, radius, MorphologyKind::Erode)?;
        let intermediate = Dataset::new(eroded, dataset.metadata.clone())?;
        let opened = apply_morphology(&intermediate, radius, MorphologyKind::Dilate)?;
        let output_dataset = Dataset::new(opened, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MorphologyCloseOp;

#[derive(Debug, Clone, Copy)]
pub struct MorphologyFillHolesOp;

#[derive(Debug, Clone, Copy)]
pub struct MorphologyOutlineOp;

#[derive(Debug, Clone, Copy)]
pub struct MorphologySkeletonizeOp;

#[derive(Debug, Clone, Copy)]
pub struct MorphologyBinaryMedianOp;

#[derive(Debug, Clone, Copy)]
pub struct MorphologyDistanceMapOp;

#[derive(Debug, Clone, Copy)]
pub struct MorphologyUltimatePointsOp;

#[derive(Debug, Clone, Copy)]
pub struct MorphologyWatershedOp;

#[derive(Debug, Clone, Copy)]
pub struct MorphologyVoronoiOp;

impl Operation for MorphologyCloseOp {
    fn name(&self) -> &'static str {
        "morphology.close"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Binary closing (dilate then erode).".to_string(),
            params: vec![ParamSpec {
                name: "radius".to_string(),
                description: "Neighborhood radius.".to_string(),
                required: false,
                kind: "int".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let radius = get_optional_usize(params, "radius", 1);
        let dilated = apply_morphology(dataset, radius, MorphologyKind::Dilate)?;
        let intermediate = Dataset::new(dilated, dataset.metadata.clone())?;
        let closed = apply_morphology(&intermediate, radius, MorphologyKind::Erode)?;
        let output_dataset = Dataset::new(closed, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

impl Operation for MorphologyFillHolesOp {
    fn name(&self) -> &'static str {
        "morphology.fill_holes"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Fill enclosed background regions in binary X/Y planes.".to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        let filled = fill_holes_xy(dataset)?;
        let output_dataset = Dataset::new(filled, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

impl Operation for MorphologyOutlineOp {
    fn name(&self) -> &'static str {
        "morphology.outline"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Convert binary foreground regions to their X/Y outlines.".to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        let outlined = outline_xy(dataset)?;
        let output_dataset = Dataset::new(outlined, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

impl Operation for MorphologySkeletonizeOp {
    fn name(&self) -> &'static str {
        "morphology.skeletonize"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Thin binary foreground regions to single-pixel X/Y skeletons."
                .to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        let skeleton = skeletonize_xy(dataset)?;
        let output_dataset = Dataset::new(skeleton, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

impl Operation for MorphologyBinaryMedianOp {
    fn name(&self) -> &'static str {
        "morphology.binary_median"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Apply ImageJ-style binary median smoothing on X/Y planes.".to_string(),
            params: vec![ParamSpec {
                name: "radius".to_string(),
                description: "Circular neighborhood radius; ImageJ default is 3.".to_string(),
                required: false,
                kind: "int".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let radius = get_optional_usize(params, "radius", 3);
        let filtered = binary_median_xy(dataset, radius)?;
        let output_dataset = Dataset::new(filtered, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

impl Operation for MorphologyDistanceMapOp {
    fn name(&self) -> &'static str {
        "morphology.distance_map"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Compute an X/Y Euclidean distance map for binary foreground.".to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        let data = distance_map_xy(dataset)?;
        let mut metadata = dataset.metadata.clone();
        metadata.pixel_type = PixelType::F32;
        let output_dataset = Dataset::new(data, metadata)?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

impl Operation for MorphologyUltimatePointsOp {
    fn name(&self) -> &'static str {
        "morphology.ultimate_points"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Mark ultimate eroded points from the X/Y Euclidean distance map."
                .to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        let data = ultimate_points_xy(dataset)?;
        let mut metadata = dataset.metadata.clone();
        metadata.pixel_type = PixelType::F32;
        let output_dataset = Dataset::new(data, metadata)?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

impl Operation for MorphologyWatershedOp {
    fn name(&self) -> &'static str {
        "morphology.watershed"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Split touching binary foreground regions using EDM watershed."
                .to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        let segmented = watershed_xy(dataset)?;
        let output_dataset = Dataset::new(segmented, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

impl Operation for MorphologyVoronoiOp {
    fn name(&self) -> &'static str {
        "morphology.voronoi"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Create X/Y Voronoi divider lines between binary particles.".to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        let data = voronoi_xy(dataset)?;
        let mut metadata = dataset.metadata.clone();
        metadata.pixel_type = PixelType::F32;
        let output_dataset = Dataset::new(data, metadata)?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

fn outline_xy(dataset: &DatasetF32) -> Result<ndarray::ArrayD<f32>> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let mut output = dataset.data.clone();

    iterate_indices(&shape, |coord| {
        if dataset.data[IxDyn(coord)] <= 0.5 {
            output[IxDyn(coord)] = 0.0;
            return;
        }
        let x = coord[x_axis] as isize;
        let y = coord[y_axis] as isize;
        let edge = (-1..=1).any(|dy| {
            (-1..=1).any(|dx| {
                if dx == 0 && dy == 0 {
                    return false;
                }
                let neighbor_x = x + dx;
                let neighbor_y = y + dy;
                if neighbor_x < 0
                    || neighbor_y < 0
                    || neighbor_x >= shape[x_axis] as isize
                    || neighbor_y >= shape[y_axis] as isize
                {
                    return true;
                }
                let mut neighbor = coord.to_vec();
                neighbor[x_axis] = neighbor_x as usize;
                neighbor[y_axis] = neighbor_y as usize;
                dataset.data[IxDyn(&neighbor)] <= 0.5
            })
        });
        output[IxDyn(coord)] = if edge { 1.0 } else { 0.0 };
    });

    Ok(output)
}

#[derive(Clone, Copy)]
struct VoronoiCell {
    label: isize,
    distance_squared: usize,
    tied: bool,
}

fn voronoi_xy(dataset: &DatasetF32) -> Result<ndarray::ArrayD<f32>> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let width = shape[x_axis];
    let height = shape[y_axis];
    let mut output = ndarray::ArrayD::<f32>::zeros(IxDyn(&shape));
    let plane_shape = shape
        .iter()
        .enumerate()
        .filter_map(|(axis, size)| (axis != x_axis && axis != y_axis).then_some(*size))
        .collect::<Vec<_>>();

    iterate_indices(&plane_shape, |plane_coord| {
        let base_coord = plane_base_coord(&shape, x_axis, y_axis, plane_coord);
        let labels = label_foreground_plane(dataset, &base_coord, x_axis, y_axis, width, height);
        let particles = labels
            .iter()
            .enumerate()
            .filter_map(|(index, label)| {
                (*label > 0).then_some((index % width, index / width, *label))
            })
            .collect::<Vec<_>>();
        let particle_count = particles
            .iter()
            .map(|(_, _, label)| *label)
            .max()
            .unwrap_or(0);
        if particle_count < 2 {
            return;
        }

        let mut nearest = vec![None; width * height];
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                if labels[index] > 0 {
                    continue;
                }
                nearest[index] = nearest_particle(x, y, &particles);
            }
        }

        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                if labels[index] > 0 {
                    continue;
                }
                let Some(cell) = nearest[index] else {
                    continue;
                };
                if cell.tied {
                    let mut coord = base_coord.clone();
                    coord[x_axis] = x;
                    coord[y_axis] = y;
                    output[IxDyn(&coord)] = (cell.distance_squared as f32).sqrt();
                }
            }
        }
    });

    Ok(output)
}

fn label_foreground_plane(
    dataset: &DatasetF32,
    base_coord: &[usize],
    x_axis: usize,
    y_axis: usize,
    width: usize,
    height: usize,
) -> Vec<isize> {
    let mut labels = vec![0isize; width * height];
    let mut next_label = 1isize;

    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if labels[index] != 0 {
                continue;
            }
            let mut coord = base_coord.to_vec();
            coord[x_axis] = x;
            coord[y_axis] = y;
            if dataset.data[IxDyn(&coord)] <= 0.5 {
                continue;
            }
            flood_foreground_label(
                dataset,
                base_coord,
                x_axis,
                y_axis,
                width,
                height,
                &mut labels,
                x,
                y,
                next_label,
            );
            next_label += 1;
        }
    }

    labels
}

fn flood_foreground_label(
    dataset: &DatasetF32,
    base_coord: &[usize],
    x_axis: usize,
    y_axis: usize,
    width: usize,
    height: usize,
    labels: &mut [isize],
    start_x: usize,
    start_y: usize,
    label: isize,
) {
    let mut stack = vec![(start_x, start_y)];
    while let Some((x, y)) = stack.pop() {
        let index = y * width + x;
        if labels[index] != 0 {
            continue;
        }
        let mut coord = base_coord.to_vec();
        coord[x_axis] = x;
        coord[y_axis] = y;
        if dataset.data[IxDyn(&coord)] <= 0.5 {
            continue;
        }
        labels[index] = label;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let neighbor_x = x as isize + dx;
                let neighbor_y = y as isize + dy;
                if neighbor_x < 0
                    || neighbor_y < 0
                    || neighbor_x >= width as isize
                    || neighbor_y >= height as isize
                {
                    continue;
                }
                stack.push((neighbor_x as usize, neighbor_y as usize));
            }
        }
    }
}

fn nearest_particle(
    x: usize,
    y: usize,
    particles: &[(usize, usize, isize)],
) -> Option<VoronoiCell> {
    let mut nearest = None;
    for (particle_x, particle_y, label) in particles {
        let dx = x.abs_diff(*particle_x);
        let dy = y.abs_diff(*particle_y);
        let distance_squared = dx * dx + dy * dy;
        match nearest {
            None => {
                nearest = Some(VoronoiCell {
                    label: *label,
                    distance_squared,
                    tied: false,
                });
            }
            Some(mut cell) if distance_squared < cell.distance_squared => {
                cell.label = *label;
                cell.distance_squared = distance_squared;
                cell.tied = false;
                nearest = Some(cell);
            }
            Some(mut cell) if distance_squared == cell.distance_squared && cell.label != *label => {
                cell.tied = true;
                nearest = Some(cell);
            }
            _ => {}
        }
    }
    nearest
}

fn watershed_xy(dataset: &DatasetF32) -> Result<ndarray::ArrayD<f32>> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let width = shape[x_axis];
    let height = shape[y_axis];
    let edm = distance_map_xy(dataset)?;
    let seeds = ultimate_points_xy(dataset)?;
    let mut output = dataset.data.clone();
    let plane_shape = shape
        .iter()
        .enumerate()
        .filter_map(|(axis, size)| (axis != x_axis && axis != y_axis).then_some(*size))
        .collect::<Vec<_>>();

    iterate_indices(&plane_shape, |plane_coord| {
        let base_coord = plane_base_coord(&shape, x_axis, y_axis, plane_coord);
        let mut labels = vec![0isize; width * height];
        let mut foreground = Vec::new();
        let mut next_label = 1isize;

        for y in 0..height {
            for x in 0..width {
                let mut coord = base_coord.clone();
                coord[x_axis] = x;
                coord[y_axis] = y;
                if dataset.data[IxDyn(&coord)] <= 0.5 {
                    output[IxDyn(&coord)] = 0.0;
                    continue;
                }
                foreground.push((x, y, edm[IxDyn(&coord)]));
                if seeds[IxDyn(&coord)] > 0.0 {
                    labels[y * width + x] = next_label;
                    next_label += 1;
                }
            }
        }

        if next_label <= 2 {
            for (x, y, _) in foreground {
                let mut coord = base_coord.clone();
                coord[x_axis] = x;
                coord[y_axis] = y;
                output[IxDyn(&coord)] = 1.0;
            }
            return;
        }

        foreground
            .sort_by(|(_, _, a), (_, _, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut changed = true;
        while changed {
            changed = false;
            for (x, y, _) in &foreground {
                let index = y * width + x;
                if labels[index] != 0 {
                    continue;
                }
                match unique_neighbor_label(&labels, width, height, *x, *y) {
                    NeighborLabels::None => {}
                    NeighborLabels::One(label) => {
                        labels[index] = label;
                        changed = true;
                    }
                    NeighborLabels::Multiple => {
                        labels[index] = -1;
                        changed = true;
                    }
                }
            }
        }

        for (x, y, _) in foreground {
            let index = y * width + x;
            let mut coord = base_coord.clone();
            coord[x_axis] = x;
            coord[y_axis] = y;
            output[IxDyn(&coord)] = if labels[index] > 0 { 1.0 } else { 0.0 };
        }
    });

    Ok(output)
}

enum NeighborLabels {
    None,
    One(isize),
    Multiple,
}

fn unique_neighbor_label(
    labels: &[isize],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> NeighborLabels {
    let mut found = None;
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let neighbor_x = x as isize + dx;
            let neighbor_y = y as isize + dy;
            if neighbor_x < 0
                || neighbor_y < 0
                || neighbor_x >= width as isize
                || neighbor_y >= height as isize
            {
                continue;
            }
            let label = labels[neighbor_y as usize * width + neighbor_x as usize];
            if label <= 0 {
                continue;
            }
            match found {
                None => found = Some(label),
                Some(existing) if existing == label => {}
                Some(_) => return NeighborLabels::Multiple,
            }
        }
    }
    match found {
        Some(label) => NeighborLabels::One(label),
        None => NeighborLabels::None,
    }
}

fn ultimate_points_xy(dataset: &DatasetF32) -> Result<ndarray::ArrayD<f32>> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let width = shape[x_axis];
    let height = shape[y_axis];
    let edm = distance_map_xy(dataset)?;
    let mut output = ndarray::ArrayD::<f32>::zeros(IxDyn(&shape));
    let plane_shape = shape
        .iter()
        .enumerate()
        .filter_map(|(axis, size)| (axis != x_axis && axis != y_axis).then_some(*size))
        .collect::<Vec<_>>();

    iterate_indices(&plane_shape, |plane_coord| {
        let base_coord = plane_base_coord(&shape, x_axis, y_axis, plane_coord);
        let mut candidates = vec![false; width * height];
        for y in 0..height {
            for x in 0..width {
                let mut coord = base_coord.clone();
                coord[x_axis] = x;
                coord[y_axis] = y;
                let value = edm[IxDyn(&coord)];
                if value <= 0.0 {
                    continue;
                }
                let is_maximum = (-1..=1).all(|dy| {
                    (-1..=1).all(|dx| {
                        if dx == 0 && dy == 0 {
                            return true;
                        }
                        let neighbor_x = x as isize + dx;
                        let neighbor_y = y as isize + dy;
                        if neighbor_x < 0
                            || neighbor_y < 0
                            || neighbor_x >= width as isize
                            || neighbor_y >= height as isize
                        {
                            return true;
                        }
                        let mut neighbor = base_coord.clone();
                        neighbor[x_axis] = neighbor_x as usize;
                        neighbor[y_axis] = neighbor_y as usize;
                        edm[IxDyn(&neighbor)] <= value
                    })
                });
                if is_maximum {
                    candidates[y * width + x] = true;
                }
            }
        }

        let mut visited = vec![false; width * height];
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                if !candidates[index] || visited[index] {
                    continue;
                }
                let mut coord = base_coord.clone();
                coord[x_axis] = x;
                coord[y_axis] = y;
                let value = edm[IxDyn(&coord)];
                let plateau = collect_maximum_plateau(
                    &edm,
                    &base_coord,
                    x_axis,
                    y_axis,
                    width,
                    height,
                    &candidates,
                    &mut visited,
                    x,
                    y,
                    value,
                );
                let (point_x, point_y) = representative_plateau_point(&plateau);
                coord[x_axis] = point_x;
                coord[y_axis] = point_y;
                output[IxDyn(&coord)] = value;
            }
        }
    });

    Ok(output)
}

fn collect_maximum_plateau(
    edm: &ndarray::ArrayD<f32>,
    base_coord: &[usize],
    x_axis: usize,
    y_axis: usize,
    width: usize,
    height: usize,
    candidates: &[bool],
    visited: &mut [bool],
    start_x: usize,
    start_y: usize,
    value: f32,
) -> Vec<(usize, usize)> {
    let mut stack = vec![(start_x, start_y)];
    let mut plateau = Vec::new();

    while let Some((x, y)) = stack.pop() {
        let index = y * width + x;
        if visited[index] || !candidates[index] {
            continue;
        }
        let mut coord = base_coord.to_vec();
        coord[x_axis] = x;
        coord[y_axis] = y;
        if edm[IxDyn(&coord)] != value {
            continue;
        }
        visited[index] = true;
        plateau.push((x, y));

        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let neighbor_x = x as isize + dx;
                let neighbor_y = y as isize + dy;
                if neighbor_x < 0
                    || neighbor_y < 0
                    || neighbor_x >= width as isize
                    || neighbor_y >= height as isize
                {
                    continue;
                }
                stack.push((neighbor_x as usize, neighbor_y as usize));
            }
        }
    }

    plateau
}

fn representative_plateau_point(plateau: &[(usize, usize)]) -> (usize, usize) {
    let count = plateau.len() as f64;
    let average_x = plateau.iter().map(|(x, _)| *x as f64).sum::<f64>() / count;
    let average_y = plateau.iter().map(|(_, y)| *y as f64).sum::<f64>() / count;
    plateau
        .iter()
        .copied()
        .min_by(|(ax, ay), (bx, by)| {
            let a_dist = (average_x - *ax as f64).powi(2) + (average_y - *ay as f64).powi(2);
            let b_dist = (average_x - *bx as f64).powi(2) + (average_y - *by as f64).powi(2);
            a_dist
                .partial_cmp(&b_dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("plateau has at least one point")
}

fn distance_map_xy(dataset: &DatasetF32) -> Result<ndarray::ArrayD<f32>> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let width = shape[x_axis];
    let height = shape[y_axis];
    let mut output = dataset.data.clone();
    let plane_shape = shape
        .iter()
        .enumerate()
        .filter_map(|(axis, size)| (axis != x_axis && axis != y_axis).then_some(*size))
        .collect::<Vec<_>>();

    iterate_indices(&plane_shape, |plane_coord| {
        let base_coord = plane_base_coord(&shape, x_axis, y_axis, plane_coord);
        let mut background = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let mut coord = base_coord.clone();
                coord[x_axis] = x;
                coord[y_axis] = y;
                if dataset.data[IxDyn(&coord)] <= 0.5 {
                    background.push((x, y));
                }
            }
        }

        for y in 0..height {
            for x in 0..width {
                let mut coord = base_coord.clone();
                coord[x_axis] = x;
                coord[y_axis] = y;
                if dataset.data[IxDyn(&coord)] <= 0.5 {
                    output[IxDyn(&coord)] = 0.0;
                    continue;
                }
                let distance_squared = background
                    .iter()
                    .map(|(bg_x, bg_y)| {
                        let dx = x.abs_diff(*bg_x);
                        let dy = y.abs_diff(*bg_y);
                        (dx * dx + dy * dy) as f32
                    })
                    .fold(f32::INFINITY, f32::min);
                output[IxDyn(&coord)] = distance_squared.sqrt();
            }
        }
    });

    Ok(output)
}

fn binary_median_xy(dataset: &DatasetF32, radius: usize) -> Result<ndarray::ArrayD<f32>> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let radius = radius as isize;
    let r2 = radius * radius;
    let mut offsets = Vec::new();
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= r2 {
                offsets.push((dx, dy));
            }
        }
    }
    let mut output = dataset.data.clone();

    iterate_indices(&shape, |coord| {
        let x = coord[x_axis] as isize;
        let y = coord[y_axis] as isize;
        let mut foreground = 0usize;
        for (dx, dy) in &offsets {
            let mut neighbor = coord.to_vec();
            neighbor[x_axis] = (x + dx).clamp(0, shape[x_axis] as isize - 1) as usize;
            neighbor[y_axis] = (y + dy).clamp(0, shape[y_axis] as isize - 1) as usize;
            if dataset.data[IxDyn(&neighbor)] > 0.5 {
                foreground += 1;
            }
        }
        output[IxDyn(coord)] = if foreground * 2 >= offsets.len() {
            1.0
        } else {
            0.0
        };
    });

    Ok(output)
}

fn skeletonize_xy(dataset: &DatasetF32) -> Result<ndarray::ArrayD<f32>> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let width = shape[x_axis];
    let height = shape[y_axis];
    let mut output = dataset.data.clone();
    let plane_shape = shape
        .iter()
        .enumerate()
        .filter_map(|(axis, size)| (axis != x_axis && axis != y_axis).then_some(*size))
        .collect::<Vec<_>>();

    iterate_indices(&plane_shape, |plane_coord| {
        let base_coord = plane_base_coord(&shape, x_axis, y_axis, plane_coord);
        let mut plane = vec![false; width * height];
        for y in 0..height {
            for x in 0..width {
                let mut coord = base_coord.clone();
                coord[x_axis] = x;
                coord[y_axis] = y;
                plane[y * width + x] = dataset.data[IxDyn(&coord)] > 0.5;
            }
        }
        thin_plane(&mut plane, width, height);
        for y in 0..height {
            for x in 0..width {
                let mut coord = base_coord.clone();
                coord[x_axis] = x;
                coord[y_axis] = y;
                output[IxDyn(&coord)] = if plane[y * width + x] { 1.0 } else { 0.0 };
            }
        }
    });

    Ok(output)
}

fn thin_plane(plane: &mut [bool], width: usize, height: usize) {
    if width < 3 || height < 3 {
        return;
    }
    loop {
        let first = thinning_pass(plane, width, height, 0);
        let second = thinning_pass(plane, width, height, 1);
        if first + second == 0 {
            break;
        }
    }
}

fn thinning_pass(plane: &mut [bool], width: usize, height: usize, pass: usize) -> usize {
    let mut remove = Vec::new();
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let index = y * width + x;
            if !plane[index] {
                continue;
            }
            let neighbors = skeleton_neighbors(plane, width, x, y);
            let count = neighbors.iter().filter(|value| **value).count();
            let transitions = foreground_transitions(&neighbors);
            if !(2..=6).contains(&count) || transitions != 1 {
                continue;
            }
            let should_remove = if pass == 0 {
                !(neighbors[0] && neighbors[2] && neighbors[4])
                    && !(neighbors[2] && neighbors[4] && neighbors[6])
            } else {
                !(neighbors[0] && neighbors[2] && neighbors[6])
                    && !(neighbors[0] && neighbors[4] && neighbors[6])
            };
            if should_remove {
                remove.push(index);
            }
        }
    }
    let removed = remove.len();
    for index in remove {
        plane[index] = false;
    }
    removed
}

fn skeleton_neighbors(plane: &[bool], width: usize, x: usize, y: usize) -> [bool; 8] {
    [
        plane[(y - 1) * width + x],
        plane[(y - 1) * width + x + 1],
        plane[y * width + x + 1],
        plane[(y + 1) * width + x + 1],
        plane[(y + 1) * width + x],
        plane[(y + 1) * width + x - 1],
        plane[y * width + x - 1],
        plane[(y - 1) * width + x - 1],
    ]
}

fn foreground_transitions(neighbors: &[bool; 8]) -> usize {
    let mut transitions = 0usize;
    for index in 0..neighbors.len() {
        if !neighbors[index] && neighbors[(index + 1) % neighbors.len()] {
            transitions += 1;
        }
    }
    transitions
}

fn fill_holes_xy(dataset: &DatasetF32) -> Result<ndarray::ArrayD<f32>> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let width = shape[x_axis];
    let height = shape[y_axis];
    let mut output = dataset.data.clone();
    let plane_shape = shape
        .iter()
        .enumerate()
        .filter_map(|(axis, size)| (axis != x_axis && axis != y_axis).then_some(*size))
        .collect::<Vec<_>>();

    iterate_indices(&plane_shape, |plane_coord| {
        let base_coord = plane_base_coord(&shape, x_axis, y_axis, plane_coord);
        let mut exterior = vec![false; width * height];
        let mut stack = Vec::new();

        for x in 0..width {
            push_background(
                dataset,
                &base_coord,
                x_axis,
                y_axis,
                &mut exterior,
                &mut stack,
                x,
                0,
            );
            push_background(
                dataset,
                &base_coord,
                x_axis,
                y_axis,
                &mut exterior,
                &mut stack,
                x,
                height - 1,
            );
        }
        for y in 0..height {
            push_background(
                dataset,
                &base_coord,
                x_axis,
                y_axis,
                &mut exterior,
                &mut stack,
                0,
                y,
            );
            push_background(
                dataset,
                &base_coord,
                x_axis,
                y_axis,
                &mut exterior,
                &mut stack,
                width - 1,
                y,
            );
        }

        while let Some((x, y)) = stack.pop() {
            if x > 0 {
                push_background(
                    dataset,
                    &base_coord,
                    x_axis,
                    y_axis,
                    &mut exterior,
                    &mut stack,
                    x - 1,
                    y,
                );
            }
            if x + 1 < width {
                push_background(
                    dataset,
                    &base_coord,
                    x_axis,
                    y_axis,
                    &mut exterior,
                    &mut stack,
                    x + 1,
                    y,
                );
            }
            if y > 0 {
                push_background(
                    dataset,
                    &base_coord,
                    x_axis,
                    y_axis,
                    &mut exterior,
                    &mut stack,
                    x,
                    y - 1,
                );
            }
            if y + 1 < height {
                push_background(
                    dataset,
                    &base_coord,
                    x_axis,
                    y_axis,
                    &mut exterior,
                    &mut stack,
                    x,
                    y + 1,
                );
            }
        }

        for y in 0..height {
            for x in 0..width {
                let mut coord = base_coord.clone();
                coord[x_axis] = x;
                coord[y_axis] = y;
                let plane_index = y * width + x;
                let foreground = dataset.data[IxDyn(&coord)] > 0.5 || !exterior[plane_index];
                output[IxDyn(&coord)] = if foreground { 1.0 } else { 0.0 };
            }
        }
    });

    Ok(output)
}

fn push_background(
    dataset: &DatasetF32,
    base_coord: &[usize],
    x_axis: usize,
    y_axis: usize,
    exterior: &mut [bool],
    stack: &mut Vec<(usize, usize)>,
    x: usize,
    y: usize,
) {
    let width = dataset.shape()[x_axis];
    let plane_index = y * width + x;
    if exterior[plane_index] {
        return;
    }
    let mut coord = base_coord.to_vec();
    coord[x_axis] = x;
    coord[y_axis] = y;
    if dataset.data[IxDyn(&coord)] > 0.5 {
        return;
    }
    exterior[plane_index] = true;
    stack.push((x, y));
}

fn plane_base_coord(
    shape: &[usize],
    x_axis: usize,
    y_axis: usize,
    plane_coord: &[usize],
) -> Vec<usize> {
    let mut base_coord = vec![0usize; shape.len()];
    let mut plane_at = 0usize;
    for axis in 0..shape.len() {
        if axis != x_axis && axis != y_axis {
            base_coord[axis] = plane_coord[plane_at];
            plane_at += 1;
        }
    }
    base_coord
}

fn axis_index(dataset: &DatasetF32, axis: AxisKind) -> Result<usize> {
    dataset
        .axis_index(axis)
        .ok_or_else(|| OpsError::UnsupportedLayout(format!("dataset has no {axis:?} axis")))
}

fn iterate_indices(shape: &[usize], mut callback: impl FnMut(&[usize])) {
    if shape.is_empty() {
        callback(&[]);
        return;
    }
    let mut index = vec![0usize; shape.len()];
    loop {
        callback(&index);
        let mut dim = shape.len();
        while dim > 0 {
            dim -= 1;
            index[dim] += 1;
            if index[dim] < shape[dim] {
                break;
            }
            index[dim] = 0;
            if dim == 0 {
                return;
            }
        }
    }
}
