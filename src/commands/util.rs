use std::f32::consts::PI;

pub(crate) fn min_max(values: &[f32]) -> (f32, f32) {
    let mut iter = values.iter().copied();
    let first = iter.next().unwrap_or(0.0);
    let mut min = first;
    let mut max = first;
    for value in iter {
        min = min.min(value);
        max = max.max(value);
    }
    (min, max)
}

pub(crate) fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return vec![1.0];
    }
    let radius = (sigma * 3.0).ceil().max(1.0) as i32;
    let mut kernel = Vec::with_capacity((radius * 2 + 1) as usize);
    let mut sum = 0.0_f32;
    for offset in -radius..=radius {
        let distance = offset as f32;
        let value =
            (-(distance * distance) / (2.0 * sigma * sigma)).exp() / (sigma * (2.0 * PI).sqrt());
        kernel.push(value);
        sum += value;
    }
    kernel
        .iter_mut()
        .for_each(|value| *value /= sum.max(f32::EPSILON));
    kernel
}

pub(crate) fn neighborhood_offsets(
    rank: usize,
    radius: usize,
    include_origin: bool,
) -> Vec<Vec<isize>> {
    fn recurse(
        rank: usize,
        radius: isize,
        axis: usize,
        current: &mut Vec<isize>,
        output: &mut Vec<Vec<isize>>,
        include_origin: bool,
    ) {
        if axis == rank {
            if include_origin || current.iter().any(|value| *value != 0) {
                output.push(current.clone());
            }
            return;
        }

        for offset in -radius..=radius {
            current.push(offset);
            recurse(rank, radius, axis + 1, current, output, include_origin);
            current.pop();
        }
    }

    let mut offsets = Vec::new();
    recurse(
        rank,
        radius as isize,
        0,
        &mut Vec::new(),
        &mut offsets,
        include_origin,
    );
    offsets
}

