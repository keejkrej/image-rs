use crate::model::{AxisKind, Dataset, DatasetF32, Dim, PixelType};
use ndarray::{ArrayD, IxDyn};
use serde_json::Value;

use super::{
    OpOutput, OpSchema, Operation, OpsError, ParamSpec, Result, get_optional_f32,
    get_optional_usize,
};

#[derive(Debug, Clone, Copy)]
pub struct ImageConvertOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageResizeOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageCanvasResizeOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageSharpenOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageFindEdgesOp;

impl Operation for ImageConvertOp {
    fn name(&self) -> &'static str {
        "image.convert"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Convert pixel metadata or grayscale/RGB channel layout.".to_string(),
            params: vec![ParamSpec {
                name: "target".to_string(),
                description: "One of u8, u16, f32, rgb.".to_string(),
                required: true,
                kind: "string".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let Some(target) = params.get("target").and_then(Value::as_str) else {
            return Err(OpsError::InvalidParams("`target` is required".to_string()));
        };
        let output = match target {
            "u8" => convert_pixel_type(dataset, PixelType::U8)?,
            "u16" => convert_pixel_type(dataset, PixelType::U16)?,
            "f32" => convert_pixel_type(dataset, PixelType::F32)?,
            "rgb" => convert_to_rgb(dataset)?,
            "gray" | "grayscale" => convert_to_grayscale(dataset)?,
            other => {
                return Err(OpsError::InvalidParams(format!(
                    "unsupported conversion target `{other}`"
                )));
            }
        };
        Ok(OpOutput::dataset_only(output))
    }
}

impl Operation for ImageResizeOp {
    fn name(&self) -> &'static str {
        "image.resize"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Resize X/Y dimensions with bilinear interpolation.".to_string(),
            params: vec![
                ParamSpec {
                    name: "width".to_string(),
                    description: "Target width.".to_string(),
                    required: true,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "height".to_string(),
                    description: "Target height.".to_string(),
                    required: true,
                    kind: "int".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let width = get_optional_usize(params, "width", 0);
        let height = get_optional_usize(params, "height", 0);
        if width == 0 || height == 0 {
            return Err(OpsError::InvalidParams(
                "`width` and `height` must be > 0".to_string(),
            ));
        }
        Ok(OpOutput::dataset_only(resize_xy(dataset, width, height)?))
    }
}

impl Operation for ImageCanvasResizeOp {
    fn name(&self) -> &'static str {
        "image.canvas_resize"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Pad or crop the X/Y canvas centered around the source.".to_string(),
            params: vec![
                ParamSpec {
                    name: "width".to_string(),
                    description: "Target width.".to_string(),
                    required: true,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "height".to_string(),
                    description: "Target height.".to_string(),
                    required: true,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "fill".to_string(),
                    description: "Fill value for padded regions.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let width = get_optional_usize(params, "width", 0);
        let height = get_optional_usize(params, "height", 0);
        if width == 0 || height == 0 {
            return Err(OpsError::InvalidParams(
                "`width` and `height` must be > 0".to_string(),
            ));
        }
        let fill = get_optional_f32(params, "fill", 0.0);
        Ok(OpOutput::dataset_only(canvas_resize_xy(
            dataset, width, height, fill,
        )?))
    }
}

impl Operation for ImageSharpenOp {
    fn name(&self) -> &'static str {
        "image.sharpen"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Apply a simple sharpening kernel on X/Y planes.".to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        Ok(OpOutput::dataset_only(filter_xy(
            dataset,
            FilterKind::Sharpen,
        )?))
    }
}

impl Operation for ImageFindEdgesOp {
    fn name(&self) -> &'static str {
        "image.find_edges"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Apply a Sobel edge magnitude filter on X/Y planes.".to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        Ok(OpOutput::dataset_only(filter_xy(
            dataset,
            FilterKind::Sobel,
        )?))
    }
}

fn convert_pixel_type(dataset: &DatasetF32, pixel_type: PixelType) -> Result<DatasetF32> {
    let mut metadata = dataset.metadata.clone();
    metadata.pixel_type = pixel_type;
    Ok(Dataset::new(dataset.data.clone(), metadata)?)
}

fn convert_to_rgb(dataset: &DatasetF32) -> Result<DatasetF32> {
    if dataset
        .axis_index(AxisKind::Channel)
        .is_some_and(|axis| dataset.shape()[axis] == 3)
    {
        return convert_pixel_type(dataset, PixelType::U8);
    }

    let mut shape = dataset.shape().to_vec();
    shape.push(3);
    let mut dims = dataset.metadata.dims.clone();
    dims.push(Dim::new(AxisKind::Channel, 3));
    let mut values = Vec::with_capacity(dataset.data.len() * 3);
    for value in dataset.data.iter().copied() {
        values.push(value);
        values.push(value);
        values.push(value);
    }

    let data = ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build RGB-converted dataset".to_string())
    })?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims = dims;
    metadata.pixel_type = PixelType::U8;
    metadata.channel_names = vec!["R".into(), "G".into(), "B".into()];
    Ok(Dataset::new(data, metadata)?)
}

fn convert_to_grayscale(dataset: &DatasetF32) -> Result<DatasetF32> {
    let Some(channel_axis) = dataset.axis_index(AxisKind::Channel) else {
        return convert_pixel_type(dataset, PixelType::U8);
    };
    let channel_count = dataset.shape()[channel_axis];
    if channel_count == 1 {
        return convert_pixel_type(dataset, PixelType::U8);
    }
    if channel_count != 3 {
        return Err(OpsError::UnsupportedLayout(format!(
            "RGB-to-grayscale expects 3 channels, found {channel_count}"
        )));
    }

    let mut output_shape = dataset.shape().to_vec();
    output_shape[channel_axis] = 1;
    let output_len: usize = output_shape.iter().product();
    let mut output = vec![0.0_f32; output_len];
    let mut output_index = vec![0usize; output_shape.len()];
    let mut write_at = 0usize;
    iterate_indices(&output_shape, |coord| {
        output_index.copy_from_slice(coord);
        let mut r_idx = coord.to_vec();
        r_idx[channel_axis] = 0;
        let mut g_idx = coord.to_vec();
        g_idx[channel_axis] = 1;
        let mut b_idx = coord.to_vec();
        b_idx[channel_axis] = 2;
        output[write_at] = dataset.data[IxDyn(&r_idx)] * 0.299
            + dataset.data[IxDyn(&g_idx)] * 0.587
            + dataset.data[IxDyn(&b_idx)] * 0.114;
        write_at += 1;
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build grayscale dataset".to_string())
    })?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims[channel_axis].size = 1;
    metadata.pixel_type = PixelType::U8;
    metadata.channel_names = vec!["Gray".into()];
    Ok(Dataset::new(data, metadata)?)
}

fn resize_xy(dataset: &DatasetF32, width: usize, height: usize) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let mut output_shape = dataset.shape().to_vec();
    output_shape[x_axis] = width;
    output_shape[y_axis] = height;
    let input_shape = dataset.shape().to_vec();
    let input_w = input_shape[x_axis].max(1);
    let input_h = input_shape[y_axis].max(1);

    let output_len: usize = output_shape.iter().product();
    let mut output = vec![0.0_f32; output_len];
    let mut write_at = 0usize;
    iterate_indices(&output_shape, |coord| {
        let src_x = if width == 1 {
            0.0
        } else {
            coord[x_axis] as f32 * (input_w.saturating_sub(1)) as f32
                / (width.saturating_sub(1)) as f32
        };
        let src_y = if height == 1 {
            0.0
        } else {
            coord[y_axis] as f32 * (input_h.saturating_sub(1)) as f32
                / (height.saturating_sub(1)) as f32
        };
        output[write_at] = bilinear_sample(dataset, coord, x_axis, y_axis, src_x, src_y);
        write_at += 1;
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output)
        .map_err(|_| OpsError::UnsupportedLayout("failed to build resized dataset".to_string()))?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims[x_axis].size = width;
    metadata.dims[y_axis].size = height;
    Ok(Dataset::new(data, metadata)?)
}

fn canvas_resize_xy(
    dataset: &DatasetF32,
    width: usize,
    height: usize,
    fill: f32,
) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let mut output_shape = dataset.shape().to_vec();
    output_shape[x_axis] = width;
    output_shape[y_axis] = height;
    let output_len: usize = output_shape.iter().product();
    let mut output = vec![fill; output_len];
    let input_shape = dataset.shape().to_vec();
    let src_w = input_shape[x_axis];
    let src_h = input_shape[y_axis];
    let x_offset = width as isize / 2 - src_w as isize / 2;
    let y_offset = height as isize / 2 - src_h as isize / 2;
    let mut write_at = 0usize;
    iterate_indices(&output_shape, |coord| {
        let src_x = coord[x_axis] as isize - x_offset;
        let src_y = coord[y_axis] as isize - y_offset;
        if src_x >= 0 && src_x < src_w as isize && src_y >= 0 && src_y < src_h as isize {
            let mut src = coord.to_vec();
            src[x_axis] = src_x as usize;
            src[y_axis] = src_y as usize;
            output[write_at] = dataset.data[IxDyn(&src)];
        }
        write_at += 1;
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build canvas-resized dataset".to_string())
    })?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims[x_axis].size = width;
    metadata.dims[y_axis].size = height;
    Ok(Dataset::new(data, metadata)?)
}

#[derive(Debug, Clone, Copy)]
enum FilterKind {
    Sharpen,
    Sobel,
}

fn filter_xy(dataset: &DatasetF32, kind: FilterKind) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let output_len = dataset.data.len();
    let mut output = vec![0.0_f32; output_len];
    let mut write_at = 0usize;

    iterate_indices(&shape, |coord| {
        let x = coord[x_axis];
        let y = coord[y_axis];
        output[write_at] = match kind {
            FilterKind::Sharpen => {
                let center = sample_clamped(dataset, coord, x_axis, y_axis, x as isize, y as isize);
                let left =
                    sample_clamped(dataset, coord, x_axis, y_axis, x as isize - 1, y as isize);
                let right =
                    sample_clamped(dataset, coord, x_axis, y_axis, x as isize + 1, y as isize);
                let up = sample_clamped(dataset, coord, x_axis, y_axis, x as isize, y as isize - 1);
                let down =
                    sample_clamped(dataset, coord, x_axis, y_axis, x as isize, y as isize + 1);
                (5.0 * center - left - right - up - down).clamp(0.0, 1.0)
            }
            FilterKind::Sobel => {
                let gx = -sample_clamped(
                    dataset,
                    coord,
                    x_axis,
                    y_axis,
                    x as isize - 1,
                    y as isize - 1,
                ) + sample_clamped(
                    dataset,
                    coord,
                    x_axis,
                    y_axis,
                    x as isize + 1,
                    y as isize - 1,
                ) - 2.0
                    * sample_clamped(dataset, coord, x_axis, y_axis, x as isize - 1, y as isize)
                    + 2.0
                        * sample_clamped(
                            dataset,
                            coord,
                            x_axis,
                            y_axis,
                            x as isize + 1,
                            y as isize,
                        )
                    - sample_clamped(
                        dataset,
                        coord,
                        x_axis,
                        y_axis,
                        x as isize - 1,
                        y as isize + 1,
                    )
                    + sample_clamped(
                        dataset,
                        coord,
                        x_axis,
                        y_axis,
                        x as isize + 1,
                        y as isize + 1,
                    );
                let gy = sample_clamped(
                    dataset,
                    coord,
                    x_axis,
                    y_axis,
                    x as isize - 1,
                    y as isize - 1,
                ) + 2.0
                    * sample_clamped(dataset, coord, x_axis, y_axis, x as isize, y as isize - 1)
                    + sample_clamped(
                        dataset,
                        coord,
                        x_axis,
                        y_axis,
                        x as isize + 1,
                        y as isize - 1,
                    )
                    - sample_clamped(
                        dataset,
                        coord,
                        x_axis,
                        y_axis,
                        x as isize - 1,
                        y as isize + 1,
                    )
                    - 2.0
                        * sample_clamped(
                            dataset,
                            coord,
                            x_axis,
                            y_axis,
                            x as isize,
                            y as isize + 1,
                        )
                    - sample_clamped(
                        dataset,
                        coord,
                        x_axis,
                        y_axis,
                        x as isize + 1,
                        y as isize + 1,
                    );
                (gx.mul_add(gx, gy * gy)).sqrt().clamp(0.0, 1.0)
            }
        };
        write_at += 1;
    });

    let data = ArrayD::from_shape_vec(IxDyn(&shape), output)
        .map_err(|_| OpsError::UnsupportedLayout("failed to build filtered dataset".to_string()))?;
    Ok(Dataset::new(data, dataset.metadata.clone())?)
}

fn bilinear_sample(
    dataset: &DatasetF32,
    coord: &[usize],
    x_axis: usize,
    y_axis: usize,
    src_x: f32,
    src_y: f32,
) -> f32 {
    let x0 = src_x.floor() as isize;
    let x1 = src_x.ceil() as isize;
    let y0 = src_y.floor() as isize;
    let y1 = src_y.ceil() as isize;
    let fx = src_x.fract();
    let fy = src_y.fract();

    let p00 = sample_clamped(dataset, coord, x_axis, y_axis, x0, y0);
    let p10 = sample_clamped(dataset, coord, x_axis, y_axis, x1, y0);
    let p01 = sample_clamped(dataset, coord, x_axis, y_axis, x0, y1);
    let p11 = sample_clamped(dataset, coord, x_axis, y_axis, x1, y1);
    let top = p00 * (1.0 - fx) + p10 * fx;
    let bottom = p01 * (1.0 - fx) + p11 * fx;
    top * (1.0 - fy) + bottom * fy
}

fn sample_clamped(
    dataset: &DatasetF32,
    coord: &[usize],
    x_axis: usize,
    y_axis: usize,
    x: isize,
    y: isize,
) -> f32 {
    let mut index = coord.to_vec();
    index[x_axis] = x.clamp(0, dataset.shape()[x_axis] as isize - 1) as usize;
    index[y_axis] = y.clamp(0, dataset.shape()[y_axis] as isize - 1) as usize;
    dataset.data[IxDyn(&index)]
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
