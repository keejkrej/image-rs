use super::lut::{LookupTable, lookup_table_color};
use crate::formats::NativeRasterImage;
use crate::model::{AxisKind, DatasetF32, Dim, PixelType};
use eframe::egui;
use ndarray::IxDyn;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) enum ViewerImageSource {
    Native(Arc<NativeRasterImage>),
    Dataset(Arc<DatasetF32>),
}

impl ViewerImageSource {
    pub(crate) fn to_dataset(&self) -> Result<Arc<DatasetF32>, String> {
        match self {
            Self::Native(image) => image
                .to_dataset()
                .map(Arc::new)
                .map_err(|error: crate::formats::IoError| error.to_string()),
            Self::Dataset(dataset) => Ok(dataset.clone()),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Default, PartialEq, Eq)]
pub(crate) struct ViewerFrameRequest {
    #[serde(default)]
    pub(crate) z: usize,
    #[serde(default)]
    pub(crate) t: usize,
    #[serde(default)]
    pub(crate) channel: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub(crate) struct ImageSummary {
    pub(crate) shape: Vec<usize>,
    pub(crate) axes: Vec<String>,
    pub(crate) channels: usize,
    pub(crate) z_slices: usize,
    pub(crate) times: usize,
    pub(crate) min: f32,
    pub(crate) max: f32,
    pub(crate) source: String,
}

#[derive(Debug, Clone)]
pub(crate) struct SliceImage {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) pixel_type: PixelType,
    pub(crate) values: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct ViewerFrameBuffer {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) values: Vec<f32>,
    pub(crate) pixels_u8: Vec<u8>,
    pub(crate) min: f32,
    pub(crate) max: f32,
}

pub(crate) fn build_frame(
    source: &ViewerImageSource,
    request: &ViewerFrameRequest,
    display_range: Option<(f32, f32)>,
) -> Result<ViewerFrameBuffer, String> {
    match source {
        ViewerImageSource::Native(image) => {
            build_native_frame(image.as_ref(), request, display_range)
        }
        ViewerImageSource::Dataset(dataset) => {
            build_dataset_frame(dataset.as_ref(), request, display_range)
        }
    }
}

fn build_dataset_frame(
    dataset: &DatasetF32,
    request: &ViewerFrameRequest,
    display_range: Option<(f32, f32)>,
) -> Result<ViewerFrameBuffer, String> {
    let slice = extract_slice(dataset, request.z, request.t, request.channel)?;
    let pixels_u8 = to_u8_samples(&slice.values, dataset.metadata.pixel_type, display_range);
    let (min, max) = min_max(&slice.values);

    Ok(ViewerFrameBuffer {
        width: slice.width,
        height: slice.height,
        values: slice.values.clone(),
        pixels_u8,
        min,
        max,
    })
}

fn build_native_frame(
    image: &NativeRasterImage,
    request: &ViewerFrameRequest,
    display_range: Option<(f32, f32)>,
) -> Result<ViewerFrameBuffer, String> {
    if request.z > 0 || request.t > 0 {
        return Err("native rasters expose only a single Z/T plane".to_string());
    }

    match image {
        NativeRasterImage::Gray8 {
            width,
            height,
            pixels,
            ..
        } => {
            let (min, max) = image.min_max();
            Ok(ViewerFrameBuffer {
                width: *width,
                height: *height,
                values: pixels.iter().map(|value| f32::from(*value)).collect(),
                pixels_u8: to_u8_samples(
                    &pixels
                        .iter()
                        .map(|value| f32::from(*value))
                        .collect::<Vec<_>>(),
                    PixelType::U8,
                    display_range,
                ),
                min,
                max,
            })
        }
        NativeRasterImage::Gray16 {
            width,
            height,
            pixels,
            ..
        } => {
            let (min, max) = image.min_max();
            Ok(ViewerFrameBuffer {
                width: *width,
                height: *height,
                values: pixels.iter().map(|value| f32::from(*value)).collect(),
                pixels_u8: to_u8_samples(
                    &pixels
                        .iter()
                        .map(|value| f32::from(*value))
                        .collect::<Vec<_>>(),
                    PixelType::U16,
                    display_range,
                ),
                min,
                max,
            })
        }
        NativeRasterImage::Rgb8 {
            width,
            height,
            pixels,
            ..
        } => {
            let channel = request.channel.min(2);
            let mut pixels_u8 = Vec::with_capacity(width * height);
            let mut min_sample = u8::MAX;
            let mut max_sample = u8::MIN;
            for chunk in pixels.chunks_exact(3) {
                let sample = chunk[channel];
                min_sample = min_sample.min(sample);
                max_sample = max_sample.max(sample);
                pixels_u8.push(sample);
            }
            let values = pixels_u8
                .iter()
                .map(|value| f32::from(*value))
                .collect::<Vec<_>>();
            Ok(ViewerFrameBuffer {
                width: *width,
                height: *height,
                pixels_u8: to_u8_samples(&values, PixelType::U8, display_range),
                values,
                min: f32::from(min_sample),
                max: f32::from(max_sample),
            })
        }
    }
}

pub(crate) fn summarize_source(source: &ViewerImageSource, path: &Path) -> ImageSummary {
    match source {
        ViewerImageSource::Native(image) => summarize_native_image(image.as_ref(), path),
        ViewerImageSource::Dataset(dataset) => summarize_dataset(dataset.as_ref(), path),
    }
}

fn summarize_dataset(dataset: &DatasetF32, source: &Path) -> ImageSummary {
    let (min, max) = dataset.min_max().unwrap_or((0.0, 0.0));
    summarize_metadata(dataset.shape(), &dataset.metadata.dims, min, max, source)
}

fn summarize_native_image(image: &NativeRasterImage, source: &Path) -> ImageSummary {
    let metadata = image.metadata();
    let shape = if image.channel_count() > 1 {
        vec![image.height(), image.width(), image.channel_count()]
    } else {
        vec![image.height(), image.width()]
    };
    let (min, max) = image.min_max();
    summarize_metadata(&shape, &metadata.dims, min, max, source)
}

fn summarize_metadata(
    shape: &[usize],
    dims: &[Dim],
    min: f32,
    max: f32,
    source: &Path,
) -> ImageSummary {
    let channel_axis = dims
        .iter()
        .position(|dimension| dimension.axis == AxisKind::Channel);
    let z_axis = dims
        .iter()
        .position(|dimension| dimension.axis == AxisKind::Z);
    let t_axis = dims
        .iter()
        .position(|dimension| dimension.axis == AxisKind::Time);

    ImageSummary {
        shape: shape.to_vec(),
        axes: dims
            .iter()
            .map(|dimension| format!("{:?}", dimension.axis))
            .collect(),
        channels: channel_axis.map(|index| shape[index]).unwrap_or(1),
        z_slices: z_axis.map(|index| shape[index]).unwrap_or(1),
        times: t_axis.map(|index| shape[index]).unwrap_or(1),
        min,
        max,
        source: source.display().to_string(),
    }
}

pub(crate) fn extract_slice(
    dataset: &DatasetF32,
    z: usize,
    t: usize,
    channel: usize,
) -> Result<SliceImage, String> {
    if dataset.ndim() < 2 {
        return Err("dataset must have at least two dimensions".to_string());
    }

    let y_axis = dataset
        .axis_index(AxisKind::Y)
        .unwrap_or(0)
        .min(dataset.ndim() - 1);
    let x_axis = dataset
        .axis_index(AxisKind::X)
        .unwrap_or(1.min(dataset.ndim() - 1));
    if y_axis == x_axis {
        return Err("could not infer distinct X/Y axes".to_string());
    }

    let mut index = vec![0usize; dataset.ndim()];
    if let Some(axis) = dataset.axis_index(AxisKind::Z) {
        index[axis] = z.min(dataset.shape()[axis].saturating_sub(1));
    }
    if let Some(axis) = dataset.axis_index(AxisKind::Time) {
        index[axis] = t.min(dataset.shape()[axis].saturating_sub(1));
    }
    if let Some(axis) = dataset.axis_index(AxisKind::Channel) {
        index[axis] = channel.min(dataset.shape()[axis].saturating_sub(1));
    }

    let height = dataset.shape()[y_axis];
    let width = dataset.shape()[x_axis];
    let mut values = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            index[y_axis] = y;
            index[x_axis] = x;
            values.push(dataset.data[IxDyn(&index)]);
        }
    }

    Ok(SliceImage {
        width,
        height,
        pixel_type: dataset.metadata.pixel_type,
        values,
    })
}

pub(crate) fn extract_slice_from_source(
    source: &ViewerImageSource,
    z: usize,
    t: usize,
    channel: usize,
) -> Result<SliceImage, String> {
    match source {
        ViewerImageSource::Native(image) => {
            extract_slice_from_native(image.as_ref(), z, t, channel)
        }
        ViewerImageSource::Dataset(dataset) => extract_slice(dataset.as_ref(), z, t, channel),
    }
}

fn extract_slice_from_native(
    image: &NativeRasterImage,
    z: usize,
    t: usize,
    channel: usize,
) -> Result<SliceImage, String> {
    if z > 0 || t > 0 {
        return Err("native rasters expose only a single Z/T plane".to_string());
    }

    match image {
        NativeRasterImage::Gray8 {
            width,
            height,
            pixels,
            ..
        } => Ok(SliceImage {
            width: *width,
            height: *height,
            pixel_type: PixelType::U8,
            values: pixels.iter().map(|value| f32::from(*value)).collect(),
        }),
        NativeRasterImage::Gray16 {
            width,
            height,
            pixels,
            ..
        } => Ok(SliceImage {
            width: *width,
            height: *height,
            pixel_type: PixelType::U16,
            values: pixels.iter().map(|value| f32::from(*value)).collect(),
        }),
        NativeRasterImage::Rgb8 {
            width,
            height,
            pixels,
            ..
        } => {
            let selected = channel.min(2);
            Ok(SliceImage {
                width: *width,
                height: *height,
                pixel_type: PixelType::U8,
                values: pixels
                    .chunks_exact(3)
                    .map(|chunk| f32::from(chunk[selected]))
                    .collect(),
            })
        }
    }
}

pub(crate) fn to_u8_samples(
    values: &[f32],
    pixel_type: PixelType,
    display_range: Option<(f32, f32)>,
) -> Vec<u8> {
    if let Some((low, high)) = display_range
        && high > low
    {
        return values
            .iter()
            .map(|value| (((*value - low) / (high - low)).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
    }

    match pixel_type {
        PixelType::U8 => values
            .iter()
            .map(|value| value.clamp(0.0, 255.0).round() as u8)
            .collect(),
        PixelType::U16 => values
            .iter()
            .map(|value| (value.clamp(0.0, 65_535.0) / 257.0).round() as u8)
            .collect(),
        PixelType::F32 => values
            .iter()
            .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect(),
    }
}

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

pub(crate) fn to_color_image(frame: &ViewerFrameBuffer, lut: LookupTable) -> egui::ColorImage {
    let mut rgba = Vec::with_capacity(frame.pixels_u8.len() * 4);
    for gray in &frame.pixels_u8 {
        let color = lookup_table_color(lut, *gray);
        rgba.extend_from_slice(&[color.r(), color.g(), color.b(), 255]);
    }
    egui::ColorImage::from_rgba_unmultiplied([frame.width, frame.height], &rgba)
}
