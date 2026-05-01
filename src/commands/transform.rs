use crate::model::{AxisKind, Dataset, DatasetF32, Dim, Metadata, PixelType};
use ndarray::{ArrayD, IxDyn};
use rustfft::{FftPlanner, num_complex::Complex};
use serde_json::{Value, json};

use super::{
    OpOutput, OpSchema, Operation, OpsError, ParamSpec, Result, get_optional_f32,
    get_optional_usize, util::gaussian_kernel,
};

#[derive(Debug, Clone, Copy)]
pub struct ImageConvertOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageResizeOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageScaleOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageCanvasResizeOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageCropOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageCoordinatesOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageSetScaleOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageCalibrateOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageStackAddSliceOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageStackDeleteSliceOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageStackSubstackOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageStackResliceOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageStackReduceOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageStackZProjectOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageStackGroupedZProjectOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageStackMontageOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageStackMontageToStackOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageStackToHyperstackOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageHyperstackToStackOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageHyperstackReduceDimensionalityOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageHyperstackSubsetOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageBinOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageFlipOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageRotate90Op;

#[derive(Debug, Clone, Copy)]
pub struct ImageRotateOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageTranslateOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageMedianFilterOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageRemoveNaNsOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageRemoveOutliersOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageSharpenOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageFindEdgesOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageFindMaximaOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageShadowOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageShadowDemoOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageRankFilterOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageSubtractBackgroundOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageUnsharpMaskOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageConvolveOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageRankFilter3dOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageSwapQuadrantsOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageFftPowerSpectrumOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageFftBandpassOp;

#[derive(Debug, Clone, Copy)]
pub struct ImageSurfacePlotOp;

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

impl Operation for ImageScaleOp {
    fn name(&self) -> &'static str {
        "image.scale"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Scale X/Y dimensions by ImageJ-style scale factors.".to_string(),
            params: vec![
                ParamSpec {
                    name: "x_scale".to_string(),
                    description: "X scale factor.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "y_scale".to_string(),
                    description: "Y scale factor; defaults to x_scale.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let x_scale = get_optional_f32(params, "x_scale", 1.0);
        let y_scale = get_optional_f32(params, "y_scale", x_scale);
        if !x_scale.is_finite() || !y_scale.is_finite() || x_scale <= 0.0 || y_scale <= 0.0 {
            return Err(OpsError::InvalidParams(
                "`x_scale` and `y_scale` must be finite and > 0".to_string(),
            ));
        }
        let x_axis = axis_index(dataset, AxisKind::X)?;
        let y_axis = axis_index(dataset, AxisKind::Y)?;
        let width = ((dataset.shape()[x_axis] as f32) * x_scale)
            .round()
            .max(1.0) as usize;
        let height = ((dataset.shape()[y_axis] as f32) * y_scale)
            .round()
            .max(1.0) as usize;
        let mut output = resize_xy(dataset, width, height)?;
        if let Some(spacing) = &mut output.metadata.dims[x_axis].spacing {
            *spacing /= x_scale;
        }
        if let Some(spacing) = &mut output.metadata.dims[y_axis].spacing {
            *spacing /= y_scale;
        }
        Ok(OpOutput::dataset_only(output))
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

impl Operation for ImageCropOp {
    fn name(&self) -> &'static str {
        "image.crop"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Crop the X/Y image bounds while preserving other axes.".to_string(),
            params: vec![
                ParamSpec {
                    name: "x".to_string(),
                    description: "Left crop coordinate in pixels.".to_string(),
                    required: true,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "y".to_string(),
                    description: "Top crop coordinate in pixels.".to_string(),
                    required: true,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "width".to_string(),
                    description: "Crop width in pixels.".to_string(),
                    required: true,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "height".to_string(),
                    description: "Crop height in pixels.".to_string(),
                    required: true,
                    kind: "int".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let x = get_optional_usize(params, "x", 0);
        let y = get_optional_usize(params, "y", 0);
        let width = get_optional_usize(params, "width", 0);
        let height = get_optional_usize(params, "height", 0);
        if width == 0 || height == 0 {
            return Err(OpsError::InvalidParams(
                "`width` and `height` must be > 0".to_string(),
            ));
        }
        Ok(OpOutput::dataset_only(crop_xy(
            dataset, x, y, width, height,
        )?))
    }
}

impl Operation for ImageCoordinatesOp {
    fn name(&self) -> &'static str {
        "image.coordinates"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Update ImageJ-style X/Y/Z coordinate calibration metadata.".to_string(),
            params: vec![
                ParamSpec {
                    name: "left".to_string(),
                    description: "Scaled coordinate at the left image edge.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "right".to_string(),
                    description: "Scaled coordinate at the right image edge.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "top".to_string(),
                    description: "Scaled coordinate at the top image edge.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "bottom".to_string(),
                    description: "Scaled coordinate at the bottom image edge.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "front".to_string(),
                    description: "Scaled coordinate at the first Z slice.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "back".to_string(),
                    description: "Scaled coordinate after the last Z slice.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "x_unit".to_string(),
                    description: "X coordinate unit.".to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "y_unit".to_string(),
                    description: "Y coordinate unit; defaults to the X unit when omitted."
                        .to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "z_unit".to_string(),
                    description: "Z coordinate unit; defaults to the X unit when omitted."
                        .to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let mut metadata = dataset.metadata.clone();
        let x_axis = axis_index(dataset, AxisKind::X)?;
        let y_axis = axis_index(dataset, AxisKind::Y)?;
        apply_coordinate_bounds(&mut metadata, x_axis, params, "left", "right", "x")?;
        apply_coordinate_bounds(&mut metadata, y_axis, params, "top", "bottom", "y")?;
        if let Some(z_axis) = dataset.axis_index(AxisKind::Z) {
            apply_coordinate_bounds(&mut metadata, z_axis, params, "front", "back", "z")?;
        }

        apply_coordinate_units(
            &mut metadata,
            x_axis,
            y_axis,
            dataset.axis_index(AxisKind::Z),
            params,
        );

        Ok(OpOutput::dataset_only(Dataset::new(
            dataset.data.clone(),
            metadata,
        )?))
    }
}

impl Operation for ImageSetScaleOp {
    fn name(&self) -> &'static str {
        "image.set_scale"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Apply ImageJ Analyze/Set Scale calibration metadata.".to_string(),
            params: vec![
                ParamSpec {
                    name: "distance_pixels".to_string(),
                    description: "Measured distance in pixels.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "known_distance".to_string(),
                    description: "Known calibrated distance.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "pixel_aspect_ratio".to_string(),
                    description: "Pixel height divided by pixel width.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "unit".to_string(),
                    description: "Unit of length; pixel or empty removes calibration.".to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "global".to_string(),
                    description: "Record whether ImageJ global calibration was requested."
                        .to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let mut metadata = dataset.metadata.clone();
        apply_set_scale(&mut metadata, dataset, params)?;
        Ok(OpOutput::dataset_only(Dataset::new(
            dataset.data.clone(),
            metadata,
        )?))
    }
}

impl Operation for ImageCalibrateOp {
    fn name(&self) -> &'static str {
        "image.calibrate"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Apply ImageJ Analyze/Calibrate value-unit metadata.".to_string(),
            params: vec![
                ParamSpec {
                    name: "function".to_string(),
                    description:
                        "Density calibration function; currently supports ImageJ None only."
                            .to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "unit".to_string(),
                    description: "Calibrated value unit, such as Gray Value or OD.".to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "global".to_string(),
                    description: "Record whether ImageJ global calibration was requested."
                        .to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let mut metadata = dataset.metadata.clone();
        apply_calibration_metadata(&mut metadata, params)?;
        Ok(OpOutput::dataset_only(Dataset::new(
            dataset.data.clone(),
            metadata,
        )?))
    }
}

impl Operation for ImageStackAddSliceOp {
    fn name(&self) -> &'static str {
        "image.stack.add_slice"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Add a blank ImageJ-style Z slice to the active image.".to_string(),
            params: vec![
                ParamSpec {
                    name: "index".to_string(),
                    description: "Insertion index along Z; defaults to append.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "fill".to_string(),
                    description: "Fill value for the inserted slice.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let z_axis = dataset.axis_index(AxisKind::Z);
        let slices = z_axis.map(|axis| dataset.shape()[axis]).unwrap_or(1);
        let index = get_optional_usize(params, "index", slices);
        let fill = get_optional_f32(params, "fill", 0.0);
        Ok(OpOutput::dataset_only(stack_add_slice(
            dataset, index, fill,
        )?))
    }
}

impl Operation for ImageStackDeleteSliceOp {
    fn name(&self) -> &'static str {
        "image.stack.delete_slice"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Delete a Z slice from the active ImageJ-style stack.".to_string(),
            params: vec![ParamSpec {
                name: "index".to_string(),
                description: "Slice index along Z.".to_string(),
                required: false,
                kind: "int".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let index = get_optional_usize(params, "index", 0);
        Ok(OpOutput::dataset_only(stack_delete_slice(dataset, index)?))
    }
}

impl Operation for ImageStackSubstackOp {
    fn name(&self) -> &'static str {
        "image.stack.substack"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Create an ImageJ-style substack from selected Z slices.".to_string(),
            params: vec![
                ParamSpec {
                    name: "slices".to_string(),
                    description: "One-based slice range/list, e.g. `2-14`, `1-10-2`, or `7,9,25`."
                        .to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "indices".to_string(),
                    description:
                        "Optional zero-based Z slice indices; used when `slices` is omitted."
                            .to_string(),
                    required: false,
                    kind: "array<int>".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        Ok(OpOutput::dataset_only(stack_substack(dataset, params)?))
    }
}

impl Operation for ImageStackResliceOp {
    fn name(&self) -> &'static str {
        "image.stack.reslice"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Create an ImageJ-style orthogonal reslice stack.".to_string(),
            params: vec![ParamSpec {
                name: "start".to_string(),
                description: "Reslice orientation: top, bottom, left, or right.".to_string(),
                required: false,
                kind: "string".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        Ok(OpOutput::dataset_only(stack_reslice(dataset, params)?))
    }
}

impl Operation for ImageStackReduceOp {
    fn name(&self) -> &'static str {
        "image.stack.reduce"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Reduce an ImageJ-style Z stack by keeping every Nth slice.".to_string(),
            params: vec![ParamSpec {
                name: "factor".to_string(),
                description: "Z slice reduction factor.".to_string(),
                required: false,
                kind: "int".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let factor = get_optional_usize(params, "factor", 2);
        Ok(OpOutput::dataset_only(stack_reduce(dataset, factor)?))
    }
}

impl Operation for ImageStackZProjectOp {
    fn name(&self) -> &'static str {
        "image.stack.z_project"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Project an ImageJ-style Z stack into one image.".to_string(),
            params: vec![
                ParamSpec {
                    name: "method".to_string(),
                    description: "Projection method: average, max, min, sum, sd, or median."
                        .to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "start".to_string(),
                    description: "First zero-based Z slice to include.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "stop".to_string(),
                    description: "Last zero-based Z slice to include, inclusive.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        Ok(OpOutput::dataset_only(stack_z_project(dataset, params)?))
    }
}

impl Operation for ImageStackMontageOp {
    fn name(&self) -> &'static str {
        "image.stack.montage"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Create an ImageJ-style montage from Z stack slices.".to_string(),
            params: vec![
                ParamSpec {
                    name: "columns".to_string(),
                    description: "Number of montage columns; defaults from stack size.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "rows".to_string(),
                    description: "Number of montage rows; defaults from stack size.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "scale".to_string(),
                    description: "Scale factor applied to each tile.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "first".to_string(),
                    description: "First zero-based Z slice to include.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "last".to_string(),
                    description: "Last zero-based Z slice to include, inclusive.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "increment".to_string(),
                    description: "Z slice step.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "border_width".to_string(),
                    description: "Pixel width of borders between tiles.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "fill".to_string(),
                    description: "Background value for empty grid cells and borders.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        Ok(OpOutput::dataset_only(stack_montage(dataset, params)?))
    }
}

impl Operation for ImageStackGroupedZProjectOp {
    fn name(&self) -> &'static str {
        "image.stack.grouped_z_project"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Project fixed-size groups of ImageJ-style Z slices.".to_string(),
            params: vec![
                ParamSpec {
                    name: "method".to_string(),
                    description: "Projection method: average, max, min, sum, sd, or median."
                        .to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "group_size".to_string(),
                    description: "Number of adjacent Z slices per output slice.".to_string(),
                    required: true,
                    kind: "int".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        Ok(OpOutput::dataset_only(stack_grouped_z_project(
            dataset, params,
        )?))
    }
}

impl Operation for ImageStackMontageToStackOp {
    fn name(&self) -> &'static str {
        "image.stack.montage_to_stack"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Convert an ImageJ-style montage image back to a Z stack.".to_string(),
            params: vec![
                ParamSpec {
                    name: "columns".to_string(),
                    description: "Number of montage columns.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "rows".to_string(),
                    description: "Number of montage rows.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "border_width".to_string(),
                    description: "Pixel width of borders between tiles.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        Ok(OpOutput::dataset_only(montage_to_stack(dataset, params)?))
    }
}

impl Operation for ImageStackToHyperstackOp {
    fn name(&self) -> &'static str {
        "image.stack.to_hyperstack"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Convert a linear ImageJ-style stack into a C/Z/T hyperstack.".to_string(),
            params: vec![
                ParamSpec {
                    name: "channels".to_string(),
                    description: "Number of channels in the output hyperstack.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "slices".to_string(),
                    description: "Number of Z slices in the output hyperstack.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "frames".to_string(),
                    description: "Number of time frames in the output hyperstack.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "order".to_string(),
                    description: "Source stack order: czt, ctz, zct, ztc, tcz, or tzc.".to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        Ok(OpOutput::dataset_only(stack_to_hyperstack(
            dataset, params,
        )?))
    }
}

impl Operation for ImageHyperstackToStackOp {
    fn name(&self) -> &'static str {
        "image.hyperstack.to_stack"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Flatten an ImageJ-style C/Z/T hyperstack into a linear Z stack."
                .to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        Ok(OpOutput::dataset_only(hyperstack_to_stack(dataset)?))
    }
}

impl Operation for ImageHyperstackReduceDimensionalityOp {
    fn name(&self) -> &'static str {
        "image.hyperstack.reduce_dimensionality"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description:
                "Reduce an ImageJ-style hyperstack by keeping or collapsing C/Z/T dimensions."
                    .to_string(),
            params: vec![
                ParamSpec {
                    name: "keep_channels".to_string(),
                    description: "When false, keep only the current channel.".to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
                ParamSpec {
                    name: "keep_slices".to_string(),
                    description: "When false, keep only the current Z slice.".to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
                ParamSpec {
                    name: "keep_frames".to_string(),
                    description: "When false, keep only the current time frame.".to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
                ParamSpec {
                    name: "channel".to_string(),
                    description: "Zero-based current channel used when channels are collapsed."
                        .to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "z".to_string(),
                    description: "Zero-based current Z slice used when slices are collapsed."
                        .to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "time".to_string(),
                    description: "Zero-based current time frame used when frames are collapsed."
                        .to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        Ok(OpOutput::dataset_only(hyperstack_reduce_dimensionality(
            dataset, params,
        )?))
    }
}

impl Operation for ImageHyperstackSubsetOp {
    fn name(&self) -> &'static str {
        "image.hyperstack.subset"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Create an ImageJ-style C/Z/T subset from a hyperstack.".to_string(),
            params: vec![
                ParamSpec {
                    name: "channels".to_string(),
                    description: "One-based channel range/list, e.g. `1-3` or `1,3`.".to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "slices".to_string(),
                    description: "One-based Z slice range/list, e.g. `2-14-3`.".to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "frames".to_string(),
                    description: "One-based time frame range/list.".to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        Ok(OpOutput::dataset_only(hyperstack_subset(dataset, params)?))
    }
}

impl Operation for ImageBinOp {
    fn name(&self) -> &'static str {
        "image.bin"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Reduce X/Y/Z dimensions by binned aggregation.".to_string(),
            params: vec![
                ParamSpec {
                    name: "x".to_string(),
                    description: "X shrink factor.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "y".to_string(),
                    description: "Y shrink factor.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "z".to_string(),
                    description: "Z shrink factor.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "method".to_string(),
                    description: "average, median, min, max, or sum.".to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let x_factor = get_optional_usize(params, "x", 2);
        let y_factor = get_optional_usize(params, "y", 2);
        let z_factor = get_optional_usize(params, "z", 1);
        let method = params
            .get("method")
            .and_then(Value::as_str)
            .map(BinMethod::parse)
            .transpose()?
            .unwrap_or(BinMethod::Average);
        Ok(OpOutput::dataset_only(bin_xyz(
            dataset, x_factor, y_factor, z_factor, method,
        )?))
    }
}

impl Operation for ImageFlipOp {
    fn name(&self) -> &'static str {
        "image.flip"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Flip an image horizontally, vertically, or across Z slices.".to_string(),
            params: vec![ParamSpec {
                name: "axis".to_string(),
                description: "One of horizontal, vertical, x, y, z, or depth.".to_string(),
                required: true,
                kind: "string".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let Some(axis) = params.get("axis").and_then(Value::as_str) else {
            return Err(OpsError::InvalidParams("`axis` is required".to_string()));
        };
        let axis = match axis {
            "horizontal" | "x" => AxisKind::X,
            "vertical" | "y" => AxisKind::Y,
            "z" | "depth" => AxisKind::Z,
            other => {
                return Err(OpsError::InvalidParams(format!(
                    "unsupported flip axis `{other}`"
                )));
            }
        };
        Ok(OpOutput::dataset_only(flip_xy(dataset, axis)?))
    }
}

impl Operation for ImageRotate90Op {
    fn name(&self) -> &'static str {
        "image.rotate_90"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Rotate X/Y planes 90 degrees left or right.".to_string(),
            params: vec![ParamSpec {
                name: "direction".to_string(),
                description: "One of right, clockwise, left, or counterclockwise.".to_string(),
                required: true,
                kind: "string".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let Some(direction) = params.get("direction").and_then(Value::as_str) else {
            return Err(OpsError::InvalidParams(
                "`direction` is required".to_string(),
            ));
        };
        let direction = match direction {
            "right" | "clockwise" | "cw" => RotateDirection::Right,
            "left" | "counterclockwise" | "ccw" => RotateDirection::Left,
            other => {
                return Err(OpsError::InvalidParams(format!(
                    "unsupported rotate direction `{other}`"
                )));
            }
        };
        Ok(OpOutput::dataset_only(rotate_90_xy(dataset, direction)?))
    }
}

impl Operation for ImageRotateOp {
    fn name(&self) -> &'static str {
        "image.rotate"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Rotate X/Y planes by an arbitrary angle in degrees.".to_string(),
            params: vec![
                ParamSpec {
                    name: "angle".to_string(),
                    description: "Clockwise rotation angle in degrees.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "fill".to_string(),
                    description: "Fill value for uncovered regions.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "interpolation".to_string(),
                    description: "nearest or bilinear.".to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "enlarge".to_string(),
                    description: "Expand the canvas to fit the rotated image.".to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let angle = get_optional_f32(params, "angle", 15.0);
        let fill = get_optional_f32(params, "fill", 0.0);
        if !angle.is_finite() || !fill.is_finite() {
            return Err(OpsError::InvalidParams(
                "`angle` and `fill` must be finite".to_string(),
            ));
        }
        let interpolation = params
            .get("interpolation")
            .and_then(Value::as_str)
            .map(TranslateInterpolation::parse)
            .transpose()?
            .unwrap_or(TranslateInterpolation::Bilinear);
        let enlarge = params
            .get("enlarge")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Ok(OpOutput::dataset_only(rotate_xy(
            dataset,
            angle,
            fill,
            interpolation,
            enlarge,
        )?))
    }
}

impl Operation for ImageTranslateOp {
    fn name(&self) -> &'static str {
        "image.translate"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Translate X/Y planes by pixel offsets.".to_string(),
            params: vec![
                ParamSpec {
                    name: "x".to_string(),
                    description: "Horizontal offset in pixels; positive moves pixels right."
                        .to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "y".to_string(),
                    description: "Vertical offset in pixels; positive moves pixels down."
                        .to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "fill".to_string(),
                    description: "Fill value for uncovered regions.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "interpolation".to_string(),
                    description: "nearest or bilinear.".to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let x_offset = get_optional_f32(params, "x", 0.0);
        let y_offset = get_optional_f32(params, "y", 0.0);
        let fill = get_optional_f32(params, "fill", 0.0);
        let interpolation = params
            .get("interpolation")
            .and_then(Value::as_str)
            .map(TranslateInterpolation::parse)
            .transpose()?
            .unwrap_or(TranslateInterpolation::Nearest);
        Ok(OpOutput::dataset_only(translate_xy(
            dataset,
            x_offset,
            y_offset,
            fill,
            interpolation,
        )?))
    }
}

impl Operation for ImageMedianFilterOp {
    fn name(&self) -> &'static str {
        "image.median_filter"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Apply an ImageJ-style circular median filter on X/Y planes.".to_string(),
            params: vec![ParamSpec {
                name: "radius".to_string(),
                description: "Circular neighborhood radius; Despeckle uses radius 1.".to_string(),
                required: false,
                kind: "float".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let radius = get_optional_f32(params, "radius", 1.0);
        if !radius.is_finite() || radius < 0.0 {
            return Err(OpsError::InvalidParams(
                "`radius` must be a finite non-negative value".to_string(),
            ));
        }
        if radius <= f32::EPSILON {
            return Ok(OpOutput::dataset_only(dataset.clone()));
        }
        Ok(OpOutput::dataset_only(median_filter_xy(dataset, radius)?))
    }
}

impl Operation for ImageRemoveNaNsOp {
    fn name(&self) -> &'static str {
        "image.remove_nans"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Replace NaN pixels with a NaN-aware neighborhood median.".to_string(),
            params: vec![ParamSpec {
                name: "radius".to_string(),
                description: "Circular neighborhood radius; ImageJ default is 2.".to_string(),
                required: false,
                kind: "float".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        if dataset.metadata.pixel_type != PixelType::F32 {
            return Err(OpsError::UnsupportedLayout(
                "Remove NaNs requires f32 pixel metadata".to_string(),
            ));
        }
        let radius = get_optional_f32(params, "radius", 2.0);
        if !radius.is_finite() || radius < 0.0 {
            return Err(OpsError::InvalidParams(
                "`radius` must be a finite non-negative value".to_string(),
            ));
        }
        Ok(OpOutput::dataset_only(remove_nans_xy(dataset, radius)?))
    }
}

impl Operation for ImageRemoveOutliersOp {
    fn name(&self) -> &'static str {
        "image.remove_outliers"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Replace bright or dark outlier pixels with the neighborhood median."
                .to_string(),
            params: vec![
                ParamSpec {
                    name: "radius".to_string(),
                    description: "Circular neighborhood radius; ImageJ default is 2.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "threshold".to_string(),
                    description: "Outlier deviation threshold; ImageJ default is 50.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "which".to_string(),
                    description: "One of bright or dark.".to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let radius = get_optional_f32(params, "radius", 2.0);
        if !radius.is_finite() || radius < 0.0 {
            return Err(OpsError::InvalidParams(
                "`radius` must be a finite non-negative value".to_string(),
            ));
        }
        let threshold = get_optional_f32(params, "threshold", 50.0);
        if !threshold.is_finite() || threshold < 0.0 {
            return Err(OpsError::InvalidParams(
                "`threshold` must be a finite non-negative value".to_string(),
            ));
        }
        let which = params
            .get("which")
            .and_then(Value::as_str)
            .map(OutlierKind::parse)
            .transpose()?
            .unwrap_or(OutlierKind::Bright);
        Ok(OpOutput::dataset_only(remove_outliers_xy(
            dataset, radius, threshold, which,
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

impl Operation for ImageFindMaximaOp {
    fn name(&self) -> &'static str {
        "image.find_maxima"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Find ImageJ-style local maxima on X/Y planes and return a mask."
                .to_string(),
            params: vec![
                ParamSpec {
                    name: "prominence".to_string(),
                    description: "Minimum difference from neighboring pixels.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "strict".to_string(),
                    description: "Require maxima to be strictly greater than all neighbors."
                        .to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
                ParamSpec {
                    name: "exclude_edges".to_string(),
                    description: "Ignore maxima touching the image edge.".to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
                ParamSpec {
                    name: "light_background".to_string(),
                    description: "Find local minima, matching ImageJ's light-background mode."
                        .to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let prominence = get_optional_f32(params, "prominence", 0.0);
        if !prominence.is_finite() || prominence < 0.0 {
            return Err(OpsError::InvalidParams(
                "`prominence` must be a finite non-negative value".to_string(),
            ));
        }
        let strict = params
            .get("strict")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let exclude_edges = params
            .get("exclude_edges")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let light_background = params
            .get("light_background")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Ok(OpOutput::dataset_only(find_maxima_mask_xy(
            dataset,
            prominence,
            strict,
            exclude_edges,
            light_background,
        )?))
    }
}

impl Operation for ImageShadowOp {
    fn name(&self) -> &'static str {
        "image.shadow"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Apply an ImageJ Process/Shadows directional 3x3 kernel.".to_string(),
            params: vec![ParamSpec {
                name: "direction".to_string(),
                description: "One of north, northeast, east, southeast, south, southwest, west, or northwest."
                    .to_string(),
                required: true,
                kind: "string".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let Some(direction) = params.get("direction").and_then(Value::as_str) else {
            return Err(OpsError::InvalidParams(
                "`direction` is required".to_string(),
            ));
        };
        let direction = ShadowDirection::parse(direction)?;
        Ok(OpOutput::dataset_only(shadow_xy(dataset, direction)?))
    }
}

impl Operation for ImageShadowDemoOp {
    fn name(&self) -> &'static str {
        "image.shadow_demo"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Create an ImageJ Process/Shadows demo stack from X/Y planes.".to_string(),
            params: vec![ParamSpec {
                name: "iterations".to_string(),
                description: "Number of eight-direction demo cycles; ImageJ default is 20."
                    .to_string(),
                required: false,
                kind: "int".to_string(),
            }],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let iterations = get_optional_usize(params, "iterations", 20);
        if iterations == 0 {
            return Err(OpsError::InvalidParams(
                "`iterations` must be > 0".to_string(),
            ));
        }
        Ok(OpOutput::dataset_only(shadow_demo_stack_xy(
            dataset, iterations,
        )?))
    }
}

impl Operation for ImageRankFilterOp {
    fn name(&self) -> &'static str {
        "image.rank_filter"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Apply ImageJ Process/Filters rank filters on X/Y planes.".to_string(),
            params: vec![
                ParamSpec {
                    name: "filter".to_string(),
                    description: "One of mean, minimum, maximum, variance, or top_hat.".to_string(),
                    required: true,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "radius".to_string(),
                    description: "Circular neighborhood radius; ImageJ default is 2.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "light_background".to_string(),
                    description: "For top_hat, use grayscale close for light backgrounds."
                        .to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
                ParamSpec {
                    name: "dont_subtract".to_string(),
                    description: "For top_hat, return the grayscale open/close image.".to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let Some(filter) = params.get("filter").and_then(Value::as_str) else {
            return Err(OpsError::InvalidParams("`filter` is required".to_string()));
        };
        let filter = RankFilterKind::parse(filter)?;
        let radius = get_optional_f32(params, "radius", 2.0);
        if !radius.is_finite() || radius < 0.0 {
            return Err(OpsError::InvalidParams(
                "`radius` must be a finite non-negative value".to_string(),
            ));
        }
        let light_background = params
            .get("light_background")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let dont_subtract = params
            .get("dont_subtract")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Ok(OpOutput::dataset_only(rank_filter_xy(
            dataset,
            filter,
            radius,
            light_background,
            dont_subtract,
        )?))
    }
}

impl Operation for ImageSubtractBackgroundOp {
    fn name(&self) -> &'static str {
        "image.subtract_background"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Estimate and subtract ImageJ-style rolling-ball background.".to_string(),
            params: vec![
                ParamSpec {
                    name: "radius".to_string(),
                    description: "Background radius in pixels; ImageJ default is 50.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "light_background".to_string(),
                    description: "Use a closing background estimate for light backgrounds."
                        .to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
                ParamSpec {
                    name: "create_background".to_string(),
                    description: "Return the estimated background instead of subtracting it."
                        .to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let radius = get_optional_f32(params, "radius", 50.0);
        if !radius.is_finite() || radius < 0.0 {
            return Err(OpsError::InvalidParams(
                "`radius` must be a finite non-negative value".to_string(),
            ));
        }
        let light_background = params
            .get("light_background")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let create_background = params
            .get("create_background")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Ok(OpOutput::dataset_only(top_hat_xy(
            dataset,
            radius,
            light_background,
            create_background,
        )?))
    }
}

impl Operation for ImageUnsharpMaskOp {
    fn name(&self) -> &'static str {
        "image.unsharp_mask"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Apply ImageJ-style Gaussian unsharp masking on X/Y planes.".to_string(),
            params: vec![
                ParamSpec {
                    name: "sigma".to_string(),
                    description: "Gaussian radius/sigma; ImageJ default is 1.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "weight".to_string(),
                    description: "Mask weight in [0, 0.99]; ImageJ default is 0.6.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let sigma = get_optional_f32(params, "sigma", 1.0);
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(OpsError::InvalidParams(
                "`sigma` must be a finite non-negative value".to_string(),
            ));
        }
        let weight = get_optional_f32(params, "weight", 0.6);
        if !weight.is_finite() || !(0.0..=0.99).contains(&weight) {
            return Err(OpsError::InvalidParams(
                "`weight` must be in the range [0, 0.99]".to_string(),
            ));
        }
        Ok(OpOutput::dataset_only(unsharp_mask_xy(
            dataset, sigma, weight,
        )?))
    }
}

impl Operation for ImageConvolveOp {
    fn name(&self) -> &'static str {
        "image.convolve"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Apply a custom odd-sized convolution kernel on X/Y planes.".to_string(),
            params: vec![
                ParamSpec {
                    name: "kernel".to_string(),
                    description: "Flat row-major kernel coefficients.".to_string(),
                    required: true,
                    kind: "array".to_string(),
                },
                ParamSpec {
                    name: "width".to_string(),
                    description: "Kernel width; must be odd and > 0.".to_string(),
                    required: true,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "height".to_string(),
                    description: "Kernel height; must be odd and > 0.".to_string(),
                    required: true,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "normalize".to_string(),
                    description: "Normalize by the kernel sum when non-zero; default true."
                        .to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let width = get_optional_usize(params, "width", 0);
        let height = get_optional_usize(params, "height", 0);
        if width == 0 || height == 0 || width % 2 == 0 || height % 2 == 0 {
            return Err(OpsError::InvalidParams(
                "`width` and `height` must be odd values > 0".to_string(),
            ));
        }

        let kernel = parse_convolution_kernel(params)?;
        if kernel.len() != width * height {
            return Err(OpsError::InvalidParams(format!(
                "`kernel` length {} does not match width * height {}",
                kernel.len(),
                width * height
            )));
        }

        let normalize = params
            .get("normalize")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        Ok(OpOutput::dataset_only(convolve_xy(
            dataset, &kernel, width, height, normalize,
        )?))
    }
}

impl Operation for ImageRankFilter3dOp {
    fn name(&self) -> &'static str {
        "image.rank_filter_3d"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Apply ImageJ Process/Filters 3D rank filters with an ellipsoid kernel."
                .to_string(),
            params: vec![
                ParamSpec {
                    name: "filter".to_string(),
                    description: "One of mean, median, minimum, maximum, or variance.".to_string(),
                    required: true,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "x_radius".to_string(),
                    description: "Ellipsoid radius along X; defaults to radius or 2.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "y_radius".to_string(),
                    description: "Ellipsoid radius along Y; defaults to radius or 2.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "z_radius".to_string(),
                    description: "Ellipsoid radius along Z; defaults to radius or 2.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let Some(filter) = params.get("filter").and_then(Value::as_str) else {
            return Err(OpsError::InvalidParams("`filter` is required".to_string()));
        };
        let filter = RankFilter3dKind::parse(filter)?;
        let radius = get_optional_f32(params, "radius", 2.0);
        let x_radius = get_optional_f32(params, "x_radius", radius);
        let y_radius = get_optional_f32(params, "y_radius", radius);
        let z_radius = get_optional_f32(params, "z_radius", radius);
        for (name, value) in [
            ("x_radius", x_radius),
            ("y_radius", y_radius),
            ("z_radius", z_radius),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(OpsError::InvalidParams(format!(
                    "`{name}` must be a finite non-negative value"
                )));
            }
        }
        Ok(OpOutput::dataset_only(rank_filter_3d(
            dataset, filter, x_radius, y_radius, z_radius,
        )?))
    }
}

impl Operation for ImageSwapQuadrantsOp {
    fn name(&self) -> &'static str {
        "image.swap_quadrants"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Swap X/Y image quadrants as used by ImageJ Process/FFT.".to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        Ok(OpOutput::dataset_only(swap_quadrants_xy(dataset)?))
    }
}

impl Operation for ImageFftPowerSpectrumOp {
    fn name(&self) -> &'static str {
        "image.fft_power_spectrum"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Compute an ImageJ-style log-scaled FFT power spectrum for X/Y planes."
                .to_string(),
            params: vec![],
        }
    }

    fn execute(&self, dataset: &DatasetF32, _params: &Value) -> Result<OpOutput> {
        Ok(OpOutput::dataset_only(fft_power_spectrum_xy(dataset)?))
    }
}

impl Operation for ImageFftBandpassOp {
    fn name(&self) -> &'static str {
        "image.fft_bandpass"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Apply an ImageJ-style FFT bandpass filter on X/Y planes.".to_string(),
            params: vec![
                ParamSpec {
                    name: "filter_large".to_string(),
                    description:
                        "Suppress structures larger than this many pixels; ImageJ default is 40."
                            .to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "filter_small".to_string(),
                    description:
                        "Suppress structures smaller than this many pixels; ImageJ default is 3."
                            .to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "suppress_stripes".to_string(),
                    description: "One of none, horizontal, or vertical.".to_string(),
                    required: false,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "tolerance".to_string(),
                    description: "Stripe direction tolerance percent; ImageJ default is 5."
                        .to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "autoscale".to_string(),
                    description: "Stretch filtered output back to the input range; default true."
                        .to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let filter_large = get_optional_f32(params, "filter_large", 40.0);
        let filter_small = get_optional_f32(params, "filter_small", 3.0);
        let tolerance = get_optional_f32(params, "tolerance", 5.0);
        for (name, value) in [
            ("filter_large", filter_large),
            ("filter_small", filter_small),
            ("tolerance", tolerance),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(OpsError::InvalidParams(format!(
                    "`{name}` must be a finite non-negative value"
                )));
            }
        }
        let stripes = params
            .get("suppress_stripes")
            .and_then(Value::as_str)
            .unwrap_or("none");
        let stripes = StripeSuppression::parse(stripes)?;
        let autoscale = params
            .get("autoscale")
            .and_then(Value::as_bool)
            .unwrap_or(true);

        Ok(OpOutput::dataset_only(fft_bandpass_xy(
            dataset,
            filter_large,
            filter_small,
            stripes,
            tolerance,
            autoscale,
        )?))
    }
}

impl Operation for ImageSurfacePlotOp {
    fn name(&self) -> &'static str {
        "image.surface_plot"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Render an ImageJ-style grayscale surface plot from an XY plane."
                .to_string(),
            params: vec![
                ParamSpec {
                    name: "z".to_string(),
                    description: "Zero-based Z index for stack inputs.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "time".to_string(),
                    description: "Zero-based time index for hyperstacks.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "channel".to_string(),
                    description: "Zero-based channel index for multichannel inputs.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "plot_width".to_string(),
                    description: "Rendered plot width in pixels.".to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "polygon_multiplier".to_string(),
                    description: "ImageJ Surface Plotter polygon multiplier percentage."
                        .to_string(),
                    required: false,
                    kind: "int".to_string(),
                },
                ParamSpec {
                    name: "source_background_lighter".to_string(),
                    description: "Invert height values when the source background is lighter."
                        .to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
                ParamSpec {
                    name: "black_fill".to_string(),
                    description: "Use a black plot background.".to_string(),
                    required: false,
                    kind: "bool".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        Ok(OpOutput::dataset_only(surface_plot_xy(dataset, params)?))
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

fn crop_xy(
    dataset: &DatasetF32,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let input_shape = dataset.shape().to_vec();
    let Some(x_end) = x.checked_add(width) else {
        return Err(OpsError::InvalidParams(
            "crop bounds must fit within the X/Y image dimensions".to_string(),
        ));
    };
    let Some(y_end) = y.checked_add(height) else {
        return Err(OpsError::InvalidParams(
            "crop bounds must fit within the X/Y image dimensions".to_string(),
        ));
    };
    if x >= input_shape[x_axis]
        || y >= input_shape[y_axis]
        || x_end > input_shape[x_axis]
        || y_end > input_shape[y_axis]
    {
        return Err(OpsError::InvalidParams(
            "crop bounds must fit within the X/Y image dimensions".to_string(),
        ));
    }

    let mut output_shape = input_shape.clone();
    output_shape[x_axis] = width;
    output_shape[y_axis] = height;
    let output_len: usize = output_shape.iter().product();
    let mut output = Vec::with_capacity(output_len);
    iterate_indices(&output_shape, |coord| {
        let mut src = coord.to_vec();
        src[x_axis] += x;
        src[y_axis] += y;
        output.push(dataset.data[IxDyn(&src)]);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output)
        .map_err(|_| OpsError::UnsupportedLayout("failed to build cropped dataset".to_string()))?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims[x_axis].size = width;
    metadata.dims[y_axis].size = height;
    shift_origin(&mut metadata, x_axis, "x", x);
    shift_origin(&mut metadata, y_axis, "y", y);
    Ok(Dataset::new(data, metadata)?)
}

fn stack_add_slice(dataset: &DatasetF32, index: usize, fill: f32) -> Result<DatasetF32> {
    let existing_z_axis = dataset.axis_index(AxisKind::Z);
    let mut output_shape = dataset.shape().to_vec();
    let z_axis = match existing_z_axis {
        Some(axis) => {
            let slices = output_shape[axis];
            if index > slices {
                return Err(OpsError::InvalidParams(
                    "`index` must be within the Z stack bounds".to_string(),
                ));
            }
            output_shape[axis] = slices + 1;
            axis
        }
        None => {
            if index > 1 {
                return Err(OpsError::InvalidParams(
                    "`index` must be 0 or 1 for a single-slice image".to_string(),
                ));
            }
            output_shape.push(2);
            output_shape.len() - 1
        }
    };

    let mut output = Vec::with_capacity(output_shape.iter().product());
    iterate_indices(&output_shape, |coord| {
        let z = coord[z_axis];
        if z == index {
            output.push(fill);
            return;
        }
        let mut src = coord.to_vec();
        if existing_z_axis.is_some() {
            src[z_axis] = if z < index { z } else { z - 1 };
        } else {
            src.pop();
        }
        output.push(dataset.data[IxDyn(&src)]);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output)
        .map_err(|_| OpsError::UnsupportedLayout("failed to build stack dataset".to_string()))?;
    let mut metadata = dataset.metadata.clone();
    if let Some(axis) = existing_z_axis {
        metadata.dims[axis].size += 1;
    } else {
        metadata.dims.push(Dim::new(AxisKind::Z, 2));
    }
    Ok(Dataset::new(data, metadata)?)
}

fn stack_delete_slice(dataset: &DatasetF32, index: usize) -> Result<DatasetF32> {
    let z_axis = axis_index(dataset, AxisKind::Z)?;
    let slices = dataset.shape()[z_axis];
    if slices <= 1 {
        return Err(OpsError::InvalidParams(
            "Delete Slice requires a stack with more than one Z slice".to_string(),
        ));
    }
    if index >= slices {
        return Err(OpsError::InvalidParams(
            "`index` must be within the Z stack bounds".to_string(),
        ));
    }

    let mut output_shape = dataset.shape().to_vec();
    output_shape[z_axis] = slices - 1;
    let mut output = Vec::with_capacity(output_shape.iter().product());
    iterate_indices(&output_shape, |coord| {
        let mut src = coord.to_vec();
        src[z_axis] = if coord[z_axis] < index {
            coord[z_axis]
        } else {
            coord[z_axis] + 1
        };
        output.push(dataset.data[IxDyn(&src)]);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output)
        .map_err(|_| OpsError::UnsupportedLayout("failed to build stack dataset".to_string()))?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims[z_axis].size -= 1;
    Ok(Dataset::new(data, metadata)?)
}

fn stack_to_hyperstack(dataset: &DatasetF32, params: &Value) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let stack_axis = axis_index(dataset, AxisKind::Z)?;
    if dataset.axis_index(AxisKind::Channel).is_some()
        || dataset.axis_index(AxisKind::Time).is_some()
    {
        return Err(OpsError::UnsupportedLayout(
            "Stack to Hyperstack requires a linear X/Y/Z stack".to_string(),
        ));
    }
    let input_shape = dataset.shape();
    let width = input_shape[x_axis];
    let height = input_shape[y_axis];
    let stack_size = input_shape[stack_axis];
    if stack_size <= 1 {
        return Err(OpsError::InvalidParams(
            "Stack to Hyperstack requires a stack".to_string(),
        ));
    }

    let channels = get_optional_usize(params, "channels", 1);
    let slices = get_optional_usize(params, "slices", stack_size);
    let frames = get_optional_usize(params, "frames", 1);
    if channels == 0 || slices == 0 || frames == 0 {
        return Err(OpsError::InvalidParams(
            "channels, slices and frames must be positive".to_string(),
        ));
    }
    if channels * slices * frames != stack_size {
        return Err(OpsError::InvalidParams(format!(
            "channels x slices x frames ({}) must equal stack size ({stack_size})",
            channels * slices * frames
        )));
    }
    let order = hyperstack_order(params)?;

    let mut output_shape = vec![height, width];
    let z_axis = if slices > 1 {
        output_shape.push(slices);
        Some(output_shape.len() - 1)
    } else {
        None
    };
    let channel_axis = if channels > 1 {
        output_shape.push(channels);
        Some(output_shape.len() - 1)
    } else {
        None
    };
    let time_axis = if frames > 1 {
        output_shape.push(frames);
        Some(output_shape.len() - 1)
    } else {
        None
    };

    let mut output = Vec::with_capacity(output_shape.iter().product());
    iterate_indices(&output_shape, |coord| {
        let z = z_axis.map(|axis| coord[axis]).unwrap_or(0);
        let channel = channel_axis.map(|axis| coord[axis]).unwrap_or(0);
        let time = time_axis.map(|axis| coord[axis]).unwrap_or(0);
        let stack_index =
            hyperstack_linear_index(order, channel, z, time, channels, slices, frames);
        let mut src = vec![0usize; dataset.ndim()];
        src[y_axis] = coord[0];
        src[x_axis] = coord[1];
        src[stack_axis] = stack_index;
        output.push(dataset.data[IxDyn(&src)]);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build hyperstack dataset".to_string())
    })?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims = vec![Dim::new(AxisKind::Y, height), Dim::new(AxisKind::X, width)];
    if slices > 1 {
        metadata.dims.push(Dim::new(AxisKind::Z, slices));
    }
    if channels > 1 {
        metadata.dims.push(Dim::new(AxisKind::Channel, channels));
    }
    if frames > 1 {
        metadata.dims.push(Dim::new(AxisKind::Time, frames));
    }
    metadata.extras.insert(
        "hyperstack_dimensions".to_string(),
        json!({
            "channels": channels,
            "slices": slices,
            "frames": frames
        }),
    );
    metadata
        .extras
        .insert("hyperstack_source_order".to_string(), json!(order));
    Ok(Dataset::new(data, metadata)?)
}

fn hyperstack_to_stack(dataset: &DatasetF32) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let channel_axis = dataset.axis_index(AxisKind::Channel);
    let z_axis = dataset.axis_index(AxisKind::Z);
    let time_axis = dataset.axis_index(AxisKind::Time);
    if channel_axis.is_none() && time_axis.is_none() {
        return Err(OpsError::UnsupportedLayout(
            "Hyperstack to Stack requires a Channel or Time axis".to_string(),
        ));
    }

    let shape = dataset.shape();
    let width = shape[x_axis];
    let height = shape[y_axis];
    let channels = channel_axis.map(|axis| shape[axis]).unwrap_or(1);
    let slices = z_axis.map(|axis| shape[axis]).unwrap_or(1);
    let frames = time_axis.map(|axis| shape[axis]).unwrap_or(1);
    let stack_size = channels * slices * frames;
    let output_shape = vec![height, width, stack_size];
    let mut output = Vec::with_capacity(output_shape.iter().product());

    iterate_indices(&output_shape, |coord| {
        let stack_index = coord[2];
        let channel = stack_index % channels;
        let z = (stack_index / channels) % slices;
        let time = stack_index / (channels * slices);
        let mut src = vec![0usize; dataset.ndim()];
        src[y_axis] = coord[0];
        src[x_axis] = coord[1];
        if let Some(axis) = channel_axis {
            src[axis] = channel;
        }
        if let Some(axis) = z_axis {
            src[axis] = z;
        }
        if let Some(axis) = time_axis {
            src[axis] = time;
        }
        output.push(dataset.data[IxDyn(&src)]);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output)
        .map_err(|_| OpsError::UnsupportedLayout("failed to build flattened stack".to_string()))?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims = vec![
        Dim::new(AxisKind::Y, height),
        Dim::new(AxisKind::X, width),
        Dim::new(AxisKind::Z, stack_size),
    ];
    metadata.extras.insert(
        "hyperstack_to_stack_dimensions".to_string(),
        json!({
            "channels": channels,
            "slices": slices,
            "frames": frames
        }),
    );
    Ok(Dataset::new(data, metadata)?)
}

fn hyperstack_reduce_dimensionality(dataset: &DatasetF32, params: &Value) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let channel_axis = dataset.axis_index(AxisKind::Channel);
    let z_axis = dataset.axis_index(AxisKind::Z);
    let time_axis = dataset.axis_index(AxisKind::Time);
    if channel_axis.is_none() && time_axis.is_none() {
        return Err(OpsError::UnsupportedLayout(
            "Reduce Dimensionality requires a hyperstack".to_string(),
        ));
    }

    let shape = dataset.shape();
    let width = shape[x_axis];
    let height = shape[y_axis];
    let channels = channel_axis.map(|axis| shape[axis]).unwrap_or(1);
    let slices = z_axis.map(|axis| shape[axis]).unwrap_or(1);
    let frames = time_axis.map(|axis| shape[axis]).unwrap_or(1);
    let keep_channels = optional_bool_param(params, "keep_channels", true);
    let keep_slices = optional_bool_param(params, "keep_slices", true);
    let keep_frames = optional_bool_param(params, "keep_frames", true);
    let current_channel = checked_axis_param(params, "channel", channels)?;
    let current_z = checked_axis_param(params, "z", slices)?;
    let current_time = checked_axis_param(params, "time", frames)?;

    let mut output_shape = vec![height, width];
    let output_z_axis = if keep_slices && slices > 1 {
        output_shape.push(slices);
        Some(output_shape.len() - 1)
    } else {
        None
    };
    let output_channel_axis = if keep_channels && channels > 1 {
        output_shape.push(channels);
        Some(output_shape.len() - 1)
    } else {
        None
    };
    let output_time_axis = if keep_frames && frames > 1 {
        output_shape.push(frames);
        Some(output_shape.len() - 1)
    } else {
        None
    };

    let mut output = Vec::with_capacity(output_shape.iter().product());
    iterate_indices(&output_shape, |coord| {
        let mut src = vec![0usize; dataset.ndim()];
        src[y_axis] = coord[0];
        src[x_axis] = coord[1];
        if let Some(axis) = z_axis {
            src[axis] = output_z_axis
                .map(|output_axis| coord[output_axis])
                .unwrap_or(current_z);
        }
        if let Some(axis) = channel_axis {
            src[axis] = output_channel_axis
                .map(|output_axis| coord[output_axis])
                .unwrap_or(current_channel);
        }
        if let Some(axis) = time_axis {
            src[axis] = output_time_axis
                .map(|output_axis| coord[output_axis])
                .unwrap_or(current_time);
        }
        output.push(dataset.data[IxDyn(&src)]);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build reduced hyperstack".to_string())
    })?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims = vec![Dim::new(AxisKind::Y, height), Dim::new(AxisKind::X, width)];
    if let Some(axis) = output_z_axis {
        metadata
            .dims
            .push(Dim::new(AxisKind::Z, output_shape[axis]));
    }
    if let Some(axis) = output_channel_axis {
        metadata
            .dims
            .push(Dim::new(AxisKind::Channel, output_shape[axis]));
    }
    if let Some(axis) = output_time_axis {
        metadata
            .dims
            .push(Dim::new(AxisKind::Time, output_shape[axis]));
    }
    metadata.extras.insert(
        "reduced_hyperstack_source_position".to_string(),
        json!({
            "channel": current_channel,
            "z": current_z,
            "time": current_time
        }),
    );
    metadata.extras.insert(
        "reduced_hyperstack_kept_axes".to_string(),
        json!({
            "channels": keep_channels,
            "slices": keep_slices,
            "frames": keep_frames
        }),
    );
    Ok(Dataset::new(data, metadata)?)
}

fn hyperstack_subset(dataset: &DatasetF32, params: &Value) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let channel_axis = dataset.axis_index(AxisKind::Channel);
    let z_axis = dataset.axis_index(AxisKind::Z);
    let time_axis = dataset.axis_index(AxisKind::Time);
    if channel_axis.is_none() && z_axis.is_none() && time_axis.is_none() {
        return Err(OpsError::UnsupportedLayout(
            "Make Subset requires a stack or hyperstack".to_string(),
        ));
    }

    let shape = dataset.shape();
    let width = shape[x_axis];
    let height = shape[y_axis];
    let channels = channel_axis.map(|axis| shape[axis]).unwrap_or(1);
    let slices = z_axis.map(|axis| shape[axis]).unwrap_or(1);
    let frames = time_axis.map(|axis| shape[axis]).unwrap_or(1);
    let selected_channels = parse_axis_selection(params, &["channels", "c"], channels)?;
    let selected_slices = parse_axis_selection(params, &["slices", "z"], slices)?;
    let selected_frames = parse_axis_selection(params, &["frames", "t", "time"], frames)?;

    let mut output_shape = vec![height, width];
    let output_z_axis = if selected_slices.len() > 1 {
        output_shape.push(selected_slices.len());
        Some(output_shape.len() - 1)
    } else {
        None
    };
    let output_channel_axis = if selected_channels.len() > 1 {
        output_shape.push(selected_channels.len());
        Some(output_shape.len() - 1)
    } else {
        None
    };
    let output_time_axis = if selected_frames.len() > 1 {
        output_shape.push(selected_frames.len());
        Some(output_shape.len() - 1)
    } else {
        None
    };

    let mut output = Vec::with_capacity(output_shape.iter().product());
    iterate_indices(&output_shape, |coord| {
        let mut src = vec![0usize; dataset.ndim()];
        src[y_axis] = coord[0];
        src[x_axis] = coord[1];
        if let Some(axis) = z_axis {
            let index = output_z_axis.map(|axis| coord[axis]).unwrap_or(0);
            src[axis] = selected_slices[index];
        }
        if let Some(axis) = channel_axis {
            let index = output_channel_axis.map(|axis| coord[axis]).unwrap_or(0);
            src[axis] = selected_channels[index];
        }
        if let Some(axis) = time_axis {
            let index = output_time_axis.map(|axis| coord[axis]).unwrap_or(0);
            src[axis] = selected_frames[index];
        }
        output.push(dataset.data[IxDyn(&src)]);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output)
        .map_err(|_| OpsError::UnsupportedLayout("failed to build hyperstack subset".to_string()))?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims = vec![Dim::new(AxisKind::Y, height), Dim::new(AxisKind::X, width)];
    if let Some(axis) = output_z_axis {
        metadata
            .dims
            .push(Dim::new(AxisKind::Z, output_shape[axis]));
    }
    if let Some(axis) = output_channel_axis {
        metadata
            .dims
            .push(Dim::new(AxisKind::Channel, output_shape[axis]));
    }
    if let Some(axis) = output_time_axis {
        metadata
            .dims
            .push(Dim::new(AxisKind::Time, output_shape[axis]));
    }
    metadata.extras.insert(
        "hyperstack_subset_channels".to_string(),
        json!(
            selected_channels
                .iter()
                .map(|index| index + 1)
                .collect::<Vec<_>>()
        ),
    );
    metadata.extras.insert(
        "hyperstack_subset_slices".to_string(),
        json!(
            selected_slices
                .iter()
                .map(|index| index + 1)
                .collect::<Vec<_>>()
        ),
    );
    metadata.extras.insert(
        "hyperstack_subset_frames".to_string(),
        json!(
            selected_frames
                .iter()
                .map(|index| index + 1)
                .collect::<Vec<_>>()
        ),
    );
    Ok(Dataset::new(data, metadata)?)
}

fn parse_axis_selection(
    params: &Value,
    keys: &[&str],
    size: usize,
) -> Result<Vec<usize>> {
    for key in keys {
        if let Some(selection) = params
            .get(*key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|selection| !selection.is_empty())
        {
            return parse_one_based_slice_selection(selection, size);
        }
    }
    Ok((0..size).collect())
}

fn optional_bool_param(params: &Value, key: &str, default: bool) -> bool {
    params.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn checked_axis_param(params: &Value, key: &str, size: usize) -> Result<usize> {
    let index = get_optional_usize(params, key, 0);
    if index >= size {
        return Err(OpsError::InvalidParams(format!(
            "`{key}` index is outside axis bounds"
        )));
    }
    Ok(index)
}

fn hyperstack_order(params: &Value) -> Result<&'static str> {
    match params
        .get("order")
        .and_then(Value::as_str)
        .unwrap_or("czt")
        .to_ascii_lowercase()
        .as_str()
    {
        "czt" | "default" => Ok("czt"),
        "ctz" => Ok("ctz"),
        "zct" => Ok("zct"),
        "ztc" => Ok("ztc"),
        "tcz" => Ok("tcz"),
        "tzc" => Ok("tzc"),
        other => Err(OpsError::InvalidParams(format!(
            "unsupported hyperstack order `{other}`"
        ))),
    }
}

fn hyperstack_linear_index(
    order: &str,
    channel: usize,
    z: usize,
    time: usize,
    channels: usize,
    slices: usize,
    frames: usize,
) -> usize {
    let (first, middle, last) = match order {
        "ctz" => ((channel, channels), (time, frames), (z, slices)),
        "zct" => ((z, slices), (channel, channels), (time, frames)),
        "ztc" => ((z, slices), (time, frames), (channel, channels)),
        "tcz" => ((time, frames), (channel, channels), (z, slices)),
        "tzc" => ((time, frames), (z, slices), (channel, channels)),
        _ => ((channel, channels), (z, slices), (time, frames)),
    };
    first.0 + middle.0 * first.1 + last.0 * first.1 * middle.1
}

fn stack_substack(dataset: &DatasetF32, params: &Value) -> Result<DatasetF32> {
    let z_axis = axis_index(dataset, AxisKind::Z)?;
    let input_shape = dataset.shape().to_vec();
    let slices = input_shape[z_axis];
    if slices <= 1 {
        return Err(OpsError::InvalidParams(
            "Make Substack requires a stack with more than one Z slice".to_string(),
        ));
    }

    let selected = parse_substack_selection(params, slices)?;
    if selected.is_empty() {
        return Err(OpsError::InvalidParams(
            "substack selection must include at least one slice".to_string(),
        ));
    }

    let mut output_shape = input_shape.clone();
    output_shape[z_axis] = selected.len();
    let mut output = Vec::with_capacity(output_shape.iter().product());
    iterate_indices(&output_shape, |coord| {
        let mut src = coord.to_vec();
        src[z_axis] = selected[coord[z_axis]];
        output.push(dataset.data[IxDyn(&src)]);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output)
        .map_err(|_| OpsError::UnsupportedLayout("failed to build substack".to_string()))?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims[z_axis].size = selected.len();
    metadata.extras.insert(
        "substack_slices".to_string(),
        json!(selected.iter().map(|slice| slice + 1).collect::<Vec<_>>()),
    );
    Ok(Dataset::new(data, metadata)?)
}

fn parse_substack_selection(params: &Value, slices: usize) -> Result<Vec<usize>> {
    if let Some(selection) = params
        .get("slices")
        .or_else(|| params.get("range"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|selection| !selection.is_empty())
    {
        return parse_one_based_slice_selection(selection, slices);
    }

    if let Some(indices) = params.get("indices").and_then(Value::as_array) {
        let mut selected = Vec::with_capacity(indices.len());
        for value in indices {
            let Some(index) = value.as_u64() else {
                return Err(OpsError::InvalidParams(
                    "`indices` must contain non-negative integers".to_string(),
                ));
            };
            let index = index as usize;
            if index >= slices {
                return Err(OpsError::InvalidParams(
                    "`indices` must be within the Z stack bounds".to_string(),
                ));
            }
            selected.push(index);
        }
        return Ok(selected);
    }

    Ok((0..slices).collect())
}

fn parse_one_based_slice_selection(selection: &str, slices: usize) -> Result<Vec<usize>> {
    let normalized = selection.replace(' ', "");
    if normalized.is_empty() {
        return Err(OpsError::InvalidParams(
            "substack slice selection cannot be empty".to_string(),
        ));
    }
    if normalized.contains(',') {
        return normalized
            .split(',')
            .map(|part| parse_one_based_slice(part, slices))
            .collect();
    }

    let parts = normalized.split('-').collect::<Vec<_>>();
    match parts.as_slice() {
        [single] => Ok(vec![parse_one_based_slice(single, slices)?]),
        [first, last] | [first, last, ""] => {
            let first = parse_one_based_slice(first, slices)?;
            let last = parse_one_based_slice(last, slices)?;
            if first > last {
                return Err(OpsError::InvalidParams(
                    "substack range start must be <= range end".to_string(),
                ));
            }
            Ok((first..=last).collect())
        }
        [first, last, increment] => {
            let first = parse_one_based_slice(first, slices)?;
            let last = parse_one_based_slice(last, slices)?;
            let increment = increment.parse::<usize>().map_err(|_| {
                OpsError::InvalidParams("substack range increment must be an integer".to_string())
            })?;
            if increment == 0 {
                return Err(OpsError::InvalidParams(
                    "substack range increment must be > 0".to_string(),
                ));
            }
            if first > last {
                return Err(OpsError::InvalidParams(
                    "substack range start must be <= range end".to_string(),
                ));
            }
            Ok((first..=last).step_by(increment).collect())
        }
        _ => Err(OpsError::InvalidParams(
            "invalid substack slice selection".to_string(),
        )),
    }
}

fn parse_one_based_slice(value: &str, slices: usize) -> Result<usize> {
    let slice = value
        .parse::<usize>()
        .map_err(|_| OpsError::InvalidParams("slice selection must be an integer".to_string()))?;
    if slice == 0 || slice > slices {
        return Err(OpsError::InvalidParams(
            "slice selection must be within the Z stack bounds".to_string(),
        ));
    }
    Ok(slice - 1)
}

#[derive(Debug, Clone, Copy)]
enum ResliceStart {
    Top,
    Bottom,
    Left,
    Right,
}

impl ResliceStart {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "top" | "y" => Ok(Self::Top),
            "bottom" => Ok(Self::Bottom),
            "left" | "x" => Ok(Self::Left),
            "right" => Ok(Self::Right),
            other => Err(OpsError::InvalidParams(format!(
                "unsupported reslice start `{other}`"
            ))),
        }
    }

    fn is_horizontal(self) -> bool {
        matches!(self, Self::Top | Self::Bottom)
    }
}

fn stack_reslice(dataset: &DatasetF32, params: &Value) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let z_axis = axis_index(dataset, AxisKind::Z)?;
    let input_shape = dataset.shape().to_vec();
    if input_shape[z_axis] <= 1 {
        return Err(OpsError::InvalidParams(
            "Reslice requires a stack with more than one Z slice".to_string(),
        ));
    }
    if dataset.ndim() != 3 {
        return Err(OpsError::UnsupportedLayout(
            "Reslice currently supports X/Y/Z stacks".to_string(),
        ));
    }

    let start = params
        .get("start")
        .or_else(|| params.get("start_at"))
        .and_then(Value::as_str)
        .map(ResliceStart::parse)
        .transpose()?
        .unwrap_or(ResliceStart::Top);

    let x_size = input_shape[x_axis];
    let y_size = input_shape[y_axis];
    let z_size = input_shape[z_axis];
    let (output_shape, output_x_source, output_z_source) = if start.is_horizontal() {
        (vec![z_size, x_size, y_size], x_axis, y_axis)
    } else {
        (vec![z_size, y_size, x_size], y_axis, x_axis)
    };

    let mut output = Vec::with_capacity(output_shape.iter().product());
    iterate_indices(&output_shape, |coord| {
        let mut src = vec![0usize; input_shape.len()];
        src[z_axis] = coord[0];
        src[output_x_source] = coord[1];
        let depth = coord[2];
        src[output_z_source] = match start {
            ResliceStart::Top | ResliceStart::Left => depth,
            ResliceStart::Bottom | ResliceStart::Right => output_shape[2] - 1 - depth,
        };
        output.push(dataset.data[IxDyn(&src)]);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output)
        .map_err(|_| OpsError::UnsupportedLayout("failed to build reslice stack".to_string()))?;
    let mut metadata = Metadata {
        dims: if start.is_horizontal() {
            vec![
                remapped_dim(&dataset.metadata.dims[z_axis], AxisKind::Y, z_size),
                remapped_dim(&dataset.metadata.dims[x_axis], AxisKind::X, x_size),
                remapped_dim(&dataset.metadata.dims[y_axis], AxisKind::Z, y_size),
            ]
        } else {
            vec![
                remapped_dim(&dataset.metadata.dims[z_axis], AxisKind::Y, z_size),
                remapped_dim(&dataset.metadata.dims[y_axis], AxisKind::X, y_size),
                remapped_dim(&dataset.metadata.dims[x_axis], AxisKind::Z, x_size),
            ]
        },
        pixel_type: dataset.metadata.pixel_type,
        channel_names: dataset.metadata.channel_names.clone(),
        source: dataset.metadata.source.clone(),
        extras: dataset.metadata.extras.clone(),
    };
    metadata
        .extras
        .insert("reslice_start".to_string(), json!(format!("{start:?}")));
    metadata.extras.insert(
        "reslice_source_axes".to_string(),
        json!(["Z", "orthogonal", "depth"]),
    );
    Ok(Dataset::new(data, metadata)?)
}

fn remapped_dim(source: &Dim, axis: AxisKind, size: usize) -> Dim {
    let mut dim = source.clone();
    dim.axis = axis;
    dim.size = size;
    dim
}

fn stack_reduce(dataset: &DatasetF32, factor: usize) -> Result<DatasetF32> {
    let z_axis = axis_index(dataset, AxisKind::Z)?;
    let input_shape = dataset.shape().to_vec();
    let slices = input_shape[z_axis];
    if slices <= 1 {
        return Err(OpsError::InvalidParams(
            "Reduce requires a stack with more than one Z slice".to_string(),
        ));
    }
    if factor == 0 {
        return Err(OpsError::InvalidParams(
            "reduction factor must be > 0".to_string(),
        ));
    }

    let reduced_slices = slices.div_ceil(factor);
    let mut output_shape = input_shape.clone();
    output_shape[z_axis] = reduced_slices;
    let mut output = Vec::with_capacity(output_shape.iter().product());
    iterate_indices(&output_shape, |coord| {
        let mut src = coord.to_vec();
        src[z_axis] = coord[z_axis] * factor;
        output.push(dataset.data[IxDyn(&src)]);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output)
        .map_err(|_| OpsError::UnsupportedLayout("failed to build reduced stack".to_string()))?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims[z_axis].size = reduced_slices;
    if let Some(spacing) = metadata.dims[z_axis].spacing {
        metadata.dims[z_axis].spacing = Some(spacing * factor as f32);
    }
    metadata
        .extras
        .insert("stack_reduce_factor".to_string(), json!(factor));
    Ok(Dataset::new(data, metadata)?)
}

fn stack_z_project(dataset: &DatasetF32, params: &Value) -> Result<DatasetF32> {
    let z_axis = axis_index(dataset, AxisKind::Z)?;
    let slices = dataset.shape()[z_axis];
    if slices <= 1 {
        return Err(OpsError::InvalidParams(
            "Z Project requires a stack with more than one Z slice".to_string(),
        ));
    }
    let start = get_optional_usize(params, "start", 0);
    let stop = get_optional_usize(params, "stop", slices - 1);
    if start > stop || stop >= slices {
        return Err(OpsError::InvalidParams(
            "Z Project start/stop must be ordered and within the Z stack bounds".to_string(),
        ));
    }
    let method = projection_method(params)?;
    let input_shape = dataset.shape().to_vec();
    let mut output_shape = input_shape.clone();
    output_shape.remove(z_axis);
    let mut output = Vec::with_capacity(output_shape.iter().product());
    iterate_indices(&output_shape, |coord| {
        let mut values = Vec::with_capacity(stop - start + 1);
        for z in start..=stop {
            let mut src = Vec::with_capacity(input_shape.len());
            for axis in 0..input_shape.len() {
                if axis == z_axis {
                    src.push(z);
                } else {
                    let output_axis = if axis < z_axis { axis } else { axis - 1 };
                    src.push(coord[output_axis]);
                }
            }
            values.push(dataset.data[IxDyn(&src)]);
        }
        output.push(project_values(&mut values, method));
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output)
        .map_err(|_| OpsError::UnsupportedLayout("failed to build Z projection".to_string()))?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims.remove(z_axis);
    metadata
        .extras
        .insert("z_projection_method".to_string(), json!(method.name()));
    metadata
        .extras
        .insert("z_projection_start".to_string(), json!(start));
    metadata
        .extras
        .insert("z_projection_stop".to_string(), json!(stop));
    Ok(Dataset::new(data, metadata)?)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectionMethod {
    Average,
    Max,
    Min,
    Sum,
    StandardDeviation,
    Median,
}

impl ProjectionMethod {
    fn name(self) -> &'static str {
        match self {
            Self::Average => "average",
            Self::Max => "max",
            Self::Min => "min",
            Self::Sum => "sum",
            Self::StandardDeviation => "sd",
            Self::Median => "median",
        }
    }
}

fn projection_method(params: &Value) -> Result<ProjectionMethod> {
    let method = params
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("average")
        .to_ascii_lowercase();
    match method.as_str() {
        "average" | "avg" | "average intensity" => Ok(ProjectionMethod::Average),
        "max" | "maximum" | "max intensity" => Ok(ProjectionMethod::Max),
        "min" | "minimum" | "min intensity" => Ok(ProjectionMethod::Min),
        "sum" | "sum slices" => Ok(ProjectionMethod::Sum),
        "sd" | "stddev" | "standard_deviation" | "standard deviation" => {
            Ok(ProjectionMethod::StandardDeviation)
        }
        "median" => Ok(ProjectionMethod::Median),
        other => Err(OpsError::InvalidParams(format!(
            "unsupported Z projection method `{other}`"
        ))),
    }
}

fn project_values(values: &mut [f32], method: ProjectionMethod) -> f32 {
    match method {
        ProjectionMethod::Average => {
            let mut sum = 0.0;
            let mut count = 0usize;
            for value in values.iter().copied().filter(|value| value.is_finite()) {
                sum += value;
                count += 1;
            }
            if count == 0 {
                f32::NAN
            } else {
                sum / count as f32
            }
        }
        ProjectionMethod::Max => values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .reduce(f32::max)
            .unwrap_or(f32::NAN),
        ProjectionMethod::Min => values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .reduce(f32::min)
            .unwrap_or(f32::NAN),
        ProjectionMethod::Sum => values.iter().copied().sum(),
        ProjectionMethod::StandardDeviation => {
            if values.len() <= 1 {
                return 0.0;
            }
            let count = values.len() as f32;
            let sum = values.iter().copied().sum::<f32>();
            let sum2 = values.iter().map(|value| value * value).sum::<f32>();
            let variance = ((count * sum2 - sum * sum) / count / (count - 1.0)).max(0.0);
            variance.sqrt()
        }
        ProjectionMethod::Median => {
            values.sort_by(|left, right| left.total_cmp(right));
            let mid = values.len() / 2;
            if values.len() % 2 == 0 {
                (values[mid - 1] + values[mid]) * 0.5
            } else {
                values[mid]
            }
        }
    }
}

fn stack_grouped_z_project(dataset: &DatasetF32, params: &Value) -> Result<DatasetF32> {
    let z_axis = axis_index(dataset, AxisKind::Z)?;
    let input_shape = dataset.shape().to_vec();
    let slices = input_shape[z_axis];
    let group_size = get_optional_usize(params, "group_size", slices);
    if group_size == 0 || group_size > slices || slices % group_size != 0 {
        return Err(OpsError::InvalidParams(
            "group_size must be a positive factor of the Z stack size".to_string(),
        ));
    }
    let method = projection_method(params)?;
    let groups = slices / group_size;
    let mut output_shape = input_shape.clone();
    output_shape[z_axis] = groups;
    let mut output = Vec::with_capacity(output_shape.iter().product());
    iterate_indices(&output_shape, |coord| {
        let group_start = coord[z_axis] * group_size;
        let mut values = Vec::with_capacity(group_size);
        for offset in 0..group_size {
            let mut src = coord.to_vec();
            src[z_axis] = group_start + offset;
            values.push(dataset.data[IxDyn(&src)]);
        }
        output.push(project_values(&mut values, method));
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build grouped Z projection".to_string())
    })?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims[z_axis].size = groups;
    if let Some(spacing) = metadata.dims[z_axis].spacing {
        metadata.dims[z_axis].spacing = Some(spacing * group_size as f32);
    }
    metadata.extras.insert(
        "grouped_z_projection_method".to_string(),
        json!(method.name()),
    );
    metadata.extras.insert(
        "grouped_z_projection_group_size".to_string(),
        json!(group_size),
    );
    Ok(Dataset::new(data, metadata)?)
}

fn stack_montage(dataset: &DatasetF32, params: &Value) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let z_axis = axis_index(dataset, AxisKind::Z)?;
    let input_shape = dataset.shape().to_vec();
    let slices = input_shape[z_axis];
    if slices <= 1 {
        return Err(OpsError::InvalidParams(
            "Make Montage requires a stack with more than one Z slice".to_string(),
        ));
    }

    let first = get_optional_usize(params, "first", 0);
    let last = get_optional_usize(params, "last", slices - 1);
    let increment = get_optional_usize(params, "increment", 1);
    if first > last || last >= slices || increment == 0 {
        return Err(OpsError::InvalidParams(
            "montage first/last/increment must describe a valid Z slice range".to_string(),
        ));
    }
    let selected_count = ((last - first) / increment) + 1;
    let (default_columns, default_rows) = default_montage_grid(selected_count);
    let columns = get_optional_usize(params, "columns", default_columns);
    let rows = get_optional_usize(params, "rows", default_rows);
    if columns == 0 || rows == 0 {
        return Err(OpsError::InvalidParams(
            "montage columns and rows must be > 0".to_string(),
        ));
    }
    let scale = get_optional_f32(params, "scale", 1.0);
    if !scale.is_finite() || scale <= 0.0 {
        return Err(OpsError::InvalidParams(
            "montage scale must be a positive finite value".to_string(),
        ));
    }
    let border = get_optional_usize(params, "border_width", 0);
    let fill = get_optional_f32(params, "fill", 0.0);
    let tile_width = ((input_shape[x_axis] as f32) * scale).floor().max(1.0) as usize;
    let tile_height = ((input_shape[y_axis] as f32) * scale).floor().max(1.0) as usize;
    let output_width = tile_width * columns + border * columns.saturating_sub(1);
    let output_height = tile_height * rows + border * rows.saturating_sub(1);

    let mut output_shape = input_shape.clone();
    output_shape.remove(z_axis);
    let output_x_axis = if x_axis < z_axis { x_axis } else { x_axis - 1 };
    let output_y_axis = if y_axis < z_axis { y_axis } else { y_axis - 1 };
    output_shape[output_x_axis] = output_width;
    output_shape[output_y_axis] = output_height;
    let mut output = vec![fill; output_shape.iter().product()];
    let output_capacity = columns * rows;

    for (tile_index, z) in (first..=last)
        .step_by(increment)
        .take(output_capacity)
        .enumerate()
    {
        let tile_column = tile_index % columns;
        let tile_row = tile_index / columns;
        let x_origin = tile_column * (tile_width + border);
        let y_origin = tile_row * (tile_height + border);
        iterate_indices(&output_shape, |coord| {
            let x = coord[output_x_axis];
            let y = coord[output_y_axis];
            if x < x_origin
                || x >= x_origin + tile_width
                || y < y_origin
                || y >= y_origin + tile_height
            {
                return;
            }
            let src_x = (((x - x_origin) as f32) / scale)
                .floor()
                .clamp(0.0, (input_shape[x_axis] - 1) as f32) as usize;
            let src_y = (((y - y_origin) as f32) / scale)
                .floor()
                .clamp(0.0, (input_shape[y_axis] - 1) as f32) as usize;
            let mut src = Vec::with_capacity(input_shape.len());
            for axis in 0..input_shape.len() {
                if axis == x_axis {
                    src.push(src_x);
                } else if axis == y_axis {
                    src.push(src_y);
                } else if axis == z_axis {
                    src.push(z);
                } else {
                    let output_axis = if axis < z_axis { axis } else { axis - 1 };
                    src.push(coord[output_axis]);
                }
            }
            output[linear_offset(&output_shape, coord)] = dataset.data[IxDyn(&src)];
        });
    }

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output)
        .map_err(|_| OpsError::UnsupportedLayout("failed to build montage".to_string()))?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims.remove(z_axis);
    metadata.dims[output_x_axis].size = output_width;
    metadata.dims[output_y_axis].size = output_height;
    metadata
        .extras
        .insert("montage_columns".to_string(), json!(columns));
    metadata
        .extras
        .insert("montage_rows".to_string(), json!(rows));
    metadata
        .extras
        .insert("montage_first".to_string(), json!(first));
    metadata
        .extras
        .insert("montage_last".to_string(), json!(last));
    metadata
        .extras
        .insert("montage_increment".to_string(), json!(increment));
    Ok(Dataset::new(data, metadata)?)
}

fn default_montage_grid(slice_count: usize) -> (usize, usize) {
    let rows = (slice_count as f64).sqrt().floor().max(1.0) as usize;
    let columns = slice_count.div_ceil(rows);
    (columns.max(1), rows.max(1))
}

fn montage_to_stack(dataset: &DatasetF32, params: &Value) -> Result<DatasetF32> {
    if dataset.axis_index(AxisKind::Z).is_some() {
        return Err(OpsError::InvalidParams(
            "Montage to Stack requires a single montage image, not a Z stack".to_string(),
        ));
    }
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let input_shape = dataset.shape().to_vec();
    let columns_default = metadata_usize(dataset, "montage_columns").unwrap_or(2);
    let rows_default = metadata_usize(dataset, "montage_rows").unwrap_or(2);
    let columns = get_optional_usize(params, "columns", columns_default);
    let rows = get_optional_usize(params, "rows", rows_default);
    let border = get_optional_usize(params, "border_width", 0);
    if columns == 0 || rows == 0 {
        return Err(OpsError::InvalidParams(
            "montage columns and rows must be > 0".to_string(),
        ));
    }
    let total_border_x = border * columns.saturating_sub(1);
    let total_border_y = border * rows.saturating_sub(1);
    if input_shape[x_axis] <= total_border_x || input_shape[y_axis] <= total_border_y {
        return Err(OpsError::InvalidParams(
            "montage border is too large for the image dimensions".to_string(),
        ));
    }
    let usable_width = input_shape[x_axis] - total_border_x;
    let usable_height = input_shape[y_axis] - total_border_y;
    if usable_width % columns != 0 || usable_height % rows != 0 {
        return Err(OpsError::InvalidParams(
            "montage dimensions must divide evenly by columns/rows after borders".to_string(),
        ));
    }
    let tile_width = usable_width / columns;
    let tile_height = usable_height / rows;
    if tile_width == 0 || tile_height == 0 {
        return Err(OpsError::InvalidParams(
            "montage tiles must have nonzero dimensions".to_string(),
        ));
    }

    let mut output_shape = input_shape.clone();
    output_shape[x_axis] = tile_width;
    output_shape[y_axis] = tile_height;
    output_shape.push(columns * rows);
    let z_axis = output_shape.len() - 1;
    let mut output = Vec::with_capacity(output_shape.iter().product());
    iterate_indices(&output_shape, |coord| {
        let z = coord[z_axis];
        let tile_column = z % columns;
        let tile_row = z / columns;
        let mut src = coord[..z_axis].to_vec();
        src[x_axis] = tile_column * (tile_width + border) + coord[x_axis];
        src[y_axis] = tile_row * (tile_height + border) + coord[y_axis];
        output.push(dataset.data[IxDyn(&src)]);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build stack from montage".to_string())
    })?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims[x_axis].size = tile_width;
    metadata.dims[y_axis].size = tile_height;
    metadata.dims.push(Dim::new(AxisKind::Z, columns * rows));
    metadata
        .extras
        .insert("montage_to_stack_columns".to_string(), json!(columns));
    metadata
        .extras
        .insert("montage_to_stack_rows".to_string(), json!(rows));
    Ok(Dataset::new(data, metadata)?)
}

fn metadata_usize(dataset: &DatasetF32, key: &str) -> Option<usize> {
    dataset
        .metadata
        .extras
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
}

fn apply_coordinate_bounds(
    metadata: &mut Metadata,
    axis: usize,
    params: &Value,
    start_key: &str,
    end_key: &str,
    label: &str,
) -> Result<()> {
    let start = optional_f32_param(params, start_key)?;
    let end = optional_f32_param(params, end_key)?;
    match (start, end) {
        (None, None) => Ok(()),
        (Some(start), Some(end)) => {
            let span = end - start;
            if !span.is_finite() || span == 0.0 {
                return Err(OpsError::InvalidParams(format!(
                    "`{end_key}` must differ from `{start_key}`"
                )));
            }
            metadata.dims[axis].spacing = Some((span / metadata.dims[axis].size as f32).abs());
            metadata
                .extras
                .insert(format!("{label}_origin_coordinate"), json!(start));
            metadata
                .extras
                .insert(format!("{label}_coordinate_inverted"), json!(span < 0.0));
            Ok(())
        }
        _ => Err(OpsError::InvalidParams(format!(
            "`{start_key}` and `{end_key}` must be provided together"
        ))),
    }
}

fn apply_coordinate_units(
    metadata: &mut Metadata,
    x_axis: usize,
    y_axis: usize,
    z_axis: Option<usize>,
    params: &Value,
) {
    let x_unit = params
        .get("x_unit")
        .and_then(Value::as_str)
        .filter(|unit| !unit.is_empty())
        .map(str::to_string);
    if let Some(unit) = &x_unit {
        metadata.dims[x_axis].unit = Some(unit.clone());
    }
    let inherited_x = metadata.dims[x_axis].unit.clone();
    let y_unit = params
        .get("y_unit")
        .and_then(Value::as_str)
        .filter(|unit| !unit.is_empty() && *unit != "<same as x unit>")
        .map(str::to_string)
        .or_else(|| x_unit.clone().or_else(|| inherited_x.clone()));
    if let Some(unit) = y_unit {
        metadata.dims[y_axis].unit = Some(unit);
    }
    if let Some(z_axis) = z_axis {
        let z_unit = params
            .get("z_unit")
            .and_then(Value::as_str)
            .filter(|unit| !unit.is_empty() && *unit != "<same as x unit>")
            .map(str::to_string)
            .or_else(|| x_unit.or(inherited_x));
        if let Some(unit) = z_unit {
            metadata.dims[z_axis].unit = Some(unit);
        }
    }
}

fn apply_set_scale(metadata: &mut Metadata, dataset: &DatasetF32, params: &Value) -> Result<()> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let measured = get_optional_f32(params, "distance_pixels", 0.0);
    let known = get_optional_f32(params, "known_distance", 0.0);
    let aspect_ratio = get_optional_f32(params, "pixel_aspect_ratio", 1.0);
    if !aspect_ratio.is_finite() {
        return Err(OpsError::InvalidParams(
            "`pixel_aspect_ratio` must be finite".to_string(),
        ));
    }

    let unit = params
        .get("unit")
        .and_then(Value::as_str)
        .unwrap_or("pixel")
        .trim();
    let removes_scale = measured <= 0.0
        || known <= 0.0
        || unit.is_empty()
        || unit.to_lowercase().starts_with("pixel");

    if removes_scale {
        metadata.dims[x_axis].spacing = Some(1.0);
        metadata.dims[y_axis].spacing = Some(1.0);
        metadata.dims[x_axis].unit = Some("pixel".to_string());
        metadata.dims[y_axis].unit = Some("pixel".to_string());
        if let Some(z_axis) = dataset.axis_index(AxisKind::Z) {
            metadata.dims[z_axis].spacing = Some(1.0);
            metadata.dims[z_axis].unit = Some("pixel".to_string());
        }
        metadata
            .extras
            .insert("global_calibration".to_string(), json!(false));
        return Ok(());
    }

    let pixel_width = known / measured;
    let pixel_height = if aspect_ratio == 0.0 {
        pixel_width
    } else {
        pixel_width * aspect_ratio
    };
    metadata.dims[x_axis].spacing = Some(pixel_width);
    metadata.dims[y_axis].spacing = Some(pixel_height);
    metadata.dims[x_axis].unit = Some(unit.to_string());
    metadata.dims[y_axis].unit = Some(unit.to_string());
    if let Some(z_axis) = dataset.axis_index(AxisKind::Z) {
        if metadata.dims[z_axis].spacing.unwrap_or(1.0) == 1.0 {
            metadata.dims[z_axis].spacing = Some(pixel_width);
        }
        metadata.dims[z_axis].unit = Some(unit.to_string());
    }
    let global = params
        .get("global")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    metadata
        .extras
        .insert("global_calibration".to_string(), json!(global));
    Ok(())
}

fn apply_calibration_metadata(metadata: &mut Metadata, params: &Value) -> Result<()> {
    let function = params
        .get("function")
        .and_then(Value::as_str)
        .unwrap_or("none")
        .trim();
    let normalized_function = function.to_ascii_lowercase().replace([' ', '_'], "");
    if !normalized_function.is_empty() && normalized_function != "none" {
        return Err(OpsError::InvalidParams(format!(
            "unsupported calibration function `{function}`"
        )));
    }

    let unit = params
        .get("unit")
        .and_then(Value::as_str)
        .filter(|unit| !unit.trim().is_empty())
        .unwrap_or("Gray Value")
        .trim();
    let global = params
        .get("global")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    metadata
        .extras
        .insert("value_unit".to_string(), json!(unit));
    metadata
        .extras
        .insert("density_calibration_function".to_string(), json!("none"));
    metadata
        .extras
        .insert("global_density_calibration".to_string(), json!(global));

    Ok(())
}

fn shift_origin(metadata: &mut Metadata, axis: usize, label: &str, offset: usize) {
    if offset == 0 {
        return;
    }
    let Some(origin) = metadata
        .extras
        .get(&format!("{label}_origin_coordinate"))
        .and_then(Value::as_f64)
    else {
        return;
    };
    let spacing = metadata.dims[axis].spacing.unwrap_or(1.0) as f64;
    let inverted = metadata
        .extras
        .get(&format!("{label}_coordinate_inverted"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let direction = if inverted { -1.0 } else { 1.0 };
    metadata.extras.insert(
        format!("{label}_origin_coordinate"),
        json!(origin + direction * spacing * offset as f64),
    );
}

fn optional_f32_param(params: &Value, key: &str) -> Result<Option<f32>> {
    let Some(value) = params.get(key) else {
        return Ok(None);
    };
    let Some(value) = value.as_f64().map(|value| value as f32) else {
        return Err(OpsError::InvalidParams(format!("`{key}` must be a number")));
    };
    if !value.is_finite() {
        return Err(OpsError::InvalidParams(format!("`{key}` must be finite")));
    }
    Ok(Some(value))
}

#[derive(Debug, Clone, Copy)]
enum BinMethod {
    Average,
    Median,
    Min,
    Max,
    Sum,
}

impl BinMethod {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "average" | "mean" => Ok(Self::Average),
            "median" => Ok(Self::Median),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            "sum" => Ok(Self::Sum),
            other => Err(OpsError::InvalidParams(format!(
                "unsupported bin method `{other}`"
            ))),
        }
    }

    fn aggregate(self, values: &mut [f32]) -> f32 {
        match self {
            Self::Average => values.iter().sum::<f32>() / values.len() as f32,
            Self::Median => {
                values.sort_by(|left, right| left.total_cmp(right));
                values[values.len() / 2]
            }
            Self::Min => values.iter().copied().fold(f32::INFINITY, f32::min),
            Self::Max => values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            Self::Sum => values.iter().sum(),
        }
    }
}

fn bin_xyz(
    dataset: &DatasetF32,
    x_factor: usize,
    y_factor: usize,
    z_factor: usize,
    method: BinMethod,
) -> Result<DatasetF32> {
    if x_factor == 0 || y_factor == 0 || z_factor == 0 {
        return Err(OpsError::InvalidParams(
            "bin factors must be greater than zero".to_string(),
        ));
    }

    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let z_axis = dataset.axis_index(AxisKind::Z);
    if z_factor > 1 && z_axis.is_none() {
        return Err(OpsError::UnsupportedLayout(
            "Z binning requires a Z axis".to_string(),
        ));
    }

    let input_shape = dataset.shape().to_vec();
    let mut output_shape = input_shape.clone();
    output_shape[x_axis] /= x_factor;
    output_shape[y_axis] /= y_factor;
    if let Some(axis) = z_axis {
        output_shape[axis] /= z_factor;
    }
    if output_shape[x_axis] == 0
        || output_shape[y_axis] == 0
        || z_axis.is_some_and(|axis| output_shape[axis] == 0)
    {
        return Err(OpsError::InvalidParams(
            "bin factors must not exceed dimension sizes".to_string(),
        ));
    }

    let mut values = Vec::with_capacity(output_shape.iter().product());
    let mut block = Vec::with_capacity(x_factor * y_factor * z_factor);
    iterate_indices(&output_shape, |coord| {
        block.clear();
        for z_offset in 0..z_factor {
            for y_offset in 0..y_factor {
                for x_offset in 0..x_factor {
                    let mut source = coord.to_vec();
                    source[x_axis] = coord[x_axis] * x_factor + x_offset;
                    source[y_axis] = coord[y_axis] * y_factor + y_offset;
                    if let Some(axis) = z_axis {
                        source[axis] = coord[axis] * z_factor + z_offset;
                    }
                    block.push(dataset.data[IxDyn(&source)]);
                }
            }
        }
        values.push(method.aggregate(&mut block));
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), values)
        .map_err(|_| OpsError::UnsupportedLayout("failed to build binned dataset".to_string()))?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims[x_axis].size = output_shape[x_axis];
    metadata.dims[y_axis].size = output_shape[y_axis];
    if let Some(spacing) = &mut metadata.dims[x_axis].spacing {
        *spacing *= x_factor as f32;
    }
    if let Some(spacing) = &mut metadata.dims[y_axis].spacing {
        *spacing *= y_factor as f32;
    }
    if let Some(axis) = z_axis {
        metadata.dims[axis].size = output_shape[axis];
        if let Some(spacing) = &mut metadata.dims[axis].spacing {
            *spacing *= z_factor as f32;
        }
    }
    Ok(Dataset::new(data, metadata)?)
}

fn flip_xy(dataset: &DatasetF32, axis: AxisKind) -> Result<DatasetF32> {
    let target_axis = axis_index(dataset, axis)?;
    let shape = dataset.shape().to_vec();
    let last = shape[target_axis] - 1;
    let mut output = Vec::with_capacity(dataset.data.len());
    iterate_indices(&shape, |coord| {
        let mut source = coord.to_vec();
        source[target_axis] = last - coord[target_axis];
        output.push(dataset.data[IxDyn(&source)]);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&shape), output)
        .map_err(|_| OpsError::UnsupportedLayout("failed to build flipped dataset".to_string()))?;
    Ok(Dataset::new(data, dataset.metadata.clone())?)
}

#[derive(Debug, Clone, Copy)]
enum RotateDirection {
    Left,
    Right,
}

fn rotate_90_xy(dataset: &DatasetF32, direction: RotateDirection) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let input_shape = dataset.shape().to_vec();
    let src_w = input_shape[x_axis];
    let src_h = input_shape[y_axis];
    let mut output_shape = input_shape.clone();
    output_shape[x_axis] = src_h;
    output_shape[y_axis] = src_w;

    let mut output = Vec::with_capacity(dataset.data.len());
    iterate_indices(&output_shape, |coord| {
        let mut source = coord.to_vec();
        match direction {
            RotateDirection::Right => {
                source[x_axis] = coord[y_axis];
                source[y_axis] = src_h - 1 - coord[x_axis];
            }
            RotateDirection::Left => {
                source[x_axis] = src_w - 1 - coord[y_axis];
                source[y_axis] = coord[x_axis];
            }
        }
        output.push(dataset.data[IxDyn(&source)]);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output)
        .map_err(|_| OpsError::UnsupportedLayout("failed to build rotated dataset".to_string()))?;
    let mut metadata = dataset.metadata.clone();
    let x_spacing = metadata.dims[x_axis].spacing;
    let y_spacing = metadata.dims[y_axis].spacing;
    let x_unit = metadata.dims[x_axis].unit.clone();
    let y_unit = metadata.dims[y_axis].unit.clone();
    metadata.dims[x_axis].size = output_shape[x_axis];
    metadata.dims[y_axis].size = output_shape[y_axis];
    metadata.dims[x_axis].spacing = y_spacing;
    metadata.dims[y_axis].spacing = x_spacing;
    metadata.dims[x_axis].unit = y_unit;
    metadata.dims[y_axis].unit = x_unit;
    Ok(Dataset::new(data, metadata)?)
}

fn rotate_xy(
    dataset: &DatasetF32,
    angle_degrees: f32,
    fill: f32,
    interpolation: TranslateInterpolation,
    enlarge: bool,
) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let input_shape = dataset.shape().to_vec();
    let src_w = input_shape[x_axis];
    let src_h = input_shape[y_axis];
    let radians = angle_degrees.to_radians();
    let cos = radians.cos();
    let sin = radians.sin();
    let mut output_shape = input_shape.clone();
    if enlarge {
        output_shape[x_axis] = ceil_dimension(src_w as f32 * cos.abs() + src_h as f32 * sin.abs());
        output_shape[y_axis] = ceil_dimension(src_w as f32 * sin.abs() + src_h as f32 * cos.abs());
    }

    let src_cx = (src_w as f32 - 1.0) / 2.0;
    let src_cy = (src_h as f32 - 1.0) / 2.0;
    let dst_cx = (output_shape[x_axis] as f32 - 1.0) / 2.0;
    let dst_cy = (output_shape[y_axis] as f32 - 1.0) / 2.0;
    let mut output = Vec::with_capacity(output_shape.iter().product());

    iterate_indices(&output_shape, |coord| {
        let dx = coord[x_axis] as f32 - dst_cx;
        let dy = coord[y_axis] as f32 - dst_cy;
        let src_x = src_cx + dx * cos + dy * sin;
        let src_y = src_cy - dx * sin + dy * cos;
        let value = match interpolation {
            TranslateInterpolation::Nearest => sample_or_fill(
                dataset,
                coord,
                x_axis,
                y_axis,
                src_x.round(),
                src_y.round(),
                fill,
            ),
            TranslateInterpolation::Bilinear => {
                bilinear_sample_or_fill(dataset, coord, x_axis, y_axis, src_x, src_y, fill)
            }
        };
        output.push(value);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output)
        .map_err(|_| OpsError::UnsupportedLayout("failed to build rotated dataset".to_string()))?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims[x_axis].size = output_shape[x_axis];
    metadata.dims[y_axis].size = output_shape[y_axis];
    Ok(Dataset::new(data, metadata)?)
}

fn ceil_dimension(value: f32) -> usize {
    let rounded = value.round();
    let adjusted = if (value - rounded).abs() < 1.0e-4 {
        rounded
    } else {
        value.ceil()
    };
    adjusted.max(1.0) as usize
}

#[derive(Debug, Clone, Copy)]
enum TranslateInterpolation {
    Nearest,
    Bilinear,
}

impl TranslateInterpolation {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "nearest" | "none" => Ok(Self::Nearest),
            "bilinear" => Ok(Self::Bilinear),
            other => Err(OpsError::InvalidParams(format!(
                "unsupported interpolation `{other}`"
            ))),
        }
    }
}

fn translate_xy(
    dataset: &DatasetF32,
    x_offset: f32,
    y_offset: f32,
    fill: f32,
    interpolation: TranslateInterpolation,
) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let mut output = Vec::with_capacity(dataset.data.len());

    iterate_indices(&shape, |coord| {
        let src_x = coord[x_axis] as f32 - x_offset;
        let src_y = coord[y_axis] as f32 - y_offset;
        let value = match interpolation {
            TranslateInterpolation::Nearest => sample_or_fill(
                dataset,
                coord,
                x_axis,
                y_axis,
                src_x.round(),
                src_y.round(),
                fill,
            ),
            TranslateInterpolation::Bilinear => {
                bilinear_sample_or_fill(dataset, coord, x_axis, y_axis, src_x, src_y, fill)
            }
        };
        output.push(value);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build translated dataset".to_string())
    })?;
    Ok(Dataset::new(data, dataset.metadata.clone())?)
}

fn median_filter_xy(dataset: &DatasetF32, radius: f32) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let line_radii = median_line_radii(radius);
    let kernel_radius = line_radii.len() / 2;
    let kernel_points = line_radii
        .iter()
        .map(|(left, right)| (right - left + 1) as usize)
        .sum::<usize>();
    let mut window = Vec::with_capacity(kernel_points);
    let mut output = Vec::with_capacity(dataset.data.len());

    iterate_indices(&shape, |coord| {
        let x = coord[x_axis] as isize;
        let y = coord[y_axis] as isize;
        window.clear();
        for (line, (left, right)) in line_radii.iter().enumerate() {
            let dy = line as isize - kernel_radius as isize;
            for dx in *left..=*right {
                window.push(sample_clamped(
                    dataset,
                    coord,
                    x_axis,
                    y_axis,
                    x + dx,
                    y + dy,
                ));
            }
        }
        window.sort_by(|left, right| left.total_cmp(right));
        output.push(window[window.len() / 2]);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build median-filtered dataset".to_string())
    })?;
    Ok(Dataset::new(data, dataset.metadata.clone())?)
}

fn remove_nans_xy(dataset: &DatasetF32, radius: f32) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let line_radii = median_line_radii(radius);
    let kernel_radius = line_radii.len() / 2;
    let kernel_points = line_radii
        .iter()
        .map(|(left, right)| (right - left + 1) as usize)
        .sum::<usize>();
    let mut window = Vec::with_capacity(kernel_points);
    let mut output = Vec::with_capacity(dataset.data.len());

    iterate_indices(&shape, |coord| {
        let value = dataset.data[IxDyn(coord)];
        if !value.is_nan() {
            output.push(value);
            return;
        }

        let x = coord[x_axis] as isize;
        let y = coord[y_axis] as isize;
        window.clear();
        for (line, (left, right)) in line_radii.iter().enumerate() {
            let dy = line as isize - kernel_radius as isize;
            for dx in *left..=*right {
                let value = sample_clamped(dataset, coord, x_axis, y_axis, x + dx, y + dy);
                if !value.is_nan() {
                    window.push(value);
                }
            }
        }
        if window.is_empty() {
            output.push(f32::NAN);
        } else {
            window.sort_by(|left, right| left.total_cmp(right));
            output.push(window[window.len() / 2]);
        }
    });

    let data = ArrayD::from_shape_vec(IxDyn(&shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build NaN-removed dataset".to_string())
    })?;
    Ok(Dataset::new(data, dataset.metadata.clone())?)
}

#[derive(Debug, Clone, Copy)]
enum OutlierKind {
    Bright,
    Dark,
}

impl OutlierKind {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "bright" | "Bright" => Ok(Self::Bright),
            "dark" | "Dark" => Ok(Self::Dark),
            other => Err(OpsError::InvalidParams(format!(
                "unsupported outlier kind `{other}`"
            ))),
        }
    }
}

fn remove_outliers_xy(
    dataset: &DatasetF32,
    radius: f32,
    threshold: f32,
    which: OutlierKind,
) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let line_radii = median_line_radii(radius);
    let kernel_radius = line_radii.len() / 2;
    let kernel_points = line_radii
        .iter()
        .map(|(left, right)| (right - left + 1) as usize)
        .sum::<usize>();
    let sample_scale = outlier_sample_scale(dataset.metadata.pixel_type);
    let mut window = Vec::with_capacity(kernel_points);
    let mut output = Vec::with_capacity(dataset.data.len());

    iterate_indices(&shape, |coord| {
        let value = dataset.data[IxDyn(coord)];
        let x = coord[x_axis] as isize;
        let y = coord[y_axis] as isize;
        window.clear();
        for (line, (left, right)) in line_radii.iter().enumerate() {
            let dy = line as isize - kernel_radius as isize;
            for dx in *left..=*right {
                window.push(sample_clamped(
                    dataset,
                    coord,
                    x_axis,
                    y_axis,
                    x + dx,
                    y + dy,
                ));
            }
        }
        window.sort_by(|left, right| left.total_cmp(right));
        let median = window[window.len() / 2];
        let replacement = match which {
            OutlierKind::Bright => {
                let local_min = window.iter().copied().fold(f32::INFINITY, f32::min);
                is_bright_outlier(value, median, local_min, threshold, sample_scale)
            }
            OutlierKind::Dark => {
                let local_max = window.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                is_dark_outlier(value, median, local_max, threshold, sample_scale)
            }
        };
        output.push(if replacement { median } else { value });
    });

    let data = ArrayD::from_shape_vec(IxDyn(&shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build outlier-removed dataset".to_string())
    })?;
    Ok(Dataset::new(data, dataset.metadata.clone())?)
}

fn is_bright_outlier(
    value: f32,
    median: f32,
    local_min: f32,
    threshold: f32,
    sample_scale: f32,
) -> bool {
    let value = value * sample_scale;
    let median = median * sample_scale;
    let local_min = local_min * sample_scale;
    value - threshold > local_min && value - threshold > median
}

fn is_dark_outlier(
    value: f32,
    median: f32,
    local_max: f32,
    threshold: f32,
    sample_scale: f32,
) -> bool {
    let value = value * sample_scale;
    let median = median * sample_scale;
    let local_max = local_max * sample_scale;
    value + threshold < local_max && value + threshold < median
}

fn outlier_sample_scale(pixel_type: PixelType) -> f32 {
    match pixel_type {
        PixelType::U8 => 255.0,
        PixelType::U16 => 65_535.0,
        PixelType::F32 => 1.0,
    }
}

fn median_line_radii(radius: f32) -> Vec<(isize, isize)> {
    let radius = if (1.5..1.75).contains(&radius) {
        1.75
    } else if (2.5..2.85).contains(&radius) {
        2.85
    } else {
        radius
    };
    let r2 = (radius * radius) as isize + 1;
    let kernel_radius = ((r2 as f32 + 1.0e-10).sqrt()) as isize;
    let mut line_radii = Vec::with_capacity(2 * kernel_radius as usize + 1);
    for y in -kernel_radius..=kernel_radius {
        let dx = ((r2 - y * y) as f32 + 1.0e-10).sqrt() as isize;
        line_radii.push((-dx, dx));
    }
    line_radii
}

#[derive(Debug, Clone, Copy)]
enum FilterKind {
    Sharpen,
    Sobel,
}

#[derive(Debug, Clone, Copy)]
enum RankFilterKind {
    Mean,
    Minimum,
    Maximum,
    Variance,
    TopHat,
}

impl RankFilterKind {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "mean" | "average" => Ok(Self::Mean),
            "minimum" | "min" => Ok(Self::Minimum),
            "maximum" | "max" => Ok(Self::Maximum),
            "variance" | "var" => Ok(Self::Variance),
            "top_hat" | "tophat" => Ok(Self::TopHat),
            other => Err(OpsError::InvalidParams(format!(
                "unsupported rank filter `{other}`"
            ))),
        }
    }

    fn aggregate(self, values: &[f32]) -> f32 {
        match self {
            Self::Mean => values.iter().sum::<f32>() / values.len() as f32,
            Self::Minimum => values.iter().copied().fold(f32::INFINITY, f32::min),
            Self::Maximum => values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            Self::Variance => {
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                values
                    .iter()
                    .map(|value| {
                        let delta = *value - mean;
                        delta * delta
                    })
                    .sum::<f32>()
                    / values.len() as f32
            }
            Self::TopHat => unreachable!("top-hat uses a multi-pass rank filter"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum RankFilter3dKind {
    Mean,
    Median,
    Minimum,
    Maximum,
    Variance,
}

impl RankFilter3dKind {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "mean" | "average" => Ok(Self::Mean),
            "median" => Ok(Self::Median),
            "minimum" | "min" => Ok(Self::Minimum),
            "maximum" | "max" => Ok(Self::Maximum),
            "variance" | "var" => Ok(Self::Variance),
            other => Err(OpsError::InvalidParams(format!(
                "unsupported 3D rank filter `{other}`"
            ))),
        }
    }

    fn aggregate(self, values: &mut [f32]) -> f32 {
        match self {
            Self::Mean => values.iter().sum::<f32>() / values.len() as f32,
            Self::Median => {
                values.sort_by(|a, b| a.total_cmp(b));
                values[values.len() / 2]
            }
            Self::Minimum => values.iter().copied().fold(f32::INFINITY, f32::min),
            Self::Maximum => values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            Self::Variance => {
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                values
                    .iter()
                    .map(|value| {
                        let delta = *value - mean;
                        delta * delta
                    })
                    .sum::<f32>()
                    / values.len() as f32
            }
        }
    }
}

fn rank_filter_xy(
    dataset: &DatasetF32,
    filter: RankFilterKind,
    radius: f32,
    light_background: bool,
    dont_subtract: bool,
) -> Result<DatasetF32> {
    if radius <= f32::EPSILON {
        return Ok(dataset.clone());
    }
    if matches!(filter, RankFilterKind::TopHat) {
        return top_hat_xy(dataset, radius, light_background, dont_subtract);
    }
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let line_radii = median_line_radii(radius);
    let kernel_radius = line_radii.len() / 2;
    let kernel_points = line_radii
        .iter()
        .map(|(left, right)| (right - left + 1) as usize)
        .sum::<usize>();
    let mut window = Vec::with_capacity(kernel_points);
    let mut output = Vec::with_capacity(dataset.data.len());

    iterate_indices(&shape, |coord| {
        let x = coord[x_axis] as isize;
        let y = coord[y_axis] as isize;
        window.clear();
        for (line, (left, right)) in line_radii.iter().enumerate() {
            let dy = line as isize - kernel_radius as isize;
            for dx in *left..=*right {
                window.push(sample_clamped(
                    dataset,
                    coord,
                    x_axis,
                    y_axis,
                    x + dx,
                    y + dy,
                ));
            }
        }
        output.push(filter.aggregate(&window));
    });

    let data = ArrayD::from_shape_vec(IxDyn(&shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build rank-filtered dataset".to_string())
    })?;
    Ok(Dataset::new(data, dataset.metadata.clone())?)
}

fn rank_filter_3d(
    dataset: &DatasetF32,
    filter: RankFilter3dKind,
    x_radius: f32,
    y_radius: f32,
    z_radius: f32,
) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let z_axis = axis_index(dataset, AxisKind::Z)?;
    let shape = dataset.shape().to_vec();
    let offsets = ellipsoid_offsets(x_radius, y_radius, z_radius);
    let mut window = Vec::with_capacity(offsets.len());
    let mut output = Vec::with_capacity(dataset.data.len());

    iterate_indices(&shape, |coord| {
        window.clear();
        let x = coord[x_axis] as isize;
        let y = coord[y_axis] as isize;
        let z = coord[z_axis] as isize;
        for (dx, dy, dz) in &offsets {
            let sample_x = x + *dx;
            let sample_y = y + *dy;
            let sample_z = z + *dz;
            if sample_x < 0
                || sample_y < 0
                || sample_z < 0
                || sample_x >= shape[x_axis] as isize
                || sample_y >= shape[y_axis] as isize
                || sample_z >= shape[z_axis] as isize
            {
                continue;
            }
            let mut sample_coord = coord.to_vec();
            sample_coord[x_axis] = sample_x as usize;
            sample_coord[y_axis] = sample_y as usize;
            sample_coord[z_axis] = sample_z as usize;
            window.push(dataset.data[IxDyn(&sample_coord)]);
        }
        output.push(filter.aggregate(&mut window));
    });

    let data = ArrayD::from_shape_vec(IxDyn(&shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build 3D rank-filtered dataset".to_string())
    })?;
    Ok(Dataset::new(data, dataset.metadata.clone())?)
}

fn ellipsoid_offsets(x_radius: f32, y_radius: f32, z_radius: f32) -> Vec<(isize, isize, isize)> {
    let x_extent = x_radius.ceil() as isize;
    let y_extent = y_radius.ceil() as isize;
    let z_extent = z_radius.ceil() as isize;
    let x_scale = if x_radius > 0.0 {
        1.0 / (x_radius * x_radius)
    } else {
        0.0
    };
    let y_scale = if y_radius > 0.0 {
        1.0 / (y_radius * y_radius)
    } else {
        0.0
    };
    let z_scale = if z_radius > 0.0 {
        1.0 / (z_radius * z_radius)
    } else {
        0.0
    };
    let mut offsets = Vec::new();

    for dz in -z_extent..=z_extent {
        for dy in -y_extent..=y_extent {
            for dx in -x_extent..=x_extent {
                let distance = (dx * dx) as f32 * x_scale
                    + (dy * dy) as f32 * y_scale
                    + (dz * dz) as f32 * z_scale;
                if distance <= 1.0 {
                    offsets.push((dx, dy, dz));
                }
            }
        }
    }
    offsets
}

fn swap_quadrants_xy(dataset: &DatasetF32) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let width = shape[x_axis];
    let height = shape[y_axis];
    if width != height {
        return Err(OpsError::InvalidParams(
            "Swap Quadrants requires square X/Y dimensions".to_string(),
        ));
    }

    let quadrant = width / 2;
    let mut output = dataset.data.clone();
    iterate_indices(&shape, |coord| {
        let x = coord[x_axis];
        let y = coord[y_axis];
        let mut source = coord.to_vec();
        if x < quadrant && y < quadrant {
            source[x_axis] = x + quadrant;
            source[y_axis] = y + quadrant;
        } else if x >= quadrant && x < quadrant * 2 && y < quadrant {
            source[x_axis] = x - quadrant;
            source[y_axis] = y + quadrant;
        } else if x < quadrant && y >= quadrant && y < quadrant * 2 {
            source[x_axis] = x + quadrant;
            source[y_axis] = y - quadrant;
        } else if x >= quadrant && x < quadrant * 2 && y >= quadrant && y < quadrant * 2 {
            source[x_axis] = x - quadrant;
            source[y_axis] = y - quadrant;
        }
        output[IxDyn(coord)] = dataset.data[IxDyn(&source)];
    });

    Ok(Dataset::new(output, dataset.metadata.clone())?)
}

fn fft_power_spectrum_xy(dataset: &DatasetF32) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let width = shape[x_axis];
    let height = shape[y_axis];
    if width == 0 || height == 0 {
        return Err(OpsError::UnsupportedLayout(
            "FFT requires non-empty X/Y dimensions".to_string(),
        ));
    }

    let plane_shape = shape
        .iter()
        .enumerate()
        .filter_map(|(axis, size)| (axis != x_axis && axis != y_axis).then_some(*size))
        .collect::<Vec<_>>();
    let mut output = vec![0.0_f32; dataset.data.len()];
    let mut planner = FftPlanner::<f32>::new();
    let row_fft = planner.plan_fft_forward(width);
    let column_fft = planner.plan_fft_forward(height);

    iterate_indices(&plane_shape, |plane_coord| {
        let mut base_coord = vec![0usize; shape.len()];
        let mut plane_at = 0usize;
        for axis in 0..shape.len() {
            if axis != x_axis && axis != y_axis {
                base_coord[axis] = plane_coord[plane_at];
                plane_at += 1;
            }
        }

        let mut complex = vec![Complex::new(0.0_f32, 0.0_f32); width * height];
        for y in 0..height {
            for x in 0..width {
                let mut coord = base_coord.clone();
                coord[x_axis] = x;
                coord[y_axis] = y;
                complex[y * width + x].re = dataset.data[IxDyn(&coord)];
            }
        }

        for y in 0..height {
            row_fft.process(&mut complex[y * width..(y + 1) * width]);
        }
        let mut column = vec![Complex::new(0.0_f32, 0.0_f32); height];
        for x in 0..width {
            for y in 0..height {
                column[y] = complex[y * width + x];
            }
            column_fft.process(&mut column);
            for y in 0..height {
                complex[y * width + x] = column[y];
            }
        }

        let powers = complex
            .iter()
            .map(|value| value.norm_sqr().max(f32::MIN_POSITIVE))
            .collect::<Vec<_>>();
        let (mut min_log, max_log) = powers.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), value| {
                let value = value.ln();
                (minimum.min(value), maximum.max(value))
            },
        );
        if !min_log.is_finite() || max_log - min_log > 50.0 {
            min_log = max_log - 50.0;
        }
        let scale = if (max_log - min_log).abs() <= f32::EPSILON {
            0.0
        } else {
            253.999 / (max_log - min_log)
        };

        for y in 0..height {
            for x in 0..width {
                let centered_x = (x + width / 2) % width;
                let centered_y = (y + height / 2) % height;
                let scaled = ((powers[y * width + x].ln() - min_log) * scale).max(0.0);
                let byte = (scaled + 1.0).clamp(1.0, 254.0);
                let mut coord = base_coord.clone();
                coord[x_axis] = centered_x;
                coord[y_axis] = centered_y;
                let offset = linear_offset(&shape, &coord);
                output[offset] = byte / 255.0;
            }
        }
    });

    let data = ArrayD::from_shape_vec(IxDyn(&shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build FFT power spectrum dataset".to_string())
    })?;
    let mut metadata = dataset.metadata.clone();
    metadata.pixel_type = PixelType::U8;
    Ok(Dataset::new(data, metadata)?)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StripeSuppression {
    None,
    Horizontal,
    Vertical,
}

impl StripeSuppression {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "none" | "" => Ok(Self::None),
            "horizontal" => Ok(Self::Horizontal),
            "vertical" => Ok(Self::Vertical),
            other => Err(OpsError::InvalidParams(format!(
                "unsupported stripe suppression `{other}`"
            ))),
        }
    }
}

fn fft_bandpass_xy(
    dataset: &DatasetF32,
    filter_large: f32,
    filter_small: f32,
    stripes: StripeSuppression,
    tolerance: f32,
    autoscale: bool,
) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let width = shape[x_axis];
    let height = shape[y_axis];
    if width == 0 || height == 0 {
        return Err(OpsError::UnsupportedLayout(
            "FFT bandpass requires non-empty X/Y dimensions".to_string(),
        ));
    }

    let plane_shape = shape
        .iter()
        .enumerate()
        .filter_map(|(axis, size)| (axis != x_axis && axis != y_axis).then_some(*size))
        .collect::<Vec<_>>();
    let mut output = vec![0.0_f32; dataset.data.len()];
    let mut planner = FftPlanner::<f32>::new();
    let row_forward = planner.plan_fft_forward(width);
    let column_forward = planner.plan_fft_forward(height);
    let row_inverse = planner.plan_fft_inverse(width);
    let column_inverse = planner.plan_fft_inverse(height);
    let max_dimension = width.max(height) as f32;
    let large_scale = (2.0 * filter_large / max_dimension).powi(2);
    let small_scale = (2.0 * filter_small / max_dimension).powi(2);
    let stripe_scale = ((100.0 - tolerance).max(0.0) / 100.0).powi(2);

    iterate_indices(&plane_shape, |plane_coord| {
        let mut base_coord = vec![0usize; shape.len()];
        let mut plane_at = 0usize;
        for axis in 0..shape.len() {
            if axis != x_axis && axis != y_axis {
                base_coord[axis] = plane_coord[plane_at];
                plane_at += 1;
            }
        }

        let mut input_values = Vec::with_capacity(width * height);
        let mut complex = vec![Complex::new(0.0_f32, 0.0_f32); width * height];
        for y in 0..height {
            for x in 0..width {
                let mut coord = base_coord.clone();
                coord[x_axis] = x;
                coord[y_axis] = y;
                let value = dataset.data[IxDyn(&coord)];
                input_values.push(value);
                complex[y * width + x].re = value;
            }
        }

        for y in 0..height {
            row_forward.process(&mut complex[y * width..(y + 1) * width]);
        }
        let mut column = vec![Complex::new(0.0_f32, 0.0_f32); height];
        for x in 0..width {
            for y in 0..height {
                column[y] = complex[y * width + x];
            }
            column_forward.process(&mut column);
            for y in 0..height {
                complex[y * width + x] = column[y];
            }
        }

        for y in 0..height {
            let ky = y.min(height - y) as f32;
            for x in 0..width {
                let kx = x.min(width - x) as f32;
                let radius2 = kx * kx + ky * ky;
                let high_pass = 1.0 - (-radius2 * large_scale).exp();
                let low_pass = (-radius2 * small_scale).exp();
                let mut factor = high_pass * low_pass;
                match stripes {
                    StripeSuppression::None => {}
                    StripeSuppression::Horizontal => {
                        factor *= 1.0 - (-(kx * kx) * stripe_scale).exp();
                    }
                    StripeSuppression::Vertical => {
                        factor *= 1.0 - (-(ky * ky) * stripe_scale).exp();
                    }
                }
                complex[y * width + x] *= factor;
            }
        }

        for x in 0..width {
            for y in 0..height {
                column[y] = complex[y * width + x];
            }
            column_inverse.process(&mut column);
            for y in 0..height {
                complex[y * width + x] = column[y];
            }
        }
        for y in 0..height {
            row_inverse.process(&mut complex[y * width..(y + 1) * width]);
        }

        let normalization = (width * height) as f32;
        let mut filtered = complex
            .iter()
            .map(|value| value.re / normalization)
            .collect::<Vec<_>>();
        if autoscale {
            stretch_to_input_range(&input_values, &mut filtered);
        }

        for y in 0..height {
            for x in 0..width {
                let mut coord = base_coord.clone();
                coord[x_axis] = x;
                coord[y_axis] = y;
                let offset = linear_offset(&shape, &coord);
                output[offset] = filtered[y * width + x];
            }
        }
    });

    let data = ArrayD::from_shape_vec(IxDyn(&shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build FFT bandpass dataset".to_string())
    })?;
    Ok(Dataset::new(data, dataset.metadata.clone())?)
}

fn stretch_to_input_range(input: &[f32], output: &mut [f32]) {
    let (input_min, input_max) = input
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
            (min.min(*value), max.max(*value))
        });
    let (output_min, output_max) = output
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
            (min.min(*value), max.max(*value))
        });
    let input_span = input_max - input_min;
    let output_span = output_max - output_min;
    if input_span <= f32::EPSILON || output_span <= f32::EPSILON {
        return;
    }
    for value in output.iter_mut() {
        *value = ((*value - output_min) / output_span) * input_span + input_min;
    }
}

fn top_hat_xy(
    dataset: &DatasetF32,
    radius: f32,
    light_background: bool,
    dont_subtract: bool,
) -> Result<DatasetF32> {
    let first = if light_background {
        RankFilterKind::Maximum
    } else {
        RankFilterKind::Minimum
    };
    let second = if light_background {
        RankFilterKind::Minimum
    } else {
        RankFilterKind::Maximum
    };
    let first_pass = rank_filter_xy(dataset, first, radius, false, false)?;
    let background = rank_filter_xy(&first_pass, second, radius, false, false)?;
    if dont_subtract {
        return Ok(background);
    }

    let offset = if light_background && dataset.metadata.pixel_type != PixelType::F32 {
        1.0
    } else {
        0.0
    };
    let output = dataset
        .data
        .iter()
        .zip(background.data.iter())
        .map(|(original, background)| *original - *background + offset)
        .collect::<Vec<_>>();
    let data = ArrayD::from_shape_vec(IxDyn(dataset.shape()), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build top-hat filtered dataset".to_string())
    })?;
    Ok(Dataset::new(data, dataset.metadata.clone())?)
}

fn unsharp_mask_xy(dataset: &DatasetF32, sigma: f32, weight: f32) -> Result<DatasetF32> {
    if sigma <= f32::EPSILON || weight <= f32::EPSILON {
        return Ok(dataset.clone());
    }
    let blurred = gaussian_blur_xy_data(dataset, sigma)?;
    let scale = 1.0 / (1.0 - weight);
    let output = dataset
        .data
        .iter()
        .zip(blurred.iter())
        .map(|(original, blurred)| (*original - weight * *blurred) * scale)
        .collect::<Vec<_>>();
    let data = ArrayD::from_shape_vec(IxDyn(dataset.shape()), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build unsharp-masked dataset".to_string())
    })?;
    Ok(Dataset::new(data, dataset.metadata.clone())?)
}

fn gaussian_blur_xy_data(dataset: &DatasetF32, sigma: f32) -> Result<ArrayD<f32>> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let kernel = gaussian_kernel(sigma);
    let radius = kernel.len() as isize / 2;
    let mut current = dataset.data.clone();

    for axis in [x_axis, y_axis] {
        if shape[axis] <= 1 {
            continue;
        }
        let source = current;
        let mut output = Vec::with_capacity(source.len());
        iterate_indices(&shape, |coord| {
            let mut sum = 0.0_f32;
            for (kernel_index, weight) in kernel.iter().enumerate() {
                let offset = kernel_index as isize - radius;
                let mut source_coord = coord.to_vec();
                source_coord[axis] =
                    (coord[axis] as isize + offset).clamp(0, shape[axis] as isize - 1) as usize;
                sum += source[IxDyn(&source_coord)] * *weight;
            }
            output.push(sum);
        });
        current = ArrayD::from_shape_vec(IxDyn(&shape), output).map_err(|_| {
            OpsError::UnsupportedLayout("failed to build blurred dataset".to_string())
        })?;
    }

    Ok(current)
}

#[derive(Debug, Clone, Copy)]
enum ShadowDirection {
    North,
    Northeast,
    East,
    Southeast,
    South,
    Southwest,
    West,
    Northwest,
}

impl ShadowDirection {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "north" => Ok(Self::North),
            "northeast" | "north_east" => Ok(Self::Northeast),
            "east" => Ok(Self::East),
            "southeast" | "south_east" => Ok(Self::Southeast),
            "south" => Ok(Self::South),
            "southwest" | "south_west" => Ok(Self::Southwest),
            "west" => Ok(Self::West),
            "northwest" | "north_west" => Ok(Self::Northwest),
            other => Err(OpsError::InvalidParams(format!(
                "unsupported shadow direction `{other}`"
            ))),
        }
    }

    fn kernel(self) -> [f32; 9] {
        match self {
            Self::North => [1.0, 2.0, 1.0, 0.0, 1.0, 0.0, -1.0, -2.0, -1.0],
            Self::Northeast => [0.0, 1.0, 2.0, -1.0, 1.0, 1.0, -2.0, -1.0, 0.0],
            Self::East => [-1.0, 0.0, 1.0, -2.0, 1.0, 2.0, -1.0, 0.0, 1.0],
            Self::Southeast => [-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0],
            Self::South => [-1.0, -2.0, -1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0],
            Self::Southwest => [0.0, -1.0, -2.0, 1.0, 1.0, -1.0, 2.0, 1.0, 0.0],
            Self::West => [1.0, 0.0, -1.0, 2.0, 1.0, -2.0, 1.0, 0.0, -1.0],
            Self::Northwest => [2.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, -1.0, -2.0],
        }
    }
}

fn shadow_xy(dataset: &DatasetF32, direction: ShadowDirection) -> Result<DatasetF32> {
    convolve3x3_xy(dataset, direction.kernel())
}

fn shadow_demo_stack_xy(dataset: &DatasetF32, iterations: usize) -> Result<DatasetF32> {
    if dataset.axis_index(AxisKind::Z).is_some() {
        return Err(OpsError::UnsupportedLayout(
            "Shadows Demo does not work with stacks".to_string(),
        ));
    }
    let directions = [
        ShadowDirection::North,
        ShadowDirection::Northeast,
        ShadowDirection::East,
        ShadowDirection::Southeast,
        ShadowDirection::South,
        ShadowDirection::Southwest,
        ShadowDirection::West,
        ShadowDirection::Northwest,
    ];
    let frames = iterations * directions.len();
    let shadowed = directions
        .iter()
        .copied()
        .map(|direction| shadow_xy(dataset, direction))
        .collect::<Result<Vec<_>>>()?;
    let input_shape = dataset.shape().to_vec();
    let mut output_shape = input_shape.clone();
    output_shape.push(frames);
    let frame_axis = output_shape.len() - 1;
    let mut output = Vec::with_capacity(output_shape.iter().product());

    iterate_indices(&output_shape, |coord| {
        let frame = coord[frame_axis];
        let source_coord = &coord[..frame_axis];
        output.push(shadowed[frame % directions.len()].data[IxDyn(source_coord)]);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&output_shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build Shadows Demo stack".to_string())
    })?;
    let mut metadata = dataset.metadata.clone();
    metadata.dims.push(Dim::new(AxisKind::Z, frames));
    Ok(Dataset::new(data, metadata)?)
}

fn parse_convolution_kernel(params: &Value) -> Result<Vec<f32>> {
    let Some(values) = params.get("kernel").and_then(Value::as_array) else {
        return Err(OpsError::InvalidParams("`kernel` is required".to_string()));
    };
    if values.is_empty() {
        return Err(OpsError::InvalidParams(
            "`kernel` must not be empty".to_string(),
        ));
    }

    let mut kernel = Vec::with_capacity(values.len());
    for value in values {
        let Some(value) = value.as_f64() else {
            return Err(OpsError::InvalidParams(
                "`kernel` values must be numeric".to_string(),
            ));
        };
        let value = value as f32;
        if !value.is_finite() {
            return Err(OpsError::InvalidParams(
                "`kernel` values must be finite".to_string(),
            ));
        }
        kernel.push(value);
    }
    Ok(kernel)
}

fn convolve_xy(
    dataset: &DatasetF32,
    kernel: &[f32],
    width: usize,
    height: usize,
    normalize: bool,
) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let x_radius = width as isize / 2;
    let y_radius = height as isize / 2;
    let scale = if normalize {
        let sum = kernel.iter().sum::<f32>();
        if sum.abs() <= f32::EPSILON {
            1.0
        } else {
            1.0 / sum
        }
    } else {
        1.0
    };
    let mut output = Vec::with_capacity(dataset.data.len());

    iterate_indices(&shape, |coord| {
        let x = coord[x_axis] as isize;
        let y = coord[y_axis] as isize;
        let mut value = 0.0_f32;
        let mut kernel_at = 0usize;
        for ky in 0..height {
            let dy = ky as isize - y_radius;
            for kx in 0..width {
                let dx = kx as isize - x_radius;
                value += kernel[kernel_at]
                    * sample_clamped(dataset, coord, x_axis, y_axis, x + dx, y + dy);
                kernel_at += 1;
            }
        }
        output.push(value * scale);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build convolved dataset".to_string())
    })?;
    Ok(Dataset::new(data, dataset.metadata.clone())?)
}

fn convolve3x3_xy(dataset: &DatasetF32, kernel: [f32; 9]) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let scale = {
        let sum = kernel.iter().sum::<f32>();
        if sum == 0.0 { 1.0 } else { 1.0 / sum }
    };
    let mut output = Vec::with_capacity(dataset.data.len());

    iterate_indices(&shape, |coord| {
        let x = coord[x_axis] as isize;
        let y = coord[y_axis] as isize;
        let mut value = 0.0_f32;
        let mut kernel_at = 0usize;
        for dy in -1..=1 {
            for dx in -1..=1 {
                value += kernel[kernel_at]
                    * sample_clamped(dataset, coord, x_axis, y_axis, x + dx, y + dy);
                kernel_at += 1;
            }
        }
        output.push(value * scale);
    });

    let data = ArrayD::from_shape_vec(IxDyn(&shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build convolved dataset".to_string())
    })?;
    Ok(Dataset::new(data, dataset.metadata.clone())?)
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

fn find_maxima_mask_xy(
    dataset: &DatasetF32,
    prominence: f32,
    strict: bool,
    exclude_edges: bool,
    light_background: bool,
) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let shape = dataset.shape().to_vec();
    let width = shape[x_axis];
    let height = shape[y_axis];
    let mut output = vec![0.0_f32; dataset.data.len()];
    let mut write_at = 0usize;

    iterate_indices(&shape, |coord| {
        let x = coord[x_axis];
        let y = coord[y_axis];
        let edge = x == 0 || y == 0 || x + 1 == width || y + 1 == height;
        if exclude_edges && edge {
            write_at += 1;
            return;
        }

        let center = maxima_sample(dataset.data[IxDyn(coord)], light_background);
        if !center.is_finite() {
            write_at += 1;
            return;
        }

        let mut max_neighbor = f32::NEG_INFINITY;
        let mut is_maximum = true;
        for dy in -1_isize..=1 {
            for dx in -1_isize..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let sx = x as isize + dx;
                let sy = y as isize + dy;
                if sx < 0 || sy < 0 || sx >= width as isize || sy >= height as isize {
                    continue;
                }
                let mut sample_coord = coord.to_vec();
                sample_coord[x_axis] = sx as usize;
                sample_coord[y_axis] = sy as usize;
                let neighbor = maxima_sample(dataset.data[IxDyn(&sample_coord)], light_background);
                if !neighbor.is_finite() {
                    continue;
                }
                max_neighbor = max_neighbor.max(neighbor);
                if (strict && center <= neighbor) || (!strict && center < neighbor) {
                    is_maximum = false;
                }
            }
        }

        if is_maximum && (max_neighbor == f32::NEG_INFINITY || center - max_neighbor >= prominence)
        {
            output[write_at] = 1.0;
        }
        write_at += 1;
    });

    let data = ArrayD::from_shape_vec(IxDyn(&shape), output).map_err(|_| {
        OpsError::UnsupportedLayout("failed to build maxima mask dataset".to_string())
    })?;
    let mut metadata = dataset.metadata.clone();
    metadata.pixel_type = PixelType::U8;
    Ok(Dataset::new(data, metadata)?)
}

fn maxima_sample(value: f32, light_background: bool) -> f32 {
    if light_background { -value } else { value }
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
    let fx = src_x - x0 as f32;
    let fy = src_y - y0 as f32;

    let p00 = sample_clamped(dataset, coord, x_axis, y_axis, x0, y0);
    let p10 = sample_clamped(dataset, coord, x_axis, y_axis, x1, y0);
    let p01 = sample_clamped(dataset, coord, x_axis, y_axis, x0, y1);
    let p11 = sample_clamped(dataset, coord, x_axis, y_axis, x1, y1);
    let top = p00 * (1.0 - fx) + p10 * fx;
    let bottom = p01 * (1.0 - fx) + p11 * fx;
    top * (1.0 - fy) + bottom * fy
}

fn bilinear_sample_or_fill(
    dataset: &DatasetF32,
    coord: &[usize],
    x_axis: usize,
    y_axis: usize,
    src_x: f32,
    src_y: f32,
    fill: f32,
) -> f32 {
    let x0 = src_x.floor();
    let x1 = src_x.ceil();
    let y0 = src_y.floor();
    let y1 = src_y.ceil();
    let fx = src_x - x0;
    let fy = src_y - y0;

    let p00 = sample_or_fill(dataset, coord, x_axis, y_axis, x0, y0, fill);
    let p10 = sample_or_fill(dataset, coord, x_axis, y_axis, x1, y0, fill);
    let p01 = sample_or_fill(dataset, coord, x_axis, y_axis, x0, y1, fill);
    let p11 = sample_or_fill(dataset, coord, x_axis, y_axis, x1, y1, fill);
    let top = p00 * (1.0 - fx) + p10 * fx;
    let bottom = p01 * (1.0 - fx) + p11 * fx;
    top * (1.0 - fy) + bottom * fy
}

fn sample_or_fill(
    dataset: &DatasetF32,
    coord: &[usize],
    x_axis: usize,
    y_axis: usize,
    x: f32,
    y: f32,
    fill: f32,
) -> f32 {
    if x < 0.0
        || y < 0.0
        || x >= dataset.shape()[x_axis] as f32
        || y >= dataset.shape()[y_axis] as f32
    {
        return fill;
    }
    let mut index = coord.to_vec();
    index[x_axis] = x as usize;
    index[y_axis] = y as usize;
    dataset.data[IxDyn(&index)]
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

fn surface_plot_xy(dataset: &DatasetF32, params: &Value) -> Result<DatasetF32> {
    let x_axis = axis_index(dataset, AxisKind::X)?;
    let y_axis = axis_index(dataset, AxisKind::Y)?;
    let input_shape = dataset.shape();
    let width = input_shape[x_axis];
    let height = input_shape[y_axis];
    if width == 0 || height == 0 {
        return Err(OpsError::UnsupportedLayout(
            "Surface Plot requires a non-empty X/Y image".to_string(),
        ));
    }

    let mut base_coord = vec![0usize; dataset.ndim()];
    for (axis_kind, key) in [
        (AxisKind::Z, "z"),
        (AxisKind::Time, "time"),
        (AxisKind::Channel, "channel"),
    ] {
        if let Some(axis) = dataset.axis_index(axis_kind) {
            base_coord[axis] = get_optional_usize(params, key, 0).min(input_shape[axis] - 1);
        }
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for y in 0..height {
        for x in 0..width {
            let mut coord = base_coord.clone();
            coord[x_axis] = x;
            coord[y_axis] = y;
            let value = dataset.data[IxDyn(&coord)];
            if value.is_finite() {
                min = min.min(value);
                max = max.max(value);
            }
        }
    }
    if min == f32::INFINITY {
        min = 0.0;
        max = 1.0;
    }
    let span = (max - min).max(f32::EPSILON);

    let plot_width = get_optional_usize(params, "plot_width", width.max(64)).max(1);
    let polygon_multiplier = get_optional_usize(params, "polygon_multiplier", 100).clamp(10, 400);
    let source_background_lighter = params
        .get("source_background_lighter")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let black_fill = params
        .get("black_fill")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let rows = ((height as f32) * (polygon_multiplier as f32 / 100.0))
        .round()
        .max(1.0) as usize;
    let height_scale = (plot_width as f32 * 0.45).max(1.0);
    let row_step = (plot_width as f32 * 0.18 / rows.max(1) as f32).max(1.0);
    let skew = (plot_width as f32 * 0.35 / rows.max(1) as f32).max(0.0);
    let margin = 8usize;
    let output_width = plot_width + (skew * rows as f32).ceil() as usize + margin * 2;
    let output_height = (height_scale + row_step * rows as f32).ceil() as usize + margin * 2 + 1;
    let background = if black_fill { 0.0 } else { 1.0 };
    let mut output = vec![background; output_width * output_height];

    for row in (0..rows).rev() {
        let src_y = ((row as f32 + 0.5) * height as f32 / rows as f32)
            .floor()
            .clamp(0.0, (height - 1) as f32) as usize;
        let depth = rows - 1 - row;
        for col in 0..plot_width {
            let src_x = ((col as f32 + 0.5) * width as f32 / plot_width as f32)
                .floor()
                .clamp(0.0, (width - 1) as f32) as usize;
            let mut coord = base_coord.clone();
            coord[x_axis] = src_x;
            coord[y_axis] = src_y;
            let mut normalized = ((dataset.data[IxDyn(&coord)] - min) / span).clamp(0.0, 1.0);
            if source_background_lighter {
                normalized = 1.0 - normalized;
            }

            let base_x = margin as f32 + col as f32 + depth as f32 * skew;
            let base_y = margin as f32 + height_scale + depth as f32 * row_step;
            let top_y = base_y - normalized * height_scale;
            let shade = (1.0 - normalized * 0.85).clamp(0.0, 1.0);
            let x = base_x.round() as usize;
            let y0 = top_y.floor().max(0.0) as usize;
            let y1 = base_y.ceil().min((output_height - 1) as f32) as usize;
            if x >= output_width {
                continue;
            }
            for y in y0..=y1 {
                output[y * output_width + x] = shade;
            }
        }
    }

    let data =
        ArrayD::from_shape_vec(IxDyn(&[output_height, output_width]), output).map_err(|_| {
            OpsError::UnsupportedLayout("failed to build surface plot dataset".to_string())
        })?;
    let mut metadata = Metadata {
        dims: vec![
            Dim::new(AxisKind::Y, output_height),
            Dim::new(AxisKind::X, output_width),
        ],
        pixel_type: PixelType::F32,
        channel_names: Vec::new(),
        source: dataset.metadata.source.clone(),
        extras: dataset.metadata.extras.clone(),
    };
    metadata.extras.insert(
        "surface_plot_source_shape".to_string(),
        json!([height, width]),
    );
    metadata
        .extras
        .insert("surface_plot_min".to_string(), json!(min));
    metadata
        .extras
        .insert("surface_plot_max".to_string(), json!(max));
    metadata.extras.insert(
        "surface_plot_polygon_multiplier".to_string(),
        json!(polygon_multiplier),
    );
    Ok(Dataset::new(data, metadata)?)
}

fn axis_index(dataset: &DatasetF32, axis: AxisKind) -> Result<usize> {
    dataset
        .axis_index(axis)
        .ok_or_else(|| OpsError::UnsupportedLayout(format!("dataset has no {axis:?} axis")))
}

fn linear_offset(shape: &[usize], coord: &[usize]) -> usize {
    let mut offset = 0usize;
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        offset += coord[axis] * stride;
        stride *= shape[axis];
    }
    offset
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
