use crate::model::{Dataset, DatasetF32, PixelType};
use ndarray::{Array, IxDyn};
use rayon::prelude::*;
use serde_json::Value;

use super::{
    OpOutput, OpSchema, Operation, OpsError, ParamSpec, Result, get_optional_f32, get_required_f32,
};

#[derive(Debug, Clone, Copy)]
pub struct IntensityNormalizeOp;

#[derive(Debug, Clone, Copy)]
pub struct IntensityInvertOp;

#[derive(Debug, Clone, Copy)]
pub struct IntensityMathOp;

#[derive(Debug, Clone, Copy)]
pub struct IntensityNaNBackgroundOp;

impl Operation for IntensityNormalizeOp {
    fn name(&self) -> &'static str {
        "intensity.normalize"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Normalize values into [0, 1] using optional min/max.".to_string(),
            params: vec![
                ParamSpec {
                    name: "min".to_string(),
                    description: "Optional lower bound. Uses dataset minimum when omitted."
                        .to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "max".to_string(),
                    description: "Optional upper bound. Uses dataset maximum when omitted."
                        .to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let (source_min, source_max) = dataset.min_max().unwrap_or((0.0, 1.0));
        let min = params
            .get("min")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(source_min);
        let max = params
            .get("max")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .unwrap_or(source_max);
        let scale = if (max - min).abs() < f32::EPSILON {
            1.0
        } else {
            max - min
        };
        let mut values = dataset.data.iter().copied().collect::<Vec<_>>();
        values.par_iter_mut().for_each(|value| {
            *value = ((*value - min) / scale).clamp(0.0, 1.0);
        });
        let normalized = Array::from_shape_vec(IxDyn(dataset.shape()), values)
            .expect("shape is unchanged and valid");
        let output_dataset = Dataset::new(normalized, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

impl Operation for IntensityInvertOp {
    fn name(&self) -> &'static str {
        "intensity.invert"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Invert intensities using ImageJ-style pixel ranges.".to_string(),
            params: vec![
                ParamSpec {
                    name: "min".to_string(),
                    description: "Optional lower display bound for f32 data.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "max".to_string(),
                    description: "Optional upper display bound for f32 data.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let (min, max) = match dataset.metadata.pixel_type {
            PixelType::U8 | PixelType::U16 => (0.0, 1.0),
            PixelType::F32 => {
                let (source_min, source_max) = dataset.min_max().unwrap_or((0.0, 1.0));
                (
                    get_optional_f32(params, "min", source_min),
                    get_optional_f32(params, "max", source_max),
                )
            }
        };
        if max < min {
            return Err(OpsError::InvalidParams(
                "`max` must be greater than or equal to `min`".to_string(),
            ));
        }

        let mut values = dataset.data.iter().copied().collect::<Vec<_>>();
        values.par_iter_mut().for_each(|value| {
            *value = max - (*value - min);
        });
        let inverted = Array::from_shape_vec(IxDyn(dataset.shape()), values)
            .expect("shape is unchanged and valid");
        let output_dataset = Dataset::new(inverted, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}

impl Operation for IntensityMathOp {
    fn name(&self) -> &'static str {
        "intensity.math"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Apply ImageJ Process/Math-style per-pixel arithmetic.".to_string(),
            params: vec![
                ParamSpec {
                    name: "operation".to_string(),
                    description:
                        "One of add, subtract, multiply, divide, and, or, xor, min, max, gamma, set, log, exp, square, sqrt, reciprocal, or abs."
                            .to_string(),
                    required: true,
                    kind: "string".to_string(),
                },
                ParamSpec {
                    name: "value".to_string(),
                    description: "Scalar value or bitmask for binary operations.".to_string(),
                    required: false,
                    kind: "float".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let Some(operation) = params.get("operation").and_then(Value::as_str) else {
            return Err(OpsError::InvalidParams(
                "`operation` is required".to_string(),
            ));
        };
        let operation = MathOperation::parse(operation)?;
        if operation.is_bitwise() {
            let mask = get_required_bitmask(params)?;
            return execute_bitwise_math(dataset, operation, mask);
        }
        let value = match operation.requires_value() {
            true => Some(get_required_f32(params, "value")?),
            false => params
                .get("value")
                .and_then(Value::as_f64)
                .map(|v| v as f32),
        };
        if matches!(operation, MathOperation::Divide) && value == Some(0.0) {
            return Err(OpsError::InvalidParams(
                "`value` must be non-zero for divide".to_string(),
            ));
        }
        if matches!(operation, MathOperation::Gamma) {
            let gamma = value.expect("gamma requires a value");
            if !(0.05..=5.0).contains(&gamma) {
                return Err(OpsError::InvalidParams(
                    "`value` for gamma must be between 0.05 and 5.0".to_string(),
                ));
            }
        }

        let mut values = dataset.data.iter().copied().collect::<Vec<_>>();
        values.par_iter_mut().for_each(|pixel| {
            *pixel = operation.apply(*pixel, value);
        });
        let output = Array::from_shape_vec(IxDyn(dataset.shape()), values)
            .expect("shape is unchanged and valid");
        Ok(OpOutput::dataset_only(Dataset::new(
            output,
            dataset.metadata.clone(),
        )?))
    }
}

impl Operation for IntensityNaNBackgroundOp {
    fn name(&self) -> &'static str {
        "intensity.nan_background"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Set pixels outside a threshold range to NaN for f32 images.".to_string(),
            params: vec![
                ParamSpec {
                    name: "lower".to_string(),
                    description: "Lower inclusive threshold.".to_string(),
                    required: true,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "upper".to_string(),
                    description: "Upper inclusive threshold.".to_string(),
                    required: true,
                    kind: "float".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        if dataset.metadata.pixel_type != PixelType::F32 {
            return Err(OpsError::UnsupportedLayout(
                "NaN Background requires f32 pixel metadata".to_string(),
            ));
        }
        let lower = get_required_f32(params, "lower")?;
        let upper = get_required_f32(params, "upper")?;
        if !lower.is_finite() || !upper.is_finite() || upper < lower {
            return Err(OpsError::InvalidParams(
                "`lower` and `upper` must be finite, with upper >= lower".to_string(),
            ));
        }

        let mut values = dataset.data.iter().copied().collect::<Vec<_>>();
        values.par_iter_mut().for_each(|pixel| {
            if *pixel < lower || *pixel > upper {
                *pixel = f32::NAN;
            }
        });
        let output = Array::from_shape_vec(IxDyn(dataset.shape()), values)
            .expect("shape is unchanged and valid");
        Ok(OpOutput::dataset_only(Dataset::new(
            output,
            dataset.metadata.clone(),
        )?))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MathOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    And,
    Or,
    Xor,
    Min,
    Max,
    Gamma,
    Set,
    Log,
    Exp,
    Square,
    Sqrt,
    Reciprocal,
    Abs,
}

impl MathOperation {
    fn parse(operation: &str) -> Result<Self> {
        match operation {
            "add" => Ok(Self::Add),
            "subtract" | "sub" => Ok(Self::Subtract),
            "multiply" | "mul" => Ok(Self::Multiply),
            "divide" | "div" => Ok(Self::Divide),
            "and" => Ok(Self::And),
            "or" => Ok(Self::Or),
            "xor" => Ok(Self::Xor),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            "gamma" => Ok(Self::Gamma),
            "set" => Ok(Self::Set),
            "log" => Ok(Self::Log),
            "exp" => Ok(Self::Exp),
            "square" | "sqr" => Ok(Self::Square),
            "sqrt" | "square_root" => Ok(Self::Sqrt),
            "reciprocal" => Ok(Self::Reciprocal),
            "abs" => Ok(Self::Abs),
            other => Err(OpsError::InvalidParams(format!(
                "unsupported math operation `{other}`"
            ))),
        }
    }

    fn requires_value(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Subtract
                | Self::Multiply
                | Self::Divide
                | Self::Min
                | Self::Max
                | Self::Gamma
                | Self::Set
        )
    }

    fn is_bitwise(self) -> bool {
        matches!(self, Self::And | Self::Or | Self::Xor)
    }

    fn apply_bitwise(self, pixel: u32, mask: u32) -> u32 {
        match self {
            Self::And => pixel & mask,
            Self::Or => pixel | mask,
            Self::Xor => pixel ^ mask,
            _ => unreachable!("only bitwise operations are routed here"),
        }
    }

    fn apply(self, pixel: f32, value: Option<f32>) -> f32 {
        match self {
            Self::Add => pixel + value.expect("add requires a value"),
            Self::Subtract => pixel - value.expect("subtract requires a value"),
            Self::Multiply => pixel * value.expect("multiply requires a value"),
            Self::Divide => pixel / value.expect("divide requires a value"),
            Self::Min => pixel.max(value.expect("min requires a value")),
            Self::Max => pixel.min(value.expect("max requires a value")),
            Self::Gamma => {
                if pixel <= 0.0 {
                    0.0
                } else {
                    pixel.powf(value.expect("gamma requires a value"))
                }
            }
            Self::Set => value.expect("set requires a value"),
            Self::Log => pixel.ln(),
            Self::Exp => pixel.exp(),
            Self::Square => pixel * pixel,
            Self::Sqrt => {
                if pixel <= 0.0 {
                    0.0
                } else {
                    pixel.sqrt()
                }
            }
            Self::Reciprocal => {
                if pixel == 0.0 {
                    f32::NAN
                } else {
                    1.0 / pixel
                }
            }
            Self::Abs => pixel.abs(),
            Self::And | Self::Or | Self::Xor => {
                unreachable!("bitwise operations are routed through integer math")
            }
        }
    }
}

fn get_required_bitmask(params: &Value) -> Result<u32> {
    let Some(value) = params.get("value") else {
        return Err(OpsError::InvalidParams(
            "missing bitmask parameter `value`".to_string(),
        ));
    };
    if let Some(mask) = value.as_u64() {
        return u32::try_from(mask)
            .map_err(|_| OpsError::InvalidParams("bitmask value is too large".to_string()));
    }
    if let Some(mask) = value.as_f64() {
        if mask < 0.0 {
            return Err(OpsError::InvalidParams(
                "bitmask value must be non-negative".to_string(),
            ));
        }
        return u32::try_from(mask.round() as u64)
            .map_err(|_| OpsError::InvalidParams("bitmask value is too large".to_string()));
    }
    if let Some(mask) = value.as_str() {
        let binary = mask.strip_prefix("0b").unwrap_or(mask);
        return u32::from_str_radix(binary, 2).map_err(|_| {
            OpsError::InvalidParams("bitmask string must be a binary number".to_string())
        });
    }
    Err(OpsError::InvalidParams(
        "bitmask parameter `value` must be an integer or binary string".to_string(),
    ))
}

fn execute_bitwise_math(
    dataset: &DatasetF32,
    operation: MathOperation,
    mask: u32,
) -> Result<OpOutput> {
    let max_sample = match dataset.metadata.pixel_type {
        PixelType::U8 => 255_u32,
        PixelType::U16 => 65_535_u32,
        PixelType::F32 => {
            return Err(OpsError::UnsupportedLayout(
                "bitwise math requires u8 or u16 pixel metadata".to_string(),
            ));
        }
    };
    let mut values = dataset.data.iter().copied().collect::<Vec<_>>();
    values.par_iter_mut().for_each(|pixel| {
        let sample = (*pixel).clamp(0.0, 1.0) * max_sample as f32;
        let sample = sample.round() as u32;
        let result = operation.apply_bitwise(sample, mask).min(max_sample);
        *pixel = result as f32 / max_sample as f32;
    });
    let output = Array::from_shape_vec(IxDyn(dataset.shape()), values)
        .expect("shape is unchanged and valid");
    Ok(OpOutput::dataset_only(Dataset::new(
        output,
        dataset.metadata.clone(),
    )?))
}

#[derive(Debug, Clone, Copy)]
pub struct IntensityWindowOp;

impl Operation for IntensityWindowOp {
    fn name(&self) -> &'static str {
        "intensity.window"
    }

    fn schema(&self) -> OpSchema {
        OpSchema {
            name: self.name().to_string(),
            description: "Apply a [low, high] window and clamp to [0, 1].".to_string(),
            params: vec![
                ParamSpec {
                    name: "low".to_string(),
                    description: "Lower window bound.".to_string(),
                    required: true,
                    kind: "float".to_string(),
                },
                ParamSpec {
                    name: "high".to_string(),
                    description: "Upper window bound.".to_string(),
                    required: true,
                    kind: "float".to_string(),
                },
            ],
        }
    }

    fn execute(&self, dataset: &DatasetF32, params: &Value) -> Result<OpOutput> {
        let low = get_required_f32(params, "low")?;
        let high = get_required_f32(params, "high")?;
        if high <= low {
            return Err(OpsError::InvalidParams(
                "`high` must be greater than `low`".to_string(),
            ));
        }
        let mut values = dataset.data.iter().copied().collect::<Vec<_>>();
        values.par_iter_mut().for_each(|value| {
            *value = ((*value - low) / (high - low)).clamp(0.0, 1.0);
        });
        let windowed = Array::from_shape_vec(IxDyn(dataset.shape()), values)
            .expect("shape is unchanged and valid");
        let output_dataset = Dataset::new(windowed, dataset.metadata.clone())?;
        Ok(OpOutput::dataset_only(output_dataset))
    }
}
