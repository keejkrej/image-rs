use serde_json::Value;

use super::{OpsError, Result};

pub(crate) fn get_required_f32(params: &Value, key: &str) -> Result<f32> {
    params
        .get(key)
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .ok_or_else(|| OpsError::InvalidParams(format!("missing float parameter `{key}`")))
}

pub(crate) fn get_optional_f32(params: &Value, key: &str, default: f32) -> f32 {
    params
        .get(key)
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(default)
}

pub(crate) fn get_optional_usize(params: &Value, key: &str, default: usize) -> usize {
    params
        .get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(default)
}

#[cfg(feature = "morpholib")]
pub(crate) fn get_optional_bool(params: &Value, key: &str, default: bool) -> bool {
    params.get(key).and_then(Value::as_bool).unwrap_or(default)
}
