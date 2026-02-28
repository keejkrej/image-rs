use std::fs;
use std::path::Path;

use super::{PipelineReport, PipelineSpec, Result};

pub fn load_spec(path: impl AsRef<Path>) -> Result<PipelineSpec> {
    let path = path.as_ref();
    let raw = fs::read_to_string(path)?;
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let spec = if matches!(extension.as_str(), "yaml" | "yml") {
        serde_yaml::from_str::<PipelineSpec>(&raw)?
    } else {
        serde_json::from_str::<PipelineSpec>(&raw)?
    };
    spec.validate()?;
    Ok(spec)
}

pub fn save_report(path: impl AsRef<Path>, report: &PipelineReport) -> Result<()> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let serialized = if matches!(extension.as_str(), "yaml" | "yml") {
        serde_yaml::to_string(report)?
    } else {
        serde_json::to_string_pretty(report)?
    };
    fs::write(path, serialized)?;
    Ok(())
}
