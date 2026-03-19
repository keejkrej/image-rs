use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopState {
    pub recent_files: Vec<PathBuf>,
    pub measurement_settings: MeasurementSettings,
    pub utility_windows: UtilityWindowsState,
}

impl Default for DesktopState {
    fn default() -> Self {
        Self {
            recent_files: Vec::new(),
            measurement_settings: MeasurementSettings::default(),
            utility_windows: UtilityWindowsState::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementSettings {
    pub area: bool,
    pub min: bool,
    pub max: bool,
    pub mean: bool,
    pub centroid: bool,
    pub bbox: bool,
    pub integrated_density: bool,
    pub slice: bool,
    pub channel: bool,
    pub time: bool,
}

impl Default for MeasurementSettings {
    fn default() -> Self {
        Self {
            area: true,
            min: true,
            max: true,
            mean: true,
            centroid: true,
            bbox: true,
            integrated_density: false,
            slice: true,
            channel: true,
            time: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UtilityWindowsState {
    pub results_open: bool,
    pub roi_manager_open: bool,
    pub measurements_open: bool,
    pub command_finder_open: bool,
    pub profile_plot_open: bool,
    pub help_about_open: bool,
    pub help_docs_open: bool,
    pub help_shortcuts_open: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ResultsTableState {
    pub rows: Vec<BTreeMap<String, serde_json::Value>>,
}

impl ResultsTableState {
    pub fn clear(&mut self) {
        self.rows.clear();
    }

    pub fn add_row(&mut self, row: BTreeMap<String, serde_json::Value>) {
        self.rows.push(row);
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn columns(&self) -> Vec<String> {
        let mut columns = Vec::new();
        for row in &self.rows {
            for key in row.keys() {
                if !columns.contains(key) {
                    columns.push(key.clone());
                }
            }
        }
        columns
    }
}

pub fn load_desktop_state() -> DesktopState {
    let path = state_file_path();
    let Ok(bytes) = fs::read(&path) else {
        return DesktopState::default();
    };
    serde_json::from_slice(&bytes).unwrap_or_default()
}

pub fn save_desktop_state(state: &DesktopState) -> std::io::Result<()> {
    let path = state_file_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload = serde_json::to_vec_pretty(state)
        .map_err(|error| std::io::Error::other(format!("serialize desktop state: {error}")))?;
    fs::write(path, payload)
}

pub fn push_recent_file(state: &mut DesktopState, path: &Path) {
    let normalized = path.to_path_buf();
    state
        .recent_files
        .retain(|candidate| candidate != &normalized);
    state.recent_files.insert(0, normalized);
    state.recent_files.truncate(10);
}

fn state_file_path() -> PathBuf {
    config_dir().join("image-rs").join("ui-state.json")
}

fn config_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home)
                .join("Library")
                .join("Application Support");
        }
    }

    if let Some(path) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(path);
    }

    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".config");
    }

    PathBuf::from(".")
}
