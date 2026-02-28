use std::path::PathBuf;

use clap::{Parser, Subcommand};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    name = "image",
    version,
    about = "ImageJ2-inspired Rust image processing CLI"
)]
pub(super) struct Cli {
    #[command(subcommand)]
    pub(super) command: Commands,
}

#[derive(Debug, Subcommand)]
pub(super) enum Commands {
    Info {
        input: PathBuf,
    },
    Convert {
        input: PathBuf,
        output: PathBuf,
    },
    Run {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        recipe: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    Ops {
        #[command(subcommand)]
        command: OpsCommand,
    },
    /// Launches the native control window and opens an initial viewer for this image.
    /// The control window remains open as the drag-and-drop entry point.
    View {
        input: PathBuf,
    },
}

#[derive(Debug, Subcommand)]
pub(super) enum OpsCommand {
    List,
}

#[derive(Debug, Serialize)]
pub(super) struct DatasetInfo {
    pub(super) shape: Vec<usize>,
    pub(super) ndim: usize,
    pub(super) pixel_type: String,
    pub(super) axes: Vec<String>,
    pub(super) channel_names: Vec<String>,
    pub(super) source: Option<String>,
    pub(super) min: Option<f32>,
    pub(super) max: Option<f32>,
}
