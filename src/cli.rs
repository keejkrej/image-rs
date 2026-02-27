use std::path::PathBuf;

use clap::{Parser, Subcommand};
use serde::Serialize;
use serde_json::json;

use crate::runtime::AppContext;

#[derive(Debug, Parser)]
#[command(
    name = "image",
    version,
    about = "ImageJ2-inspired Rust image processing CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
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
enum OpsCommand {
    List,
}

#[derive(Debug, Serialize)]
struct DatasetInfo {
    shape: Vec<usize>,
    ndim: usize,
    pixel_type: String,
    axes: Vec<String>,
    channel_names: Vec<String>,
    source: Option<String>,
    min: Option<f32>,
    max: Option<f32>,
}

pub fn run_cli() -> Result<(), String> {
    let cli = Cli::parse();
    let app = AppContext::new();

    match cli.command {
        Commands::Info { input } => {
            let dataset = app
                .io_service()
                .read(&input)
                .map_err(|error| error.to_string())?;
            app.dataset_service()
                .validate(&dataset)
                .map_err(|error| error.to_string())?;
            let (min, max) = dataset.min_max().unwrap_or((0.0, 0.0));
            let info = DatasetInfo {
                shape: dataset.shape().to_vec(),
                ndim: dataset.ndim(),
                pixel_type: format!("{:?}", dataset.metadata.pixel_type),
                axes: dataset
                    .metadata
                    .dims
                    .iter()
                    .map(|dimension| format!("{:?}", dimension.axis))
                    .collect(),
                channel_names: dataset.metadata.channel_names.clone(),
                source: dataset
                    .metadata
                    .source
                    .as_ref()
                    .map(|path| path.display().to_string()),
                min: Some(min),
                max: Some(max),
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&info).map_err(|error| error.to_string())?
            );
        }
        Commands::Convert { input, output } => {
            let dataset = app
                .io_service()
                .read(&input)
                .map_err(|error| error.to_string())?;
            app.io_service()
                .write(&output, &dataset)
                .map_err(|error| error.to_string())?;
            println!("{}", json!({"status": "ok", "output": output}));
        }
        Commands::Run {
            input,
            recipe,
            output,
            report,
        } => {
            let dataset = app
                .io_service()
                .read(&input)
                .map_err(|error| error.to_string())?;
            let spec = app
                .pipeline_service()
                .load_spec(&recipe)
                .map_err(|error| error.to_string())?;
            let (result, run_report) = app
                .pipeline_service()
                .run(&spec, &dataset)
                .map_err(|error| error.to_string())?;
            app.io_service()
                .write(&output, &result)
                .map_err(|error| error.to_string())?;
            if let Some(report_path) = report {
                app.pipeline_service()
                    .save_report(report_path, &run_report)
                    .map_err(|error| error.to_string())?;
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&run_report).map_err(|error| error.to_string())?
            );
        }
        Commands::Ops { command } => match command {
            OpsCommand::List => {
                let schemas = app.ops_service().list();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&schemas).map_err(|error| error.to_string())?
                );
            }
        },
        Commands::View { input } => {
            launch_native_viewer(&input)?;
        }
    }

    Ok(())
}

fn launch_native_viewer(input: &PathBuf) -> Result<(), String> {
    crate::ui::run(Some(input.clone()))
}

