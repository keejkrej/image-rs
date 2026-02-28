pub mod cli;
pub mod commands;
pub mod formats;
pub mod model;
pub mod runtime;
pub mod ui;
pub mod workflow;

pub fn run_cli() -> Result<(), String> {
    cli::run_cli()
}
