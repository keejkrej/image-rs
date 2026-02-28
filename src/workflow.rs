mod error;
mod execute;
mod io;
mod report;
mod spec;

#[cfg(test)]
mod tests;

pub use error::{PipelineError, Result};
pub use execute::run_pipeline;
pub use io::{load_spec, save_report};
pub use report::{PipelineReport, StepReport};
pub use spec::{OpInvocation, PipelineSpec};
