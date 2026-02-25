# image-rs Architecture

## Workspace layout

- `crates/ijrs-core`: N-dimensional dataset model and metadata contracts.
- `crates/ijrs-io`: TIFF/PNG/JPEG readers and writers behind codec traits.
- `crates/ijrs-ops`: Core operation registry and implementations.
- `crates/ijrs-pipeline`: JSON/YAML pipeline spec and deterministic executor.
- `crates/ijrs-app`: Application context exposing dataset, I/O, ops, and pipeline services.
- `crates/ijrs-cli`: Stable command interface (`ijrs info/convert/run/ops/view`).
- `apps/ijrs-tauri`: Thin Tauri viewer calling app services through tauri commands.

## Data model

- Primary container: `ijrs_core::Dataset<T>` with `ndarray::ArrayD<T>`.
- Metadata is explicit and validated on construction.
- Axes are annotated using `AxisKind` (`X`, `Y`, `Z`, `Channel`, `Time`, `Unknown`).
- Pixel precision metadata is tracked with `PixelType` (`U8`, `U16`, `F32`).

## Processing model

- Operations are compile-time registered; no runtime plugin discovery exists in V1.
- Each operation exposes:
  - Stable `name`
  - Parameter schema (`OpSchema`)
  - Deterministic `execute(dataset, params)` behavior
- Pipelines are ordered lists of operation invocations (`PipelineSpec`).

## Error model

- Every crate has typed `thiserror` errors.
- Lower-layer errors are wrapped and surfaced at the app/CLI/UI boundaries.
- CLI/UI present user-readable messages while preserving operation-specific context.

## UI integration

- Tauri commands call into `ijrs-app` services.
- The UI remains thin:
  - It requests metadata/slices/histograms/preview/export.
  - It does not implement image processing logic itself.
