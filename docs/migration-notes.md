# Migration Notes (ImageJ2 -> image-rs)

## Intent

`image-rs` is a conceptual rewrite of core ImageJ2 workflows in Rust.

## What is intentionally different in V1

- No dynamic SciJava plugin graph.
- No ImageJ legacy macro/plugin compatibility layer.
- No updater/uploader/scripting stack port.
- No runtime service discovery.

## What is preserved conceptually

- Centralized app context and service boundaries.
- Deterministic operation pipelines.
- Explicit multidimensional data model and metadata.
- Headless + interactive workflows from the same core APIs.

## Recommended migration path

1. Replace ad hoc script chains with `ijrs run` recipes.
2. Validate operation parity using fixture images and reports.
3. Use `ijrs view` for exploratory slice-level analysis.
4. Gradually move external tooling onto `ijrs-app` crate APIs.
