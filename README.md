# image-rs

Rust-first core rewrite inspired by ImageJ2, with a thin Tauri viewer and deterministic CLI pipelines.

## Workspace

- Core crates: `ijrs-core`, `ijrs-io`, `ijrs-ops`, `ijrs-pipeline`, `ijrs-app`
- CLI: `ijrs`
- Viewer: `ijrs-tauri`

## Quick start

```bash
cargo test --workspace
cargo run -p ijrs-cli -- ops list
```

Run a pipeline:

```bash
cargo run -p ijrs-cli -- run \
  --input ./input.tiff \
  --recipe ./fixtures/recipes/normalize-threshold.json \
  --output ./output.tiff \
  --report ./report.json
```

Open viewer:

```bash
cargo run -p ijrs-tauri -- ./input.tiff
```
