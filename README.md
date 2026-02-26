# image-rs

Rust-first core rewrite inspired by ImageJ2, with a thin Tauri viewer and deterministic CLI pipelines.

## Workspace

- Core crates: `image-model`, `image-formats`, `image-commands`, `image-workflow`, `image-runtime`
- CLI: `image`
- UI crate: `image-ui`

## Quick start

```bash
cargo test --workspace
cargo run -p image-cli -- ops list
```

Run a pipeline:

```bash
cargo run -p image-cli -- run \
  --input ./input.tiff \
  --recipe ./fixtures/recipes/normalize-threshold.json \
  --output ./output.tiff \
  --report ./report.json
```

Open viewer:

```bash
cargo run -p image-cli -- view ./input.tiff
```
