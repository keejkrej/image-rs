# image-rs

Rust-first core rewrite inspired by ImageJ2, with a native egui desktop UI and deterministic CLI pipelines.

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

Launcher notes:

- Native compact launcher shell: menu bar, ImageJ-style tool icon strip, and status row.
- Use `File > Open...` (native file dialog) or drag-and-drop TIFF files onto the launcher.
- Tool shortcuts (`R`, `O`, `G`, `F`, `L`, `P`, `W`, `T`, `Z`, `H`, `D`) are shared across launcher and viewer windows.
