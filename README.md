# image-rs

Rust-first core rewrite inspired by ImageJ2, with a native egui desktop UI and deterministic CLI pipelines.

## Workspace

- Core crates: `image-model`, `image-formats`, `image-commands`, `image-workflow`, `image-runtime`
- CLI binary: `image` (from `image-cli`)
- Desktop UI crate/binary: `image-ui`

## What currently works

- CLI image IO for `png`, `jpg`/`jpeg`, `tif`/`tiff`
- Deterministic pipeline execution from JSON or YAML recipes
- Operation introspection with `image ops list`
- Native ImageJ-style launcher + viewer shell
- MorphoLibJ-style operations integrated via [`morpholib-rs`](https://github.com/keejkrej/morpholib-rs)

## Quick start

```bash
cargo test --workspace
cargo run -p image-cli -- ops list
```

Basic CLI examples:

```bash
cargo run -p image-cli -- info ./input.tiff
cargo run -p image-cli -- convert ./input.png ./output.tiff
cargo run -p image-cli -- view ./input.tiff
```

Run a pipeline (recipe path is JSON or YAML):

```bash
cargo run -p image-cli -- run \
  --input ./input.tiff \
  --recipe ./pipeline.json \
  --output ./output.tiff \
  --report ./report.json
```

Pipeline recipe shape:

```json
{
  "name": "normalize-threshold-chamfer",
  "operations": [
    { "op": "intensity.normalize", "params": {} },
    { "op": "threshold.otsu", "params": {} },
    {
      "op": "morpholibj.distance.chamfer",
      "params": { "connectivity": 8, "normalize": true }
    }
  ]
}
```

## MorphoLib integration

Added operations:

- `morpholibj.distance.chamfer`
- `morpholibj.reconstruct.by_dilation`
- `morpholibj.reconstruct.by_erosion`

Current constraints:

- MorphoLib operations currently support 2D datasets only.
- `connectivity` currently supports `4` or `8` (default `8`).

## UI launcher notes

- Native launcher shell: menu bar, ImageJ-style tool icon strip, and status row.
- Launcher window is resizable and starts at minimum size (`600x200`, 3:1).
- Use `File > Open...` (native file dialog) or drag-and-drop TIFF files (`.tif`/`.tiff`) onto the launcher.
- Tool shortcuts (`R`, `O`, `G`, `F`, `L`, `P`, `W`, `T`, `Z`, `H`, `D`) are shared across launcher and viewer windows.
