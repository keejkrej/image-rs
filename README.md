# image-rs

Rust-first core rewrite inspired by ImageJ, with a native GPUI desktop UI and deterministic CLI pipelines.

## Architecture

- Single package: `image-rs`
- Single binary: `image`
- Internal modules: `cli`, `ui`, `model`, `formats`, `commands`, `workflow`, `runtime`

## What currently works

- CLI image IO for `png`, `jpg`/`jpeg`, `tif`/`tiff`
- Deterministic pipeline execution from JSON or YAML recipes
- Operation introspection with `image ops list`
- Native ImageJ-style GPUI workspace with a persistent launcher and one native window per image
- Shared tools, macros, command routing, ROI clipboard edits, ROI Manager, undo/redo, and persistent Results across viewers
- MorphoLibJ-style operations integrated via [`morpholib-rs`](https://github.com/keejkrej/morpholib-rs)

## Quick start

```bash
cargo test
cargo run -- ops list
```

Basic CLI examples:

```bash
cargo run -- info ./input.tiff
cargo run -- convert ./input.png ./output.tiff
cargo run -- view ./input.tiff
```

Run a pipeline (recipe path is JSON or YAML):

```bash
cargo run -- run \
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

- Run `image` with no arguments to launch the native UI.
- The startup launcher remains the command and tool surface. Every open image gets a native viewer window with an independent `viewer-N` session; reopening the same path focuses its existing viewer.
- ImageJ-aligned behavior is presented with a modern zinc/blue surface, compact shadcn-inspired controls, and Lucide tool icons.
- Closing a modified viewer, Close All, and Quit use a cancelable Save / Don’t Save guard. The most recently activated viewer remains the command target.
- Use `File > Open...` (native file dialog) or drag-and-drop supported image files: `png`, `jpg`/`jpeg`, `tif`/`tiff`.
- Tool shortcuts (`R`, `O`, `G`, `F`, `L`, `A`, `P`, `W`, `T`, `Z`, `H`, `D`) work across image windows.
- Rectangle, oval, polygon, freehand, line, angle, point, wand, and text selections render as ImageJ-style yellow overlays; `Image > Crop` crops to the active selection.
- Cut/copy/paste, clear, fill, and select-all operate on the active plane and ROI bounds, with clipboard contents available in their own viewer.
- Parameterized processing commands open viewer-bound GPUI dialogs, operation edits support undo/redo, and calibrated measurement rows accumulate in a persistent, horizontally scrollable Results window with CSV export and unit-safe summaries.
- Brightness/Contrast and Window/Level share a modeless, ROI-aware histogram utility with Minimum, Maximum, Brightness, Contrast, Window, and Level controls. Scientific channels retain independent display ranges; RGB Auto uses weighted luminance and repeated Auto progressively rejects sparse tails.
- Display changes remain non-destructive until Apply LUT. Apply asks before changing pixels, honors exact ROI shapes, offers current-plane or full Z/T-stack scope, and records ImageJ-compatible `slice`/`stack` macros.
- The modeless ROI Manager stores named selections with C/Z/T positions, supports stable Ctrl/Cmd and Shift multi-selection, restores selections into the active viewer, previews Show All without mutating overlays, measures real ROI geometry, and supports ImageJ `roiManager(...)` macro actions.
- Analyze Particles supports threshold ranges, 4/8 connectivity, size and circularity filters, edge exclusion, calibrated areas, and one ImageJ-style Results row per particle.
- ImageJ macro files can be run and installed; the native Recorder can pause, clear, and save `run(...)`, `setMinAndMax(...)`, and `resetMinAndMax()` statements, and `RunAtStartup.ijm` is supported.
- The Window menu lists every live viewer and focuses it directly; Next/Previous/Put Behind follow viewer activation order.
- Stack and hyperstack datasets expose compact C/Z/T navigation controls in their viewer window.
