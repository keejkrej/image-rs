# image-rs

Rust-first core rewrite inspired by ImageJ, with a native GPUI desktop UI and deterministic CLI pipelines.

## Architecture

- Single package: `image-rs`
- Single binary: `image`
- Internal modules: `cli`, `ui`, `model`, `formats`, `commands`, `plugins`, `workflow`, `runtime`

## What currently works

- CLI image IO for `png`, `jpg`/`jpeg`, `tif`/`tiff`
- Deterministic pipeline execution from JSON or YAML recipes
- Operation introspection with `image ops list`
- Native ImageJ-style GPUI workspace with a persistent launcher and one native window per image
- Shared tools, macros, command routing, ROI clipboard edits, ROI Manager, undo/redo, and persistent Results across viewers
- Safe plugin package discovery with strict manifests, SemVer compatibility, namespaced contributions, and a versioned WebAssembly Component contract; a host/library registration API executes compatible all-plane operations through a bounded no-WASI Wasmtime sandbox, while automatic CLI/UI discovery remains future work; see [`docs/plugin-system.md`](docs/plugin-system.md)
- MorphoLibJ-style operations integrated via [`morpholib-rs`](https://github.com/keejkrej/morpholib-rs)

## Application-owned Bio-Formats storage

The optional `bioformats` feature opens TIFF/OME-TIFF, ND2, CZI, NRRD, MRC,
and DCIMG datasets from image-rs-owned range storage. The returned
`BioformatsDataset` is the native lazy dataset: callers keep the exact
`PixelLayout` and issue explicit series, resolution, Z/C/T, and XY-region
requests.

```rust,ignore
use std::sync::Arc;
use image_rs::formats::{
    AssetSnapshot, PlaneCoordinates, RangeStorage, ReadRequest, Rect, Region,
    open_bioformats_asset,
};

let primary = AssetSnapshot::new(
    "objects/acquisition-42@etag-9d2",
    "sample.czi",
    object_length,
)?;
let dataset = open_bioformats_asset(Arc::new(application_store), primary)?;
let request = ReadRequest::new(0, PlaneCoordinates::new(4, 1, 2))
    .with_resolution(0)
    .with_region(Region::Rect(Rect::new(128, 64, 512, 256)?));
let info = dataset.plane_info(request)?;
let mut bytes = vec![0; info.byte_len];
dataset.read_plane_into(request, &mut bytes)?;
```

`RangeStorage::read_exact_at` adapts the application's versioned object key,
logical name, and length to `bioformats_rs::RandomAccessSource`. It must fill
the requested range exactly and remain safe for unordered concurrent calls.
The library checks ranges before forwarding them and retains the store for lazy
pixel reads; it does not create temporary files, emulate a filesystem, or load
an entire asset merely to open it.

Companion lookup deliberately stays in the same application namespace:

- `resolve_named` handles detached NRRD data and metadata-declared OME-TIFF members.
- `siblings` supplies the complete candidate set for split CZI assets; bioformats-rs performs CZI-specific filtering, de-duplication, and ordering.

The `RangeStorage` instance passed to `open_bioformats_asset` defines that one
logical namespace: every returned companion snapshot must remain readable by
the same instance, with a unique stable identity, for the dataset's lifetime.
An application adapter may federate multiple physical providers internally.
There is no separate companion-index seam because no supported companion flow
currently requires independently owned providers; one should be introduced
only for a concrete cross-provider use case.

`materialize_bioformats_plane` is an explicitly eager image-rs conversion for
one requested plane or region. It interprets native byte order and
planar/interleaved samples without normalization, then creates a
`Dataset<f32>`. Numeric samples are converted directly without normalization,
rescaling, or clamping: exactly representable values retain their value;
`Int32`, `Uint32`, and `Float64` use Rust's `as f32` semantics and may round,
with out-of-range floating-point values becoming infinities. The native layout
is retained in metadata extras. Whole-series materialization is not implicit.

Existing convenience paths remain unchanged. `read_dataset`,
`read_native_image`, `DefaultImageCodec::read`, `IoService::read`, and the
CLI/UI open flows are filesystem-based and eager. `read_dataset_bytes`,
`read_native_image_bytes`, and `IoService::read_bytes` remain eager
whole-buffer helpers and cannot resolve companions. Writers and `source_path`
also remain path-oriented.

Enable and test the integration with:

```bash
cargo test --features bioformats --test bioformats_storage
```

This is not full Java Bio-Formats parity. Notable gaps include ND2 JPEG 2000;
CZI JPEG-XR, pyramid exposure, complex pixels, and some axes/LUT behavior;
DCIMG multi-file Z grouping and timestamps; and documented TIFF packed-sample,
FillOrder 2, Predictor 3, CMYK/WhiteIsZero, and non-JPEG YCbCr variants. Some
implemented compressed paths still need broader real-fixture verification.

`bioformats-rs` is GPL-2.0-or-later. The feature is disabled by default so
ordinary image-rs builds retain their existing dependency set, but distributing
a build that enables it can carry GPL obligations.

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

The current command counts, semantic gaps, and recommended parity sequence are tracked in [`docs/imagej-parity.md`](docs/imagej-parity.md).

- Run `image` with no arguments to launch the native UI.
- The startup launcher remains the command and tool surface. Every open image gets a native viewer window with an independent `viewer-N` session; reopening the same path focuses its existing viewer.
- ImageJ-aligned behavior is presented with a modern zinc/blue surface, compact shadcn-inspired controls, and Lucide tool icons.
- Closing a modified viewer, Close All, and Quit use a cancelable Save / Don’t Save guard. The most recently activated viewer remains the command target.
- Use `File > Open...` (native file dialog) or drag-and-drop supported image files: `png`, `jpg`/`jpeg`, `tif`/`tiff`.
- Tool shortcuts (`R`, `O`, `G`, `F`, `L`, `A`, `P`, `W`, `T`, `Z`, `H`, `D`) work across image windows.
- Rectangle, oval, polygon, freehand, line, angle, point, wand, and text selections render as ImageJ-style yellow overlays; `Image > Crop` crops to the active selection.
- Cut/copy/paste, clear, fill, and select-all operate on the active plane and ROI bounds, with clipboard contents available in their own viewer.
- Parameterized processing commands open viewer-bound GPUI dialogs and operation edits support undo/redo. Analyze > Measure samples only the active C/Z/T plane and exact ROI, with calibrated ImageJ-style columns controlled by application-wide Set Measurements settings; Measure Stack appends one row per Z plane at the active C/T. Stored results retain full precision while the Results window and CSV export honor the selected decimal places.
- Brightness/Contrast and Window/Level share a modeless, ROI-aware histogram utility with Minimum, Maximum, Brightness, Contrast, Window, and Level controls. Scientific channels retain independent display ranges; RGB Auto uses weighted luminance and repeated Auto progressively rejects sparse tails.
- Display changes remain non-destructive until Apply LUT. Apply asks before changing pixels, honors exact ROI shapes, offers current-plane or full Z/T-stack scope, and records ImageJ-compatible `slice`/`stack` macros.
- The modeless ROI Manager stores named selections with C/Z/T positions, supports stable Ctrl/Cmd and Shift multi-selection, restores selections into the active viewer, previews Show All without mutating overlays, measures real ROI geometry, and supports ImageJ `roiManager(...)` macro actions.
- Analyze Particles supports threshold ranges, 4/8 connectivity, size and circularity filters, edge exclusion, calibrated areas, and one ImageJ-style Results row per particle.
- ImageJ macro files can be run and installed; the native Recorder can pause, clear, and save `run(...)`, `setMinAndMax(...)`, and `resetMinAndMax()` statements, and `RunAtStartup.ijm` is supported.
- The Window menu lists every live viewer and focuses it directly; Next/Previous/Put Behind follow viewer activation order.
- Stack and hyperstack datasets expose compact C/Z/T navigation controls in their viewer window.
