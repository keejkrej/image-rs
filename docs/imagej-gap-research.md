# ImageJ gap research

- Snapshot: 2026-08-12
- image-rs pre-milestone comparison baseline: `ba7afae1f47825d006d0510cdb75d1a239d4543c`
- ImageJ baseline: `4c4975d6df7cf89334e7bc7bb56c48f7b204f244` (ImageJ 1.54u8)

This note compares the native GPUI application with the bundled primary
ImageJ source tree. It measures both routed commands and behavior. A routed
menu command is not automatically compatible: active plane, stack scope, ROI
mask, calibration, dialog state, undo, macro behavior, and result columns are
part of the contract. The catalog deliberately enforces that distinction by
combining declared metadata with the real GPUI router
([source](../src/ui/command_registry.rs#L148-L153)), and its tests reject
capability drift ([source](../src/ui/command_registry.rs#L2428-L2441)).

## Implemented surface

- **Native desktop lifecycle.** The launcher owns application-wide command,
  tool, results, macro, and ROI Manager state, while each image has a stable
  `viewer-N` identity and its own native viewer handle
  ([source](../src/ui/gpui_app.rs#L362-L388),
  [source](../src/ui/gpui_app.rs#L572-L617)). Opening an already-loaded
  normalized path focuses its existing viewer, and closing the launcher exits
  the application ([source](../src/ui/gpui_app.rs#L1314-L1335),
  [source](../src/ui/gpui_app.rs#L1338-L1383)).
- **Core image model and operations.** The model has named X/Y/Z/channel/time
  axes, per-axis spacing and units, channel names, extensible metadata, and
  U8/U16/F32 logical pixel types
  ([source](../src/model/axis.rs#L3-L29),
  [source](../src/model/metadata.rs#L8-L34)). The operation registry exposes a
  large built-in transform/segmentation/measurement surface through one
  injectable service used by the UI, CLI, and workflows
  ([source](../src/commands/registry.rs#L39-L104),
  [source](../src/commands/registry.rs#L112-L205)).
- **ImageJ-style interaction.** The application has the ImageJ tool vocabulary
  and shortcuts, ROI overlays, an internal image clipboard, undo/redo, LUT and
  display-range controls, C/Z/T navigation, persistent Results, and a modeless
  ROI Manager. Those states are represented directly on the image/application
  model rather than as menu-only acknowledgements
  ([source](../src/ui/toolbar.rs#L5-L40),
  [source](../src/ui/gpui_app.rs#L362-L388),
  [source](../src/ui/gpui_app.rs#L572-L609)).
- **Measurement milestone.** `Analyze > Measure` samples the active C/Z/T
  plane and exact ROI, with calibrated geometry and application-wide column
  settings. `Measure Stack` holds C/T and the ROI while iterating Z
  ([source](../src/ui/gpui_app.rs#L3434-L3507),
  [source](../src/ui/gpui_app.rs#L7469-L7498),
  [source](../src/ui/gpui_app.rs#L7559-L7794)).
- **Deterministic automation.** JSON/YAML workflows and CLI commands share the
  operation service, while the desktop macro layer supports recorded/literal
  `run(...)` commands, file/window calls, selection constructors, ROI Manager
  actions, and display-range calls
  ([source](../src/workflow/execute.rs#L13-L84),
  [source](../src/ui/macros.rs#L17-L153)).
- **I/O foundations.** The default eager path reads and writes PNG, JPEG, and
  TIFF, including multi-page grayscale TIFF stacks
  ([source](../src/formats/api.rs#L13-L70),
  [source](../src/formats/tiff.rs#L23-L78)). The optional Bio-Formats adapter
  preserves native layouts and lazy series/resolution/ZCT/region reads over
  application-owned range storage, but only an explicitly requested plane or
  region is materialized into the core F32 dataset
  ([source](../src/formats/bioformats.rs#L184-L211),
  [source](../src/formats/bioformats.rs#L251-L320)).
- **Plugin foundation.** Schema-1 packages are discovered, bounded, validated
  as WebAssembly Components, and indexed with namespaced operations, handlers,
  and commands. The versioned WIT contract now adds stateful image-operation,
  bounded command-handler, and combined worlds plus host-side buffer, mask,
  schedule, metadata, measurement, replacement, and progress validation.
  Discovery intentionally does not instantiate or execute code
  ([source](../src/plugins/mod.rs#L1-L5),
  [source](../src/plugins/contract.rs),
  [source](../wit/image-rs-plugin.wit),
  [source](../src/plugins/mod.rs#L77-L97),
  [source](../src/plugins/mod.rs#L121-L203)). The runtime operation registry
  already accepts dynamically owned names and rejects collisions
  ([source](../src/commands/registry.rs#L39-L95)).

## Command-surface snapshot

The checked-in menu manifest contains 251 leaf entries. At this baseline, 189
route through GPUI and 62 are reported unavailable. One unavailable entry is
the intentional `(empty)` Open Recent placeholder, leaving 61 substantive
unrouted entries. The source of truth is the checked-in manifest
([source](../src/ui/menu/imagej-menu-manifest.json)) plus the real routing
predicate ([source](../src/ui/gpui_app.rs#L8257-L8349)); the catalog test keeps
the two synchronized ([source](../src/ui/command_registry.rs#L2407-L2441)).

| Menu | Unavailable |
| --- | ---: |
| File | 5 |
| Edit | 5 |
| Image | 42 |
| Process | 4 |
| Analyze | 3 |
| Plugins | 1 |
| Window | 2 |

This is a compatibility subset, not a complete inventory of ImageJ's menus,
runtime plugin contributions, macro language, or public Java API. In
particular, a routed acknowledgement is weaker than native behavior.

## Remaining gaps, ordered by risk

### 1. Processing scope and semantics

This is the largest correctness gap. With one Analyze Particles exception,
the GPUI dispatcher sends the complete dataset to an operation and replaces
the complete dataset with its output
([source](../src/ui/gpui_app.rs#L2594-L2648)). An active-plane extractor
already exists, but it is not the common operation path
([source](../src/ui/gpui_app.rs#L8211-L8239)). As a result, a routed 2D
command may process every C/Z/T plane even when an ImageJ user expects the
current plane or an explicit “Process Stack?” choice, and ordinary operations
do not consistently preserve pixels outside an irregular ROI.

ImageJ makes this host behavior explicit: `IJ.setupDialog` chooses current
slice versus the full stack, records `slice`/`stack`, and warns that stack
processing has no undo
([primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/IJ.java#L1245-L1293)).
Its filter runner installs the active area ROI and restores pixels outside an
irregular mask for filters that declare masking support
([primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/plugin/filter/PlugInFilterRunner.java#L205-L275)).
ImageJ's Smooth/Sharpen/Edges/Noise filter uses exactly that scope machinery
([primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/plugin/filter/Filters.java#L18-L47)).

Threshold also remains a destructive dataset operation in the GPUI route
([source](../src/ui/gpui_app.rs#L8351-L8363)), rather than ImageJ's modeless
threshold state that downstream analysis and a separate Apply action can
consume
([primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/plugin/frame/ThresholdAdjuster.java#L16-L50)).

### 2. File, import, and export workflows

The default image-rs codec recognizes only PNG, JPEG, and TIFF
([source](../src/formats/api.rs#L13-L70)). ImageJ's core opener also recognizes
DICOM, FITS, PGM, GIF, BMP, LUT, ROI, ZIP, text/table, AVI, and raw paths and can
delegate unknown formats to plugins
([primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/io/Opener.java#L25-L45),
[primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/io/Opener.java#L308-L360)).
ImageJ also has first-class image-sequence, raw, URL, virtual-stack, LUT/ROI,
and table import commands plus a broad Save As surface
([primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/IJ_Props.txt#L25-L68)).
Its sequence importer supports filtering, numeric/metadata sort, start/count/
step/scale, type conversion, separate images, and virtual stacks
([primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/plugin/FolderOpener.java#L49-L108)).

Image sequence, raw, URL, text-image, LUT/ROI imports, persisted Open Recent,
several exports, and consistent Bio-Formats entry points are therefore still
absent from the GPUI/CLI workflow.

### 3. Plugin and macro ecosystem

The plugin catalog currently retains component paths and handler exports as
private future-runtime data and exposes metadata only
([source](../src/plugins/mod.rs#L77-L97)). No catalog call site adapts a
component export into `OperationRegistry`, so validated packages cannot yet
change an image or handle a command.

ImageJ's minimum plugin contract separates general commands
(`PlugIn.run(String)`) from image filters
([primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/plugin/PlugIn.java#L3-L11),
[primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/plugin/filter/PlugInFilter.java#L5-L36)).
The filter capability word covers pixel kinds, stacks, ROI masks, no-change/
no-undo behavior, snapshots, parallel work, thresholds, and final processing
([primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/plugin/filter/PlugInFilter.java#L38-L93)); the extended lifecycle adds parameter
dialogs, preview, progress passes, cancellation, and a final callback
([primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/plugin/filter/ExtendedPlugInFilter.java#L13-L31)).

The macro gap is larger than command-label coverage. image-rs explicitly parses
only literal, command-oriented statements and acknowledges expression-bearing
calls for a future interpreter ([source](../src/ui/macros.rs#L17-L21),
[source](../src/ui/macros.rs#L45-L153)). ImageJ has a recursive-descent
interpreter with variables, functions, arrays, conditionals, loops, return
values, batch-mode images, and a large built-in function surface
([primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/macro/Interpreter.java#L16-L93),
[primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/macro/Interpreter.java#L593-L719)).

### 4. Data-model fidelity

All core pixels are stored in one `ArrayD<f32>` even when metadata records U8
or U16 ([source](../src/model/dataset.rs#L5-L19),
[source](../src/model/dataset.rs#L39-L57)). ImageJ keeps native byte, unsigned
short, float, indexed-color, and packed RGB representations
([primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/ImagePlus.java#L30-L45),
[primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/ImageStack.java#L54-L87)).
ImageJ also models composite-channel display/LUT state
([primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/CompositeImage.java#L10-L38)), disk-resident virtual stacks
([primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/VirtualStack.java#L12-L51)), and spatial, temporal, origin, density-curve, and value-unit calibration
([primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/measure/Calibration.java#L5-L75)).

Composite images, indexed color models, virtual datasets, calibration curves,
and metadata round-tripping remain partial. The lazy Bio-Formats API is a good
foundation, but the ordinary UI dataset still eagerly materializes F32 data.

### 5. Interaction, concurrency, and verification

Some advertised tools are selection-only or no-op shells. The viewer currently
creates the same single-point selection for Point, Wand, and Text; Hand,
Dropper, and More do nothing on canvas mouse-down
([source](../src/ui/gpui_app.rs#L3085-L3128)). ImageJ distinguishes those tools
and supports multiple line/point/oval/rectangle modes plus installable toolbar
tools
([primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/gui/Toolbar.java#L24-L59),
[primary source](https://github.com/imagej/ImageJ/blob/4c4975d6df7cf89334e7bc7bb56c48f7b204f244/ij/gui/Toolbar.java#L166-L244)).
Window Tile/Cascade are intentionally left to the desktop instead of arranging
viewers ([source](../src/ui/gpui_app.rs#L2842-L2844)). Long reads and operation
execution are synchronous on the application path
([source](../src/ui/gpui_app.rs#L1338-L1383),
[source](../src/ui/gpui_app.rs#L2619-L2641)).

The codebase has strong unit coverage for operation kernels and the completed
measurement slice, but semantic parity still needs fixture/oracle tests against
the bundled ImageJ baseline. Routed-command counts alone cannot catch a wrong
plane, mask, calibration, or undo boundary.

## Implemented milestone: freeze the plugin WIT contract

This change completes the design seam deliberately created by the plugin
foundation: it defines and validates version `0.1.0` of the
WebAssembly Component interface plus its sandbox policy. The catalog already
advertises that API version while keeping component exports private until a
safe adapter exists ([source](../src/plugins/mod.rs#L30-L31),
[source](../src/plugins/mod.rs#L44-L54),
[source](../src/plugins/mod.rs#L77-L97)). A stable contract is therefore the
smallest coherent increment; runtime execution comes after the interface has
parser, code-generation, invariant, and compatibility tests.

The WIT contract is **host-scoped and plane-oriented**, not a raw export of
`Dataset<f32>`, so the new public compatibility seam does not bake in the
whole-dataset scope bug described above. It models the durable parts of
ImageJ's filter lifecycle without reproducing its Java ABI:

1. A filter descriptor declares accepted pixel representations, whether it
   changes pixels, ROI-mask support, and its supported plane/stack scopes.
2. `begin` receives validated parameters, bounded metadata, the active C/Z/T
   position, and a host-selected scope; it returns opaque per-invocation state.
3. `process-plane` receives a bounded row-major full plane, its C/Z/T position,
   calibration, ROI bounds, and an optional exact mask. The future adapter
   preserves excluded pixels, owns undo, and validates every scheduled plane.
4. `finish` consumes the invocation. The adapter must then recheck
   cancellation and atomically commit all staged pixels, metadata,
   measurements, and status. One cumulative output budget spans every plane
   result and `finish`; pixel-changing final processing is outside v0.1.
5. General command handlers use a separate world from pixel operations, with a
   combined world for packages that declare both. Handlers receive structured
   parameters and results, not application internals.

The accompanying sandbox policy states that components have no ambient
filesystem, network, environment, clock, random, or process capability;
datasets stay host-owned; all buffers and metadata are bounded; and future
runtime limits include memory, fuel, and epoch/time budgets. The existing
discovery limits are not execution limits
([source](../src/plugins/manifest.rs#L18-L29)).

### Delivered acceptance criteria

- The checked-in WIT package defines shared error/metadata/measurement types,
  separate stateful image-operation and bounded command-handler worlds, and a
  combined world for packages that contribute both shapes.
- Ordinary test builds generate Rust guest bindings for all three worlds.
  `wit-parser` verifies the package and version, while a normalized semantic
  fingerprint prevents an unchanged `0.1.0` from silently changing ABI types,
  field order, ownership, imports, or exports.
- Tests prove exact world import/export surfaces and reject unsupported scopes,
  invalid or duplicate plane schedules, malformed or oversized U8/U16/F32
  full planes, non-binary ROI masks, position-changing replacements, changed
  metadata shapes, unschedulable axes, cumulative output over-budget,
  unbounded JSON/measurements/text, and non-monotonic progress.
- The contract carries C/Z/T positions, calibrated axes, structured JSON
  parameters, measurements, cancellation, and structured errors without host
  pointers or raw application state.
- The sandbox policy explicitly denies ambient capabilities and names the
  memory, fuel, and epoch/time budgets the Wasmtime adapter must enforce.

### Explicit non-goals

This contract milestone does not instantiate components, register plugin
operations, expose dynamic menu entries, install/update packages, provide
preview UI, stream tiled/virtual datasets, claim Java/JAR binary compatibility,
or interpret the full ImageJ macro language. Those are later adapters and
policies, not hidden requirements of the ABI definition.

## Recommended sequence after that milestone

1. Add a capability-limited Wasmtime Component adapter and prove one fixture
   image operation end to end through `OperationRegistry`, `OpsService`, and a
   CLI/workflow test.
2. Add plugin command/menu adapters and enable/disable/reload state behind the
   same capability boundary.
3. Restore File/Open parity: image sequence, raw, URL, Open Recent, export, and
   Bio-Formats entry points.
4. Apply the host-owned active-plane/ROI/stack scope adapter to built-in 2D
   operations and keep explicit 3D operations separate.
5. Repair tool/shortcut/window fidelity, move I/O and processing off the GPUI
   thread, and expand macro compatibility behind stable interfaces.
