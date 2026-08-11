# ImageJ parity ledger

Snapshot: 2026-08-12

This ledger compares the native GPUI application with the bundled ImageJ source tree. A routed menu command is not automatically behavior-compatible: scope, calibration, active plane, ROI, dialog state, macro behavior, and result columns all matter.

## Command surface

The checked-in ImageJ menu manifest currently contains 251 leaf entries. After the measurement milestone, 189 route through GPUI and 62 are reported unavailable. One unavailable entry is the intentional `(empty)` Open Recent placeholder, leaving 61 substantive unrouted entries.

| Menu | Unavailable |
| --- | ---: |
| File | 5 |
| Edit | 5 |
| Image | 42 |
| Process | 4 |
| Analyze | 3 |
| Plugins | 1 |
| Window | 2 |

The manifest is itself a compatibility subset, not a complete inventory of ImageJ's menus, plugin contributions, macro language, or public API.

## Completed native measurement milestone

- `Analyze > Measure` uses the active C/Z/T plane and exact active ROI, with full-plane fallback.
- `Analyze > Set Measurements` controls an application-wide first subset of ImageJ columns—area, mean, standard deviation, min/max, centroid, perimeter, bounding rectangle, integrated density, median, stack position, label, inverted Y—and decimal display/export precision.
- Measurements share calibrated area, centroid, bounds, perimeter, line, point, angle, intensity-statistics, and stack-position logic across Measure, overlays, and ROI Manager.
- `Image > Stacks > Measure Stack` reuses the exact ROI, forces position columns, and appends one row for every Z plane at the active C/T without changing the viewer position.
- Invalid X/Y calibration is rejected rather than silently replaced.

ImageJ's Measure Stack macro can also iterate user-selected C/Z/T axes in six orders. The first native slice deliberately holds C/T and iterates Z; the hyperstack axes/order dialog remains future work.

The remaining Set Measurements options include mode, center of mass, ellipse/shape descriptors, Feret, skewness, kurtosis, area fraction, threshold limiting, redirect images, and scientific notation. This milestone establishes one shared measurement path; it does not claim the entire Analyzer option surface.

## Completed plugin execution milestone

- `image-rs:plugin@0.1.0` defines separate Component Model worlds for image operations and command handlers plus a combined world for mixed packages.
- Image operations use invocation-local `begin` → repeated `process-plane` → consuming `finish` state, matching the durable part of ImageJ's filter lifecycle without exposing the host dataset.
- The contract assigns C/Z/T scheduling, exact ROI masking, undo, cancellation, staging, and atomic commit to the future host adapter. Components receive only bounded owned full U8/U16/F32 planes and calibrated metadata.
- Named capabilities replace ImageJ's integer filter flags; the host chooses scope per invocation, and v0.1 replacement planes must preserve dimensions, position, representation, and byte count.
- The only callable guest imports are monotonic progress and cancellation. Ambient filesystem, network, process, environment, clock, random, and WASI capabilities are absent.
- Contract tests parse, version-check, code-generate, and fingerprint the WIT package, then verify plane schedules, buffers, ROI masks, replacement layouts, metadata/JSON/measurement budgets, and progress invariants.
- The host-facing `PluginCatalog::register_operations` library API now compiles image-operation Components with Wasmtime, capability-checks them, and atomically adapts them into the ordinary operation registry.
- Every invocation gets fresh guest state, fixed memory, table, stack, fuel, and epoch/time limits, and only the contract host imports. The host stages pixels and metadata in a cloned dataset and publishes them only after the consuming `finish` call succeeds.
- Registered operations preserve validated guest status and ordered measurement rows through `OpsService` and workflow reports. Structured guest errors, traps, and contract failures do not expose partial output.

The current generic operation interface has no active C/Z/T or ROI context, so
the adapter deliberately accepts only `all-planes` operations that do not
require an ROI. Command-handler execution, dynamic menus, install/update
packages, automatic CLI/GPUI discovery and registration, and UI-aware
active-plane, stack, and exact-ROI invocation remain future work. The
command-surface count is therefore unchanged.

## Largest remaining gaps

1. **Processing scope and semantics.** Several routed operations still process an entire dataset where ImageJ acts on an active plane or ROI. Threshold is destructive instead of modeless state, and ordinary smoothing/binary operations need explicit 2D versus 3D behavior.
2. **File and import workflows.** Image sequence, raw, URL, text-image, LUT/ROI imports, Open Recent, and several export flows are absent from GPUI. Optional Bio-Formats support is not yet exposed consistently through UI and CLI.
3. **Image/stack workflows.** Channels, stack composition, hyperstack tools, transforms, overlays, and lookup-table commands account for most unavailable menu entries.
4. **Plugin and macro ecosystem.** Capability-compatible image-operation Components can now execute through the registry, but command handlers, dynamic menu exposure, installation and dependency management, and UI-aware active-plane/ROI invocation remain absent. The macro layer parses a useful literal-command subset, not the ImageJ language/runtime.
5. **Data model fidelity.** The core stores `f32` samples and exposes a smaller pixel-type set than ImageJ. Composite images, color models, virtual stacks, calibration curves, and metadata round-tripping remain partial.
6. **Interaction fidelity.** Several advertised tools and shortcuts are incomplete, some Window behavior differs from ImageJ, and long I/O/processing work still blocks the UI thread.
7. **Architecture and verification.** GPUI application state is concentrated in one large module and lacks deep lifecycle/window tests. Compatibility needs oracle tests against the bundled ImageJ behavior, not only route-count growth.

## Recommended sequence

1. Introduce a reusable active-plane/ROI processing scope adapter and separate explicit 3D operations, then use it for built-in and plugin operations.
2. Restore File/Open parity: image sequence, raw, URL, Open Recent, export, and Bio-Formats entry points.
3. Add plugin command-handler/menu adapters and a staged install/update policy.
4. Repair tool/shortcut/window fidelity and move I/O and processing off the GPUI thread.
5. Expand macro and plugin compatibility incrementally behind stable command, dataset, and results interfaces.
