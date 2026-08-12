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
- The contract assigns C/Z/T scheduling, exact ROI masking, cancellation, staging, and application commit to host adapters. Components receive only bounded owned full U8/U16/F32 planes and calibrated metadata.
- Named capabilities replace ImageJ's integer filter flags; the host chooses scope per invocation, and v0.1 replacement planes must preserve dimensions, position, representation, and byte count.
- The only callable guest imports are monotonic progress and cancellation. Ambient filesystem, network, process, environment, clock, random, and WASI capabilities are absent.
- Contract tests parse, version-check, code-generate, and fingerprint the WIT package, then verify plane schedules, buffers, ROI masks, replacement layouts, metadata/JSON/measurement budgets, and progress invariants.
- The host-facing `PluginCatalog::register_operations` library API now compiles image-operation Components with Wasmtime, capability-checks them, and atomically adapts them into the scoped operation registry.
- Every invocation gets fresh guest state, fixed memory, table, stack, fuel, and epoch/time limits, and only the contract host imports. The host stages pixels and metadata in a cloned dataset and publishes them only after the consuming `finish` call succeeds.
- Registered operations preserve validated guest status and ordered measurement rows through `OpsService` and workflow reports. Structured guest errors, traps, and contract failures do not expose partial output.

Command-handler execution, dynamic menus, install/update packages, and
automatic CLI/GPUI plugin discovery and registration remain future work. The
command-surface count is therefore unchanged.

## Completed scoped invocation milestone

- `OpsService::describe` exposes each operation's schema, supported execution
  scopes, and area-mask policy. `OpsService::invoke` owns the source snapshot,
  active C/Z/T position, exact mask, parameters, and execution control, and
  returns an explicit unchanged/replaced dataset effect with measurements and
  status.
- `WholeDataset` is a single n-dimensional invocation. `ActivePlane`, `ZStack`,
  and `AllPlanes` are host-scheduled 2D invocations; Z-stack holds C/T, while
  all-planes visits C fastest, then Z, then T. Explicit 3D and shape-changing
  operations remain whole-dataset operations instead of being silently sliced.
- Selected same-shape native 2D operations use a common plane adapter. It gives
  kernels the full X/Y plane for neighborhood context and scatters only exact
  rectangle, oval, polygon, or freehand mask members into staged output.
- Sandboxed Wasm operations now advertise and receive their compatible scope
  and optional/required exact area mask. Host-side scatter defensively retains
  source pixels outside the mask even if the guest replaces the entire plane.
- Native and Wasm adapters stage a clone, validate output before exposure, check
  a caller cancellation token, and report plane progress through one control
  interface. Guest progress is validated and forwarded; epoch interruption also
  reaches a guest that does not call the cooperative cancellation import.
- Legacy CLI/workflow `execute` calls retain built-ins' prior whole-dataset
  behavior. Their output is still discarded after cancellation, but a single
  long native call is not yet cooperatively interruptible.
- GPUI runs shared operation invocation off the UI thread. Plane-wise commands
  default to the active C/Z/T plane and add `Process stack` for eligible Z
  stacks. Escape, viewer close, and quit request cancellation.
- Macro operation steps use the same background invocation path sequentially;
  recorded `Process stack` options replay as the active C/T Z stack instead of
  silently falling back to whole-dataset execution.
- Completion rechecks the viewer ID, source `Arc`, and monotonic dataset
  revision, renders the candidate before mutation, and discards stale or
  unrenderable results. A successful pixel replacement publishes one dataset
  swap and exactly one undo entry; failure or cancellation publishes no pixel
  change or undo entry.

This is a reusable execution and commit boundary, not a claim that every
routed command now has ImageJ-identical processing semantics. The GPUI exposes
host-authoritative plane progress and validated guest detail through a bounded
latest-update slot; interactive processing previews remain future work.

## Largest remaining gaps

1. **Processing scope and semantics.** The shared adapter now scopes selected same-shape 2D operations, but remaining commands still need an ImageJ semantic audit. Threshold is destructive instead of modeless state, and additional smoothing/binary operations need explicit 2D versus 3D decisions.
2. **File and import workflows.** Image sequence, raw, URL, text-image, LUT/ROI imports, Open Recent, and several export flows are absent from GPUI. Optional Bio-Formats support is not yet exposed consistently through UI and CLI.
3. **Image/stack workflows.** Channels, stack composition, hyperstack tools, transforms, overlays, and lookup-table commands account for most unavailable menu entries.
4. **Plugin and macro ecosystem.** Capability-compatible image-operation Components can execute through the scoped registry, but command handlers, dynamic menu exposure, automatic discovery, installation, and dependency management remain absent. The macro layer parses a useful literal-command subset, not the ImageJ language/runtime.
5. **Data model fidelity.** The core stores `f32` samples and exposes a smaller pixel-type set than ImageJ. Composite images, color models, virtual stacks, calibration curves, and metadata round-tripping remain partial.
6. **Interaction fidelity.** Several advertised tools and shortcuts are incomplete, some Window behavior differs from ImageJ, long I/O paths can still block the UI thread, and legacy whole-dataset kernels are not cooperatively interruptible.
7. **Architecture and verification.** GPUI application state is concentrated in one large module and lacks deep lifecycle/window tests. Compatibility needs oracle tests against the bundled ImageJ behavior, not only route-count growth.

## Recommended sequence

1. Restore the next bounded File/Open import-export slice: image sequence, raw,
   URL, and real Open Recent entry points, followed by the missing export and
   optional Bio-Formats UI/CLI entry points.
2. Audit remaining processing commands against ImageJ, moving same-shape 2D
   kernels onto the scoped adapter while keeping explicit 3D/shape-changing
   behavior whole-dataset.
3. Add plugin command-handler/menu adapters and a staged install/update policy.
4. Repair remaining tool/shortcut/window fidelity and move long I/O and legacy
   processing paths off the GPUI thread.
5. Expand macro and plugin compatibility incrementally behind stable command,
   dataset, and results interfaces.
