# Plugin system architecture

## Status

The first plugin milestone added a safe package catalog. `PluginCatalog::discover`
finds, parses, validates, and indexes declarative operation, handler, and
command contributions. The second froze the guest-facing
`image-rs:plugin@0.1.0` WIT package and the host-side buffer, mask, schedule,
metadata, measurement, replacement, payload, and progress invariants. The
third executes compatible image-operation Components through Wasmtime. The
fourth adds the shared scoped invocation and GPUI commit lifecycle.

`PluginCatalog::register_operations` compiles each discovered package once,
checks every declared operation's capabilities, and atomically adds compatible
scoped adapters to an `OperationRegistry`. Registered operations therefore use
the same `OpsService` path as built-in operations. Each invocation gets a fresh
Store, Component instance, and guest resource; the runtime exposes only the
contract's progress and cancellation imports and does not link WASI. This is
still a host/library registration API: neither the CLI nor the GPUI application
automatically discovers a plugin directory or calls the registration method.

`OpsService::describe` returns the operation schema, supported scopes, and
whether an exact area mask is unsupported, optional, or required.
`OpsService::invoke` accepts an owned dataset snapshot, parameters, active C/Z/T
position, selected scope, optional mask, and caller-owned `ExecutionControl`.
Its result makes the dataset effect explicit (`Unchanged` or a replacement
that identifies both the source and staged result) alongside measurements and
status. The scopes deliberately separate two kinds of execution:

- `WholeDataset` invokes an n-dimensional operation once. It is not an alias
  for processing every 2D plane; current whole-dataset descriptors do not
  accept an area mask.
- `ActivePlane` processes exactly the requested C/Z/T plane.
- `ZStack` holds the active C/T position and visits every Z plane.
- `AllPlanes` visits every C/Z/T plane with C fastest, then Z, then T.

The native plane adapter gathers a full calibrated X/Y plane so neighborhood
kernels retain surrounding context, then scatters only exact mask members into
an isolated clone. The Wasm adapter uses the same host-authoritative schedule
and passes the exact binary mask to the guest, but also ignores every returned
pixel outside the mask. That defensive scatter preserves source pixels even
if an untrusted component modifies its entire replacement plane. Both adapters
check caller cancellation between planes and report structured plane progress;
validated guest progress is forwarded as detail on the same progress sink.
Wasm cancellation also interrupts non-cooperative guest execution on an epoch
tick. A cancellation, trap, structured guest error, timeout, or contract
violation discards the staged clone.

The legacy `OpsService::execute` and workflow path retains each built-in's
original whole-dataset behavior for compatibility. Callers that need explicit
active-plane or stack semantics use `describe` plus `invoke`. Cancellation of
a legacy whole-dataset native call is checked before and after that call; its
result remains atomic, but an individual long native call cannot yet be
preempted cooperatively.

GPUI processing uses operation descriptors to choose `ActivePlane` by default
and offers `Process stack` when the descriptor supports `ZStack`. Rectangle,
oval, polygon, and freehand selections are rasterized to exact masks before
dispatch. Invocation runs on a background executor. Escape, viewer close, and
quit signal the caller token; completion is accepted only when the original
viewer and dataset revision still match. The candidate plane is rendered
before mutation; completion publishes the dataset, measurements, and status,
with exactly one undo entry when pixels were replaced. The GPUI currently
updates its progress bar from host plane counts and may show validated guest
progress messages. A single overwriteable slot retains only the latest available
update so an operation cannot queue unbounded UI work.
Recorded macro operation steps enter the same path one at a time, preserving
active-plane versus active-Z-stack scope while retaining macro statement order.

This separates two decisions that classic ImageJ combines: what a plugin
contributes, and how its code is loaded. Execution remains behind the plugin
module, and native dynamic libraries are not part of the default trust model.
Command-handler execution, application command/menu contributions, automatic
discovery, installation and updates, and interactive operation previews remain
future work.

## Package format (schema 1)

Each immediate child of the configured plugin root may contain
`image-rs-plugin.toml`:

```toml
schema_version = 1
id = "org.example.contrast"
name = "Example Contrast"
version = "1.0.0"
description = "Example plugin"
authors = ["Image Scientist"]

[runtime]
kind = "wasm-component"
api_version = "0.1.0"
path = "extension.wasm"

[[operations]]
id = "stretch"
description = "Stretch image contrast."
export = "stretch"

[[operations.params]]
name = "minimum"
description = "Lower display bound."
required = false
kind = "float"

[[commands]]
id = "stretch"
label = "Stretch Contrast..."
menu_path = ["Plugins", "Examples"]
target = { kind = "operation", id = "stretch" }
default_params = { minimum = 2.5 }
```

ImageJ-style commands that are not dataset operations use a separately declared
handler. A command-only package is valid without an `operations` entry:

```toml
[[handlers]]
id = "about"
description = "Show plugin information."
export = "show-about"

[[commands]]
id = "about"
label = "About Example..."
menu_path = ["Plugins", "Examples"]
target = { kind = "handler", id = "about" }
default_params = { argument = "credits" }
```

The tagged target makes operation and handler dispatch unambiguous. A manifest
`export` is a component-local entrypoint selector passed to the fixed WIT
dispatcher; it is not an arbitrary Component Model export name. Handler and
operation selectors remain private to their sandbox adapters; UI and workflow
callers see only qualified contribution identities.

Manifest fields reject unknown keys so authoring mistakes fail locally instead
of being silently ignored. Package discovery is one directory deep and does
not follow symlinked package directories or manifests.

## Component contract (API 0.1.0)

The canonical contract is [`wit/image-rs-plugin.wit`](../wit/image-rs-plugin.wit).
It defines separate `image-operation-plugin` and `command-handler-plugin`
worlds so ImageJ-style image filters do not receive the authority of general
commands, plus a `combined-plugin` world for packages that declare both kinds.

Image operations use an invocation-local lifecycle:

1. `capabilities(entrypoint)` declares accepted pixel representations,
   supported scopes, ROI requirements, masking support, and whether pixels
   change.
2. `begin(entrypoint, request)` receives validated parameters, immutable image
   metadata, and the host-selected scope and active C/Z/T position, then
   returns fresh invocation state.
3. The host adapter calls `process-plane` in a checked deterministic
   schedule with bounded owned full planes and an optional exact area mask.
4. `finish` consumes the invocation. The adapter then checks cancellation and
   validates every lifted value before atomically committing staged pixels,
   metadata, measurements, and status. Errors leave application state intact.

This preserves the useful ImageJ `PlugInFilter` lifecycle without copying its
integer flags or handing a guest the active image object. The registry adapter
implements active-plane, active-Z-stack, and all-plane schedules, exact ROI
masking, caller cancellation/progress, and atomic staged output. GPUI supplies
the viewer-aware background execution, stale-result guard, and undo commit;
interactive previews remain future work.
Version 0.1 replacements must preserve the input width, height, C/Z/T position,
pixel representation, and byte count. Planes and masks are row-major with X
fastest; U16 and F32 samples are little-endian.
Shape-changing commands, tiles, pixel-changing final processing, and new-image
creation are deliberately outside this first filter contract. Image operations
also reject non-singleton unknown axes because v0.1 positions schedule only
X/Y planes across C/Z/T; handlers may still receive unknown-axis metadata.

Command handlers receive identifiers, labels, arguments, JSON parameters, and
an optional immutable active-image summary. They may return status and
measurement rows; they cannot mutate datasets or create native windows.

## Compatibility and identity

- Schema compatibility is exact: this release accepts only schema `1`.
- Plugin and runtime interface versions are SemVer. The host interface is
  `0.1.0`; a plugin must be on the same SemVer compatibility line and may not
  be newer than the host.
- Plugin IDs are lowercase reverse-domain-style identifiers. Operation,
  handler, and command IDs in a manifest are single local identifiers and may
  not contain dots; the catalog exposes them as `<plugin-id>.<local-id>`. This
  makes qualification injective.
- Duplicate local IDs reject the package. Packages are visited in
  lexicographic path order, and the first valid package deterministically owns
  a duplicate plugin ID. A rejected package contributes nothing.
- Commands can target only operations or handlers declared by the same
  package. Menu paths must begin with `Plugins`, preventing a plugin from
  injecting entries into core menus without a future explicit capability.

## Safety invariants

- Runtime paths must be normalized, relative `.wasm` paths. `..`, absolute
  paths, and symlinks that resolve outside the package are rejected.
- Runtime artifacts are parsed and validated with `wasmparser`; a component is
  limited to 64 MiB. A core Wasm module is not a component and is rejected.
- Registering image operations applies a narrower deterministic admission pass
  before `Component::new`: at most 16 MiB per compilable artifact, eight levels
  of embedding, 32 embedded core modules, 16 nested Components, 32 MiB of
  aggregate embedded-binary ranges, and 100,000 aggregate section items. Core
  code is additionally limited to 10,000 defined functions, 12 MiB of function
  bodies, 512 KiB per body, one million decoded operators (50,000 per body), and
  one million expanded locals (100,000 per body). Custom sections are limited
  to 64 entries, 1 MiB each, and 2 MiB total. This catches compact count/local
  expansion and code-complexity bombs before native compilation, not merely
  after Wasmtime starts compiling them.
- Manifests are limited to 256 KiB before parsing. Schema 1 permits at most 32
  authors, 128 operations, 128 handlers, 256 commands, 256 aggregate
  contributions, 64 parameters per operation, and 1024 parameters overall.
- A bad package is reported independently and does not hide valid siblings.
- TOML datetimes and non-finite numbers are rejected from command defaults.
  Operation-target defaults may refer only to declared operation parameters
  and must match one of the core schema kinds (`bool`, `float`,
  `int`/`integer`, `string`, `array`, `array<int>`, or `object`). Handler
  defaults are free-form structured values within the same resource limits.
- Defaults across one package are limited to 8 levels, 1024 value nodes, and
  64 KiB of aggregate scalar/key data.
- Each full plane or ROI mask crossing the component seam is limited to 64 MiB.
  Plane samples are tightly packed U8, little-endian U16, or little-endian F32
  bytes. Because the core stores all samples as f32, finite values in logically
  U8/U16 datasets are clamped and rounded at this typed boundary, matching the
  native image writers; non-finite integer samples are rejected. ROI masks use
  exactly one validated zero-or-one byte per pixel.
- One cumulative 4 MiB output budget covers every staged `process-plane`
  result plus `finish` for an invocation; the limit cannot be multiplied by
  the number of scheduled planes. Handler input/output is bounded per
  invocation as well.
- Host-side contract checks reject empty/out-of-bounds regions, size overflow,
  malformed buffer or mask lengths, C/Z/T schedule drift, non-identical or
  forbidden replacement layouts, changed metadata shapes, invalid calibration
  (including f64 values that cannot narrow to finite positive host f32), or
  JSON, duplicate result columns/properties, non-finite measurements,
  over-budget strings/collections/payloads, regressing progress, and unstable
  progress totals before an adapter can commit output.
- Discovery performs no native library loading, process execution, network
  access, or WebAssembly instantiation. Instantiation occurs only when the host
  explicitly calls `PluginCatalog::register_operations` or invokes a registered
  operation.
- The only callable guest imports are the contract host's monotonic progress
  and cooperative cancellation functions; the other local import carries
  shared types only. There are no filesystem, network, process, environment,
  clock, random, or other WASI imports. The Wasmtime linker preserves that
  denial and rejects Components requiring unavailable imports.
- Each invocation has 100 million fuel units and a five-second epoch deadline,
  a 512 KiB Wasm stack limit, at most one 160 MiB memory, 100,000 table elements
  per table, 64 instances, and 16 tables. Growth failures trap rather than
  weakening the limits.
- One registration call accepts at most 32 executable packages, 256 operations,
  and 256 MiB of component artifacts. Capability probes get 10 million fuel
  units and 500 ms each within a 15-second aggregate registration deadline.
  Wasmtime compilation itself is synchronous and not interruptible, so the
  deadline is observed only between structurally bounded artifacts and probes;
  it is not a compilation deadline or a guarantee of wall-clock compile time.
- Tests parse and generate Rust bindings for all three worlds. A normalized
  semantic fingerprint freezes record fields, variants, function signatures,
  resource ownership, imports, and exports for API `0.1.0`. A freestanding
  Component fixture also exercises active-plane and Z-stack invocation, exact
  masking, caller cancellation, U8/U16/F32 execution, progress, measurement
  rows, status, and failure without WASI.
- The resolved module path and operation/handler exports remain private
  implementation details. Callers see the catalog registration interface and
  scoped operation descriptors without loader mechanics leaking into UI, CLI,
  or workflow layers.
- Component results are lifted into host memory before Rust can apply the
  structural payload validators. The single 160 MiB guest-memory ceiling and
  immediate raw collection prechecks bound that exposure, while process-level
  isolation remains a possible later hardening step for hostile plugins.

## Why this shape

Classic ImageJ discovers class/JAR plugins, reads menu declarations from
`plugins.config`, and then dispatches either `PlugIn.run(String)` or the richer
`PlugInFilter` lifecycle through a custom JVM class loader:

- [ImageJ `PlugIn`](https://github.com/imagej/ImageJ/blob/master/ij/plugin/PlugIn.java)
- [ImageJ `PlugInFilter`](https://github.com/imagej/ImageJ/blob/master/ij/plugin/filter/PlugInFilter.java)
- [ImageJ menu/JAR discovery](https://github.com/imagej/ImageJ/blob/master/ij/Menus.java)
- [ImageJ `PluginClassLoader`](https://github.com/imagej/ImageJ/blob/master/ij/io/PluginClassLoader.java)

The useful compatibility concepts are preserved—discovery, stable command
names, menu placement, operation parameters, and declared processing entry
points—without adopting JVM reflection or unrestricted process-native code.

Zed provides the reference for the Rust-native direction: repository packages
with a versioned manifest, declarative contributions, Rust procedural code
compiled to WebAssembly, a host-owned extension index, and explicit host
capabilities:

- [Zed extension development guide](https://zed.dev/docs/extensions/developing-extensions)
- [Zed manifest implementation](https://github.com/zed-industries/zed/blob/main/crates/extension/src/extension_manifest.rs)
- [Zed extension store/host](https://github.com/zed-industries/zed/blob/main/crates/extension_host/src/extension_host.rs)
- [Life of a Zed Extension: Rust, WIT, Wasm](https://zed.dev/blog/zed-decoded-extensions)

## Next milestones

1. Implement the command-handler world and adapt validated plugin commands into
   the shared application command/menu catalog.
2. Add install/update staging, archive integrity/signature policy, persistent
   enable/disable state, and atomic catalog reloads.
3. Extend the contract deliberately for previews, shape-changing/new-image
   outputs, tiled virtual datasets, and richer command actions. Add
   capabilities only when a concrete adapter or use case makes the seam real.
