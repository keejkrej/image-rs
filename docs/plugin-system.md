# Plugin system architecture

## Status

The first plugin milestone is a safe package catalog. `PluginCatalog::discover`
finds, parses, validates, and indexes declarative operation, handler, and
command contributions. The second milestone freezes the guest-facing
`image-rs:plugin@0.1.0` WIT package and the host-side buffer, mask, schedule,
metadata, measurement, replacement, payload, and progress invariants. Tests
generate Rust guest bindings for all three contract worlds.

The system still deliberately does **not** instantiate or execute external
code. A `.wasm` file is checked for location and regular-file status, then
fully validated as WebAssembly Component Model binary encoding. Core Wasm
modules are rejected.

The host operation path is ready for the next layer: `OperationRegistry`
accepts runtime-owned identifiers and is injected through `OpsService`,
`AppContext`, workflows, and the GPUI launcher. The catalog still cannot turn a
declared operation into an `Operation`; the Wasmtime and contribution adapters
are the next layer behind the now-versioned contract.

This separates two decisions that classic ImageJ combines: what a plugin
contributes, and how its code is loaded. Keeping execution behind the plugin
module leaves one seam for a future WebAssembly Component adapter and avoids
making native dynamic libraries part of the default trust model.

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
operation selectors remain private to the future sandbox adapter; UI and
workflow callers see only qualified contribution identities.

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
3. The future host adapter calls `process-plane` in a checked deterministic
   schedule with bounded owned full planes and an optional exact area mask.
4. `finish` consumes the invocation. The adapter then checks cancellation and
   validates every lifted value before atomically committing staged pixels,
   metadata, measurements, and status. Errors leave application state intact.

This preserves the useful ImageJ `PlugInFilter` lifecycle without copying its
integer flags or handing a guest the active image object. Dataset selection,
plane/stack scheduling, ROI masking, undo, preview UI, and commit remain future
host-adapter behavior. Version 0.1 replacements must preserve the input width,
height, C/Z/T position, pixel representation, and byte count. Planes and masks
are row-major with X fastest; U16 and F32 samples are little-endian.
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
  bytes. ROI masks use exactly one validated zero-or-one byte per pixel.
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
  access, or WebAssembly instantiation.
- The only callable guest imports are the contract host's monotonic progress
  and cooperative cancellation functions; the other local import carries
  shared types only. There are no filesystem, network, process, environment,
  clock, random, or other WASI imports. The Wasmtime adapter must preserve that
  denial and additionally enforce memory, fuel, and epoch/time limits.
- Tests parse and generate Rust bindings for all three worlds. A normalized
  semantic fingerprint freezes record fields, variants, function signatures,
  resource ownership, imports, and exports for API `0.1.0`.
- The resolved module path and operation/handler exports remain private
  implementation details. Callers see a small catalog interface; the eventual
  sandbox adapter can execute through that same module without leaking loader
  mechanics into the UI, CLI, or workflow layers.

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

1. Add a Wasmtime Component adapter with no ambient filesystem, network, or
   process capabilities; enforce memory, fuel, and epoch/time limits.
2. Adapt validated plugin operations into the runtime operation registry, then
   adapt plugin commands into the shared application command/menu catalog.
3. Add install/update staging, archive integrity/signature policy, persistent
   enable/disable state, and atomic catalog reloads.
4. Extend the contract deliberately for previews, shape-changing/new-image
   outputs, tiled virtual datasets, and richer command actions. Add
   capabilities only when a concrete adapter or use case makes the seam real.
