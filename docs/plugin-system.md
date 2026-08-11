# Plugin system architecture

## Status

The first plugin milestone is a safe package catalog. `PluginCatalog::discover`
finds, parses, validates, and indexes declarative operation, handler, and
command contributions. It deliberately does **not** instantiate or execute
external code. A `.wasm` file is checked for location and regular-file status,
then fully validated as WebAssembly Component Model binary encoding. Core Wasm
modules are rejected.

The host operation path is ready for the next layer: `OperationRegistry`
accepts runtime-owned identifiers and is injected through `OpsService`,
`AppContext`, workflows, and the GPUI launcher. The catalog still cannot turn a
declared component export into an `Operation`; that adapter begins only after
the WIT contract and sandbox policy exist.

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

The tagged target makes operation and handler dispatch unambiguous. Handler
and operation export names are retained privately for the future sandbox
adapter; UI and workflow callers see only qualified contribution identities.

Manifest fields reject unknown keys so authoring mistakes fail locally instead
of being silently ignored. Package discovery is one directory deep and does
not follow symlinked package directories or manifests.

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
- Discovery performs no native library loading, process execution, network
  access, or WebAssembly instantiation.
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

1. Define command-handler and image-operation contracts in WIT. The operation
   world needs owned plane/tile buffers, metadata, parameters, progress,
   cancellation, measurements, and structured errors. Keep datasets host-owned
   and pass bounded buffers rather than raw pointers.
2. Add a Wasmtime Component adapter with no ambient filesystem, network, or
   process capabilities; enforce memory, fuel, and epoch/time limits.
3. Adapt validated plugin operations into the runtime operation registry, then
   adapt plugin commands into the shared application command/menu catalog.
4. Add install/update staging, archive integrity/signature policy, persistent
   enable/disable state, and atomic catalog reloads.
5. Extend the contract deliberately for ImageJ filter concerns: active plane
   versus stack scope, ROI masks, supported pixel types, previews, undo, and
   final processing. Add capabilities only when a concrete second adapter or
   use case makes the seam real.
