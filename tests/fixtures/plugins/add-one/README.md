# Add-one Component fixture

This package is a checked-in, no-WASI `image-operation-plugin` component used
by the runtime adapter's integration tests. Its manifest exposes seven selectors:

- `add-one` increments every U8, little-endian U16, or little-endian F32 sample
  on every host-scheduled plane. Each plane returns one measurement row whose
  integer `plane` column is the 1-based invocation order. It also reports the
  same monotonic completed count against the fixed `begin.plane-count` total.
  A successful `finish` reports that same completed total with a distinct
  `finish` message, returns the status `add-one complete`, and applies an
  idempotent X-axis calibration update used to exercise metadata publication.
- `fail-finish` returns the same replacements, rows, and progress updates, then
  returns an `internal` error from `finish`. This lets tests prove that the host
  does not commit staged pixels or measurements when finalization fails.
- `spin`, `grow-memory`, `bad-progress`, and `bad-replacement` deliberately
  violate execution limits or the host contract so adversarial wiring is
  exercised end to end. They are test-only selectors, not examples to copy.
- `needs-roi` requires an exact area ROI on every scheduled plane. It modifies
  the full replacement deliberately so tests can verify that the host restores
  every non-member pixel before committing.

Successful `finish` also returns a labeled summary row covering number,
boolean, text, and missing values; the per-plane row covers integers.

All seven selectors advertise `active-plane`, `z-stack`, and `all-planes`;
`needs-roi` additionally declares its required exact-mask input. Every selector
preserves each plane's dimensions, C/Z/T position, sample type, and byte length
unless its name describes the intentional violation. The guest is freestanding:
it uses the small bump allocator in `src/support.c`, links no libc, and imports
only the versioned image-rs Component contract. It is a test fixture rather than
a general plugin SDK example.

## Rebuild

The checked-in binary is generated from the repository's
`wit/image-rs-plugin.wit` with:

- `wit-bindgen-cli 0.51.0`
- `wasm-tools 1.244.0`
- Clang 22.1.8 with the `wasm32-unknown-unknown` target

From this directory, run:

```sh
WIT_BINDGEN=/path/to/wit-bindgen \
WASM_TOOLS=/path/to/wasm-tools \
CLANG=/path/to/clang \
./rebuild.sh
```

The script generates bindings and intermediate core Wasm in a fresh temporary
directory, componentizes without a WASI adapter, validates the result, rejects
any `wasi:` import, and replaces only `add-one.component.wasm`. It prints the
component's decoded WIT at the end; the root world imports only
`image-rs:plugin/types@0.1.0` and `image-rs:plugin/host@0.1.0`, and exports
`image-rs:plugin/image-operation@0.1.0`. Rebuilds reject tool-version drift;
CI consumes the checked-in binary and does not need a WebAssembly C toolchain.
