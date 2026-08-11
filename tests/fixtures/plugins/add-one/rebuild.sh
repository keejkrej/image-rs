#!/bin/sh
set -eu

fixture_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$fixture_dir/../../../.." && pwd)
wit_bindgen=${WIT_BINDGEN:-wit-bindgen}
wasm_tools=${WASM_TOOLS:-wasm-tools}
clang=${CLANG:-clang}

if [ "$($wit_bindgen --version)" != "wit-bindgen-cli 0.51.0" ]; then
  echo "rebuild requires wit-bindgen-cli 0.51.0" >&2
  exit 1
fi
if [ "$($wasm_tools --version)" != "wasm-tools 1.244.0" ]; then
  echo "rebuild requires wasm-tools 1.244.0" >&2
  exit 1
fi
case "$($clang --version | sed -n '1p')" in
  "clang version 22.1.8"*) ;;
  *)
    echo "rebuild requires Clang 22.1.8" >&2
    exit 1
    ;;
esac

build_dir=$(mktemp -d "${TMPDIR:-/tmp}/image-rs-add-one.XXXXXX")
trap 'rm -rf -- "$build_dir"' EXIT HUP INT TERM

"$wit_bindgen" c \
  --world image-operation-plugin \
  --out-dir "$build_dir" \
  "$repo_root/wit"

"$clang" \
  --target=wasm32-unknown-unknown \
  -std=c11 \
  -O2 \
  -Wall \
  -Wextra \
  -Werror \
  -fno-builtin \
  -nostdlib \
  -I "$build_dir" \
  -I "$fixture_dir/src/include" \
  "$fixture_dir/src/add_one.c" \
  "$fixture_dir/src/support.c" \
  "$build_dir/image_operation_plugin.c" \
  "$build_dir/image_operation_plugin_component_type.o" \
  -Wl,--no-entry \
  -Wl,--export-memory \
  -Wl,--allow-undefined \
  -Wl,--initial-memory=131072 \
  -Wl,--max-memory=536870912 \
  -Wl,--strip-all \
  -o "$build_dir/add-one.core.wasm"

"$wasm_tools" component new \
  "$build_dir/add-one.core.wasm" \
  -o "$build_dir/add-one.component.wasm"
"$wasm_tools" validate "$build_dir/add-one.component.wasm"

component_wit=$($wasm_tools component wit "$build_dir/add-one.component.wasm")
root_world=$(printf '%s\n' "$component_wit" | sed -n '/^world root {$/,/^}$/p')
root_imports=$(printf '%s\n' "$root_world" | grep '^  import ')
expected_imports='  import image-rs:plugin/types@0.1.0;
  import image-rs:plugin/host@0.1.0;'
if [ "$root_imports" != "$expected_imports" ]; then
  echo "generated fixture has imports outside image-rs:plugin@0.1.0" >&2
  printf '%s\n' "$root_imports" >&2
  exit 1
fi
case "$component_wit" in
  *"export image-rs:plugin/image-operation@0.1.0;"*) ;;
  *)
    echo "generated fixture does not export the image-operation contract" >&2
    exit 1
    ;;
esac

cp "$build_dir/add-one.component.wasm" "$fixture_dir/add-one.component.wasm"
printf '%s\n' "$component_wit"
