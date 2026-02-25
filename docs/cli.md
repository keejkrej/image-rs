# CLI Guide

Binary: `ijrs`

## Commands

### `ijrs info <input>`

Reads image metadata and prints JSON summary.

Example:

```bash
ijrs info ./fixtures/samples/cells.tiff
```

### `ijrs convert <input> <output>`

Converts between supported formats (`tif/tiff/png/jpg/jpeg`).

Example:

```bash
ijrs convert ./input.tiff ./output.png
```

### `ijrs run --input <path> --recipe <path> --output <path> [--report <path>]`

Runs deterministic operation pipelines from JSON/YAML recipe files.

Example:

```bash
ijrs run \
  --input ./input.tiff \
  --recipe ./fixtures/recipes/normalize-threshold.json \
  --output ./output.tiff \
  --report ./report.json
```

### `ijrs ops list`

Prints available operations and parameter schemas.

Example:

```bash
ijrs ops list
```

### `ijrs view <input>`

Launches the Tauri viewer (`ijrs-tauri`) and opens the given input.

Example:

```bash
ijrs view ./input.tiff
```

## Recipe schema

```json
{
  "name": "optional pipeline name",
  "operations": [
    {
      "op": "intensity.normalize",
      "params": {}
    },
    {
      "op": "threshold.fixed",
      "params": { "threshold": 0.5 }
    }
  ]
}
```
