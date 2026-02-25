# Tauri UI Guide

App: `ijrs-tauri`

## Window model

The desktop UX uses one process with two window roles:

- Control window (`main`): small persistent launcher/drop target.
- Viewer windows (`viewer-*`): one image per file, focused on duplicate open.

## ImageJ2/Fiji-style interaction baseline

- Control window stays open and accepts drag/drop.
- Each image opens as an independent viewer window.
- Viewer centers around image canvas + status bar.
- Operations run from a process dialog (`Preview` / `Apply`) instead of always-live side panel updates.

## Runtime behavior

- Viewer sessions are bound to window labels and keep datasets in memory.
- Frame rendering uses a single command path (`viewer_frame`) with caching.
- Long-running operations execute in background tasks and emit `viewer-op-event`.
- Job cancellation is generation-based: stale or cancelled jobs are ignored when they finish.

## Commands

- `open_images(paths: Vec<String>) -> OpenResult`
- `viewer_init(window) -> ViewerInit`
- `viewer_frame(window, request) -> ViewerFrame`
- `viewer_start_op(window, request) -> JobTicket`
- `viewer_cancel_op(window, request) -> ()`
- `export_preview(window, request) -> ()`

Compatibility commands are still registered for one milestone:

- `viewer_context(window) -> ViewerContext`
- `render_slice(window, request) -> data-url-png`
- `slice_histogram(window, request) -> Vec<u32>`
- `slice_plane(window, request) -> SlicePlane`

## Event payload

`viewer-op-event` emits:

- `job_id`
- `mode` (`preview` | `apply`)
- `op`
- `status` (`completed` | `failed` | `cancelled`)
- `message` (optional)

## Invoke fallback

Frontend command calls go through `ui/tauri-api.js`, which checks:

1. `window.__TAURI__.core.invoke`
2. `window.__TAURI__.invoke`
3. `window.__TAURI_INTERNALS__.invoke`

If none exist, the UI reports runtime diagnostics instead of failing silently.
