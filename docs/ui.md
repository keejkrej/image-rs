# Tauri UI Guide

App: `ijrs-tauri`

Frontend stack: `React + TypeScript + Vite + Tailwind + shadcn/ui` in `apps/ijrs-tauri/web`.

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
- Frame rendering uses a single command path (`viewer_frame_buffer`) with caching.
- Long-running operations execute in background tasks and emit `viewer-op-event`.
- Job cancellation is generation-based: stale or cancelled jobs are ignored when they finish.

## Commands

- `open_images(paths: Vec<String>) -> OpenResult`
- `viewer_init(window) -> ViewerInit`
- `viewer_frame_buffer(window, request) -> ViewerFrameBuffer`
- `cycle_window(window, direction) -> String`
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

Frontend command calls use `@tauri-apps/api` (`core.invoke`, `event.listen`, `window.getCurrentWindow`) via `web/src/lib/tauri.ts`.
