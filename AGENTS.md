# AGENTS.md

## UI Architecture

The native egui UI intentionally uses a single application window. The startup launcher remains the main window, and every opened image is represented as an in-window tab with its own viewer session. Opening another image must create or focus a tab in that same window, not spawn another OS window.

This differs from the upstream ImageJ-style multi-window desktop model. The divergence is deliberate: egui works best when one `eframe` window owns the shared menu, toolbar, utility dialogs, command routing, and image canvas state. Keep ImageJ-compatible labels such as `viewer-1` as internal image/session identifiers for macros and commands, but render them as tabs in the main window.
