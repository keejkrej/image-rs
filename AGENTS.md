# AGENTS.md

## UI Architecture

The native GPUI UI follows ImageJ's multi-window desktop model. The persistent launcher owns the shared menu bar and tool palette, while every opened image is shown in its own native viewer window. Opening an already-loaded path focuses its existing viewer instead of duplicating it.

Shared command routing, selected-tool state, macros, and ImageJ-compatible identifiers such as `viewer-1` remain application-level state. Closing a viewer closes that image session; closing the launcher exits the application and its viewers. Preserve this workflow while presenting it with the modern zinc/blue visual system used by the GPUI implementation.
