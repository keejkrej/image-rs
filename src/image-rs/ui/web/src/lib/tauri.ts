import { invoke } from "@tauri-apps/api/core";
import { listen, type Event } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";

export async function tauriInvoke<T>(command: string, payload?: Record<string, unknown>): Promise<T> {
  const runtimeAvailable = typeof window !== "undefined" && (
    typeof (window as Window & { __TAURI__?: unknown }).__TAURI__ !== "undefined" ||
    typeof (window as Window & { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ !== "undefined"
  );

  if (!runtimeAvailable) {
    throw new Error(
      "Tauri runtime unavailable: launch this UI via `cargo tauri dev` or a packaged desktop build."
    );
  }

  try {
    return await invoke<T>(command, payload);
  } catch (error) {
    throw new Error(String(error));
  }
}

export async function tauriListen<T>(
  eventName: string,
  handler: (event: Event<T>) => void
): Promise<() => void> {
  return listen<T>(eventName, handler);
}

export function currentWindow() {
  return getCurrentWindow();
}

export type OpenResult = {
  opened: string[];
  focused: string[];
  skipped: string[];
  errors: string[];
};

export type ImageSummary = {
  shape: number[];
  axes: string[];
  channels: number;
  z_slices: number;
  times: number;
  min: number;
  max: number;
  source: string;
};

export type ViewerInit = {
  window_label: string;
  path: string;
  summary: ImageSummary;
  default_frame: ViewerFrameBuffer;
};

export type ViewerFrameBuffer = {
  width: number;
  height: number;
  pixels_u8: number[];
  histogram: number[];
  min: number;
  max: number;
};
