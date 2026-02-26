(function () {
  function resolveInvoke() {
    if (typeof window.__TAURI__?.core?.invoke === "function") {
      return window.__TAURI__.core.invoke;
    }
    if (typeof window.__TAURI__?.invoke === "function") {
      return window.__TAURI__.invoke;
    }
    if (typeof window.__TAURI_INTERNALS__?.invoke === "function") {
      return window.__TAURI_INTERNALS__.invoke;
    }
    return null;
  }

  function resolveListen() {
    if (typeof window.__TAURI__?.event?.listen === "function") {
      return window.__TAURI__.event.listen;
    }
    if (typeof window.__TAURI_INTERNALS__?.event?.listen === "function") {
      return window.__TAURI_INTERNALS__.event.listen;
    }
    return null;
  }

  function diagnostics() {
    return {
      has_tauri: !!window.__TAURI__,
      has_core_invoke: typeof window.__TAURI__?.core?.invoke === "function",
      has_legacy_invoke: typeof window.__TAURI__?.invoke === "function",
      has_internal_invoke: typeof window.__TAURI_INTERNALS__?.invoke === "function",
      has_event_listen: typeof window.__TAURI__?.event?.listen === "function",
    };
  }

  async function invoke(command, payload) {
    const fn = resolveInvoke();
    if (!fn) {
      const details = JSON.stringify(diagnostics());
      throw new Error(
        `Tauri invoke API not available. Run this page inside Tauri runtime. ${details}`
      );
    }
    return fn(command, payload);
  }

  async function listen(eventName, handler) {
    const fn = resolveListen();
    if (!fn) {
      return () => {};
    }
    return fn(eventName, handler);
  }

  window.tauriApi = {
    invoke,
    listen,
    canInvoke: () => !!resolveInvoke(),
    diagnostics,
  };
})();
