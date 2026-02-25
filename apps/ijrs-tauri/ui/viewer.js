const api = window.tauriApi;
const OP_EVENT = "viewer-op-event";
const ALLOWED_OPS = ["gaussian.blur", "threshold.fixed", "threshold.otsu"];

const state = {
  context: null,
  frame: null,
  frameImage: null,
  z: 0,
  t: 0,
  channel: 0,
  zoom: 1,
  panX: 18,
  panY: 18,
  dragging: false,
  lastMouseX: 0,
  lastMouseY: 0,
  hoverX: null,
  hoverY: null,
  hoverValue: null,
  activeJobId: null,
  requestToken: 0,
  busy: false,
};

const ids = {
  menuFile: document.getElementById("menu-file"),
  menuProcess: document.getElementById("menu-process"),
  menuAnalyze: document.getElementById("menu-analyze"),
  menuWindow: document.getElementById("menu-window"),
  menuHelp: document.getElementById("menu-help"),
  toggleDrawer: document.getElementById("toggle-drawer"),
  drawer: document.getElementById("drawer"),
  windowTitle: document.getElementById("window-title"),
  sourcePath: document.getElementById("source-path"),
  canvas: document.getElementById("image-canvas"),
  zSlider: document.getElementById("z-slider"),
  zValue: document.getElementById("z-value"),
  tSlider: document.getElementById("t-slider"),
  tValue: document.getElementById("t-value"),
  channelSlider: document.getElementById("channel-slider"),
  channelValue: document.getElementById("channel-value"),
  histogram: document.getElementById("histogram"),
  openProcessDialog: document.getElementById("open-process-dialog"),
  resetView: document.getElementById("reset-view"),
  exportPath: document.getElementById("export-path"),
  exportButton: document.getElementById("export-button"),
  meta: document.getElementById("meta"),
  statusBar: document.getElementById("status-bar"),
  processDialog: document.getElementById("process-dialog"),
  processOp: document.getElementById("process-op"),
  processParams: document.getElementById("process-params"),
  processStatus: document.getElementById("process-status"),
  previewOp: document.getElementById("preview-op"),
  applyOp: document.getElementById("apply-op"),
  cancelOp: document.getElementById("cancel-op"),
  closeDialog: document.getElementById("close-dialog"),
};

function runtimeMessage() {
  const details = api?.diagnostics ? JSON.stringify(api.diagnostics()) : "{}";
  return `Tauri runtime unavailable. ${details}`;
}

async function invoke(command, payload = {}) {
  if (!api?.canInvoke?.()) {
    throw new Error(runtimeMessage());
  }
  return api.invoke(command, payload);
}

function setStatus(message) {
  ids.statusBar.textContent = message;
}

function setProcessStatus(message) {
  ids.processStatus.textContent = message;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function defaultParamsFor(op) {
  if (op === "gaussian.blur") {
    return '{"sigma":1.0}';
  }
  if (op === "threshold.fixed") {
    return '{"threshold":0.5}';
  }
  return "{}";
}

function parseProcessParams() {
  const raw = ids.processParams.value.trim();
  if (!raw) {
    return {};
  }
  return JSON.parse(raw);
}

async function loadFrameImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to decode rendered frame."));
    image.src = dataUrl;
  });
}

function drawHistogram(histogram) {
  const canvas = ids.histogram;
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);

  const max = Math.max(...histogram, 1);
  const barWidth = width / histogram.length;
  context.fillStyle = "#5a5a5a";
  for (let i = 0; i < histogram.length; i += 1) {
    const ratio = histogram[i] / max;
    const barHeight = ratio * (height - 6);
    context.fillRect(i * barWidth, height - barHeight, barWidth, barHeight);
  }
}

function redraw() {
  const context = ids.canvas.getContext("2d");
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, ids.canvas.width, ids.canvas.height);
  context.fillStyle = "#121212";
  context.fillRect(0, 0, ids.canvas.width, ids.canvas.height);

  if (!state.frameImage) {
    return;
  }

  context.imageSmoothingEnabled = false;
  context.setTransform(state.zoom, 0, 0, state.zoom, state.panX, state.panY);
  context.drawImage(state.frameImage, 0, 0);
  context.setTransform(1, 0, 0, 1, 0, 0);
}

function fitCanvasToContainer() {
  const rect = ids.canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  if (ids.canvas.width !== width || ids.canvas.height !== height) {
    ids.canvas.width = width;
    ids.canvas.height = height;
    redraw();
  }
}

function toImageCoordinates(clientX, clientY) {
  if (!state.frame) {
    return null;
  }

  const rect = ids.canvas.getBoundingClientRect();
  const localX = clientX - rect.left;
  const localY = clientY - rect.top;
  const imageX = Math.floor((localX - state.panX) / state.zoom);
  const imageY = Math.floor((localY - state.panY) / state.zoom);
  if (
    imageX < 0 ||
    imageY < 0 ||
    imageX >= state.frame.width ||
    imageY >= state.frame.height
  ) {
    return null;
  }
  return { x: imageX, y: imageY };
}

function updateStatusBar() {
  const x = state.hoverX == null ? "-" : state.hoverX;
  const y = state.hoverY == null ? "-" : state.hoverY;
  const value =
    state.hoverValue == null ? "-" : Number(state.hoverValue).toFixed(4);

  const busyNote = state.busy ? "  [Processing...]" : "";
  setStatus(
    `X:${x} Y:${y} Value:${value}  Z:${state.z} T:${state.t} C:${state.channel}  Zoom:${(
      state.zoom * 100
    ).toFixed(0)}%${busyNote}`
  );
}

async function applyFrame(frame) {
  const image = await loadFrameImage(frame.png_data_url);
  state.frame = frame;
  state.frameImage = image;
  drawHistogram(frame.histogram);
  redraw();
  updateStatusBar();
}

async function requestFrame() {
  if (!state.context) {
    return;
  }
  const token = ++state.requestToken;
  setStatus("Rendering...");
  const frame = await invoke("viewer_frame", {
    request: {
      z: state.z,
      t: state.t,
      channel: state.channel,
    },
  });
  if (token !== state.requestToken) {
    return;
  }
  await applyFrame(frame);
  updateStatusBar();
}

function configureSliders(summary) {
  ids.zSlider.max = String(Math.max(0, summary.z_slices - 1));
  ids.tSlider.max = String(Math.max(0, summary.times - 1));
  ids.channelSlider.max = String(Math.max(0, summary.channels - 1));

  state.z = clamp(state.z, 0, Number(ids.zSlider.max));
  state.t = clamp(state.t, 0, Number(ids.tSlider.max));
  state.channel = clamp(state.channel, 0, Number(ids.channelSlider.max));

  ids.zSlider.value = String(state.z);
  ids.tSlider.value = String(state.t);
  ids.channelSlider.value = String(state.channel);
  ids.zValue.textContent = String(state.z);
  ids.tValue.textContent = String(state.t);
  ids.channelValue.textContent = String(state.channel);
}

function bindSliders() {
  ids.zSlider.addEventListener("input", async (event) => {
    state.z = Number(event.target.value);
    ids.zValue.textContent = String(state.z);
    await requestFrame();
  });

  ids.tSlider.addEventListener("input", async (event) => {
    state.t = Number(event.target.value);
    ids.tValue.textContent = String(state.t);
    await requestFrame();
  });

  ids.channelSlider.addEventListener("input", async (event) => {
    state.channel = Number(event.target.value);
    ids.channelValue.textContent = String(state.channel);
    await requestFrame();
  });
}

function bindCanvas() {
  ids.canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const previousZoom = state.zoom;
    const nextZoom = clamp(
      event.deltaY < 0 ? previousZoom * 1.12 : previousZoom / 1.12,
      0.1,
      64
    );

    const rect = ids.canvas.getBoundingClientRect();
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    state.panX = localX - ((localX - state.panX) * nextZoom) / previousZoom;
    state.panY = localY - ((localY - state.panY) * nextZoom) / previousZoom;
    state.zoom = nextZoom;
    redraw();
    updateStatusBar();
  });

  ids.canvas.addEventListener("mousedown", (event) => {
    state.dragging = true;
    state.lastMouseX = event.clientX;
    state.lastMouseY = event.clientY;
    ids.canvas.style.cursor = "grabbing";
  });

  window.addEventListener("mouseup", () => {
    state.dragging = false;
    ids.canvas.style.cursor = "crosshair";
  });

  ids.canvas.addEventListener("mousemove", (event) => {
    if (state.dragging) {
      const dx = event.clientX - state.lastMouseX;
      const dy = event.clientY - state.lastMouseY;
      state.panX += dx;
      state.panY += dy;
      state.lastMouseX = event.clientX;
      state.lastMouseY = event.clientY;
      redraw();
    }

    const coord = toImageCoordinates(event.clientX, event.clientY);
    if (!coord || !state.frame) {
      state.hoverX = null;
      state.hoverY = null;
      state.hoverValue = null;
    } else {
      const offset = coord.y * state.frame.width + coord.x;
      state.hoverX = coord.x;
      state.hoverY = coord.y;
      state.hoverValue = state.frame.values[offset];
    }
    updateStatusBar();
  });

  ids.canvas.addEventListener("mouseleave", () => {
    state.hoverX = null;
    state.hoverY = null;
    state.hoverValue = null;
    updateStatusBar();
  });
}

async function startOperation(mode) {
  if (state.busy) {
    setProcessStatus("Another operation is already running.");
    return;
  }

  const op = ids.processOp.value;
  const params = parseProcessParams();
  state.busy = true;
  setProcessStatus(`Running ${op} (${mode})...`);
  updateStatusBar();

  try {
    const ticket = await invoke("viewer_start_op", {
      request: { op, params, mode },
    });
    if (state.busy) {
      state.activeJobId = ticket.job_id;
    } else {
      state.activeJobId = null;
    }
  } catch (error) {
    state.busy = false;
    state.activeJobId = null;
    setProcessStatus(String(error));
    updateStatusBar();
  }
}

async function cancelOperation() {
  if (state.activeJobId == null) {
    setProcessStatus("No running job.");
    return;
  }

  try {
    await invoke("viewer_cancel_op", {
      request: { job_id: state.activeJobId },
    });
  } catch (error) {
    setProcessStatus(String(error));
  }
}

function handleOpEvent(payload) {
  if (!payload || payload.job_id == null) {
    return;
  }
  if (state.activeJobId != null && payload.job_id !== state.activeJobId) {
    return;
  }

  if (payload.status === "completed") {
    state.activeJobId = null;
    state.busy = false;
    setProcessStatus(`${payload.op} completed.`);
    requestFrame().catch((error) => setStatus(String(error)));
  } else if (payload.status === "failed") {
    state.activeJobId = null;
    state.busy = false;
    setProcessStatus(payload.message || `${payload.op} failed.`);
  } else if (payload.status === "cancelled") {
    state.activeJobId = null;
    state.busy = false;
    setProcessStatus(payload.message || "Operation cancelled.");
  }
  updateStatusBar();
}

function bindProcessDialog() {
  ids.processOp.addEventListener("change", (event) => {
    ids.processParams.value = defaultParamsFor(event.target.value);
  });

  ids.previewOp.addEventListener("click", async () => {
    try {
      await startOperation("preview");
    } catch (error) {
      setProcessStatus(String(error));
    }
  });

  ids.applyOp.addEventListener("click", async () => {
    try {
      await startOperation("apply");
    } catch (error) {
      setProcessStatus(String(error));
    }
  });

  ids.cancelOp.addEventListener("click", async () => {
    await cancelOperation();
  });

  ids.closeDialog.addEventListener("click", () => {
    ids.processDialog.close();
  });
}

function bindMenus() {
  ids.toggleDrawer.addEventListener("click", () => {
    ids.drawer.classList.toggle("hidden");
    fitCanvasToContainer();
  });

  ids.menuFile.addEventListener("click", () => {
    ids.exportPath.focus();
    setStatus("File: use Export to write the current rendered image.");
  });

  ids.menuProcess.addEventListener("click", () => {
    openProcessDialog();
  });
  ids.menuAnalyze.addEventListener("click", () => {
    ids.drawer.classList.remove("hidden");
    setStatus("Analyze: histogram and metadata are in the side panel.");
  });
  ids.menuWindow.addEventListener("click", () => {
    ids.drawer.classList.toggle("hidden");
    fitCanvasToContainer();
  });
  ids.menuHelp.addEventListener("click", () => {
    setStatus("Help: zoom with wheel, pan by dragging, process via Process menu.");
  });
}

function openProcessDialog() {
  if (!ids.processDialog.open) {
    ids.processDialog.showModal();
  }
}

async function initOpsMenu() {
  try {
    const ops = await invoke("list_preview_ops");
    const filtered = ops.filter((name) => ALLOWED_OPS.includes(name));
    if (filtered.length > 0) {
      ids.processOp.innerHTML = "";
      for (const op of filtered) {
        const option = document.createElement("option");
        option.value = op;
        option.textContent = op;
        ids.processOp.appendChild(option);
      }
    }
  } catch (error) {
    setProcessStatus(String(error));
  }
  ids.processParams.value = defaultParamsFor(ids.processOp.value);
}

async function init() {
  if (!api?.canInvoke?.()) {
    setStatus(runtimeMessage());
    return;
  }

  bindMenus();
  bindCanvas();
  bindSliders();
  bindProcessDialog();

  ids.openProcessDialog.addEventListener("click", () => {
    openProcessDialog();
  });
  ids.resetView.addEventListener("click", () => {
    state.zoom = 1;
    state.panX = 18;
    state.panY = 18;
    redraw();
    updateStatusBar();
  });
  ids.exportButton.addEventListener("click", async () => {
    const outputPath = ids.exportPath.value.trim();
    if (!outputPath) {
      setStatus("Provide an export path.");
      return;
    }
    try {
      await invoke("export_preview", {
        request: {
          output_path: outputPath,
          preview: null,
        },
      });
      setStatus(`Exported ${outputPath}`);
      updateStatusBar();
    } catch (error) {
      setStatus(String(error));
    }
  });

  const initPayload = await invoke("viewer_init");
  state.context = initPayload;
  ids.windowTitle.textContent =
    initPayload.path.split(/[\\/]/).pop() || "Image Viewer";
  ids.sourcePath.textContent = initPayload.path;
  ids.meta.textContent = JSON.stringify(initPayload.summary, null, 2);
  configureSliders(initPayload.summary);

  await initOpsMenu();
  await applyFrame(initPayload.default_frame);

  await api.listen(OP_EVENT, (event) => {
    handleOpEvent(event.payload);
  });

  window.addEventListener("resize", fitCanvasToContainer);
  fitCanvasToContainer();
  updateStatusBar();
}

init().catch((error) => setStatus(String(error)));
