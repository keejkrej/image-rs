const tauriInvoke =
  window.__TAURI__?.core?.invoke ||
  window.__TAURI__?.invoke;

const state = {
  inputPath: "",
  z: 0,
  t: 0,
  channel: 0,
};

const ids = {
  path: document.getElementById("path-input"),
  open: document.getElementById("open-button"),
  z: document.getElementById("z-slider"),
  zValue: document.getElementById("z-value"),
  t: document.getElementById("t-slider"),
  tValue: document.getElementById("t-value"),
  channel: document.getElementById("channel-slider"),
  channelValue: document.getElementById("channel-value"),
  previewOp: document.getElementById("preview-op"),
  previewParam: document.getElementById("preview-param"),
  applyPreview: document.getElementById("apply-preview"),
  exportPath: document.getElementById("export-path"),
  export: document.getElementById("export-button"),
  image: document.getElementById("slice-image"),
  status: document.getElementById("status"),
  meta: document.getElementById("meta"),
  histogram: document.getElementById("histogram"),
};

function setStatus(message) {
  ids.status.textContent = message;
}

async function invoke(command, payload = {}) {
  if (!tauriInvoke) {
    throw new Error("Tauri invoke API not available.");
  }
  return tauriInvoke(command, payload);
}

function clampToSlider(value, slider) {
  const min = Number(slider.min || 0);
  const max = Number(slider.max || 0);
  return Math.max(min, Math.min(max, value));
}

function currentPreview() {
  const op = ids.previewOp.value.trim();
  if (!op) {
    return null;
  }
  const raw = ids.previewParam.value.trim();
  if (!raw) {
    return { op, params: {} };
  }
  return { op, params: JSON.parse(raw) };
}

async function inspectAndConfigure(path) {
  const summary = await invoke("inspect_image", { path });
  ids.meta.textContent = JSON.stringify(summary, null, 2);
  ids.z.max = String(Math.max(0, summary.z_slices - 1));
  ids.t.max = String(Math.max(0, summary.times - 1));
  ids.channel.max = String(Math.max(0, summary.channels - 1));
  state.z = clampToSlider(state.z, ids.z);
  state.t = clampToSlider(state.t, ids.t);
  state.channel = clampToSlider(state.channel, ids.channel);
  ids.z.value = String(state.z);
  ids.t.value = String(state.t);
  ids.channel.value = String(state.channel);
  ids.zValue.textContent = String(state.z);
  ids.tValue.textContent = String(state.t);
  ids.channelValue.textContent = String(state.channel);
}

async function render() {
  if (!state.inputPath) {
    return;
  }
  setStatus("Rendering...");
  const preview = currentPreview();
  const imageData = await invoke("render_slice", {
    request: {
      path: state.inputPath,
      z: Number(state.z),
      t: Number(state.t),
      channel: Number(state.channel),
      preview,
    },
  });
  ids.image.src = imageData;
  const histogram = await invoke("slice_histogram", {
    request: {
      path: state.inputPath,
      z: Number(state.z),
      t: Number(state.t),
      channel: Number(state.channel),
      preview,
    },
  });
  drawHistogram(histogram);
  setStatus("Ready");
}

function drawHistogram(histogram) {
  const canvas = ids.histogram;
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#f5f7fb";
  context.fillRect(0, 0, width, height);
  const max = Math.max(...histogram, 1);
  const barWidth = width / histogram.length;
  context.fillStyle = "#007a6a";
  histogram.forEach((value, index) => {
    const ratio = value / max;
    const barHeight = ratio * (height - 12);
    context.fillRect(index * barWidth, height - barHeight, barWidth, barHeight);
  });
}

async function load(path) {
  if (!path.trim()) {
    setStatus("Provide an input path.");
    return;
  }
  state.inputPath = path.trim();
  ids.path.value = state.inputPath;
  await inspectAndConfigure(state.inputPath);
  await render();
}

function updateAxisFromSlider(slider, key, label) {
  slider.addEventListener("input", async (event) => {
    state[key] = Number(event.target.value);
    label.textContent = String(state[key]);
    await render();
  });
}

async function init() {
  if (!tauriInvoke) {
    setStatus("This page must run inside Tauri.");
    return;
  }

  const previewOps = await invoke("list_preview_ops");
  for (const name of previewOps) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    ids.previewOp.appendChild(option);
  }

  ids.open.addEventListener("click", async () => {
    try {
      await load(ids.path.value);
    } catch (error) {
      setStatus(String(error));
    }
  });

  ids.applyPreview.addEventListener("click", async () => {
    try {
      await render();
    } catch (error) {
      setStatus(String(error));
    }
  });

  ids.export.addEventListener("click", async () => {
    const outputPath = ids.exportPath.value.trim();
    if (!outputPath) {
      setStatus("Provide an export path.");
      return;
    }
    try {
      await invoke("export_preview", {
        request: {
          path: state.inputPath,
          output_path: outputPath,
          preview: currentPreview(),
        },
      });
      setStatus(`Exported to ${outputPath}`);
    } catch (error) {
      setStatus(String(error));
    }
  });

  updateAxisFromSlider(ids.z, "z", ids.zValue);
  updateAxisFromSlider(ids.t, "t", ids.tValue);
  updateAxisFromSlider(ids.channel, "channel", ids.channelValue);

  try {
    const startupPath = await invoke("startup_input");
    if (startupPath) {
      await load(startupPath);
    }
  } catch (error) {
    setStatus(String(error));
  }
}

init().catch((error) => {
  setStatus(String(error));
});
