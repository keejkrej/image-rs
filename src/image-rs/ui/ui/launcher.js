const api = window.tauriApi;
const RESULT_EVENT = "launcher-open-result";

const ids = {
  menuFile: document.getElementById("menu-file"),
  menuWindow: document.getElementById("menu-window"),
  menuHelp: document.getElementById("menu-help"),
  pathInput: document.getElementById("path-input"),
  openButton: document.getElementById("open-button"),
  summary: document.getElementById("summary-line"),
  list: document.getElementById("result-list"),
};

function setSummary(message) {
  ids.summary.textContent = message;
}

function runtimeMessage() {
  const details = api?.diagnostics ? JSON.stringify(api.diagnostics()) : "{}";
  return `Tauri runtime unavailable. ${details}`;
}

function renderResult(result) {
  if (!result) {
    setSummary("No result payload.");
    return;
  }

  const opened = result.opened?.length || 0;
  const focused = result.focused?.length || 0;
  const skipped = result.skipped?.length || 0;
  const errors = result.errors?.length || 0;
  setSummary(
    `Opened ${opened}, focused ${focused}, skipped ${skipped}, errors ${errors}`
  );

  ids.list.innerHTML = "";
  const addEntries = (prefix, items) => {
    for (const item of items || []) {
      const row = document.createElement("li");
      row.textContent = `${prefix}: ${item}`;
      ids.list.appendChild(row);
    }
  };

  addEntries("Opened", result.opened);
  addEntries("Focused", result.focused);
  addEntries("Skipped", result.skipped);
  addEntries("Error", result.errors);
}

async function openPathInput() {
  const path = ids.pathInput.value.trim();
  if (!path) {
    setSummary("Provide an image path.");
    return;
  }

  if (!api?.canInvoke?.()) {
    setSummary(runtimeMessage());
    return;
  }

  try {
    const result = await api.invoke("open_images", { paths: [path] });
    renderResult(result);
  } catch (error) {
    setSummary(String(error));
  }
}

async function init() {
  if (!api?.canInvoke?.()) {
    setSummary(runtimeMessage());
    return;
  }

  ids.openButton.addEventListener("click", openPathInput);
  ids.pathInput.addEventListener("keydown", async (event) => {
    if (event.key === "Enter") {
      await openPathInput();
    }
  });

  ids.menuFile.addEventListener("click", () => {
    ids.pathInput.focus();
    setSummary("File: type a path and click Open, or drag files onto this window.");
  });
  ids.menuWindow.addEventListener("click", () => {
    setSummary("Window: image viewers are managed automatically per file.");
  });
  ids.menuHelp.addEventListener("click", () => {
    setSummary("Help: supported formats are TIFF, PNG, JPEG.");
  });

  await api.listen(RESULT_EVENT, (event) => {
    renderResult(event.payload);
  });
}

init().catch((error) => setSummary(String(error)));
