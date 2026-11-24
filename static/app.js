const USE_MOCK = true;

// Semua akses API terpusat di modul ini agar mock dan server memakai kontrak yang sama.
const api = (() => {
  let mockModels = [];
  let mockActive = null;
  let mockConf = 0.25;
  let mockJobs = new Map();

  async function request(url, options = {}) {
    const response = await fetch(url, { credentials: "same-origin", ...options });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Permintaan gagal (${response.status}).`);
    return data;
  }

  function copyModels() {
    return mockModels.map((model) => ({ ...model }));
  }

  return {
    async getModels() {
      if (USE_MOCK) return { models: copyModels(), active: mockActive, fallback: false };
      return request("/api/models");
    },

    async uploadModel(file) {
      if (USE_MOCK) {
        if (!file || !file.name.toLowerCase().endsWith(".pt")) {
          return { ok: false, error: "Pilih file model dengan ekstensi .pt." };
        }
        const model = {
          name: file.name,
          size_mb: Number((file.size / (1024 * 1024)).toFixed(2)),
          uploaded: new Date().toISOString(),
        };
        mockModels = [model, ...mockModels.filter((item) => item.name !== model.name)];
        mockActive = model.name;
        return { ok: true, active: mockActive };
      }
      const form = new FormData();
      form.append("file", file);
      return request("/api/upload_model", { method: "POST", body: form });
    },

    async selectModel(name) {
      if (USE_MOCK) {
        if (!mockModels.some((model) => model.name === name)) throw new Error("Model tidak ditemukan.");
        mockActive = name;
        return { ok: true, active: mockActive };
      }
      return request("/api/select_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
    },

    async deleteModel(name) {
      if (USE_MOCK) {
        mockModels = mockModels.filter((model) => model.name !== name);
        if (mockActive === name) mockActive = null;
        return { ok: true };
      }
      return request("/api/delete_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
    },

    async getSettings() {
      if (USE_MOCK) return { conf: mockConf };
      return request("/api/settings");
    },

    async saveSettings(conf) {
      if (USE_MOCK) {
        mockConf = Number(conf);
        return { ok: true };
      }
      return request("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conf: Number(conf) }),
      });
    },

    async detectImage({ file, conf, enhance = "none", brightness = 0, region = "" }) {
      if (USE_MOCK) {
        const imageUrl = URL.createObjectURL(file);
        const total = Math.floor(Math.random() * 10) + 3;
        return {
          original_url: imageUrl,
          enhanced_url: imageUrl,
          result_url: imageUrl,
          total,
          inside_region: Math.floor(Math.random() * (total + 1)),
          avg_conf: Number((0.68 + Math.random() * 0.27).toFixed(2)),
          baseline: null,
          model: mockActive,
        };
      }
      const form = new FormData();
      form.append("file", file);
      form.append("conf", String(conf));
      form.append("enhance", enhance);
      form.append("brightness", String(brightness));
      form.append("region", region);
      return request("/api/detect_image", { method: "POST", body: form });
    },

    async processVideo({ file, conf, enhance = "none", brightness = 0, region = "" }) {
      if (USE_MOCK) {
        const jobId = `mock-${Date.now()}`;
        mockJobs.set(jobId, {
          status: "done",
          percent: 100,
          result_url: URL.createObjectURL(file),
          stats: { inside_now: 0, entered_region: 0, unique_total: Math.floor(Math.random() * 10) + 3 },
          baseline: null,
          message: "Video selesai diproses.",
        });
        return { job_id: jobId };
      }
      const form = new FormData();
      form.append("file", file);
      form.append("conf", String(conf));
      form.append("enhance", enhance);
      form.append("brightness", String(brightness));
      form.append("region", region);
      return request("/api/process_video", { method: "POST", body: form });
    },

    async getProgress(jobId) {
      if (USE_MOCK) {
        return mockJobs.get(jobId) || {
          status: "error",
          percent: 0,
          result_url: null,
          stats: { inside_now: 0, entered_region: 0, unique_total: 0 },
          baseline: null,
          message: "Pekerjaan video tidak ditemukan.",
        };
      }
      return request(`/api/progress/${encodeURIComponent(jobId)}`);
    },

    async getReport() {
      if (USE_MOCK) return { rows: [] };
      return request("/api/report");
    },

    async clearReport() {
      if (USE_MOCK) return { ok: true };
      return request("/api/report/clear", { method: "POST" });
    },

    async exportExcel() {
      if (USE_MOCK) {
        return new Blob(["Mock laporan Arus\n"], {
          type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        });
      }
      const response = await fetch("/api/export_excel", { credentials: "same-origin" });
      if (!response.ok) throw new Error("Gagal mengunduh laporan Excel.");
      return response.blob();
    },
  };
})();

const state = {
  models: [],
  activeModel: null,
  confidence: 0.25,
  currentFile: null,
  previewUrl: null,
  mockResultUrl: null,
  activeTab: "foto",
  saveTimer: null,
};

const elements = {
  tabs: [...document.querySelectorAll("[data-tab]")],
  panels: [...document.querySelectorAll("[data-panel]")],
  modelState: document.getElementById("model-state"),
  activeModel: document.getElementById("active-model"),
  noModel: document.getElementById("no-model-state"),
  photoWorkspace: document.getElementById("photo-workspace"),
  modelInput: document.getElementById("model-file-input"),
  photoInput: document.getElementById("photo-file-input"),
  dropzone: document.getElementById("photo-dropzone"),
  modelTable: document.getElementById("model-table-body"),
  modelCount: document.getElementById("model-count"),
  confidence: document.getElementById("confidence-slider"),
  confidenceValue: document.getElementById("confidence-value"),
  settingsConfidence: document.getElementById("settings-confidence"),
  settingsConfidenceValue: document.getElementById("settings-confidence-value"),
  saveNote: document.getElementById("settings-save-note"),
  globalMessage: document.getElementById("global-message"),
  toast: document.getElementById("toast"),
  error: document.getElementById("detection-error"),
  resultSection: document.getElementById("result-section"),
  resultImage: document.getElementById("result-image"),
  imageLoading: document.getElementById("image-loading"),
  totalCount: document.getElementById("total-count"),
  insideCount: document.getElementById("inside-count"),
  averageConfidence: document.getElementById("average-confidence"),
  resultModel: document.getElementById("result-model"),
  redetect: document.getElementById("redetect-button"),
};

function formatDecimal(value, places = 2) {
  return Number(value).toFixed(places).replace(".", ",");
}

function setToast(message, isError = false) {
  elements.toast.textContent = message;
  elements.toast.classList.toggle("toast-error", isError);
  elements.toast.hidden = false;
  window.clearTimeout(setToast.timer);
  setToast.timer = window.setTimeout(() => { elements.toast.hidden = true; }, 3600);
}

function setGlobalError(message) {
  elements.globalMessage.textContent = message;
  elements.globalMessage.hidden = !message;
  elements.globalMessage.classList.toggle("message-error", Boolean(message));
}

function setTab(tabName) {
  state.activeTab = tabName;
  elements.tabs.forEach((tab) => {
    const active = tab.dataset.tab === tabName;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  elements.panels.forEach((panel) => {
    panel.hidden = panel.dataset.panel !== tabName;
  });
}

function formatDate(value) {
  if (!value || typeof value !== "string") return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("id-ID", { day: "numeric", month: "short", year: "numeric" }).format(date);
}

function makeCell(text, className = "") {
  const cell = document.createElement("td");
  cell.textContent = text;
  if (className) cell.className = className;
  return cell;
}

function renderModels() {
  const active = state.models.find((model) => model.name === state.activeModel) || null;
  elements.modelState.textContent = active ? "Aktif" : "Belum dipilih";
  elements.modelState.classList.toggle("is-ready", Boolean(active));
  elements.activeModel.textContent = active ? active.name : "Belum ada model aktif";
  elements.noModel.hidden = Boolean(active);
  elements.photoWorkspace.hidden = !active;
  elements.modelCount.textContent = `${state.models.length} model`;
  elements.modelTable.replaceChildren();

  if (!state.models.length) {
    const row = document.createElement("tr");
    const cell = makeCell("Belum ada model. Upload file .pt untuk memulai.", "table-empty");
    cell.colSpan = 4;
    row.append(cell);
    elements.modelTable.append(row);
    return;
  }

  state.models.forEach((model) => {
    const row = document.createElement("tr");
    row.append(makeCell(model.name, "model-name-cell"));
    row.append(makeCell(`${Number(model.size_mb || 0).toFixed(2)} MB`, "numeric-cell"));
    row.append(makeCell(formatDate(model.uploaded), "date-cell"));
    const actionCell = document.createElement("td");
    actionCell.className = "row-actions";
    if (model.name === state.activeModel) {
      const activeLabel = document.createElement("span");
      activeLabel.className = "active-label";
      activeLabel.textContent = "Model aktif";
      actionCell.append(activeLabel);
    } else {
      const selectButton = document.createElement("button");
      selectButton.type = "button";
      selectButton.className = "text-button";
      selectButton.dataset.action = "select";
      selectButton.dataset.name = model.name;
      selectButton.textContent = "Jadikan Aktif";
      actionCell.append(selectButton);
    }
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "text-button text-button-danger";
    deleteButton.dataset.action = "delete";
    deleteButton.dataset.name = model.name;
    deleteButton.textContent = "Hapus";
    actionCell.append(deleteButton);
    row.append(actionCell);
    elements.modelTable.append(row);
  });
}

async function refreshModels() {
  const result = await api.getModels();
  state.models = Array.isArray(result.models) ? result.models : [];
  state.activeModel = result.active || null;
  renderModels();
}

function setConfidence(value) {
  const parsed = Math.min(0.9, Math.max(0.1, Number(value)));
  state.confidence = parsed;
  const text = formatDecimal(parsed);
  elements.confidence.value = String(parsed);
  elements.confidenceValue.value = text;
  elements.confidenceValue.textContent = text;
  elements.settingsConfidence.value = String(parsed);
  elements.settingsConfidenceValue.value = text;
  elements.settingsConfidenceValue.textContent = text;
}

function scheduleSaveSettings() {
  window.clearTimeout(state.saveTimer);
  elements.saveNote.textContent = "Menyimpan…";
  state.saveTimer = window.setTimeout(async () => {
    try {
      await api.saveSettings(state.confidence);
      elements.saveNote.textContent = "Perubahan tersimpan otomatis.";
    } catch (error) {
      elements.saveNote.textContent = error.message || "Gagal menyimpan pengaturan.";
    }
  }, 350);
}

function clearImageUrls() {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  if (USE_MOCK && state.mockResultUrl) URL.revokeObjectURL(state.mockResultUrl);
  state.previewUrl = null;
  state.mockResultUrl = null;
}

function setDetectionLoading(isLoading) {
  elements.imageLoading.hidden = !isLoading;
  elements.redetect.disabled = isLoading;
  elements.dropzone.classList.toggle("is-disabled", isLoading);
}

async function runDetection() {
  if (!state.currentFile || !state.activeModel) return;
  elements.error.hidden = true;
  elements.resultSection.hidden = false;
  setDetectionLoading(true);

  try {
    const result = await api.detectImage({
      file: state.currentFile,
      conf: state.confidence,
      enhance: "none",
      brightness: 0,
      region: "",
    });
    if (USE_MOCK && state.mockResultUrl) URL.revokeObjectURL(state.mockResultUrl);
    state.mockResultUrl = USE_MOCK ? result.result_url : null;
    elements.resultImage.src = result.result_url || result.original_url;
    elements.totalCount.textContent = String(result.total ?? 0);
    elements.insideCount.textContent = `${result.inside_region ?? 0} di dalam region`;
    elements.averageConfidence.textContent = result.avg_conf == null ? "—" : formatDecimal(result.avg_conf);
    elements.resultModel.textContent = result.model || state.activeModel || "—";
  } catch (error) {
    elements.error.textContent = error.message || "Deteksi gagal. Periksa foto dan koneksi ke server, lalu coba lagi.";
    elements.error.hidden = false;
  } finally {
    setDetectionLoading(false);
  }
}

function selectPhoto(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    elements.error.textContent = "File tidak dikenali sebagai foto. Pilih JPG, PNG, atau WEBP.";
    elements.error.hidden = false;
    return;
  }
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  if (USE_MOCK && state.mockResultUrl) URL.revokeObjectURL(state.mockResultUrl);
  state.mockResultUrl = null;
  state.currentFile = file;
  state.previewUrl = URL.createObjectURL(file);
  elements.resultImage.src = state.previewUrl;
  elements.resultImage.alt = `Pratinjau foto ${file.name}`;
  elements.totalCount.textContent = "—";
  elements.insideCount.textContent = "— di dalam region";
  elements.averageConfidence.textContent = "—";
  elements.resultModel.textContent = state.activeModel || "—";
  elements.resultSection.hidden = false;
  elements.error.hidden = true;
  runDetection();
}

async function handleModelUpload(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pt")) {
    setToast("Format model harus .pt.", true);
    return;
  }
  try {
    const result = await api.uploadModel(file);
    if (!result.ok) throw new Error(result.error || "Upload model gagal.");
    await refreshModels();
    setToast(`Model ${result.active || file.name} berhasil diunggah.`);
    setGlobalError("");
  } catch (error) {
    setToast(error.message || "Upload model gagal. Silakan coba lagi.", true);
  } finally {
    elements.modelInput.value = "";
  }
}

async function handleModelAction(event) {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const { action, name } = button.dataset;
  button.disabled = true;
  try {
    if (action === "select") {
      const result = await api.selectModel(name);
      if (!result.ok) throw new Error("Model gagal diaktifkan.");
      state.activeModel = result.active || name;
      await refreshModels();
      setToast(`${name} menjadi model aktif.`);
    } else if (action === "delete") {
      const result = await api.deleteModel(name);
      if (!result.ok) throw new Error("Model gagal dihapus.");
      await refreshModels();
      setToast(`${name} dihapus.`);
    }
  } catch (error) {
    setToast(error.message || "Aksi model gagal.", true);
  } finally {
    button.disabled = false;
  }
}

function bindEvents() {
  elements.tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => setTab(tab.dataset.tab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      let nextIndex = index;
      if (event.key === "ArrowRight") nextIndex = (index + 1) % elements.tabs.length;
      if (event.key === "ArrowLeft") nextIndex = (index - 1 + elements.tabs.length) % elements.tabs.length;
      if (event.key === "Home") nextIndex = 0;
      if (event.key === "End") nextIndex = elements.tabs.length - 1;
      elements.tabs[nextIndex].focus();
      setTab(elements.tabs[nextIndex].dataset.tab);
    });
  });

  ["upload-model-button", "empty-upload-button", "settings-upload-button"].forEach((id) => {
    document.getElementById(id).addEventListener("click", () => elements.modelInput.click());
  });
  elements.modelInput.addEventListener("change", () => handleModelUpload(elements.modelInput.files[0]));
  elements.photoInput.addEventListener("change", () => selectPhoto(elements.photoInput.files[0]));
  elements.redetect.addEventListener("click", runDetection);
  elements.modelTable.addEventListener("click", handleModelAction);

  [elements.confidence, elements.settingsConfidence].forEach((slider) => {
    slider.addEventListener("input", () => {
      setConfidence(slider.value);
      scheduleSaveSettings();
    });
  });

  elements.dropzone.addEventListener("keydown", (event) => {
    if (event.target === elements.photoInput || !["Enter", " "].includes(event.key)) return;
    event.preventDefault();
    elements.photoInput.click();
  });
  ["dragenter", "dragover"].forEach((eventName) => {
    elements.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropzone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    elements.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropzone.classList.remove("is-dragging");
    });
  });
  elements.dropzone.addEventListener("drop", (event) => {
    selectPhoto(event.dataTransfer.files[0]);
  });

  window.addEventListener("beforeunload", clearImageUrls);
}

async function initialize() {
  bindEvents();
  setTab("foto");
  const [modelsResult, settingsResult] = await Promise.allSettled([api.getModels(), api.getSettings()]);
  if (modelsResult.status === "fulfilled") {
    state.models = Array.isArray(modelsResult.value.models) ? modelsResult.value.models : [];
    state.activeModel = modelsResult.value.active || null;
    renderModels();
  } else {
    state.models = [];
    state.activeModel = null;
    renderModels();
    setGlobalError(modelsResult.reason.message || "Daftar model gagal dimuat.");
  }

  if (settingsResult.status === "fulfilled") {
    setConfidence(settingsResult.value.conf ?? 0.25);
  } else {
    setConfidence(0.25);
    setGlobalError(settingsResult.reason.message || "Pengaturan gagal dimuat.");
  }
}

initialize();
