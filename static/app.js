const USE_MOCK = false;

// Semua akses API terpusat di modul ini agar mock dan server memakai kontrak yang sama.
const api = (() => {
  let mockModels = [];
  let mockActive = null;
  let mockConf = 0.25;
  let mockJobs = new Map();
  let mockReport = [
    { created_at: new Date(Date.now() - 22 * 60 * 1000).toISOString(), file_name: "simpang-barat.jpg", type: "Foto", enhance: "clahe", total: 8, avg_conf: 0.84, model: "yolo11_vehicle.pt" },
    { created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(), file_name: "akses-pasar.png", type: "Foto", enhance: "none", total: 4, avg_conf: 0.79, model: "yolo11_vehicle.pt" },
    { created_at: new Date(Date.now() - 26 * 60 * 60 * 1000).toISOString(), file_name: "jalan-utara.mp4", type: "Video", enhance: "he", total: 16, avg_conf: 0.87, model: "yolo11_vehicle.pt" },
  ];

  async function request(url, options = {}) {
    const response = await fetch(url, { credentials: "same-origin", ...options });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Permintaan gagal (${response.status}).`);
    return data;
  }

  function copyModels() {
    return mockModels.map((model) => ({ ...model }));
  }

  function addReportRow(row) {
    mockReport.unshift({ ...row, created_at: row.created_at || new Date().toISOString() });
  }

  function xlsxBlob(rows) {
    const encoder = new TextEncoder();
    const xmlEscape = (value) => String(value ?? "").replace(/[&<>\"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[character]));
    const headers = ["Tanggal", "File", "Jenis", "Status", "Enhancement", "Total", "Avg. confidence", "Model"];
    const values = [headers, ...rows.map((row) => [row.created_at, row.file_name, row.type, row.status || "Selesai", row.enhance, row.total, row.avg_conf, row.model])];
    const columnName = (number) => {
      let name = "";
      for (number += 1; number > 0; number = Math.floor((number - 1) / 26)) name = String.fromCharCode(65 + ((number - 1) % 26)) + name;
      return name;
    };
    const sheetRows = values.map((row, rowIndex) => {
      const cells = row.map((value, columnIndex) => {
        const reference = `${columnName(columnIndex)}${rowIndex + 1}`;
        if (typeof value === "number") return `<c r="${reference}"><v>${value}</v></c>`;
        return `<c r="${reference}" t="inlineStr"><is><t>${xmlEscape(value)}</t></is></c>`;
      }).join("");
      return `<row r="${rowIndex + 1}">${cells}</row>`;
    }).join("");
    const files = [
      ["[Content_Types].xml", `<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>`],
      ["_rels/.rels", `<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>`],
      ["xl/workbook.xml", `<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Laporan" sheetId="1" r:id="rId1"/></sheets></workbook>`],
      ["xl/_rels/workbook.xml.rels", `<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>`],
      ["xl/worksheets/sheet1.xml", `<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>${sheetRows}</sheetData></worksheet>`],
    ].map(([name, content]) => ({ name: encoder.encode(name), data: encoder.encode(content) }));

    const checksum = (bytes) => {
      let crc = 0xffffffff;
      for (const byte of bytes) {
        crc ^= byte;
        for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
      }
      return (crc ^ 0xffffffff) >>> 0;
    };
    const join = (parts) => {
      const size = parts.reduce((total, part) => total + part.length, 0);
      const result = new Uint8Array(size);
      let offset = 0;
      parts.forEach((part) => { result.set(part, offset); offset += part.length; });
      return result;
    };
    const localParts = [];
    const centralParts = [];
    let offset = 0;
    files.forEach(({ name, data }) => {
      const crc = checksum(data);
      const local = new Uint8Array(30);
      const localView = new DataView(local.buffer);
      localView.setUint32(0, 0x04034b50, true);
      localView.setUint16(4, 20, true);
      localView.setUint16(12, 33, true);
      localView.setUint32(14, crc, true);
      localView.setUint32(18, data.length, true);
      localView.setUint32(22, data.length, true);
      localView.setUint16(26, name.length, true);
      localParts.push(local, name, data);

      const central = new Uint8Array(46);
      const centralView = new DataView(central.buffer);
      centralView.setUint32(0, 0x02014b50, true);
      centralView.setUint16(4, 20, true);
      centralView.setUint16(6, 20, true);
      centralView.setUint16(14, 33, true);
      centralView.setUint32(16, crc, true);
      centralView.setUint32(20, data.length, true);
      centralView.setUint32(24, data.length, true);
      centralView.setUint16(28, name.length, true);
      centralView.setUint32(42, offset, true);
      centralParts.push(central, name);
      offset += local.length + name.length + data.length;
    });
    const centralDirectory = join(centralParts);
    const end = new Uint8Array(22);
    const endView = new DataView(end.buffer);
    endView.setUint32(0, 0x06054b50, true);
    endView.setUint16(8, files.length, true);
    endView.setUint16(10, files.length, true);
    endView.setUint32(12, centralDirectory.length, true);
    endView.setUint32(16, offset, true);
    return new Blob([...localParts, centralDirectory, end], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
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

    async detectImage({ file, conf, enhance = "none", brightness = 0 }) {
      if (USE_MOCK) {
        const imageUrl = URL.createObjectURL(file);
        const total = Math.floor(Math.random() * 10) + 3;
        const avgConf = Number((0.68 + Math.random() * 0.27).toFixed(2));
        const baselineTotal = Math.max(0, total + Math.floor(Math.random() * 5) - 2);
        const baseline = enhance !== "none" || Number(brightness) !== 0
          ? {
            total: baselineTotal,
            avg_conf: Number((0.65 + Math.random() * 0.28).toFixed(2)),
          }
          : null;
        const result = {
          original_url: imageUrl,
          enhanced_url: imageUrl,
          result_url: imageUrl,
          total,
          avg_conf: avgConf,
          baseline,
          model: mockActive,
        };
        addReportRow({
          file_name: file.name,
          type: "Foto",
          status: "Selesai",
          enhance,
          total,
          avg_conf: avgConf,
          model: mockActive,
        });
        return result;
      }
      const form = new FormData();
      form.append("file", file);
      form.append("conf", String(conf));
      form.append("enhance", enhance);
      form.append("brightness", String(brightness));
      return request("/api/detect_image", { method: "POST", body: form });
    },

    async processVideo({ file, conf, enhance = "none", brightness = 0 }) {
      if (USE_MOCK) {
        const jobId = `mock-${Date.now()}`;
        mockJobs.set(jobId, {
          polls: 0,
          reported: false,
          file_name: file.name,
          enhance,
          brightness,
          model: mockActive,
          status: "queued",
          percent: 0,
          result_url: URL.createObjectURL(file),
          stats: { unique_total: 0 },
          baseline: enhance !== "none" || Number(brightness) !== 0
            ? { total: 9, avg_conf: 0.78 }
            : null,
          message: "Video masuk antrean.",
        });
        return { job_id: jobId };
      }
      const form = new FormData();
      form.append("file", file);
      form.append("conf", String(conf));
      form.append("enhance", enhance);
      form.append("brightness", String(brightness));
      return request("/api/process_video", { method: "POST", body: form });
    },

    async getProgress(jobId) {
      if (USE_MOCK) {
        const job = mockJobs.get(jobId);
        if (!job) return {
          status: "error",
          percent: 0,
          result_url: null,
          stats: { unique_total: 0 },
          baseline: null,
          message: "Pekerjaan video tidak ditemukan.",
        };
        if (job.status !== "done") {
          job.polls += 1;
          const steps = [20, 47, 73, 100];
          job.percent = steps[Math.min(job.polls - 1, steps.length - 1)];
          job.status = job.percent === 100 ? "done" : "running";
          job.message = job.status === "done" ? "Pemrosesan video selesai." : `Memproses video (${job.percent}%).`;
          job.stats = { unique_total: job.status === "done" ? 14 : Math.min(12, job.polls * 3) };
          if (job.status === "done" && !job.reported) {
            addReportRow({
              file_name: job.file_name,
              type: "Video",
              status: "Selesai",
              enhance: job.enhance,
              total: job.stats.unique_total,
              avg_conf: 0.83,
              model: job.model,
            });
            job.reported = true;
          }
        }
        return {
          status: job.status,
          percent: job.percent,
          result_url: job.status === "done" ? job.result_url : null,
          stats: { ...job.stats },
          baseline: job.baseline,
          message: job.message,
        };
      }
      return request(`/api/progress/${encodeURIComponent(jobId)}`);
    },

    async getReport() {
      if (USE_MOCK) return { rows: mockReport.map((row) => ({ ...row })) };
      return request("/api/report", { cache: "no-store" });
    },

    async clearReport() {
      if (USE_MOCK) {
        mockReport = [];
        return { ok: true };
      }
      return request("/api/report/clear", { method: "POST" });
    },

    async exportExcel() {
      if (USE_MOCK) {
        return xlsxBlob(mockReport);
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
  detectTimer: null,
  detectSequence: 0,
  enhancement: "none",
  brightness: 0,
  previewType: null,
  videoFile: null,
  videoSourceUrl: null,
  videoResultUrl: null,
  videoPollSequence: 0,
  reportLoaded: false,
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
  photoFileName: document.getElementById("photo-file-name"),
  photoPreviewSlot: document.getElementById("photo-preview-slot"),
  videoPreviewSlot: document.getElementById("video-preview-slot"),
  mediaPreview: document.getElementById("media-preview"),
  mediaPreviewTitle: document.getElementById("media-preview-title"),
  mediaPreviewHelp: document.getElementById("media-preview-help"),
  photoPreview: document.getElementById("photo-preview"),
  advancedOptions: document.getElementById("advanced-options"),
  enhancement: document.getElementById("enhancement-select"),
  brightness: document.getElementById("brightness-slider"),
  brightnessValue: document.getElementById("brightness-value"),
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
  standingWaterCount: document.getElementById("count-standing-water"),
  buildingFloodedCount: document.getElementById("count-building-flooded"),
  rubbishCount: document.getElementById("count-rubbish"),
  vehicleCount: document.getElementById("count-vehicle"),
  personCount: document.getElementById("count-person"),
  averageConfidence: document.getElementById("average-confidence"),
  resultModel: document.getElementById("result-model"),
  redetect: document.getElementById("redetect-button"),
  comparison: document.getElementById("comparison-section"),
  comparisonMethod: document.getElementById("comparison-method"),
  baselineTotal: document.getElementById("baseline-total"),
  afterTotal: document.getElementById("after-total"),
  baselineConfidence: document.getElementById("baseline-confidence"),
  afterConfidence: document.getElementById("after-confidence"),
  comparisonOriginal: document.getElementById("comparison-original"),
  comparisonEnhanced: document.getElementById("comparison-enhanced"),
  comparisonResult: document.getElementById("comparison-result"),
  videoNoModel: document.getElementById("video-no-model"),
  videoWorkspace: document.getElementById("video-workspace"),
  videoInput: document.getElementById("video-file-input"),
  videoDropzone: document.getElementById("video-dropzone"),
  videoFileName: document.getElementById("video-file-name"),
  videoError: document.getElementById("video-error"),
  processVideo: document.getElementById("process-video-button"),
  videoProcessHint: document.getElementById("video-process-hint"),
  sourceVideo: document.getElementById("source-video"),
  videoProgress: document.getElementById("video-progress"),
  progressMessage: document.getElementById("progress-message"),
  progressPercent: document.getElementById("progress-percent"),
  progressBar: document.getElementById("progress-bar"),
  videoResult: document.getElementById("video-result"),
  outputVideo: document.getElementById("output-video"),
  videoUniqueTotal: document.getElementById("video-unique-total"),
  videoClassCounts: document.getElementById("video-class-counts"),
  videoBaseline: document.getElementById("video-baseline"),
  reportBody: document.getElementById("report-table-body"),
  reportSummary: document.getElementById("report-summary"),
  reportError: document.getElementById("report-error"),
  clearReport: document.getElementById("clear-report-button"),
  exportReport: document.getElementById("export-report-button"),
  downloadVideo: document.getElementById("download-video-button"),
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
  updateMediaPreviewLocation();
  if (tabName === "laporan") loadReport();
}

function formatDate(value) {
  if (!value || typeof value !== "string") return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("id-ID", {
    day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit",
  }).format(date);
}

function makeCell(text, className = "") {
  const cell = document.createElement("td");
  cell.textContent = text;
  if (className) cell.className = className;
  return cell;
}

function reportComparison(before, after, difference, formatter) {
  if (before == null && after == null) return "—";
  if (before == null) return `— → ${formatter(after)}`;
  if (after == null) return `${formatter(before)} → —`;
  const delta = difference == null ? Number(after || 0) - Number(before || 0) : Number(difference);
  const sign = delta > 0 ? "+" : "";
  return `${formatter(before)} → ${formatter(after)} (Δ ${sign}${formatter(delta)})`;
}

function renderModels() {
  const active = state.models.find((model) => model.name === state.activeModel) || null;
  elements.modelState.textContent = active ? "Aktif" : "Belum dipilih";
  elements.modelState.classList.toggle("is-ready", Boolean(active));
  elements.activeModel.textContent = active ? active.name : "Belum ada model aktif";
  elements.noModel.hidden = Boolean(active);
  elements.photoWorkspace.hidden = !active;
  elements.videoNoModel.hidden = Boolean(active);
  elements.videoWorkspace.hidden = !active;
  updateMediaPreviewLocation();
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
      selectButton.className = "text-button model-action-button model-action-select";
      selectButton.dataset.action = "select";
      selectButton.dataset.name = model.name;
      selectButton.textContent = "Jadikan Aktif";
      actionCell.append(selectButton);
    }
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "text-button model-action-button model-action-delete";
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
  if (state.videoSourceUrl) URL.revokeObjectURL(state.videoSourceUrl);
  if (USE_MOCK && state.videoResultUrl) URL.revokeObjectURL(state.videoResultUrl);
  state.previewUrl = null;
  state.mockResultUrl = null;
  state.videoSourceUrl = null;
  state.videoResultUrl = null;
}

function setDetectionLoading(isLoading) {
  elements.imageLoading.hidden = !isLoading;
  elements.redetect.disabled = isLoading;
  elements.dropzone.classList.toggle("is-disabled", isLoading);
}

function formatMetric(value, places = 2) {
  return value == null ? "—" : Number(value).toFixed(places).replace(".", ",");
}

function showImageComparison(result) {
  const baseline = result.baseline;
  elements.comparison.hidden = !baseline;
  if (!baseline) return;

  elements.comparisonMethod.textContent = `${state.enhancement.toUpperCase()} · brightness ${state.brightness > 0 ? "+" : ""}${state.brightness}`;
  elements.baselineTotal.textContent = String(baseline.total ?? "—");
  elements.afterTotal.textContent = String(result.total ?? "—");
  elements.baselineConfidence.textContent = formatMetric(baseline.avg_conf);
  elements.afterConfidence.textContent = formatMetric(result.avg_conf);
  elements.comparisonOriginal.src = result.original_url || "";
  elements.comparisonEnhanced.src = result.enhanced_url || result.original_url || "";
  elements.comparisonResult.src = result.result_url || result.enhanced_url || result.original_url || "";
}

function updateObjectCounts(result) {
  const counts =
    result?.class_counts ||
    result?.classCounts ||
    result?.counts_by_class ||
    result?.countsByClass ||
    result?.counts ||
    {};

  const normalized = Object.fromEntries(
    Object.entries(counts).map(([key, value]) => [
      String(key).trim().toLowerCase().replace(/[\s-]+/g, "_"),
      Number(value) || 0,
    ])
  );

  const getCount = (...keys) => {
    for (const key of keys) {
      const normalizedKey = String(key).trim().toLowerCase().replace(/[\s-]+/g, "_");
      if (Object.prototype.hasOwnProperty.call(normalized, normalizedKey)) {
        return normalized[normalizedKey];
      }
    }
    return 0;
  };

  const values = {
    standingWater: getCount("standing_water"),
    buildingFlooded: getCount("building_flooded"),
    rubbish: getCount("rubbish"),
    vehicle: getCount("vehicle"),
    person: getCount("person"),
  };

  if (elements.standingWaterCount) {
    elements.standingWaterCount.textContent = String(values.standingWater);
  }

  if (elements.buildingFloodedCount) {
    elements.buildingFloodedCount.textContent = String(values.buildingFlooded);
  }

  if (elements.rubbishCount) {
    elements.rubbishCount.textContent = String(values.rubbish);
  }

  if (elements.vehicleCount) {
    elements.vehicleCount.textContent = String(values.vehicle);
  }

  if (elements.personCount) {
    elements.personCount.textContent = String(values.person);
  }

  const hasClassCounts = Object.keys(normalized).length > 0;
  if (hasClassCounts && (result.total == null || Number.isNaN(Number(result.total)))) {
    result.total = Object.values(values).reduce((sum, value) => sum + value, 0);
  }
}

function resetObjectCounts(value = "—") {
  [
    elements.totalCount,
    elements.standingWaterCount,
    elements.buildingFloodedCount,
    elements.rubbishCount,
    elements.vehicleCount,
    elements.personCount,
  ].forEach((element) => {
    if (element) element.textContent = value;
  });
}

async function runDetection() {
  if (!state.currentFile || !state.activeModel) return;
  const requestId = ++state.detectSequence;
  elements.error.hidden = true;
  elements.resultSection.hidden = false;
  setDetectionLoading(true);

  try {
    const result = await api.detectImage({
      file: state.currentFile,
      conf: state.confidence,
      enhance: state.enhancement,
      brightness: state.brightness,
    });
    if (requestId !== state.detectSequence) {
      if (USE_MOCK && result.result_url) URL.revokeObjectURL(result.result_url);
      return;
    }
    if (USE_MOCK && state.mockResultUrl && state.mockResultUrl !== state.previewUrl) URL.revokeObjectURL(state.mockResultUrl);
    state.mockResultUrl = USE_MOCK ? result.result_url : null;
    elements.resultImage.src = result.result_url || result.original_url;
    elements.totalCount.textContent = String(result.total ?? 0);
    updateObjectCounts(result);
    elements.averageConfidence.textContent = result.avg_conf == null ? "—" : formatDecimal(result.avg_conf);
    elements.resultModel.textContent = result.model || state.activeModel || "—";
    showImageComparison(result);
  } catch (error) {
    if (requestId !== state.detectSequence) return;
    elements.error.textContent = error.message || "Deteksi gagal. Periksa foto dan koneksi ke server, lalu coba lagi.";
    elements.error.hidden = false;
  } finally {
    if (requestId === state.detectSequence) setDetectionLoading(false);
  }
}

function selectPhoto(file) {
  if (!file) return;
  if (!file.type.startsWith("image/") && !/\.(jpe?g|png|webp|bmp|gif)$/i.test(file.name)) {
    elements.error.textContent = "File tidak dikenali sebagai foto. Pilih JPG, PNG, atau WEBP.";
    elements.error.hidden = false;
    return;
  }
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  if (USE_MOCK && state.mockResultUrl && state.mockResultUrl !== state.previewUrl) URL.revokeObjectURL(state.mockResultUrl);
  state.detectSequence += 1;
  state.mockResultUrl = null;
  state.currentFile = file;
  state.previewUrl = URL.createObjectURL(file);
  state.previewType = "foto";
  elements.photoPreview.src = state.previewUrl;
  elements.photoPreview.hidden = false;
  elements.sourceVideo.hidden = true;
  updateMediaPreviewLocation();
  const previewImage = new Image();
  previewImage.onload = runDetection;
  previewImage.onerror = () => {
    elements.imageLoading.hidden = true;
    elements.error.textContent = "Foto tidak dapat dibuka. Pilih file gambar lain.";
    elements.error.hidden = false;
  };
  previewImage.src = state.previewUrl;
  elements.resultImage.src = state.previewUrl;
  elements.resultImage.alt = `Pratinjau foto ${file.name}`;
  elements.photoFileName.textContent = file.name;
  elements.photoFileName.hidden = false;
  elements.totalCount.textContent = "—";
  if (elements.standingWaterCount) {
    elements.standingWaterCount.textContent = "—";
  }
  if (elements.buildingFloodedCount) {
    elements.buildingFloodedCount.textContent = "—";
  }
  if (elements.rubbishCount) {
    elements.rubbishCount.textContent = "—";
  }
  if (elements.vehicleCount) {
    elements.vehicleCount.textContent = "—";
  }
  if (elements.personCount) {
    elements.personCount.textContent = "—";
  }
  elements.averageConfidence.textContent = "—";
  elements.resultModel.textContent = state.activeModel || "—";
  elements.resultSection.hidden = false;
  elements.comparison.hidden = true;
  elements.error.hidden = true;
  elements.imageLoading.hidden = false;
}

function updateMediaPreviewLocation() {
  const type = state.previewType;
  const hasMedia = type === "foto" ? Boolean(state.currentFile) : Boolean(state.videoFile);
  const destination = state.activeTab === "video" ? elements.videoPreviewSlot : elements.photoPreviewSlot;
  if (!type || !hasMedia || !state.activeModel || state.activeTab !== type || !destination) {
    elements.mediaPreview.hidden = true;
  } else {
    destination.prepend(elements.mediaPreview);
    elements.mediaPreview.hidden = false;
    elements.mediaPreviewTitle.textContent = type === "foto" ? "Preview Foto" : "Preview Video";
    elements.mediaPreviewHelp.textContent = type === "foto"
      ? "Pratinjau foto yang diunggah."
      : "Pratinjau video yang diunggah.";
  }
  destination.append(elements.advancedOptions);
}

function scheduleDetection() {
  window.clearTimeout(state.detectTimer);
  state.detectTimer = window.setTimeout(runDetection, 240);
}

function setVideoSource(file) {
  if (!file) return;
  if (!file.type.startsWith("video/") && !/\.(mp4|mov|webm|m4v|avi|mkv)$/i.test(file.name)) {
    elements.videoError.textContent = "Format video tidak dikenali. Pilih file video yang didukung browser.";
    elements.videoError.hidden = false;
    return;
  }
  if (state.videoSourceUrl) URL.revokeObjectURL(state.videoSourceUrl);
  if (USE_MOCK && state.videoResultUrl) URL.revokeObjectURL(state.videoResultUrl);
  state.videoResultUrl = null;
  state.videoFile = file;
  state.previewType = "video";
  elements.photoPreview.hidden = true;
  elements.sourceVideo.hidden = false;
  updateMediaPreviewLocation();
  elements.videoError.hidden = true;
  elements.videoResult.hidden = true;
  elements.videoProgress.hidden = true;
  elements.videoFileName.textContent = file.name;
  elements.videoFileName.hidden = false;
  elements.processVideo.disabled = true;
  elements.videoProcessHint.textContent = "Memuat frame pertama…";
  state.videoSourceUrl = URL.createObjectURL(file);
  const renderFirstFrame = (attempt = 0) => {
    if (state.videoFile !== file) return;
    const video = elements.sourceVideo;
    if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
      if (attempt < 30) {
        window.requestAnimationFrame(() => renderFirstFrame(attempt + 1));
      } else {
        elements.videoError.textContent = "Frame pertama video belum dapat dibaca. Coba format MP4 atau WebM.";
        elements.videoError.hidden = false;
        elements.videoProcessHint.textContent = "Frame video belum siap.";
      }
      return;
    }

    video.pause();
    elements.processVideo.disabled = false;
    elements.videoProcessHint.textContent = "Video siap diproses.";
  };
  elements.sourceVideo.onloadeddata = () => {
    if (!state.videoFile || state.videoFile !== file) return;
    window.requestAnimationFrame(() => renderFirstFrame());
  };
  elements.sourceVideo.onerror = () => {
    elements.videoError.textContent = "Frame video tidak dapat dibaca oleh browser. Coba format MP4 atau WebM.";
    elements.videoError.hidden = false;
    elements.videoProcessHint.textContent = "Video tidak dapat dibuka.";
  };
  elements.sourceVideo.src = state.videoSourceUrl;
  elements.sourceVideo.load();
}

function renderProgress(progress) {
  const percent = Math.min(100, Math.max(0, Number(progress.percent) || 0));
  elements.progressMessage.textContent = progress.message || "Memproses video…";
  elements.progressPercent.textContent = `${percent}%`;
  elements.progressBar.setAttribute("aria-valuenow", String(percent));
  elements.progressBar.firstElementChild.style.width = `${percent}%`;
}

const classColors = {
    standing_water: "rgb(184, 34, 225)",
    building_flooded: "#D6A600",
    rubbish: "#3E8E5B",
    vehicle: "#3B82C4",
    person: "#C94A4A"
};

function renderVideoClassCounts(counts) {
  elements.videoClassCounts.replaceChildren();
  const rows = Object.entries(counts || {}).sort(([left], [right]) => left.localeCompare(right));
  if (!rows.length) {
    const empty = document.createElement("p");
    empty.className = "video-class-empty";
    empty.textContent = "Tidak ada objek terlacak dengan ID kelas.";
    elements.videoClassCounts.append(empty);
    return;
  }

  rows.forEach(([className, count]) => {
    const card = document.createElement("article");
    card.className = "video-class-card";
    const label = document.createElement("p");
    label.className = "video-class-label";
    label.textContent = className.replace(/[_-]+/g, " ");
    const value = document.createElement("p");
    value.className = "video-class-value";
    value.textContent = String(count);
    card.append(label, value);
    elements.videoClassCounts.append(card);
  });
}

async function pollVideoProgress(jobId, sequence) {
  while (sequence === state.videoPollSequence) {
    const progress = await api.getProgress(jobId);
    renderProgress(progress);
    if (progress.status === "done") {
      if (USE_MOCK && state.videoResultUrl) URL.revokeObjectURL(state.videoResultUrl);
      state.videoResultUrl = progress.result_url;
      elements.outputVideo.src = progress.result_url;
      elements.outputVideo.load();
      elements.videoUniqueTotal.textContent = String(progress.stats?.unique_total ?? 0);
      renderVideoClassCounts(progress.stats?.class_counts);
      elements.videoBaseline.hidden = !progress.baseline;
      elements.videoBaseline.textContent = progress.baseline
        ? `Sebelum enhancement: ${progress.baseline.total} objek, confidence ${formatDecimal(progress.baseline.avg_conf)}.`
        : "";
      elements.videoResult.hidden = false;
      elements.processVideo.disabled = false;
      elements.videoProcessHint.textContent = "Video selesai diproses.";
      if (state.reportLoaded) loadReport();
      return;
    }
    if (progress.status === "error") {
      elements.videoError.textContent = progress.message || "Pemrosesan video gagal.";
      elements.videoError.hidden = false;
      elements.processVideo.disabled = false;
      return;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
  }
}

async function processVideo() {
  if (!state.videoFile || !state.activeModel) return;
  const sequence = ++state.videoPollSequence;
  elements.videoError.hidden = true;
  elements.videoResult.hidden = true;
  elements.videoProgress.hidden = false;
  elements.processVideo.disabled = true;
  elements.videoProcessHint.textContent = "Video sedang diproses…";
  try {
    const job = await api.processVideo({
      file: state.videoFile,
      conf: state.confidence,
      enhance: state.enhancement,
      brightness: state.brightness,
    });
    await pollVideoProgress(job.job_id, sequence);
  } catch (error) {
    elements.videoError.textContent = error.message || "Video gagal diproses. Silakan coba lagi.";
    elements.videoError.hidden = false;
    elements.processVideo.disabled = false;
  }
}

async function loadReport() {
  if (state.reportLoading) return;
  state.reportLoading = true;
  elements.reportError.hidden = true;
  try {
    const result = await api.getReport();
    const rows = Array.isArray(result.rows)
      ? [...result.rows].sort((left, right) => new Date(right.waktu || right.created_at || right.date || 0) - new Date(left.waktu || left.created_at || left.date || 0))
      : [];
    elements.reportBody.replaceChildren();
    if (!rows.length) {
      const row = document.createElement("tr");
      const cell = makeCell("Belum ada aktivitas deteksi. Hasil foto dan video akan muncul di sini.", "table-empty");
      cell.colSpan = 8;
      row.append(cell);
      elements.reportBody.append(row);
    } else {
      rows.forEach((item) => {
        const row = document.createElement("tr");
        const method = item.metode_enhancement || item.enhance || item.method || "none";
        const brightness = Number(item.brightness || 0);
        const confidence = item.conf ?? item.confidence;
        const type = item.jenis || item.type || item.kind || "foto";
        const typeLabel = String(type).toLowerCase() === "video" ? "Video" : "Foto";
        const rawStatus = String(item.status || "Selesai").toLowerCase();
        const statusKind = ["gagal", "error", "failed"].includes(rawStatus)
          ? "failed"
          : ["diproses", "processing", "running"].includes(rawStatus) ? "working" : "done";
        const statusLabel = statusKind === "failed" ? "Gagal" : statusKind === "working" ? "Diproses" : "Selesai";
        const countFormat = (value) => value == null ? "—" : String(Math.round(Number(value)));
        const confidenceFormat = (value) => value == null ? "—" : formatMetric(value);
        row.append(makeCell(formatDate(item.waktu || item.created_at || item.date)));
        row.append(makeCell(item.nama_file || item.file_name || item.filename || item.file || "—", "report-file"));
        row.append(makeCell(typeLabel));
        const statusCell = document.createElement("td");
        statusCell.className = "report-status-cell";
        const statusBadge = document.createElement("span");
        statusBadge.className = `report-status-badge is-${statusKind}`;
        statusBadge.textContent = statusLabel;
        if (item.pesan) statusBadge.title = item.pesan;
        statusCell.append(statusBadge);
        row.append(statusCell);
        row.append(makeCell(`${method} · ${brightness > 0 ? "+" : ""}${brightness} · conf ${confidence == null ? "—" : formatMetric(confidence)}`));
        row.append(makeCell(reportComparison(item.total_sebelum ?? item.baseline?.total, item.total_sesudah ?? item.total, item.selisih_total, countFormat), "numeric-cell"));
        row.append(makeCell(reportComparison(item.avgconf_sebelum ?? item.baseline?.avg_conf, item.avgconf_sesudah ?? item.avg_conf, item.selisih_avgconf, confidenceFormat), "numeric-cell"));
        row.append(makeCell(item.model || "—", "report-model"));
        elements.reportBody.append(row);
      });
    }
    const failedCount = rows.filter((item) => ["gagal", "error", "failed"].includes(String(item.status || "").toLowerCase())).length;
    elements.reportSummary.textContent = failedCount
      ? `${rows.length} aktivitas · ${failedCount} gagal`
      : `${rows.length} aktivitas deteksi`;
    state.reportLoaded = true;
  } catch (error) {
    elements.reportError.textContent = error.message || "Laporan gagal dimuat.";
    elements.reportError.hidden = false;
  } finally {
    state.reportLoading = false;
  }
}

async function clearReport() {
  if (!window.confirm("Kosongkan semua laporan deteksi?")) return;
  try {
    const result = await api.clearReport();
    if (!result.ok) throw new Error("Laporan gagal dikosongkan.");
    await loadReport();
    setToast("Laporan berhasil dikosongkan.");
  } catch (error) {
    elements.reportError.textContent = error.message || "Laporan gagal dikosongkan.";
    elements.reportError.hidden = false;
  }
}

async function exportReport() {
  try {
    const file = await api.exportExcel();
    const url = URL.createObjectURL(file);
    const link = document.createElement("a");
    link.href = url;
    link.download = "laporan-deteksi.xlsx";
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) {
    elements.reportError.textContent = error.message || "Excel gagal diunduh.";
    elements.reportError.hidden = false;
  }
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
      if (!result.ok) throw new Error(result.error || "Model gagal dihapus.");
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
  elements.photoInput.addEventListener("change", () => {
    selectPhoto(elements.photoInput.files[0]);
    elements.photoInput.value = "";
  });
  elements.redetect.addEventListener("click", runDetection);
  elements.modelTable.addEventListener("click", handleModelAction);

  [elements.confidence, elements.settingsConfidence].forEach((slider) => {
    slider.addEventListener("input", () => {
      setConfidence(slider.value);
      scheduleSaveSettings();
      if (state.activeTab === "foto" && state.currentFile) scheduleDetection();
    });
  });

  elements.enhancement.addEventListener("change", () => {
    state.enhancement = elements.enhancement.value;
    if (state.activeTab === "foto" && state.currentFile) scheduleDetection();
  });
  elements.brightness.addEventListener("input", () => {
    state.brightness = Number(elements.brightness.value);
    elements.brightnessValue.value = String(state.brightness);
    elements.brightnessValue.textContent = String(state.brightness);
    if (state.activeTab === "foto" && state.currentFile) scheduleDetection();
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

  elements.videoInput.addEventListener("change", () => {
    setVideoSource(elements.videoInput.files[0]);
    elements.videoInput.value = "";
  });
  ["dragenter", "dragover"].forEach((eventName) => {
    elements.videoDropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.videoDropzone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    elements.videoDropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.videoDropzone.classList.remove("is-dragging");
    });
  });
  elements.videoDropzone.addEventListener("drop", (event) => setVideoSource(event.dataTransfer.files[0]));
  elements.processVideo.addEventListener("click", processVideo);
  elements.clearReport.addEventListener("click", clearReport);
  elements.exportReport.addEventListener("click", exportReport);
  elements.downloadVideo.addEventListener("click", () => {
    if (!state.videoResultUrl) return;
    const link = document.createElement("a");
    link.href = state.videoResultUrl;
    const sourceName = state.videoFile?.name || "video.mp4";
    const baseName = sourceName.replace(/\.[^.]+$/, "");
    link.download = `${baseName}-hasil.mp4`;
    document.body.append(link);
    link.click();
    link.remove();
  });

  document.getElementById("video-upload-model-button").addEventListener("click", () => elements.modelInput.click());

  window.addEventListener("beforeunload", clearImageUrls);
}

async function initialize() {
  bindEvents();
  setTab("foto");
  elements.brightnessValue.value = "0";
  elements.brightnessValue.textContent = "0";
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
