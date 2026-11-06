// ── Inline validation styles ──
(function injectValidationStyles() {
  const style = document.createElement("style");
  style.textContent =
    "input.is-invalid{border-color:var(--danger,#dc3545)!important;background:color-mix(in srgb,var(--danger,#dc3545) 10%,transparent)!important}" +
    "select.is-invalid{border-color:var(--danger,#dc3545)!important}" +
    ".validation-summary{color:var(--danger,#dc3545);font-size:0.9em;padding:0.25em 0}";
  document.head.appendChild(style);
})();

const themeSelect = document.getElementById("themeSelect");
const languageSelect = document.getElementById("languageSelect");
const fileInput = document.getElementById("fileInput");
const uploadZone = document.getElementById("uploadZone");
const datasetInfo = document.getElementById("datasetInfo");
const datasetFilename = document.getElementById("datasetFilename");
const datasetRows = document.getElementById("datasetRows");
const previewTable = document.getElementById("previewTable");
const featuresCheckboxes = document.getElementById("featuresCheckboxes");
const targetRadios = document.getElementById("targetRadios");
const clearDataset = document.getElementById("clearDataset");
const uploadStatus = document.getElementById("uploadStatus");
const algorithmSelect = document.getElementById("algorithmSelect");
const algoTooltipIcon = document.getElementById("algoTooltipIcon");
const algoTooltipText = document.getElementById("algoTooltipText");
const hyperParams = document.getElementById("hyperParams");
const trainForm = document.getElementById("trainForm");
const trainBtn = document.getElementById("trainBtn");
const outputLog = document.getElementById("outputLog");
const clearOutputBtn = document.getElementById("clearOutputBtn");
const trainSection = document.getElementById("trainSection");
const outputSection = document.getElementById("outputSection");
const resultsSection = document.getElementById("resultsSection");
const metricsDisplay = document.getElementById("metricsDisplay");
const downloadBtn = document.getElementById("downloadBtn");
const saveConfirmBtn = document.getElementById("saveConfirmBtn");
const saveStatus = document.getElementById("saveStatus");
const modelList = document.getElementById("modelList");
const modelFilter = document.getElementById("modelFilter");
const imgFileInput = document.getElementById("imgFileInput");
const imgUploadZone = document.getElementById("imgUploadZone");
const imgUploadStatus = document.getElementById("imgUploadStatus");
const imgList = document.getElementById("imgList");
const imgThumbnails = document.getElementById("imgThumbnails");
const clearAllImagesBtn = document.getElementById("clearAllImagesBtn");
const imgAnnotateSection = document.getElementById("imgAnnotateSection");
const canvas = document.getElementById("annotationCanvas");
const canvasPlaceholder = document.getElementById("canvasPlaceholder");
const prevImgBtn = document.getElementById("prevImgBtn");
const nextImgBtn = document.getElementById("nextImgBtn");
const imgCounter = document.getElementById("imgCounter");
const boxCount = document.getElementById("boxCount");
const undoBoxBtn = document.getElementById("undoBoxBtn");
const redoBoxBtn = document.getElementById("redoBoxBtn");
const clearBoxesBtn = document.getElementById("clearBoxesBtn");
const imgTrainSection = document.getElementById("imgTrainSection");
const imgTrainBtn = document.getElementById("imgTrainBtn");
const imgOutputSection = document.getElementById("imgOutputSection");
const imgOutputLog = document.getElementById("imgOutputLog");
const imgClearOutputBtn = document.getElementById("imgClearOutputBtn");
const imgResultsSection = document.getElementById("imgResultsSection");
const imgMetricsDisplay = document.getElementById("imgMetricsDisplay");
const imgDownloadBtn = document.getElementById("imgDownloadBtn");
const imgSaveBtn = document.getElementById("imgSaveBtn");
const imgSaveStatus = document.getElementById("imgSaveStatus");
const imgModelList = document.getElementById("imgModelList");
const imgModelFilter = document.getElementById("imgModelFilter");
const imgModelsSection = document.getElementById("imgModelsSection");
const imgTestGallerySection = document.getElementById("imgTestGallerySection");
const testGalleryCanvas = document.getElementById("testGalleryCanvas");
const testCanvasPlaceholder = document.getElementById("testCanvasPlaceholder");
const prevTestImgBtn = document.getElementById("prevTestImgBtn");
const nextTestImgBtn = document.getElementById("nextTestImgBtn");
const testImgCounter = document.getElementById("testImgCounter");
const testGalleryCtx = testGalleryCanvas ? testGalleryCanvas.getContext("2d") : null;

let translations = {};
let ADMIN_PASSWORD = "88274";
const MAX_LOG_LINES = 500;

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  const response = await fetch(path, { headers, ...options });
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!response.ok) throw new Error((data && data.error) ? data.error : `HTTP ${response.status}`);
  return data;
}

async function loadTranslations(language) {
  let resp = await fetch(`/static/i18n/${language}.json`);
  if (!resp.ok) resp = await fetch("/static/i18n/en.json");
  translations = await resp.json();
}

function t(key) { return translations[key] ?? key; }

function balancedColumns(count) {
  if (count <= 1) return 1;
  for (let cols = Math.min(5, count); cols >= 2; cols--) {
    if (count % cols === 0) return cols;
  }
  return Math.max(2, Math.ceil(count / Math.ceil(count / 5)));
}

function setMetricColumns(container, count) {
  if (!container) return;
  container.style.setProperty("--metric-cols", balancedColumns(count));
}

function pt(prefix, n) {
  const _n = Math.abs(n);
  if (_n === 1) return t(prefix + "_1");
  if (_n % 10 >= 2 && _n % 10 <= 4 && (_n % 100 < 10 || _n % 100 >= 20)) return t(prefix + "_2");
  return t(prefix + "_5");
}

async function loadConfig() {
  try {
    const resp = await fetch("/static/config/train.json");
    const cfg = await resp.json();
    if (cfg.admin_password) ADMIN_PASSWORD = cfg.admin_password;
  } catch {}
}

function requireAdmin() {
  return prompt(t("admin_prompt")) === ADMIN_PASSWORD;
}

async function loadLanguage() {
  const data = await api("/api/language");
  await loadTranslations(data.language);
}

document.querySelectorAll(".mode-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mode-tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".mode-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.mode + "Mode").classList.add("active");
  });
});

function appendLog(line) {
  const text = outputLog.textContent;
  const updated = text ? text + "\n" + line : line;
  const lines = updated.split("\n");
  if (lines.length > MAX_LOG_LINES) {
    outputLog.textContent = lines.slice(lines.length - MAX_LOG_LINES).join("\n");
  } else {
    outputLog.textContent = updated;
  }
  outputLog.scrollTop = outputLog.scrollHeight;
}

function imgAppendLog(line) {
  const text = imgOutputLog.textContent;
  const updated = text ? text + "\n" + line : line;
  const lines = updated.split("\n");
  if (lines.length > MAX_LOG_LINES) {
    imgOutputLog.textContent = lines.slice(lines.length - MAX_LOG_LINES).join("\n");
  } else {
    imgOutputLog.textContent = updated;
  }
  imgOutputLog.scrollTop = imgOutputLog.scrollHeight;
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

function disableResults() {
  downloadBtn.classList.add("btn");
  downloadBtn.setAttribute("aria-disabled", "true");
  downloadBtn.removeAttribute("href");
  saveConfirmBtn.disabled = true;
}

function enableResults() {
  downloadBtn.setAttribute("aria-disabled", "false");
  downloadBtn.href = "/api/train/download/token";
  saveConfirmBtn.disabled = false;
  resultsSection.style.display = "";
}

function disableImgResults() {
  imgDownloadBtn.classList.add("btn");
  imgDownloadBtn.setAttribute("aria-disabled", "true");
  imgDownloadBtn.removeAttribute("href");
  imgSaveBtn.disabled = true;
}

function enableImgResults() {
  imgDownloadBtn.setAttribute("aria-disabled", "false");
  imgDownloadBtn.href = "/api/image_train/download/token";
  imgSaveBtn.disabled = false;
  imgResultsSection.style.display = "";
}
