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

let translations = {};
let currentAlgorithms = {};
let pollInterval = null;
let modelRefreshInterval = null;
let imgModelRefreshInterval = null;

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

let ADMIN_PASSWORD = "88274";

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

function pt(prefix, n) {
  const _n = Math.abs(n);
  if (_n === 1) return t(prefix + "_1");
  if (_n % 10 >= 2 && _n % 10 <= 4 && (_n % 100 < 10 || _n % 100 >= 20)) return t(prefix + "_2");
  return t(prefix + "_5");
}

async function loadLanguage() {
  const data = await api("/api/language");
  await loadTranslations(data.language);
}

// ── Mode Tabs ──

document.querySelectorAll(".mode-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mode-tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".mode-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.mode + "Mode").classList.add("active");
  });
});

// ── CSV Upload ──

uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

async function handleFile(file) {
  if (!file.name.endsWith(".csv")) {
    alert(t("csv_only"));
    return;
  }
  const text = await file.text();
  const base64 = btoa(unescape(encodeURIComponent(text)));
  try {
    const result = await api("/api/upload", {
      method: "POST",
      body: JSON.stringify({ filename: file.name, data: base64 })
    });
    showDatasetInfo(result);
  } catch (err) {
    alert(err.message);
  }
  fileInput.value = "";
}

function showDatasetInfo(data) {
  datasetInfo.hidden = false;
  trainSection.style.display = "";
  uploadStatus.textContent = data.filename;
  datasetFilename.textContent = t("dataset_file").replace("{file}", data.filename);
  const r = pt("rows", data.total_lines).replace("{n}", data.total_lines);
  const c = pt("cols", data.columns.length).replace("{n}", data.columns.length);
  datasetRows.textContent = r + ", " + c;

  const thead = previewTable.querySelector("thead");
  const tbody = previewTable.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";

  if (data.columns) {
    const tr = document.createElement("tr");
    for (const col of data.columns) {
      const th = document.createElement("th");
      th.textContent = col;
      tr.appendChild(th);
    }
    thead.appendChild(tr);
  }

  if (data.preview) {
    for (const row of data.preview) {
      const tr = document.createElement("tr");
      for (const val of row) {
        const td = document.createElement("td");
        td.textContent = val;
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
  }

  featuresCheckboxes.innerHTML = "";
  targetRadios.innerHTML = "";
  const colTypes = data.col_types || {};

  for (const col of (data.columns || [])) {
    const label = document.createElement("label");
    label.className = "checkbox-label";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.name = "feature";
    cb.value = col;
    cb.checked = true;
    label.appendChild(cb);
    const txt = document.createElement("span");
    txt.textContent = col;
    label.appendChild(txt);
    if (colTypes[col]) {
      const badge = document.createElement("span");
      badge.className = "col-badge";
      badge.textContent = colTypes[col];
      label.appendChild(badge);
    }
    featuresCheckboxes.appendChild(label);
  }

  const cols = data.columns || [];
  for (const col of cols) {
    const label = document.createElement("label");
    label.className = "radio-label";
    const rb = document.createElement("input");
    rb.type = "radio";
    rb.name = "target";
    rb.value = col;
    label.appendChild(rb);
    label.appendChild(document.createTextNode(" " + col));
    targetRadios.appendChild(label);
  }
  if (targetRadios.lastElementChild) {
    targetRadios.lastElementChild.querySelector("input").checked = true;
  }
  syncFeatureTarget();
  targetRadios.querySelectorAll("input").forEach(rb => {
    rb.addEventListener("change", syncFeatureTarget);
  });
}

function syncFeatureTarget() {
  const targetRb = document.querySelector('input[name="target"]:checked');
  if (!targetRb) return;
  const targetVal = targetRb.value;
  document.querySelectorAll('input[name="feature"]').forEach(cb => {
    const isTarget = cb.value === targetVal;
    cb.disabled = isTarget;
    if (isTarget) cb.checked = false;
  });
}

clearDataset.addEventListener("click", async () => {
  await api("/api/dataset/clear", { method: "POST", body: "{}" });
  datasetInfo.hidden = true;
  uploadStatus.textContent = t("upload_pending");
  trainSection.style.display = "none";
  outputSection.style.display = "none";
  resultsSection.style.display = "none";
  outputLog.textContent = "";
  disableResults();
  if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
  fileInput.value = "";
});

// ── Algorithms ──

async function loadAlgorithms() {
  currentAlgorithms = await api("/api/algorithms") || {};
  algorithmSelect.innerHTML = "";
  for (const [key, cfg] of Object.entries(currentAlgorithms)) {
    if (cfg.available === false) continue;
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = t("algo_label_" + key) || cfg.label || key;
    algorithmSelect.appendChild(opt);
  }
  renderHyperParams();
  updateTooltip();
}

function updateTooltip() {
  const algoKey = algorithmSelect.value;
  const cfg = currentAlgorithms[algoKey];
  if (cfg && cfg.description) {
    algoTooltipText.textContent = t(`algo_desc_${algoKey}`) || cfg.description;
    algoTooltipIcon.style.display = "inline-flex";
  } else {
    algoTooltipText.textContent = "";
    algoTooltipIcon.style.display = "none";
  }
}

function renderHyperParams() {
  const algoKey = algorithmSelect.value;
  const cfg = currentAlgorithms[algoKey];
  if (!cfg || !cfg.params) { hyperParams.innerHTML = ""; return; }

  let html = '<p class="params-title">' + t("hyperparams_title") + '</p>';
  for (const [key, param] of Object.entries(cfg.params)) {
    const label = t("param_" + key) || key.replace(/_/g, " ");
    const val = param.default;
    if (param.type === "int") {
      html += `<label class="param-field"><span>${label}</span><input type="number" name="param_${key}" value="${val}" min="${param.min}" max="${param.max}" step="1"></label>`;
    } else if (param.type === "float") {
      html += `<label class="param-field"><span>${label}</span><input type="number" name="param_${key}" value="${val}" min="${param.min}" max="${param.max}" step="any"></label>`;
    } else if (param.type === "select") {
      html += `<label class="param-field"><span>${label}</span><select name="param_${key}">`;
      for (const opt of (param.options || [])) {
        html += `<option value="${opt}"${opt === val ? " selected" : ""}>${opt}</option>`;
      }
      html += `</select></label>`;
    } else if (param.type === "string") {
      html += `<label class="param-field"><span>${label}</span><input type="text" name="param_${key}" value="${val}" placeholder="${param.hint || ""}"></label>`;
    }
  }
  hyperParams.innerHTML = html;
}

algorithmSelect.addEventListener("change", () => {
  renderHyperParams();
  updateTooltip();
});

async function loadImageAlgorithms() {
  imgCurrentAlgorithms = await api("/api/image_algorithms") || {};
  imgAlgorithmSelect.innerHTML = "";
  for (const [key, cfg] of Object.entries(imgCurrentAlgorithms)) {
    if (cfg.available === false) continue;
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = t("img_algo_label_" + key) || cfg.label || key;
    imgAlgorithmSelect.appendChild(opt);
  }
  renderImgHyperParams();
  updateImgTooltip();
}

function updateImgTooltip() {
  const algoKey = imgAlgorithmSelect.value;
  const cfg = imgCurrentAlgorithms[algoKey];
  if (cfg && cfg.description) {
    imgAlgoTooltipText.textContent = t("img_algo_desc_" + algoKey) || cfg.description;
    imgAlgoTooltipIcon.style.display = "inline-flex";
  } else {
    imgAlgoTooltipText.textContent = "";
    imgAlgoTooltipIcon.style.display = "none";
  }
}

function renderImgHyperParams() {
  const algoKey = imgAlgorithmSelect.value;
  const cfg = imgCurrentAlgorithms[algoKey];
  if (!cfg || !cfg.params) { imgHyperParams.innerHTML = ""; return; }

  let html = '<p class="params-title">' + t("hyperparams_title") + '</p>';
  for (const [key, param] of Object.entries(cfg.params)) {
    const label = t("param_" + key) || key.replace(/_/g, " ");
    const val = param.default;
    if (param.type === "int") {
      html += `<label class="param-field"><span>${label}</span><input type="number" name="imgparam_${key}" value="${val}" min="${param.min}" max="${param.max}" step="1"></label>`;
    } else if (param.type === "float") {
      html += `<label class="param-field"><span>${label}</span><input type="number" name="imgparam_${key}" value="${val}" min="${param.min}" max="${param.max}" step="any"></label>`;
    } else if (param.type === "select") {
      html += `<label class="param-field"><span>${label}</span><select name="imgparam_${key}">`;
      for (const opt of (param.options || [])) {
        html += `<option value="${opt}"${opt === val ? " selected" : ""}>${opt}</option>`;
      }
      html += `</select></label>`;
    } else if (param.type === "string") {
      html += `<label class="param-field"><span>${label}</span><input type="text" name="imgparam_${key}" value="${val}" placeholder="${param.hint || ""}"></label>`;
    }
  }
  imgHyperParams.innerHTML = html;
}

// Image seed and negative toggles
const imgShuffleCheckbox = document.getElementById("imgShuffle");
function toggleImgSeedField() {
  const seedLabel = document.getElementById("imgSeedLabel");
  if (!seedLabel) return;
  seedLabel.style.display = imgShuffleCheckbox.checked ? "" : "none";
}
imgShuffleCheckbox.addEventListener("change", toggleImgSeedField);

const imgEnableNegCheckbox = document.getElementById("imgEnableNeg");
function toggleImgNegField() {
  const negLabel = document.getElementById("imgNegLabel");
  if (!negLabel) return;
  negLabel.style.display = imgEnableNegCheckbox.checked ? "" : "none";
}
imgEnableNegCheckbox.addEventListener("change", toggleImgNegField);

const shuffleCheckbox = document.querySelector('input[name="shuffle"]');
function toggleSeedField() {
  const seedField = document.getElementById("randomSeedField");
  if (!seedField) return;
  seedField.style.display = shuffleCheckbox.checked ? "" : "none";
}
shuffleCheckbox.addEventListener("change", toggleSeedField);

const dualSlider = document.getElementById("dualSlider");
const thumbTrain = document.getElementById("thumbTrain");
const thumbTest = document.getElementById("thumbTest");
const fillTrain = document.getElementById("fillTrain");
const fillVal = document.getElementById("fillVal");
const fillTest = document.getElementById("fillTest");

function getThumbPos(thumb) {
  return parseFloat(thumb.style.left) || 70;
}

function setThumbPos(thumb, pct) {
  thumb.style.left = Math.round(pct) + "%";
}

function updateSplitBar() {
  const t = getThumbPos(thumbTrain);
  const e = getThumbPos(thumbTest);
  const train = t;
  const val = e - t;
  const test = 100 - e;
  fillTrain.style.width = train + "%";
  fillTrain.textContent = Math.round(train) + "%";
  fillVal.style.width = val + "%";
  fillVal.textContent = Math.round(val) + "%";
  fillTest.style.width = test + "%";
  fillTest.textContent = Math.round(test) + "%";
}

(function initSlider() {
  const sliderRect = dualSlider.getBoundingClientRect();
  let activeThumb = null;

  function posFromMouse(clientX) {
    const rect = dualSlider.getBoundingClientRect();
    return ((clientX - rect.left) / rect.width) * 100;
  }

  function onDown(e) {
    const target = e.target.closest(".dual-thumb");
    if (!target) return;
    activeThumb = target;
    e.preventDefault();
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }

  function onMove(e) {
    if (!activeThumb) return;
    let pct = posFromMouse(e.clientX);
    pct = Math.max(5, Math.min(95, pct));
    const t = getThumbPos(thumbTrain);
    const e2 = getThumbPos(thumbTest);
    if (activeThumb === thumbTrain) {
      pct = Math.min(pct, e2 - 5);
    } else {
      pct = Math.max(pct, t + 5);
    }
    setThumbPos(activeThumb, pct);
    updateSplitBar();
  }

  function onUp() {
    activeThumb = null;
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
  }

  dualSlider.addEventListener("mousedown", onDown);
})();

function toggleValStrategy() {
  const val = document.querySelector('input[name="val_strategy"]:checked');
  const splitBar = document.getElementById("splitBarSection");
  const kfoldConfig = document.getElementById("kfoldConfig");
  if (val && val.value === "kfold") {
    splitBar.style.display = "none";
    kfoldConfig.style.display = "";
  } else {
    splitBar.style.display = "";
    kfoldConfig.style.display = "none";
  }
}

document.querySelectorAll('input[name="val_strategy"]').forEach(rb => {
  rb.addEventListener("change", toggleValStrategy);
});

// ── Training ──

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

trainForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const targetRb = document.querySelector('input[name="target"]:checked');
  if (!targetRb) { alert(t("select_target")); return; }
  const featureCbs = document.querySelectorAll('input[name="feature"]:checked');
  if (featureCbs.length === 0) { alert(t("select_features")); return; }
  const targetCol = targetRb.value;
  const featureCols = Array.from(featureCbs).map(cb => cb.value).filter(c => c !== targetCol);
  if (featureCols.length === 0) { alert(t("target_not_feature")); return; }

  const params = {};
  for (const input of hyperParams.querySelectorAll("input, select")) {
    const match = input.name.match(/^param_(.+)/);
    if (match) {
      params[match[1]] = input.type === "number" ? parseFloat(input.value) : input.value;
    }
  }

  const fd = trainForm.elements;
  const algorithm = algorithmSelect.value;
  const valStrategy = document.querySelector('input[name="val_strategy"]:checked').value;
  const taskType = fd.task_type.value;
  const shuffle = shuffleCheckbox.checked;
  const randomState = parseInt(fd.random_state.value) || 42;

  trainBtn.disabled = true;
  trainBtn.textContent = t("training") + "...";
  outputSection.style.display = "";
  disableResults();

  try {
    await api("/api/train/start", {
      method: "POST",
      body: JSON.stringify({
        target_col: targetCol,
        feature_cols: featureCols,
        algorithm: algorithm,
        params: params,
        task_type: taskType,
        shuffle: shuffle,
        random_state: randomState,
        val_strategy: valStrategy,
        test_size: valStrategy === "simple" ? (100 - getThumbPos(thumbTest)) / 100 : 0.0,
        val_size: valStrategy === "simple" ? (getThumbPos(thumbTest) - getThumbPos(thumbTrain)) / 100 : 0.0,
        k_folds: valStrategy === "kfold" ? parseInt(fd.k_folds.value) || 5 : 5,
      })
    });
    startPolling();
  } catch (err) {
    appendLog("ERROR: " + err.message);
    trainBtn.disabled = false;
    trainBtn.textContent = t("train_btn");
  }
});

function startPolling() {
  if (pollInterval) clearInterval(pollInterval);
  let lastLen = 0;

  pollInterval = setInterval(async () => {
    try {
      const status = await api("/api/train/status");
      const log = status.log || [];

      if (log.length > lastLen) {
        for (let i = lastLen; i < log.length; i++) {
          appendLog(log[i]);
        }
        lastLen = log.length;
      }

      if (status.status === "done") {
        clearInterval(pollInterval);
        pollInterval = null;
        trainBtn.disabled = false;
        trainBtn.textContent = t("train_btn");
        showResults(status.metrics);
        if (status.predictions) {
          const task = status.metrics ? status.metrics.task : "regression";
          renderPredictionChart(status.predictions, task);
        }
        enableResults();
      } else if (status.status === "error") {
        clearInterval(pollInterval);
        pollInterval = null;
        trainBtn.disabled = false;
        trainBtn.textContent = t("train_btn");
        appendLog("ERROR: " + (status.error || "Unknown error"));
      }
    } catch (err) {
      // ignore polling errors
    }
  }, 500);
}

const MAX_LOG_LINES = 500;

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

function showResults(metrics) {
  if (!metrics) return;
  resultsSection.style.display = "";
  metricsDisplay.innerHTML = "";

  const task = metrics.task || "classification";
  const metricKeys = task === "classification"
    ? ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    : ["mae", "mse", "rmse", "r2"];

  const labels = {
    accuracy: "Accuracy", precision: "Precision", recall: "Recall",
    f1_score: "F1 Score", roc_auc: "ROC AUC",
    mae: "MAE", mse: "MSE", rmse: "RMSE", r2: "R\u00b2"
  };

  for (const key of metricKeys) {
    const val = metrics[key];
    if (val !== undefined && val !== null && (typeof val === "string" || !isNaN(val))) {
      const div = document.createElement("div");
      div.className = "metric-card";
      div.innerHTML = `<strong>${labels[key] || key}</strong><span class="metric-value">${val}</span>`;
      metricsDisplay.appendChild(div);
    }
  }
}

// ── Save to Server ──

saveConfirmBtn.addEventListener("click", async () => {
  if (!requireAdmin()) return;
  const name = prompt(t("save_prompt"));
  if (!name) return;
  try {
    const result = await api("/api/models/save", {
      method: "POST",
      body: JSON.stringify({ name: name.trim() })
    });
    saveStatus.textContent = t("saved_ok") + ": " + name.trim() + ".model";
    await loadModels();
  } catch (err) {
    saveStatus.textContent = "Error: " + err.message;
  }
});

// ── Output ──

clearOutputBtn.addEventListener("click", async () => {
  await api("/api/output/clear", { method: "POST", body: "{}" });
  outputLog.textContent = "";
});

// ── Server Models ──

async function loadModels() {
  try {
    const data = await api("/api/models?type=csv");
    renderModels(data || []);
  } catch (err) {
    modelList.innerHTML = `<li class="model-empty">Error: ${err.message}</li>`;
  }
}

function renderModels(models) {
  const filter = modelFilter.value.trim().toLowerCase();
  const items = Array.isArray(models) ? models : [];
  modelList.innerHTML = "";
  const filtered = filter
    ? items.filter(m => m.name.toLowerCase().includes(filter))
    : items;
  if (!filtered.length) {
    modelList.innerHTML = `<li class="model-empty">${t("no_models")}</li>`;
    return;
  }
  for (const item of filtered) {
    const li = document.createElement("li");
    li.className = "model-item";

    const meta = document.createElement("div");
    meta.className = "model-meta";
    const nameEl = document.createElement("strong");
    nameEl.textContent = item.stem || item.name;
    const details = document.createElement("span");
    const size = Number(item.size || 0);
    const modified = item.modified ? new Date(item.modified * 1000).toLocaleString() : "";
    details.textContent = formatSize(size) + (modified ? ` \u00b7 ${modified}` : "");
    meta.appendChild(nameEl);
    meta.appendChild(details);

    const actions = document.createElement("div");
    actions.className = "model-actions";

    const dlBtn = document.createElement("a");
    dlBtn.href = `/api/models/download/${encodeURIComponent(item.name)}`;
    dlBtn.className = "button-secondary";
    dlBtn.textContent = t("download");
    dlBtn.download = item.stem ? item.stem + ".model" : item.name;

    const statsBtn = document.createElement("button");
    statsBtn.type = "button";
    statsBtn.className = "button-secondary";
    statsBtn.textContent = t("stats");
    statsBtn.addEventListener("click", async () => {
      try {
        const metrics = await api("/api/models/stats", {
          method: "POST",
          body: JSON.stringify({ name: item.name })
        });
        if (!metrics || Object.keys(metrics).length === 0) {
          alert(t("no_stats"));
          return;
        }
        const lines = Object.entries(metrics).map(([k, v]) => `${k}: ${v}`);
        alert(lines.join("\n"));
      } catch (err) {
        alert("Error: " + err.message);
      }
    });

    const renameBtn = document.createElement("button");
    renameBtn.type = "button";
    renameBtn.className = "button-secondary";
    renameBtn.textContent = t("rename");
    renameBtn.addEventListener("click", async () => {
      if (!requireAdmin()) return;
      const displayName = item.stem || item.name;
      const newName = prompt(t("rename_prompt").replace("{name}", displayName), displayName);
      if (!newName || newName === displayName) return;
      const finalName = newName.endsWith(".model") ? newName : newName + ".model";
      try {
        await api("/api/models/rename", {
          method: "POST",
          body: JSON.stringify({ name: item.name, new_name: finalName })
        });
        await loadModels();
      } catch (err) {
        alert("Error: " + err.message);
      }
    });

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "danger";
    delBtn.textContent = t("delete");
    delBtn.addEventListener("click", async () => {
      if (!requireAdmin()) return;
      const delDisplay = item.stem || item.name;
      if (!confirm(t("delete_confirm").replace("{name}", delDisplay))) return;
      await api("/api/models/delete", {
        method: "POST",
        body: JSON.stringify({ name: item.name })
      });
      await loadModels();
    });

    actions.appendChild(dlBtn);
    actions.appendChild(statsBtn);
    actions.appendChild(renameBtn);
    actions.appendChild(delBtn);
    li.appendChild(meta);
    li.appendChild(actions);
    modelList.appendChild(li);
  }
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

function startModelRefresh() {
  if (modelRefreshInterval) clearInterval(modelRefreshInterval);
  modelRefreshInterval = setInterval(loadModels, 30000);
}
function startImageModelRefresh() {
  if (imgModelRefreshInterval) clearInterval(imgModelRefreshInterval);
  imgModelRefreshInterval = setInterval(loadImageModels, 30000);
}

const modelFilter = document.getElementById("modelFilter");

modelFilter.addEventListener("input", () => {
  loadModels();
});

async function loadImageModels() {
  try {
    const data = await api("/api/models?type=image");
    renderImageModels(data || []);
  } catch (err) {
    imgModelList.innerHTML = `<li class="model-empty">Error: ${err.message}</li>`;
  }
}

function renderImageModels(models) {
  const filter = imgModelFilter.value.trim().toLowerCase();
  const filtered = (Array.isArray(models) ? models : [])
    .filter(m => !filter || m.name.toLowerCase().includes(filter));

  imgModelList.innerHTML = "";
  if (!filtered.length) {
    imgModelList.innerHTML = `<li class="model-empty">${t("no_models")}</li>`;
    return;
  }
  for (const item of filtered) {
    const li = document.createElement("li");
    li.className = "model-item";
    const meta = document.createElement("div");
    meta.className = "model-meta";
    const nameEl = document.createElement("strong");
    nameEl.textContent = item.stem || item.name;
    const details = document.createElement("span");
    const size = Number(item.size || 0);
    const modified = item.modified ? new Date(item.modified * 1000).toLocaleString() : "";
    details.textContent = formatSize(size) + (modified ? ` \u00b7 ${modified}` : "");
    meta.appendChild(nameEl);
    meta.appendChild(details);

    const actions = document.createElement("div");
    actions.className = "model-actions";

    const dlBtn = document.createElement("a");
    dlBtn.href = `/api/models/download/${encodeURIComponent(item.name)}?type=image`;
    dlBtn.className = "button-secondary";
    dlBtn.textContent = t("download");
    dlBtn.download = item.stem ? item.stem + ".model" : item.name;

    const statsBtn = document.createElement("button");
    statsBtn.type = "button";
    statsBtn.className = "button-secondary";
    statsBtn.textContent = t("stats");
    statsBtn.addEventListener("click", async () => {
      try {
        const metrics = await api("/api/models/stats", {
          method: "POST",
          body: JSON.stringify({ name: item.name, type: "image" })
        });
        if (!metrics || Object.keys(metrics).length === 0) {
          alert(t("no_stats"));
          return;
        }
        const lines = Object.entries(metrics).map(([k, v]) => `${k}: ${v}`);
        alert(lines.join("\n"));
      } catch (e) {
        alert("Error: " + e.message);
      }
    });

    const renameBtn = document.createElement("button");
    renameBtn.type = "button";
    renameBtn.className = "button-secondary";
    renameBtn.textContent = t("rename");
    renameBtn.addEventListener("click", async () => {
      if (!requireAdmin()) return;
      const displayName = item.stem || item.name;
      const newName = prompt(t("rename_prompt").replace("{name}", displayName), displayName);
      if (!newName || newName === displayName) return;
      const finalName = newName.endsWith(".model") ? newName : newName + ".model";
      try {
        await api("/api/models/rename", {
          method: "POST",
          body: JSON.stringify({ name: item.name, new_name: finalName, type: "image" })
        });
        await loadImageModels();
      } catch (err) {
        alert("Error: " + err.message);
      }
    });

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "danger";
    delBtn.textContent = t("delete");
    delBtn.addEventListener("click", async () => {
      if (!requireAdmin()) return;
      const delDisplay = item.stem || item.name;
      if (!confirm(t("delete_confirm").replace("{name}", delDisplay))) return;
      try {
        await api("/api/models/delete", {
          method: "POST",
          body: JSON.stringify({ name: item.name, type: "image" })
        });
        await loadImageModels();
      } catch (err) {
        alert("Error: " + err.message);
      }
    });

    actions.appendChild(dlBtn);
    actions.appendChild(statsBtn);
    actions.appendChild(renameBtn);
    actions.appendChild(delBtn);

    li.appendChild(meta);
    li.appendChild(actions);
    imgModelList.appendChild(li);
  }
}

// ── Prediction Chart ──

function renderPredictionChart(predictions, task) {
  const svg = document.getElementById("chartSvg");
  const section = document.getElementById("chartSection");
  const wrap = document.getElementById("chartWrap");
  section.style.display = "";

  let yTest = predictions.y_test || [];
  let yPred = predictions.y_pred || [];
  if (yTest.length === 0 || yPred.length === 0) return;

  const maxPts = 500;
  if (yTest.length > maxPts) {
    const step = Math.ceil(yTest.length / maxPts);
    yTest = yTest.filter((_, i) => i % step === 0);
    yPred = yPred.filter((_, i) => i % step === 0);
  }

  const w = wrap.clientWidth || 600;
  const h = 300;
  const pad = { top: 20, right: 20, bottom: 40, left: 50 };
  const pw = w - pad.left - pad.right;
  const ph = h - pad.top - pad.bottom;

  const all = yTest.concat(yPred);
  const xMin = Math.min(...all);
  const xMax = Math.max(...all);
  const margin = (xMax - xMin) * 0.1 || 1;
  const xMin2 = xMin - margin;
  const xMax2 = xMax + margin;

  const sx = (v) => pad.left + ((v - xMin2) / (xMax2 - xMin2)) * pw;
  const sy = (v) => pad.top + ph - ((v - xMin2) / (xMax2 - xMin2)) * ph;

  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("width", w);
  svg.setAttribute("height", h);
  svg.innerHTML = "";

  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", sx(xMin2));
  line.setAttribute("y1", sy(xMin2));
  line.setAttribute("x2", sx(xMax2));
  line.setAttribute("y2", sy(xMax2));
  line.setAttribute("stroke", "#ccc");
  line.setAttribute("stroke-dasharray", "4");
  svg.appendChild(line);

  for (let i = 0; i < yTest.length; i++) {
    const cx = sx(yPred[i]);
    const cy = sy(yTest[i]);
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", cx);
    circle.setAttribute("cy", cy);
    circle.setAttribute("r", 3);
    circle.setAttribute("fill", "var(--accent, #4361ee)");
    circle.setAttribute("opacity", "0.6");
    svg.appendChild(circle);
  }

  const lblX = document.createElementNS("http://www.w3.org/2000/svg", "text");
  lblX.setAttribute("x", pad.left + pw / 2);
  lblX.setAttribute("y", h - 5);
  lblX.setAttribute("text-anchor", "middle");
  lblX.setAttribute("font-size", "12");
  lblX.setAttribute("fill", "var(--muted, #888)");
  lblX.textContent = t("chart_predicted");
  svg.appendChild(lblX);

  const lblY = document.createElementNS("http://www.w3.org/2000/svg", "text");
  lblY.setAttribute("x", 12);
  lblY.setAttribute("y", pad.top + ph / 2);
  lblY.setAttribute("text-anchor", "middle");
  lblY.setAttribute("font-size", "12");
  lblY.setAttribute("fill", "var(--muted, #888)");
  lblY.setAttribute("transform", `rotate(-90, 12, ${pad.top + ph / 2})`);
  lblY.textContent = t("chart_actual");
  svg.appendChild(lblY);

  const tbody = document.getElementById("predTbody");
  tbody.innerHTML = "";
  const fullYTest = predictions.y_test || [];
  const fullYPred = predictions.y_pred || [];
  const maxRows = Math.min(fullYTest.length, 1000);
  for (let i = 0; i < maxRows; i++) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i + 1}</td><td>${fullYTest[i]}</td><td>${fullYPred[i]}</td>`;
    tbody.appendChild(tr);
  }
}

// ═══════════════════════════════════════════════════════════════════
// IMAGE MODE
// ═══════════════════════════════════════════════════════════════════

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
const imgDualSlider = document.getElementById("imgDualSlider");
const imgThumbTrain = document.getElementById("imgThumbTrain");
const imgThumbTest = document.getElementById("imgThumbTest");
const imgFillTrain = document.getElementById("imgFillTrain");
const imgFillVal = document.getElementById("imgFillVal");
const imgFillTest = document.getElementById("imgFillTest");
const imgSaveStatus = document.getElementById("imgSaveStatus");
const imgModelList = document.getElementById("imgModelList");
const imgModelFilter = document.getElementById("imgModelFilter");
const imgAlgorithmSelect = document.getElementById("imgAlgorithmSelect");
const imgAlgoTooltipIcon = document.getElementById("imgAlgoTooltipIcon");
const imgAlgoTooltipText = document.getElementById("imgAlgoTooltipText");
const imgHyperParams = document.getElementById("imgHyperParams");
let imgCurrentAlgorithms = {};
const imgModelsSection = document.getElementById("imgModelsSection");
const imgPredSection = document.getElementById("imgPredSection");
const imgPredGrid = document.getElementById("imgPredGrid");
const imgKFolds = document.getElementById("imgKFolds");
const imgTestGallerySection = document.getElementById("imgTestGallerySection");
const testGalleryCanvas = document.getElementById("testGalleryCanvas");
const testCanvasPlaceholder = document.getElementById("testCanvasPlaceholder");
const prevTestImgBtn = document.getElementById("prevTestImgBtn");
const nextTestImgBtn = document.getElementById("nextTestImgBtn");
const testImgCounter = document.getElementById("testImgCounter");
const testGalleryCtx = testGalleryCanvas ? testGalleryCanvas.getContext("2d") : null;
let testGalleryState = { testImageIds: [], predictionsMap: {}, currentIndex: 0 };

imgAlgorithmSelect.addEventListener("change", () => {
  renderImgHyperParams();
  updateImgTooltip();
});

imgModelFilter.addEventListener("input", () => {
  loadImageModels();
});

// ── Image dual-slider init ──

function updateImgSplitBar() {
  if (!imgThumbTrain || !imgThumbTest || !imgFillTrain || !imgFillVal || !imgFillTest) return;
  const t = parseFloat(imgThumbTrain.style.left) || 70;
  const e = parseFloat(imgThumbTest.style.left) || 85;
  const train = t;
  const val = e - t;
  const test = 100 - e;
  imgFillTrain.style.width = train + "%";
  imgFillTrain.textContent = Math.round(train) + "%";
  imgFillVal.style.width = val + "%";
  imgFillVal.textContent = Math.round(val) + "%";
  imgFillTest.style.width = test + "%";
  imgFillTest.textContent = Math.round(test) + "%";
  updateImgLockedStates();
}

function updateImgLockedStates() {
  if (!imgState || !imgState.images.length) return;
  const valRb = document.querySelector('input[name="img_val_strategy"]:checked');
  if (valRb && valRb.value === "kfold") {
    imgState.images.forEach(img => img.locked = false);
    updateNavButtons();
    return;
  }
  const e = parseFloat(imgThumbTest.style.left) || 85;
  const t = parseFloat(imgThumbTrain.style.left) || 70;
  const test = 100 - e;
  const val = e - t;
  const total = imgState.images.length;
  // Same logic as backend: round for test/val, train absorbs leftovers
  let testCount = Math.round(total * test / 100);
  let valCount = Math.round(total * val / 100);
  if (testCount + valCount > total) {
    if (val <= test) valCount = total - testCount; else testCount = total - valCount;
  }
  imgState.images.forEach((img, i) => {
    img.locked = (i >= total - testCount);
    img.split = (!img.locked && i >= total - testCount - valCount) ? "val" : img.locked ? "test" : "train";
  });
  updateNavButtons();
}

// Image training state
let imgState = {
  images: [],
  currentIndex: 0,
  isDrawing: false,
  startX: 0,
  startY: 0,
  currentRect: null,
  selectedAnnotation: -1,
  interactionMode: 'idle',
  resizeHandle: null,
  dragOrigAnn: null,
};

(function initImgSlider() {
  if (!imgDualSlider || !imgThumbTrain || !imgThumbTest) return;
  let activeThumb = null;

  function posFromMouse(clientX) {
    const rect = imgDualSlider.getBoundingClientRect();
    return ((clientX - rect.left) / rect.width) * 100;
  }

  function onDown(e) {
    const target = e.target.closest(".dual-thumb");
    if (!target) return;
    activeThumb = target;
    e.preventDefault();
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }

  function onMove(e) {
    if (!activeThumb) return;
    let pct = posFromMouse(e.clientX);
    pct = Math.max(5, Math.min(95, pct));
    const t = parseFloat(imgThumbTrain.style.left) || 70;
    const e2 = parseFloat(imgThumbTest.style.left) || 85;
    if (activeThumb === imgThumbTrain) {
      pct = Math.min(pct, e2 - 5);
    } else {
      pct = Math.max(pct, t + 5);
    }
    activeThumb.style.left = Math.round(pct) + "%";
    updateImgSplitBar();
    updateImageList();
    if (imgState.images[imgState.currentIndex]?.locked) {
      for (let i = imgState.images.length - 1; i >= 0; i--) {
        if (!imgState.images[i].locked) { selectImage(i); break; }
      }
    }
  }

  function onUp() {
    activeThumb = null;
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
  }

  imgDualSlider.addEventListener("mousedown", onDown);
  updateImgSplitBar();
})();

// ── Image val strategy toggle ──

function toggleImgValStrategy() {
  const val = document.querySelector('input[name="img_val_strategy"]:checked');
  const splitBar = document.getElementById("imgSplitBarSection");
  const kfoldConfig = document.getElementById("imgKfoldConfig");
  if (val && val.value === "kfold") {
    splitBar.style.display = "none";
    kfoldConfig.style.display = "";
  } else {
    splitBar.style.display = "";
    kfoldConfig.style.display = "none";
  }
  updateImageList();
}

document.querySelectorAll('input[name="img_val_strategy"]').forEach(rb => {
  rb.addEventListener("change", toggleImgValStrategy);
});

const ctx = canvas.getContext("2d");

imgUploadZone.addEventListener("click", () => imgFileInput.click());
imgUploadZone.addEventListener("dragover", (e) => { e.preventDefault(); imgUploadZone.classList.add("drag-over"); });
imgUploadZone.addEventListener("dragleave", () => imgUploadZone.classList.remove("drag-over"));
imgUploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  imgUploadZone.classList.remove("drag-over");
  const files = e.dataTransfer.files;
  for (const file of files) handleImageFile(file);
});
imgFileInput.addEventListener("change", () => {
  for (const file of imgFileInput.files) handleImageFile(file);
  imgFileInput.value = "";
});

function handleImageFile(file) {
  const validTypes = ["image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"];
  if (!validTypes.includes(file.type)) {
    if (file.name.match(/\.(png|jpg|jpeg|gif|webp|bmp)$/i) === null) {
      alert(t("img_format_error") || "Unsupported image format");
      return;
    }
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    const img = new Image();
    img.onload = () => {
      const id = "img_" + Date.now() + "_" + Math.random().toString(36).slice(2, 6);
      imgState.images.push({
        id,
        filename: file.name,
        dataUrl,
        width: img.width,
        height: img.height,
        annotations: [],
        uploaded: false,
        locked: false,
      });
      updateImageList();
      if (imgState.images.length === 1) {
        selectImage(0);
      } else {
        updateNavButtons();
        updateBoxCount();
      }
    };
    img.src = dataUrl;
  };
  reader.readAsDataURL(file);
}

function updateImageList() {
  updateImgLockedStates();
  const imgs = imgState.images;
  if (imgs.length === 0) {
    imgList.style.display = "none";
    imgUploadStatus.textContent = t("img_no_images") || "no images";
    imgAnnotateSection.style.display = "none";
    imgTrainSection.style.display = "none";
    imgOutputSection.style.display = "none";
    imgResultsSection.style.display = "none";
    imgPredSection.style.display = "none";
    imgTestGallerySection.style.display = "none";
    return;
  }

  imgList.style.display = "";
  imgUploadStatus.textContent = imgs.length + " " + (imgs.length === 1 ? t("img_image") || "image" : t("img_images") || "images");
  imgAnnotateSection.style.display = "";
  imgTrainSection.style.display = "";

  imgThumbnails.innerHTML = "";
  imgs.forEach((imgData, i) => {
    const thumb = document.createElement("div");
    const isLocked = imgData.locked;
    thumb.className = "img-thumb" + (i === imgState.currentIndex ? " active" : "") + (isLocked ? " img-thumb-locked" : "");
    thumb.innerHTML = `<img src="${imgData.dataUrl}" alt="${imgData.filename}"><span class="img-thumb-name">${imgData.filename}</span>`;
    if (imgData.annotations.length > 0) {
      const badge = document.createElement("span");
      badge.className = "img-thumb-badge";
      badge.textContent = imgData.annotations.length;
      thumb.appendChild(badge);
    }
    if (isLocked) {
      thumb.style.cursor = "not-allowed";
    } else {
      thumb.addEventListener("click", () => selectImage(i));
    }
    thumb.draggable = true;
    thumb.dataset.index = i;

    thumb.addEventListener("dragstart", (e) => {
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", i);
      thumb.classList.add("dragging");
    });

    thumb.addEventListener("dragend", () => {
      document.querySelectorAll(".img-thumb.dragging, .img-thumb.drag-over").forEach(el => {
        el.classList.remove("dragging", "drag-over");
      });
    });

    thumb.addEventListener("dragenter", (e) => {
      e.preventDefault();
      if (!thumb.classList.contains("dragging")) {
        thumb.classList.add("drag-over");
      }
    });

    thumb.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
    });

    thumb.addEventListener("dragleave", () => {
      thumb.classList.remove("drag-over");
    });

    thumb.addEventListener("drop", (e) => {
      e.preventDefault();
      e.stopPropagation();
      thumb.classList.remove("drag-over");

      const fromIdx = parseInt(e.dataTransfer.getData("text/plain"));
      const toIdx = parseInt(thumb.dataset.index);
      if (isNaN(fromIdx) || isNaN(toIdx) || fromIdx === toIdx) return;

      const [moved] = imgState.images.splice(fromIdx, 1);
      imgState.images.splice(toIdx, 0, moved);

      if (imgState.currentIndex === fromIdx) {
        imgState.currentIndex = toIdx;
      } else if (fromIdx < toIdx) {
        if (imgState.currentIndex > fromIdx && imgState.currentIndex <= toIdx) {
          imgState.currentIndex--;
        }
      } else {
        if (imgState.currentIndex >= toIdx && imgState.currentIndex < fromIdx) {
          imgState.currentIndex++;
        }
      }

      updateImageList();
    });
    imgThumbnails.appendChild(thumb);
  });
}

function selectImage(index) {
  if (index < 0 || index >= imgState.images.length) return;
  if (imgState.images[index].locked) return;
  imgState.currentIndex = index;
  imgState.selectedAnnotation = -1;
  imgState.interactionMode = 'idle';
  imgState.resizeHandle = null;
  imgState.dragOrigAnn = null;
  imgState.currentRect = null;
  imgState.isDrawing = false;

  updateImageList();
  renderCanvas(index);
  updateNavButtons();
  updateBoxCount();
}

function updateNavButtons() {
  const idx = imgState.currentIndex;
  const total = imgState.images.length;
  if (total === 0) { prevImgBtn.disabled = true; nextImgBtn.disabled = true; imgCounter.textContent = "0 / 0"; return; }
  let prev = idx - 1;
  while (prev >= 0 && imgState.images[prev].locked) prev--;
  let next = idx + 1;
  while (next < total && imgState.images[next].locked) next++;
  prevImgBtn.disabled = prev < 0;
  nextImgBtn.disabled = next >= total;
  imgCounter.textContent = `${idx + 1} / ${total}`;
}

function updateBoxCount() {
  const imgData = imgState.images[imgState.currentIndex];
  if (imgData) {
    boxCount.textContent = imgData.annotations.length;
  }
}

prevImgBtn.addEventListener("click", () => selectImage(imgState.currentIndex - 1));
nextImgBtn.addEventListener("click", () => selectImage(imgState.currentIndex + 1));

clearAllImagesBtn.addEventListener("click", () => {
  if (imgState.images.length === 0) return;
  if (!confirm(t("img_clear_all_confirm") || "Clear all images?")) return;
  imgState.images = [];
  imgState.currentIndex = 0;
  updateImageList();
  canvas.style.display = "none";
  canvasPlaceholder.style.display = "";
  canvas.width = 0;
  canvas.height = 0;
});

// ── Canvas rendering & annotation ──

function renderCanvas(index) {
  const imgData = imgState.images[index];
  if (!imgData) return;

  canvasPlaceholder.style.display = "none";
  canvas.style.display = "";

  if (imgData.locked) {
    canvas.style.pointerEvents = "none";
    const container = canvas.parentElement;
    canvas.width = container.clientWidth - 4;
    canvas.height = 60;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(239,68,68,0.08)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ef4444";
    ctx.font = "bold 14px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("TEST — image locked, not available for annotation", canvas.width / 2, canvas.height / 2);
    return;
  }
  canvas.style.pointerEvents = "";

  const container = canvas.parentElement;
  const maxW = container.clientWidth - 4;
  const maxH = Math.min(window.innerHeight * 0.6, 600);

  let scale = Math.min(maxW / imgData.width, maxH / imgData.height, 1);
  scale = Math.max(scale, 0.1);

  const cw = Math.round(imgData.width * scale);
  const ch = Math.round(imgData.height * scale);
  canvas.width = cw;
  canvas.height = ch;
  canvas.style.width = cw + "px";
  canvas.style.height = ch + "px";

  // Store scale for coordinate transforms
  canvas._scale = scale;
  imgState._baseRendered = false;

  renderBaseLayer(index);
}

// ── Offscreen canvas for base layer caching ──

let offscreenCanvas = null;
let offscreenCtx = null;

function ensureOffscreen(w, h) {
  if (!offscreenCanvas || offscreenCanvas.width !== w || offscreenCanvas.height !== h) {
    offscreenCanvas = document.createElement("canvas");
    offscreenCanvas.width = w;
    offscreenCanvas.height = h;
    offscreenCtx = offscreenCanvas.getContext("2d");
  }
}

function renderBaseLayer(index) {
  const imgData = imgState.images[index];
  if (!imgData) return;
  const w = canvas.width;
  const h = canvas.height;

  ensureOffscreen(w, h);

  const img = new Image();
  img.onload = () => {
    offscreenCtx.drawImage(img, 0, 0, w, h);
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(offscreenCanvas, 0, 0);
    drawAnnotations();
    imgState._baseRendered = true;
  };
  img.src = imgData.dataUrl;
}

function drawAnnotations() {
  const imgData = imgState.images[imgState.currentIndex];
  if (!imgData) return;
  const scale = canvas._scale || 1;

  for (let i = 0; i < imgData.annotations.length; i++) {
    const ann = imgData.annotations[i];
    const x = ann.x * scale, y = ann.y * scale;
    const bw = ann.width * scale, bh = ann.height * scale;
    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = i === imgState.selectedAnnotation ? 3 : 2;
    ctx.strokeRect(x, y, bw, bh);
  }

  if (imgState.selectedAnnotation >= 0 && imgState.selectedAnnotation < imgData.annotations.length) {
    const ann = imgData.annotations[imgState.selectedAnnotation];
    const hs = 6;
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "#ffcc00";
    ctx.lineWidth = 2;
    const hx = ann.x * scale, hy = ann.y * scale;
    const hw = ann.width * scale, hh = ann.height * scale;
    const hp = [
      [hx, hy], [hx + hw / 2, hy], [hx + hw, hy],
      [hx + hw, hy + hh / 2],
      [hx + hw, hy + hh], [hx + hw / 2, hy + hh], [hx, hy + hh],
      [hx, hy + hh / 2],
    ];
    for (const [px, py] of hp) {
      ctx.fillRect(px - hs, py - hs, hs * 2, hs * 2);
      ctx.strokeRect(px - hs, py - hs, hs * 2, hs * 2);
    }
  }

  if (imgState.currentRect) {
    ctx.strokeStyle = "#ff0000";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 3]);
    const r = imgState.currentRect;
    ctx.strokeRect(r.x, r.y, r.w, r.h);
    ctx.setLineDash([]);
  }
}

function renderCurrentFrame() {
  if (!offscreenCanvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(offscreenCanvas, 0, 0);
  drawAnnotations();
}

// ── Canvas mouse events for drawing / moving / resizing boxes ──

const HANDLE_HIT_RADIUS = 10;

function hitTestHandles(cx, cy, x, y, w, h) {
  const r = HANDLE_HIT_RADIUS;
  const mids = [
    ['nw', x, y], ['n', x + w / 2, y], ['ne', x + w, y],
    ['e', x + w, y + h / 2],
    ['se', x + w, y + h], ['s', x + w / 2, y + h], ['sw', x, y + h],
    ['w', x, y + h / 2],
  ];
  for (const [dir, hx, hy] of mids) {
    if (Math.abs(cx - hx) <= r && Math.abs(cy - hy) <= r) return dir;
  }
  return null;
}

function getAnnotationAt(cx, cy, annotations, scale) {
  for (let i = annotations.length - 1; i >= 0; i--) {
    const a = annotations[i];
    const ax = a.x * scale, ay = a.y * scale;
    if (cx >= ax && cx <= ax + a.width * scale && cy >= ay && cy <= ay + a.height * scale) return i;
  }
  return -1;
}

function cursorForHandle(dir) {
  const map = { nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize', n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize' };
  return map[dir] || 'default';
}

canvas.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return;
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const scale = canvas._scale || 1;
  const imgData = imgState.images[imgState.currentIndex];
  if (!imgData) return;

  // 1. Check handles of selected annotation
  if (imgState.selectedAnnotation >= 0 && imgState.selectedAnnotation < imgData.annotations.length) {
    const ann = imgData.annotations[imgState.selectedAnnotation];
    const hx = ann.x * scale, hy = ann.y * scale, hw = ann.width * scale, hh = ann.height * scale;
    const handle = hitTestHandles(cx, cy, hx, hy, hw, hh);
    if (handle) {
      imgState.interactionMode = 'resize';
      imgState.resizeHandle = handle;
      imgState.dragStartX = cx;
      imgState.dragStartY = cy;
      imgState.dragOrigAnn = { x: ann.x, y: ann.y, width: ann.width, height: ann.height };
      return;
    }
  }

  // 2. Check if clicking inside an existing annotation
  const annIdx = getAnnotationAt(cx, cy, imgData.annotations, scale);
  if (annIdx >= 0) {
    imgState.selectedAnnotation = annIdx;
    imgState.interactionMode = 'move';
    imgState.dragStartX = cx;
    imgState.dragStartY = cy;
    const ann = imgData.annotations[annIdx];
    imgState.dragOrigAnn = { x: ann.x, y: ann.y, width: ann.width, height: ann.height };
    renderCurrentFrame();
    return;
  }

  // 3. Deselect and start drawing a new box
  imgState.selectedAnnotation = -1;
  imgState.isDrawing = true;
  imgState.interactionMode = 'draw';
  imgState.startX = cx;
  imgState.startY = cy;
  imgState.currentRect = { x: cx, y: cy, w: 0, h: 0 };
  renderCurrentFrame();
});

canvas.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const scale = canvas._scale || 1;
  const imgData = imgState.images[imgState.currentIndex];
  if (!imgData) return;

  if (imgState.interactionMode === 'draw') {
    imgState.currentRect = {
      x: Math.min(imgState.startX, cx),
      y: Math.min(imgState.startY, cy),
      w: Math.abs(cx - imgState.startX),
      h: Math.abs(cy - imgState.startY),
    };
    renderCurrentFrame();
    return;
  }

  if (imgState.interactionMode === 'move' && imgState.selectedAnnotation >= 0) {
    const dx = (cx - imgState.dragStartX) / scale;
    const dy = (cy - imgState.dragStartY) / scale;
    const orig = imgState.dragOrigAnn;
    const ann = imgData.annotations[imgState.selectedAnnotation];
    if (ann) {
      ann.x = Math.round(Math.max(0, orig.x + dx));
      ann.y = Math.round(Math.max(0, orig.y + dy));
      renderCurrentFrame();
    }
    return;
  }

  if (imgState.interactionMode === 'resize' && imgState.selectedAnnotation >= 0) {
    const dx = (cx - imgState.dragStartX) / scale;
    const dy = (cy - imgState.dragStartY) / scale;
    const orig = imgState.dragOrigAnn;
    const ann = imgData.annotations[imgState.selectedAnnotation];
    if (!ann) return;

    let x = orig.x, y = orig.y, w = orig.width, h = orig.height;
    switch (imgState.resizeHandle) {
      case 'nw': x = orig.x + dx; y = orig.y + dy; w = orig.width - dx; h = orig.height - dy; break;
      case 'n':  y = orig.y + dy; h = orig.height - dy; break;
      case 'ne': y = orig.y + dy; w = orig.width + dx; h = orig.height - dy; break;
      case 'e':  w = orig.width + dx; break;
      case 'se': w = orig.width + dx; h = orig.height + dy; break;
      case 's':  h = orig.height + dy; break;
      case 'sw': x = orig.x + dx; w = orig.width - dx; h = orig.height + dy; break;
      case 'w':  x = orig.x + dx; w = orig.width - dx; break;
    }
    if (w < 3) { w = 3; if (imgState.resizeHandle === 'nw' || imgState.resizeHandle === 'sw' || imgState.resizeHandle === 'w') x = orig.x + orig.width - 3; }
    if (h < 3) { h = 3; if (imgState.resizeHandle === 'nw' || imgState.resizeHandle === 'ne' || imgState.resizeHandle === 'n') y = orig.y + orig.height - 3; }
    ann.x = Math.round(Math.max(0, x));
    ann.y = Math.round(Math.max(0, y));
    ann.width = Math.round(w);
    ann.height = Math.round(h);
    renderCurrentFrame();
    return;
  }

  // Idle: update cursor
  if (imgState.selectedAnnotation >= 0 && imgState.selectedAnnotation < imgData.annotations.length) {
    const ann = imgData.annotations[imgState.selectedAnnotation];
    const ax = ann.x * scale, ay = ann.y * scale, aw = ann.width * scale, ah = ann.height * scale;
    const h = hitTestHandles(cx, cy, ax, ay, aw, ah);
    if (h) { canvas.style.cursor = cursorForHandle(h); return; }
  }
  const idx = getAnnotationAt(cx, cy, imgData.annotations, scale);
  canvas.style.cursor = idx >= 0 ? 'move' : 'default';
});

canvas.addEventListener("mouseup", () => {
  if (imgState.interactionMode === 'draw') {
    imgState.isDrawing = false;
    const r = imgState.currentRect;
    if (r && r.w > 5 && r.h > 5) {
      const scale = canvas._scale || 1;
      const imgData = imgState.images[imgState.currentIndex];
      if (imgData) {
        const idx = imgData.annotations.length;
        imgData.annotations.push({
          x: Math.round(r.x / scale),
          y: Math.round(r.y / scale),
          width: Math.round(r.w / scale),
          height: Math.round(r.h / scale),
        });
        imgState.selectedAnnotation = idx;
        updateBoxCount();
        updateImageList();
      }
    }
    imgState.currentRect = null;
    imgState.interactionMode = 'idle';
    renderCurrentFrame();
    return;
  }

  if (imgState.interactionMode === 'move' || imgState.interactionMode === 'resize') {
    imgState.interactionMode = 'idle';
    imgState.dragOrigAnn = null;
    imgState.resizeHandle = null;
    updateBoxCount();
    updateImageList();
    renderCurrentFrame();
  }
});

canvas.addEventListener("mouseleave", () => {
  if (imgState.interactionMode !== 'idle') {
    imgState.currentRect = null;
    imgState.isDrawing = false;
    imgState.interactionMode = 'idle';
    imgState.dragOrigAnn = null;
    imgState.resizeHandle = null;
    renderCurrentFrame();
  }
});

// ── Annotation tools ──

undoBoxBtn.addEventListener("click", () => {
  const imgData = imgState.images[imgState.currentIndex];
  if (!imgData || imgData.annotations.length === 0) return;
  if (imgData.locked) return;
  imgData.annotations.pop();
  updateBoxCount();
  updateImageList();
  renderBaseLayer(imgState.currentIndex);
});

clearBoxesBtn.addEventListener("click", () => {
  const imgData = imgState.images[imgState.currentIndex];
  if (!imgData || imgData.annotations.length === 0) return;
  if (imgData.locked) return;
  if (!confirm(t("img_clear_boxes_confirm") || "Clear all boxes on this image?")) return;
  imgData.annotations = [];
  updateBoxCount();
  updateImageList();
  renderBaseLayer(imgState.currentIndex);
});

// ── Image Training ──

let imgPollInterval = null;

imgTrainBtn.addEventListener("click", async () => {
  const hasAnnotations = imgState.images.some(img => img.annotations.length > 0);
  if (!hasAnnotations) {
    alert(t("img_no_annotations") || "Please annotate at least one object on at least one image.");
    return;
  }

  // Upload images that haven't been uploaded yet
  try {
    for (const imgData of imgState.images) {
      if (!imgData.uploaded) {
        const base64 = imgData.dataUrl.split(",")[1];
        const result = await api("/api/image_train/upload", {
          method: "POST",
          body: JSON.stringify({ filename: imgData.filename, data: base64 })
        });
        imgData.serverId = result.image_id;
        imgData.uploaded = true;
      }

      // Save annotations
      await api("/api/image_train/annotations", {
        method: "POST",
        body: JSON.stringify({ image_id: imgData.serverId, annotations: imgData.annotations })
      });
    }
  } catch (err) {
    imgAppendLog("ERROR: " + err.message);
    return;
  }

  imgTrainBtn.disabled = true;
  imgTrainBtn.textContent = (t("training") || "Training") + "...";
  imgOutputSection.style.display = "";
  imgResultsSection.style.display = "none";
  imgPredSection.style.display = "none";
  imgTestGallerySection.style.display = "none";

  const valStrategy = document.querySelector('input[name="img_val_strategy"]:checked').value;
  const algorithm = imgAlgorithmSelect.value;
  const enableNeg = document.getElementById("imgEnableNeg").checked;
  const negRaw = document.getElementById("imgNegPerAnn").value;
  const negPerAnn = enableNeg ? (negRaw === "" ? 10 : parseInt(negRaw) || 0) : 0;
  const params = {};
  for (const input of imgHyperParams.querySelectorAll("input, select")) {
    const name = input.getAttribute("name");
    if (name && name.startsWith("imgparam_")) {
      params[name.replace("imgparam_", "")] = input.value;
    }
  }
  const shuffle = document.getElementById("imgShuffle").checked;
  const randomState = parseInt(document.getElementById("imgRandomSeed").value) || 42;

  let valSize = 0, testSize = 0, kFolds = 5;
  if (valStrategy === "simple") {
    const t = parseFloat(imgThumbTrain.style.left || 70);
    const e = parseFloat(imgThumbTest.style.left || 85);
    testSize = (100 - e) / 100;
    valSize = (e - t) / 100;
  } else {
    kFolds = parseInt(imgKFolds.value) || 5;
  }

  try {
    await api("/api/image_train/start", {
      method: "POST",
      body: JSON.stringify({
        val_strategy: valStrategy,
        val_size: valSize,
        test_size: testSize,
        algorithm,
        neg_per_ann: negPerAnn,
        params,
        shuffle,
        random_state: randomState,
        k_folds: kFolds,
        test_image_ids: imgState.images.filter(im => im.locked).map(im => im.serverId),
      })
    });
    startImagePolling();
  } catch (err) {
    imgAppendLog("ERROR: " + err.message);
    imgTrainBtn.disabled = false;
    imgTrainBtn.textContent = t("img_train_btn") || "Train Detector";
  }
});

if (imgClearOutputBtn) {
  imgClearOutputBtn.addEventListener("click", () => {
    imgOutputLog.textContent = "";
  });
}

function startImagePolling() {
  if (imgPollInterval) clearInterval(imgPollInterval);
  let lastLen = 0;

  imgPollInterval = setInterval(async () => {
    try {
      const status = await api("/api/image_train/status");
      const log = status.log || [];

      if (log.length > lastLen) {
        for (let i = lastLen; i < log.length; i++) {
          imgAppendLog(log[i]);
        }
        lastLen = log.length;
      }

      if (status.status === "done") {
        clearInterval(imgPollInterval);
        imgPollInterval = null;
        imgTrainBtn.disabled = false;
        imgTrainBtn.textContent = t("img_train_btn") || "Train Detector";
        showImageResults(status.metrics);
        showImagePredictions(status.predictions);
        showImageTestGallery(status.test_image_ids, status.predictions);
      } else if (status.status === "error") {
        clearInterval(imgPollInterval);
        imgPollInterval = null;
        imgTrainBtn.disabled = false;
        imgTrainBtn.textContent = t("img_train_btn") || "Train Detector";
        imgAppendLog("ERROR: " + (status.error || "Unknown error"));
      }
    } catch (err) {
      // ignore
    }
  }, 500);
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

function showImageResults(metrics) {
  if (!metrics) return;
  imgResultsSection.style.display = "";
  imgMetricsDisplay.innerHTML = "";

  const labels = {
    accuracy: "Training accuracy",
    val_accuracy: "Validation accuracy",
    samples: "Total samples",
    positive_samples: "Positive",
    negative_samples: "Negative",
    annotations: "Annotations",
  };

  const skipKeys = new Set(["algorithm", "val_strategy", "task"]);
  for (const [key, val] of Object.entries(metrics)) {
    if (val !== undefined && val !== null && !skipKeys.has(key)) {
      const div = document.createElement("div");
      div.className = "metric-card";
      div.innerHTML = `<strong>${labels[key] || key}</strong><span class="metric-value">${val}</span>`;
      imgMetricsDisplay.appendChild(div);
    }
  }

  imgDownloadBtn.setAttribute("aria-disabled", "false");
  imgDownloadBtn.href = "/api/image_train/download/token";
  imgSaveBtn.disabled = false;
}

function showImagePredictions(predictions) {
  const filtered = predictions ? predictions.filter(p => !p.sliding_window) : [];
  if (filtered.length === 0) return;
  imgPredSection.style.display = "";
  imgPredGrid.innerHTML = "";
  const predLabel = (v) => v === 1
    ? (t("img_pred_positive") || "Positive")
    : (t("img_pred_negative") || "Negative");
  for (const pred of filtered) {
    const card = document.createElement("div");
    const correct = pred.actual === pred.predicted;
    card.className = "img-pred-card " + (correct ? "correct" : "incorrect");
    card.innerHTML = `
      <img src="data:image/png;base64,${pred.image}" alt="patch">
      <span class="img-pred-label">${t("chart_actual") || "Actual"}: ${predLabel(pred.actual)}</span>
      <span class="img-pred-label">${t("chart_predicted") || "Predicted"}: ${predLabel(pred.predicted)}</span>
      <span class="img-pred-badge">${correct ? "✓" : "✗"}</span>
    `;
    imgPredGrid.appendChild(card);
  }
}

// ── Test Image Gallery ──

function showImageTestGallery(testImageIds, predictions) {
  if (!testImageIds || testImageIds.length === 0) return;
  imgTestGallerySection.style.display = "";

  const predMap = {};
  if (predictions) {
    for (const p of predictions) {
      if (!predMap[p.image_id]) predMap[p.image_id] = [];
      predMap[p.image_id].push(p);
    }
  }

  testGalleryState.testImageIds = testImageIds;
  testGalleryState.predictionsMap = predMap;
  testGalleryState.currentIndex = 0;

  renderTestGalleryImage(0);
  updateTestGalleryNav();
}

function renderTestGalleryImage(index) {
  const imgId = testGalleryState.testImageIds[index];
  if (!imgId) return;

  const imgData = imgState.images.find(im => im.serverId === imgId);
  if (!imgData) return;

  testCanvasPlaceholder.style.display = "none";
  testGalleryCanvas.style.display = "";

  const predictions = testGalleryState.predictionsMap[imgId] || [];
  const container = testGalleryCanvas.parentElement;
  const maxW = container.clientWidth - 4;
  const maxH = window.innerHeight * 0.6;
  let scale = Math.min(maxW / imgData.width, maxH / imgData.height, 1);
  scale = Math.max(scale, 0.1);
  const cw = Math.round(imgData.width * scale);
  const ch = Math.round(imgData.height * scale);
  testGalleryCanvas.width = cw;
  testGalleryCanvas.height = ch;
  testGalleryCanvas.style.width = cw + "px";
  testGalleryCanvas.style.height = ch + "px";
  testGalleryCanvas._scale = scale;

  const img = new Image();
  img.onload = () => {
    testGalleryCtx.clearRect(0, 0, cw, ch);
    testGalleryCtx.drawImage(img, 0, 0, cw, ch);

    if (predictions.length === 0) {
      testGalleryCtx.fillStyle = "rgba(0,0,0,0.4)";
      testGalleryCtx.font = "bold 16px sans-serif";
      testGalleryCtx.textAlign = "center";
      testGalleryCtx.textBaseline = "middle";
      testGalleryCtx.fillText(t("test_no_detections") || "No objects detected", cw / 2, ch / 2);
      return;
    }

    for (const pred of predictions) {
      const x = pred.x * scale;
      const y = pred.y * scale;
      const w = pred.w * scale;
      const h = pred.h * scale;

      if (pred.sliding_window) {
        testGalleryCtx.strokeStyle = "#22c55e";
        testGalleryCtx.lineWidth = 2;
        testGalleryCtx.strokeRect(x, y, w, h);
        testGalleryCtx.fillStyle = "rgba(34,197,94,0.15)";
        testGalleryCtx.fillRect(x, y, w, h);
        testGalleryCtx.fillStyle = "#22c55e";
        testGalleryCtx.font = "bold 11px sans-serif";
        testGalleryCtx.textBaseline = "top";
        testGalleryCtx.fillText("D", x + 2, y + 2);
      } else {
        const correct = pred.actual === pred.predicted;
        testGalleryCtx.strokeStyle = correct ? "#22c55e" : "#ef4444";
        testGalleryCtx.lineWidth = 2;
        testGalleryCtx.strokeRect(x, y, w, h);
        testGalleryCtx.fillStyle = correct ? "rgba(34,197,94,0.15)" : "rgba(239,68,68,0.15)";
        testGalleryCtx.fillRect(x, y, w, h);
        const label = correct
          ? (pred.actual === 1 ? "TP" : "TN")
          : (pred.actual === 1 ? "FN" : "FP");
        testGalleryCtx.fillStyle = correct ? "#22c55e" : "#ef4444";
        testGalleryCtx.font = "bold 11px sans-serif";
        testGalleryCtx.textBaseline = "top";
        testGalleryCtx.fillText(label, x + 2, y + 2);
      }
    }
  };
  img.onerror = () => {
    testGalleryCtx.clearRect(0, 0, cw, ch);
    testGalleryCtx.fillStyle = "#666";
    testGalleryCtx.font = "14px sans-serif";
    testGalleryCtx.textAlign = "center";
    testGalleryCtx.fillText(t("test-canvas-hint") || "Image not available", cw / 2, ch / 2);
  };
  img.src = imgData.dataUrl;
}

function updateTestGalleryNav() {
  const total = testGalleryState.testImageIds.length;
  const idx = testGalleryState.currentIndex;
  testImgCounter.textContent = (idx + 1) + " / " + total;
  prevTestImgBtn.disabled = idx <= 0;
  nextTestImgBtn.disabled = idx >= total - 1;
}

prevTestImgBtn.addEventListener("click", () => {
  if (testGalleryState.currentIndex > 0) {
    testGalleryState.currentIndex--;
    renderTestGalleryImage(testGalleryState.currentIndex);
    updateTestGalleryNav();
  }
});

nextTestImgBtn.addEventListener("click", () => {
  if (testGalleryState.currentIndex < testGalleryState.testImageIds.length - 1) {
    testGalleryState.currentIndex++;
    renderTestGalleryImage(testGalleryState.currentIndex);
    updateTestGalleryNav();
  }
});

// ── Image Save to Server ──

imgSaveBtn.addEventListener("click", async () => {
  if (!requireAdmin()) return;
  const name = prompt(t("save_prompt"));
  if (!name) return;
  try {
    const result = await api("/api/image_train/save", {
      method: "POST",
      body: JSON.stringify({ name: name.trim() })
    });
    imgSaveStatus.textContent = t("saved_ok") + ": " + name.trim() + ".model";
    await loadImageModels();
  } catch (err) {
    imgSaveStatus.textContent = "Error: " + err.message;
  }
});

// ── Init ──

document.addEventListener("DOMContentLoaded", async () => {
  try {
    disableResults();
    await loadConfig();
    await loadLanguage();

    // clear server state from previous sessions (theme/lang survive)
    await api("/api/dataset/clear", { method: "POST", body: "{}" });
    await api("/api/output/clear", { method: "POST", body: "{}" });
    await api("/api/image_train/clear", { method: "POST", body: "{}" });

    await loadAlgorithms();
    await loadImageAlgorithms();
    await loadModels();
    await loadImageModels();
    startModelRefresh();
    startImageModelRefresh();
    toggleSeedField();
    toggleImgSeedField();
    toggleImgNegField();
    updateSplitBar();
    updateImgSplitBar();
    toggleValStrategy();
    toggleImgValStrategy();
  } catch (err) {
    outputLog.textContent = String(err.message || err);
  }
});
