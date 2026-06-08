const themeSelect = document.getElementById("themeSelect");
const languageSelect = document.getElementById("languageSelect");
const modelFilter = document.getElementById("modelFilter");
const modelSelect = document.getElementById("modelSelect");
const selectModelBtn = document.getElementById("selectModelBtn");
const refreshModelsBtn = document.getElementById("refreshModelsBtn");
const modelUploadZone = document.getElementById("modelUploadZone");
const modelFileInput = document.getElementById("modelFileInput");
const uploadModelStatus = document.getElementById("uploadModelStatus");
const confirmUploadBtn = document.getElementById("confirmUploadBtn");
const csvSection = document.getElementById("csvSection");
const loadedModelPill = document.getElementById("loadedModelPill");
const csvUploadZone = document.getElementById("csvUploadZone");
const csvFileInput = document.getElementById("csvFileInput");
const csvStatus = document.getElementById("csvStatus");
const featureStatus = document.getElementById("featureStatus");
const predictBtn = document.getElementById("predictBtn");
const clearAllBtn = document.getElementById("clearAllBtn");
const resultsSection = document.getElementById("resultsSection");
const resultsInfo = document.getElementById("resultsInfo");
const resultsTable = document.getElementById("resultsTable");
const downloadResultsBtn = document.getElementById("downloadResultsBtn");

let translations = {};
let currentModelState = null;
let uploadToken = null;
let csvToken = null;
let csvColumnNames = [];

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

function pt(prefix, n) {
  const _n = Math.abs(n);
  if (_n === 1) return t(prefix + "_1");
  if (_n % 10 >= 2 && _n % 10 <= 4 && (_n % 100 < 10 || _n % 100 >= 20)) return t(prefix + "_2");
  return t(prefix + "_5");
}

function applyTranslations() {
  document.documentElement.lang = languageSelect.value || "en";
  document.querySelectorAll("[data-i18n]").forEach(node => {
    const key = node.getAttribute("data-i18n");
    const tag = node.tagName.toLowerCase();
    if (tag === "input" || tag === "textarea") {
      node.placeholder = t(key);
    } else {
      node.textContent = t(key);
    }
  });
}

async function loadTheme() {
  const data = await api("/api/theme");
  themeSelect.innerHTML = "";
  const themeLabels = translations.themes || {};
  for (const th of (data.available || [])) {
    const opt = document.createElement("option");
    opt.value = th.key;
    opt.textContent = themeLabels[th.key] || th.label || th.key;
    opt.selected = th.key === data.theme;
    themeSelect.appendChild(opt);
  }
  themeSelect.value = data.theme;
}

async function loadLanguage() {
  const data = await api("/api/language");
  languageSelect.innerHTML = "";
  for (const lang of (data.available || [])) {
    const opt = document.createElement("option");
    opt.value = lang.key;
    opt.textContent = lang.label;
    opt.selected = lang.key === data.language;
    languageSelect.appendChild(opt);
  }
  languageSelect.value = data.language;
  await loadTranslations(data.language);
  applyTranslations();
  await loadTheme();
}

themeSelect.addEventListener("change", async () => {
  await api("/api/theme", { method: "POST", body: JSON.stringify({ theme: themeSelect.value }) });
  window.location.reload();
});
languageSelect.addEventListener("change", async () => {
  await api("/api/language", { method: "POST", body: JSON.stringify({ language: languageSelect.value }) });
  window.location.reload();
});

// ── Tabs ──

document.querySelectorAll("#modelTabBar button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#modelTabBar button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab" + btn.dataset.tab.charAt(0).toUpperCase() + btn.dataset.tab.slice(1)).classList.add("active");
  });
});

// ── Load Models (with filter) ──

async function loadModels() {
  try {
    const models = await api("/api/models") || [];
    const filter = modelFilter.value.trim().toLowerCase();
    const filtered = filter ? models.filter(m => (m.stem || m.name).toLowerCase().includes(filter)) : models;
    modelSelect.innerHTML = `<option value="">${t("test_select_hint")}</option>`;
    for (const m of filtered) {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = m.stem || m.name;
      modelSelect.appendChild(opt);
    }
  } catch (err) {
    modelSelect.innerHTML = `<option value="">Error: ${err.message}</option>`;
  }
}

modelFilter.addEventListener("input", loadModels);
refreshModelsBtn.addEventListener("click", loadModels);

// ── Select model from server ──

selectModelBtn.addEventListener("click", async () => {
  const name = modelSelect.value;
  if (!name) return;
  try {
    const info = await api("/api/test/model_info", {
      method: "POST",
      body: JSON.stringify({ model_name: name })
    });
    currentModelState = { source: "server", name, feature_cols: info.feature_cols || [] };
    loadedModelPill.textContent = (name.endsWith(".model") ? name.slice(0, -6) : name);
    csvSection.style.display = "";
    resultsSection.style.display = "none";
    revalidateFeatures();
  } catch (err) {
    alert("Error: " + err.message);
  }
});

// ── Upload model ──

modelUploadZone.addEventListener("click", () => modelFileInput.click());
modelUploadZone.addEventListener("dragover", (e) => { e.preventDefault(); modelUploadZone.classList.add("drag-over"); });
modelUploadZone.addEventListener("dragleave", () => modelUploadZone.classList.remove("drag-over"));
modelUploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  modelUploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleModelFile(file);
});
modelFileInput.addEventListener("change", () => {
  if (modelFileInput.files[0]) handleModelFile(modelFileInput.files[0]);
});

async function handleModelFile(file) {
  if (!file.name.endsWith(".model")) {
    alert(t("test_model_only"));
    return;
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  const base64 = btoa(binary);
  try {
    const result = await api("/api/test/upload", {
      method: "POST",
      body: JSON.stringify({ data: base64 })
    });
    uploadToken = result.token;
    uploadModelStatus.textContent = t("test_uploaded") + ": " + file.name;
  } catch (err) {
    alert(err.message);
  }
}

confirmUploadBtn.addEventListener("click", () => {
  if (!uploadToken) {
    alert(t("test_upload_first"));
    return;
  }
  currentModelState = { source: "upload", token: uploadToken, feature_cols: [] };
  loadedModelPill.textContent = t("test_uploaded_model");
  csvSection.style.display = "";
  resultsSection.style.display = "none";
  revalidateFeatures();
});

// ── CSV Upload ──

csvUploadZone.addEventListener("click", () => csvFileInput.click());
csvUploadZone.addEventListener("dragover", (e) => { e.preventDefault(); csvUploadZone.classList.add("drag-over"); });
csvUploadZone.addEventListener("dragleave", () => csvUploadZone.classList.remove("drag-over"));
csvUploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  csvUploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleCsvFile(file);
});
csvFileInput.addEventListener("change", () => {
  if (csvFileInput.files[0]) handleCsvFile(csvFileInput.files[0]);
});

async function handleCsvFile(file) {
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
    csvToken = file.name;
    csvColumnNames = result.columns || [];
    csvStatus.textContent = file.name + " (" + pt("rows", result.total_lines).replace("{n}", result.total_lines) + ", " + pt("cols", result.columns.length).replace("{n}", result.columns.length) + ")";

    revalidateFeatures();
  } catch (err) {
    alert(err.message);
  }
  csvFileInput.value = "";
}

// ── Re-validate features after model or CSV change ──

function revalidateFeatures() {
  const modelCols = currentModelState ? currentModelState.feature_cols || [] : [];
  const csvCols = csvColumnNames || [];
  if (csvCols.length > 0 && modelCols.length > 0) {
    const found = modelCols.filter(c => csvCols.includes(c));
    const missing = modelCols.filter(c => !csvCols.includes(c));
    if (missing.length > 0) {
      featureStatus.innerHTML = `<span style="color:var(--muted,#888)">${t("test_skip_cols")}: ${missing.join(", ")}</span>`;
    } else {
      featureStatus.innerHTML = `<span style="color:var(--positive,#4caf50)">\u2713 ${found.length} ${t("test_features_found")}</span>`;
    }
    predictBtn.disabled = missing.length > 0;
  } else if (csvCols.length > 0 && modelCols.length === 0) {
    featureStatus.innerHTML = `<span style="color:var(--muted,#888)">${t("test_columns_loaded").replace("{n}", csvCols.length)}</span>`;
    predictBtn.disabled = false;
  } else {
    featureStatus.innerHTML = "";
    predictBtn.disabled = !csvToken;
  }
}

// ── Predict ──

predictBtn.addEventListener("click", async () => {
  if (!currentModelState || !csvToken) return;

  predictBtn.disabled = true;
  predictBtn.textContent = t("test_predicting") + "...";

  try {
    const body = {};
    if (currentModelState.source === "server") {
      body.model_name = currentModelState.name;
    } else {
      body.upload_token = currentModelState.token;
    }
    const result = await api("/api/test/predict_csv", {
      method: "POST",
      body: JSON.stringify(body)
    });
    showResults(result);
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = t("test_predict_btn");
  }
});

function showResults(result) {
  resultsSection.style.display = "";
  const rows = result.rows || [];
  const predictions = result.predictions || [];
  const hasProba = result.probabilities && result.probabilities.length > 0;

  resultsInfo.textContent = t("test_results_info").replace("{n}", rows.length);

  const thead = resultsTable.querySelector("thead");
  const tbody = resultsTable.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";

  const allCols = Object.keys(rows[0] || {});
  const shownCols = allCols.concat(["prediction"]);
  if (hasProba) shownCols.push("probabilities");

  const tr = document.createElement("tr");
  for (const col of shownCols) {
    const th = document.createElement("th");
    th.textContent = col;
    tr.appendChild(th);
  }
  thead.appendChild(tr);

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const tr2 = document.createElement("tr");
    for (const col of allCols) {
      const td = document.createElement("td");
      td.textContent = row[col] !== undefined ? row[col] : "";
      tr2.appendChild(td);
    }
    const tdPred = document.createElement("td");
    tdPred.textContent = predictions[i] !== undefined ? predictions[i] : "";
    tr2.appendChild(tdPred);
    if (hasProba) {
      const tdProb = document.createElement("td");
      const proba = result.probabilities[i] || [];
      tdProb.textContent = proba.map((p, j) => `c${j}:${(p * 100).toFixed(1)}%`).join(" ");
      tr2.appendChild(tdProb);
    }
    tbody.appendChild(tr2);
  }

  // Download CSV
  let csv = shownCols.join(",") + "\n";
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const vals = [];
    for (const col of shownCols) {
      if (col === "prediction") vals.push(predictions[i]);
      else if (col === "probabilities" && hasProba) {
        vals.push("\"" + (result.probabilities[i] || []).join(";") + "\"");
      } else {
        vals.push(row[col] !== undefined ? row[col] : "");
      }
    }
    csv += vals.join(",") + "\n";
  }
  downloadResultsBtn.style.display = "inline-flex";
  const blob = new Blob([csv], { type: "text/csv" });
  downloadResultsBtn.href = URL.createObjectURL(blob);
  const csvBase = csvToken ? csvToken.replace(/\.csv$/i, "") : "predictions";
  downloadResultsBtn.download = csvBase + "_predictions.csv";
}

// ── Clear CSV & results (keeps model) ──

clearAllBtn.addEventListener("click", () => {
  csvToken = null;
  csvColumnNames = [];
  resultsSection.style.display = "none";
  featureStatus.innerHTML = "";
  csvStatus.textContent = "";
  predictBtn.disabled = true;
  csvFileInput.value = "";
});

// ── Init ──

document.addEventListener("DOMContentLoaded", async () => {
  try {
    await loadLanguage();
    await loadModels();
  } catch (err) {
    console.error(err);
  }
});