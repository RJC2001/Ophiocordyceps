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

// ── CSV: Tabs (server/upload) ──

document.querySelectorAll("#modelTabBar button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#modelTabBar button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll("#modelSection .tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab" + btn.dataset.tab.charAt(0).toUpperCase() + btn.dataset.tab.slice(1)).classList.add("active");
  });
});

// ── CSV: Load Models ──

async function loadModels() {
  try {
    const models = await api("/api/models?type=csv") || [];
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

// ── CSV: Select model from server ──

selectModelBtn.addEventListener("click", async () => {
  const name = modelSelect.value;
  if (!name) return;
  try {
    const info = await api("/api/test/model_info", {
      method: "POST",
      body: JSON.stringify({ model_name: name, type: "csv" })
    });
    if (info.type === "image_detector") {
      // Switch to Image tab and select there
      document.querySelector('.mode-tab[data-mode="image"]').click();
      await loadImageModels();
      const opt = imgModelSelect.querySelector(`option[value="${CSS.escape(name)}"]`);
      if (opt) {
        imgModelSelect.value = name;
        selectImageModel(name);
      }
      return;
    }
    currentModelState = { source: "server", name, feature_cols: info.feature_cols || [] };
    loadedModelPill.textContent = (name.endsWith(".model") ? name.slice(0, -6) : name);
    csvSection.style.display = "";
    resultsSection.style.display = "none";
    revalidateFeatures();
  } catch (err) {
    alert("Error: " + err.message);
  }
});

// ── CSV: Upload model ──

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

// ── CSV: Upload ──

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

// ── CSV: Re-validate features ──

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

// ── CSV: Predict ──

let csvPollInterval = null;

predictBtn.addEventListener("click", async () => {
  if (!currentModelState || !csvToken) return;

  predictBtn.disabled = true;
  predictBtn.textContent = t("test_predicting") + "...";
  const csvCancelBtn = document.getElementById("csvTestCancelBtn");
  if (csvCancelBtn) {
    csvCancelBtn.style.display = "";
    csvCancelBtn.disabled = false;
    csvCancelBtn.textContent = t("cancel_train") || "Cancel";
  }

  try {
    const body = {};
    if (currentModelState.source === "server") {
      body.model_name = currentModelState.name;
    } else {
      body.upload_token = currentModelState.token;
    }
    const resp = await api("/api/test/predict_csv", {
      method: "POST",
      body: JSON.stringify(body)
    });

    // Poll for completion
    await new Promise((resolve, reject) => {
      csvPollInterval = setInterval(async () => {
        try {
          const st = await api("/api/test/status");
          if (st.status === "done") {
            clearInterval(csvPollInterval);
            csvPollInterval = null;
            showResults(st.results);
            resolve();
          } else if (st.status === "cancelled") {
            clearInterval(csvPollInterval);
            csvPollInterval = null;
            alert("Prediction cancelled.");
            resolve();
          } else if (st.status === "error") {
            clearInterval(csvPollInterval);
            csvPollInterval = null;
            throw new Error(st.error || "Prediction failed");
          }
        } catch (e) {
          clearInterval(csvPollInterval);
          csvPollInterval = null;
          reject(e);
        }
      }, 500);
    });
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = t("test_predict_btn");
    const csvCancelBtn = document.getElementById("csvTestCancelBtn");
    if (csvCancelBtn) csvCancelBtn.style.display = "none";
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

// ── CSV: Clear ──

clearAllBtn.addEventListener("click", () => {
  csvToken = null;
  csvColumnNames = [];
  resultsSection.style.display = "none";
  featureStatus.innerHTML = "";
  csvStatus.textContent = "";
  predictBtn.disabled = true;
  csvFileInput.value = "";
});

// ═══════════════════════════════════════════════════════════════════
// IMAGE MODE
// ═══════════════════════════════════════════════════════════════════

const imgModelFilter = document.getElementById("imgModelFilter");
const imgModelSelect = document.getElementById("imgModelSelect");
const imgSelectModelBtn = document.getElementById("imgSelectModelBtn");
const imgRefreshModelsBtn = document.getElementById("imgRefreshModelsBtn");
const imgModelUploadZone = document.getElementById("imgModelUploadZone");
const imgModelFileInput = document.getElementById("imgModelFileInput");
const imgUploadModelStatus = document.getElementById("imgUploadModelStatus");
const imgConfirmUploadBtn = document.getElementById("imgConfirmUploadBtn");
const imgTestUploadSection = document.getElementById("imgTestUploadSection");
const imgTestModelPill = document.getElementById("imgTestModelPill");
const imgTestUploadZone = document.getElementById("imgTestUploadZone");
const imgTestFileInput = document.getElementById("imgTestFileInput");
const imgTestStatus = document.getElementById("imgTestStatus");
const detectBtn = document.getElementById("detectBtn");
const imgTestClearBtn = document.getElementById("imgTestClearBtn");
const imgTestResultsSection = document.getElementById("imgTestResultsSection");
const imgTestPreview = document.getElementById("imgTestPreview");
const downloadDetectionsBtn = document.getElementById("downloadDetectionsBtn");
const confidenceSlider = document.getElementById("confidenceSlider");
const confidenceValue = document.getElementById("confidenceValue");
const mergeSlider = document.getElementById("mergeSlider");
const mergeValue = document.getElementById("mergeValue");

if (confidenceSlider && confidenceValue) {
  confidenceSlider.addEventListener("input", () => {
    confidenceValue.textContent = parseFloat(confidenceSlider.value).toFixed(2);
  });
}

if (mergeSlider && mergeValue) {
  mergeSlider.addEventListener("input", () => {
    mergeValue.textContent = parseFloat(mergeSlider.value).toFixed(2);
  });
}

let imgModelState = null;
let imgUploadToken = null;
let imgTestImages = []; // { filename, dataUrl, data: base64content }
let lastZipBlob = null;

// ── Image: Tabs (server/upload) ──

document.querySelectorAll("#imgModelTabBar button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#imgModelTabBar button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll("#imgModelSection .tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("imgTab" + btn.dataset.tab.charAt(0).toUpperCase() + btn.dataset.tab.slice(1)).classList.add("active");
  });
});

// ── Image: Load Models (filter to image detectors) ──

async function loadImageModels() {
  try {
    const models = await api("/api/models?type=image") || [];
    const filter = imgModelFilter.value.trim().toLowerCase();
    const filtered = filter ? models.filter(m => (m.stem || m.name).toLowerCase().includes(filter)) : models;

    imgModelSelect.innerHTML = `<option value="">${t("test_select_hint")}</option>`;
    for (const m of filtered) {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = (m.stem || m.name) + " \u{1F5BC}";
      imgModelSelect.appendChild(opt);
    }
    if (filtered.length === 0) {
      imgModelSelect.innerHTML = `<option value="">${t("img_no_models") || "-- No image models --"}</option>`;
    }
  } catch (err) {
    imgModelSelect.innerHTML = `<option value="">Error: ${err.message}</option>`;
  }
}

imgModelFilter.addEventListener("input", loadImageModels);
imgRefreshModelsBtn.addEventListener("click", loadImageModels);

// ── Image: Select model from server ──

async function selectImageModel(name) {
  try {
    imgModelState = { source: "server", name };
    imgTestModelPill.textContent = name.endsWith(".model") ? name.slice(0, -6) : name;
    imgTestUploadSection.style.display = "";
    imgTestResultsSection.style.display = "none";
  } catch (err) {
    alert("Error: " + err.message);
  }
}

imgSelectModelBtn.addEventListener("click", async () => {
  const name = imgModelSelect.value;
  if (!name) return;
  await selectImageModel(name);
});

// ── Image: Upload model ──

imgModelUploadZone.addEventListener("click", () => imgModelFileInput.click());
imgModelUploadZone.addEventListener("dragover", (e) => { e.preventDefault(); imgModelUploadZone.classList.add("drag-over"); });
imgModelUploadZone.addEventListener("dragleave", () => imgModelUploadZone.classList.remove("drag-over"));
imgModelUploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  imgModelUploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleImageModelFile(file);
});
imgModelFileInput.addEventListener("change", () => {
  if (imgModelFileInput.files[0]) handleImageModelFile(imgModelFileInput.files[0]);
});

async function handleImageModelFile(file) {
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
    imgUploadToken = result.token;
    imgUploadModelStatus.textContent = t("test_uploaded") + ": " + file.name;
  } catch (err) {
    alert(err.message);
  }
}

imgConfirmUploadBtn.addEventListener("click", () => {
  if (!imgUploadToken) {
    alert(t("test_upload_first"));
    return;
  }
  imgModelState = { source: "upload", token: imgUploadToken };
  imgTestModelPill.textContent = t("test_uploaded_model");
  imgTestUploadSection.style.display = "";
  imgTestResultsSection.style.display = "none";
});

// ── Image: Upload test images ──

imgTestUploadZone.addEventListener("click", () => imgTestFileInput.click());
imgTestUploadZone.addEventListener("dragover", (e) => { e.preventDefault(); imgTestUploadZone.classList.add("drag-over"); });
imgTestUploadZone.addEventListener("dragleave", () => imgTestUploadZone.classList.remove("drag-over"));
imgTestUploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  imgTestUploadZone.classList.remove("drag-over");
  const files = e.dataTransfer.files;
  for (const file of files) handleTestImageFile(file);
});
imgTestFileInput.addEventListener("change", () => {
  for (const file of imgTestFileInput.files) handleTestImageFile(file);
  imgTestFileInput.value = "";
});

function handleTestImageFile(file) {
  const validTypes = ["image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"];
  if (!validTypes.includes(file.type) && !file.name.match(/\.(png|jpg|jpeg|gif|webp|bmp)$/i)) {
    alert(t("img_format_error") || "Unsupported image format");
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    const base64Content = dataUrl.split(",")[1];
    imgTestImages.push({
      filename: file.name,
      dataUrl,
      data: base64Content,
    });
    updateTestImageStatus();
  };
  reader.readAsDataURL(file);
}

function updateTestImageStatus() {
  const n = imgTestImages.length;
  if (n === 0) {
    imgTestStatus.textContent = "";
    detectBtn.disabled = true;
    return;
  }
  imgTestStatus.textContent = n + " " + (n === 1 ? "image loaded" : "images loaded");
  detectBtn.disabled = !imgModelState;
}

// ── Image: Detect ──

let imgPollInterval = null;

detectBtn.addEventListener("click", async () => {
  if (!imgModelState || imgTestImages.length === 0) return;

  detectBtn.disabled = true;
  detectBtn.textContent = (t("test_predicting") || "Detecting") + "...";
  const imgCancelBtn = document.getElementById("imgTestCancelBtn");
  if (imgCancelBtn) {
    imgCancelBtn.style.display = "";
    imgCancelBtn.disabled = false;
    imgCancelBtn.textContent = t("img_cancel_test") || "Cancel";
  }

  try {
    const confidence = confidenceSlider ? parseFloat(confidenceSlider.value) : 0.5;
    const mergeThreshold = mergeSlider ? parseFloat(mergeSlider.value) : 0.2;
    const body = {
      images: imgTestImages.map(img => ({ filename: img.filename, data: img.data })),
      confidence: confidence,
      merge_threshold: mergeThreshold,
    };
    if (imgModelState.source === "server") {
      body.model_name = imgModelState.name;
    } else {
      body.upload_token = imgModelState.token;
    }

    const resp = await fetch("/api/image_test/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const errText = await resp.text();
      let errMsg = errText;
      try { const j = JSON.parse(errText); errMsg = j.error || errText; } catch {}
      throw new Error(errMsg);
    }

    const data = await resp.json();
    const zipToken = data.zip_token;

    // Poll for completion
    await new Promise((resolve, reject) => {
      imgPollInterval = setInterval(async () => {
        try {
          const st = await api("/api/image_test/status");
          if (st.status === "done") {
            clearInterval(imgPollInterval);
            imgPollInterval = null;
            // Fetch ZIP
            const zipResp = await fetch("/api/image_test/download/" + zipToken);
            if (!zipResp.ok) throw new Error("Failed to download ZIP");
            lastZipBlob = await zipResp.blob();
            showImageDetectionResults(st.detections || []);
            resolve();
          } else if (st.status === "cancelled") {
            clearInterval(imgPollInterval);
            imgPollInterval = null;
            alert("Detection cancelled.");
            resolve();
          } else if (st.status === "error") {
            clearInterval(imgPollInterval);
            imgPollInterval = null;
            throw new Error(st.error || "Detection failed");
          }
        } catch (e) {
          clearInterval(imgPollInterval);
          imgPollInterval = null;
          reject(e);
        }
      }, 500);
    });
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    detectBtn.disabled = false;
    detectBtn.textContent = t("img_detect_btn") || "Detect Objects";
    const imgCancelBtn = document.getElementById("imgTestCancelBtn");
    if (imgCancelBtn) imgCancelBtn.style.display = "none";
  }
});

function drawDetectionsOnCanvas(img, detections, canvas) {
  const ctx = canvas.getContext("2d");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);
  ctx.strokeStyle = "#00ff00";
  ctx.lineWidth = 3;
  ctx.font = "14px sans-serif";
  ctx.fillStyle = "#00ff00";
  for (const d of detections) {
    ctx.strokeRect(d.x, d.y, d.w, d.h);
    ctx.fillText(d.confidence.toFixed(2), d.x, Math.max(0, d.y - 4));
  }
}

function showImageDetectionResults(allDetections) {
  imgTestResultsSection.style.display = "";
  imgTestPreview.innerHTML = "";

  const statusMsg = document.createElement("p");
  statusMsg.className = "table-info";
  const totalDets = allDetections.reduce((s, im) => s + im.detections.length, 0);
  statusMsg.textContent = (t("img_test_detection_done") || "Detection complete.") + " (" + totalDets + " objects in " + imgTestImages.length + " " + (t("img_images") || "images") + ")";
  imgTestPreview.appendChild(statusMsg);

  // Build detections map by filename
  const detMap = {};
  for (const im of (allDetections || [])) {
    detMap[im.filename] = im;
  }

  // Show preview thumbnails with boxes drawn on canvas
  const previewGrid = document.createElement("div");
  previewGrid.className = "img-test-thumbs";
  for (const img of imgTestImages) {
    const thumb = document.createElement("div");
    thumb.className = "img-thumb";
    const detInfo = detMap[img.filename];
    if (detInfo && detInfo.detections && detInfo.detections.length > 0) {
      const canvas = document.createElement("canvas");
      canvas.className = "img-thumb-canvas";
      canvas.style.width = "100%";
      canvas.style.maxWidth = "200px";
      const image = new Image();
      image.onload = () => {
        drawDetectionsOnCanvas(image, detInfo.detections, canvas);
      };
      image.src = img.dataUrl;
      thumb.appendChild(canvas);
    } else {
      thumb.innerHTML = `<img src="${img.dataUrl}" alt="${img.filename}">`;
    }
    const name = document.createElement("span");
    name.className = "img-thumb-name";
    name.textContent = img.filename + (detInfo ? " (" + detInfo.detections.length + ")" : "");
    thumb.appendChild(name);
    previewGrid.appendChild(thumb);
  }
  imgTestPreview.appendChild(previewGrid);

  // Enable download
  downloadDetectionsBtn.style.display = "inline-flex";
  downloadDetectionsBtn.href = URL.createObjectURL(lastZipBlob);
  downloadDetectionsBtn.download = "detections.zip";
}

// ── Image: Clear ──

imgTestClearBtn.addEventListener("click", () => {
  imgTestImages = [];
  lastZipBlob = null;
  imgTestStatus.textContent = "";
  detectBtn.disabled = true;
  imgTestResultsSection.style.display = "none";
  downloadDetectionsBtn.style.display = "none";
  imgTestFileInput.value = "";
});

// ── Cancel handlers ──

document.getElementById("imgTestCancelBtn").addEventListener("click", async () => {
  const btn = document.getElementById("imgTestCancelBtn");
  btn.disabled = true;
  btn.textContent = t("cancelling") || "Cancelling...";
  try {
    await api("/api/image_test/cancel", { method: "POST", body: "{}" });
  } catch (_) {}
});

document.getElementById("csvTestCancelBtn").addEventListener("click", async () => {
  const btn = document.getElementById("csvTestCancelBtn");
  btn.disabled = true;
  btn.textContent = t("cancelling") || "Cancelling...";
  try {
    await api("/api/test/cancel", { method: "POST", body: "{}" });
  } catch (_) {}
});

// ── Init ──

document.addEventListener("DOMContentLoaded", async () => {
  try {
    await loadLanguage();
    await loadModels();
    await loadImageModels();
  } catch (err) {
    console.error(err);
  }
});
