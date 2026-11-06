let currentAlgorithms = {};
let pollInterval = null;
let modelRefreshInterval = null;
let csvTotalRows = 0;

const dualSlider = document.getElementById("dualSlider");
const thumbTrain = document.getElementById("thumbTrain");
const thumbTest = document.getElementById("thumbTest");
const fillTrain = document.getElementById("fillTrain");
const fillVal = document.getElementById("fillVal");
const fillTest = document.getElementById("fillTest");
const csvCountTrain = document.getElementById("csvCountTrain");
const csvCountVal = document.getElementById("csvCountVal");
const csvCountTest = document.getElementById("csvCountTest");

// ── Upload ──

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
  csvTotalRows = Number.isFinite(data.rows) ? data.rows : data.total_lines || 0;
  updateSplitBar();

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
  csvTotalRows = 0;
  updateSplitBar();
});

// ── Algorithms ──

async function loadAlgorithms() {
  currentAlgorithms = await api("/api/algorithms") || {};
  algorithmSelect.innerHTML = "";
  for (const [key, cfg] of Object.entries(currentAlgorithms)) {
    if (cfg.available === false) continue;
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = translations["algo_label_" + key] || cfg.label || key;
    algorithmSelect.appendChild(opt);
  }
  renderHyperParams();
  updateTooltip();
}

function prettyName(value) {
  return String(value).replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function updateTooltip() {
  const algoKey = algorithmSelect.value;
  const cfg = currentAlgorithms[algoKey];
  if (cfg && cfg.description) {
    algoTooltipText.textContent = translations[`algo_desc_${algoKey}`] || cfg.description;
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
    const label = translations["param_" + key] || prettyName(key);
    const placeholder = translations["param_" + key + "_hint"] || param.hint || "";
    const val = param.default;
    if (param.type === "int") {
      html += `<label class="param-field"><span>${label}</span><input type="number" name="param_${key}" value="${val}"${param.min !== undefined ? ` min="${param.min}"` : ""}${param.max !== undefined ? ` max="${param.max}"` : ""} step="1"></label>`;
    } else if (param.type === "float") {
      html += `<label class="param-field"><span>${label}</span><input type="number" name="param_${key}" value="${val}"${param.min !== undefined ? ` min="${param.min}"` : ""}${param.max !== undefined ? ` max="${param.max}"` : ""} step="any"></label>`;
    } else if (param.type === "select") {
      html += `<label class="param-field"><span>${label}</span><select name="param_${key}">`;
      for (const opt of (param.options || [])) {
        const display = translations["param_" + key + "_" + opt]
          || translations["param_strategy_" + opt]
          || prettyName(opt);
        html += `<option value="${opt}"${opt === val ? " selected" : ""}>${display}</option>`;
      }
      html += `</select></label>`;
    } else if (param.type === "string") {
      html += `<label class="param-field"><span>${label}</span><input type="text" name="param_${key}" value="${val}" placeholder="${placeholder}"></label>`;
    }
  }
  hyperParams.innerHTML = html;
}

algorithmSelect.addEventListener("change", () => {
  renderHyperParams();
  updateTooltip();
});

// ── Dual slider ──

function getThumbPos(thumb) {
  const value = parseFloat(thumb && thumb.style.left);
  return Number.isFinite(value) ? value : 70;
}

function setThumbPos(thumb, pct) {
  thumb.style.left = Math.round(pct) + "%";
}

function updateSplitCounts(trainPct, valPct, testPct) {
  const total = csvTotalRows;
  const set = (element, value) => {
    if (element) element.textContent = total ? value : "";
  };
  if (!total) {
    [csvCountTrain, csvCountVal, csvCountTest].forEach((element) => {
      if (element) element.textContent = "";
    });
    return;
  }
  const test = Math.ceil(total * testPct / 100);
  const remainder = total - test;
  const val = Math.ceil(remainder * valPct / Math.max(1, 100 - testPct));
  const train = Math.max(0, remainder - val);
  set(csvCountTrain, train);
  set(csvCountVal, val);
  set(csvCountTest, test);
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
  updateSplitCounts(train, val, test);
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

// ── Validation strategy ──

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

// ── Validation helper (Task 1) ──

function validateNumberInput(input) {
  input.classList.remove("is-invalid");
  const min = parseFloat(input.min);
  const max = parseFloat(input.max);
  const val = parseFloat(input.value);
  if (input.value === "" || isNaN(val) || (!isNaN(min) && val < min) || (!isNaN(max) && val > max)) {
    input.classList.add("is-invalid");
    return false;
  }
  return true;
}

function validateParamsContainer(container) {
  let valid = true;
  for (const input of container.querySelectorAll("input[type=number]")) {
    if (!validateNumberInput(input)) valid = false;
  }
  return valid;
}

// ── Live validation on input ──
hyperParams.addEventListener("input", (e) => {
  if (e.target.type === "number") validateNumberInput(e.target);
});
for (const el of document.querySelectorAll('input[name="random_state"], input[name="k_folds"]')) {
  el.addEventListener("input", () => validateNumberInput(el));
}

// ── Training ──

trainForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const targetRb = document.querySelector('input[name="target"]:checked');
  if (!targetRb) { alert(t("select_target")); return; }
  const featureCbs = document.querySelectorAll('input[name="feature"]:checked');
  if (featureCbs.length === 0) { alert(t("select_features")); return; }
  const targetCol = targetRb.value;
  const featureCols = Array.from(featureCbs).map(cb => cb.value).filter(c => c !== targetCol);
  if (featureCols.length === 0) { alert(t("target_not_feature")); return; }

  // Validate hyperparameter ranges — highlight invalid fields in red
  let valid = validateParamsContainer(hyperParams);
  for (const el of [document.querySelector('input[name="random_state"]'), document.querySelector('input[name="k_folds"]')]) {
    if (el && !validateNumberInput(el)) valid = false;
  }
  if (!valid) {
    alert(t("validation_fix_highlighted") || "Fix the highlighted fields.");
    return;
  }

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
  const useShuffle = document.getElementById("csvUseShuffle").checked;
  const shuffle = useShuffle;
  const randomState = useShuffle ? parseInt(fd.random_state.value) || 42 : -1;

  trainBtn.disabled = true;
  trainBtn.textContent = t("training") + "...";
  const csvCancelBtn = document.getElementById("csvCancelBtn");
  csvCancelBtn.disabled = false;
  csvCancelBtn.textContent = t("cancel_train") || "Cancel";
  csvCancelBtn.style.display = "";
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
        document.getElementById("csvCancelBtn").style.display = "none";
        showResults(status.metrics);
        if (status.predictions) {
          const task = status.metrics ? status.metrics.task : "regression";
          renderPredictionChart(status.predictions, task);
        }
        enableResults();
      } else if (status.status === "cancelled") {
        clearInterval(pollInterval);
        pollInterval = null;
        trainBtn.disabled = false;
        trainBtn.textContent = t("train_btn");
        document.getElementById("csvCancelBtn").style.display = "none";
        appendLog(t("img_training_cancelled") || "Training cancelled.");
      } else if (status.status === "error") {
        clearInterval(pollInterval);
        pollInterval = null;
        trainBtn.disabled = false;
        trainBtn.textContent = t("train_btn");
        document.getElementById("csvCancelBtn").style.display = "none";
        appendLog(t("error_prefix") + ": " + (status.error || t("unknown_error")));
      }
    } catch (err) {
      // ignore polling errors
    }
  }, 500);
}

// ── Results ──

function showResults(metrics) {
  if (!metrics) return;
  resultsSection.style.display = "";
  metricsDisplay.innerHTML = "";

  const task = metrics.task || "classification";
  const metricKeys = task === "classification"
    ? ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    : ["mae", "mse", "rmse", "r2"];

  const labels = {
    accuracy: "metric_accuracy", precision: "metric_precision",
    recall: "metric_recall", f1_score: "metric_f1", roc_auc: "metric_roc_auc",
    mae: "metric_mae", mse: "metric_mse", rmse: "metric_rmse", r2: "metric_r2"
  };

  const visibleKeys = metricKeys.filter(key => {
    const val = metrics[key];
    return val !== undefined && val !== null && (typeof val === "string" || !isNaN(val));
  });

  setMetricColumns(metricsDisplay, visibleKeys.length);
  for (const key of visibleKeys) {
    const val = metrics[key];
    const div = document.createElement("div");
    div.className = "metric-card";
    div.innerHTML = `<strong>${t(labels[key]) || key}</strong><span class="metric-value">${val}</span>`;
    metricsDisplay.appendChild(div);
  }
}

// ── Save ──

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
    saveStatus.textContent = t("error_prefix") + ": " + err.message;
  }
});

clearOutputBtn.addEventListener("click", async () => {
  await api("/api/output/clear", { method: "POST", body: "{}" });
  outputLog.textContent = "";
});

document.getElementById("csvCancelBtn").addEventListener("click", async () => {
  const btn = document.getElementById("csvCancelBtn");
  btn.disabled = true;
  btn.textContent = t("cancelling") || "Cancelling...";
  appendLog(t("cancelling"));
  try {
    await api("/api/train/cancel", { method: "POST", body: "{}" });
  } catch (_) {}
});

// ── Model list ──

async function loadModels() {
  try {
    const data = await api("/api/models?type=csv");
    renderModels(data || []);
  } catch (err) {
    modelList.innerHTML = `<li class="model-empty">${t("error_prefix")}: ${err.message}</li>`;
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
        alert(t("error_prefix") + ": " + err.message);
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
        alert(t("error_prefix") + ": " + err.message);
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

modelFilter.addEventListener("input", () => {
  loadModels();
});

function startModelRefresh() {
  if (modelRefreshInterval) clearInterval(modelRefreshInterval);
  modelRefreshInterval = setInterval(loadModels, 30000);
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
