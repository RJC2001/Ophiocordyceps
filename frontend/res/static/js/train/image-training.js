let imgPollInterval = null;
let imgModelRefreshInterval = null;
let testGalleryState = { testImageIds: [], predictionsMap: {}, currentIndex: 0 };

imgModelFilter.addEventListener("input", () => {
  loadImageModels();
});

// ── Image Models ──

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

function startImageModelRefresh() {
  if (imgModelRefreshInterval) clearInterval(imgModelRefreshInterval);
  imgModelRefreshInterval = setInterval(loadImageModels, 30000);
}

// ── Image validation (Task 1) ──

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

function validateImgParams() {
  let valid = true;
  for (const input of imgHyperParams.querySelectorAll("input[type=number]")) {
    if (!validateNumberInput(input)) valid = false;
  }
  for (const id of ["imgPatchSize", "imgNegPerAnn", "imgRandomSeed", "imgStride", "imgKFolds"]) {
    const el = document.getElementById(id);
    if (el && el.type === "number" && !el.disabled && !validateNumberInput(el)) valid = false;
  }
  return valid;
}

// ── Live validation on input ──
imgHyperParams.addEventListener("input", (e) => {
  if (e.target.type === "number") validateNumberInput(e.target);
});
for (const id of ["imgPatchSize", "imgNegPerAnn", "imgRandomSeed", "imgStride", "imgKFolds"]) {
  const el = document.getElementById(id);
  if (el && el.type === "number") el.addEventListener("input", () => validateNumberInput(el));
}

// Checkbox enables/disables its linked input
function bindEnableToggle(cbId, selector) {
  const cb = document.getElementById(cbId);
  const input = document.getElementById(selector) || document.querySelector(`[name="${selector}"]`);
  if (cb && input) {
    const toggle = () => { input.disabled = !cb.checked; };
    cb.addEventListener("change", toggle);
    toggle();
  }
}
document.addEventListener("DOMContentLoaded", () => {
  bindEnableToggle("imgUsePatchSize", "imgPatchSize");
  bindEnableToggle("imgUseNeg", "imgNegPerAnn");
  bindEnableToggle("imgUseStride", "imgStride");
  bindEnableToggle("imgUseShuffle", "imgRandomSeed");
  bindEnableToggle("csvUseShuffle", "random_state");
});

// ── Image Training ──

imgTrainBtn.addEventListener("click", async () => {
  const hasAnnotations = imgState.images.some(img => img.annotations.length > 0);
  if (!hasAnnotations) {
    alert(t("img_no_annotations") || "Please annotate at least one object on at least one image.");
    return;
  }

  // Validate parameter ranges — highlight invalid fields in red
  if (!validateImgParams()) {
    alert(t("validation_fix_highlighted") || "Fix the highlighted fields.");
    return;
  }

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
  const imgCancelBtn = document.getElementById("imgCancelBtn");
  imgCancelBtn.disabled = false;
  imgCancelBtn.textContent = t("cancel_train") || "Cancel";
  imgCancelBtn.style.display = "";
  tuningBound = false;
  imgOutputSection.style.display = "";
  imgResultsSection.style.display = "none";
  imgPredSection.style.display = "none";
  imgTestGallerySection.style.display = "none";
  imgTuningSection.style.display = "none";

  const valStrategy = document.querySelector('input[name="img_val_strategy"]:checked').value;
  const algorithm = imgAlgorithmSelect.value;
  const useNeg = document.getElementById("imgUseNeg").checked;
  const negRaw = document.getElementById("imgNegPerAnn").value;
  const negPerAnn = useNeg ? (parseInt(negRaw) || 10) : 0;
  const params = {};
  for (const input of imgHyperParams.querySelectorAll("input, select")) {
    const name = input.getAttribute("name");
    if (name && name.startsWith("imgparam_")) {
      params[name.replace("imgparam_", "")] = input.value;
    }
  }
  const useShuffle = document.getElementById("imgUseShuffle").checked;
  const shuffle = useShuffle;
  const randomState = useShuffle ? (parseInt(document.getElementById("imgRandomSeed").value) || 42) : -1;

  let valSize = 0, testSize = 0, kFolds = 5;
  if (valStrategy === "simple") {
    const t2 = parseFloat(imgThumbTrain.style.left || 70);
    const e = parseFloat(imgThumbTest.style.left || 85);
    testSize = (100 - e) / 100;
    valSize = (e - t2) / 100;
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
        patch_size: document.getElementById("imgUsePatchSize").checked
          ? parseInt(document.getElementById("imgPatchSize").value) || 32
          : 0,
        stride: document.getElementById("imgUseStride").checked
          ? parseInt(document.getElementById("imgStride").value) || 0
          : 0,
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

document.getElementById("imgCancelBtn").addEventListener("click", async () => {
  const btn = document.getElementById("imgCancelBtn");
  btn.disabled = true;
  btn.textContent = t("cancelling") || "Cancelling...";
  imgAppendLog("Cancelling...");
  try {
    await api("/api/image_train/cancel", { method: "POST", body: "{}" });
  } catch (_) {}
});

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
        document.getElementById("imgCancelBtn").style.display = "none";
        showImageResults(status.metrics);
        showImagePredictions(status.predictions);
        const raw = status.raw_predictions || status.predictions || [];
        showImageTuning(status.test_image_ids, raw);
      } else if (status.status === "cancelled") {
        clearInterval(imgPollInterval);
        imgPollInterval = null;
        imgTrainBtn.disabled = false;
        imgTrainBtn.textContent = t("img_train_btn") || "Train Detector";
        document.getElementById("imgCancelBtn").style.display = "none";
        imgAppendLog("Training cancelled.");
      } else if (status.status === "error") {
        clearInterval(imgPollInterval);
        imgPollInterval = null;
        imgTrainBtn.disabled = false;
        imgTrainBtn.textContent = t("img_train_btn") || "Train Detector";
        document.getElementById("imgCancelBtn").style.display = "none";
        imgAppendLog("ERROR: " + (status.error || "Unknown error"));
      }
    } catch (err) {
      // ignore
    }
  }, 500);
}

// ── Image Results ──

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

  enableImgResults();
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

// ── Image Test Gallery ──

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
        const label = pred.confidence !== undefined
          ? (pred.confidence * 100).toFixed(0) + "%"
          : "D";
        testGalleryCtx.fillStyle = "#22c55e";
        testGalleryCtx.font = "bold 11px sans-serif";
        testGalleryCtx.textBaseline = "top";
        testGalleryCtx.fillText(label, x + 2, y + 2);
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
    testGalleryCtx.fillText(t("test_canvas_hint") || "Image not available", cw / 2, ch / 2);
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

// ── Client-side IoU / merge helpers ──

function computeIoU(a, b) {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.w, b.x + b.w);
  const bottom = Math.min(a.y + a.h, b.y + b.h);
  if (left < right && top < bottom) {
    const inter = (right - left) * (bottom - top);
    const union = a.w * a.h + b.w * b.h - inter;
    return inter / (union || 1);
  }
  return 0;
}

function filterAndMerge(detections, confThresh, iouThresh) {
  const kept = detections.filter(d => d.confidence >= confThresh);
  const byImg = {};
  for (const d of kept) {
    if (!byImg[d.image_id]) byImg[d.image_id] = [];
    byImg[d.image_id].push(d);
  }
  const result = [];
  for (const boxes of Object.values(byImg)) {
    const used = new Array(boxes.length).fill(false);
    for (let i = 0; i < boxes.length; i++) {
      if (used[i]) continue;
      used[i] = true;
      let gx1 = boxes[i].x, gy1 = boxes[i].y;
      let gx2 = boxes[i].x + boxes[i].w, gy2 = boxes[i].y + boxes[i].h;
      for (let j = 0; j < boxes.length; j++) {
        if (used[j]) continue;
        const bx1 = boxes[j].x, by1 = boxes[j].y;
        const bx2 = boxes[j].x + boxes[j].w, by2 = boxes[j].y + boxes[j].h;
        if (computeIoU({x: gx1, y: gy1, w: gx2 - gx1, h: gy2 - gy1}, {x: bx1, y: by1, w: bx2 - bx1, h: by2 - by1}) > iouThresh) {
          used[j] = true;
          gx1 = Math.min(gx1, bx1);
          gy1 = Math.min(gy1, by1);
          gx2 = Math.max(gx2, bx2);
          gy2 = Math.max(gy2, by2);
        }
      }
      result.push({
        x: gx1, y: gy1,
        w: gx2 - gx1, h: gy2 - gy1,
        image_id: boxes[i].image_id,
        sliding_window: true,
        confidence: boxes[i].confidence,
      });
    }
  }
  return result;
}

// ── Detection Tuning ──

let tuningRawPredictions = [];
let tuningBound = false;

function showImageTuning(testImageIds, rawPredictions) {
  if (!testImageIds || testImageIds.length === 0) return;
  tuningRawPredictions = rawPredictions || [];
  testGalleryState.testImageIds = testImageIds;
  testGalleryState.currentIndex = 0;
  imgTuningSection.style.display = "";
  imgTestGallerySection.style.display = "";

  const confSlider = document.getElementById("tuningConfSlider");
  const nmsSlider = document.getElementById("tuningNmsSlider");
  const confVal = document.getElementById("tuningConfValue");
  const nmsVal = document.getElementById("tuningNmsValue");

  const applyTuning = () => {
    const ct = parseFloat(confSlider.value);
    const nt = parseFloat(nmsSlider.value);
    confVal.textContent = ct.toFixed(2);
    nmsVal.textContent = nt.toFixed(2);
    rebuildTestGallery(ct, nt);
  };

  if (!tuningBound) {
    tuningBound = true;
    confSlider.addEventListener("input", applyTuning);
    nmsSlider.addEventListener("input", applyTuning);
  }

  applyTuning();
}

function rebuildTestGallery(confThresh, nmsThresh) {
  const filtered = filterAndMerge(tuningRawPredictions, confThresh, nmsThresh);
  const predMap = {};
  for (const p of filtered) {
    if (!predMap[p.image_id]) predMap[p.image_id] = [];
    predMap[p.image_id].push(p);
  }
  testGalleryState.predictionsMap = predMap;
  renderTestGalleryImage(testGalleryState.currentIndex);
  updateTestGalleryNav();
}

// ── Image Save ──

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
