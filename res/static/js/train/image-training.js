let imgPollInterval = null;
let imgModelRefreshInterval = null;
let testGalleryState = { imageIds: [], predictionsMap: {}, currentIndex: 0 };

const imgDualSlider = document.getElementById("imgDualSlider");
const imgThumbTrain = document.getElementById("imgThumbTrain");
const imgThumbTest = document.getElementById("imgThumbTest");
const imgFillTrain = document.getElementById("imgFillTrain");
const imgFillVal = document.getElementById("imgFillVal");
const imgFillTest = document.getElementById("imgFillTest");
const imgCountTrain = document.getElementById("imgCountTrain");
const imgCountVal = document.getElementById("imgCountVal");
const imgCountTest = document.getElementById("imgCountTest");

const IMG_DETECTOR_DEFAULTS = {
  anchor_mode: "fixed",
  center_x: 0.52,
  center_y: 0.51,
  radius: 0.5,
  radius_min: 0.42,
  radius_max: 0.61,
};

const imgAnchorMode = document.getElementById("imgAnchorMode");

function imgNumber(id, fallback) {
  const value = parseFloat(document.getElementById(id)?.value);
  return Number.isFinite(value) ? value : fallback;
}

function imgDetectorConfig() {
  return {
    anchor_mode: imgAnchorMode ? imgAnchorMode.value : IMG_DETECTOR_DEFAULTS.anchor_mode,
    center_x: imgNumber("imgAnchorCenterX", IMG_DETECTOR_DEFAULTS.center_x),
    center_y: imgNumber("imgAnchorCenterY", IMG_DETECTOR_DEFAULTS.center_y),
    radius: imgNumber("imgAnchorRadius", IMG_DETECTOR_DEFAULTS.radius),
    radius_min: imgNumber("imgHoughRadiusMin", IMG_DETECTOR_DEFAULTS.radius_min),
    radius_max: imgNumber("imgHoughRadiusMax", IMG_DETECTOR_DEFAULTS.radius_max),
  };
}

function imgSplitPercentages() {
  const readLeft = (element, fallback) => {
    const value = parseFloat(element && element.style.left);
    return Number.isFinite(value) ? value : fallback;
  };
  const train = Math.min(100, Math.max(0, readLeft(imgThumbTrain, 70)));
  const testStart = Math.min(100, Math.max(train, readLeft(imgThumbTest, 85)));
  return { train, val: Math.max(0, testStart - train), test: Math.max(0, 100 - testStart) };
}

function imgSplitCounts(total, valPct, testPct) {
  if (total <= 0) return { train: 0, val: 0, test: 0 };

  // Largest-remainder apportionment over exact integer units, so floating
  // point noise cannot decide which split gets a leftover image.  When two
  // shares have the same remainder, Train wins first, then Validation, then Test.
  const basis = 1000000;
  const pcts = [
    Math.max(0, Math.round((100 - valPct - testPct) * basis / 100)),
    Math.max(0, Math.round(valPct * basis / 100)),
    Math.max(0, Math.round(testPct * basis / 100)),
  ];
  const products = pcts.map(pct => total * pct);
  const counts = products.map(product => Math.floor(product / basis));
  const remainders = products.map(product => product % basis);
  const order = [0, 1, 2].sort((a, b) => {
    const diff = remainders[b] - remainders[a];
    return diff !== 0 ? diff : a - b;
  });
  const left = total - counts.reduce((sum, value) => sum + value, 0);
  for (let i = 0; i < left; i++) counts[order[i]] += 1;

  function ensureOne(target, donors, minimumTotal) {
    if (total < minimumTotal || pcts[target] <= 0 || counts[target] > 0) return;
    for (const donor of donors) {
      if (donor !== target && counts[donor] > (donor === 0 ? 1 : 0)) {
        counts[donor] -= 1;
        counts[target] += 1;
        return;
      }
    }
  }

  // Training must keep at least one image whenever the dataset allows it.
  ensureOne(0, [2, 1], 1);
  // Test is the locked check set, so keep one image there when possible.
  ensureOne(2, [0, 1], 2);
  // Then keep a validation image when there is room for all three splits.
  ensureOne(1, [0, 2], 3);

  return { train: counts[0], val: counts[1], test: counts[2] };
}

// Test images are the LAST part of the list, just like the old workflow.
// This also makes drag-reordering a way to choose which images are test images.
function syncImgSplitState() {
  const total = imgState.images.length;
  const p = imgSplitPercentages();
  const c = imgSplitCounts(total, p.val, p.test);
  const testStart = total - c.test;
  const valStart = testStart - c.val;
  imgState.images.forEach((img, index) => {
    img.split = index >= testStart ? "test" : (index >= valStart ? "val" : "train");
    img.locked = img.split === "test";
  });
  return c;
}

function updateImgSplitBar() {
  if (!imgDualSlider) return;
  const p = imgSplitPercentages();
  imgFillTrain.style.width = p.train + "%";
  imgFillTrain.textContent = Math.round(p.train) + "%";
  imgFillVal.style.width = p.val + "%";
  imgFillVal.textContent = Math.round(p.val) + "%";
  imgFillTest.style.width = p.test + "%";
  imgFillTest.textContent = Math.round(p.test) + "%";
}

function updateImgSplitCounts() {
  const c = syncImgSplitState();
  const total = imgState.images.length;
  const set = (element, value) => {
    if (element) element.textContent = total ? value : "";
  };
  set(imgCountTrain, c.train);
  set(imgCountVal, c.val);
  set(imgCountTest, c.test);
}

function refreshImgSplitUI() {
  updateImgSplitBar();
  updateImgSplitCounts();
  const current = imgState.images[imgState.currentIndex];
  if (current && current.locked) {
    for (let i = imgState.images.length - 1; i >= 0; i--) {
      if (!imgState.images[i].locked) {
        selectImage(i);
        return;
      }
    }
  }
  updateImageList();
}

(function initImgSplitSlider() {
  if (!imgDualSlider || !imgThumbTrain || !imgThumbTest) return;
  let activeThumb = null;

  function posFromEvent(e) {
    const point = e.touches && e.touches[0] ? e.touches[0] : e;
    const rect = imgDualSlider.getBoundingClientRect();
    return ((point.clientX - rect.left) / rect.width) * 100;
  }

  function moveThumb(e) {
    if (!activeThumb) return;
    let pct = Math.max(5, Math.min(95, posFromEvent(e)));
    const startValue = parseFloat(imgThumbTrain.style.left);
    const endValue = parseFloat(imgThumbTest.style.left);
    const start = Number.isFinite(startValue) ? startValue : 70;
    const end = Number.isFinite(endValue) ? endValue : 85;
    if (activeThumb === imgThumbTrain) pct = Math.min(pct, end - 5);
    else pct = Math.max(pct, start + 5);
    activeThumb.style.left = Math.round(pct) + "%";
    refreshImgSplitUI();
  }

  function stopDrag() {
    activeThumb = null;
    document.removeEventListener("mousemove", moveThumb);
    document.removeEventListener("mouseup", stopDrag);
    document.removeEventListener("touchmove", moveThumb);
    document.removeEventListener("touchend", stopDrag);
  }

  function startDrag(thumb) {
    return (e) => {
      e.preventDefault();
      activeThumb = thumb;
      document.addEventListener("mousemove", moveThumb);
      document.addEventListener("mouseup", stopDrag);
      document.addEventListener("touchmove", moveThumb, { passive: false });
      document.addEventListener("touchend", stopDrag);
    };
  }

  imgThumbTrain.addEventListener("mousedown", startDrag(imgThumbTrain));
  imgThumbTest.addEventListener("mousedown", startDrag(imgThumbTest));
  imgThumbTrain.addEventListener("touchstart", startDrag(imgThumbTrain), { passive: false });
  imgThumbTest.addEventListener("touchstart", startDrag(imgThumbTest), { passive: false });
})();

updateImgSplitBar();
updateImgSplitCounts();

imgModelFilter.addEventListener("input", () => {
  loadImageModels();
});

// ── Saved image models ──

async function loadImageModels() {
  try {
    const data = await api("/api/models?type=image");
    renderImageModels(data || []);
  } catch (err) {
    imgModelList.innerHTML = `<li class="model-empty">${t("error_prefix")}: ${err.message}</li>`;
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
        alert(Object.entries(metrics).map(([k, v]) => `${k}: ${v}`).join("\n"));
      } catch (e) {
        alert(t("error_prefix") + ": " + e.message);
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
        alert(t("error_prefix") + ": " + err.message);
      }
    });

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "danger";
    delBtn.textContent = t("delete");
    delBtn.addEventListener("click", async () => {
      if (!requireAdmin()) return;
      const displayName = item.stem || item.name;
      if (!confirm(t("delete_confirm").replace("{name}", displayName))) return;
      try {
        await api("/api/models/delete", {
          method: "POST",
          body: JSON.stringify({ name: item.name, type: "image" })
        });
        await loadImageModels();
      } catch (err) {
        alert(t("error_prefix") + ": " + err.message);
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

// ── Train ──

imgTrainBtn.addEventListener("click", async () => {
  // Recalculate the split from the current image order.  The last images are
  // the locked TEST set and must stay unannotated.
  const splitCounts = syncImgSplitState();
  const annotated = imgState.images.filter(img => !img.locked && img.annotations.length > 0);
  const trainAnnotated = imgState.images.filter(img => img.split === "train" && img.annotations.length > 0);
  if (annotated.length === 0 || trainAnnotated.length === 0) {
    alert(t("img_no_annotations") || "Annotate the objects on at least one TRAIN image first. TEST images are locked on purpose.");
    return;
  }

  const counts = new Set(trainAnnotated.map(img => img.annotations.length));
  if (counts.size > 1) {
    alert(t("img_same_box_count_train") || "Every annotated TRAIN image must have the same number of boxes. Fix the annotations first.");
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
    imgAppendLog(t("error_prefix") + ": " + err.message);
    return;
  }

  imgTrainBtn.disabled = true;
  imgTrainBtn.textContent = (t("training") || "Training") + "...";
  const cancelBtn = document.getElementById("imgCancelBtn");
  cancelBtn.disabled = false;
  cancelBtn.textContent = t("cancel_train") || "Cancel";
  cancelBtn.style.display = "";
  imgOutputSection.style.display = "";
  imgOutputLog.textContent = "";
  imgResultsSection.style.display = "none";
  imgTestGallerySection.style.display = "none";

  const split = imgSplitPercentages();
  const testIds = imgState.images.filter(im => im.locked && im.serverId).map(im => im.serverId);
  const valIds = imgState.images.filter(im => im.split === "val" && im.serverId).map(im => im.serverId);

  try {
    await api("/api/image_train/start", {
      method: "POST",
      body: JSON.stringify({
        val_size: split.val / 100,
        test_size: split.test / 100,
        test_image_ids: testIds,
        val_image_ids: valIds,
        detector_config: imgDetectorConfig(),
      })
    });
    startImagePolling();
  } catch (err) {
    imgAppendLog(t("error_prefix") + ": " + err.message);
    imgTrainBtn.disabled = false;
    imgTrainBtn.textContent = t("img_train_btn") || "Train detector";
    cancelBtn.style.display = "none";
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
  imgAppendLog(t("cancelling"));
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
        for (let i = lastLen; i < log.length; i++) imgAppendLog(log[i]);
        lastLen = log.length;
      }

      if (status.status === "done") {
        clearInterval(imgPollInterval);
        imgPollInterval = null;
        resetTrainButton();
        showImageResults(status.metrics);
        showImageTestGallery(status.test_image_ids, status.predictions);
      } else if (status.status === "cancelled") {
        clearInterval(imgPollInterval);
        imgPollInterval = null;
        resetTrainButton();
        imgAppendLog(t("img_training_cancelled") || "Training cancelled.");
      } else if (status.status === "error") {
        clearInterval(imgPollInterval);
        imgPollInterval = null;
        resetTrainButton();
        imgAppendLog(t("error_prefix") + ": " + (status.error || t("unknown_error")));
        alert(status.error || t("img_training_failed") || "Training failed");
      }
    } catch (_) {
      // transient network error; keep polling
    }
  }, 500);
}

function resetTrainButton() {
  imgTrainBtn.disabled = false;
  imgTrainBtn.textContent = t("img_train_btn") || "Train detector";
  document.getElementById("imgCancelBtn").style.display = "none";
}

// ── Metrics ──

function showImageResults(metrics) {
  if (!metrics) return;
  imgResultsSection.style.display = "";
  imgMetricsDisplay.innerHTML = "";

  const labels = {
    train_images: "img_metric_train_images",
    train_annotated: "img_metric_train_annotated",
    val_images: "img_metric_val_images",
    val_annotated: "img_metric_val_annotated",
    test_images: "img_metric_test_images",
    images: "img_metric_images",
    annotations: "img_metric_annotations",
    boxes_per_image: "img_metric_boxes_per_image",
    mean_iou: "img_metric_mean_iou",
    val_mean_iou: "img_metric_val_mean_iou",
    loo_mean_iou: "img_metric_loo_mean_iou",
  };

  const visible = Object.entries(metrics).filter(([, value]) => value !== undefined && value !== null);
  setMetricColumns(imgMetricsDisplay, visible.length);
  for (const [key, value] of visible) {
    const div = document.createElement("div");
    div.className = "metric-card";
    const labelKey = labels[key];
    const label = labelKey ? t(labelKey) : key;
    div.innerHTML = `<strong>${label}</strong><span class="metric-value">${value}</span>`;
    imgMetricsDisplay.appendChild(div);
  }

  enableImgResults();
}

// ── Locked TEST-set prediction gallery ──

function showImageTestGallery(imageIds, predictions) {
  if (!imageIds || imageIds.length === 0) return;
  imgTestGallerySection.style.display = "";
  testGalleryState.imageIds = imageIds;
  testGalleryState.currentIndex = 0;
  testGalleryState.predictionsMap = {};
  for (const pred of (predictions || [])) {
    if (!testGalleryState.predictionsMap[pred.image_id]) {
      testGalleryState.predictionsMap[pred.image_id] = [];
    }
    testGalleryState.predictionsMap[pred.image_id].push(pred);
  }
  renderTestGalleryImage(0);
  updateTestGalleryNav();
}

function renderTestGalleryImage(index) {
  const imageId = testGalleryState.imageIds[index];
  if (!imageId) return;
  const imgData = imgState.images.find(im => im.serverId === imageId);
  if (!imgData) return;

  testCanvasPlaceholder.style.display = "none";
  testGalleryCanvas.style.display = "";

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

  const predictions = testGalleryState.predictionsMap[imageId] || [];
  const img = new Image();
  img.onload = () => {
    testGalleryCtx.clearRect(0, 0, cw, ch);
    testGalleryCtx.drawImage(img, 0, 0, cw, ch);
    predictions.forEach((pred, i) => {
      const x = pred.x * scale;
      const y = pred.y * scale;
      const w = pred.w * scale;
      const h = pred.h * scale;
      testGalleryCtx.strokeStyle = "#22c55e";
      testGalleryCtx.lineWidth = 2;
      testGalleryCtx.strokeRect(x, y, w, h);
      testGalleryCtx.fillStyle = "rgba(34,197,94,0.12)";
      testGalleryCtx.fillRect(x, y, w, h);
      testGalleryCtx.fillStyle = "#22c55e";
      testGalleryCtx.font = "bold 12px sans-serif";
      testGalleryCtx.textBaseline = "top";
      testGalleryCtx.fillText(`${i + 1}`, x + 4, y + 3);
    });
  };
  img.src = imgData.dataUrl;
}

function updateTestGalleryNav() {
  const total = testGalleryState.imageIds.length;
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
  if (testGalleryState.currentIndex < testGalleryState.imageIds.length - 1) {
    testGalleryState.currentIndex++;
    renderTestGalleryImage(testGalleryState.currentIndex);
    updateTestGalleryNav();
  }
});

// ── Save ──

imgSaveBtn.addEventListener("click", async () => {
  if (!requireAdmin()) return;
  const name = prompt(t("save_prompt"));
  if (!name) return;
  try {
    await api("/api/image_train/save", {
      method: "POST",
      body: JSON.stringify({ name: name.trim() })
    });
    imgSaveStatus.textContent = t("saved_ok") + ": " + name.trim() + ".model";
    await loadImageModels();
  } catch (err) {
    imgSaveStatus.textContent = t("error_prefix") + ": " + err.message;
  }
});
