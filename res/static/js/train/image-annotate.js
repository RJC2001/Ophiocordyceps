let imgCurrentAlgorithms = {};

const ctx = canvas.getContext("2d");

let offscreenCanvas = null;
let offscreenCtx = null;

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
  shapeMode: 'rect',
  polygonPoints: [],
  polygonCursorX: 0,
  polygonCursorY: 0,
};

// ── Image Algorithms ──

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
      html += `<label class="param-field"><span>${label}</span><input type="number" name="imgparam_${key}" value="${val}"${param.min !== undefined ? ` min="${param.min}"` : ""}${param.max !== undefined ? ` max="${param.max}"` : ""} step="1"></label>`;
    } else if (param.type === "float") {
      html += `<label class="param-field"><span>${label}</span><input type="number" name="imgparam_${key}" value="${val}"${param.min !== undefined ? ` min="${param.min}"` : ""}${param.max !== undefined ? ` max="${param.max}"` : ""} step="any"></label>`;
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

imgAlgorithmSelect.addEventListener("change", () => {
  renderImgHyperParams();
  updateImgTooltip();
});

// ── Image dual slider ──

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
    imgState.images.forEach(img => { img.locked = false; img.split = "train"; });
    updateNavButtons();
    return;
  }
  const trainPctLeft = parseFloat(imgThumbTrain.style.left) || 70;
  const testPctLeft = parseFloat(imgThumbTest.style.left) || 85;
  const test = 100 - testPctLeft;
  const val = testPctLeft - trainPctLeft;
  const total = imgState.images.length;
  // Same logic as backend: round for test/val, train absorbs leftovers
  let testCount = Math.round(total * test / 100);
  let valCount = Math.round(total * val / 100);
  if (testCount + valCount > total) {
    if (val <= test) valCount = total - testCount; else testCount = total - valCount;
  }
  const trainCount = total - testCount - valCount;
  imgState.images.forEach((img, i) => {
    img.locked = (i >= total - testCount);
    if (img.locked) {
      img.split = "test";
    } else if (i >= total - testCount - valCount) {
      img.split = "val";
    } else {
      img.split = "train";
    }
  });
  updateNavButtons();
}

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
    updateImgLockedStates();
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

// ── Image Upload ──

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
        split: "train",
      });
      updateImgLockedStates();
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
    const label = document.createElement("span");
    label.className = "img-thumb-label " + (imgData.split || "train");
    label.textContent = (imgData.split || "train").toUpperCase();
    thumb.appendChild(label);
    if (imgData.annotations.length > 0) {
      const badge = document.createElement("span");
      badge.className = "img-thumb-badge";
      badge.textContent = imgData.annotations.length;
      thumb.appendChild(badge);
    }
    const delBtn = document.createElement("button");
    delBtn.className = "img-thumb-del";
    delBtn.title = t("delete") || "Delete";
    delBtn.textContent = "\u2715";
    delBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(t("img_delete_confirm") || "Delete this image?")) return;
      if (imgData.uploaded && imgData.serverId) {
        try {
          await api("/api/image_train/delete", {
            method: "POST",
            body: JSON.stringify({ image_id: imgData.serverId })
          });
        } catch (_) {}
      }
      imgState.images.splice(i, 1);
      if (imgState.currentIndex >= imgState.images.length && imgState.images.length > 0) {
        imgState.currentIndex = imgState.images.length - 1;
      } else if (imgState.images.length === 0) {
        imgState.currentIndex = 0;
        canvas.style.display = "none";
        canvasPlaceholder.style.display = "";
        canvas.width = 0;
        canvas.height = 0;
      }
      imgState.selectedAnnotation = -1;
      updateImageList();
      updateNavButtons();
      if (imgState.images.length > 0) {
        selectImage(Math.min(imgState.currentIndex, imgState.images.length - 1));
      }
    });
    thumb.appendChild(delBtn);
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

// ── Canvas ──

// ── Shape mode switching ──

document.querySelectorAll(".shape-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    if (imgState.polygonPoints.length > 0) {
      imgState.polygonPoints = [];
      renderCurrentFrame();
    }
    document.querySelectorAll(".shape-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    imgState.shapeMode = btn.dataset.shape;
    const hint = document.getElementById("polygonHint");
    if (hint) hint.style.display = imgState.shapeMode === "polygon" ? "" : "none";
    const placeholder = document.getElementById("canvasPlaceholder");
    if (placeholder && placeholder.style.display !== "none") {
      const p = placeholder.querySelector("p:first-child");
      if (p) {
        if (imgState.shapeMode === "rect") p.textContent = "Click and drag to draw a rectangle.";
        else if (imgState.shapeMode === "circle") p.textContent = "Click and drag to draw an ellipse.";
        else p.textContent = "Click to place vertices, double-click to close polygon.";
      }
    }
  });
});

function renderCanvas(index) {
  const imgData = imgState.images[index];
  if (!imgData) return;

  canvasPlaceholder.style.display = "none";
  canvas.style.display = "";
  imgState.polygonPoints = [];

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

  canvas._scale = scale;
  imgState._baseRendered = false;

  renderBaseLayer(index);
}

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

function _drawAnnotationShape(ann, scale, strokeColor, lineWidth) {
  const t = ann.type || "rect";
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = lineWidth;
  if (t === "circle") {
    const cx = (ann.x + ann.width / 2) * scale;
    const cy = (ann.y + ann.height / 2) * scale;
    const rx = (ann.width / 2) * scale;
    const ry = (ann.height / 2) * scale;
    ctx.beginPath();
    ctx.ellipse(cx, cy, Math.abs(rx), Math.abs(ry), 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = "rgba(0,255,0,0.08)";
    ctx.fill();
  } else if (t === "polygon") {
    const pts = ann.points || [];
    if (pts.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(pts[0].x * scale, pts[0].y * scale);
    for (let k = 1; k < pts.length; k++) {
      ctx.lineTo(pts[k].x * scale, pts[k].y * scale);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = "rgba(0,255,0,0.08)";
    ctx.fill();
  } else {
    const x = ann.x * scale, y = ann.y * scale;
    const bw = ann.width * scale, bh = ann.height * scale;
    ctx.fillStyle = "rgba(0,255,0,0.08)";
    ctx.fillRect(x, y, bw, bh);
    ctx.strokeRect(x, y, bw, bh);
  }
}

function _drawSelectionHandles(ann, scale) {
  const t = ann.type || "rect";
  const hs = 6;
  ctx.fillStyle = "#fff";
  ctx.strokeStyle = "#ffcc00";
  ctx.lineWidth = 2;

  let handles = [];
  if (t === "circle") {
    const cx = (ann.x + ann.width / 2) * scale;
    const cy = (ann.y + ann.height / 2) * scale;
    const rx = (ann.width / 2) * scale;
    const ry = (ann.height / 2) * scale;
    handles = [
      [cx - rx, cy], [cx + rx, cy],
      [cx, cy - ry], [cx, cy + ry],
    ];
  } else if (t === "polygon") {
    const pts = ann.points || [];
    for (const p of pts) {
      handles.push([p.x * scale, p.y * scale]);
    }
  } else {
    const hx = ann.x * scale, hy = ann.y * scale;
    const hw = ann.width * scale, hh = ann.height * scale;
    handles = [
      [hx, hy], [hx + hw / 2, hy], [hx + hw, hy],
      [hx + hw, hy + hh / 2],
      [hx + hw, hy + hh], [hx + hw / 2, hy + hh], [hx, hy + hh],
      [hx, hy + hh / 2],
    ];
  }
  for (const [px, py] of handles) {
    ctx.fillRect(px - hs, py - hs, hs * 2, hs * 2);
    ctx.strokeRect(px - hs, py - hs, hs * 2, hs * 2);
  }
}

function drawAnnotations() {
  const imgData = imgState.images[imgState.currentIndex];
  if (!imgData) return;
  const scale = canvas._scale || 1;

  for (let i = 0; i < imgData.annotations.length; i++) {
    _drawAnnotationShape(imgData.annotations[i], scale, "#00ff00", i === imgState.selectedAnnotation ? 3 : 2);
  }

  if (imgState.selectedAnnotation >= 0 && imgState.selectedAnnotation < imgData.annotations.length) {
    _drawSelectionHandles(imgData.annotations[imgState.selectedAnnotation], scale);
  }

  // Drawing preview
  if (imgState.currentRect) {
    ctx.strokeStyle = "#ff0000";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 3]);
    const r = imgState.currentRect;
    if (imgState.shapeMode === "circle") {
      ctx.beginPath();
      ctx.ellipse(r.x + r.w / 2, r.y + r.h / 2, Math.abs(r.w / 2), Math.abs(r.h / 2), 0, 0, Math.PI * 2);
      ctx.stroke();
    } else {
      ctx.strokeRect(r.x, r.y, r.w, r.h);
    }
    ctx.setLineDash([]);
  }

  // Polygon drawing preview
  if (imgState.shapeMode === "polygon" && imgState.polygonPoints.length > 0) {
    ctx.strokeStyle = "#ff0000";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 3]);
    ctx.beginPath();
    ctx.moveTo(imgState.polygonPoints[0].x, imgState.polygonPoints[0].y);
    for (let k = 1; k < imgState.polygonPoints.length; k++) {
      ctx.lineTo(imgState.polygonPoints[k].x, imgState.polygonPoints[k].y);
    }
    ctx.lineTo(imgState.polygonCursorX, imgState.polygonCursorY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Vertices
    ctx.fillStyle = "#ff0000";
    for (const p of imgState.polygonPoints) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function renderCurrentFrame() {
  if (!offscreenCanvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(offscreenCanvas, 0, 0);
  drawAnnotations();
}

// ── Canvas mouse events ──

function cursorForHandle(dir) {
  const map = { nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize', n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize' };
  return map[dir] || 'default';
}

function _pointInPolygon(px, py, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x, yi = polygon[i].y;
    const xj = polygon[j].x, yj = polygon[j].y;
    if ((yi > py) !== (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

function _hitTestAnnotation(cx, cy, ann, scale) {
  const t = ann.type || "rect";
  if (t === "circle") {
    const acx = (ann.x + ann.width / 2) * scale;
    const acy = (ann.y + ann.height / 2) * scale;
    const rx = (ann.width / 2) * scale;
    const ry = (ann.height / 2) * scale;
    return ((cx - acx) / rx) ** 2 + ((cy - acy) / ry) ** 2 <= 1.02;
  } else if (t === "polygon") {
    const pts = (ann.points || []).map(p => ({ x: p.x * scale, y: p.y * scale }));
    return pts.length >= 3 && _pointInPolygon(cx, cy, pts);
  } else {
    const tol = 2;
    const ax = ann.x * scale, ay = ann.y * scale;
    return cx >= ax - tol && cx <= ax + ann.width * scale + tol && cy >= ay - tol && cy <= ay + ann.height * scale + tol;
  }
}

function _getBoundingBoxCorners(ann, scale) {
  const t = ann.type || "rect";
  const hs = 10; // hit radius
  const hx = ann.x * scale, hy = ann.y * scale;
  const hw = ann.width * scale, hh = ann.height * scale;

  if (t === "circle") {
    const cx = hx + hw / 2, cy = hy + hh / 2;
    return [
      ['n', cx, hy], ['s', cx, hy + hh],
      ['w', hx, cy], ['e', hx + hw, cy],
    ];
  } else if (t === "polygon") {
    const pts = ann.points || [];
    return pts.map((p, i) => [`v${i}`, p.x * scale, p.y * scale]);
  } else {
    return [
      ['nw', hx, hy], ['n', hx + hw / 2, hy], ['ne', hx + hw, hy],
      ['e', hx + hw, hy + hh / 2],
      ['se', hx + hw, hy + hh], ['s', hx + hw / 2, hy + hh], ['sw', hx, hy + hh],
      ['w', hx, hy + hh / 2],
    ];
  }
}

canvas.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return;
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const scale = canvas._scale || 1;
  const imgData = imgState.images[imgState.currentIndex];
  if (!imgData) return;

  // 1. Always check resize handles first (any mode)
  if (imgState.selectedAnnotation >= 0 && imgState.selectedAnnotation < imgData.annotations.length) {
    const ann = imgData.annotations[imgState.selectedAnnotation];
    const corners = _getBoundingBoxCorners(ann, scale);
    for (const [handle, hx, hy] of corners) {
      if (Math.abs(cx - hx) <= 10 && Math.abs(cy - hy) <= 10) {
        imgState.interactionMode = 'resize';
        imgState.resizeHandle = handle;
        imgState.dragStartX = cx;
        imgState.dragStartY = cy;
        imgState.dragOrigAnn = JSON.parse(JSON.stringify(ann));
        return;
      }
    }
  }

  // 2. Check if clicking on existing annotation (select + move)
  for (let i = imgData.annotations.length - 1; i >= 0; i--) {
    if (_hitTestAnnotation(cx, cy, imgData.annotations[i], scale)) {
      imgState.selectedAnnotation = i;
      imgState.interactionMode = 'move';
      imgState.dragStartX = cx;
      imgState.dragStartY = cy;
      imgState.dragOrigAnn = JSON.parse(JSON.stringify(imgData.annotations[i]));
      renderCurrentFrame();
      return;
    }
  }

  // 3. Polygon mode
  if (imgState.shapeMode === "polygon") {
    if (imgState.polygonPoints.length > 0) {
      if (imgState.polygonPoints.length >= 3) {
        const first = imgState.polygonPoints[0];
        const dist = Math.sqrt((cx - first.x) ** 2 + (cy - first.y) ** 2);
        if (dist < 12) {
          _closePolygon(imgData);
          return;
        }
      }
      imgState.polygonPoints.push({ x: cx, y: cy });
      renderCurrentFrame();
      return;
    }
    // No active polygon — deselect if something is selected
    if (imgState.selectedAnnotation >= 0) {
      imgState.selectedAnnotation = -1;
      renderCurrentFrame();
      return;
    }
    // Start new polygon
    imgState.polygonPoints.push({ x: cx, y: cy });
    renderCurrentFrame();
    return;
  }

  // 4. Start new shape drawing
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

  // Track polygon cursor position (for preview line) — always runs in polygon mode
  if (imgState.shapeMode === "polygon" && imgState.interactionMode !== 'move' && imgState.interactionMode !== 'resize') {
    imgState.polygonCursorX = cx;
    imgState.polygonCursorY = cy;
  }

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
      if ((ann.type || "rect") === "polygon" && ann.points) {
        ann.points = orig.points.map((p, i) => ({
          x: Math.round(Math.max(0, p.x + dx)),
          y: Math.round(Math.max(0, p.y + dy)),
        }));
        ann.x = Math.round(Math.max(0, orig.x + dx));
        ann.y = Math.round(Math.max(0, orig.y + dy));
      } else {
        ann.x = Math.round(Math.max(0, orig.x + dx));
        ann.y = Math.round(Math.max(0, orig.y + dy));
      }
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
    const t = ann.type || "rect";

    if (t === "polygon" && ann.points) {
      const vi = parseInt(imgState.resizeHandle.replace("v", ""));
      if (!isNaN(vi) && vi >= 0 && vi < ann.points.length) {
        ann.points[vi] = {
          x: Math.round(Math.max(0, orig.points[vi].x + dx)),
          y: Math.round(Math.max(0, orig.points[vi].y + dy)),
        };
        _recomputePolygonBounds(ann);
        renderCurrentFrame();
        return;
      }
    }

    let x = orig.x, y = orig.y, w2 = orig.width, h2 = orig.height;
    if (t === "circle") {
      switch (imgState.resizeHandle) {
        case 'n': y = orig.y + dy; h2 = orig.height - dy; break;
        case 's': h2 = orig.height + dy; break;
        case 'w': x = orig.x + dx; w2 = orig.width - dx; break;
        case 'e': w2 = orig.width + dx; break;
      }
    } else {
      switch (imgState.resizeHandle) {
        case 'nw': x = orig.x + dx; y = orig.y + dy; w2 = orig.width - dx; h2 = orig.height - dy; break;
        case 'n':  y = orig.y + dy; h2 = orig.height - dy; break;
        case 'ne': y = orig.y + dy; w2 = orig.width + dx; h2 = orig.height - dy; break;
        case 'e':  w2 = orig.width + dx; break;
        case 'se': w2 = orig.width + dx; h2 = orig.height + dy; break;
        case 's':  h2 = orig.height + dy; break;
        case 'sw': x = orig.x + dx; w2 = orig.width - dx; h2 = orig.height + dy; break;
        case 'w':  x = orig.x + dx; w2 = orig.width - dx; break;
      }
    }
    if (w2 < 3) { w2 = 3; if (imgState.resizeHandle === 'nw' || imgState.resizeHandle === 'sw' || imgState.resizeHandle === 'w') x = orig.x + orig.width - 3; }
    if (h2 < 3) { h2 = 3; if (imgState.resizeHandle === 'nw' || imgState.resizeHandle === 'ne' || imgState.resizeHandle === 'n') y = orig.y + orig.height - 3; }
    ann.x = Math.round(Math.max(0, x));
    ann.y = Math.round(Math.max(0, y));
    ann.width = Math.round(w2);
    ann.height = Math.round(h2);
    renderCurrentFrame();
    return;
  }

  if (imgState.selectedAnnotation >= 0 && imgState.selectedAnnotation < imgData.annotations.length) {
    const ann = imgData.annotations[imgState.selectedAnnotation];
    const corners = _getBoundingBoxCorners(ann, scale);
    for (const [handle, hx, hy] of corners) {
      if (Math.abs(cx - hx) <= 10 && Math.abs(cy - hy) <= 10) {
        canvas.style.cursor = handle.startsWith("v") ? "pointer" : cursorForHandle(handle);
        if (imgState.polygonPoints.length > 0) { renderCurrentFrame(); }
        return;
      }
    }
  }
  for (let i = imgData.annotations.length - 1; i >= 0; i--) {
    if (_hitTestAnnotation(cx, cy, imgData.annotations[i], scale)) {
      canvas.style.cursor = 'move';
      if (imgState.polygonPoints.length > 0) { renderCurrentFrame(); }
      return;
    }
  }
  canvas.style.cursor = 'default';

  if (imgState.polygonPoints.length > 0) {
    renderCurrentFrame();
  }
});

function _closePolygon(imgData) {
  const scale = canvas._scale || 1;
  const pts = imgState.polygonPoints.map(p => ({
    x: Math.round(p.x / scale),
    y: Math.round(p.y / scale),
  }));
  if (pts.length < 3) return;

  // Compute bounding box
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }

  const idx = imgData.annotations.length;
  imgData.annotations.push({
    type: "polygon",
    points: pts,
    x: minX, y: minY,
    width: maxX - minX, height: maxY - minY,
  });
  imgState.selectedAnnotation = idx;
  imgState.polygonPoints = [];
  imgState.interactionMode = 'idle';
  updateBoxCount();
  updateImageList();
  renderCurrentFrame();
}

function _recomputePolygonBounds(ann) {
  if (!ann.points || ann.points.length === 0) return;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of ann.points) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  ann.x = minX;
  ann.y = minY;
  ann.width = maxX - minX;
  ann.height = maxY - minY;
}

canvas.addEventListener("mouseup", () => {
  if (imgState.shapeMode === "polygon" && imgState.polygonPoints.length > 0 && imgState.interactionMode !== 'move' && imgState.interactionMode !== 'resize') {
    return; // vertices added on mousedown
  }

  if (imgState.interactionMode === 'draw') {
    imgState.isDrawing = false;
    const r = imgState.currentRect;
    if (r && r.w > 5 && r.h > 5) {
      const scale = canvas._scale || 1;
      const imgData = imgState.images[imgState.currentIndex];
      if (imgData) {
        const idx = imgData.annotations.length;
        const x = Math.round(r.x / scale);
        const y = Math.round(r.y / scale);
        const w = Math.round(r.w / scale);
        const h = Math.round(r.h / scale);
        if (imgState.shapeMode === "circle") {
          imgData.annotations.push({ type: "circle", x, y, width: w, height: h });
        } else {
          imgData.annotations.push({ x, y, width: w, height: h });
        }
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

canvas.addEventListener("dblclick", (e) => {
  if (imgState.shapeMode === "polygon" && imgState.polygonPoints.length >= 3) {
    const imgData = imgState.images[imgState.currentIndex];
    if (imgData) _closePolygon(imgData);
  }
});

canvas.addEventListener("mouseleave", () => {
  if (imgState.interactionMode !== 'idle') {
    if (imgState.shapeMode === "polygon" && imgState.polygonPoints.length > 0) {
      // Keep polygon points but stop tracking cursor
      renderCurrentFrame();
      return;
    }
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
