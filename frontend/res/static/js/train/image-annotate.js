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

// ── Canvas mouse events ──

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
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const scale = canvas._scale || 1;
  const imgData = imgState.images[imgState.currentIndex];
  if (!imgData) return;

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

    let x = orig.x, y = orig.y, w2 = orig.width, h2 = orig.height;
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
  if (imgState.interactionMode === 'draw') {
    imgState.currentRect = null;
    imgState.isDrawing = false;
    imgState.interactionMode = 'idle';
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
