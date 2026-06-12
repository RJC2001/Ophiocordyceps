from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageDraw

from core.config import MODELS_DIR, MODEL_TEMP_DIR
from modules.model_manager import get_model_path


def _iou(
    x1: float, y1: float, w1: float, h1: float,
    x2: float, y2: float, w2: float, h2: float,
) -> float:
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)
    if left < right and top < bottom:
        inter = (right - left) * (bottom - top)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0
    return 0


def _nms(detections: list[dict], iou_threshold: float = 1) -> list[dict]:
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)
    result: list[dict] = []
    while sorted_dets:
        best = sorted_dets.pop(0)
        result.append(best)
        sorted_dets = [
            d for d in sorted_dets
            if _iou(d["x"], d["y"], d["width"], d["height"],
                    best["x"], best["y"], best["width"], best["height"]) < iou_threshold
        ]
    return result


def load_detector(model_name: str = "", upload_token: str = "", session_id: str = "") -> tuple[object, dict]:
    if model_name:
        model_path = get_model_path(model_name, "image")
    elif upload_token:
        if not session_id:
            raise ValueError("session_id required with upload_token")
        model_path = MODEL_TEMP_DIR / f"{session_id}_test_{upload_token}.model"
        if not model_path.exists():
            raise FileNotFoundError(f"Uploaded model not found: {upload_token}")
    else:
        raise ValueError("Either model_name or upload_token required")

    data = joblib.load(model_path)
    if isinstance(data, dict) and "model" in data:
        meta = data.get("metadata", {})
        # Normalize metadata to include pw/ph regardless of version
        if "patch_width" not in meta:
            ps = meta.get("patch_size", 32)
            meta["patch_width"] = ps
            meta["patch_height"] = ps
        return data["model"], meta
    return data, {"patch_width": 32, "patch_height": 32}


def _cancelled_default() -> bool:
    return False


def predict_images(
    model: object,
    metadata: dict,
    images_b64: list[dict[str, str]],
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.0,
    scales: list[float] | None = None,
    cancelled_fn=_cancelled_default,
) -> tuple[bytes, list[dict]]:
    if scales is None:
        scales = [1.0, 1.5, 0.75]
    pw = metadata.get("patch_width") or metadata.get("patch_size", 32)
    ph = metadata.get("patch_height") or metadata.get("patch_size", 32)
    stride_x = max(pw // 2, 4)
    stride_y = max(ph // 2, 4)

    all_results: list[dict] = []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_entry in images_b64:
            if cancelled_fn():
                return b"", []
            filename = img_entry.get("filename", "image.png")
            raw = base64.b64decode(img_entry["data"])
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            orig_w, orig_h = img.size

            all_detections: list[dict] = []

            for scale in scales:
                if cancelled_fn():
                    return b"", []
                sw = int(orig_w * scale)
                sh = int(orig_h * scale)
                if sw < pw or sh < ph:
                    continue
                resized = img.resize((sw, sh), Image.Resampling.LANCZOS)
                resized_arr = np.array(resized, dtype=np.float32) / 255.0

                patch_features: list[np.ndarray] = []
                patch_coords: list[tuple[int, int]] = []
                for y in range(0, sh - ph + 1, stride_y):
                    for x in range(0, sw - pw + 1, stride_x):
                        patch_features.append(resized_arr[y:y+ph, x:x+pw].flatten())
                        patch_coords.append((x, y))

                if not patch_features:
                    continue

                feat_matrix = np.array(patch_features)
                try:
                    probabilities = model.predict_proba(feat_matrix)
                    positive_idx = 1
                    if hasattr(model, "classes_"):
                        classes = list(model.classes_)
                        positive_idx = classes.index(1) if 1 in classes else (
                            classes.index(True) if True in classes else 1
                        )
                    for (x, y), confidence in zip(patch_coords, probabilities[:, positive_idx]):
                        if confidence >= confidence_threshold:
                            all_detections.append({
                                "x": x / scale,
                                "y": y / scale,
                                "width": pw / scale,
                                "height": ph / scale,
                                "confidence": float(confidence),
                            })
                except Exception:
                    pass

            filtered = _nms(all_detections, iou_threshold=nms_threshold)
            filtered.sort(key=lambda d: d.get("confidence", 0), reverse=True)

            draw = ImageDraw.Draw(img)
            for det in filtered:
                x, y, w, h = det["x"], det["y"], det["width"], det["height"]
                draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)
                label = f"{det['confidence']:.2f}"
                draw.text((x, max(0, y - 14)), label, fill=(0, 255, 0))

            stem = Path(filename).stem
            annotated_buf = io.BytesIO()
            img.save(annotated_buf, format="PNG")
            zf.writestr(f"{stem}_annotated.png", annotated_buf.getvalue())

            result_dets: list[dict] = []
            for idx, det in enumerate(filtered, start=1):
                result_dets.append({
                    "x": int(det["x"]),
                    "y": int(det["y"]),
                    "w": int(det["width"]),
                    "h": int(det["height"]),
                    "confidence": det["confidence"],
                })
                x, y, w, h = det["x"], det["y"], det["width"], det["height"]
                x = max(0, int(x))
                y = max(0, int(y))
                w = min(int(w), orig_w - x)
                h = min(int(h), orig_h - y)
                if w > 4 and h > 4:
                    crop = img.crop((x, y, x + w, y + h))
                    crop_buf = io.BytesIO()
                    crop.save(crop_buf, format="PNG")
                    zf.writestr(f"{stem}_object_{idx}.png", crop_buf.getvalue())

            all_results.append({
                "filename": filename,
                "width": orig_w,
                "height": orig_h,
                "detections": result_dets,
            })

    return buf.getvalue(), all_results
