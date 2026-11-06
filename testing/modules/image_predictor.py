"""Inference for the normalized-layout detector."""

from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageDraw

from core.config import MODEL_TEMP_DIR
from modules import layout_detector as ld
from modules.model_manager import get_model_path


def load_detector(
    model_name: str = "", upload_token: str = "", session_id: str = ""
) -> tuple[object | None, dict]:
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
    if isinstance(data, dict):
        metadata = dict(data.get("metadata") or {})
        model_type = metadata.get("type")
        if model_type == "layout_detector":
            return None, metadata
        if model_type == "image_detector":
            raise ValueError(
                "This is a legacy patch-classifier model. Retrain it with the "
                "current layout detector."
            )
    raise ValueError("Unsupported model file. Train a new detector in the Training tab.")


def _cancelled_default() -> bool:
    return False


def _clamp_box(
    x: float, y: float, width: float, height: float, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """Clamp to the image while always keeping a usable, non-empty box."""
    x1 = max(0, min(image_width - 2, int(round(x))))
    y1 = max(0, min(image_height - 2, int(round(y))))
    x2 = max(x1 + 2, min(image_width, int(round(x + width))))
    y2 = max(y1 + 2, min(image_height, int(round(y + height))))
    return x1, y1, x2, y2


def predict_images(
    model: object | None,
    metadata: dict,
    images_b64: list[dict[str, str]],
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.2,
    scales: list[float] | None = None,
    min_support: int = 1,
    cancelled_fn=_cancelled_default,
) -> tuple[bytes, list[dict]]:
    """Detect the learned objects on every image and build a ZIP of results.

    ``confidence_threshold``/``nms_threshold`` are accepted for backwards
    compatibility and intentionally ignored: the layout detector always
    returns exactly the number of objects it was trained on.
    """
    if metadata.get("type") != "layout_detector":
        raise ValueError("Unsupported model metadata")
    if not metadata.get("layout"):
        raise ValueError("Model has no learned layout")

    expected_count = int(metadata.get("count", len(metadata["layout"])))
    results: list[dict[str, Any]] = []
    zip_buffer = io.BytesIO()
    used_stems: set[str] = set()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for entry in images_b64:
            if cancelled_fn():
                return b"", []

            filename = entry.get("filename", "image.png")
            stem = Path(filename).stem
            unique_stem = stem
            suffix = 2
            while unique_stem in used_stems:
                unique_stem = f"{stem}_{suffix}"
                suffix += 1
            used_stems.add(unique_stem)
            raw = base64.b64decode(entry["data"])
            buffer = io.BytesIO(raw)
            pil_image = Image.open(buffer).convert("RGB")
            image_width, image_height = pil_image.size

            rgb = np.array(pil_image, dtype=np.uint8)
            bgr = rgb[:, :, ::-1].copy()

            detections, _anchor = ld.predict_boxes(metadata, bgr)
            # The layout guarantees this; keep the check as a hard invariant.
            if len(detections) != expected_count:
                raise RuntimeError(
                    f"Internal error: expected {expected_count} boxes, got {len(detections)}"
                )

            draw = ImageDraw.Draw(pil_image)
            line_width = max(2, int(round(min(image_width, image_height) / 300)))
            result_detections: list[dict[str, Any]] = []

            for index, detection in enumerate(detections, start=1):
                x1, y1, x2, y2 = _clamp_box(
                    detection["x"], detection["y"], detection["w"], detection["h"],
                    image_width, image_height,
                )
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=line_width)
                label = f"object {index}"
                text_y = y1 - 14 if y1 >= 16 else y1 + 2
                try:
                    draw.text(
                        (x1 + 2, text_y), label, fill=(0, 255, 0),
                        stroke_width=2, stroke_fill=(0, 0, 0),
                    )
                except TypeError:  # very old Pillow
                    draw.text((x1 + 2, text_y), label, fill=(0, 255, 0))

                result_detections.append({
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "confidence": 1.0,
                    "label": label,
                })

                crop = pil_image.crop((x1, y1, x2, y2))
                crop_buffer = io.BytesIO()
                crop.save(crop_buffer, format="PNG")
                archive.writestr(f"{unique_stem}_object_{index}.png", crop_buffer.getvalue())

            annotated_buffer = io.BytesIO()
            pil_image.save(annotated_buffer, format="PNG")
            archive.writestr(f"{unique_stem}_annotated.png", annotated_buffer.getvalue())

            results.append({
                "filename": filename,
                "width": image_width,
                "height": image_height,
                "count": len(result_detections),
                "detections": result_detections,
            })

    return zip_buffer.getvalue(), results
