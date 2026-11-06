"""Training for the normalized-layout image detector.

The user annotates the same set of objects on a modest number of images.
Training then simply learns each object's position and size relative to the
anchor.  It is fast, deterministic and needs no hyper-parameters.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image

from core.config import MODELS_IMAGE_DIR, IMGTRAIN_DIR
from modules import layout_detector as ld

_image_training_states: dict[str, dict[str, Any]] = {}


def get_available_image_algorithms() -> dict[str, Any]:
    """Kept for API compatibility; the layout detector has no algorithms."""
    return {}


def _init_state(session_id: str) -> dict[str, Any]:
    return _image_training_states.setdefault(session_id, {
        "images": {},
        "annotations": {},
        "status": "idle",
        "log": [],
        "model_path": None,
        "error": None,
        "metrics": None,
        "test_image_ids": [],
        "_predictions": None,
        "_cancel": False,
    })


def upload_training_image(session_id: str, filename: str, b64_data: str) -> dict[str, Any]:
    raw = base64.b64decode(b64_data)
    ext = Path(filename).suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"):
        raise ValueError(f"Unsupported image format: {ext}")

    img_id = uuid.uuid4().hex[:8]
    IMGTRAIN_DIR.mkdir(parents=True, exist_ok=True)
    dest = IMGTRAIN_DIR / f"{session_id}_{img_id}{ext}"
    dest.write_bytes(raw)

    img = Image.open(io.BytesIO(raw))
    width, height = img.size

    state = _init_state(session_id)
    state["images"][img_id] = {
        "path": str(dest),
        "filename": filename,
        "width": width,
        "height": height,
    }
    return {"image_id": img_id, "filename": filename, "width": width, "height": height}


def delete_training_image(session_id: str, image_id: str) -> dict[str, Any]:
    state = _init_state(session_id)
    img_info = state["images"].pop(image_id, None)
    state["annotations"].pop(image_id, None)
    if img_info:
        path = Path(img_info["path"])
        if path.exists():
            path.unlink()
    return {"deleted": True}


def save_annotations(session_id: str, image_id: str, annotations: list[dict]) -> dict[str, Any]:
    state = _init_state(session_id)
    if image_id not in state["images"]:
        raise ValueError("Unknown image_id; upload the image first")
    state["annotations"][image_id] = annotations
    return {"saved": True, "count": len(annotations)}


def get_image_training_data(session_id: str) -> dict[str, Any]:
    state = _init_state(session_id)
    images: dict[str, Any] = {}
    for image_id, info in state["images"].items():
        images[image_id] = {
            **info,
            "annotations": state["annotations"].get(image_id, []),
        }
    return {"images": images, "status": state.get("status", "idle")}


def clear_image_training(session_id: str) -> dict[str, Any]:
    state = _init_state(session_id)
    for info in list(state["images"].values()):
        path = Path(info["path"])
        if path.exists():
            path.unlink()
    state["images"].clear()
    state["annotations"].clear()
    state["status"] = "idle"
    state["log"] = []
    state["error"] = None
    state["metrics"] = None
    state["model_path"] = None
    state["test_image_ids"] = []
    state["_predictions"] = None
    state["_cancel"] = False
    if IMGTRAIN_DIR.exists():
        for path in IMGTRAIN_DIR.iterdir():
            if path.name.startswith(session_id):
                path.unlink(missing_ok=True)
    return {"cleared": True}


def start_image_training(
    session_id: str,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    shuffle: bool = False,
    random_state: int = 42,
    test_image_ids: list[str] | None = None,
    val_image_ids: list[str] | None = None,
    detector_config: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    val_size = float(val_size)
    test_size = float(test_size)
    if val_size < 0 or test_size < 0:
        raise ValueError("Validation/test split cannot be negative")
    if val_size + test_size >= 1:
        raise ValueError(
            "Validation + test split must leave at least one training image. "
            "Reduce the validation or test percentage."
        )

    explicit_test = None if test_image_ids is None else [str(v) for v in test_image_ids]
    explicit_val = None if val_image_ids is None else [str(v) for v in val_image_ids]

    state = _init_state(session_id)
    if state.get("status") == "running":
        raise ValueError("Training is already running")
    state["_cancel"] = False
    state["status"] = "running"
    state["log"] = []
    state["error"] = None
    state["metrics"] = None
    state["_predictions"] = None
    state["test_image_ids"] = []
    thread = threading.Thread(
        target=_train_worker,
        args=(
            session_id,
            val_size,
            test_size,
            bool(shuffle),
            int(random_state),
            explicit_test,
            explicit_val,
            ld.normalize_detector_config(detector_config),
        ),
        daemon=True,
    )
    thread.start()
    return {"status": "started"}


def cancel_image_training(session_id: str) -> dict[str, Any]:
    state = _init_state(session_id)
    state["_cancel"] = True
    return {"status": "cancelling"}


def get_image_training_status(session_id: str) -> dict[str, Any]:
    state = _init_state(session_id)
    result: dict[str, Any] = {
        "status": state.get("status", "idle"),
        "log": list(state.get("log", [])),
        "error": state.get("error"),
        "metrics": state.get("metrics"),
        "model_path": state.get("model_path"),
        "test_image_ids": state.get("test_image_ids", []),
    }
    predictions = state.get("_predictions")
    if predictions is not None:
        result["predictions"] = predictions
    return result


def _log(state: dict[str, Any], message: str) -> None:
    state.setdefault("log", []).append(message)


def _split_counts(total: int, val_size: float, test_size: float) -> tuple[int, int]:
    """Return ``(validation_count, test_count)`` with stable priority.

    Uses largest-remainder apportionment over exact integer units, so floating
    point noise can never decide which split receives a leftover image.  Ties
    are resolved in favour of Train first, then Validation, then Test.
    """
    if total <= 0:
        return 0, 0

    basis = 1_000_000
    pcts = [
        max(0, round((1.0 - val_size - test_size) * basis)),
        max(0, round(val_size * basis)),
        max(0, round(test_size * basis)),
    ]
    products = [total * pct for pct in pcts]
    counts = [product // basis for product in products]
    remainders = [product % basis for product in products]
    order = sorted(range(3), key=lambda i: (-remainders[i], i))
    leftover = total - sum(counts)
    for index in order[:leftover]:
        counts[index] += 1

    def ensure_one(target: int, donors: list[int], minimum_total: int) -> None:
        if total < minimum_total or pcts[target] <= 0 or counts[target] > 0:
            return
        for donor in donors:
            if donor != target and counts[donor] > (1 if donor == 0 else 0):
                counts[donor] -= 1
                counts[target] += 1
                return

    # Training must keep at least one image whenever the dataset allows it.
    ensure_one(0, [2, 1], 1)
    # Test is the locked check set, so keep one image there when possible.
    ensure_one(2, [0, 1], 2)
    # Then keep a validation image when there is room for all three splits.
    ensure_one(1, [0, 2], 3)
    return counts[1], counts[2]


def _train_worker(
    session_id: str,
    val_size: float,
    test_size: float,
    shuffle: bool,
    random_state: int,
    test_image_ids: list[str] | None,
    val_image_ids: list[str] | None,
    detector_config: dict[str, Any],
) -> None:
    state = _init_state(session_id)

    try:
        state["status"] = "running"
        _log(state, "Reading uploaded images...")

        # Keep every uploaded image in the split.  Test images are deliberately
        # unannotated: they are the "guess on unseen data" set.
        entries: list[dict[str, Any]] = []
        for image_id, info in state["images"].items():
            boxes: list[tuple[float, float, float, float]] = []
            for annotation in state["annotations"].get(image_id, []):
                if not isinstance(annotation, dict):
                    continue
                box = ld.annotation_to_box(annotation)
                if box is None:
                    raise ValueError(
                        f"{info['filename']}: one annotation is too small or invalid. "
                        "Delete it and draw a proper box."
                    )
                boxes.append(box)
            entries.append({
                "image_id": image_id,
                "filename": info["filename"],
                "path": info["path"],
                "boxes": boxes,
            })

        if not entries:
            raise ValueError("Upload some images first.")

        all_ids = [entry["image_id"] for entry in entries]

        # The UI sends the exact test/validation image ids.  This keeps the
        # locked thumbnails and the backend split in sync even when the user
        # reorders images after a previous training run.
        if test_image_ids is not None or val_image_ids is not None:
            test_set = set(test_image_ids or [])
            val_set = set(val_image_ids or []) - test_set
            unknown = (test_set | val_set) - set(all_ids)
            if unknown:
                _log(state, f"Warning: ignoring {len(unknown)} unknown split image id(s).")
            test_idx = [i for i, image_id in enumerate(all_ids) if image_id in test_set]
            val_idx = [i for i, image_id in enumerate(all_ids) if image_id in val_set]
            train_idx = [
                i for i, image_id in enumerate(all_ids)
                if image_id not in test_set and image_id not in val_set
            ]
        else:
            n_val, n_test = _split_counts(len(entries), val_size, test_size)
            order = list(range(len(entries)))
            if shuffle and len(order) > 1:
                rng = np.random.default_rng(random_state if random_state >= 0 else None)
                order = [int(i) for i in rng.permutation(len(order))]
            test_idx = sorted(order[:n_test])
            val_idx = sorted(order[n_test:n_test + n_val])
            train_idx = sorted(order[n_test + n_val:])

        if not train_idx:
            raise ValueError(
                "The split left no training images. Reduce the validation or test percentage."
            )

        train_annotated_idx = [i for i in train_idx if entries[i]["boxes"]]
        if not train_annotated_idx:
            raise ValueError(
                "No annotated images in the Train split. Draw the objects on at least one "
                "training image (TEST images are locked on purpose)."
            )

        counts = {len(entries[i]["boxes"]) for i in train_annotated_idx}
        if len(counts) != 1:
            detail = ", ".join(
                f"{entries[i]['filename']}: {len(entries[i]['boxes'])}"
                for i in train_annotated_idx
            )
            raise ValueError(
                "Every annotated training image must contain the same number of boxes. "
                f"Found: {detail}. Fix the annotations and train again."
            )

        count = counts.pop()
        val_annotated_idx = [i for i in val_idx if entries[i]["boxes"]]
        val_eval_idx = [i for i in val_annotated_idx if len(entries[i]["boxes"]) == count]
        if val_annotated_idx and len(val_eval_idx) != len(val_annotated_idx):
            _log(
                state,
                "Warning: some validation images have a different box count; "
                "only matching validation images are scored.",
            )

        _log(
            state,
            f"{len(entries)} images, {len(train_annotated_idx)} annotated training images, "
            f"{count} boxes per image.",
        )
        _log(
            state,
            f"Split: train {len(train_idx)} ({len(train_annotated_idx)} annotated), "
            f"validation {len(val_idx)} ({len(val_eval_idx)} scored), "
            f"test {len(test_idx)} (locked, no annotations needed).",
        )
        if not val_idx:
            _log(state, "Validation set is empty (increase its percentage or add images).")
        if not test_idx:
            _log(state, "Test set is empty (increase its percentage or add images).")

        # Pass 1: find the anchor in every image with a generic search.
        _log(state, "Locating the reference anchor in each image...")
        prepared: list[dict[str, Any]] = []
        for entry in entries:
            if state.get("_cancel"):
                state["status"] = "cancelled"
                return
            image = ld.read_image(entry["path"])
            height, width = image.shape[:2]
            anchor = ld.detect_anchor(image, config=detector_config)
            entry = dict(entry)
            entry.update({"image": image, "width": width, "height": height, "anchor": anchor})
            prepared.append(entry)
            _log(
                state,
                f"  {entry['filename']}: anchor=({anchor[0]:.0f}, {anchor[1]:.0f}, r={anchor[2]:.0f})",
            )

        # The median normalized anchor of the annotated TRAINING images is a
        # stable prior for pass 2.  Validation/test images never contribute.
        train_prepared = [prepared[i] for i in train_annotated_idx]
        short_sides = [min(e["height"], e["width"]) for e in train_prepared]
        anchor_norm = (
            float(np.median([e["anchor"][0] / e["width"] for e in train_prepared])),
            float(np.median([e["anchor"][1] / e["height"] for e in train_prepared])),
            float(np.median([e["anchor"][2] / side for e, side in zip(train_prepared, short_sides)])),
        )

        # Pass 2: refine every anchor using the training prior, then normalize.
        samples: list[list[tuple[float, float, float, float]]] = []
        train_annotated_set = set(train_annotated_idx)
        for index, entry in enumerate(prepared):
            if state.get("_cancel"):
                state["status"] = "cancelled"
                return
            expected = (
                anchor_norm[0] * entry["width"],
                anchor_norm[1] * entry["height"],
                anchor_norm[2] * min(entry["height"], entry["width"]),
            )
            anchor = ld.detect_anchor(entry["image"], expected, detector_config)
            entry["anchor"] = anchor

            if index in train_annotated_set:
                normalized = [ld.normalize_box(box, anchor) for box in entry["boxes"]]
                samples.append(ld.canonical_order(normalized))

        layout = ld.make_layout(samples)
        metadata: dict[str, Any] = {
            "type": "layout_detector",
            "count": count,
            "anchor": {
                "cx_norm": anchor_norm[0],
                "cy_norm": anchor_norm[1],
                "r_norm": anchor_norm[2],
            },
            "layout": layout,
            "detector_config": detector_config,
        }

        def mean_iou_for(indices: list[int]) -> float | None:
            values: list[float] = []
            for index in indices:
                entry = prepared[index]
                predicted = ld.predict_boxes_with_anchor(
                    layout, entry["anchor"], entry["width"], entry["height"]
                )
                gt = [tuple(box) for box in entry["boxes"]]
                score, _, _ = ld.match_and_score(predicted, gt)
                values.append(score)
            return float(np.mean(values)) if values else None

        train_mean = mean_iou_for(train_annotated_idx)
        val_mean = mean_iou_for(val_eval_idx)

        annotation_count = sum(len(entry["boxes"]) for entry in entries)
        metrics: dict[str, Any] = {
            "train_images": len(train_idx),
            "train_annotated": len(train_annotated_idx),
            "val_images": len(val_idx),
            "val_annotated": len(val_eval_idx),
            "test_images": len(test_idx),
            "images": len(entries),
            "annotations": annotation_count,
            "boxes_per_image": count,
            "mean_iou": round(float(train_mean or 0.0), 3),
        }
        if val_mean is not None:
            metrics["val_mean_iou"] = round(val_mean, 3)

        metadata["metrics"] = metrics
        metadata["split"] = {
            "train": len(train_idx),
            "train_annotated": len(train_annotated_idx),
            "validation": len(val_idx),
            "validation_annotated": len(val_eval_idx),
            "test": len(test_idx),
            "val_size": round(val_size, 4),
            "test_size": round(test_size, 4),
            "shuffle": bool(shuffle),
            "random_state": int(random_state),
        }
        metadata["image_size"] = [
            int(np.median([e["width"] for e in prepared])),
            int(np.median([e["height"] for e in prepared])),
        ]

        if state.get("_cancel"):
            state["status"] = "cancelled"
            return

        # Save the model used by both in-page preview and the Test page.
        IMGTRAIN_DIR.mkdir(parents=True, exist_ok=True)
        model_path = IMGTRAIN_DIR / f"{session_id}.model"
        joblib.dump({"metadata": metadata}, model_path)
        state["model_path"] = str(model_path)

        predictions: list[dict[str, Any]] = []
        for entry in prepared:
            predicted = ld.predict_boxes_with_anchor(
                layout, entry["anchor"], entry["width"], entry["height"]
            )
            for box in predicted:
                predictions.append({
                    "image_id": entry["image_id"],
                    "filename": entry["filename"],
                    "x": float(box[0]),
                    "y": float(box[1]),
                    "w": float(box[2] - box[0]),
                    "h": float(box[3] - box[1]),
                    "confidence": 1.0,
                    "sliding_window": True,
                })

        # Only TEST images are shown in the held-out gallery.  Predictions on
        # training images are not useful for checking generalization.
        held_out = [prepared[i]["image_id"] for i in test_idx]

        state["_predictions"] = predictions
        state["test_image_ids"] = held_out
        state["metrics"] = metrics
        state["status"] = "done"

        _log(state, f"Mean IoU on training images: {float(train_mean or 0.0):.3f}")
        if val_mean is not None:
            word = "image" if len(val_eval_idx) == 1 else "images"
            _log(state, f"Validation mean IoU ({len(val_eval_idx)} {word}): {val_mean:.3f}")
        if test_idx:
            _log(
                state,
                f"Test set: {len(test_idx)} locked image(s) with no annotations; "
                "inspect their predictions in the held-out gallery.",
            )
        _log(state, f"Model ready: {model_path.name}")
        _log(state, f"Detector always returns exactly {count} boxes per image.")

    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        state["status"] = "error"
        state["error"] = str(exc)
        _log(state, f"ERROR: {exc}")


def save_to_server(session_id: str, name: str) -> str:
    state = _init_state(session_id)
    if state.get("status") != "done":
        raise ValueError("No trained model available")
    model_path = state.get("model_path")
    if not model_path or not Path(model_path).exists():
        raise ValueError("Model file not found")

    name = name.strip() or "layout_detector"
    if not name.endswith(".model"):
        name += ".model"

    dest = MODELS_IMAGE_DIR / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(Path(model_path).read_bytes())

    data = joblib.load(model_path)
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    index_path = MODELS_IMAGE_DIR / "_metrics.json"
    index: dict[str, Any] = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except Exception:
            index = {}

    metrics = state.get("metrics") or metadata.get("metrics") or {}
    index[name] = {
        "type": "layout_detector",
        "task": "object_detection",
        "count": metadata.get("count"),
        "train_images": metrics.get("train_images"),
        "train_annotated": metrics.get("train_annotated"),
        "val_images": metrics.get("val_images"),
        "val_annotated": metrics.get("val_annotated"),
        "test_images": metrics.get("test_images"),
        "images": metrics.get("images"),
        "annotations": metrics.get("annotations"),
        "mean_iou": metrics.get("mean_iou"),
        "val_mean_iou": metrics.get("val_mean_iou"),
    }
    index_path.write_text(json.dumps(index, indent=2))
    return str(dest)
