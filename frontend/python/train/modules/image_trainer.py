from __future__ import annotations

import base64
import io
import json
import threading
import uuid
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold

from core.config import MODELS_IMAGE_DIR, IMGTRAIN_DIR

_image_training_states: dict[str, dict[str, Any]] = {}

IMAGE_ALGORITHMS: dict[str, Any] = {
    "cnn": {
        "label": "CNN (Convolutional Neural Network)",
        "description": "Convolutional Neural Network for image recognition. Uses a multi-layer perceptron with deep architecture optimized for visual pattern recognition.",
        "params": {
            "conv_layers": {
                "type": "string",
                "default": "512,256,128",
                "hint": "comma-separated layer sizes, e.g. 512,256,128",
            },
            "learning_rate": {
                "type": "float",
                "default": 0.001,
                "min": 0.00001,
                "max": 1.0,
            },
            "epochs": {"type": "int", "default": 1000, "min": 1, "max": 10000},
            "batch_size": {"type": "int", "default": 0, "min": 0, "max": 1024},
        },
    },
    "random_forest": {
        "label": "Random Forest",
        "description": "Ensemble of decision trees. Handles non-linear relationships, resistant to overfitting. Good default choice for classification and regression.",
        "params": {
            "n_estimators": {"type": "int", "default": 100, "min": 1, "max": 2000},
            "max_depth": {"type": "int", "default": 0, "min": 0, "max": 200},
        },
    },
    "logistic": {
        "label": "Logistic Regression",
        "description": "Simple, fast, interpretable. Logistic Regression for classification, Linear Regression for regression. Works best with linear relationships.",
        "params": {
            "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100.0},
        },
    },
    "svm": {
        "label": "SVM",
        "description": "Support Vector Machine. Effective in high-dimensional spaces. Works well with clear margins of separation. Kernel trick handles non-linearity.",
        "params": {
            "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100.0},
            "kernel": {
                "type": "select",
                "default": "rbf",
                "options": ["linear", "rbf", "poly", "sigmoid"],
            },
        },
    },
    "neural_network": {
        "label": "Neural Network (MLP)",
        "description": "Multi-layer Perceptron. Can learn complex non-linear patterns. Requires more data and hyperparameter tuning. Slower training.",
        "params": {
            "hidden_layers": {
                "type": "string",
                "default": "100",
                "hint": "comma-separated, e.g. 100,50",
            },
            "learning_rate": {
                "type": "float",
                "default": 0.001,
                "min": 0.00001,
                "max": 1.0,
            },
            "epochs": {"type": "int", "default": 1000, "min": 1, "max": 10000},
            "batch_size": {"type": "int", "default": 0, "min": 0, "max": 1024},
        },
    },
}


def get_available_image_algorithms() -> dict[str, Any]:
    return {k: dict(v, available=True) for k, v in IMAGE_ALGORITHMS.items()}


def _init_state(session_id: str) -> dict[str, Any]:
    state = _image_training_states.setdefault(session_id, {
        "images": {},
        "annotations": {},
        "status": "idle",
        "log": [],
        "model_path": None,
        "error": None,
        "metrics": None,
    })
    return state


def upload_training_image(session_id: str, filename: str, b64_data: str) -> dict[str, Any]:
    raw = base64.b64decode(b64_data)
    ext = Path(filename).suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"):
        raise ValueError(f"Unsupported image format: {ext}")

    img_id = uuid.uuid4().hex[:8]
    IMGTRAIN_DIR.mkdir(parents=True, exist_ok=True)
    safe = f"{session_id}_{img_id}{ext}"
    dest = IMGTRAIN_DIR / safe
    dest.write_bytes(raw)

    img = Image.open(io.BytesIO(raw))
    w, h = img.size

    state = _init_state(session_id)
    state["images"][img_id] = {
        "path": str(dest),
        "filename": filename,
        "width": w,
        "height": h,
    }

    return {"image_id": img_id, "filename": filename, "width": w, "height": h}


def delete_training_image(session_id: str, image_id: str) -> dict[str, Any]:
    state = _init_state(session_id)
    img_info = state["images"].pop(image_id, None)
    state["annotations"].pop(image_id, None)
    if img_info:
        p = Path(img_info["path"])
        if p.exists():
            p.unlink()
    return {"deleted": True}


def save_annotations(session_id: str, image_id: str, annotations: list[dict]) -> dict[str, Any]:
    state = _init_state(session_id)
    state["annotations"][image_id] = annotations
    return {"saved": True, "count": len(annotations)}


def get_image_training_data(session_id: str) -> dict[str, Any]:
    state = _init_state(session_id)
    imgs = {}
    for img_id, info in state["images"].items():
        anns = state["annotations"].get(img_id, [])
        imgs[img_id] = {**info, "annotations": anns}
    return {
        "images": imgs,
        "status": state.get("status", "idle"),
    }


def clear_image_training(session_id: str) -> dict[str, Any]:
    state = _init_state(session_id)
    for img_id, info in list(state["images"].items()):
        p = Path(info["path"])
        if p.exists():
            p.unlink()
    state["images"].clear()
    state["annotations"].clear()
    state["status"] = "idle"
    state["log"] = []
    state["error"] = None
    state["metrics"] = None
    state["model_path"] = None
    state["_predictions"] = None
    if IMGTRAIN_DIR.exists():
        for f in IMGTRAIN_DIR.iterdir():
            if f.name.startswith(session_id):
                f.unlink()
    return {"cleared": True}


def _feature_to_image(feature: np.ndarray, pw: int = 32, ph: int = 32) -> str:
    n_features = feature.size
    if n_features == pw * ph:
        arr = (feature * 255).astype(np.uint8).reshape(ph, pw)
        img = Image.fromarray(arr, mode="L")
    elif n_features == pw * ph * 3:
        arr = (feature * 255).astype(np.uint8).reshape(ph, pw, 3)
        img = Image.fromarray(arr, mode="RGB")
    else:
        side = int(np.sqrt(n_features // 3))
        arr = (feature * 255).astype(np.uint8).reshape(side, side, 3)
        img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def start_image_training(
    session_id: str,
    val_strategy: str = "simple",
    val_size: float = 0.0,
    test_size: float = 0.0,
    algorithm: str = "random_forest",
    neg_per_ann: int = 10,
    params: dict[str, Any] | None = None,
    shuffle: bool = True,
    random_state: int = 42,
    k_folds: int = 5,
    patch_size: int = 0,
    test_image_ids: list[str] | None = None,
    stride: int = 0,
) -> dict[str, Any]:
    state = _init_state(session_id)
    state["_cancel"] = False
    thread = threading.Thread(
        target=_train_worker,
        args=(session_id, val_strategy, val_size, test_size, algorithm, neg_per_ann, params, shuffle, random_state, k_folds, patch_size, test_image_ids, stride),
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
    preds = state.get("_predictions")
    if preds is not None:
        result["predictions"] = preds
    raw = state.get("_raw_predictions")
    if raw is not None:
        result["raw_predictions"] = raw
    return result


def _log(state: dict[str, Any], msg: str) -> None:
    state["log"].append(msg)
    if len(state["log"]) > 500:
        state["log"] = state["log"][-500:]


def _extract_patch(img: Image.Image, x: int, y: int, w: int, h: int, pw: int = 32, ph: int = 32) -> np.ndarray | None:
    x, y, w, h = int(x), int(y), int(w), int(h)
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.width - x)
    h = min(h, img.height - y)
    if w <= 0 or h <= 0:
        return None
    patch = img.crop((x, y, x + w, y + h))
    patch = patch.resize((pw, ph), Image.Resampling.LANCZOS)
    patch = patch.convert("RGB")
    arr = np.array(patch, dtype=np.float32) / 255.0
    return arr.flatten()


def _augment_patch(features: np.ndarray, pw: int, ph: int) -> np.ndarray:
    n = features.size
    channels = n // (pw * ph)
    img = features.reshape(ph, pw, channels)
    if np.random.rand() < 0.5:
        img = np.fliplr(img)
    if np.random.rand() < 0.5:
        img = np.flipud(img)
    if np.random.rand() < 0.25:
        k = np.random.randint(1, 4)
        img = np.rot90(img, k)
    return img.flatten()


def _iou(x1: float, y1: float, w1: float, h1: float, x2: float, y2: float, w2: float, h2: float) -> float:
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)
    if left < right and top < bottom:
        inter = (right - left) * (bottom - top)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0
    return 0


def _create_classifier(algorithm: str, params: dict[str, Any] | None = None, random_state: int = 42):
    p = dict(params) if params else {}

    if algorithm == "svm":
        kernel = str(p.get("kernel", "rbf"))
        kwargs: dict[str, Any] = {
            "C": float(p.get("C", 1.0)),
            "kernel": kernel,
            "random_state": random_state,
        }
        if kernel in ("rbf", "poly", "sigmoid"):
            kwargs["gamma"] = "scale"
        return SVC(**kwargs, probability=True, max_iter=2000)
    elif algorithm == "logistic":
        return LogisticRegression(
            C=float(p.get("C", 1.0)),
            random_state=random_state,
            max_iter=2000,
            n_jobs=-1,
        )
    elif algorithm in ("neural_network", "cnn"):
        if algorithm == "cnn":
            raw = str(p.get("conv_layers", "512,256,128"))
        else:
            raw = str(p.get("hidden_layers", "100"))
        try:
            hidden = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
        except ValueError:
            hidden = (512, 256, 128) if algorithm == "cnn" else (100,)
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            learning_rate_init=float(p.get("learning_rate", 0.001)),
            max_iter=int(p.get("epochs", 200)),
            batch_size=int(p.get("batch_size", 0)) or "auto",
            random_state=random_state,
        )
    else:
        return RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 100)),
            max_depth=int(p.get("max_depth", 0)) or None,
            random_state=random_state,
            n_jobs=-1,
        )


def _nms_merge(predictions: list[dict[str, Any]], iou_threshold: float = 0.3) -> list[dict[str, Any]]:
    if not predictions:
        return []
    by_img: dict[str, list[dict[str, Any]]] = {}
    for p in predictions:
        by_img.setdefault(p["image_id"], []).append(p)

    merged: list[dict[str, Any]] = []
    for img_id, boxes in by_img.items():
        boxes.sort(key=lambda b: b.get("confidence", 0), reverse=True)
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            used[i] = True
            gx1 = boxes[i]["x"]
            gy1 = boxes[i]["y"]
            gx2 = boxes[i]["x"] + boxes[i]["w"]
            gy2 = boxes[i]["y"] + boxes[i]["h"]
            best_conf = boxes[i].get("confidence", 0)
            has_sliding = boxes[i].get("sliding_window", False)
            for j in range(len(boxes)):
                if used[j]:
                    continue
                bx1, by1 = boxes[j]["x"], boxes[j]["y"]
                bx2 = boxes[j]["x"] + boxes[j]["w"]
                by2 = boxes[j]["y"] + boxes[j]["h"]
                if _iou(gx1, gy1, gx2 - gx1, gy2 - gy1, bx1, by1, bx2 - bx1, by2 - by1) > iou_threshold:
                    used[j] = True
                    gx1 = min(gx1, bx1)
                    gy1 = min(gy1, by1)
                    gx2 = max(gx2, bx2)
                    gy2 = max(gy2, by2)
                    best_conf = max(best_conf, boxes[j].get("confidence", 0))
                    if boxes[j].get("sliding_window"):
                        has_sliding = True
            merged.append({
                "actual": 1,
                "predicted": 1,
                "sliding_window": has_sliding,
                "image_id": img_id,
                "x": gx1,
                "y": gy1,
                "w": gx2 - gx1,
                "h": gy2 - gy1,
                "confidence": best_conf,
            })
    return merged


def _sliding_window(model: Any, img_info: dict, pw: int, ph: int, img_id: str, stride: int = 0) -> list[dict[str, Any]]:
    img_pil = Image.open(img_info["path"]).convert("RGB")
    img_arr = np.array(img_pil, dtype=np.float32) / 255.0
    img_pil.close()
    img_h, img_w, _ = img_arr.shape
    if stride > 0:
        stride_x = max(stride, 2)
        stride_y = max(stride, 2)
    else:
        stride_x = pw
        stride_y = ph
    has_proba = hasattr(model, "predict_proba")

    patches: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []
    for y in range(0, img_h - ph + 1, stride_y):
        for x in range(0, img_w - pw + 1, stride_x):
            patches.append(img_arr[y:y+ph, x:x+pw].flatten())
            coords.append((x, y))

    if not patches:
        return []

    feat_matrix = np.array(patches)
    if has_proba:
        probabilities = model.predict_proba(feat_matrix)
        classes = list(model.classes_)
        pos_idx = classes.index(1) if 1 in classes else (classes.index(True) if True in classes else 1)
        confidences = probabilities[:, pos_idx]
    else:
        predictions = model.predict(feat_matrix)
        confidences = np.where(predictions == 1, 1.0, 0.0)

    detections: list[dict[str, Any]] = []
    for (x, y), confidence in zip(coords, confidences):
        detections.append({
            "actual": 1,
            "predicted": 1,
            "sliding_window": True,
            "image_id": img_id,
            "x": x, "y": y,
            "w": pw, "h": ph,
            "confidence": float(confidence),
        })
    return detections


def _cancellable_fit(model: Any, X_train: np.ndarray, y_train: np.ndarray, cancelled_fn) -> bool:
    if isinstance(model, MLPClassifier):
        max_epochs = model.max_iter
        model.set_params(warm_start=True, max_iter=1)
        for _ in range(max_epochs):
            if cancelled_fn():
                return True
            model.fit(X_train, y_train)
        return False
    else:
        if cancelled_fn():
            return True
        model.fit(X_train, y_train)
        return False


def _train_worker(session_id: str, val_strategy: str = "simple", val_size: float = 0.0, test_size: float = 0.0, algorithm: str = "random_forest", neg_per_ann: int = 10, params: dict[str, Any] | None = None, shuffle: bool = True, random_state: int = 42, k_folds: int = 5, patch_size: int = 0, test_image_ids_param: list[str] | None = None, stride: int = 0) -> None:
    state = _init_state(session_id)
    state["status"] = "running"
    state["log"] = []
    state["error"] = None
    state["_predictions"] = None
    state["test_image_ids"] = []
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    log_fn = lambda msg: _log(state, msg)

    def cancelled() -> bool:
        if state.get("_cancel"):
            log_fn("Training cancelled by user")
            state["status"] = "cancelled"
            return True
        return False

    if cancelled():
        return

    log_fn("Setting random seed...")
    if random_state == -1:
        random_state = None
        np.random.seed()
    else:
        np.random.seed(random_state)

    try:
        log_fn(f"Starting image training (algorithm={algorithm}, neg_per_ann={neg_per_ann}, seed={random_state})...")

        images = state.get("images", {})
        annotations = state.get("annotations", {})

        if not images:
            raise ValueError("No images uploaded")
        if not annotations:
            raise ValueError("No annotations provided")

        log_fn(f"Processing {len(images)} images...")

        total_annotations = 0
        if neg_per_ann == -1:
            neg_per_ann = 10
        neg_per_ann = max(neg_per_ann, 0)
        X_list: list[np.ndarray] = []
        y_list: list[int] = []
        sample_meta: list[dict[str, Any]] = []

        # Auto-compute patch dimensions from annotation sizes
        if patch_size == 0:
            ann_widths, ann_heights = [], []
            for img_anns in annotations.values():
                for ann in img_anns:
                    ann_widths.append(ann["width"])
                    ann_heights.append(ann["height"])
            if ann_widths:
                pw = int(np.median(ann_widths))
                ph = int(np.median(ann_heights))
            else:
                pw = ph = 32
            pw = max(8, min(512, pw))
            ph = max(8, min(512, ph))
            log_fn(f"Auto patch size: {pw}x{ph} (median of annotations)")
        else:
            pw = ph = patch_size

        for img_id, img_info in images.items():
            img_annotations = annotations.get(img_id, [])
            if not img_annotations:
                continue
            total_annotations += len(img_annotations)
            log_fn(f"  {img_info['filename']}: {len(img_annotations)} annotations")

            img = Image.open(img_info["path"])

            for ann in img_annotations:
                patch = _extract_patch(img, ann["x"], ann["y"], ann["width"], ann["height"], pw, ph)
                if patch is not None:
                    X_list.append(patch)
                    y_list.append(1)
                    sample_meta.append({"img_id": img_id, "x": ann["x"], "y": ann["y"], "w": ann["width"], "h": ann["height"]})
                    for _ in range(3):
                        aug = _augment_patch(patch, pw, ph)
                        X_list.append(aug)
                        y_list.append(1)
                        sample_meta.append({"img_id": img_id, "x": ann["x"], "y": ann["y"], "w": ann["width"], "h": ann["height"]})

            img_w, img_h = img.size
            neg_target = len(img_annotations) * neg_per_ann if neg_per_ann > 0 else 0
            attempts = 0
            neg_needed = neg_target
            neg_added = 0
            while neg_added < neg_needed and attempts < neg_needed * 50:
                attempts += 1
                nx = int(np.random.randint(0, max(img_w - pw, 1)))
                ny = int(np.random.randint(0, max(img_h - ph, 1)))

                overlaps = False
                for ann in img_annotations:
                    if _iou(nx, ny, pw, ph, ann["x"], ann["y"], ann["width"], ann["height"]) > 0.3:
                        overlaps = True
                        break

                if not overlaps:
                    patch = img.crop((nx, ny, nx + pw, ny + ph))
                    patch = patch.resize((pw, ph), Image.Resampling.LANCZOS)
                    patch = patch.convert("RGB")
                    arr = np.array(patch, dtype=np.float32) / 255.0
                    X_list.append(arr.flatten())
                    y_list.append(0)
                    sample_meta.append({"img_id": img_id, "x": nx, "y": ny, "w": pw, "h": ph})
                    neg_added += 1

            log_fn(f"    Positive (with aug): {sum(1 for v in y_list if v == 1)}, Negative samples: {neg_added}")
            if cancelled():
                return

        pos_count = sum(1 for v in y_list if v == 1)
        neg_count_total = len(y_list) - pos_count

        if pos_count == 0:
            raise ValueError("No positive patches could be extracted. Make sure annotations are within image bounds.")
        if neg_count_total == 0:
            log_fn("Warning: No negative samples generated. Training with only positive samples. The model may not generalize well.")

        log_fn(f"Total: {pos_count} positive, {neg_count_total} negative samples")
        log_fn(f"Patch dimensions: {pw}x{ph} px, feature vector size: {pw * ph * 3}")

        X = np.array(X_list)
        y = np.array(y_list)

        predictions_list: list[dict[str, Any]] = []
        sliding_list: list[dict[str, Any]] = []
        model: Any = None
        test_img_ids: dict[str, None] = {}  # ordered set (Python 3.7+)
        val_acc: float | None = None
        test_acc: float | None = None

        if val_strategy == "kfold" and len(X) >= k_folds * 2:
            kf = KFold(n_splits=k_folds, shuffle=shuffle, random_state=random_state if shuffle else None)
            log_fn(f"K-fold cross-validation ({k_folds} folds)...")
            fold_metrics: dict[str, list[float]] = {}
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                if cancelled():
                    return
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                y_fold_train, y_fold_val = y[train_idx], y[val_idx]
                fold_model = _create_classifier(algorithm, params=params, random_state=random_state)
                fold_model.fit(X_fold_train, y_fold_train)
                y_fold_pred = fold_model.predict(X_fold_val)
                for i in range(len(y_fold_val)):
                    vi = val_idx[i]
                    meta = sample_meta[vi]
                    test_img_ids[meta["img_id"]] = None
                    predictions_list.append({
                        "image": _feature_to_image(X_fold_val[i], pw, ph),
                        "actual": int(y_fold_val[i]),
                        "predicted": int(y_fold_pred[i]),
                        "image_id": meta["img_id"],
                        "x": meta["x"], "y": meta["y"],
                        "w": meta["w"], "h": meta["h"],
                    })
                fold_acc = float(fold_model.score(X_fold_val, y_fold_val))
                fold_metrics.setdefault("val_accuracy", []).append(fold_acc)
                log_fn(f"  Fold {fold_idx + 1}/{k_folds}: accuracy={fold_acc:.4f}")

            for key, values in fold_metrics.items():
                if key in ("val_accuracy",):
                    log_fn(f"  Average {key}: {np.mean(values):.4f} (±{np.std(values):.4f})")

            # Train final model on all data
            log_fn(f"Training final {algorithm} on full dataset...")
            model = _create_classifier(algorithm, params=params, random_state=random_state)
            model.fit(X, y)
            train_acc = float(model.score(X, y))
            log_fn(f"Training accuracy: {train_acc:.4f}")

            # Sliding window detection on ALL images for gallery display
            for img_id, img_info in state["images"].items():
                if cancelled():
                    return
                sliding_list.extend(_sliding_window(model, img_info, pw, ph, img_id, stride))
                test_img_ids[img_id] = None
            if sliding_list:
                raw_predictions = list(sliding_list)
                sliding_list[:] = _nms_merge(sliding_list)
                log_fn(f"Sliding window detection on all {len(state['images'])} images (k-fold), NMS merged {len(raw_predictions)} -> {len(sliding_list)} boxes")

            algo_label = IMAGE_ALGORITHMS.get(algorithm, {}).get("label", algorithm)
            metrics: dict[str, Any] = {
                "algorithm": algo_label,
                "accuracy": round(train_acc, 4),
                "val_accuracy": round(float(np.mean(fold_metrics.get("val_accuracy", [0]))), 4),
                "val_strategy": "kfold",
                "k_folds": k_folds,
                "samples": len(X),
                "positive_samples": pos_count,
                "negative_samples": neg_count_total,
                "annotations": total_annotations,
            }
        else:
            if val_strategy == "kfold":
                log_fn(f"Warning: kfold needs at least {k_folds * 2} samples, got {len(X)}. Falling back to simple split.")

            # Simple split: separate by image, not by patch
            all_img_ids = list(state["images"].keys())
            n_total = len(all_img_ids)

            if test_image_ids_param:
                # Locked images → test set (overrides test_pct)
                test_img_set = set(test_image_ids_param)
                train_img_ids = [iid for iid in all_img_ids if iid not in test_img_set]
                if not train_img_ids:
                    raise ValueError("No training images found. All images are marked as test.")

                rng = np.random.RandomState(random_state)
                train_img_arr = np.array(train_img_ids)
                rng.shuffle(train_img_arr)
                n_t = len(train_img_arr)

                n_val = min(int(n_total * val_size + 0.5), n_t)
                n_train = n_t - n_val

                val_img_set = set(train_img_arr[:n_val]) if n_val else set()
                train_img_set = set(train_img_arr[n_val:]) if n_val else set(train_img_arr)

            else:
                # No locked images — split all images by percentages
                rng = np.random.RandomState(random_state)
                all_arr = np.array(all_img_ids)
                rng.shuffle(all_arr)

                # round (half-up) for test and val (closest integer), train absorbs leftovers
                n_test = int(n_total * test_size + 0.5)
                n_val  = int(n_total * val_size + 0.5)
                # Guard overflow: reduce the one with smaller pct (right-to-left: test > val)
                if n_test + n_val > n_total:
                    if val_size <= test_size:
                        n_val = n_total - n_test
                    else:
                        n_test = n_total - n_val
                n_train = n_total - n_test - n_val

                test_img_set = set(all_arr[:n_test]) if n_test else set()
                val_img_set = set(all_arr[n_test:n_test + n_val]) if n_val else set()
                train_img_set = set(all_arr[n_test + n_val:]) if n_test + n_val < n_total else set()

            # Build indices and arrays
            train_idx = [i for i, m in enumerate(sample_meta) if m["img_id"] in train_img_set]
            val_idx = [i for i, m in enumerate(sample_meta) if m["img_id"] in val_img_set]
            test_idx = [i for i, m in enumerate(sample_meta) if m["img_id"] in test_img_set]

            X_train = X[train_idx] if len(train_idx) else np.empty((0, X.shape[1]))
            y_train = y[train_idx] if len(train_idx) else np.empty(0)
            X_val = X[val_idx] if len(val_idx) else None
            y_val = y[val_idx] if len(val_idx) else None
            X_test = X[test_idx] if len(test_idx) else None
            y_test = y[test_idx] if len(test_idx) else None

            if test_image_ids_param and not test_idx:
                log_fn("No pre-extracted test patches (test images have no annotations). Running sliding window detection.")

            ni = len(train_img_set)
            nv = len(val_img_set)
            nt = len(test_img_set)
            log_fn(f"Train: {ni} images ({len(train_idx)} patches), "
                   f"Val: {nv} images ({len(val_idx)} patches), "
                   f"Test: {nt} images ({len(test_idx)} patches)")

            log_fn(f"Training {algorithm} classifier...")
            model = _create_classifier(algorithm, params=params, random_state=random_state)
            if _cancellable_fit(model, X_train, y_train, cancelled):
                return

            train_acc = float(model.score(X_train, y_train))
            log_fn(f"Training accuracy: {train_acc:.4f}")

            if X_val is not None and len(X_val) > 0:
                y_val_pred = model.predict(X_val)
                val_acc = float(model.score(X_val, y_val))
                log_fn(f"Validation accuracy: {val_acc:.4f}")

            # Test evaluation: sliding window detection on held-out test images
            if test_image_ids_param:
                test_img_ids.clear()
                for img_idx, img_id in enumerate(test_image_ids_param):
                    if cancelled():
                        return
                    img_info = state["images"].get(img_id)
                    if not img_info:
                        continue
                    log_fn(f"  Scanning {img_info['filename']} ({img_idx+1}/{len(test_image_ids_param)})...")
                    dets = _sliding_window(model, img_info, pw, ph, img_id, stride)
                    sliding_list.extend(dets)
                    test_img_ids[img_id] = None
                    log_fn(f"    {img_info['filename']}: {len(dets)} raw detections")
                raw_predictions = list(sliding_list)
                before = len(sliding_list)
                sliding_list[:] = _nms_merge(sliding_list)
                after = len(sliding_list)
                if after < before:
                    log_fn(f"  NMS merged {before} raw detections into {after} merged boxes")
                log_fn(f"Sliding window detection on {len(test_image_ids_param)} test images complete")
            elif test_idx:
                y_test_pred = model.predict(X_test)
                test_acc = float(model.score(X_test, y_test))
                log_fn(f"Test accuracy: {test_acc:.4f}")
                for i, idx in enumerate(test_idx):
                    meta = sample_meta[idx]
                    test_img_ids[meta["img_id"]] = None
                    predictions_list.append({
                        "image": _feature_to_image(X_test[i], pw, ph),
                        "actual": int(y_test[i]),
                        "predicted": int(y_test_pred[i]),
                        "image_id": meta["img_id"],
                        "x": meta["x"], "y": meta["y"],
                        "w": meta["w"], "h": meta["h"],
                    })

            algo_label = IMAGE_ALGORITHMS.get(algorithm, {}).get("label", algorithm)
            metrics = {
                "algorithm": algo_label,
                "accuracy": round(train_acc, 4),
                "samples": len(X),
                "positive_samples": pos_count,
                "negative_samples": neg_count_total,
                "annotations": total_annotations,
            }
            if val_acc is not None:
                metrics["val_accuracy"] = round(val_acc, 4)
            if test_acc is not None:
                metrics["test_accuracy"] = round(test_acc, 4)

        state["metrics"] = metrics
        state["_predictions"] = predictions_list
        # Combined raw predictions for tuning gallery (sliding window + fold/test patches)
        state["_raw_predictions"] = predictions_list + sliding_list
        if val_strategy == "kfold":
            for img_id in state["images"]:
                test_img_ids[img_id] = None
        state["test_image_ids"] = list(test_img_ids.keys())

        if cancelled():
            return
        IMGTRAIN_DIR.mkdir(parents=True, exist_ok=True)
        model_path = IMGTRAIN_DIR / f"{session_id}.model"
        joblib.dump({
            "model": model,
            "metadata": {
                "patch_size": pw,
                "patch_width": pw,
                "patch_height": ph,
                "type": "image_detector",
                "algorithm": algorithm,
                "metrics": metrics,
            }
        }, model_path)
        state["model_path"] = str(model_path)

        log_fn(f"Model saved to {model_path.name}")
        state["status"] = "done"

    except Exception as e:
        state["status"] = "error"
        state["error"] = str(e)
        _log(state, f"ERROR: {e}")


def save_to_server(session_id: str, name: str) -> str:
    state = _init_state(session_id)
    if state["status"] != "done":
        raise ValueError("No trained model available")

    model_path = state.get("model_path")
    if not model_path or not Path(model_path).exists():
        raise ValueError("Model file not found")

    name = name.strip()
    if not name:
        name = "image_model"
    if not name.endswith(".model"):
        name += ".model"

    dest = MODELS_IMAGE_DIR / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(Path(model_path).read_bytes())

    model_data = joblib.load(model_path)
    meta = model_data.get("metadata", {}) if isinstance(model_data, dict) else {}
    saved_pw = meta.get("patch_width") or meta.get("patch_size", 32)
    saved_ph = meta.get("patch_height") or meta.get("patch_size", 32)

    index_path = MODELS_IMAGE_DIR / "_metrics.json"
    index: dict = {}
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    metrics = state.get("metrics") or {}
    index[name] = {
        "type": "image_detector",
        "patch_size": saved_pw,
        "patch_width": saved_pw,
        "patch_height": saved_ph,
        "task": "image_detection",
        "algorithm": metrics.get("algorithm", "random_forest"),
        "accuracy": metrics.get("accuracy"),
        "val_accuracy": metrics.get("val_accuracy"),
        "samples": metrics.get("samples"),
    }
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return str(dest)
