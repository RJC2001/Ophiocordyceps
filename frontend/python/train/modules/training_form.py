from __future__ import annotations

from typing import Any

from core.config import FORM_CONFIG_PATH, load_json, save_json

DEFAULT_FORM = {
    "dataset": "",
    "model_name": "",
    "epochs": 25,
    "batch_size": 16,
    "learning_rate": 0.0003
}

def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback

def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback

def get_training_form() -> dict[str, Any]:
    data = load_json(FORM_CONFIG_PATH, DEFAULT_FORM)
    if not isinstance(data, dict):
        return DEFAULT_FORM.copy()
    merged = DEFAULT_FORM.copy()
    merged.update(data)
    return merged

def save_training_form(payload: dict[str, Any]) -> dict[str, Any]:
    merged = DEFAULT_FORM.copy()
    merged.update(payload)
    merged["dataset"] = str(merged.get("dataset", "")).strip()
    merged["model_name"] = str(merged.get("model_name", "")).strip()
    merged["epochs"] = _as_int(merged.get("epochs"), DEFAULT_FORM["epochs"])
    merged["batch_size"] = _as_int(merged.get("batch_size"), DEFAULT_FORM["batch_size"])
    merged["learning_rate"] = _as_float(merged.get("learning_rate"), DEFAULT_FORM["learning_rate"])
    save_json(FORM_CONFIG_PATH, merged)
    return merged