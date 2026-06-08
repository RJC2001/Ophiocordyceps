from __future__ import annotations

from typing import Any

from core.config import FORM_CONFIG_PATH, load_json, save_json

DEFAULT_FORM = {
    "test_size": 0.2,
    "algorithm": "random_forest",
}


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
    save_json(FORM_CONFIG_PATH, merged)
    return merged
