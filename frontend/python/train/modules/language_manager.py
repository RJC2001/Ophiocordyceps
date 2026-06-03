from __future__ import annotations

from typing import Any

from core.config import APP_CONFIG_PATH, load_json, save_json

DEFAULT_LANGUAGE = "en"

LANGUAGES = {
    "en": "English",
    "pl": "Polski",
    "de": "Deutsch",
    "es": "Español",
    "fr": "Français",
    "it": "Italiano",
}

def get_app_config() -> dict[str, Any]:
    config = load_json(APP_CONFIG_PATH, {})
    if not isinstance(config, dict):
        return {}
    return config

def save_app_config(config: dict[str, Any]) -> None:
    save_json(APP_CONFIG_PATH, config)

def get_current_language() -> str:
    config = get_app_config()
    language = str(config.get("language", DEFAULT_LANGUAGE)).strip().lower()
    return language if language in LANGUAGES else DEFAULT_LANGUAGE

def set_current_language(language: str) -> str:
    normalized = str(language).strip().lower()
    if normalized not in LANGUAGES:
        raise ValueError("Unknown language")
    config = get_app_config()
    config["language"] = normalized
    save_app_config(config)
    return normalized

def get_available_languages() -> list[dict[str, str]]:
    return [{"key": key, "label": label} for key, label in LANGUAGES.items()]