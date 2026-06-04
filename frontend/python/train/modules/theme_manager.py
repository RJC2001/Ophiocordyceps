from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import APP_CONFIG_PATH, CSS_DIR, THEME_DIR, load_json, save_json

DEFAULT_THEME = "light"

THEME_FILES = {
    "light": THEME_DIR / "light.css",
    "dark": THEME_DIR / "dark.css",
    "monochrome": THEME_DIR / "monochrome.css",
    "forest": THEME_DIR / "forest.css",
    "ocean": THEME_DIR / "ocean.css",
    "sunset": THEME_DIR / "sunset.css",
    "steampunk": THEME_DIR / "steampunk.css",
    "cyberpunk": THEME_DIR / "cyberpunk.css",
}

def get_app_config() -> dict[str, Any]:
    config = load_json(APP_CONFIG_PATH, {})
    if not isinstance(config, dict):
        return {}
    return config

def save_app_config(config: dict[str, Any]) -> None:
    save_json(APP_CONFIG_PATH, config)

def get_current_theme() -> str:
    config = get_app_config()
    theme = str(config.get("theme", DEFAULT_THEME)).strip().lower()
    return theme if theme in THEME_FILES else DEFAULT_THEME

def validate_theme(theme: str) -> str:
    normalized = str(theme).strip().lower()
    if normalized not in THEME_FILES:
        raise ValueError("Unknown theme")
    return normalized

def set_current_theme(theme: str) -> str:
    normalized = validate_theme(theme)
    config = get_app_config()
    config["theme"] = normalized
    save_app_config(config)
    return normalized

def resolve_theme(session_config: dict | None = None) -> str:
    if session_config and "theme" in session_config:
        theme = session_config["theme"]
        if theme in THEME_FILES:
            return theme
    return get_current_theme()

def get_available_themes() -> list[dict[str, str]]:
    return [{"key": key, "label": key.capitalize()} for key in THEME_FILES]

def get_theme_css_path(theme: str | None = None) -> Path:
    chosen = get_current_theme() if theme is None else str(theme).strip().lower()
    return THEME_FILES.get(chosen, THEME_FILES[DEFAULT_THEME])

def read_theme_css(theme: str | None = None) -> str:
    base_css = (CSS_DIR / "base.css").read_text(encoding="utf-8")
    theme_css = get_theme_css_path(theme).read_text(encoding="utf-8")
    return f"{base_css}\n\n{theme_css}"