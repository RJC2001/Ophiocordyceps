from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[4]
FRONTEND_DIR = ROOT_DIR / "frontend"
BACKEND_DIR = ROOT_DIR / "backend"
DB_DIR = ROOT_DIR / "db"

LAYOUT_DIR = FRONTEND_DIR / "res" / "layout" / "train"
STATIC_DIR = FRONTEND_DIR / "res" / "static"
CSS_DIR = STATIC_DIR / "css"
THEME_DIR = CSS_DIR / "themes"
I18N_DIR = STATIC_DIR / "i18n"

CONFIG_DIR = STATIC_DIR / "config"
APP_CONFIG_PATH = CONFIG_DIR / "train.json"
FORM_CONFIG_PATH = CONFIG_DIR / "train_form.json"

RUNTIME_DIR = DB_DIR / "runtime"
TRAIN_LOG_PATH = RUNTIME_DIR / "train.log"

MODELS_DIR = DB_DIR / "models"
DATASETS_DIR = DB_DIR / "datasets"

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default

def save_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")