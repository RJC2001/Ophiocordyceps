from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
RES_DIR = ROOT_DIR / "res"
DB_DIR = ROOT_DIR / "db"

LAYOUT_DIR = RES_DIR / "layout" / "train"
TEST_LAYOUT_DIR = RES_DIR / "layout" / "test"
STATIC_DIR = RES_DIR / "static"
CSS_DIR = STATIC_DIR / "css"
THEME_DIR = CSS_DIR / "themes"
I18N_DIR = STATIC_DIR / "i18n"

CONFIG_DIR = STATIC_DIR / "config"
APP_CONFIG_PATH = CONFIG_DIR / "train.json"

MODELS_DIR = DB_DIR / "models"
MODELS_CSV_DIR = MODELS_DIR / "csv"
MODELS_IMAGE_DIR = MODELS_DIR / "image"
TEMP_DIR = DB_DIR / "temp"
IMGTRAIN_DIR = DB_DIR / "temp_imgtrain"
MODEL_TEMP_DIR = DB_DIR / "temp_models"


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
