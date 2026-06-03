from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import MODELS_DIR

def _ensure_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR

def _safe_model_path(name: str) -> Path:
    candidate = (_ensure_models_dir() / name).resolve()
    base = _ensure_models_dir().resolve()
    if base not in candidate.parents and candidate != base:
        raise ValueError("Invalid model name")
    return candidate

def list_models() -> list[dict[str, Any]]:
    root = _ensure_models_dir()
    items: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.name != ".gitkeep":
            stat = path.stat()
            items.append({
                "name": path.name,
                "stem": path.stem,
                "suffix": path.suffix,
                "size": stat.st_size,
                "modified": stat.st_mtime
            })
    return items

def rename_model(old_name: str, new_name: str) -> dict[str, str]:
    source = _safe_model_path(old_name)
    target = _safe_model_path(new_name)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(old_name)
    if target.exists():
        raise FileExistsError(new_name)
    source.rename(target)
    return {"old_name": old_name, "new_name": new_name}

def delete_model(name: str) -> dict[str, str]:
    target = _safe_model_path(name)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(name)
    target.unlink()
    return {"deleted": name}