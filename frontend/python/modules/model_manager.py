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
        if path.is_file() and path.name not in (".gitkeep", "_metrics.json"):
            stat = path.stat()
            items.append({
                "name": path.name,
                "stem": path.stem,
                "suffix": path.suffix,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
    return items


def _remove_metrics_entry(name: str) -> None:
    try:
        import json as _json
        index_path = _ensure_models_dir() / "_metrics.json"
        if not index_path.exists():
            return
        with open(index_path) as f:
            index = _json.load(f)
        if name in index:
            del index[name]
            with open(index_path, "w") as f:
                _json.dump(index, f, indent=2)
    except Exception:
        pass


def delete_model(name: str) -> dict[str, str]:
    target = _safe_model_path(name)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(name)
    target.unlink()
    _remove_metrics_entry(name)
    return {"deleted": name}


def get_model_metrics(name: str) -> dict[str, Any] | None:
    try:
        import json as _json
        index_path = _ensure_models_dir() / "_metrics.json"
        if not index_path.exists():
            return None
        with open(index_path) as f:
            index = _json.load(f)
        return index.get(name)
    except Exception:
        return None


def rename_model(old_name: str, new_name: str) -> dict[str, str]:
    old_path = _safe_model_path(old_name)
    if not old_path.exists() or not old_path.is_file():
        raise FileNotFoundError(old_name)
    new_path = _safe_model_path(new_name)
    if new_path.exists():
        raise FileExistsError(new_name)
    old_path.rename(new_path)

    try:
        import json as _json
        index_path = _ensure_models_dir() / "_metrics.json"
        if index_path.exists():
            with open(index_path) as f:
                index = _json.load(f)
            if old_name in index:
                index[new_name] = index.pop(old_name)
                with open(index_path, "w") as f:
                    _json.dump(index, f, indent=2)
    except Exception:
        pass

    return {"renamed": old_name, "to": new_name}


def get_model_path(name: str) -> Path:
    target = _safe_model_path(name)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(name)
    return target
