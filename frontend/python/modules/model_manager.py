from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import MODELS_DIR, MODELS_CSV_DIR, MODELS_IMAGE_DIR


def _ensure_subdir(subdir: str) -> Path:
    root = MODELS_DIR / subdir if subdir else MODELS_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _subdir_for_type(model_type: str) -> str:
    if model_type == "image_detector":
        return "image"
    if model_type == "csv" or model_type == "unknown":
        return "csv"
    return model_type


def _safe_model_path(name: str, subdir: str = "") -> Path:
    root = _ensure_subdir(subdir).resolve()
    candidate = (root / name).resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError("Invalid model name")
    return candidate


def list_models(subdir: str = "") -> list[dict[str, Any]]:
    root = _ensure_subdir(subdir)
    items: list[dict[str, Any]] = []
    index: dict[str, Any] = {}
    index_path = root / "_metrics.json"
    if index_path.exists():
        try:
            import json as _json
            index = _json.loads(index_path.read_text())
        except Exception:
            index = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.name not in (".gitkeep", "_metrics.json"):
            stat = path.stat()
            meta = index.get(path.name, {})
            items.append({
                "name": path.name,
                "stem": path.stem,
                "suffix": path.suffix,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "type": meta.get("type", "unknown"),
            })
    return items


def list_all_models() -> list[dict[str, Any]]:
    csv_items = list_models("csv")
    image_items = list_models("image")
    csv_items.extend(image_items)
    return csv_items


def _remove_metrics_entry(name: str, subdir: str = "") -> None:
    try:
        import json as _json
        index_path = _ensure_subdir(subdir) / "_metrics.json"
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


def delete_model(name: str, subdir: str = "") -> dict[str, str]:
    target = _safe_model_path(name, subdir)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(name)
    target.unlink()
    _remove_metrics_entry(name, subdir)
    return {"deleted": name}


def get_model_metrics(name: str, subdir: str = "") -> dict[str, Any] | None:
    try:
        import json as _json
        index_path = _ensure_subdir(subdir) / "_metrics.json"
        if not index_path.exists():
            return None
        with open(index_path) as f:
            index = _json.load(f)
        return index.get(name)
    except Exception:
        return None


def get_model_metrics_anywhere(name: str) -> dict[str, Any] | None:
    for subdir in ("csv", "image"):
        result = get_model_metrics(name, subdir)
        if result:
            return result
    return None


def rename_model(old_name: str, new_name: str, subdir: str = "") -> dict[str, str]:
    old_path = _safe_model_path(old_name, subdir)
    if not old_path.exists() or not old_path.is_file():
        raise FileNotFoundError(old_name)
    new_path = _safe_model_path(new_name, subdir)
    if new_path.exists():
        raise FileExistsError(new_name)
    old_path.rename(new_path)

    try:
        import json as _json
        index_path = _ensure_subdir(subdir) / "_metrics.json"
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


def get_model_path(name: str, subdir: str = "") -> Path:
    target = _safe_model_path(name, subdir)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(name)
    return target
