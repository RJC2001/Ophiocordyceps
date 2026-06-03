from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import TRAIN_LOG_PATH, ensure_parent

MAX_LINES = 500

def _ensure_log_file() -> Path:
    ensure_parent(TRAIN_LOG_PATH)
    if not TRAIN_LOG_PATH.exists():
        TRAIN_LOG_PATH.write_text("", encoding="utf-8")
    return TRAIN_LOG_PATH

def _read_lines() -> list[str]:
    path = _ensure_log_file()
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[-MAX_LINES:]

def get_training_output() -> dict[str, Any]:
    path = _ensure_log_file()
    stat = path.stat()
    return {
        "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "lines": _read_lines()
    }

def append_training_output(line: str) -> dict[str, Any]:
    path = _ensure_log_file()
    text = str(line).rstrip("\n")
    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")
    return get_training_output()

def clear_training_output() -> dict[str, Any]:
    path = _ensure_log_file()
    path.write_text("", encoding="utf-8")
    return get_training_output()