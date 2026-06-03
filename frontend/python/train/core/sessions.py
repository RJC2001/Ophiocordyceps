from __future__ import annotations

import uuid
import threading
from typing import Any

from core.config import CONFIG_DIR, load_json, save_json

SESSIONS: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
COOKIE_NAME = "ophio_session"
SESSIONS_PATH = CONFIG_DIR / "sessions.json"


def _load() -> None:
    data = load_json(SESSIONS_PATH, {})
    if isinstance(data, dict):
        SESSIONS.clear()
        for sid, config in data.items():
            if isinstance(config, dict):
                SESSIONS[sid] = dict(config)


def _save() -> None:
    save_json(SESSIONS_PATH, SESSIONS)


def create_id() -> str:
    return str(uuid.uuid4())


def get(session_id: str) -> dict[str, Any] | None:
    with _lock:
        return SESSIONS.get(session_id)


def get_or_create(session_id: str | None) -> tuple[str, dict[str, Any]]:
    if session_id:
        with _lock:
            config = SESSIONS.get(session_id)
            if config is not None:
                return session_id, config
    sid = create_id()
    with _lock:
        SESSIONS[sid] = {}
    _save()
    return sid, {}


def set_theme(session_id: str, theme: str) -> None:
    with _lock:
        config = SESSIONS.get(session_id)
        if config is not None:
            config["theme"] = theme
    _save()


def set_language(session_id: str, language: str) -> None:
    with _lock:
        config = SESSIONS.get(session_id)
        if config is not None:
            config["language"] = language
    _save()


_load()
