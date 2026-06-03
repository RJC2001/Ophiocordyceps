from __future__ import annotations

import uuid
import threading
from typing import Any

SESSIONS: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
COOKIE_NAME = "ophio_session"


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
    return sid, {}


def set_theme(session_id: str, theme: str) -> None:
    with _lock:
        config = SESSIONS.get(session_id)
        if config is not None:
            config["theme"] = theme


def set_language(session_id: str, language: str) -> None:
    with _lock:
        config = SESSIONS.get(session_id)
        if config is not None:
            config["language"] = language
