from __future__ import annotations

import json
import base64
from typing import Any

COOKIE_NAME = "ophio_session"

def _encode_cookie(data: dict[str, Any]) -> str:
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

def _decode_cookie(value: str) -> dict[str, Any] | None:
    try:
        padding = "=" * (-len(value) % 4)
        raw = base64.urlsafe_b64decode((value + padding).encode("ascii"))
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None

def create_id() -> str:
    return _encode_cookie({})

def get(session_id: str) -> dict[str, Any] | None:
    if not session_id:
        return None
    return _decode_cookie(session_id)

def get_or_create(session_id: str | None) -> tuple[str, dict[str, Any]]:
    if session_id:
        config = _decode_cookie(session_id)
        if isinstance(config, dict):
            return session_id, config
    sid = create_id()
    return sid, {}

def set_theme(session_id: str, theme: str) -> str:
    config = _decode_cookie(session_id) or {}
    config["theme"] = theme
    return _encode_cookie(config)

def set_language(session_id: str, language: str) -> str:
    config = _decode_cookie(session_id) or {}
    config["language"] = language
    return _encode_cookie(config)