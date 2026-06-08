from __future__ import annotations

from typing import Any

_log: list[str] = []
MAX_LINES = 500


def get_output() -> dict[str, Any]:
    return {"lines": list(_log)}


def append_output(line: str) -> dict[str, Any]:
    _log.append(str(line).rstrip("\n"))
    if len(_log) > MAX_LINES:
        _log[: len(_log) - MAX_LINES] = []
    return get_output()


def clear_output() -> dict[str, Any]:
    _log.clear()
    return get_output()
