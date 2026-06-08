from __future__ import annotations

import base64
import csv
import io
import os
import shutil
from pathlib import Path
from typing import Any

from core.config import TEMP_DIR

_uploaded_csv: dict[str, Any] | None = None


def _clean_temp() -> None:
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def upload_csv(filename: str, b64_data: str) -> dict[str, Any]:
    global _uploaded_csv
    _clean_temp()

    raw = base64.b64decode(b64_data)
    ext = Path(filename).suffix.lower()
    if ext != ".csv":
        raise ValueError("Only CSV files are supported")

    dest = TEMP_DIR / filename
    dest.write_bytes(raw)
    return _parse_csv(dest)


def _parse_csv(path: Path) -> dict[str, Any]:
    global _uploaded_csv
    text = path.read_text(encoding="utf-8-sig")
    all_lines = [l for l in text.splitlines() if l.strip()]
    total_lines = max(0, len(all_lines) - 1)
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, str]] = []
    columns: list[str] = []

    for i, row in enumerate(reader):
        if i == 0:
            columns = list(row.keys())
        rows.append(row)
        if i >= 49:
            break

    preview_rows: list[list[str]] = []
    for row in rows:
        preview_rows.append([row.get(c, "") for c in columns])

    result: dict[str, Any] = {
        "filename": path.name,
        "columns": columns,
        "rows": len(rows),
        "total_lines": total_lines,
        "preview": preview_rows,
        "path": str(path),
    }
    col_types: dict[str, str] = {}
    for col in columns:
        vals = [row[col] for row in rows if row[col]]
        if _is_numeric_col(vals):
            unique_vals = len(set(vals))
            if unique_vals < 10:
                col_types[col] = "categorical_int"
            else:
                col_types[col] = "numeric"
        else:
            col_types[col] = "text"
    result["col_types"] = col_types
    _uploaded_csv = result
    return result


def _is_numeric_col(vals: list[str]) -> bool:
    if not vals:
        return False
    num_count = 0
    for v in vals:
        try:
            float(v)
            num_count += 1
        except ValueError:
            pass
    return num_count > len(vals) * 0.8


def get_uploaded_csv() -> dict[str, Any] | None:
    return _uploaded_csv


def get_csv_path() -> str | None:
    if _uploaded_csv is None:
        return None
    return _uploaded_csv.get("path")


def clear_uploaded() -> None:
    global _uploaded_csv
    _uploaded_csv = None
    _clean_temp()
