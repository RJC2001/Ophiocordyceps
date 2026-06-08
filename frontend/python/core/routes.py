from __future__ import annotations

import base64
import datetime
import json
import mimetypes
import re
import uuid

from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd

from core.config import LAYOUT_DIR, TEST_LAYOUT_DIR, FRONTEND_DIR, STATIC_DIR, TEMP_DIR, MODEL_TEMP_DIR, MODELS_DIR

from core.sessions import (
    COOKIE_NAME,
    get_or_create as session_get_or_create,
    set_theme as session_set_theme,
    set_language as session_set_language,
)

from modules.theme_manager import (
    get_available_themes,
    read_theme_css,
    resolve_theme,
    validate_theme,
)

from modules.language_manager import (
    get_available_languages,
    resolve_language,
    validate_language,
)

from train.modules.training_form import (
    get_training_form,
    save_training_form,
)

from train.modules.training_output import (
    get_output,
    append_output,
    clear_output,
)

from modules.model_manager import (
    list_models,
    delete_model,
    rename_model,
    get_model_metrics,
    get_model_path,
)

from modules.csv_handler import (
    upload_csv,
    get_uploaded_csv,
    get_csv_path,
    clear_uploaded,
    clean_model_temp,
)

from train.modules.trainer import (
    get_available_algorithms,
    start_training,
    get_training_status,
    save_to_server,
)

from testing.modules.predictor import (
    load_model,
    predict_csv,
)

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
    ".model": "application/octet-stream",
    ".pkl": "application/octet-stream",
}


def _set_session_cookie(
    handler: BaseHTTPRequestHandler,
    session_id: str,
) -> None:
    expires = (
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).strftime("%a, %d %b %Y %H:%M:%S GMT")
    handler.send_header(
        "Set-Cookie",
        f"{COOKIE_NAME}={session_id}; Path=/; HttpOnly; SameSite=Lax; Expires={expires}",
    )


def json_bytes(data: object) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def read_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    if not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        return {}
    return data


def send_json(
    handler: BaseHTTPRequestHandler,
    status: int,
    data: object,
    session_id: str | None = None,
) -> None:
    payload = json_bytes(data)
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    if session_id:
        _set_session_cookie(handler, session_id)
    handler.end_headers()
    handler.wfile.write(payload)


def send_text(
    handler: BaseHTTPRequestHandler,
    status: int,
    text: str,
    content_type: str,
    session_id: str | None = None,
) -> None:
    payload = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    if session_id:
        _set_session_cookie(handler, session_id)
    handler.end_headers()
    handler.wfile.write(payload)


def send_file(
    handler: BaseHTTPRequestHandler,
    file_path: Path,
    session_id: str | None = None,
) -> None:
    data = file_path.read_bytes()
    content_type = (
        CONTENT_TYPES.get(file_path.suffix.lower())
        or mimetypes.guess_type(str(file_path))[0]
        or "application/octet-stream"
    )
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header(
        "Content-Disposition",
        f'attachment; filename="{file_path.name}"',
    )
    if session_id:
        _set_session_cookie(handler, session_id)
    handler.end_headers()
    handler.wfile.write(data)


def safe_static_path(url_path: str) -> Path | None:
    relative = url_path.removeprefix("/static/").lstrip("/")
    candidate = (STATIC_DIR / relative).resolve()
    base = STATIC_DIR.resolve()
    if base not in candidate.parents and candidate != base:
        return None
    if not candidate.exists():
        return None
    if not candidate.is_file():
        return None
    return candidate


class TrainHandler(BaseHTTPRequestHandler):
    server_version = "OphiocordycepsTrainHTTP/3.0"

    def _resolve_session(self) -> tuple[str, dict]:
        cookie_header = self.headers.get("Cookie", "")
        cookies = SimpleCookie(cookie_header)
        existing = cookies.get(COOKIE_NAME)
        sid, config = session_get_or_create(existing.value if existing else None)
        self._session_id = sid
        self._session_config = config
        return sid, config

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        sid, session_config = self._resolve_session()

        if path in ("/", "/index.html"):
            clear_uploaded(sid)
            clean_model_temp(sid)
            html = (FRONTEND_DIR / "res" / "layout" / "index.html").read_text(encoding="utf-8")
            send_text(self, HTTPStatus.OK, html, "text/html; charset=utf-8", session_id=sid)
            return

        train_paths = ("/train", "/train/", "/train/index.html")
        if path in train_paths:
            clear_uploaded(sid)
            clean_model_temp(sid)
            html = (LAYOUT_DIR / "index.html").read_text(encoding="utf-8")
            send_text(self, HTTPStatus.OK, html, "text/html; charset=utf-8", session_id=sid)
            return

        test_paths = ("/test", "/test/", "/test/index.html")
        if path in test_paths:
            clear_uploaded(sid)
            clean_model_temp(sid)
            html = (TEST_LAYOUT_DIR / "index.html").read_text(encoding="utf-8")
            send_text(self, HTTPStatus.OK, html, "text/html; charset=utf-8", session_id=sid)
            return

        if path == "/api/theme":
            send_json(
                self,
                HTTPStatus.OK,
                {"theme": resolve_theme(session_config), "available": get_available_themes()},
                session_id=sid,
            )
            return

        if path == "/api/language":
            send_json(
                self,
                HTTPStatus.OK,
                {"language": resolve_language(session_config), "available": get_available_languages()},
                session_id=sid,
            )
            return

        if path == "/api/form":
            send_json(self, HTTPStatus.OK, get_training_form(), session_id=sid)
            return

        if path == "/api/output":
            send_json(self, HTTPStatus.OK, get_output(), session_id=sid)
            return

        if path == "/api/dataset":
            data = get_uploaded_csv(sid)
            if data is None:
                send_json(self, HTTPStatus.OK, {"uploaded": False}, session_id=sid)
            else:
                send_json(self, HTTPStatus.OK, {"uploaded": True, **data}, session_id=sid)
            return

        if path == "/api/models":
            send_json(self, HTTPStatus.OK, list_models(), session_id=sid)
            return

        if path == "/api/algorithms":
            send_json(self, HTTPStatus.OK, get_available_algorithms(), session_id=sid)
            return

        if path == "/api/train/status":
            status = get_training_status(sid)
            send_json(self, HTTPStatus.OK, status, session_id=sid)
            return

        # Model download from server db/
        model_dl_match = re.match(r"^/api/models/download/(.+)$", path)
        if model_dl_match:
            try:
                model_name = model_dl_match.group(1)
                model_path = get_model_path(model_name)
                send_file(self, model_path, session_id=sid)
            except (FileNotFoundError, ValueError):
                self.send_error(HTTPStatus.NOT_FOUND, "Model not found")
            return

        # Trained model download (temp)
        if path.startswith("/api/train/download/"):
            token = path.removeprefix("/api/train/download/")
            if not token:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            from train.modules.trainer import training_states
            state = training_states.get(sid)
            if state is None or state["status"] != "done":
                self.send_error(HTTPStatus.BAD_REQUEST, "No trained model")
                return
            model_path_str = state.get("model_path")
            if not model_path_str:
                self.send_error(HTTPStatus.NOT_FOUND, "Model file not found")
                return
            mp = Path(model_path_str)
            if not mp.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Model file not found")
                return
            # Use .model extension for download
            csv_path = state.get("_csv_path")
            dl_name = (Path(csv_path).stem if csv_path else "model") + ".model"
            data = mp.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition", f'attachment; filename="{dl_name}"')
            _set_session_cookie(self, sid)
            self.end_headers()
            self.wfile.write(data)
            return

        # Serve static files
        if path == "/static/css/current.css":
            send_text(
                self,
                HTTPStatus.OK,
                read_theme_css(resolve_theme(session_config)),
                "text/css; charset=utf-8",
                session_id=sid,
            )
            return

        if path.startswith("/static/"):
            file_path = safe_static_path(path)
            if file_path is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            content_type = (
                CONTENT_TYPES.get(file_path.suffix.lower())
                or mimetypes.guess_type(str(file_path))[0]
                or "application/octet-stream"
            )
            data = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            _set_session_cookie(self, sid)
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        sid, session_config = self._resolve_session()

        try:
            payload = read_body(self)

            if path == "/api/theme":
                theme = validate_theme(payload.get("theme", ""))
                new_sid = session_set_theme(sid, theme)
                send_json(self, HTTPStatus.OK, {"theme": theme}, session_id=new_sid)
                return

            if path == "/api/language":
                language = validate_language(payload.get("language", ""))
                new_sid = session_set_language(sid, language)
                send_json(self, HTTPStatus.OK, {"language": language}, session_id=new_sid)
                return

            if path == "/api/form":
                form = save_training_form(payload)
                send_json(self, HTTPStatus.OK, form, session_id=sid)
                return

            if path == "/api/output/append":
                line = str(payload.get("line", "")).strip()
                if line:
                    result = append_output(line)
                else:
                    result = get_output()
                send_json(self, HTTPStatus.OK, result, session_id=sid)
                return

            if path == "/api/output/clear":
                send_json(self, HTTPStatus.OK, clear_output(), session_id=sid)
                return

            if path == "/api/upload":
                filename = str(payload.get("filename", "data.csv"))
                b64_data = str(payload.get("data", ""))
                if not b64_data:
                    raise ValueError("No file data provided")
                result = upload_csv(filename, b64_data, sid)
                send_json(self, HTTPStatus.OK, result, session_id=sid)
                return

            if path == "/api/train/start":
                csv_path = get_csv_path(sid)
                if csv_path is None:
                    raise ValueError("No CSV uploaded")
                target_col = str(payload.get("target_col", ""))
                feature_cols = list(payload.get("feature_cols", []))
                algorithm = str(payload.get("algorithm", "random_forest"))
                params = dict(payload.get("params", {}))
                test_size = float(payload.get("test_size", 0.2))
                val_size = float(payload.get("val_size", 0.0))
                task_type = str(payload.get("task_type", "auto"))
                shuffle = str(payload.get("shuffle", "true")).lower() == "true"
                random_state = int(payload.get("random_state", 42))
                val_strategy = str(payload.get("val_strategy", "simple"))
                k_folds = int(payload.get("k_folds", 5))

                if not target_col:
                    raise ValueError("No target column selected")
                if not feature_cols:
                    raise ValueError("No feature columns selected")
                if target_col in feature_cols:
                    raise ValueError("Target column cannot be a feature")

                result = start_training(
                    sid, csv_path, target_col, feature_cols, algorithm, params,
                    test_size, task_type, random_state, shuffle, val_size,
                    val_strategy, k_folds
                )
                send_json(self, HTTPStatus.OK, result, session_id=sid)
                return

            if path == "/api/models/rename":
                result = rename_model(
                    str(payload.get("name", "")),
                    str(payload.get("new_name", "")),
                )
                send_json(self, HTTPStatus.OK, result, session_id=sid)
                return

            if path == "/api/models/delete":
                result = delete_model(str(payload.get("name", "")))
                send_json(self, HTTPStatus.OK, result, session_id=sid)
                return

            if path == "/api/models/stats":
                name = str(payload.get("name", ""))
                result = get_model_metrics(name) or {}
                send_json(self, HTTPStatus.OK, result, session_id=sid)
                return

            if path == "/api/models/save":
                name = str(payload.get("name", "")).strip()
                if not name:
                    raise ValueError("Model name is required")
                path_result = save_to_server(sid, name)
                send_json(
                    self,
                    HTTPStatus.OK,
                    {"saved": path_result},
                    session_id=sid,
                )
                return

            if path == "/api/dataset/clear":
                clear_uploaded(sid)
                send_json(self, HTTPStatus.OK, {"cleared": True}, session_id=sid)
                return

            # ── Test app endpoints ──

            if path == "/api/test/model_info":
                model_name = str(payload.get("model_name", ""))
                if not model_name:
                    raise ValueError("model_name required")
                metrics = get_model_metrics(model_name)
                if metrics:
                    send_json(self, HTTPStatus.OK, {
                        "feature_cols": metrics.get("feature_cols", []),
                        "target_col": metrics.get("target_col", ""),
                        "task": metrics.get("task", "auto"),
                        "has_metadata": True,
                    }, session_id=sid)
                else:
                    send_json(self, HTTPStatus.OK, {
                        "feature_cols": [],
                        "target_col": "",
                        "task": "auto",
                        "has_metadata": False,
                    }, session_id=sid)
                return

            if path == "/api/test/upload":
                b64_data = str(payload.get("data", ""))
                if not b64_data:
                    raise ValueError("No file data provided")
                raw = base64.b64decode(b64_data)
                token = str(uuid.uuid4())
                clean_model_temp(sid)
                MODEL_TEMP_DIR.mkdir(parents=True, exist_ok=True)
                dest = MODEL_TEMP_DIR / f"{sid}_test_{token}.model"
                dest.write_bytes(raw)
                send_json(self, HTTPStatus.OK, {"token": token, "filename": dest.name}, session_id=sid)
                return

            if path == "/api/test/predict_csv":
                model_name = str(payload.get("model_name", ""))
                upload_token = str(payload.get("upload_token", ""))
                raw_cols = payload.get("feature_cols")
                feature_cols = list(raw_cols) if raw_cols else None

                model = load_model(model_name=model_name, upload_token=upload_token, session_id=sid)

                csv_path = get_csv_path(sid)
                if not csv_path:
                    raise ValueError("No CSV uploaded. Upload a CSV file first.")

                result = predict_csv(model, csv_path, feature_cols)
                send_json(self, HTTPStatus.OK, result, session_id=sid)
                return

        except (FileNotFoundError, FileExistsError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            send_json(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)}, session_id=sid)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args) -> None:
        return
