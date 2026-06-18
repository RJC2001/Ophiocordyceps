from __future__ import annotations

import base64
import datetime
import json
import mimetypes
import re
import threading
import uuid

from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

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


from train.modules.training_output import (
    get_output,
    append_output,
    clear_output,
)

from modules.model_manager import (
    list_models,
    list_all_models,
    delete_model,
    rename_model,
    get_model_metrics,
    get_model_metrics_anywhere,
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
    cancel_training,
)

from testing.modules.predictor import (
    load_model,
    predict_csv,
)

from train.modules.image_trainer import (
    upload_training_image,
    delete_training_image,
    save_annotations,
    get_image_training_data,
    clear_image_training,
    get_available_image_algorithms,
    start_image_training as start_img_training,
    get_image_training_status as get_img_training_status,
    cancel_image_training,
    save_to_server as img_save_to_server,
)

from testing.modules.image_predictor import (
    load_detector,
    predict_images,
)

image_test_states: dict[str, dict[str, Any]] = {}
csv_test_states: dict[str, dict[str, Any]] = {}
image_test_uploads: dict[str, list[dict[str, str]]] = {}


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
    ".zip": "application/zip",
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
            qs = parse_qs(urlparse(self.path).query)
            model_type = qs.get("type", [None])[0]
            if model_type == "csv":
                result = list_models("csv")
            elif model_type == "image":
                result = list_models("image")
            else:
                result = list_all_models()
            send_json(self, HTTPStatus.OK, result, session_id=sid)
            return

        if path == "/api/algorithms":
            send_json(self, HTTPStatus.OK, get_available_algorithms(), session_id=sid)
            return

        if path == "/api/train/status":
            status = get_training_status(sid)
            send_json(self, HTTPStatus.OK, status, session_id=sid)
            return

        if path == "/api/image_train/status":
            status = get_img_training_status(sid)
            send_json(self, HTTPStatus.OK, status, session_id=sid)
            return

        if path == "/api/image_test/status":
            state = image_test_states.get(sid, {})
            st_resp: dict[str, Any] = {
                "status": state.get("status", "idle"),
            }
            if state.get("detections"):
                st_resp["detections"] = state["detections"]
            if state.get("error"):
                st_resp["error"] = state["error"]
            if state.get("zip_token"):
                st_resp["zip_token"] = state["zip_token"]
            send_json(self, HTTPStatus.OK, st_resp, session_id=sid)
            return

        if path.startswith("/api/image_test/download/"):
            zip_token = path.removeprefix("/api/image_test/download/")
            tmp = TEMP_DIR / f"img_test_{sid}_{zip_token}.zip"
            if not tmp.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "ZIP not found")
                return
            data = tmp.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition", 'attachment; filename="detections.zip"')
            _set_session_cookie(self, sid)
            self.end_headers()
            self.wfile.write(data)
            tmp.unlink(missing_ok=True)
            return

        if path == "/api/test/status":
            state = csv_test_states.get(sid, {})
            st_resp: dict[str, Any] = {
                "status": state.get("status", "idle"),
            }
            if state.get("results") is not None:
                st_resp["results"] = state["results"]
            if state.get("error"):
                st_resp["error"] = state["error"]
            send_json(self, HTTPStatus.OK, st_resp, session_id=sid)
            return

        if path == "/api/image_algorithms":
            send_json(self, HTTPStatus.OK, get_available_image_algorithms(), session_id=sid)
            return

        if path == "/api/image_train/data":
            data = get_image_training_data(sid)
            send_json(self, HTTPStatus.OK, data, session_id=sid)
            return

        # Model download from server db/
        model_dl_match = re.match(r"^/api/models/download/(.+)$", path)
        if model_dl_match:
            try:
                qs = parse_qs(urlparse(self.path).query)
                dl_type = qs.get("type", [None])[0]
                dl_subdir = "image" if dl_type == "image" else "csv"
                model_name = model_dl_match.group(1)
                model_path = get_model_path(model_name, dl_subdir)
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
            csv_info = get_uploaded_csv(sid)
            orig = Path(csv_info["filename"]).stem if csv_info and csv_info.get("filename") else "model"
            dl_name = orig + ".model"
            data = mp.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition", f'attachment; filename="{dl_name}"')
            _set_session_cookie(self, sid)
            self.end_headers()
            self.wfile.write(data)
            return

        # Image model download (temp)
        if path.startswith("/api/image_train/download/"):
            from train.modules.image_trainer import _image_training_states
            state = _image_training_states.get(sid)
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
            img_ids = list(state.get("images", {}).keys())
            first_img = state["images"].get(img_ids[0]) if img_ids else None
            stem = Path(first_img["filename"]).stem if first_img else "image_detector"
            dl_name = stem + ".model"
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

            if path == "/api/train/cancel":
                result = cancel_training(sid)
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
                model_type = str(payload.get("type", "csv"))
                subdir = "image" if model_type == "image_detector" or model_type == "image" else "csv"
                result = rename_model(
                    str(payload.get("name", "")),
                    str(payload.get("new_name", "")),
                    subdir,
                )
                send_json(self, HTTPStatus.OK, result, session_id=sid)
                return

            if path == "/api/models/delete":
                model_type = str(payload.get("type", "csv"))
                subdir = "image" if model_type == "image_detector" or model_type == "image" else "csv"
                result = delete_model(str(payload.get("name", "")), subdir)
                send_json(self, HTTPStatus.OK, result, session_id=sid)
                return

            if path == "/api/models/stats":
                model_type = str(payload.get("type", "csv"))
                subdir = "image" if model_type == "image_detector" or model_type == "image" else "csv"
                name = str(payload.get("name", ""))
                result = get_model_metrics(name, subdir) or {}
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
                model_type_hint = str(payload.get("type", ""))
                subdir = "image" if model_type_hint in ("image", "image_detector") else "csv"
                metrics = get_model_metrics(model_name, subdir) if subdir else get_model_metrics_anywhere(model_name)
                if not metrics:
                    metrics = get_model_metrics(model_name, "csv") or get_model_metrics(model_name, "image")
                if metrics and metrics.get("type") == "image_detector":
                    pw = metrics.get("patch_width") or metrics.get("patch_size", 32)
                    ph = metrics.get("patch_height") or metrics.get("patch_size", 32)
                    send_json(self, HTTPStatus.OK, {
                        "type": "image_detector",
                        "patch_size": pw,
                        "patch_width": pw,
                        "patch_height": ph,
                        "has_metadata": True,
                    }, session_id=sid)
                elif metrics:
                    send_json(self, HTTPStatus.OK, {
                        "type": "csv",
                        "feature_cols": metrics.get("feature_cols", []),
                        "target_col": metrics.get("target_col", ""),
                        "task": metrics.get("task", "auto"),
                        "has_metadata": True,
                    }, session_id=sid)
                else:
                    # Unknown model type - try to detect from the model file
                    for try_subdir in ("csv", "image"):
                        try:
                            model_path = get_model_path(model_name, try_subdir)
                            data = joblib.load(model_path)
                            if isinstance(data, dict) and data.get("metadata", {}).get("type") == "image_detector":
                                meta = data["metadata"]
                                pw = meta.get("patch_width") or meta.get("patch_size", 32)
                                ph = meta.get("patch_height") or meta.get("patch_size", 32)
                                send_json(self, HTTPStatus.OK, {
                                    "type": "image_detector",
                                    "patch_size": pw,
                                    "patch_width": pw,
                                    "patch_height": ph,
                                    "has_metadata": True,
                                }, session_id=sid)
                                return
                            else:
                                send_json(self, HTTPStatus.OK, {
                                    "type": "csv",
                                    "feature_cols": [],
                                    "target_col": "",
                                    "task": "auto",
                                    "has_metadata": False,
                                }, session_id=sid)
                                return
                        except Exception:
                            continue
                    send_json(self, HTTPStatus.OK, {
                        "type": "csv",
                        "feature_cols": [],
                        "target_col": "",
                        "task": "auto",
                        "has_metadata": False,
                    }, session_id=sid)
                return

            if path == "/api/test/upload_csv":
                filename = str(payload.get("filename", "data.csv"))
                b64_data = str(payload.get("data", ""))
                if not b64_data:
                    raise ValueError("No file data provided")
                result = upload_csv(filename, b64_data, sid)
                send_json(self, HTTPStatus.OK, result, session_id=sid)
                return

            if path == "/api/image_test/upload":
                # Supports both a single image {"filename","data"} and a list under "images"
                images = payload.get("images")
                if isinstance(images, list) and images:
                    normalized: list[dict[str, str]] = []
                    for item in images:
                        if not isinstance(item, dict):
                            continue
                        data = str(item.get("data", ""))
                        if not data:
                            continue
                        normalized.append({
                            "filename": str(item.get("filename", "image.png")),
                            "data": data,
                        })
                else:
                    data = str(payload.get("data", ""))
                    if not data:
                        raise ValueError("No image data provided")
                    normalized = [{
                        "filename": str(payload.get("filename", "image.png")),
                        "data": data,
                    }]

                if not normalized:
                    raise ValueError("No image data provided")

                image_test_uploads[sid] = normalized
                send_json(self, HTTPStatus.OK, {"uploaded": True, "count": len(normalized)}, session_id=sid)
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

            if path == "/api/test/predict":
                path = "/api/test/predict_csv"

            if path == "/api/test/predict_csv":
                model_name = str(payload.get("model_name", ""))
                upload_token = str(payload.get("upload_token", ""))
                raw_cols = payload.get("feature_cols")
                feature_cols = list(raw_cols) if raw_cols else None

                state = csv_test_states.get(sid)
                if state and state.get("status") == "running":
                    raise ValueError("A prediction is already in progress")

                model = load_model(model_name=model_name, upload_token=upload_token, session_id=sid)
                csv_path = get_csv_path(sid)
                if not csv_path:
                    raise ValueError("No CSV uploaded. Upload a CSV file first.")

                csv_test_states[sid] = {
                    "status": "running",
                    "_cancel": False,
                    "results": None,
                    "error": None,
                }

                def _csv_cancelled() -> bool:
                    return csv_test_states.get(sid, {}).get("_cancel", False)

                def _csv_worker():
                    st = csv_test_states.get(sid, {})
                    try:
                        if _csv_cancelled():
                            st["status"] = "cancelled"
                            return
                        result = predict_csv(model, csv_path, feature_cols)
                        if _csv_cancelled():
                            st["status"] = "cancelled"
                            return
                        st["results"] = result
                        st["status"] = "done"
                    except Exception as e:
                        st["status"] = "error"
                        st["error"] = str(e)

                t = threading.Thread(target=_csv_worker, daemon=True)
                t.start()

                send_json(self, HTTPStatus.OK, {"status": "started"}, session_id=sid)
                return

            # ── Image training endpoints ──

            if path == "/api/image_train/upload":
                filename = str(payload.get("filename", "image.png"))
                b64_data = str(payload.get("data", ""))
                if not b64_data:
                    raise ValueError("No image data provided")
                result = upload_training_image(sid, filename, b64_data)
                send_json(self, HTTPStatus.OK, result, session_id=sid)
                return

            if path == "/api/image_train/delete":
                image_id = str(payload.get("image_id", ""))
                if not image_id:
                    raise ValueError("image_id required")
                result = delete_training_image(sid, image_id)
                send_json(self, HTTPStatus.OK, result, session_id=sid)
                return

            if path == "/api/image_train/annotations":
                image_id = str(payload.get("image_id", ""))
                annotations = list(payload.get("annotations", []))
                if not image_id:
                    raise ValueError("image_id required")
                result = save_annotations(sid, image_id, annotations)
                send_json(self, HTTPStatus.OK, result, session_id=sid)
                return

            if path == "/api/image_train/clear":
                result = clear_image_training(sid)
                send_json(self, HTTPStatus.OK, result, session_id=sid)
                return

            if path == "/api/image_train/cancel":
                result = cancel_image_training(sid)
                send_json(self, HTTPStatus.OK, result, session_id=sid)
                return

            if path == "/api/image_train/start":
                val_strategy = str(payload.get("val_strategy", "simple"))
                val_size = float(payload.get("val_size", 0.2))
                val_size = max(0.0, min(0.8, val_size))
                test_size = float(payload.get("test_size", 0.0))
                test_size = max(0.0, min(0.5, test_size))
                algorithm = str(payload.get("algorithm", "random_forest"))
                neg_per_ann = int(payload.get("neg_per_ann", 10))
                if neg_per_ann != -1:
                    neg_per_ann = max(0, neg_per_ann)
                params = dict(payload.get("params", {}))
                shuffle = bool(payload.get("shuffle", True))
                random_state = int(payload.get("random_state", 42))
                k_folds = int(payload.get("k_folds", 5))
                k_folds = max(2, min(20, k_folds))
                patch_size = int(payload.get("patch_size", 0))
                if patch_size != 0:
                    patch_size = max(8, min(512, patch_size))
                stride = int(payload.get("stride", 0))
                test_image_ids = payload.get("test_image_ids", [])
                result = start_img_training(sid, val_strategy=val_strategy, val_size=val_size, test_size=test_size, algorithm=algorithm, neg_per_ann=neg_per_ann, params=params, shuffle=shuffle, random_state=random_state, k_folds=k_folds, patch_size=patch_size, test_image_ids=test_image_ids, stride=stride)
                send_json(self, HTTPStatus.OK, result, session_id=sid)
                return

            if path == "/api/image_train/save":
                name = str(payload.get("name", "")).strip()
                if not name:
                    raise ValueError("Model name is required")
                path_result = img_save_to_server(sid, name)
                send_json(self, HTTPStatus.OK, {"saved": path_result}, session_id=sid)
                return

            # ── Image test/predict endpoints ──

            if path == "/api/image_test/predict":
                model_name = str(payload.get("model_name", ""))
                upload_token = str(payload.get("upload_token", ""))
                images_list = list(payload.get("images", []))
                if not images_list:
                    images_list = list(image_test_uploads.get(sid, []))
                confidence = float(payload.get("confidence", 0.5))
                merge_threshold = float(payload.get("merge_threshold", 0.0))
                merge_threshold = max(0.0, min(0.95, merge_threshold))

                if not images_list:
                    raise ValueError("No images provided")

                model_obj, metadata = load_detector(
                    model_name=model_name, upload_token=upload_token, session_id=sid
                )

                state = image_test_states.get(sid)
                if state and state.get("status") == "running":
                    raise ValueError("A detection is already in progress")

                zip_token = uuid.uuid4().hex[:12]
                image_test_states[sid] = {
                    "status": "running",
                    "zip_token": zip_token,
                    "_cancel": False,
                    "detections": None,
                    "error": None,
                }
                image_test_uploads.pop(sid, None)

                def _cancelled() -> bool:
                    return image_test_states.get(sid, {}).get("_cancel", False)

                def _worker():
                    st = image_test_states.get(sid, {})
                    try:
                        zip_bytes, detections = predict_images(
                            model_obj, metadata, images_list,
                            confidence_threshold=confidence,
                            nms_threshold=merge_threshold,
                            cancelled_fn=_cancelled,
                        )
                        if _cancelled():
                            st["status"] = "cancelled"
                            return
                        tmp = TEMP_DIR / f"img_test_{sid}_{zip_token}.zip"
                        tmp.parent.mkdir(parents=True, exist_ok=True)
                        tmp.write_bytes(zip_bytes)
                        st["status"] = "done"
                        st["detections"] = detections
                    except Exception as e:
                        st["status"] = "error"
                        st["error"] = str(e)

                t = threading.Thread(target=_worker, daemon=True)
                t.start()

                send_json(self, HTTPStatus.OK, {
                    "status": "started",
                    "zip_token": zip_token,
                }, session_id=sid)
                return

            if path == "/api/image_test/cancel":
                state = image_test_states.get(sid)
                if state:
                    state["_cancel"] = True
                image_test_uploads.pop(sid, None)
                send_json(self, HTTPStatus.OK, {"status": "cancelling"}, session_id=sid)
                return

            # ── CSV test/predict endpoints ──

            if path == "/api/test/cancel":
                state = csv_test_states.get(sid)
                if state:
                    state["_cancel"] = True
                send_json(self, HTTPStatus.OK, {"status": "cancelling"}, session_id=sid)
                return

        except (FileNotFoundError, FileExistsError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            send_json(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)}, session_id=sid)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args) -> None:
        return
