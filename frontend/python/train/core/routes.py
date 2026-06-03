from __future__ import annotations

import datetime
import json
import mimetypes

from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from core.config import LAYOUT_DIR, STATIC_DIR

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

from modules.training_form import (
    get_training_form,
    save_training_form,
)

from modules.training_output import (
    get_training_output,
    append_training_output,
    clear_training_output,
)

from modules.model_manager import (
    list_models,
    rename_model,
    delete_model,
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
}


def _set_session_cookie(
    handler: BaseHTTPRequestHandler,
    session_id: str,
) -> None:
    expires = (
        datetime.datetime.utcnow()
        + datetime.timedelta(days=365)
    ).strftime("%a, %d %b %Y %H:%M:%S GMT")
    handler.send_header(
        "Set-Cookie",
        f"{COOKIE_NAME}={session_id}; Path=/; HttpOnly; SameSite=Lax; Expires={expires}",
    )


def json_bytes(data: object) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")


def read_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(
        handler.headers.get(
            "Content-Length",
            "0",
        )
    )

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

    handler.send_header(
        "Content-Type",
        "application/json; charset=utf-8",
    )

    handler.send_header(
        "Content-Length",
        str(len(payload)),
    )

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

    handler.send_header(
        "Content-Type",
        content_type,
    )

    handler.send_header(
        "Content-Length",
        str(len(payload)),
    )

    if session_id:
        _set_session_cookie(handler, session_id)

    handler.end_headers()

    handler.wfile.write(payload)


def safe_static_path(url_path: str) -> Path | None:

    relative = url_path.removeprefix(
        "/static/"
    ).lstrip("/")

    candidate = (
        STATIC_DIR / relative
    ).resolve()

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
        sid, config = session_get_or_create(
            existing.value if existing else None
        )
        self._session_id = sid
        self._session_config = config
        return sid, config

    def do_GET(self) -> None:

        parsed = urlparse(self.path)

        path = parsed.path

        sid, session_config = self._resolve_session()

        if path in (
            "/",
            "/index.html",
        ):

            html = (
                LAYOUT_DIR / "index.html"
            ).read_text(
                encoding="utf-8"
            )

            send_text(
                self,
                HTTPStatus.OK,
                html,
                "text/html; charset=utf-8",
                session_id=sid,
            )

            return

        if path == "/api/theme":

            send_json(
                self,
                HTTPStatus.OK,
                {
                    "theme": resolve_theme(session_config),
                    "available": get_available_themes(),
                },
                session_id=sid,
            )

            return

        if path == "/api/language":

            send_json(
                self,
                HTTPStatus.OK,
                {
                    "language": resolve_language(session_config),
                    "available": get_available_languages(),
                },
                session_id=sid,
            )

            return

        if path == "/api/form":

            send_json(
                self,
                HTTPStatus.OK,
                get_training_form(),
                session_id=sid,
            )

            return

        if path == "/api/output":

            send_json(
                self,
                HTTPStatus.OK,
                get_training_output(),
                session_id=sid,
            )

            return

        if path == "/api/models":

            send_json(
                self,
                HTTPStatus.OK,
                list_models(),
                session_id=sid,
            )

            return

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

                self.send_error(
                    HTTPStatus.NOT_FOUND,
                    "Not found",
                )

                return

            content_type = (
                CONTENT_TYPES.get(
                    file_path.suffix.lower()
                )
                or mimetypes.guess_type(
                    str(file_path)
                )[0]
                or "application/octet-stream"
            )

            data = file_path.read_bytes()

            self.send_response(
                HTTPStatus.OK
            )

            self.send_header(
                "Content-Type",
                content_type,
            )

            self.send_header(
                "Content-Length",
                str(len(data)),
            )

            _set_session_cookie(self, sid)

            self.end_headers()

            self.wfile.write(data)

            return

        self.send_error(
            HTTPStatus.NOT_FOUND,
            "Not found",
        )

    def do_POST(self) -> None:

        parsed = urlparse(self.path)

        path = parsed.path

        sid, session_config = self._resolve_session()

        try:

            payload = read_body(self)

            if path == "/api/theme":

                theme = validate_theme(
                    payload.get(
                        "theme",
                        "",
                    )
                )

                session_set_theme(sid, theme)

                send_json(
                    self,
                    HTTPStatus.OK,
                    {
                        "theme": theme,
                    },
                    session_id=sid,
                )

                return

            if path == "/api/language":

                language = validate_language(
                    payload.get(
                        "language",
                        "",
                    )
                )

                session_set_language(sid, language)

                send_json(
                    self,
                    HTTPStatus.OK,
                    {
                        "language": language,
                    },
                    session_id=sid,
                )

                return

            if path == "/api/form":

                form = save_training_form(
                    payload
                )

                send_json(
                    self,
                    HTTPStatus.OK,
                    form,
                    session_id=sid,
                )

                return

            if path == "/api/output/append":

                line = str(
                    payload.get(
                        "line",
                        "",
                    )
                ).strip()

                if line:
                    result = append_training_output(
                        line
                    )
                else:
                    result = get_training_output()

                send_json(
                    self,
                    HTTPStatus.OK,
                    result,
                    session_id=sid,
                )

                return

            if path == "/api/output/clear":

                send_json(
                    self,
                    HTTPStatus.OK,
                    clear_training_output(),
                    session_id=sid,
                )

                return

            if path == "/api/models/delete":

                result = delete_model(
                    str(
                        payload.get(
                            "name",
                            "",
                        )
                    )
                )

                send_json(
                    self,
                    HTTPStatus.OK,
                    result,
                    session_id=sid,
                )

                return

            if path == "/api/models/rename":

                result = rename_model(
                    str(
                        payload.get(
                            "old_name",
                            "",
                        )
                    ),
                    str(
                        payload.get(
                            "new_name",
                            "",
                        )
                    ),
                )

                send_json(
                    self,
                    HTTPStatus.OK,
                    result,
                    session_id=sid,
                )

                return

        except (
            FileNotFoundError,
            FileExistsError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:

            send_json(
                self,
                HTTPStatus.BAD_REQUEST,
                {
                    "error": str(exc),
                },
                session_id=sid,
            )

            return

        self.send_error(
            HTTPStatus.NOT_FOUND,
            "Not found",
        )

    def log_message(
        self,
        format: str,
        *args,
    ) -> None:
        return