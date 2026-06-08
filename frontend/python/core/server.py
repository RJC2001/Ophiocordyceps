from __future__ import annotations

from http.server import ThreadingHTTPServer

from core.config import APP_CONFIG_PATH, load_json
from core.routes import TrainHandler

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def get_host() -> str:
    config = load_json(APP_CONFIG_PATH, {})
    if isinstance(config, dict):
        host = config.get("host", DEFAULT_HOST)
        if isinstance(host, str):
            return host
    return DEFAULT_HOST


def get_port() -> int:
    config = load_json(APP_CONFIG_PATH, {})
    if isinstance(config, dict):
        try:
            return int(config.get("port", DEFAULT_PORT))
        except (TypeError, ValueError):
            return DEFAULT_PORT
    return DEFAULT_PORT


def run_server() -> None:
    host = get_host()
    port = get_port()
    server = ThreadingHTTPServer((host, port), TrainHandler)
    try:
        print(f"Serving on http://localhost:{port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
