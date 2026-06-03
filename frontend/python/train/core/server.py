from __future__ import annotations

from http.server import ThreadingHTTPServer

from core.config import APP_CONFIG_PATH, load_json
from core.routes import TrainHandler

DEFAULT_PORT = 8000

def get_port() -> int:
    config = load_json(APP_CONFIG_PATH, {})
    if isinstance(config, dict):
        try:
            return int(config.get("port", DEFAULT_PORT))
        except (TypeError, ValueError):
            return DEFAULT_PORT
    return DEFAULT_PORT

def run_server() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", get_port()), TrainHandler)
    try:
        print(f"Serving on http://localhost:{server.server_address[1]}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()