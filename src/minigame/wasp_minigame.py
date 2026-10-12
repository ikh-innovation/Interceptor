#!/usr/bin/env python3
"""Local bounding-box-driven drone playground. Run: python3 src/minigame/wasp_minigame.py.

Serves a self-contained third-person simulator; no models, camera, FC connection,
or Mission execution. Settings are read from the app's presets.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from wasp import Wasp  # noqa: E402


def camera_fov(name, default):
    try:
        value = float(os.environ.get(name, default))
        return value if math.isfinite(value) and 1 <= value <= 179 else default
    except ValueError:
        return default


def settings():
    """Read saved profile values without writing the live application's defaults."""
    wasp = Wasp()
    source = "Application presets"
    path = Path.home() / ".config/wasp/wasp.json"
    # Match the application's configurable profile file when supplied.
    path = Path(os.environ.get("WASP_DEFAULTS", path)).expanduser()
    try:
        saved = json.loads(path.read_text())
        if isinstance(saved, dict):
            wasp.load_defaults(saved)
            source = "Saved application profiles"
    except (OSError, ValueError, TypeError):
        pass
    return dict(profiles=wasp.profiles, selected=wasp.active_profile,
                throttle=wasp.params["throttle"], launch_s=wasp.params["launch_s"],
                throttle_choices=Wasp.SETTINGS["throttle"], source=source,
                profile_choices={k: v for k, v in Wasp.SETTINGS.items() if k != "throttle"},
                camera=dict(hfov=camera_fov("CAM_HFOV_DEG", 62.2),
                            vfov=camera_fov("CAM_VFOV_DEG", 48.8)))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        assets = {"/guidance.js": ("guidance.js", "text/javascript; charset=utf-8"),
                  "/": ("minigame.html", "text/html; charset=utf-8"),
                  "/simulator.js": ("simulator.js", "text/javascript; charset=utf-8")}
        if path == "/settings":
            body, mime = json.dumps(settings(), allow_nan=False).encode(), "application/json"
        elif path in assets:
            filename, mime = assets[path]
            body = (HERE / filename).read_bytes()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve(port, host="127.0.0.1"):
    with ThreadingHTTPServer((host, port), Handler) as server:
        print(f"Drone playground: http://{host}:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("MINIGAME_PORT", 8091)))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    serve(args.port, args.host)
