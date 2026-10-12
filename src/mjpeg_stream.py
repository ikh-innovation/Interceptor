
from __future__ import annotations

import socket
import threading
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from time import perf_counter, sleep

import cv2

_BOUNDARY = "waspframe"

import os
from pathlib import Path

_UI_FILE = Path(__file__).resolve().parent / "mission_ui.html"
_FONT_DIR = Path(__file__).resolve().parent / "fonts"
_FONTS = ("Ubuntu-R.ttf", "Ubuntu-M.ttf", "Ubuntu-B.ttf", "Ubuntu-C.ttf",
          "UbuntuMono-B.ttf")
_IMG_DIR = Path(__file__).resolve().parent / "img"
_IMGS = ("ikd-logo.svg",)


def _load_page():
    try:
        return _UI_FILE.read_bytes()
    except OSError as exc:
        return (f"<h1>WASP</h1><p>UI file missing: {_UI_FILE} ({exc})</p>"
                ).encode("utf-8")



class MJPEGServer:

    CONTROL_KEYS = ("hold", "release", "step", "engage", "lvl", "mode", "lock", "auto", "autoen", "autop", "autoreset",
                    "sky", "follow", "cam", "engine", "stabilization", "def", "waspen", "waspp", "mount", "flightmode", "idles", "testmode", "testtarget")
    ALL_KEYS = CONTROL_KEYS + ("abort",)
    LEASE_S = 2.0

    def __init__(self, port: int = 8080, quality: int = 70,
                 fps: float = 15.0, host: str = "0.0.0.0",
                 state_get=None, command=None, console_get=None):
        self.state_get = state_get
        self.command = command
        self.console_get = console_get
        self._lease = None
        self._lease_lock = threading.Lock()
        self.port    = int(port)
        self.host    = host
        self.quality = max(1, min(100, int(quality)))
        self.min_dt  = 1.0 / max(1.0, float(fps))

        self._raw       = None
        self._jpeg      = None
        self._seq       = 0
        self._clients   = 0
        self._running   = False
        self._lock      = threading.Lock()
        self._new_raw   = threading.Condition(self._lock)
        self._new_jpeg  = threading.Condition(self._lock)
        self._httpd     = None
        self._encoder   = None
        self._n_encoded = 0

    def start(self) -> "MJPEGServer":
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_a):
                pass

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    self._send_page()
                elif path in ("/stream.mjpg", "/stream", "/video"):
                    self._send_stream()
                elif path in ("/snapshot.jpg", "/snapshot"):
                    self._send_snapshot()
                elif path in ("/state", "/sens"):
                    self._state()
                elif path == "/cmd":
                    self._cmd()
                elif path == "/console":
                    self._console()
                elif path.startswith("/font/") and path[6:] in _FONTS:
                    self._send_font(path[6:])
                elif path.startswith("/img/") and path[5:] in _IMGS:
                    self._send_img(path[5:])
                else:
                    self.send_error(404)

            def _json(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _read_state(self):
                if server.state_get is None:
                    return None
                try:
                    return server.state_get()
                except Exception:
                    return None

            def _state(self):
                st = self._read_state()
                if st is None:
                    self.send_error(503, "no tracker wired")
                    return
                q = parse_qs(urlparse(self.path).query)
                st["control"] = server._lease_status((q.get("c") or [""])[0],
                                                     renew=True)
                self._json(st)

            def _console(self):
                q = parse_qs(urlparse(self.path).query)
                try:
                    since = int((q.get("since") or ["0"])[0])
                except ValueError:
                    since = 0
                rows = server.console_get(since) if server.console_get else []
                self._json([list(r) for r in rows])

            def _cmd(self):
                if server.command is None:
                    self.send_error(503, "no tracker wired")
                    return
                q = parse_qs(urlparse(self.path).query)
                client = (q.get("c") or [""])[0]
                keys = [k for k in server.ALL_KEYS if k in q]
                if any(k in server.CONTROL_KEYS for k in keys):
                    if not server._take_lease(client):
                        self._json(dict(error="another phone has control"),
                                   code=409)
                        return
                applied, refused = [], ""
                for key in keys:
                    try:
                        why = server.command(key, q[key][0])
                        applied.append(key)
                        if why:
                            refused = why
                    except Exception as exc:
                        refused = f"{key}: {exc}"
                st = self._read_state() or {}
                st["applied"] = applied
                st["refused"] = refused
                st["control"] = server._lease_status(client, renew=True)
                self._json(st)

            def _send_page(self):
                page = _load_page()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(page)

            def _send_font(self, name):
                self._send_file(_FONT_DIR / name, "font/ttf", "max-age=31536000, immutable")

            def _send_img(self, name):
                self._send_file(_IMG_DIR / name, "image/svg+xml", "max-age=3600")

            def _send_file(self, path, ctype, cache):
                try:
                    body = path.read_bytes()
                except OSError:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", cache)
                self.end_headers()
                self.wfile.write(body)

            def _send_snapshot(self):
                server._client_in()
                try:
                    jpeg = server.latest_jpeg(timeout=2.0)
                finally:
                    server._client_out()
                if jpeg is None:
                    self.send_error(503, "no frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(jpeg)

            def _send_stream(self):
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control",
                                 "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary="
                                 + _BOUNDARY)
                self.end_headers()
                server._client_in()
                last = -1
                try:
                    while server._running:
                        jpeg, last = server._wait_for_frame(last, timeout=1.0)
                        if jpeg is None:
                            continue
                        self.wfile.write(("--%s\r\n" % _BOUNDARY).encode())
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            ("Content-Length: %d\r\n\r\n" % len(jpeg)).encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    server._client_out()

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._httpd.daemon_threads = True
        self._running = True

        threading.Thread(target=self._httpd.serve_forever,
                         name="mjpeg-http", daemon=True).start()
        self._encoder = threading.Thread(target=self._encode_loop,
                                         name="mjpeg-encode", daemon=True)
        self._encoder.start()

        for url in self.urls():
            pass


        return self

    def _take_lease(self, client):
        now = perf_counter()
        with self._lease_lock:
            if not client:
                return False
            cur = self._lease
            if cur is None or cur[0] == client or (now - cur[1]) > self.LEASE_S:
                self._lease = (client, now)
                return True
            return False

    def _lease_status(self, client, renew=False):
        now = perf_counter()
        with self._lease_lock:
            cur = self._lease
            if cur is None or (now - cur[1]) > self.LEASE_S:
                return "free"
            if client and cur[0] == client:
                if renew:
                    self._lease = (client, now)
                return "you"
            return "other"

    def stop(self):
        if not self._running:
            return
        self._running = False
        with self._lock:
            self._new_raw.notify_all()
            self._new_jpeg.notify_all()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


    def publish(self, frame_bgr):
        if not self._running or self._clients == 0 or frame_bgr is None:
            return
        with self._lock:
            self._raw = frame_bgr.copy()
            self._new_raw.notify()

    def latest_jpeg(self, timeout: float = 1.0):
        with self._lock:
            self._new_jpeg.wait(timeout)
            return self._jpeg

    def _wait_for_frame(self, last_seq: int, timeout: float):
        with self._lock:
            if self._seq == last_seq:
                self._new_jpeg.wait(timeout)
            if self._seq == last_seq:
                return None, last_seq
            return self._jpeg, self._seq

    def _client_in(self):
        with self._lock:
            self._clients += 1
            n = self._clients


    def _client_out(self):
        with self._lock:
            self._clients = max(0, self._clients - 1)
            n = self._clients


    def _encode_loop(self):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        t_last = 0.0
        while self._running:
            with self._lock:
                while self._running and self._raw is None:
                    self._new_raw.wait(0.5)
                if not self._running:
                    return
                frame     = self._raw
                self._raw = None

            now = perf_counter()
            if now - t_last < self.min_dt:
                continue
            t_last = now

            ok, buf = cv2.imencode(".jpg", frame, params)
            if not ok:
                continue
            with self._lock:
                self._jpeg = buf.tobytes()
                self._seq += 1
                self._n_encoded += 1
                self._new_jpeg.notify_all()

    def urls(self):
        out = ["http://%s:%d/" % (ip, self.port) for ip in local_ips()]
        if not out:
            out.append("http://<jetson-ip>:%d/" % self.port)
        return out


def local_ips():
    ips = []

    try:
        import array
        import fcntl
        import struct
        SIOCGIFADDR = 0x8915
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for _idx, name in socket.if_nameindex():
                try:
                    packed = fcntl.ioctl(
                        s.fileno(), SIOCGIFADDR,
                        struct.pack("256s", name.encode()[:15]))
                    ip = socket.inet_ntoa(packed[20:24])
                except OSError:
                    continue
                if ip and not ip.startswith("127.") and ip not in ips:
                    ips.append(ip)
        finally:
            s.close()
    except Exception:
        pass

    for probe in ("8.8.8.8", "192.168.1.1", "10.0.0.1"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((probe, 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
        except OSError:
            pass
        finally:
            s.close()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    return ips


if __name__ == "__main__":
    import numpy as np

    srv = MJPEGServer(port=8080).start()

    try:
        i = 0
        while True:
            img = np.zeros((480, 640, 3), np.uint8)
            x = (i * 4) % 640
            cv2.rectangle(img, (x, 0), (x + 40, 480), (0, 200, 255), -1)
            cv2.putText(img, "WASP stream test  frame %d" % i, (12, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            srv.publish(img)
            i += 1
            sleep(1 / 30)
    except KeyboardInterrupt:
        srv.stop()
