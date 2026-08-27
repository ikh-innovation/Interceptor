"""
mjpeg_stream.py  -  view the annotated detection window from another device
==============================================================================
The Jetson flies headless.  `cv2.imshow` needs a display the Orin Nano does not
have in the field, and X-forwarding over WiFi is both fragile and slow.  So the
tracker publishes the SAME annotated frame it would have shown in the window as
an MJPEG stream over HTTP, and the operator opens it in a phone or laptop
browser on the same WiFi:

    http://<jetson-ip>:8080/

No client app, no plugin, no GStreamer on the receiving end - MJPEG-over-HTTP
is just a multipart response that every browser has understood for twenty years.

Why MJPEG and not RTSP / WebRTC
-------------------------------
RTSP is lower bitrate but needs VLC (or an app) on the phone and a server
process on the Jetson.  WebRTC is lower latency still but needs signalling and
a STUN path.  On a closed field WiFi, with one or two viewers and a 640x640
frame, MJPEG costs a few Mbit/s and ~150 ms of glass-to-glass, which is what a
spotter watching the HUD actually needs.

Cost control - this runs on the flight computer
-----------------------------------------------
Nothing is encoded unless somebody is actually watching:

  * `publish()` is called from the tracker's hot loop.  It does ONE memcpy of
    the display frame and returns.  It never encodes, never blocks, never
    waits on a socket.  If no client is connected it does not even copy.
  * A separate daemon thread does the JPEG encode, rate-limited to `fps`
    (default 15, half the capture rate) so the stream cannot steal cycles from
    YOLO or from the 20 Hz command tx thread.
  * Latest-frame-wins.  A phone on a weak signal falls behind and simply skips
    frames; it can never apply back-pressure to the control loop.

The server is bound on all interfaces and is UNAUTHENTICATED.  It is meant for
a private field WiFi / hotspot.  Do not expose the Jetson to a public network
with this running.
"""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import perf_counter, sleep

import cv2

_BOUNDARY = "waspframe"

_PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WASP seeker</title>
<style>
  html,body{margin:0;height:100%;background:#0b0d10;color:#8b949e;
            font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}
  .wrap{height:100%;display:flex;flex-direction:column;align-items:center;
        justify-content:center;gap:8px;padding:8px;box-sizing:border-box}
  img{max-width:100%;max-height:calc(100% - 28px);object-fit:contain;
      border:1px solid #21262d;border-radius:4px;background:#000}
  .bar{display:flex;gap:14px;align-items:center}
  a{color:#58a6ff;text-decoration:none}
</style></head>
<body><div class="wrap">
  <img id="v" src="/stream.mjpg" alt="seeker view">
  <div class="bar"><span>WASP seeker &mdash; live</span>
    <a href="/snapshot.jpg" target="_blank">snapshot</a>
    <a href="/stream.mjpg">raw stream</a></div>
</div>
<script>
  // A dropped WiFi packet ends the multipart response and the <img> just stops.
  // Re-arm the src so the operator does not have to reload the page by hand.
  var v = document.getElementById('v');
  v.onerror = function () {
    setTimeout(function () { v.src = '/stream.mjpg?t=' + Date.now(); }, 1000);
  };
</script>
</body></html>
"""


class MJPEGServer:
    """Serves the latest published frame as MJPEG over HTTP.

    Usage:
        stream = MJPEGServer(port=8080).start()
        ...
        stream.publish(frame_disp)      # every frame, from the tracker loop
        ...
        stream.stop()
    """

    def __init__(self, port: int = 8080, quality: int = 70,
                 fps: float = 15.0, host: str = "0.0.0.0"):
        self.port    = int(port)
        self.host    = host
        self.quality = max(1, min(100, int(quality)))
        self.min_dt  = 1.0 / max(1.0, float(fps))

        self._raw       = None          # newest BGR frame awaiting encode
        self._jpeg      = None          # newest encoded frame
        self._seq       = 0             # bumped per encoded frame
        self._clients   = 0
        self._running   = False
        self._lock      = threading.Lock()
        self._new_raw   = threading.Condition(self._lock)
        self._new_jpeg  = threading.Condition(self._lock)
        self._httpd     = None
        self._encoder   = None
        self._n_encoded = 0

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> "MJPEGServer":
        server = self                       # captured by the handler below

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"   # one response per connection

            def log_message(self, *_a):     # keep the flight console readable
                pass

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    self._send_page()
                elif path in ("/stream.mjpg", "/stream", "/video"):
                    self._send_stream()
                elif path in ("/snapshot.jpg", "/snapshot"):
                    self._send_snapshot()
                else:
                    self.send_error(404)

            def _send_page(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_PAGE)))
                self.end_headers()
                self.wfile.write(_PAGE)

            def _send_snapshot(self):
                # Count as a viewer for the duration: publish() is a no-op with
                # nobody watching, so without this a bare /snapshot.jpg on a
                # stream nobody has opened yet would never get a frame.
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
                    pass            # viewer closed the tab / left WiFi range
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
            print("[STREAM] seeker view -> " + url)
        print("[STREAM] MJPEG q=%d cap=%.0f fps - encodes only while a viewer "
              "is connected" % (self.quality, 1.0 / self.min_dt))
        return self

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
        print("[STREAM] stopped - %d frame(s) served" % self._n_encoded)

    # -- producer side (called from the tracker's hot loop) ------------------
    def publish(self, frame_bgr):
        """Hand the newest annotated frame to the stream.  Never blocks.

        Returns immediately with no copy at all when nobody is watching, so the
        cost on the Jetson of leaving streaming enabled is one integer read per
        frame until an operator actually opens the page.
        """
        if not self._running or self._clients == 0 or frame_bgr is None:
            return
        with self._lock:
            self._raw = frame_bgr.copy()    # decouple from the caller's buffer
            self._new_raw.notify()

    # -- consumer side -------------------------------------------------------
    def latest_jpeg(self, timeout: float = 1.0):
        """The next freshly encoded frame; the last one if none arrives in time."""
        with self._lock:
            self._new_jpeg.wait(timeout)
            return self._jpeg

    def _wait_for_frame(self, last_seq: int, timeout: float):
        with self._lock:
            if self._seq == last_seq:
                self._new_jpeg.wait(timeout)
            if self._seq == last_seq:
                return None, last_seq       # timed out; loop re-checks _running
            return self._jpeg, self._seq

    def _client_in(self):
        with self._lock:
            self._clients += 1
            n = self._clients
        print("[STREAM] viewer connected (%d watching)" % n)

    def _client_out(self):
        with self._lock:
            self._clients = max(0, self._clients - 1)
            n = self._clients
        print("[STREAM] viewer left (%d watching)" % n)

    # -- encoder thread ------------------------------------------------------
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

            # Rate-limit the encode, not the capture: the tracker keeps running
            # at full speed and we simply skip the frames in between.
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

    # -- addressing ----------------------------------------------------------
    def urls(self):
        """Every address the operator could type on the other device."""
        out = ["http://%s:%d/" % (ip, self.port) for ip in local_ips()]
        if not out:
            out.append("http://<jetson-ip>:%d/" % self.port)
        return out


def local_ips():
    """LAN IPv4 addresses of this machine, best-effort, no extra dependencies.

    The UDP 'connect' sends nothing - it only asks the kernel which local
    address would be used to reach that destination, which is exactly the
    interface the phone on the same WiFi will come in on.
    """
    ips = []
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
    # Smoke test with no camera and no tracker: a moving bar, so the network
    # path (Jetson -> WiFi -> phone) can be proven before a flight.
    import numpy as np

    srv = MJPEGServer(port=8080).start()
    print("[STREAM] test pattern - Ctrl-C to stop")
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
            sleep(1 / 30)      # not cv2.waitKey: there is no window to pump
    except KeyboardInterrupt:
        srv.stop()
