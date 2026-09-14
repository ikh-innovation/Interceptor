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
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from time import perf_counter, sleep

import cv2

_BOUNDARY = "waspframe"

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,viewport-fit=cover">
<title>WASP</title>
<style>
  *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
  html,body{margin:0;height:100%;background:#000;color:#e8e8e8;
            font:16px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
                 ui-sans-serif,system-ui,sans-serif;
            -webkit-text-size-adjust:100%;overflow:hidden}
  .app{height:100%;display:flex;flex-direction:column}

  /* ── header: the colour IS the message.  Black = pilot has it,
       blue = ALT_HOLD, orange = ACRO.  Readable across a field. ── */
  .hdr{flex:0 0 auto;padding:16px 18px 14px;transition:background .18s linear}
  .hdr.m_stab{background:#101010}
  .hdr.m_alt {background:#0d3a7a}
  .hdr.m_acro{background:#8a4409}
  .hdr .title{font-size:clamp(28px,9vw,46px);font-weight:800;letter-spacing:-.01em;
              line-height:1.05}
  .hdr .desc{margin-top:5px;font-size:clamp(13px,3.8vw,16px);color:#cfd6e0;
             line-height:1.35}
  .hdr.m_stab .desc{color:#9a9a9a}
  .chips{display:flex;gap:7px;margin-top:11px;flex-wrap:wrap}
  .chip{font-size:clamp(11px,3.1vw,13px);font-weight:700;letter-spacing:.04em;
        padding:5px 10px;border-radius:999px;background:rgba(255,255,255,.14);
        color:#fff;white-space:nowrap}
  .chip.hot{background:#fff;color:#000}

  .vid{flex:1 1 auto;min-height:0;display:flex;align-items:center;
       justify-content:center;background:#000}
  img{max-width:100%;max-height:100%;object-fit:contain;display:block}

  /* ── the panel the active tab shows ── */
  .panel{flex:0 0 auto;background:#0b0b0b;border-top:1px solid #242424}
  .panel .plabel{padding:9px 16px 0;font-size:12px;font-weight:700;
                 letter-spacing:.1em;color:#6d6d6d}
  .opts{display:flex;gap:8px;padding:9px 12px 12px}
  .opts button{flex:1;min-width:0;padding:15px 6px;border:1px solid #2e2e2e;
               border-radius:12px;background:#151515;color:#a8a8a8;
               font:inherit;font-size:clamp(12px,3.5vw,15px);font-weight:700;
               letter-spacing:.03em}
  .opts button.on{background:#f2f2f2;color:#000;border-color:#f2f2f2}

  /* ── bottom nav, app style ── */
  .nav{flex:0 0 auto;display:flex;background:#0b0b0b;border-top:1px solid #242424;
       padding-bottom:env(safe-area-inset-bottom)}
  .nav button{flex:1;padding:11px 4px 13px;background:none;border:0;
              color:#6d6d6d;font:inherit;font-size:11px;font-weight:700;
              letter-spacing:.07em}
  .nav button .ic{display:block;font-size:19px;line-height:1.1;margin-bottom:3px;
                  font-weight:400}
  .nav button.on{color:#fff}
  .msg{flex:0 0 auto;padding:5px 12px;font-size:11px;color:#5e5e5e;
       text-align:center;background:#0b0b0b;height:22px;overflow:hidden}
</style></head>
<body><div class="app">

  <div class="hdr m_stab" id="hdr">
    <div class="title" id="h_title">--</div>
    <div class="desc" id="h_desc">σύνδεση...</div>
    <div class="chips">
      <span class="chip" id="c_arm">--</span>
      <span class="chip" id="c_drv">--</span>
      <span class="chip" id="c_lock">--</span>
    </div>
  </div>

  <div class="vid"><img id="v" src="/stream.mjpg" alt=""></div>

  <div class="panel">
    <div class="plabel" id="p_label">ΤΑΧΥΤΗΤΑ</div>
    <div class="opts" id="p_opts"></div>
  </div>

  <div class="msg" id="msg">connecting</div>

  <div class="nav">
    <button data-tab="test"><span class="ic">&#9680;</span>TEST</button>
    <button data-tab="speed" class="on"><span class="ic">&#9654;</span>ΤΑΧΥΤΗΤΑ</button>
    <button data-tab="target"><span class="ic">&#9678;</span>ΣΤΟΧΟΣ</button>
  </div>
</div>
<script>
  var v = document.getElementById('v');
  v.onerror = function () {
    setTimeout(function () { v.src = '/stream.mjpg?t=' + Date.now(); }, 1000);
  };
  function T(id, s) { document.getElementById(id).textContent = s; }

  // Title + one line.  Not a manual -- the pilot has the manual; this is the
  // reminder you can read at arm's length while something is in the air.
  var MODES = {
    ALT_HOLD: {t: 'ALT_HOLD', d: 'Το AI οδηγεί και το γκάζι. Κέντρο γκαζιού = κράτημα ύψους.'},
    ACRO:     {t: 'ACRO',     d: 'Το γκάζι το κρατάς ΕΣΥ. Το AI οδηγεί μόνο roll / pitch / yaw.'},
    _pilot:   {t: null,       d: 'Ο χειριστής έχει τον πλήρη έλεγχο.'},
    _nolink:  {t: 'NO LINK',  d: 'Καμία σύνδεση με τον FC.'}
  };

  // TEMPORARY-ish: forces the header to draw a mode the FC is not in, so the
  // three looks can be checked on the bench.  Only changes what THIS page
  // draws -- it never reaches the tracker.  LIVE clears it.
  var PREVIEW = '';
  var TAB = 'speed';

  var TABS = {
    test:   {label: 'ΔΟΚΙΜΗ ΟΘΟΝΩΝ', key: 'pv',
             opts: [['', 'LIVE'], ['STABILIZE', 'STABILIZE'],
                    ['ALT_HOLD', 'ALT_HOLD'], ['ACRO', 'ACRO']]},
    speed:  {label: 'ΤΑΧΥΤΗΤΑ', key: 'mode',
             opts: [['SLOW', 'SLOW'], ['MEDIUM', 'MEDIUM'], ['FAST', 'FAST']]},
    target: {label: 'ΣΤΟΧΟΣ', key: 'target',
             opts: [['HUMAN', 'HUMAN'], ['DRONE', 'DRONE']]}
  };

  var LAST = {};

  function drawPanel() {
    var t = TABS[TAB];
    T('p_label', t.label);
    var box = document.getElementById('p_opts');
    box.innerHTML = '';
    for (var i = 0; i < t.opts.length; i++) {
      var b = document.createElement('button');
      b.textContent = t.opts[i][1];
      b.setAttribute('data-k', t.key);
      b.setAttribute('data-v', t.opts[i][0]);
      box.appendChild(b);
    }
    markPanel();
  }

  function markPanel() {
    var t = TABS[TAB];
    var cur = (t.key === 'pv') ? PREVIEW
            : (t.key === 'mode') ? LAST.mode : LAST.target;
    var bs = document.querySelectorAll('#p_opts button');
    for (var i = 0; i < bs.length; i++) {
      bs[i].classList.toggle('on', bs[i].getAttribute('data-v') === (cur || ''));
    }
  }

  // Painted from the SERVER's reply, never from the click -- if the tracker
  // refused a command, or the pilot flipped the switch, this shows what is
  // actually true.
  function paint(j) {
    LAST = j;
    var linked = j.linked, mode = j.fc_mode || '?';
    var armed = j.armed, steering = j.steering, state = j.state;
    if (PREVIEW) { linked = true; mode = PREVIEW;
                   steering = (PREVIEW !== 'STABILIZE') && steering; }

    var spec = !linked ? MODES._nolink : (MODES[mode] || MODES._pilot);
    T('h_title', spec.t || mode);
    T('h_desc', spec.d);
    document.getElementById('hdr').className = 'hdr ' +
      (linked && mode === 'ALT_HOLD' ? 'm_alt'
       : linked && mode === 'ACRO' ? 'm_acro' : 'm_stab');

    T('c_arm', armed ? 'ΟΠΛΙΣΜΕΝΟ' : 'ΑΣΦΑΛΕΙΑ');
    document.getElementById('c_arm').className = 'chip' + (armed ? ' hot' : '');
    T('c_drv', steering ? 'AI ΟΔΗΓΕΙ' : 'ΧΕΙΡΙΣΤΗΣ');
    document.getElementById('c_drv').className = 'chip' + (steering ? ' hot' : '');
    T('c_lock', state === 'LOCK' ? 'ΚΛΕΙΔΩΜΕΝΟΣ'
              : state === 'PREDICT' ? 'PREDICT' : 'ΑΝΑΖΗΤΗΣΗ');
    document.getElementById('c_lock').className =
      'chip' + (state === 'LOCK' ? ' hot' : '');
    markPanel();
  }

  var busy = false;
  function req(url, note) {
    if (busy) return;
    busy = true;
    fetch(url, {cache: 'no-store'}).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(function (j) {
      paint(j);
      if (note) T('msg', note + ' \u2014 ' + new Date().toLocaleTimeString());
    }).catch(function (e) {
      T('msg', 'link error: ' + e.message);
    }).then(function () { busy = false; });
  }

  document.addEventListener('click', function (ev) {
    var b = ev.target.closest ? ev.target.closest('button') : null;
    if (!b) return;
    if (b.hasAttribute('data-tab')) {
      TAB = b.getAttribute('data-tab');
      var nb = document.querySelectorAll('.nav button');
      for (var i = 0; i < nb.length; i++) nb[i].classList.toggle('on', nb[i] === b);
      drawPanel();
      return;
    }
    var k = b.getAttribute('data-k'), val = b.getAttribute('data-v');
    if (k === 'pv') {
      PREVIEW = val;
      T('msg', val ? ('ΔΟΚΙΜΗ: ' + val + ' (δεν αλλάζει τίποτα στο drone)') : 'LIVE');
      req('/state');
      return;
    }
    T('msg', 'sending ' + k + '=' + val);
    req('/cmd?' + k + '=' + encodeURIComponent(val), k + '=' + val);
  });

  drawPanel();
  req('/state');
  setInterval(function () { req('/state'); }, 500);
</script>
</body></html>
""".encode("utf-8")


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
                 fps: float = 15.0, host: str = "0.0.0.0",
                 state_get=None, command=None):
        # state_get()      -> dict of what the tracker is ACTUALLY doing.
        # command(k, v)    -> applies an operator command, returns nothing.
        # Both are None when nothing is wired, and the dashboard then 503s
        # rather than pretending to have changed something.
        self.state_get = state_get
        self.command = command
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
                elif path in ("/state", "/sens"):
                    self._state()
                elif path == "/cmd":
                    self._cmd()
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
                except Exception:            # never take the flight down
                    return None

            def _state(self):
                """What the tracker is doing right now, read off the live loop."""
                st = self._read_state()
                if st is None:
                    self.send_error(503, "no tracker wired")
                    return
                self._json(st)

            def _cmd(self):
                """Apply an operator command, then report the resulting state.

                GET /cmd?sens=STANDARD
                GET /cmd?target=DRONE

                The reply is the state read back AFTER the command, so the
                dashboard can only ever display what the control loop actually
                holds.  A button that lit up on its own click would confirm
                nothing.
                """
                if server.command is None:
                    self.send_error(503, "no tracker wired")
                    return
                q = parse_qs(urlparse(self.path).query)
                applied = []
                for key in ("mode", "speed", "sens", "target"):
                    if key in q:
                        try:
                            server.command(key, q[key][0])
                            applied.append(key)
                        except Exception as exc:
                            self.send_error(400, f"{key}: {exc}")
                            return
                st = self._read_state() or {}
                st["applied"] = applied
                self._json(st)

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

    # FIRST, enumerate the interfaces directly.  The route probe below only
    # finds an address if one of its hardcoded destinations is routable, so on
    # a field network with no internet and a gateway that is not 192.168.1.1 or
    # 10.0.0.1 it found NOTHING and the banner printed a literal
    # "http://<jetson-ip>:8080/" — which is what happened on 2026-09-10 after
    # the Jetson moved to a different WiFi.  Asking the kernel what addresses
    # are actually configured cannot fail that way.
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
        pass                      # fall through to the route probe

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
