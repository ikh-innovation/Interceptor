"""Laptop (ground) side of the RFD radio link, the EKETA-EAB gateway and the map cache.

Everything between the laptop and the drone goes over the RFD radio:

  laptop -> drone   $RFD,<tid>,<lat>,<lon>,<alt>,<REL|MSL>*HH   targets, 5 Hz
                    $RFDSYNC,<tid>,...*HH                       which targets are alive
                    $RFDCMD,#<seq>,<command>*HH                 SEEK commands, resent until acked
  drone -> laptop   $RFDACK,<seq>,<1|0>,<message>*HH            one per command (runs once)
                    $RFDTEL,<json>*HH                           status for the map, 2 Hz
                    $RFDPAR,<json>*HH                           SEEK settings, every 10 s

Wi-Fi is only needed at the start, to fill the map cache (tiles and Leaflet).

EKETA-EAB (ICD "Διεπαφή φίλιων drones" v2.0): CAT062 arrives on UDP 18200; for each
datagram an ACK {"type":"ack","crc":"<crc32>"} and, at least once a second, the
drone's telemetry go to ws://192.168.1.10:18100/gcs over one WebSocket.
"""
from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
import urllib.request
import zlib

from asterix_cue import parse_rfd_msg, rfd_cmd_line, rfd_line, rfd_sync_line

TEL_STALE_S = 5.0


class RfdLink:
    """The laptop's RFD radio: targets and commands out, drone status in."""

    def __init__(self, dev, baud=57600, hz=5.0, serial_factory=None):
        self.dev, self.baud = dev, int(baud)
        self._factory = serial_factory
        self.period = 1.0 / max(0.5, float(hz))
        self._lock = threading.Lock()          # _tracks, _pending
        self._wlock = threading.Lock()         # one writer at a time: lines never interleave
        self._tracks = {}
        self._pending = {}
        self._seq = int(time.time() * 10) % 900000 + 1000
        self.tel = None
        self.tel_t = 0.0
        self.par = None
        self.tx = self.rx_lines = self.rx_bad = 0
        self.error = ""
        self.on_tel = None                     # callback(msg) on every status line
        self._stop = threading.Event()
        self.port = None
        self._open()
        threading.Thread(target=self._tx_loop, daemon=True, name="rfd-tx").start()
        threading.Thread(target=self._rx_loop, daemon=True, name="rfd-rx").start()
        print(f"[RFD] {dev} @{baud}: targets {1 / self.period:.0f} Hz, commands with ack, drone status in")

    def _open(self):
        try:
            if self._factory is not None:
                self.port = self._factory()
            else:
                import serial
                self.port = serial.Serial(self.dev, self.baud, timeout=0.2)
            self.error = ""
        except Exception as exc:
            self.port = None
            self.error = f"cannot open {self.dev}: {exc}"[:120]

    # ---- out -----------------------------------------------------------------

    def update_tracks(self, tracks):
        """[{tid, lat, lon, alt_m, frame}]: frame REL = metres above the drone's home."""
        with self._lock:
            self._tracks = {str(t["tid"]): t for t in tracks}

    def _write(self, line):
        port = self.port
        if port is None:
            return False
        try:
            with self._wlock:
                port.write((line + "\r\n").encode("ascii", "replace"))
            return True
        except Exception as exc:
            self.error = f"write: {exc}"[:120]
            return False

    def command(self, cmd, timeout=4.0, resend_s=0.7):
        """Send one SEEK command and wait for the drone's answer: (ok, message).
        Resent every resend_s until acked; the drone runs a resent one only once."""
        with self._lock:
            seq = str(self._seq)
            self._seq += 1
            ev = threading.Event()
            self._pending[seq] = [ev, None]
        line = rfd_cmd_line(f"#{seq},{cmd}")
        deadline = time.monotonic() + timeout
        while True:
            self._write(line)
            left = deadline - time.monotonic()
            if left <= 0 or ev.wait(min(resend_s, left)):
                break
        with self._lock:
            _ev, res = self._pending.pop(seq)
        if res is None:
            return False, "no answer from the drone over the RFD"
        return res

    def _tx_loop(self):
        while not self._stop.is_set():
            with self._lock:
                snap = list(self._tracks.values())
            if self._write(rfd_sync_line([t["tid"] for t in snap])):
                for t in snap:
                    if self._write(rfd_line(t["lat"], t["lon"], t["alt_m"], t.get("frame", "MSL"), tid=str(t["tid"]))):
                        self.tx += 1
            self._stop.wait(self.period)

    # ---- in ------------------------------------------------------------------

    def _rx_loop(self):
        buf = b""
        while not self._stop.is_set():
            port = self.port
            if port is None:
                self._stop.wait(1.0)
                self._open()
                continue
            try:
                chunk = port.read(getattr(port, "in_waiting", 0) or 1)   # no waiting for a full buffer
            except Exception as exc:
                self.error = f"read: {exc}"[:120]
                try:
                    port.close()
                except Exception:
                    pass
                self.port = None
                continue
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                self.ingest(raw.decode("ascii", "replace"))
            if len(buf) > 8192:
                buf = b""

    def ingest(self, line, now=None):
        now = time.monotonic() if now is None else now
        line = line.strip()
        if not line:
            return
        p = parse_rfd_msg(line, "RFDTEL")
        if p is not None:
            try:
                msg = json.loads(p)
            except ValueError:
                self.rx_bad += 1
                return
            self.tel, self.tel_t = msg, now
            self.rx_lines += 1
            cb = self.on_tel
            if cb is not None:
                try:
                    cb(msg)
                except Exception:
                    pass
            return
        p = parse_rfd_msg(line, "RFDACK")
        if p is not None:
            seq, _, rest = p.partition(",")
            ok, _, text = rest.partition(",")
            with self._lock:
                slot = self._pending.get(seq.strip())
                if slot is not None and slot[1] is None:
                    slot[1] = (ok.strip() == "1", text.strip())
                    slot[0].set()
            self.rx_lines += 1
            return
        p = parse_rfd_msg(line, "RFDPAR")
        if p is not None:
            try:
                self.par = json.loads(p)
                self.rx_lines += 1
            except ValueError:
                self.rx_bad += 1
            return
        self.rx_bad += 1

    def age(self, now=None):
        now = time.monotonic() if now is None else now
        return None if self.tel is None else now - self.tel_t

    def drone_state(self, now=None):
        """The drone's state in the shape the planner page reads (the Jetson's /state)."""
        m, age = self.tel, self.age(now)
        if m is None:
            return None
        live = age < TEL_STALE_S
        fc = m.get("fc") or [False, False, None]
        tracks = [dict(key=x[0], label=x[1], fresh=bool(x[2]) and live, dist_m=x[3], alt_rel=x[4],
                       age_s=x[5], lat=x[6], lon=x[7], speed=x[8], bearing=x[9])
                  for x in m.get("tr") or [] if len(x) >= 10]
        target = next((t for t in tracks if t["key"] == m.get("tg")), None) if live else None
        run = None
        r = m.get("run")
        if r:
            of = r.get("of")
            run = dict(id=r.get("id"), state=r.get("st"), why=r.get("why"), seek=r.get("seek"),
                       follow_only=r.get("fo"), alt=r.get("alt"), target_alt=r.get("ta"), gs=r.get("gs"),
                       home_m=r.get("hm"), elapsed_s=r.get("el"), battery_pct=r.get("bat"), cue=r.get("cue"),
                       offer=None if not of else dict(n=of.get("n"), kind=of.get("k"), text=of.get("tx")),
                       events=r.get("ev") or [])
        params = [dict(k=x[0], t=x[1], u=x[2], v=x[3], lo=x[4], hi=x[5], step=x[6])
                  for x in self.par or [] if len(x) >= 7]
        why = m.get("why") or ""
        if not live:
            why = f"no status from the drone over the RFD for {age:.0f} s"
        g = m.get("g") or [None, None]
        return dict(
            fc_mode=fc[2], armed=bool(fc[1]), linked=bool(fc[0]) and live, rfd_age=round(age, 1), rfd_live=live,
            mission=dict(position=m.get("p"), height_m=m.get("h"), home=m.get("hm"),
                         gps=dict(fix=g[0], sats=g[1]), battery_pct=m.get("bat"),
                         seeker=dict(cue=dict(tracks=tracks, selected=m.get("sel"), target=target,
                                              packets=m.get("rx")),
                                     run=run, result=m.get("res"), alert=m.get("al"),
                                     seek=dict(ready=bool(m.get("rd")) and live, why=why, params=params))))


# ---- the drone's RFD wired to the flight controller: MAVLink TUNNEL ------------

class MavTunnelPort:
    """A serial-like port for RfdLink that carries our lines inside MAVLink TUNNEL
    messages: for a drone whose RFD is wired to the FC's TELEM port (MAVLink2).
    The FC forwards them to the Jetson and back (see rfd_tunnel.py).  A heartbeat
    once a second (sysid 250, never the Jetson's 255) makes the FC route to us."""

    def __init__(self, dev, baud=57600, sysid=None, compid=None):
        import rfd_tunnel as T
        os.environ.setdefault("MAVLINK20", "1")       # the FC's TELEM port speaks MAVLink 2
        from pymavlink import mavutil
        self.T, self.mavutil = T, mavutil
        sysid, compid = sysid or T.LAPTOP_SYSID, compid or T.LAPTOP_COMPID
        self.m = mavutil.mavlink_connection(dev, baud=int(baud), source_system=sysid, source_component=compid)
        self.enc = T.encoder(self.m, sysid, compid)
        self._buf = bytearray()
        self._cv = threading.Condition()
        self._wbuf = b""
        self._tx = threading.Lock()
        self._re = T.Reassembler()
        self._id = 0
        self._stop = threading.Event()
        self.fc_hb_t = None             # the FC's own heartbeat heard over the radio
        self.fc_sysid = None
        self.tunnel_rx = self.tunnel_tx = 0
        threading.Thread(target=self._rx, daemon=True, name="mav-tunnel-rx").start()
        threading.Thread(target=self._hb, daemon=True, name="mav-tunnel-hb").start()

    @property
    def in_waiting(self):
        with self._cv:
            return len(self._buf)

    def read(self, n):
        with self._cv:
            if not self._buf:
                self._cv.wait(0.2)
            out = bytes(self._buf[:n])
            del self._buf[:n]
            return out

    def write(self, data):
        self._wbuf += bytes(data)
        while b"\n" in self._wbuf:
            line, self._wbuf = self._wbuf.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if line:
                self._send_line(line)
        return len(data)

    def _send_line(self, line):
        T = self.T
        with self._tx:
            self._id = (self._id + 1) & 0xFF
            for frag in T.fragments(line, self._id):
                self.enc.tunnel_send(0, 0, T.PAYLOAD_TYPE, len(frag), T.padded(frag))
        self.tunnel_tx += 1

    def _hb(self):
        mav = self.mavutil.mavlink
        while not self._stop.wait(1.0):
            try:
                with self._tx:
                    self.enc.heartbeat_send(mav.MAV_TYPE_ONBOARD_CONTROLLER, mav.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            except Exception:
                pass

    def _rx(self):
        T = self.T
        mav = self.mavutil.mavlink
        while not self._stop.is_set():
            try:
                msg = self.m.recv_match(blocking=True, timeout=0.5)
            except Exception:
                time.sleep(0.2)
                continue
            if msg is None:
                continue
            t = msg.get_type()
            if t == "HEARTBEAT" and msg.autopilot != mav.MAV_AUTOPILOT_INVALID:
                self.fc_hb_t, self.fc_sysid = time.monotonic(), msg.get_srcSystem()
            elif t == "TUNNEL" and int(msg.payload_type) == T.PAYLOAD_TYPE:
                n = max(0, min(int(msg.payload_length), T.PAYLOAD_MAX))
                line = self._re.feed(bytes(msg.payload[:n]))
                if line is not None:
                    self.tunnel_rx += 1
                    with self._cv:
                        self._buf += line + b"\n"
                        self._cv.notify_all()

    def fc_age(self):
        return None if self.fc_hb_t is None else time.monotonic() - self.fc_hb_t

    def close(self):
        self._stop.set()
        try:
            self.m.close()
        except Exception:
            pass


# ---- EKETA-EAB gateway (WebSocket) -------------------------------------------

def eketa_crc(datagram):
    """CRC-32/ISO-HDLC of the whole UDP payload, 8 lowercase hex digits."""
    return f"{zlib.crc32(bytes(datagram)) & 0xFFFFFFFF:08x}"


def eketa_record(msg, drone_id):
    """One telemetry record (ICD table 5) from an $RFDTEL status, or None."""
    p, h = msg.get("p"), msg.get("h")
    if not p or h is None:
        return None
    rec = dict(droneId=str(drone_id)[:32], latitude=p[0], longitude=p[1], relativeAltitude=h)
    gs = msg.get("gs")
    fc = msg.get("fc") or [None, None, None]
    opt = dict(altitude=msg.get("ms"), heading=msg.get("hd"), course=msg.get("co"),
               speed=None if gs is None else round(gs * 3.6, 1), verticalSpeed=msg.get("vz"),
               battery=msg.get("bat"),
               inAir=None if fc[1] is None else bool(fc[1]) and h > 1.0)
    rec.update({k: v for k, v in opt.items() if v is not None})
    return rec


class EketaGateway:
    """One WebSocket to the gateway: ACKs and telemetry; reconnects every 1 s.
    Nothing is queued while it is down (the ICD: missed ones are not resent)."""

    def __init__(self, url, drone_id):
        self.url, self.drone_id = url, drone_id
        self._q = queue.Queue(maxsize=200)
        self.connected = False
        self.sent_acks = self.sent_tel = 0
        self.replies = {}
        self.last_error = ""
        self._ws = None
        threading.Thread(target=self._run, daemon=True, name="eketa-ws").start()

    def ack(self, datagram):
        self._put(dict(type="ack", crc=eketa_crc(datagram)), "ack")

    def telemetry(self, msg):
        rec = eketa_record(msg, self.drone_id)
        if rec is not None:
            self._put(dict(type="telemetry", data=[rec]), "tel")

    def _put(self, obj, kind):
        if not self.connected:
            return
        try:
            self._q.put_nowait((kind, json.dumps(obj, separators=(",", ":"))))
        except queue.Full:
            pass

    def _run(self):
        try:
            import websocket
        except ImportError:
            self.last_error = "pip install websocket-client"
            print("[EKETA] websocket-client is missing: pip install websocket-client")
            return
        while True:
            try:
                ws = websocket.create_connection(self.url, timeout=3)
            except Exception as exc:
                self.last_error = f"connect: {exc}"[:160]
                time.sleep(1.0)
                continue
            self._ws, self.connected, self.last_error = ws, True, ""
            print(f"[EKETA] connected to {self.url}")
            threading.Thread(target=self._reader, args=(ws,), daemon=True, name="eketa-rx").start()
            try:
                while self.connected:
                    try:
                        kind, text = self._q.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    ws.send(text)
                    if kind == "ack":
                        self.sent_acks += 1
                    else:
                        self.sent_tel += 1
            except Exception as exc:
                self.last_error = f"send: {exc}"[:160]
            self.connected = False
            try:
                ws.close()
            except Exception:
                pass
            while not self._q.empty():
                self._q.get_nowait()
            print(f"[EKETA] disconnected ({self.last_error}); retrying every 1 s")
            time.sleep(1.0)

    def _reader(self, ws):
        while self.connected:
            try:
                text = ws.recv()
            except Exception as exc:
                if self.connected:
                    self.last_error = f"recv: {exc}"[:160]
                self.connected = False
                return
            try:
                rep = json.loads(text)
            except (TypeError, ValueError):
                continue
            key = f"{rep.get('status')} {rep.get('message')}"
            self.replies[key] = self.replies.get(key, 0) + 1
            if rep.get("status") != 200:
                self.last_error = f"{key}: {rep.get('details', '')}"[:160]

    def status(self):
        return dict(url=self.url, drone_id=self.drone_id, connected=self.connected, acks=self.sent_acks,
                    telemetry=self.sent_tel, replies=dict(self.replies), error=self.last_error)


# ---- offline map: tiles and Leaflet, cached on disk ----------------------------

CACHE_DIR = os.path.expanduser(os.environ.get("WASP_MAP_CACHE", "~/.cache/wasp-planner"))
TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
VENDOR = {"leaflet.min.js": "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js",
          "leaflet.min.css": "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"}


def _fetch(url, path, timeout=6.0):
    """The cached file at path, else download it once (None when offline)."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "wasp-ground-planner"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
    except Exception:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return data


def tile(z, y, x):
    return _fetch(TILE_URL.format(z=z, y=y, x=x), os.path.join(CACHE_DIR, "tiles", str(z), str(y), f"{x}.jpg"))


def vendor(name):
    url = VENDOR.get(name)
    return None if url is None else _fetch(url, os.path.join(CACHE_DIR, "vendor", name))


def _tile_xy(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


class Prefetch:
    """Download the tiles around a point (while there is Wi-Fi), once per area."""

    def __init__(self):
        self.done = self.total = self.failed = 0
        self.running = False
        self._areas = []
        self._lock = threading.Lock()

    def start(self, lat, lon, radius_m=1500.0, zmin=12, zmax=18):
        with self._lock:
            for a in self._areas:
                if math.hypot((a[0] - lat) * 111320, (a[1] - lon) * 111320 * math.cos(math.radians(lat))) < radius_m / 3:
                    return False
            self._areas.append((lat, lon))
        threading.Thread(target=self._run, args=(lat, lon, radius_m, zmin, zmax), daemon=True,
                         name="tile-prefetch").start()
        return True

    def _run(self, lat, lon, radius_m, zmin, zmax):
        dlat = radius_m / 111320.0
        dlon = radius_m / (111320.0 * max(0.1, math.cos(math.radians(lat))))
        jobs = []
        for z in range(zmin, zmax + 1):
            x0, y0 = _tile_xy(lat + dlat, lon - dlon, z)
            x1, y1 = _tile_xy(lat - dlat, lon + dlon, z)
            jobs += [(z, y, x) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
        with self._lock:
            self.total += len(jobs)
            self.running = True
        q = queue.Queue()
        for j in jobs:
            q.put(j)

        def worker():
            while True:
                try:
                    z, y, x = q.get_nowait()
                except queue.Empty:
                    return
                ok = tile(z, y, x) is not None
                with self._lock:
                    self.done += 1
                    self.failed += 0 if ok else 1

        ts = [threading.Thread(target=worker, daemon=True) for _ in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        vendor("leaflet.min.js")
        vendor("leaflet.min.css")
        with self._lock:
            self.running = False
        print(f"[MAP] cached {self.done - self.failed}/{self.total} tiles around {lat:.5f},{lon:.5f}")

    def status(self):
        with self._lock:
            return dict(done=self.done, total=self.total, failed=self.failed, running=self.running,
                        areas=len(self._areas))
