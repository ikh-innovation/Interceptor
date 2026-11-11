"""ASTERIX CAT062 cue for the Seeker: target tracks over UDP, straight to the Jetson.

The same feed asterix062_bridge.py reads (unicast or multicast UDP), decoded
the same way (its extract_track), so the Seeker can fly to a track's latitude,
longitude and altitude until the camera sees the target.  One track is the
cue: the operator picks it on the phone, or the nearest one is used.

Altitude: I136 (measured flight level) is pressure altitude, I130 geometric;
both are taken as metres above sea level, so the height above home is that
minus the FC's home altitude (the QNH difference is the error).

The RFD radio: the laptop runs asterix062_bridge.py --rfd-dev, which sends the
one followed track as  $RFD,<lat>,<lon>,<alt_m>,<REL|MSL>*HH  lines (HH: XOR
of the characters between $ and *).  The Jetson reads them from its RFD radio
(RFD_DEV, or the first USB serial adapter that is not the flight controller)
as the track "RFD".
"""
from __future__ import annotations

import glob
import math
import os
import socket
import struct
import threading
from time import perf_counter

from asterix062_bridge import FT_TO_M, KT_PER_MS, extract_track

try:
    import asterix
except ImportError:          # pip install asterix_decoder
    asterix = None

KEEP_S = 15.0
RFD_KEY = "RFD"


def rfd_checksum(body):
    n = 0
    for ch in body:
        n ^= ord(ch)
    return f"{n:02X}"


def parse_rfd(line):
    """{lat, lon, alt_m, frame} from one $RFD line, or None (bad checksum or values)."""
    line = str(line).strip()
    if not line.startswith("$") or "*" not in line:
        return None
    body, _, cs = line[1:].partition("*")
    if len(cs.strip()) < 2 or rfd_checksum(body) != cs.strip().upper()[:2] or not body.startswith("RFD,"):
        return None
    parts = body.split(",")
    if len(parts) < 5:
        return None
    try:
        lat, lon, alt = float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return None
    frame = parts[4].strip().upper()
    if frame not in ("REL", "MSL") or not (-90 <= lat <= 90 and -180 <= lon <= 180 and math.isfinite(alt)):
        return None
    return dict(lat=lat, lon=lon, alt_m=alt, frame=frame)


def rfd_line(lat, lon, alt_m, frame="MSL"):
    body = f"RFD,{lat:.7f},{lon:.7f},{alt_m:.2f},{frame}"
    return f"${body}*{rfd_checksum(body)}"


def find_rfd_device():
    """RFD_DEV, else the first USB serial adapter that is not the flight controller."""
    dev = os.environ.get("RFD_DEV", "").strip()
    if dev:
        return dev
    for p in sorted(glob.glob("/dev/serial/by-id/*")):
        if "ardupilot" not in p.lower() and ("ftdi" in p.lower() or "rfd" in p.lower() or "uart" in p.lower()):
            return p
    usb = sorted(glob.glob("/dev/ttyUSB*"))
    return usb[0] if usb else ""


def _dist(a, b):
    lat0 = math.radians(a[0])
    return math.hypot(math.radians(b[1]-a[1])*math.cos(lat0)*6371000.0, math.radians(b[0]-a[0])*6371000.0)


def _bearing(a, b):
    lat0 = math.radians(a[0])
    return math.degrees(math.atan2(math.radians(b[1]-a[1])*math.cos(lat0), math.radians(b[0]-a[0]))) % 360


class AsterixCue:
    def __init__(self, port=8600, mcast="", iface="0.0.0.0", fresh_s=3.0):
        self.port, self.mcast, self.iface = int(port), mcast or "", iface or "0.0.0.0"
        self.fresh_s = float(fresh_s)
        self.selected = None          # track key, or None: the nearest
        self.error = "" if asterix is not None else "asterix_decoder not installed (pip install asterix_decoder)"
        self.packets = self.records = self.bad = 0
        self.rx_t = None
        self._tracks = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sock = None
        self._thread = None
        self.rfd_dev = ""
        self.rfd_baud = int(os.environ.get("RFD_BAUD", "57600") or 57600)
        self.rfd_open = False
        self.rfd_error = ""
        self.rfd_lines = self.rfd_bad = 0
        self.rfd_t = None
        self._rfd_thread = None

    @staticmethod
    def key(t):
        return t.get("key") or f"{t['icao']:06X}"

    # ---- the RFD radio (serial $RFD lines) --------------------------------------

    def start_rfd(self):
        if self._rfd_thread is None:
            self._rfd_thread = threading.Thread(target=self._rfd_run, name="rfd-cue", daemon=True)
            self._rfd_thread.start()

    def _rfd_run(self):
        try:
            import serial
        except ImportError:
            self.rfd_error = "pyserial not installed"
            return
        while not self._stop.is_set():
            dev = find_rfd_device()
            if not dev:
                self.rfd_dev, self.rfd_open, self.rfd_error = "", False, "no RFD radio on USB"
                self._stop.wait(2.0)
                continue
            try:
                port = serial.Serial(dev, self.rfd_baud, timeout=0.5)
            except (OSError, ValueError, serial.SerialException) as exc:
                self.rfd_dev, self.rfd_open, self.rfd_error = dev, False, str(exc)[:120]
                self._stop.wait(2.0)
                continue
            self.rfd_dev, self.rfd_open, self.rfd_error = dev, True, ""
            buf = b""
            try:
                while not self._stop.is_set():
                    chunk = port.read(256)
                    if not chunk:
                        continue
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        self.ingest_rfd(line.decode("ascii", "replace"))
                    buf = buf[-512:]
            except (OSError, serial.SerialException) as exc:
                self.rfd_error = f"lost: {exc}"[:120]
            finally:
                self.rfd_open = False
                try:
                    port.close()
                except Exception:
                    pass
            self._stop.wait(1.0)

    def ingest_rfd(self, line, now=None):
        """One $RFD line from the radio; the track 'RFD'."""
        now = perf_counter() if now is None else now
        fix = parse_rfd(line)
        with self._lock:
            if fix is None:
                if line.strip():
                    self.rfd_bad += 1
                return False
            self.rfd_lines += 1
            self.rfd_t = self.rx_t = now
            t = dict(key=RFD_KEY, icao=0, callsign="RFD", track_no=None, lat=fix["lat"], lon=fix["lon"],
                     gs_kt=None, track_deg=None)
            if fix["frame"] == "MSL":
                t["alt_ft"] = fix["alt_m"]/FT_TO_M
            else:
                t["alt_ft"], t["rel_m"] = None, fix["alt_m"]
            self._tracks[RFD_KEY] = (now, t)
        return True

    # ---- the UDP listener ----------------------------------------------------

    def start(self):
        self._stop.clear()
        self.start_rfd()
        if self._thread is not None or asterix is None:
            return self
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            s.bind(("", self.port))
            if self.mcast:
                mreq = struct.pack("4s4s", socket.inet_aton(self.mcast), socket.inet_aton(self.iface))
                s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            s.settimeout(0.5)
        except OSError as exc:
            self.error = f"UDP {self.port}: {exc}"
            return self
        self.error = ""
        self._sock = s
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="asterix-cue", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        for th in (self._thread, self._rfd_thread):
            if th is not None:
                th.join(timeout=2)
        if self._sock is not None:
            self._sock.close()
        self._thread = self._sock = self._rfd_thread = None

    def configure(self, port, mcast, iface):
        self.stop()
        self.port, self.mcast, self.iface = int(port), mcast or "", iface or "0.0.0.0"
        return self.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                data, _addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            self.ingest(data)

    def ingest(self, data, now=None):
        """One UDP datagram of ASTERIX data blocks.  Tries the EKETA raw
        parser first (no dependency); falls back to asterix_decoder."""
        now = perf_counter() if now is None else now
        tracks = parse_eketa_block(data)
        if tracks:
            with self._lock:
                for t in tracks:
                    self._tracks[self.key(t)] = (now, t)
                self.packets += 1
                self.records += len(tracks)
                self.rx_t = now
                for k in [k for k, (t0, _) in self._tracks.items() if now - t0 > KEEP_S]:
                    del self._tracks[k]
            return len(tracks)
        if asterix is None:
            self.bad += 1
            return 0
        try:
            records = asterix.parse(data)
        except Exception:
            self.bad += 1
            return 0
        n = 0
        for rec in records:
            try:
                t = extract_track(rec)
            except (TypeError, ValueError, KeyError):
                t = None
            if t:
                n += 1
                with self._lock:
                    self._tracks[self.key(t)] = (now, t)
        with self._lock:
            self.packets += 1
            self.records += n
            self.rx_t = now
            for k in [k for k, (t0, _) in self._tracks.items() if now - t0 > KEEP_S]:
                del self._tracks[k]
        return n

    # ---- what the Seeker reads ---------------------------------------------

    def select(self, key):
        key = (key or "").strip().upper()
        with self._lock:
            if key and key not in self._tracks:
                return "no such track (it may have gone)"
            self.selected = key or None
        return ""

    def tracks(self, now, pos=None, home_amsl=None):
        with self._lock:
            items = list(self._tracks.items())
        out = []
        for k, (t0, t) in items:
            age = now - t0
            alt_m = None if t.get("alt_ft") is None else t["alt_ft"]*FT_TO_M
            rel = t.get("rel_m")
            if rel is None and alt_m is not None and home_amsl is not None:
                rel = alt_m - home_amsl
            out.append(dict(key=k, label=t.get("callsign") or k, icao=k, track_no=t.get("track_no"),
                            lat=t["lat"], lon=t["lon"], age_s=round(age, 1), fresh=age <= self.fresh_s,
                            source="RFD" if k == RFD_KEY else "UDP",
                            alt_amsl=None if alt_m is None else round(alt_m, 1),
                            alt_rel=None if rel is None else round(rel, 1),
                            speed=None if t.get("gs_kt") is None else round(t["gs_kt"]/KT_PER_MS, 1),
                            course=None if t.get("track_deg") is None else round(t["track_deg"]),
                            dist_m=None if pos is None else round(_dist(pos, (t["lat"], t["lon"]))),
                            bearing=None if pos is None else round(_bearing(pos, (t["lat"], t["lon"])))))
        out.sort(key=lambda x: (not x["fresh"], x["dist_m"] if x["dist_m"] is not None else 1e12, x["key"]))
        return out

    def target(self, now, pos=None, home_amsl=None):
        """The cue: the selected track, or the nearest fresh one; None if stale."""
        tr = self.tracks(now, pos, home_amsl)
        if self.selected:
            t = next((x for x in tr if x["key"] == self.selected), None)
        else:
            t = next((x for x in tr if x["fresh"]), None)
        return t if t is not None and t["fresh"] else None

    def status(self, now):
        with self._lock:
            n = len(self._tracks)
        return dict(port=self.port, mcast=self.mcast, iface=self.iface, error=self.error,
                    listening=self._thread is not None, packets=self.packets, records=self.records, bad=self.bad,
                    rx_age=None if self.rx_t is None else round(now - self.rx_t, 1), tracks=n,
                    selected=self.selected,
                    rfd=dict(dev=self.rfd_dev, baud=self.rfd_baud, open=self.rfd_open, error=self.rfd_error,
                             lines=self.rfd_lines, bad=self.rfd_bad,
                             rx_age=None if self.rfd_t is None else round(now - self.rfd_t, 1)))


# ---- CAT062 encoding, for the simulator and the tests ---------------------------

def encode_cat062(tracks, sac=0, sic=1):
    """One CAT062 data block: I010, I105, I185, I380/ADR, I040, I136 per track
    (dicts with icao, track_no, lat, lon, alt_ft, vx, vy in m/s)."""
    recs = b""
    for t in tracks:
        fspec = bytes((0x8B, 0x19, 0x20))
        lat = int(round(t["lat"] / (180.0/2**25)))
        lon = int(round(t["lon"] / (180.0/2**25)))
        body = struct.pack(">BB", sac, sic)
        body += struct.pack(">ii", lat, lon)
        body += struct.pack(">hh", int(round(t.get("vx", 0.0)/0.25)), int(round(t.get("vy", 0.0)/0.25)))
        icao = int(t["icao"]) & 0xFFFFFF
        body += bytes((0x80,)) + icao.to_bytes(3, "big")
        body += struct.pack(">H", int(t["track_no"]) & 0xFFFF)
        body += struct.pack(">h", int(round(t["alt_ft"]/100.0*4)))
        recs += fspec + body
    return struct.pack(">BH", 62, 3 + len(recs)) + recs


# ---- EKETA-EAB CAT062 with SP CH v1 (raw parser, no asterix_decoder needed) ----

EKETA_FSPEC = b"\x9B\x0D\x01\x01\x02"
_DEG25 = 180.0 / 2**25

_CLASSIF = {0: "not_evaluated", 1: "unknown", 2: "friendly"}


def parse_eketa_record(buf, off):
    """Parse one EKETA CAT062 record starting at *off* (just past the data-block
    header).  Returns (track_dict, next_offset) or (None, next_offset) on error.
    The track_dict is compatible with the existing extract_track() output."""
    start = off
    if len(buf) - off < len(EKETA_FSPEC):
        return None, len(buf)
    fspec = buf[off:off + len(EKETA_FSPEC)]
    off += len(EKETA_FSPEC)
    if fspec != EKETA_FSPEC:
        return None, off

    need = 2 + 3 + 8 + 4 + 2 + 1   # I010 + I070 + I105 + I185 + I040 + I080
    if len(buf) - off < need:
        return None, len(buf)

    sac, sic = buf[off], buf[off + 1]; off += 2
    tod = int.from_bytes(buf[off:off + 3], "big"); off += 3
    lat_raw, lon_raw = struct.unpack_from(">ii", buf, off); off += 8
    vx_raw, vy_raw = struct.unpack_from(">hh", buf, off); off += 4
    track_no = struct.unpack_from(">H", buf, off)[0]; off += 2
    status = buf[off]; off += 1

    lat = lat_raw * _DEG25
    lon = lon_raw * _DEG25
    vx = vx_raw * 0.25
    vy = vy_raw * 0.25
    tod_s = tod / 128.0

    t = dict(
        key=f"TRK{track_no}", icao=0xF00000 | (track_no & 0xFFFF),
        synthetic_icao=True, callsign=f"TRK{track_no}",
        track_no=track_no, sac=sac, sic=sic,
        lat=lat, lon=lon,
        gs_kt=math.hypot(vx, vy) * KT_PER_MS,
        track_deg=math.degrees(math.atan2(vx, vy)) % 360.0 if (vx or vy) else 0.0,
        vx=vx, vy=vy,
        tod_s=tod_s, status=status,
        alt_ft=None, roc_fpm=None, squawk=None,
    )

    # SP field
    if off < len(buf):
        sp_len = buf[off]
        sp_end = off + sp_len
        if sp_len >= 21 and sp_end <= len(buf) and buf[off + 1:off + 3] == b"CH":
            sp = buf[off:sp_end]
            ver = sp[3]
            classif_byte = sp[4]
            classif_code = (classif_byte >> 6) & 0x03
            theta = struct.unpack_from(">H", sp, 5)[0] * 0.01
            phi = struct.unpack_from(">h", sp, 7)[0] * 0.01
            R = struct.unpack_from(">I", sp, 9)[0] * 0.01
            hrel = struct.unpack_from(">i", sp, 13)[0] * 0.01
            h_msl = struct.unpack_from(">i", sp, 17)[0] * 0.01

            t["alt_ft"] = h_msl / FT_TO_M
            t["sp"] = dict(
                ver=ver, classif=_CLASSIF.get(classif_code, classif_code),
                azimuth_deg=theta, elevation_deg=phi,
                slant_range_m=R, hrel_m=hrel, h_msl_m=h_msl,
            )
        off = sp_end if sp_len > 0 else off + 1
    return t, off


def parse_eketa_block(data):
    """Parse one ASTERIX data block in EKETA CAT062 format.
    Returns a list of track dicts (may be empty)."""
    if len(data) < 3:
        return []
    cat = data[0]
    block_len = struct.unpack_from(">H", data, 1)[0]
    if cat != 62 or block_len > len(data):
        return []
    tracks = []
    off = 3
    while off < block_len:
        t, off = parse_eketa_record(data, off)
        if t is not None:
            tracks.append(t)
    return tracks


def encode_eketa_cat062(tracks, sac=0, sic=1, tod_s=None):
    """Encode tracks in EKETA CAT062 format (FSPEC 9B 0D 01 01 02 + SP CH v1).
    Each track dict: lat, lon, vx (m/s east), vy (m/s north), track_no,
    h_msl_m, and optionally azimuth_deg, elevation_deg, slant_range_m, hrel_m,
    classif ('not_evaluated'|'unknown'|'friendly')."""
    from datetime import datetime, timezone
    if tod_s is None:
        now = datetime.now(timezone.utc)
        tod_s = now.hour * 3600 + now.minute * 60 + now.second + now.microsecond / 1e6
    tod_raw = int(round(tod_s * 128)) & 0xFFFFFF

    recs = b""
    for t in tracks:
        body = EKETA_FSPEC
        body += struct.pack(">BB", sac & 0xFF, sic & 0xFF)
        body += tod_raw.to_bytes(3, "big")
        lat_raw = int(round(t["lat"] / _DEG25))
        lon_raw = int(round(t["lon"] / _DEG25))
        body += struct.pack(">ii", lat_raw, lon_raw)
        body += struct.pack(">hh", int(round(t.get("vx", 0.0) / 0.25)),
                                   int(round(t.get("vy", 0.0) / 0.25)))
        body += struct.pack(">H", int(t["track_no"]) & 0xFFFF)
        body += b"\x80"   # I062/080: MON=1

        # SP CH v1
        classif_str = t.get("classif", "unknown")
        classif_code = {"not_evaluated": 0, "unknown": 1, "friendly": 2}.get(classif_str, 1)
        theta = int(round(t.get("azimuth_deg", 0.0) / 0.01)) & 0xFFFF
        phi = int(round(t.get("elevation_deg", 0.0) / 0.01))
        R = int(round(t.get("slant_range_m", 0.0) / 0.01)) & 0xFFFFFFFF
        hrel = int(round(t.get("hrel_m", 0.0) / 0.01))
        h_msl = int(round(t["h_msl_m"] / 0.01))
        sp = struct.pack(">B2sB B HhI ii",
                         21, b"CH", 1,
                         (classif_code << 6) & 0xFF,
                         theta, phi, R, hrel, h_msl)
        body += sp
        recs += body
    return struct.pack(">BH", 62, 3 + len(recs)) + recs
