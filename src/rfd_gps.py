"""Live target position from the laptop RFD radio.

The laptop sends one line per fix, with an NMEA checksum::

    $RFD,<lat>,<lon>,<alt_m>,<REL|MSL>*HH

HH is the XOR of the characters between ``$`` and ``*``. REL is metres above
the same home the flight controller uses. MSL is metres above mean sea level.
The Jetson keeps the newest valid line (``RFD_DEV``, baud ``RFD_BAUD``).
"""
from __future__ import annotations

import math
import os
import threading
from time import perf_counter

EARTH_R = 6378137.0


def checksum(body):
    n = 0
    for ch in body:
        n ^= ord(ch)
    return f"{n:02X}"


def format_line(lat, lon, alt_m, frame="REL"):
    frame = str(frame).strip().upper()
    if frame not in ("REL", "MSL"):
        raise ValueError("frame: REL or MSL")
    lat, lon, alt_m = float(lat), float(lon), float(alt_m)
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 and math.isfinite(alt_m)):
        raise ValueError("latitude, longitude or height out of range")
    body = f"RFD,{lat:.7f},{lon:.7f},{alt_m:.2f},{frame}"
    return f"${body}*{checksum(body)}"


def parse_line(line):
    line = str(line).strip()
    if not line.startswith("$") or "*" not in line:
        return None
    body, _, cs = line[1:].partition("*")
    cs = cs.strip().upper()
    if len(cs) < 2 or checksum(body) != cs[:2]:
        return None
    if not body.startswith("RFD,"):
        return None
    parts = body.split(",")
    if len(parts) < 5:
        return None
    try:
        lat, lon, alt = float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return None
    frame = parts[4].strip().upper()
    if frame not in ("REL", "MSL"):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 and math.isfinite(alt)):
        return None
    other = [tok for tok in parts[5:] if "=" not in tok]
    return dict(lat=lat, lon=lon, alt_m=alt, frame=frame, other=other)


def ned_m(lat0, lon0, lat1, lon1):
    north = math.radians(lat1 - lat0) * EARTH_R
    east = math.radians(lon1 - lon0) * EARTH_R * math.cos(math.radians(lat0))
    return north, east


class RfdLink:
    """Newest fix from a transparent serial radio. ``ingest`` is for tests."""

    def __init__(self, dev, baud=57600, fresh_s=2.0):
        self.dev = dev
        self.baud = int(baud)
        self.fresh_s = float(fresh_s)
        self.error = ""
        self._lock = threading.Lock()
        self._fix = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rfd-gps", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def ingest(self, line, now=None):
        fix = parse_line(line)
        if fix is None:
            return False
        fix["t"] = perf_counter() if now is None else float(now)
        with self._lock:
            self._fix = fix
            self.error = ""
        return True

    def snapshot(self, now=None):
        now = perf_counter() if now is None else float(now)
        with self._lock:
            fix = None if self._fix is None else dict(self._fix)
        if fix is None:
            return None
        fix["age_s"] = max(0.0, now - fix["t"])
        fix["fresh"] = fix["age_s"] <= self.fresh_s
        return fix

    def _run(self):
        try:
            import serial
        except ImportError:
            self.error = "pyserial is not installed"
            return
        try:
            port = serial.Serial(self.dev, self.baud, timeout=0.5)
        except Exception as exc:
            self.error = str(exc)
            return
        buf = b""
        try:
            while not self._stop.is_set():
                chunk = port.read(256)
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    self.ingest(raw.decode("ascii", "ignore"))
        except Exception as exc:
            self.error = str(exc)
        finally:
            try:
                port.close()
            except Exception:
                pass


def start_from_env():
    dev = os.environ.get("RFD_DEV", "").strip()
    if not dev:
        return None
    try:
        baud = int(os.environ.get("RFD_BAUD", "57600") or 57600)
    except ValueError:
        baud = 57600
    link = RfdLink(dev, baud)
    link.start()
    return link
