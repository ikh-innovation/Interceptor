#!/usr/bin/env python3
"""
asterix062_bridge.py - Feed ASTERIX CAT062 track data into the Interceptor stack.

    ASTERIX CAT062 (UDP / multicast)
            |
            v
      this bridge
            |
            +-->  SBS-1 / BaseStation text on TCP :30003    ---> Mission Planner map
            |
            +-->  (optional) MAVLink ADSB_VEHICLE messages  ---> autopilot / GCS
            |
            +-->  $RFD,lat,lon,alt_m,REL|MSL*HH on serial   ---> Jetson rfd_gps.RfdLink
                  (laptop's ground RFD radio --> airborne RFD radio on the Jetson)

The RFD leg is the one that hands coordinates to the Interceptor. Nothing on the
Jetson needs to change: rfd_gps.RfdLink already reads the same $RFD,...*HH frames
rfd_send.py emits, from whatever serial device RFD_DEV points at. This bridge
replaces rfd_send.py when the source of truth is an ASTERIX feed instead of a
manual lat/lon on the laptop.

Target selection (important): rfd_gps.RfdLink only keeps the NEWEST fix, so
blasting every aircraft would just overwrite itself. Pick exactly one track with
--follow-icao / --follow-callsign / --follow-track, or let the bridge pick the
closest to a reference point with --follow-nearest LAT,LON.

Altitude: ASTERIX I136 MFL is barometric MSL in feet; the bridge converts to
metres and sends with --rfd-frame MSL (default). Pick REL only if the Jetson's
home elevation matches the pressure-altitude reference.

Freshness: rfd_gps.RfdLink treats a fix as stale after 2 s. The bridge re-emits
the latest line at --rfd-hz (default 5 Hz) so brief gaps in the ASTERIX feed do
not flap the 'fresh' flag on the Jetson.

Mission Planner setup (once per PC, only if you also want the SBS leg):
    Tick Config -> Planner -> ADSB, or Mission Planner will not connect to
    the SBS server. ("Display ADSB" alone only draws aircraft on the map.)

Requirements:
    pip install asterix_decoder         # ASTERIX parser (Croatia Control)
    pip install pymavlink               # only needed for --mavlink
    pip install pyserial                # only needed for --rfd-dev

Examples:
    # Unicast UDP feed on port 8600, SBS-only (Mission Planner map)
    python asterix062_bridge.py --udp-port 8600

    # Multicast feed, received on a specific network interface
    python asterix062_bridge.py --udp-port 8600 --mcast 239.1.1.1 --iface 192.168.1.50

    # Also send MAVLink ADSB_VEHICLE messages
    python asterix062_bridge.py --udp-port 8600 --mavlink udpout:127.0.0.1:14550

    # Forward ONE target to the Jetson over the laptop's RFD radio
    python asterix062_bridge.py --udp-port 8600 --no-sbs \\
        --rfd-dev /dev/ttyUSB0 --rfd-baud 57600 --rfd-frame MSL \\
        --follow-callsign TARGET1

    # Follow a specific Mode S / synthetic ICAO address (hex)
    python asterix062_bridge.py --udp-port 8600 --rfd-dev /dev/ttyUSB0 \\
        --follow-icao 3C0A55

    # Follow whichever aircraft is closest to a reference point (10-s window)
    python asterix062_bridge.py --udp-port 8600 --rfd-dev /dev/ttyUSB0 \\
        --follow-nearest 37.9,23.7

    # Print the decoded records to see which fields your feed contains
    python asterix062_bridge.py --udp-port 8600 --dump

    # Test Mission Planner without any ASTERIX feed: one fake aircraft circling a point
    python asterix062_bridge.py --demo 37.9,23.7
"""

import argparse
import math
import pprint
import socket
import struct
import sys
import threading
import time
from datetime import datetime, timezone

KT_PER_MS = 1.943844
FT_TO_M = 0.3048


# --------------------------------------------------------------------------
# ASTERIX CAT062 -> simple track dictionary
# --------------------------------------------------------------------------

def find_val(node, keys):
    """Find the first 'val' for any of the given field names, searching nested
    compound items (asterix_decoder nests compound subfields one level deeper)."""
    if isinstance(node, dict):
        for k in keys:
            v = node.get(k)
            if isinstance(v, dict) and "val" in v:
                return v["val"]
        for v in node.values():
            if isinstance(v, dict):
                r = find_val(v, keys)
                if r is not None:
                    return r
    return None


def parse_icao(value):
    """ICAO 24-bit address from the decoder (usually a hex string like '3C0A55')."""
    if value is None:
        return None
    if isinstance(value, int):
        return value & 0xFFFFFF
    try:
        return int(str(value).strip(), 16) & 0xFFFFFF
    except ValueError:
        return None


def extract_track(rec):
    """Turn one decoded CAT062 record into a dict, or None if it has no position."""
    if rec.get("category") != 62:
        return None

    i105 = rec.get("I105")                       # position, WGS-84
    if not i105:
        return None
    lat = find_val(i105, ["Lat"])
    lon = find_val(i105, ["Lon"])
    if lat is None or lon is None:
        return None

    t = {"lat": float(lat), "lon": float(lon)}

    trk_no = find_val(rec.get("I040", {}), ["TrkN"])
    t["track_no"] = int(trk_no) if trk_no is not None else None

    # Altitude in feet. asterix_decoder already scales I136 MFL to feet
    # (FL157 is returned as 15700). Fall back to calculated barometric
    # altitude (I135) and then geometric altitude (I130).
    alt = find_val(rec.get("I136", {}), ["MFL"])
    if alt is None:
        alt = find_val(rec.get("I135", {}), ["CTBA"])
    if alt is None:
        alt = find_val(rec.get("I130", {}), ["Alt"])
    t["alt_ft"] = float(alt) if alt is not None else None

    # Velocity: Vx (east) and Vy (north) in m/s
    vx = find_val(rec.get("I185", {}), ["Vx"])
    vy = find_val(rec.get("I185", {}), ["Vy"])
    if vx is not None and vy is not None:
        vx, vy = float(vx), float(vy)
        t["gs_kt"] = math.hypot(vx, vy) * KT_PER_MS
        t["track_deg"] = math.degrees(math.atan2(vx, vy)) % 360.0
    else:
        t["gs_kt"] = t["track_deg"] = None

    roc = find_val(rec.get("I220", {}), ["RoC"])  # ft/min
    t["roc_fpm"] = float(roc) if roc is not None else None

    sq = find_val(rec.get("I060", {}), ["Mode3A"])  # octal string, e.g. '4276'
    t["squawk"] = str(sq).strip().zfill(4) if sq not in (None, "") else None

    # Mode S address and callsign: aircraft-derived data first, then
    # target identification (I245), then flight plan callsign (I390).
    t["icao"] = parse_icao(find_val(rec.get("I380", {}), ["ADR"]))
    cs = (find_val(rec.get("I380", {}), ["ACID"])
          or find_val(rec.get("I245", {}), ["TId"])
          or find_val(rec.get("I390", {}), ["CS"]))
    t["callsign"] = str(cs).strip() if cs else None

    # Tracks without a Mode S address (e.g. primary-radar only) get a
    # synthetic address in 0xF0xxxx built from the track number.
    t["synthetic_icao"] = False
    if t["icao"] is None:
        if t["track_no"] is None:
            return None
        t["icao"] = 0xF00000 | (t["track_no"] & 0xFFFF)
        t["synthetic_icao"] = True
        if not t["callsign"]:
            t["callsign"] = f"TRK{t['track_no']}"

    return t


# --------------------------------------------------------------------------
# SBS-1 (BaseStation) output over TCP
# --------------------------------------------------------------------------

def sbs_lines(t):
    now = datetime.now(timezone.utc)
    d = now.strftime("%Y/%m/%d")
    tm = now.strftime("%H:%M:%S.%f")[:-3]
    hexid = f"{t['icao']:06X}"

    def msg(tt, cs="", alt="", gs="", trk="", lat="", lon="", vr="", sq=""):
        f = ["MSG", str(tt), "1", "1", hexid, "1", d, tm, d, tm,
             cs, alt, gs, trk, lat, lon, vr, sq, "0", "0", "0", "0"]
        return ",".join(f) + "\r\n"

    alt = str(int(round(t["alt_ft"]))) if t["alt_ft"] is not None else ""
    out = []
    if t["callsign"]:
        out.append(msg(1, cs=t["callsign"][:8]))
    out.append(msg(3, alt=alt, lat=f"{t['lat']:.6f}", lon=f"{t['lon']:.6f}"))
    if t["gs_kt"] is not None:
        vr = str(int(round(t["roc_fpm"]))) if t["roc_fpm"] is not None else ""
        out.append(msg(4, gs=f"{t['gs_kt']:.1f}", trk=f"{t['track_deg']:.1f}", vr=vr))
    if t["squawk"]:
        out.append(msg(6, alt=alt, sq=t["squawk"]))
    return "".join(out)


class SbsServer:
    """Tiny TCP server; every connected client receives every SBS line."""

    def __init__(self, host, port):
        self.clients = []
        self.lock = threading.Lock()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(5)
        threading.Thread(target=self._accept_loop, daemon=True).start()
        print(f"[SBS] Listening on {host}:{port}")

    def _accept_loop(self):
        while True:
            conn, addr = self.sock.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[SBS] Client connected: {addr[0]}:{addr[1]}")
            with self.lock:
                self.clients.append(conn)

    def send(self, text):
        data = text.encode("ascii", "replace")
        with self.lock:
            for c in list(self.clients):
                try:
                    c.sendall(data)
                except OSError:
                    print("[SBS] Client disconnected")
                    self.clients.remove(c)
                    c.close()


# --------------------------------------------------------------------------
# Optional MAVLink ADSB_VEHICLE output
# --------------------------------------------------------------------------

class MavlinkSender:
    def __init__(self, url, sysid):
        from pymavlink import mavutil
        self.m = mavutil.mavlink
        comp = getattr(self.m, "MAV_COMP_ID_ADSB", 156)
        self.conn = mavutil.mavlink_connection(url, source_system=sysid,
                                               source_component=comp)
        self.last_hb = 0.0
        print(f"[MAVLink] Sending ADSB_VEHICLE to {url} (sysid {sysid})")

    def _heartbeat(self):
        now = time.time()
        if now - self.last_hb >= 1.0:
            mav_type = getattr(self.m, "MAV_TYPE_ADSB", 27)
            self.conn.mav.heartbeat_send(mav_type, self.m.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            self.last_hb = now

    def send(self, t):
        self._heartbeat()
        m = self.m
        flags = m.ADSB_FLAGS_VALID_COORDS
        alt_mm = heading = hor = ver = squawk = 0

        if t["alt_ft"] is not None:
            alt_mm = int(t["alt_ft"] * FT_TO_M * 1000)
            flags |= m.ADSB_FLAGS_VALID_ALTITUDE
        if t["gs_kt"] is not None:
            heading = int(t["track_deg"] * 100) % 36000
            hor = int(t["gs_kt"] / KT_PER_MS * 100)            # cm/s
            flags |= m.ADSB_FLAGS_VALID_HEADING | m.ADSB_FLAGS_VALID_VELOCITY
        if t["roc_fpm"] is not None:
            ver = int(t["roc_fpm"] * FT_TO_M / 60 * 100)       # cm/s, up positive
        callsign = (t["callsign"] or "")[:8]
        if callsign:
            flags |= m.ADSB_FLAGS_VALID_CALLSIGN
        if t["squawk"] and t["squawk"].isdigit():
            squawk = int(t["squawk"])
            flags |= m.ADSB_FLAGS_VALID_SQUAWK

        self.conn.mav.adsb_vehicle_send(
            t["icao"],
            int(t["lat"] * 1e7), int(t["lon"] * 1e7),
            m.ADSB_ALTITUDE_TYPE_PRESSURE_QNH, alt_mm,
            heading, hor, ver,
            callsign.encode("ascii", "replace"),
            m.ADSB_EMITTER_TYPE_NO_INFO,
            1,                       # time since last communication, s
            flags, squawk)


# --------------------------------------------------------------------------
# RFD radio output: $RFD,...*HH on serial, read by rfd_gps.RfdLink on the Jetson
# --------------------------------------------------------------------------

class RfdSender:
    """Open the laptop's RFD serial port and resend the latest fix at a steady
    rate. One track at a time, because rfd_gps.RfdLink only keeps the newest
    line it receives."""

    def __init__(self, dev, baud=57600, frame="MSL", hz=5.0):
        try:
            import serial
        except ImportError:
            sys.exit("pyserial is not installed: pip install pyserial")
        from rfd_gps import format_line
        self._format = format_line
        self.frame = frame
        self.period = 1.0 / max(0.5, float(hz))
        self.port = serial.Serial(dev, baud, timeout=0.2)
        self._lock = threading.Lock()
        self._latest = None  # (lat, lon, alt_m)
        self._stop = threading.Event()
        threading.Thread(target=self._run, name="rfd-tx", daemon=True).start()
        print(f"[RFD] Sending $RFD,...*HH on {dev} @ {baud} "
              f"({frame} frame, {hz:.1f} Hz)")

    def update(self, lat, lon, alt_m):
        if alt_m is None or not math.isfinite(float(alt_m)):
            return
        with self._lock:
            self._latest = (float(lat), float(lon), float(alt_m))

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                latest = self._latest
            if latest is not None:
                try:
                    line = self._format(latest[0], latest[1], latest[2],
                                        frame=self.frame)
                    self.port.write((line + "\r\n").encode("ascii"))
                    self.port.flush()
                except OSError as exc:
                    print(f"[RFD] write error: {exc}")
                except ValueError as exc:
                    print(f"[RFD] format error: {exc}")
            time.sleep(self.period)


def pick_target(t, args, state):
    """Return the track to forward to the RFD leg, or None.

    One of the --follow-* filters narrows to a single aircraft; without any
    filter every track passes through (last one wins at the Jetson, which is
    usually not what you want -- always set one of --follow-* in practice)."""
    if args.follow_icao is not None:
        return t if (t.get("icao") or 0) & 0xFFFFFF == args.follow_icao else None
    if args.follow_callsign:
        want = args.follow_callsign.strip().upper()
        cs = (t.get("callsign") or "").strip().upper()
        return t if cs == want else None
    if args.follow_track is not None:
        return t if t.get("track_no") == args.follow_track else None
    if state.get("nearest_center") is not None:
        lat0, lon0 = state["nearest_center"]
        now = time.time()
        state["recent"][t["icao"]] = (t, now)
        for k in list(state["recent"]):
            if now - state["recent"][k][1] > 10.0:
                del state["recent"][k]
        if not state["recent"]:
            return None
        return min(state["recent"].values(),
                   key=lambda v: math.hypot(v[0]["lat"] - lat0,
                                            v[0]["lon"] - lon0))[0]
    return t


# --------------------------------------------------------------------------
# Input: UDP ASTERIX feed, or demo generator
# --------------------------------------------------------------------------

def open_udp(port, mcast=None, iface="0.0.0.0"):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    s.bind(("", port))
    if mcast:
        mreq = struct.pack("4s4s", socket.inet_aton(mcast), socket.inet_aton(iface))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        print(f"[UDP] Joined multicast {mcast} on interface {iface}, port {port}")
    else:
        print(f"[UDP] Listening on port {port}")
    return s


def asterix_tracks(args):
    try:
        import asterix
    except ImportError:
        sys.exit("asterix_decoder is not installed: pip install asterix_decoder")

    sock = open_udp(args.udp_port, args.mcast, args.iface)
    while True:
        data, addr = sock.recvfrom(65535)
        try:
            records = asterix.parse(data)
        except Exception as e:  # malformed or unsupported data block
            print(f"[ASTERIX] Parse error from {addr[0]}: {e}")
            continue
        for rec in records:
            if args.dump:
                pprint.pprint(rec, width=120)
            t = extract_track(rec)
            if t:
                yield t


def demo_tracks(center):
    lat0, lon0 = (float(x) for x in center.split(","))
    radius_deg, period_s = 0.05, 120.0
    print(f"[DEMO] Fake aircraft circling {lat0},{lon0}")
    start = time.time()
    while True:
        a = 2 * math.pi * ((time.time() - start) / period_s)
        yield {
            "lat": lat0 + radius_deg * math.cos(a),
            "lon": lon0 + radius_deg * math.sin(a) / math.cos(math.radians(lat0)),
            "alt_ft": 3000.0, "gs_kt": 140.0,
            "track_deg": (math.degrees(a) + 90) % 360,
            "roc_fpm": 0.0, "squawk": "7000", "track_no": 1,
            "icao": 0xABCDEF, "callsign": "DEMO01", "synthetic_icao": False,
        }
        time.sleep(1.0)


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="ASTERIX CAT062 -> SBS-1 / MAVLink bridge")
    p.add_argument("--udp-port", type=int, help="UDP port of the ASTERIX feed")
    p.add_argument("--mcast", help="multicast group to join (e.g. 239.1.1.1)")
    p.add_argument("--iface", default="0.0.0.0",
                   help="local interface IP for multicast (default: any)")
    p.add_argument("--sbs-host", default="127.0.0.1",
                   help="address for the SBS server (0.0.0.0 to allow other PCs)")
    p.add_argument("--sbs-port", type=int, default=30003)
    p.add_argument("--no-sbs", action="store_true", help="disable SBS output")
    p.add_argument("--mavlink", help="pymavlink URL, e.g. udpout:127.0.0.1:14550")
    p.add_argument("--mav-sysid", type=int, default=250)
    p.add_argument("--rfd-dev",
                   help="serial device of the laptop's RFD radio (e.g. /dev/ttyUSB0); "
                        "writes $RFD,...*HH lines the Jetson's rfd_gps.RfdLink reads")
    p.add_argument("--rfd-baud", type=int, default=57600)
    p.add_argument("--rfd-frame", default="MSL", choices=("REL", "MSL"),
                   help="altitude frame to send; MSL matches ASTERIX I136 MFL (default)")
    p.add_argument("--rfd-hz", type=float, default=5.0,
                   help="rate the latest fix is re-emitted on the RFD port (default 5)")
    p.add_argument("--follow-icao",
                   help="forward only this Mode S / synthetic ICAO address (hex)")
    p.add_argument("--follow-callsign", help="forward only this callsign (exact match)")
    p.add_argument("--follow-track", type=int,
                   help="forward only this ASTERIX track number")
    p.add_argument("--follow-nearest", metavar="LAT,LON",
                   help="forward whichever track is closest to LAT,LON (10-s window)")
    p.add_argument("--dump", action="store_true", help="print decoded ASTERIX records")
    p.add_argument("--demo", metavar="LAT,LON",
                   help="no ASTERIX input; generate a fake aircraft around LAT,LON")
    p.add_argument("--quiet", action="store_true", help="don't print each track")
    args = p.parse_args()

    if not args.demo and not args.udp_port:
        p.error("give --udp-port for an ASTERIX feed, or --demo LAT,LON for a test")

    if args.follow_icao:
        try:
            args.follow_icao = int(args.follow_icao, 16) & 0xFFFFFF
        except ValueError:
            p.error("--follow-icao expects a hex address, e.g. 3C0A55")
    else:
        args.follow_icao = None

    state = {"recent": {}}
    if args.follow_nearest:
        try:
            lat0, lon0 = (float(x) for x in args.follow_nearest.split(","))
        except ValueError:
            p.error("--follow-nearest expects LAT,LON")
        state["nearest_center"] = (lat0, lon0)

    sbs = None if args.no_sbs else SbsServer(args.sbs_host, args.sbs_port)
    mav = MavlinkSender(args.mavlink, args.mav_sysid) if args.mavlink else None
    rfd = (RfdSender(args.rfd_dev, args.rfd_baud, args.rfd_frame, args.rfd_hz)
           if args.rfd_dev else None)
    if not sbs and not mav and not rfd:
        p.error("nothing to output: enable SBS, give --mavlink, or give --rfd-dev")

    source = demo_tracks(args.demo) if args.demo else asterix_tracks(args)
    try:
        for t in source:
            if sbs:
                sbs.send(sbs_lines(t))
            if mav:
                mav.send(t)
            if rfd:
                picked = pick_target(t, args, state)
                if picked is not None and picked.get("alt_ft") is not None:
                    rfd.update(picked["lat"], picked["lon"],
                               picked["alt_ft"] * FT_TO_M)
            if not args.quiet:
                alt = f"{t['alt_ft']:.0f}ft" if t["alt_ft"] is not None else "?"
                gs = f"{t['gs_kt']:.0f}kt" if t["gs_kt"] is not None else "?"
                tag = " (synthetic)" if t["synthetic_icao"] else ""
                print(f"{t['icao']:06X}{tag} {t['callsign'] or '':8} "
                      f"{t['lat']:.5f},{t['lon']:.5f} {alt} {gs}")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()