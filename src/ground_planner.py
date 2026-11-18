#!/usr/bin/env python3
"""
ground_planner.py — Laptop-side ground station for ASTERIX testing and operations.

Two tabs:
  1. TEST  — Circle / Line / Draw path on the map, emits fake ASTERIX to Jetson
  2. ELKAK — Listens for real ASTERIX CAT062 from EKETA-EAB + manual target input

Via UDP (Jetson on same network):
    python ground_planner.py --jetson 192.168.205.160

Via RFD radio (ADSB_VEHICLE through FC):
    python ground_planner.py --mavlink /dev/ttyUSB0

Then open http://localhost:9000 in a browser.
"""
import argparse
import json
import math
import os
import socket
import struct
import sys
import threading
import time
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(__file__) or ".")
from asterix_cue import encode_eketa_cat062, parse_eketa_block
from asterix062_bridge import FT_TO_M, KT_PER_MS

# ---- state ------------------------------------------------------------------

STATE = {
    "mode": "test",           # "test" | "elkak"
    "targets": [],            # test targets
    "elkak_tracks": [],       # live tracks from EKETA
    "elkak_packets": 0,
    "elkak_last_rx": 0,
    "elkak_manual": None,     # {lat, lon, alt, enabled}
    "ref": None,
    "hz": 2.0,
}
LOCK = threading.Lock()
JETSON_IP = "192.168.205.160"
JETSON_HTTP = 8080
JETSON_AST = 8600
LISTEN_PORT = 8600
SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_PKT_COUNT = [0]
_MAV = None
_MAV_STATUS = {"fc_hb": 0, "fc_last": 0, "tx": 0, "tx_last": 0, "linked": False}
_MAV_STATUS_LOCK = threading.Lock()


def _mav_rx_loop():
    """Read MAVLink from the RFD to detect FC heartbeats (proves the radio link)."""
    while True:
        try:
            msg = _MAV.conn.recv_match(blocking=True, timeout=1)
        except Exception:
            time.sleep(0.5)
            continue
        if msg is None:
            continue
        t = msg.get_type()
        if t == "HEARTBEAT" and msg.get_srcSystem() != 250:
            with _MAV_STATUS_LOCK:
                first = _MAV_STATUS["fc_hb"] == 0
                _MAV_STATUS["fc_hb"] += 1
                _MAV_STATUS["fc_last"] = time.time()
                _MAV_STATUS["linked"] = True
            if first:
                print(f"[RFD] FC heartbeat received — radio link is ALIVE (sysid {msg.get_srcSystem()})")


def _mav_send_tracks(tracks):
    """Send tracks as ADSB_VEHICLE via the MAVLink/RFD link."""
    if _MAV is None:
        return
    sent = 0
    for t in tracks:
        alt_m = t.get("h_msl_m", 50)
        vx, vy = t.get("vx", 0), t.get("vy", 0)
        gs = math.hypot(vx, vy)
        hdg = math.degrees(math.atan2(vx, vy)) % 360.0 if gs > 0.01 else 0.0
        tno = int(t.get("track_no", 1)) & 0xFFFF
        mav_t = dict(icao=0xF00000 | tno, callsign=f"TRK{tno}",
                     lat=t["lat"], lon=t["lon"],
                     alt_ft=alt_m / FT_TO_M, gs_kt=gs * KT_PER_MS,
                     track_deg=hdg, roc_fpm=None, squawk=None,
                     synthetic_icao=True)
        try:
            _MAV.send(mav_t)
            sent += 1
        except Exception as exc:
            print(f"[RFD] send error: {exc}")
    if sent:
        with _MAV_STATUS_LOCK:
            first_tx = _MAV_STATUS["tx"] == 0
            _MAV_STATUS["tx"] += sent
            _MAV_STATUS["tx_last"] = time.time()
        if first_tx:
            print(f"[RFD] First ADSB_VEHICLE sent ({sent} track(s))")
    with _MAV_STATUS_LOCK:
        linked = _MAV_STATUS["linked"]
    if sent and not linked:
        if _MAV_STATUS["tx"] % 10 == 1:
            print("[RFD] WARNING: sending but no FC heartbeat received — check radio link")


def _polar(lat0, lon0, h0, lat, lon, h_msl):
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0) * math.cos(math.radians(lat0))
    n = dlat * 6371000.0
    e = dlon * 6371000.0
    d = math.hypot(n, e)
    hr = h_msl - h0
    R = math.hypot(d, hr)
    th = math.degrees(math.atan2(e, n)) % 360.0
    ph = math.degrees(math.atan2(hr, d)) if R > 0.01 else 0.0
    return dict(azimuth_deg=th, elevation_deg=ph, slant_range_m=R, hrel_m=hr)


def _target_pos(t, elapsed):
    if t.get("type") == "circle":
        r = t.get("radius", 200)
        spd = max(0.1, t.get("speed", 10))
        period = (2 * math.pi * r) / spd
        a = 2 * math.pi * (elapsed / max(0.01, period))
        lat0, lon0 = t["lat"], t["lon"]
        dlat = r * math.cos(a) / 111320.0
        dlon = r * math.sin(a) / (111320.0 * math.cos(math.radians(lat0)))
        w = 2 * math.pi / max(0.01, period)
        vx = r * w * math.cos(a)
        vy = -r * w * math.sin(a)
        return lat0 + dlat, lon0 + dlon, t.get("alt", 50), vx, vy
    elif t.get("type") == "line":
        d = math.hypot((t.get("lat2", 0) - t["lat"]) * 111320,
                       (t.get("lon2", 0) - t["lon"]) * 111320 * math.cos(math.radians(t["lat"])))
        spd = t.get("speed", 10)
        period = (d / max(0.1, spd)) * 2 if d > 0.1 else 1.0
        frac_t = (elapsed % period) / period
        frac = 2 * frac_t if frac_t < 0.5 else 2 * (1 - frac_t)
        lat = t["lat"] + frac * (t.get("lat2", 0) - t["lat"])
        lon = t["lon"] + frac * (t.get("lon2", 0) - t["lon"])
        brg = math.atan2((t.get("lon2", 0) - t["lon"]) * math.cos(math.radians(t["lat"])),
                         t.get("lat2", 0) - t["lat"])
        sign = 1.0 if frac_t < 0.5 else -1.0
        return lat, lon, t.get("alt", 50), sign * spd * math.sin(brg), sign * spd * math.cos(brg)
    elif t.get("type") == "path":
        wps = t.get("waypoints", [])
        if not wps or len(wps) < 2:
            return t.get("lat", 0), t.get("lon", 0), t.get("alt", 50), 0.0, 0.0
        total = 0
        segs = []
        for i in range(len(wps) - 1):
            p1, p2 = wps[i], wps[i + 1]
            dx = (p2[0] - p1[0]) * 111320
            dy = (p2[1] - p1[1]) * 111320 * math.cos(math.radians(p1[0]))
            sl = math.hypot(dx, dy)
            segs.append(sl)
            total += sl
        if total < 0.1:
            return wps[0][0], wps[0][1], t.get("alt", 50), 0.0, 0.0
        spd = max(0.1, t.get("speed", 10))
        period = (total / spd) * 2
        frac_t = (elapsed % period) / period
        frac = 2 * frac_t if frac_t < 0.5 else 2 * (1 - frac_t)
        dist = frac * total
        cum = 0
        for i, sl in enumerate(segs):
            if cum + sl >= dist:
                sf = (dist - cum) / max(0.001, sl)
                p1, p2 = wps[i], wps[i + 1]
                lat = p1[0] + sf * (p2[0] - p1[0])
                lon = p1[1] + sf * (p2[1] - p1[1])
                brg = math.atan2((p2[1] - p1[1]) * math.cos(math.radians(p1[0])), p2[0] - p1[0])
                sign = 1.0 if frac_t < 0.5 else -1.0
                return lat, lon, t.get("alt", 50), sign * spd * math.sin(brg), sign * spd * math.cos(brg)
            cum += sl
        return wps[-1][0], wps[-1][1], t.get("alt", 50), 0.0, 0.0
    else:
        return t["lat"], t["lon"], t.get("alt", 50), 0.0, 0.0


# ---- ELKAK listener ---------------------------------------------------------

def _elkak_listener():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", LISTEN_PORT))
    except OSError as exc:
        print(f"[ELKAK] Cannot bind UDP :{LISTEN_PORT}: {exc}")
        return
    print(f"[ELKAK] Listening for ASTERIX CAT062 on UDP :{LISTEN_PORT}")
    s.settimeout(1.0)
    while True:
        try:
            data, addr = s.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            return
        tracks = parse_eketa_block(data)
        if not tracks:
            continue
        now = time.time()
        with LOCK:
            STATE["elkak_packets"] += 1
            STATE["elkak_last_rx"] = now
            existing = {t["key"]: t for t in STATE["elkak_tracks"]}
            for t in tracks:
                existing[t.get("key", f"TRK{t.get('track_no', 0)}")] = dict(
                    key=t.get("key"), label=t.get("callsign", t.get("key", "")),
                    track_no=t.get("track_no", 0),
                    lat=t["lat"], lon=t["lon"],
                    alt=t.get("sp", {}).get("h_msl_m", t.get("alt_ft", 0) * 0.3048 if t.get("alt_ft") else 50),
                    vx=t.get("vx", 0), vy=t.get("vy", 0),
                    speed=math.hypot(t.get("vx", 0), t.get("vy", 0)),
                    classif=t.get("sp", {}).get("classif", "unknown"),
                    azimuth=t.get("sp", {}).get("azimuth_deg"),
                    elevation=t.get("sp", {}).get("elevation_deg"),
                    slant_range=t.get("sp", {}).get("slant_range_m"),
                    rx_t=now,
                )
            STATE["elkak_tracks"] = [v for v in existing.values() if now - v.get("rx_t", 0) < 15]
        try:
            SOCK.sendto(data, (JETSON_IP, JETSON_AST))
        except Exception:
            pass
        mav_tracks = []
        for t in tracks:
            alt_m = t.get("sp", {}).get("h_msl_m") or (t.get("alt_ft", 0) * FT_TO_M if t.get("alt_ft") else 50)
            mav_tracks.append(dict(lat=t["lat"], lon=t["lon"], h_msl_m=alt_m,
                                   vx=t.get("vx", 0), vy=t.get("vy", 0),
                                   track_no=t.get("track_no", 0)))
        _mav_send_tracks(mav_tracks)


# ---- emitter ----------------------------------------------------------------

def _emitter_loop():
    start = time.time()
    while True:
        with LOCK:
            mode = STATE["mode"]
            tgt_list = [dict(t) for t in STATE["targets"]]
            ref = STATE["ref"]
            hz = STATE["hz"]
            elkak_man = STATE.get("elkak_manual")
        if mode == "elkak":
            if elkak_man and elkak_man.get("enabled"):
                lat, lon, alt = elkak_man["lat"], elkak_man["lon"], elkak_man.get("alt", 50)
                if ref is None:
                    ref = (lat, lon, alt)
                    with LOCK:
                        STATE["ref"] = ref
                polar = _polar(ref[0], ref[1], ref[2], lat, lon, alt)
                tracks = [dict(lat=lat, lon=lon, h_msl_m=alt, vx=0, vy=0,
                               track_no=9999, classif="unknown", **polar)]
                data = encode_eketa_cat062(tracks)
                try:
                    SOCK.sendto(data, (JETSON_IP, JETSON_AST))
                except Exception:
                    pass
                _mav_send_tracks(tracks)
                _PKT_COUNT[0] += 1
            time.sleep(1.0 / hz)
            continue
        elapsed = time.time() - start
        tracks = []
        for t in tgt_list:
            if not t.get("enabled") or (t.get("lat", 0) == 0 and t.get("lon", 0) == 0):
                continue
            lat, lon, alt, vx, vy = _target_pos(t, elapsed)
            if ref is None:
                ref = (lat, lon, alt)
                with LOCK:
                    STATE["ref"] = ref
            polar = _polar(ref[0], ref[1], ref[2], lat, lon, alt)
            tracks.append(dict(lat=lat, lon=lon, h_msl_m=alt, vx=vx, vy=vy,
                               track_no=t.get("id", 1), classif="unknown", **polar))
        if tracks:
            data = encode_eketa_cat062(tracks)
            try:
                SOCK.sendto(data, (JETSON_IP, JETSON_AST))
            except Exception:
                pass
            _mav_send_tracks(tracks)
            _PKT_COUNT[0] += 1
        time.sleep(1.0 / hz)


# ---- HTML page --------------------------------------------------------------

HTML_PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ground Planner</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root{--bg:#ffffff;--panel:#f8f9fb;--text:#1e293b;--muted:#64748b;--accent:#2563eb;
  --ok:#16a34a;--warn:#d97706;--err:#dc2626;--border:#e2e8f0;--op:#7c3aed}
*{box-sizing:border-box;margin:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:var(--bg);color:var(--text);display:flex;flex-direction:column;height:100vh}
header{padding:10px 16px;display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--border);
  background:var(--panel)}
header h1{font-size:16px;letter-spacing:.3px;font-weight:700}
.status{margin-left:auto;font-size:12px;color:var(--muted)}
.status b{color:var(--ok);font-weight:600}
.status b.off{color:var(--err)}
.main{flex:1;display:flex;min-height:0}
.map-wrap{flex:1;min-width:0;position:relative}
#map{width:100%;height:100%;background:#e5e7eb}
.toolbar{position:absolute;top:10px;left:56px;z-index:1000;display:flex;gap:6px}
.toolbar button{padding:8px 14px;border:none;border-radius:8px;font:600 13px/1 sans-serif;
  cursor:pointer;background:#fff;color:var(--text);border:1px solid var(--border);
  box-shadow:0 1px 3px rgba(0,0,0,.1);transition:all .15s}
.toolbar button:hover{border-color:var(--accent);box-shadow:0 2px 6px rgba(37,99,235,.15)}
.toolbar button.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.draw-hint{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);z-index:1000;
  background:rgba(30,41,59,.9);color:#fff;padding:8px 16px;border-radius:8px;
  font:600 13px/1 sans-serif;pointer-events:none;display:none}
.draw-panel{position:absolute;bottom:16px;left:16px;z-index:1000;
  background:#fff;border:1px solid var(--border);border-radius:12px;
  padding:14px 16px;min-width:260px;max-width:300px;box-shadow:0 4px 20px rgba(0,0,0,.12)}
.dp-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.dp-header span{font:700 14px/1 sans-serif}
.dp-close{background:none;border:none;font-size:18px;color:var(--muted);cursor:pointer;padding:0 2px}
.dp-close:hover{color:var(--err)}
.dprow{display:flex;gap:6px;align-items:center;margin:6px 0}
.dprow label{font-size:12px;color:var(--muted);min-width:54px}
.dprow input{flex:1;background:var(--panel);border:1px solid var(--border);color:var(--text);
  border-radius:6px;padding:5px 8px;font-size:12px}
.dprow .unit{font-size:11px;color:var(--muted);min-width:24px}
.dp-status{font-size:11px;color:var(--muted);margin:8px 0;line-height:1.4}
.dp-status b{color:var(--text)}
.dp-actions{display:flex;gap:8px;margin-top:10px}
.dp-actions button{flex:1;padding:9px;border:none;border-radius:8px;font:600 12px/1 sans-serif;
  cursor:pointer;transition:all .15s}
.dp-cancel{background:var(--panel);color:var(--muted);border:1px solid var(--border)!important}
.dp-cancel:hover{color:var(--err);border-color:var(--err)!important}
.dp-send{background:var(--accent);color:#fff}
.dp-send:hover:not(:disabled){opacity:.85}
.dp-send:disabled{opacity:.35;cursor:not-allowed}
.sidebar{width:340px;overflow-y:auto;padding:0;display:flex;flex-direction:column;
  border-left:1px solid var(--border);background:var(--panel)}
.mode-tabs{display:flex;border-bottom:1px solid var(--border);background:#fff}
.mode-tabs button{flex:1;padding:12px 4px;border:none;background:transparent;color:var(--muted);
  font:700 13px/1 sans-serif;cursor:pointer;border-bottom:3px solid transparent;transition:all .15s;
  letter-spacing:.3px}
.mode-tabs button.active{color:var(--accent);border-bottom-color:var(--accent)}
.mode-tabs button.op-active{color:var(--op);border-bottom-color:var(--op)}
.mode-tabs .badge{font-size:9px;padding:2px 5px;border-radius:3px;margin-left:5px;
  vertical-align:middle;font-weight:700}
.mode-tabs .test-badge{background:#dbeafe;color:var(--accent)}
.mode-tabs .op-badge{background:#ede9fe;color:var(--op)}
.mode-content{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px}
.card{padding:10px 12px;border-radius:10px;border:1px solid var(--border);background:#fff}
.card h3{font-size:13px;margin:0 0 6px;display:flex;align-items:center;gap:8px}
.card h3 .dot{width:10px;height:10px;border-radius:50%;flex:0 0 auto}
.card h3 .spacer{flex:1}
.card .del-btn{background:none;border:none;color:var(--err);cursor:pointer;font-size:16px;
  padding:0 4px;opacity:.5}
.card .del-btn:hover{opacity:1}
.card .info{font-size:11px;color:var(--muted);line-height:1.5}
.row{display:flex;gap:6px;align-items:center;margin:3px 0}
.row label{font-size:11px;color:var(--muted);min-width:46px}
.row input,.row select{flex:1;background:#fff;border:1px solid var(--border);color:var(--text);
  border-radius:6px;padding:4px 6px;font-size:12px}
.row .unit{font-size:11px;color:var(--muted);min-width:24px}
.drone-info{font-size:11px;color:var(--muted);line-height:1.5}
.drone-info b{color:var(--text)}
.empty-hint{font-size:12px;color:var(--muted);text-align:center;padding:16px 8px;line-height:1.6}
.add-btn{width:100%;padding:10px;border:1px dashed var(--border);border-radius:8px;
  background:transparent;color:var(--accent);font:600 13px/1 sans-serif;cursor:pointer}
.add-btn:hover{border-color:var(--accent);background:#f0f7ff}
/* ELKAK */
.elkak-status{padding:12px;border-radius:10px;border:1px solid var(--border);background:#fff;
  font-size:12px;line-height:1.6}
.elkak-status b{color:var(--op)}
.elkak-track{padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:#fff;
  font-size:11px;line-height:1.5}
.elkak-track .tk-label{font-weight:700;font-size:12px}
.elkak-track .tk-classif{font-size:10px;padding:1px 5px;border-radius:3px;margin-left:6px}
.elkak-track .tk-classif.unknown{background:#fef3c7;color:#92400e}
.elkak-track .tk-classif.friendly{background:#dcfce7;color:#166534}
.elkak-track .tk-classif.not_evaluated{background:#f1f5f9;color:#475569}
.elkak-manual{padding:12px;border-radius:10px;border:1px solid var(--border);background:#fff}
.elkak-manual h3{font-size:13px;margin:0 0 8px;font-weight:700}
.elkak-manual .mrow{display:flex;gap:6px;align-items:center;margin:4px 0}
.elkak-manual .mrow label{font-size:11px;color:var(--muted);min-width:36px}
.elkak-manual .mrow input{flex:1;background:#fff;border:1px solid var(--border);color:var(--text);
  border-radius:6px;padding:5px 8px;font-size:12px}
.elkak-manual .mrow .unit{font-size:11px;color:var(--muted);min-width:24px}
.send-btn{width:100%;padding:10px;border:none;border-radius:8px;background:var(--op);color:#fff;
  font:700 13px/1 sans-serif;cursor:pointer;margin-top:8px;transition:all .15s}
.send-btn:hover{opacity:.85}
.send-btn:disabled{opacity:.35;cursor:not-allowed}
.stop-btn{width:100%;padding:10px;border:none;border-radius:8px;background:var(--err);color:#fff;
  font:700 13px/1 sans-serif;cursor:pointer;margin-top:6px;transition:all .15s}
.stop-btn:hover{opacity:.85}
/* modal overlay */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.3);z-index:2000;display:flex;
  align-items:center;justify-content:center}
.modal{background:#fff;border:1px solid var(--border);border-radius:14px;padding:20px 24px;
  min-width:320px;max-width:400px;box-shadow:0 8px 30px rgba(0,0,0,.12)}
.modal h2{font-size:15px;margin:0 0 14px;display:flex;align-items:center;gap:8px}
.modal h2 .mbadge{font-size:10px;padding:2px 6px;border-radius:4px;font-weight:700}
.modal h2 .mbadge.circle{background:#dbeafe;color:var(--accent)}
.modal h2 .mbadge.line{background:#fef3c7;color:#92400e}
.modal h2 .mbadge.path{background:#dcfce7;color:#166534}
.modal .mrow{display:flex;gap:8px;align-items:center;margin:8px 0}
.modal .mrow label{font-size:12px;color:var(--muted);min-width:56px}
.modal .mrow input{flex:1;background:var(--panel);border:1px solid var(--border);color:var(--text);
  border-radius:6px;padding:6px 8px;font-size:13px}
.modal .mrow .unit{font-size:11px;color:var(--muted);min-width:30px}
.modal .mrow-info{font-size:11px;color:var(--muted);margin:2px 0 2px 64px}
.modal .btn-row{display:flex;gap:10px;margin-top:18px}
.modal .btn-row button{flex:1;padding:10px;border:none;border-radius:8px;font:600 13px/1 sans-serif;
  cursor:pointer;transition:all .15s}
.modal .btn-send{background:var(--accent);color:#fff}
.modal .btn-send:hover{opacity:.85}
.modal .btn-send:disabled{opacity:.35;cursor:not-allowed}
.modal .btn-cancel{background:var(--panel);color:var(--err);border:1px solid var(--border)}
.modal .btn-cancel:hover{border-color:var(--err)}
/* SEEK button */
.seek-btn{position:absolute;bottom:20px;right:20px;z-index:1000;width:80px;height:80px;
  border-radius:50%;border:3px solid var(--border);background:#fff;color:var(--muted);
  font:700 14px/1 sans-serif;cursor:not-allowed;transition:all .3s;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:2px;box-shadow:0 4px 16px rgba(0,0,0,.1);
  user-select:none;opacity:.7}
.seek-btn .seek-icon{font-size:22px}
.seek-btn.ready{background:#dcfce7;border-color:var(--ok);color:var(--ok);cursor:pointer;
  opacity:1;animation:seekPulse 2s ease-in-out infinite}
.seek-btn.ready:hover{background:#bbf7d0;transform:scale(1.05)}
.seek-btn.active{background:#fee2e2;border-color:var(--err);color:var(--err);cursor:pointer;
  opacity:1;animation:none}
.seek-btn.active:hover{background:#fecaca}
@keyframes seekPulse{0%,100%{box-shadow:0 0 8px rgba(22,163,74,.2)}
  50%{box-shadow:0 0 24px rgba(22,163,74,.4)}}
/* glowing target blob */
.tgt-blob{width:20px;height:20px;border-radius:50%;position:relative;
  transition:transform .05s linear}
.tgt-blob .core{width:20px;height:20px;border-radius:50%;position:relative;z-index:2}
.tgt-blob .glow{position:absolute;inset:-8px;border-radius:50%;opacity:.35;
  animation:blobPulse 1.2s ease-in-out infinite}
.tgt-blob .ring{position:absolute;inset:-4px;border-radius:50%;border:2px solid;opacity:.5;
  animation:blobRing 1.2s ease-in-out infinite}
@keyframes blobPulse{0%,100%{transform:scale(1);opacity:.35}50%{transform:scale(1.6);opacity:.1}}
@keyframes blobRing{0%,100%{transform:scale(1);opacity:.5}50%{transform:scale(1.3);opacity:.2}}
.seek-panel{padding:0 12px 12px;display:flex;flex-direction:column;gap:6px;
  border-top:1px solid var(--border)}
.seek-panel h3{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;
  margin:8px 0 2px;cursor:pointer;user-select:none}
.seek-panel h3:hover{color:var(--accent)}
.fparam{display:flex;gap:6px;align-items:center;margin:2px 0}
.fparam label{font-size:11px;color:var(--muted);min-width:70px}
.fparam input{flex:1;background:var(--panel);border:1px solid var(--border);color:var(--text);
  border-radius:6px;padding:3px 6px;font-size:12px;max-width:80px}
.fparam input:disabled{opacity:.4}
.fparam .unit{font-size:10px;color:var(--muted);min-width:28px}
.msn-box{padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:#fff;
  font-size:11px;line-height:1.6}
.msn-box b{color:var(--text)}
.msn-box .c-ok{color:var(--ok)}.msn-box .c-warn{color:var(--warn)}.msn-box .c-err{color:var(--err)}
.offer-box{padding:10px 12px;border-radius:8px;background:#eff6ff;border:1px solid var(--accent);margin:4px 0}
.offer-box .offer-text{font-size:12px;margin:0 0 8px;line-height:1.4}
.offer-btns{display:flex;gap:8px}
.offer-btns button{flex:1;padding:8px;border:none;border-radius:6px;font:600 12px/1 sans-serif;cursor:pointer}
.offer-yes{background:var(--ok);color:#fff}
.offer-no{background:var(--err);color:#fff}
@media(max-width:700px){
  .main{flex-direction:column}
  .sidebar{width:100%;max-height:50vh;border-left:none;border-top:1px solid var(--border)}
  .map-wrap{min-height:40vh}
  .modal{min-width:280px;max-width:90vw}
}
</style></head><body>
<header>
  <h1>Ground Planner</h1>
  <span class="status">Drone: <b id="drone_st" class="off">no GPS</b></span>
  <span class="status">RFD: <b id="rfd_st" class="off">off</b></span>
  <span class="status">TX: <b id="tx_st">idle</b> <span id="tx_n"></span></span>
</header>
<div class="main">
  <div class="map-wrap">
    <div id="map"></div>
    <div class="toolbar" id="draw_toolbar">
      <button id="draw_circle">&#9711; Circle</button>
      <button id="draw_line">&#9586; Line</button>
      <button id="draw_path">&#9998; Draw</button>
    </div>
    <div class="draw-hint" id="draw_hint"></div>
    <div class="draw-panel" id="draw_panel" style="display:none">
      <div class="dp-header"><span id="dp_title"></span><button class="dp-close" id="dp_close">&times;</button></div>
      <div id="dp_fields"></div>
      <div id="dp_status" class="dp-status"></div>
      <div class="dp-actions">
        <button class="dp-cancel" id="dp_cancel">Cancel</button>
        <button class="dp-send" id="dp_send" disabled>Send to Jetson</button>
      </div>
    </div>
    <button class="seek-btn" id="seek_btn" disabled><span class="seek-icon">&#9678;</span><span id="seek_label">SEEK</span></button>
  </div>
  <div class="sidebar">
    <div class="mode-tabs">
      <button id="tab_test" class="active" data-mode="test">TEST<span class="badge test-badge">SIM</span></button>
      <button id="tab_elkak" data-mode="elkak">ELKAK<span class="badge op-badge">LIVE</span></button>
    </div>
    <div class="mode-content" id="mode_content">
      <div class="card drone-info" id="drone_card">
        <b>Drone</b><br>
        Lat: <span id="d_lat">--</span> Lon: <span id="d_lon">--</span><br>
        Alt: <span id="d_alt">--</span> m &nbsp; Mode: <span id="d_mode">--</span>
        &nbsp; Armed: <span id="d_armed">--</span>
      </div>
      <div id="target_list"></div>
      <div id="mode_hint"></div>
    </div>
    <div id="seek_panel" class="seek-panel" style="display:none">
      <h3 id="fp_toggle">&#9656; SEEKER FLIGHT PARAMETERS</h3>
      <div id="fp_body" style="display:none"></div>
      <div id="msn_status"></div>
    </div>
  </div>
</div>
<div id="modal_root"></div>

<script>
var JETSON = "JETSON_PLACEHOLDER";
var map = L.map("map", {zoomControl:true}).setView([37.9,23.7],16);
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
  maxZoom:19, attribution:"Esri"
}).addTo(map);

var COLORS = ["#2563eb","#d97706","#dc2626","#7c3aed","#16a34a","#ea580c","#0891b2","#c026d3"];
var droneMarker = null, centered = false;
var targets = [], elkakMarkers = {};
var animStart = Date.now();
var currentMode = "test";
var modalOpen = false;
var drawState = {mode:null, center:null, start:null, end:null, waypoints:[]};
var drawTempLayers = [];
var elkakManualActive = false;

function hint(msg){ var el=document.getElementById("draw_hint"); el.textContent=msg||""; el.style.display=msg?"":"none"; }
function tgtColor(i){ return COLORS[i%COLORS.length]; }
function droneIcon(){
  return L.divIcon({className:"",html:'<svg width="28" height="28" viewBox="-14 -14 28 28"><circle r="10" fill="#16a34a" stroke="#fff" stroke-width="2"/><circle r="3" fill="#fff"/></svg>',iconSize:[28,28],iconAnchor:[14,14]});
}
function tgtIcon(color){
  return L.divIcon({className:"",html:'<svg width="20" height="20" viewBox="-10 -10 20 20"><polygon points="0,-8 6,6 -6,6" fill="'+color+'" stroke="#fff" stroke-width="1.5"/></svg>',iconSize:[20,20],iconAnchor:[10,10]});
}
function elkakIcon(){
  return L.divIcon({className:"",html:'<svg width="24" height="24" viewBox="-12 -12 24 24"><polygon points="0,-10 7,7 -7,7" fill="#7c3aed" stroke="#fff" stroke-width="1.5"/></svg>',iconSize:[24,24],iconAnchor:[12,12]});
}
function blobIcon(color){
  return L.divIcon({className:"",
    html:'<div class="tgt-blob"><div class="glow" style="background:'+color+'"></div><div class="ring" style="border-color:'+color+'"></div><div class="core" style="background:'+color+';box-shadow:0 0 8px '+color+'"></div></div>',
    iconSize:[20,20],iconAnchor:[10,10]});
}

// ---- modal system ----
function showModal(type, geo, onConfirm){
  modalOpen = true;
  var root = document.getElementById("modal_root");
  var fields = "";
  if(type === "circle"){
    fields = '<div class="mrow"><label>Radius</label><input id="m_radius" type="number" value="200" min="10" step="10"><span class="unit">m</span></div>'
      +'<div class="mrow"><label>Speed</label><input id="m_speed" type="number" value="10" min="0.1" step="1"><span class="unit">m/s</span></div>'
      +'<div class="mrow-info">Orbit time = 2&pi; &times; radius / speed</div>'
      +'<div class="mrow"><label>Altitude</label><input id="m_alt" type="number" value="50" min="0" step="5"><span class="unit">m</span></div>';
  } else if(type === "line"){
    fields = '<div class="mrow"><label>Speed</label><input id="m_speed" type="number" value="10" min="0.1" step="1"><span class="unit">m/s</span></div>'
      +'<div class="mrow"><label>Altitude</label><input id="m_alt" type="number" value="50" min="0" step="5"><span class="unit">m</span></div>';
  } else if(type === "path"){
    var n = (geo.waypoints||[]).length;
    fields = '<div class="mrow-info">Path with <b>'+n+'</b> waypoints</div>'
      +'<div class="mrow"><label>Speed</label><input id="m_speed" type="number" value="10" min="0.1" step="1"><span class="unit">m/s</span></div>'
      +'<div class="mrow"><label>Altitude</label><input id="m_alt" type="number" value="50" min="0" step="5"><span class="unit">m</span></div>';
  }
  var badge = '<span class="mbadge '+type+'">'+type.toUpperCase()+'</span>';
  var titles = {circle:"New Circle Target", line:"New Line Target", path:"New Path Target"};
  root.innerHTML = '<div class="modal-overlay" id="modal_overlay">'
    +'<div class="modal">'
    +'<h2>'+badge+' '+(titles[type]||"New Target")+'</h2>'
    + fields
    +'<div class="btn-row">'
    +'<button class="btn-cancel" id="m_cancel">Cancel</button>'
    +'<button class="btn-send" id="m_send">Send to Jetson</button>'
    +'</div>'
    +'</div></div>';
  document.getElementById("m_cancel").onclick = function(){ closeModal(); };
  document.getElementById("m_send").onclick = function(){
    var params = {alt: parseFloat(document.getElementById("m_alt").value)||50};
    if(type === "circle"){
      params.radius = parseFloat(document.getElementById("m_radius").value);
      params.speed = parseFloat(document.getElementById("m_speed").value);
      if(!params.radius || params.radius <= 0 || !params.speed || params.speed <= 0) return;
    } else if(type === "line" || type === "path"){
      params.speed = parseFloat(document.getElementById("m_speed").value);
      if(!params.speed || params.speed <= 0) return;
    }
    if(type === "path") params.waypoints = geo.waypoints;
    closeModal();
    onConfirm(params);
  };
  document.getElementById("modal_overlay").onclick = function(e){
    if(e.target === this) closeModal();
  };
  setTimeout(function(){
    var first = root.querySelector(".modal input");
    if(first) first.focus();
  }, 50);
}

function closeModal(){
  modalOpen = false;
  document.getElementById("modal_root").innerHTML = "";
}

// ---- mode switching ----
document.querySelectorAll(".mode-tabs button").forEach(function(btn){
  btn.onclick = function(){
    currentMode = btn.dataset.mode;
    document.querySelectorAll(".mode-tabs button").forEach(function(b){ b.className = ""; });
    btn.className = currentMode === "elkak" ? "op-active" : "active";
    document.getElementById("draw_toolbar").style.display = currentMode === "test" ? "" : "none";
    hideDrawPanel();
    fetch("/api/mode",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({mode:currentMode})});
    renderMode();
  };
});

function renderMode(){
  var list = document.getElementById("target_list");
  var mh = document.getElementById("mode_hint");
  list.innerHTML = ""; mh.innerHTML = "";
  clearElkakMarkers();
  if(currentMode === "test") renderTestMode(list, mh);
  else renderElkakMode(list, mh);
}

// ---- TEST tab ----
function renderTestMode(list, mh){
  if(!targets.length) mh.innerHTML = '<div class="empty-hint">Use the map toolbar to add targets:<br><b>Circle</b> &mdash; orbiting target<br><b>Line</b> &mdash; back-and-forth between two points<br><b>Draw</b> &mdash; click waypoints to draw a path</div>';
  targets.forEach(function(t, idx){
    var c = tgtColor(idx), card = document.createElement("div");
    card.className = "card";
    var typeLabel = t.type === "circle" ? "Circle" : t.type === "line" ? "Line" : t.type === "path" ? "Path" : "Static";
    var details = "";
    if(t.type === "circle"){
      var period = (2*Math.PI*t.radius / Math.max(0.1,t.speed)).toFixed(1);
      details = 'Radius: '+t.radius+' m &nbsp; Speed: '+t.speed+' m/s &nbsp; Orbit: '+period+' s<br>';
    } else if(t.type === "line"){
      details = 'Speed: '+t.speed+' m/s<br>';
    } else if(t.type === "path"){
      details = (t.waypoints?t.waypoints.length:0)+' waypoints &nbsp; Speed: '+t.speed+' m/s<br>';
    }
    details += 'Alt: '+t.alt+' m';
    card.innerHTML = '<h3><span class="dot" style="background:'+c+'"></span>T'+t.id+' '+typeLabel
      +'<span class="spacer"></span>'
      +'<button class="del-btn" data-delidx="'+idx+'" title="Remove target">&times;</button></h3>'
      +'<div class="info">'+details+'</div>';
    list.appendChild(card);
  });
  list.querySelectorAll("[data-delidx]").forEach(function(btn){
    btn.addEventListener("click",function(){removeTarget(parseInt(btn.dataset.delidx));});
  });
}

// ---- ELKAK tab ----
var elkakPollTimer = null;
function renderElkakMode(list, mh){
  mh.innerHTML = '<div class="elkak-status"><b>ELKAK — Operational Mode</b><br>'
    +'Listening for ASTERIX CAT062 on UDP <b>:LISTEN_PORT_PH</b><br>'
    +'Packets are decoded and relayed to Jetson <b>JETSON_PLACEHOLDER:JETSON_AST_PH</b><br><br>'
    +'<span id="elkak_st">Waiting for data...</span></div>'
    +'<div class="elkak-manual" style="margin-top:8px">'
    +'<h3>Manual Target Input</h3>'
    +'<div style="font-size:11px;color:var(--muted);margin-bottom:6px">Enter target coordinates manually (e.g. from ELKAK verbal or text)</div>'
    +'<div class="mrow"><label>Lat</label><input id="elkak_lat" type="number" step="0.00001" placeholder="e.g. 37.80700"></div>'
    +'<div class="mrow"><label>Lon</label><input id="elkak_lon" type="number" step="0.00001" placeholder="e.g. 23.78500"></div>'
    +'<div class="mrow"><label>Alt</label><input id="elkak_alt" type="number" value="50" min="0" step="5"><span class="unit">m MSL</span></div>'
    +'<button class="send-btn" id="elkak_send">Send Target to Drone</button>'
    +'<button class="stop-btn" id="elkak_stop" style="display:none">Stop Manual Target</button>'
    +'</div>';
  mh.innerHTML = mh.innerHTML.replace(/LISTEN_PORT_PH/g, "LISTEN_PORT_VAL");
  mh.innerHTML = mh.innerHTML.replace(/JETSON_PLACEHOLDER/g, JETSON);
  mh.innerHTML = mh.innerHTML.replace(/JETSON_AST_PH/g, "JETSON_AST_VAL");
  document.getElementById("elkak_send").onclick = function(){
    var lat = parseFloat(document.getElementById("elkak_lat").value);
    var lon = parseFloat(document.getElementById("elkak_lon").value);
    var alt = parseFloat(document.getElementById("elkak_alt").value) || 50;
    if(!lat || !lon || lat < -90 || lat > 90 || lon < -180 || lon > 180){ alert("Invalid coordinates"); return; }
    fetch("/api/elkak-manual",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({lat:lat,lon:lon,alt:alt,enabled:true})})
      .then(function(r){return r.json();}).then(function(){
        elkakManualActive = true;
        document.getElementById("elkak_send").style.display = "none";
        document.getElementById("elkak_stop").style.display = "";
      });
  };
  document.getElementById("elkak_stop").onclick = function(){
    fetch("/api/elkak-manual",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({enabled:false})})
      .then(function(r){return r.json();}).then(function(){
        elkakManualActive = false;
        document.getElementById("elkak_send").style.display = "";
        document.getElementById("elkak_stop").style.display = "none";
      });
  };
  if(elkakManualActive){
    document.getElementById("elkak_send").style.display = "none";
    document.getElementById("elkak_stop").style.display = "";
  }
  pollElkak();
  if(elkakPollTimer) clearInterval(elkakPollTimer);
  elkakPollTimer = setInterval(pollElkak, 1000);
}

function pollElkak(){
  if(currentMode !== "elkak"){ if(elkakPollTimer){clearInterval(elkakPollTimer);elkakPollTimer=null;} return; }
  fetch("/api/elkak").then(function(r){return r.json();}).then(function(j){
    var el = document.getElementById("elkak_st");
    if(!el) return;
    var age = j.age !== null ? j.age.toFixed(1) + " s ago" : "never";
    el.innerHTML = "Packets: <b>"+j.packets+"</b> &nbsp; Last RX: <b>"+age+"</b> &nbsp; Tracks: <b>"+j.tracks.length+"</b>";
    if(j.manual_active && !elkakManualActive){
      elkakManualActive = true;
      var sb = document.getElementById("elkak_send"), stb = document.getElementById("elkak_stop");
      if(sb) sb.style.display = "none";
      if(stb) stb.style.display = "";
    }
    var list = document.getElementById("target_list");
    list.innerHTML = "";
    j.tracks.forEach(function(t){
      var card = document.createElement("div");
      card.className = "elkak-track";
      var cls = t.classif || "unknown";
      card.innerHTML = '<span class="tk-label">'+t.label+'</span>'
        +'<span class="tk-classif '+cls+'">'+cls+'</span><br>'
        +'Lat: '+t.lat.toFixed(6)+' Lon: '+t.lon.toFixed(6)+'<br>'
        +'Alt: '+(t.alt != null ? t.alt.toFixed(1)+' m' : '--')
        +' &nbsp; Speed: '+(t.speed != null ? t.speed.toFixed(1)+' m/s' : '--')
        +(t.azimuth != null ? '<br>Az: '+t.azimuth.toFixed(1)+'&deg; El: '+(t.elevation||0).toFixed(1)+'&deg; R: '+(t.slant_range||0).toFixed(0)+' m' : '');
      list.appendChild(card);
      updateElkakMarker(t);
    });
    var keys = {};
    j.tracks.forEach(function(t){keys[t.key]=true;});
    Object.keys(elkakMarkers).forEach(function(k){
      if(!keys[k]){elkakMarkers[k].remove(); delete elkakMarkers[k];}
    });
    document.getElementById("tx_n").textContent = j.packets + " rx";
    document.getElementById("tx_st").textContent = j.tracks.length > 0 ? "live" : "waiting";
    document.getElementById("tx_st").className = j.tracks.length > 0 ? "" : "off";
  }).catch(function(){});
}

function updateElkakMarker(t){
  if(!t.lat || !t.lon) return;
  if(elkakMarkers[t.key]){
    elkakMarkers[t.key].setLatLng([t.lat,t.lon]);
  } else {
    elkakMarkers[t.key] = L.marker([t.lat,t.lon],{icon:elkakIcon(),zIndexOffset:500}).addTo(map);
    elkakMarkers[t.key].bindTooltip(t.label,{permanent:true,direction:"right",className:"",offset:[12,0]});
  }
}

function clearElkakMarkers(){
  Object.keys(elkakMarkers).forEach(function(k){elkakMarkers[k].remove();});
  elkakMarkers = {};
}

// ---- target data model ----
function commitTarget(type, lat, lon, lat2, lon2, params){
  var id = targets.length+1;
  var t = {id:id, enabled:true, type:type, lat:lat, lon:lon, alt:params.alt||50,
    lat2:lat2||0, lon2:lon2||0, speed:params.speed||10, radius:params.radius||200, layers:{}};
  if(params.waypoints) t.waypoints = params.waypoints;
  targets.push(t);
  drawTargetLayers(targets.length-1);
  syncToServer();
  renderMode();
}
function removeTarget(idx){
  clearLayers(idx); targets.splice(idx,1);
  targets.forEach(function(t,i){t.id=i+1;});
  redrawAllLayers(); renderMode(); syncToServer();
}
function clearLayers(idx){
  var ly=targets[idx].layers;
  if(ly.m)ly.m.remove(); if(ly.m2)ly.m2.remove();
  if(ly.path)ly.path.remove(); if(ly.circle)ly.circle.remove();
  if(ly.anim)ly.anim.remove();
  if(ly.wps){ ly.wps.forEach(function(w){w.remove();}); }
  targets[idx].layers={};
}
function drawTargetLayers(idx){
  var t=targets[idx],c=tgtColor(idx),ly=t.layers;
  if(ly.m)ly.m.remove(); if(ly.m2)ly.m2.remove();
  if(ly.path)ly.path.remove(); if(ly.circle)ly.circle.remove();
  if(ly.wps){ ly.wps.forEach(function(w){w.remove();}); }
  if(!t.lat&&!t.lon&&!(t.waypoints&&t.waypoints.length))return;
  if(t.type==="path"&&t.waypoints&&t.waypoints.length>=2){
    ly.path=L.polyline(t.waypoints,{color:c,weight:2,dashArray:"6 4"}).addTo(map);
    ly.wps=t.waypoints.map(function(wp){
      return L.circleMarker(wp,{radius:3,color:c,fillColor:c,fillOpacity:1,weight:0}).addTo(map);
    });
  } else {
    if(t.lat||t.lon) ly.m=L.marker([t.lat,t.lon],{icon:tgtIcon(c),draggable:false}).addTo(map);
    if(t.type==="line"&&t.lat2&&t.lon2){
      ly.m2=L.marker([t.lat2,t.lon2],{icon:tgtIcon(c),draggable:false}).addTo(map);
      ly.path=L.polyline([[t.lat,t.lon],[t.lat2,t.lon2]],{color:c,weight:2,dashArray:"6 4"}).addTo(map);
    }
    if(t.type==="circle") ly.circle=L.circle([t.lat,t.lon],{radius:t.radius,color:c,weight:2,fill:false,dashArray:"6 4"}).addTo(map);
  }
}
function redrawAllLayers(){targets.forEach(function(_,i){drawTargetLayers(i);});}

function syncToServer(){
  var data=targets.map(function(t){
    var obj={id:t.id, enabled:t.enabled, type:t.type, lat:t.lat, lon:t.lon, alt:t.alt,
            lat2:t.lat2, lon2:t.lon2, speed:t.speed, radius:t.radius};
    if(t.waypoints) obj.waypoints=t.waypoints;
    return obj;
  });
  fetch("/api/targets",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({targets:data})});
}

// ---- drawing tools (panel-based) ----
function clearDrawTemp(){
  drawTempLayers.forEach(function(l){l.remove();}); drawTempLayers=[];
  drawState.center=null; drawState.start=null; drawState.end=null; drawState.waypoints=[];
}

function showDrawPanel(mode){
  drawState.mode=mode;
  clearDrawTemp();
  var panel=document.getElementById("draw_panel");
  var title=document.getElementById("dp_title");
  var fields=document.getElementById("dp_fields");
  var status=document.getElementById("dp_status");
  var send=document.getElementById("dp_send");
  panel.style.display="";
  send.disabled=true;
  if(mode==="circle"){
    title.textContent="New Circle Target";
    fields.innerHTML='<div class="dprow"><label>Radius</label><input id="dp_radius" type="number" value="200" min="10" step="10"><span class="unit">m</span></div>'
      +'<div class="dprow"><label>Speed</label><input id="dp_speed" type="number" value="10" min="0.1" step="1"><span class="unit">m/s</span></div>'
      +'<div class="dprow"><label>Altitude</label><input id="dp_alt" type="number" value="50" min="0" step="5"><span class="unit">m</span></div>';
    status.textContent="Click map to place circle center";
  } else if(mode==="line"){
    title.textContent="New Line Target";
    fields.innerHTML='<div class="dprow"><label>Speed</label><input id="dp_speed" type="number" value="10" min="0.1" step="1"><span class="unit">m/s</span></div>'
      +'<div class="dprow"><label>Altitude</label><input id="dp_alt" type="number" value="50" min="0" step="5"><span class="unit">m</span></div>';
    status.textContent="Click map for start point";
  } else if(mode==="path"){
    title.textContent="New Path Target";
    fields.innerHTML='<div class="dprow"><label>Speed</label><input id="dp_speed" type="number" value="10" min="0.1" step="1"><span class="unit">m/s</span></div>'
      +'<div class="dprow"><label>Altitude</label><input id="dp_alt" type="number" value="50" min="0" step="5"><span class="unit">m</span></div>';
    status.textContent="Click map to add waypoints (0 points)";
  }
}

function hideDrawPanel(){
  document.getElementById("draw_panel").style.display="none";
  drawState.mode=null;
  clearDrawTemp();
  document.querySelectorAll(".toolbar button").forEach(function(b){b.classList.remove("active");});
  hint("");
}

function updateDrawStatus(){
  var status=document.getElementById("dp_status");
  var send=document.getElementById("dp_send");
  if(drawState.mode==="circle"){
    if(drawState.center){
      status.innerHTML="Center: <b>"+drawState.center[0].toFixed(5)+", "+drawState.center[1].toFixed(5)+"</b>";
      send.disabled=false;
    }
  } else if(drawState.mode==="line"){
    if(drawState.start&&drawState.end){
      status.innerHTML="Start &rarr; End placed";
      send.disabled=false;
    } else if(drawState.start){
      status.innerHTML="Start: <b>"+drawState.start[0].toFixed(5)+"</b> &mdash; click for end";
      send.disabled=true;
    }
  } else if(drawState.mode==="path"){
    var n=drawState.waypoints.length;
    if(n>=2){
      status.innerHTML="<b>"+n+"</b> waypoints. Click more or send.";
      send.disabled=false;
    } else {
      status.textContent="Click map to add waypoints ("+n+" point"+(n!==1?"s":"")+")";
      send.disabled=true;
    }
  }
}

function sendDrawTarget(){
  var alt=parseFloat((document.getElementById("dp_alt")||{}).value)||50;
  var speed=parseFloat((document.getElementById("dp_speed")||{}).value)||10;
  var radius=parseFloat((document.getElementById("dp_radius")||{}).value)||200;
  if(drawState.mode==="circle"&&drawState.center){
    commitTarget("circle",drawState.center[0],drawState.center[1],0,0,{alt:alt,speed:speed,radius:radius});
  } else if(drawState.mode==="line"&&drawState.start&&drawState.end){
    commitTarget("line",drawState.start[0],drawState.start[1],drawState.end[0],drawState.end[1],{alt:alt,speed:speed});
  } else if(drawState.mode==="path"&&drawState.waypoints.length>=2){
    var wps=drawState.waypoints.slice();
    commitTarget("path",wps[0][0],wps[0][1],0,0,{alt:alt,speed:speed,waypoints:wps});
  } else return;
  clearDrawTemp();
  updateDrawStatus();
  hint("Sent! Click map to add another target.");
}

function setDrawMode(mode){
  var wasActive=drawState.mode===mode;
  hideDrawPanel();
  if(wasActive) return;
  document.querySelectorAll(".toolbar button").forEach(function(b){b.classList.remove("active");});
  document.getElementById("draw_"+mode).classList.add("active");
  showDrawPanel(mode);
}

document.getElementById("draw_circle").onclick=function(){ setDrawMode("circle"); };
document.getElementById("draw_line").onclick=function(){ setDrawMode("line"); };
document.getElementById("draw_path").onclick=function(){ setDrawMode("path"); };
document.getElementById("dp_close").onclick=function(){ hideDrawPanel(); };
document.getElementById("dp_cancel").onclick=function(){ hideDrawPanel(); };
document.getElementById("dp_send").onclick=function(){ sendDrawTarget(); };

map.on("click",function(e){
  if(modalOpen) return;
  if(currentMode!=="test"||!drawState.mode) return;
  if(drawState.mode==="circle"){
    clearDrawTemp();
    drawState.center=[e.latlng.lat,e.latlng.lng];
    var dot=L.circleMarker(e.latlng,{radius:5,color:"#2563eb",fillColor:"#2563eb",fillOpacity:1}).addTo(map);
    drawTempLayers.push(dot);
    var r=parseFloat((document.getElementById("dp_radius")||{}).value)||200;
    var circ=L.circle(e.latlng,{radius:r,color:"#2563eb",weight:2,fill:false,dashArray:"6 4"}).addTo(map);
    drawTempLayers.push(circ);
    updateDrawStatus();
  } else if(drawState.mode==="line"){
    if(!drawState.start){
      drawState.start=[e.latlng.lat,e.latlng.lng];
      var dot=L.circleMarker(e.latlng,{radius:5,color:"#d97706",fillColor:"#d97706",fillOpacity:1}).addTo(map);
      drawTempLayers.push(dot);
      updateDrawStatus();
    } else if(!drawState.end){
      drawState.end=[e.latlng.lat,e.latlng.lng];
      var dot=L.circleMarker(e.latlng,{radius:5,color:"#d97706",fillColor:"#d97706",fillOpacity:1}).addTo(map);
      drawTempLayers.push(dot);
      var line=L.polyline([drawState.start,drawState.end],{color:"#d97706",weight:2,dashArray:"6 4"}).addTo(map);
      drawTempLayers.push(line);
      updateDrawStatus();
    }
  } else if(drawState.mode==="path"){
    drawState.waypoints.push([e.latlng.lat,e.latlng.lng]);
    var dot=L.circleMarker(e.latlng,{radius:4,color:"#16a34a",fillColor:"#16a34a",fillOpacity:1,weight:0}).addTo(map);
    drawTempLayers.push(dot);
    if(drawState.waypoints.length>1){
      var wps=drawState.waypoints;
      var seg=L.polyline([wps[wps.length-2],wps[wps.length-1]],{color:"#16a34a",weight:2,dashArray:"4 4"}).addTo(map);
      drawTempLayers.push(seg);
    }
    updateDrawStatus();
  }
});

document.addEventListener("keydown",function(e){
  if(e.key==="Escape"){
    if(modalOpen){closeModal();return;}
    if(drawState.mode){hideDrawPanel();return;}
  }
});

// ---- drone polling ----
function pollDrone(){
  fetch("/drone/state").then(function(r){return r.ok?r.json():null;}).then(function(j){
    if(!j){document.getElementById("drone_st").className="off";document.getElementById("drone_st").textContent="offline";updateSeek(null);return;}
    document.getElementById("d_mode").textContent=j.fc_mode||"--";
    document.getElementById("d_armed").textContent=j.armed?"YES":"no";
    document.getElementById("drone_st").className=j.linked?"":"off";
    document.getElementById("drone_st").textContent=j.linked?"linked":"no FC";
    var m=j.mission||{},pos=m.position,alt=m.height_m;
    document.getElementById("d_alt").textContent=alt!=null?alt.toFixed(1):"--";
    if(pos&&pos.length===2&&pos[0]) updateDrone(pos[0],pos[1],alt);
    else if(!centered&&m.home&&m.home.length===2&&m.home[0]){map.setView(m.home,17);centered=true;}
    var sk=m.seeker||m,cue=(sk.cue||{});
    if(currentMode!=="elkak"){
      if(cue.packets!=null) document.getElementById("tx_n").textContent=cue.packets+" rx";
    }
    updateSeek(m.seeker||null, m);
  }).catch(function(){
    document.getElementById("drone_st").className="off";
    document.getElementById("drone_st").textContent="offline";
    updateSeek(null);
  });
}
function updateDrone(lat,lon,alt){
  if(!lat||!lon)return;
  document.getElementById("d_lat").textContent=lat.toFixed(6);
  document.getElementById("d_lon").textContent=lon.toFixed(6);
  if(alt!=null) document.getElementById("d_alt").textContent=alt.toFixed(1);
  if(!droneMarker){
    droneMarker=L.marker([lat,lon],{icon:droneIcon(),zIndexOffset:1000}).addTo(map);
    if(!centered){map.setView([lat,lon],17);centered=true;}
  } else droneMarker.setLatLng([lat,lon]);
}

// ---- animate moving targets ----
function animateTargets(){
  if(currentMode==="elkak"){requestAnimationFrame(animateTargets);return;}
  var elapsed=(Date.now()-animStart)/1000;
  targets.forEach(function(t,i){
    if(!t.enabled){if(t.layers.anim){t.layers.anim.remove();t.layers.anim=null;}return;}
    var lat,lon;
    if(t.type==="circle"){
      var period=2*Math.PI*t.radius/Math.max(0.1,t.speed);
      var a=2*Math.PI*(elapsed/Math.max(0.01,period));
      lat=t.lat+t.radius*Math.cos(a)/111320;
      lon=t.lon+t.radius*Math.sin(a)/(111320*Math.cos(t.lat*Math.PI/180));
    } else if(t.type==="line"&&t.lat2&&t.lon2){
      var d=Math.hypot((t.lat2-t.lat)*111320,(t.lon2-t.lon)*111320*Math.cos(t.lat*Math.PI/180));
      var period=d>0.1?(d/Math.max(0.1,t.speed))*2:1;
      var ft=(elapsed%period)/period;
      var frac=ft<0.5?2*ft:2*(1-ft);
      lat=t.lat+frac*(t.lat2-t.lat);lon=t.lon+frac*(t.lon2-t.lon);
    } else if(t.type==="path"&&t.waypoints&&t.waypoints.length>=2){
      var totalLen=0, segs=[];
      for(var k=0;k<t.waypoints.length-1;k++){
        var p1=t.waypoints[k],p2=t.waypoints[k+1];
        var dx=(p2[0]-p1[0])*111320, dy=(p2[1]-p1[1])*111320*Math.cos(p1[0]*Math.PI/180);
        var sl=Math.hypot(dx,dy); segs.push(sl); totalLen+=sl;
      }
      if(totalLen>0.1){
        var period2=(totalLen/Math.max(0.1,t.speed))*2;
        var ft2=(elapsed%period2)/period2;
        var frac2=ft2<0.5?2*ft2:2*(1-ft2);
        var dist=frac2*totalLen, cum=0;
        for(var k=0;k<segs.length;k++){
          if(cum+segs[k]>=dist){
            var sf=(dist-cum)/Math.max(0.001,segs[k]);
            lat=t.waypoints[k][0]+sf*(t.waypoints[k+1][0]-t.waypoints[k][0]);
            lon=t.waypoints[k][1]+sf*(t.waypoints[k+1][1]-t.waypoints[k][1]);
            break;
          }
          cum+=segs[k];
        }
      }
    }
    if(lat&&lon){
      var c=tgtColor(i);
      if(!t.layers.anim) t.layers.anim=L.marker([lat,lon],{icon:blobIcon(c),zIndexOffset:800}).addTo(map);
      else t.layers.anim.setLatLng([lat,lon]);
    } else {
      if(t.layers.anim){t.layers.anim.remove();t.layers.anim=null;}
    }
  });
  requestAnimationFrame(animateTargets);
}
requestAnimationFrame(animateTargets);

setInterval(pollDrone,2000); pollDrone();
setInterval(function(){
  fetch("/api/rfd").then(function(r){return r.json();}).then(function(j){
    var el=document.getElementById("rfd_st");
    if(!j.enabled){el.textContent="off";el.className="off";}
    else if(j.linked){el.textContent="LINKED ("+j.tx+" tx, "+j.fc_hb+" rx)";el.className="";}
    else if(j.tx>0){el.textContent="TX only (no FC reply)";el.className="off";}
    else{el.textContent="waiting";el.className="off";}
  }).catch(function(){});
},2000);
setInterval(function(){
  if(currentMode==="elkak")return;
  fetch("/api/status").then(function(r){return r.json();}).then(function(j){
    document.getElementById("tx_n").textContent=j.packets+" pkt";
    document.getElementById("tx_st").textContent=j.targets>0?"sending":"idle";
    document.getElementById("tx_st").className=j.targets>0?"":"off";
  }).catch(function(){});
},1000);

// ---- SEEKER integration ----
var seekReady=false, seekRunning=false, seekChecksOk=false, seekWhy="", seekLastOfferN=null, fpOpen=false;

function seekCmd(v){
  return fetch("/drone/cmd?key=msn&value="+encodeURIComponent(v))
    .then(function(r){return r.json();})
    .catch(function(){return {};});
}

document.getElementById("fp_toggle").onclick=function(){
  fpOpen=!fpOpen;
  document.getElementById("fp_body").style.display=fpOpen?"":"none";
  this.innerHTML=(fpOpen?"&#9662;":"&#9656;")+" SEEKER FLIGHT PARAMETERS";
};

document.getElementById("seek_btn").onclick=function(){
  if(seekRunning){
    seekCmd("abort");
  } else if(seekReady){
    modalOpen=true;
    var root=document.getElementById("modal_root");
    if(!seekChecksOk){
      root.innerHTML='<div class="modal-overlay" id="modal_overlay"><div class="modal">'
        +'<h2 style="color:var(--err)">&#9888; Cannot SEEK</h2>'
        +'<p style="font-size:13px;color:var(--muted);margin:0 0 16px;line-height:1.5">'
        +'Pre-flight checks failed:<br><b style="color:var(--err)">'+seekWhy+'</b></p>'
        +'<div class="btn-row">'
        +'<button class="btn-cancel" id="m_cancel" style="flex:1">OK</button>'
        +'</div></div></div>';
      document.getElementById("m_cancel").onclick=function(){closeModal();};
      document.getElementById("modal_overlay").onclick=function(e){if(e.target===this)closeModal();};
    } else {
      root.innerHTML='<div class="modal-overlay" id="modal_overlay"><div class="modal">'
        +'<h2>&#9678; Confirm SEEK</h2>'
        +'<p style="font-size:13px;color:var(--muted);margin:0 0 16px;line-height:1.5">'
        +'Arm the drone and fly to the ASTERIX target?<br>The drone will take off autonomously in <b>GUIDED</b> mode.</p>'
        +'<div class="btn-row">'
        +'<button class="btn-cancel" id="m_cancel">Cancel</button>'
        +'<button class="btn-send" id="m_send">&#9678; Start SEEK</button>'
        +'</div></div></div>';
      document.getElementById("m_cancel").onclick=function(){closeModal();};
      document.getElementById("m_send").onclick=function(){closeModal();seekCmd("start:SEEK");};
      document.getElementById("modal_overlay").onclick=function(e){if(e.target===this)closeModal();};
    }
  }
};

function updateSeek(sk, m){
  var panel=document.getElementById("seek_panel");
  var btn=document.getElementById("seek_btn");
  var lbl=document.getElementById("seek_label");
  if(!sk){
    panel.style.display="none";
    btn.className="seek-btn"; btn.disabled=true; lbl.textContent="SEEK";
    return;
  }
  panel.style.display="";
  var seek=sk.seek||{}, run=sk.run||null, cue=sk.cue||{};
  var hasTgt=cue.target!=null;
  seekRunning=run!=null && run.id==="SEEK";
  seekReady=!!(hasTgt && !run);
  seekChecksOk=!!seek.ready;
  seekWhy=seek.why||"";

  if(seekRunning){
    btn.className="seek-btn active"; btn.disabled=false; lbl.textContent="ABORT";
  } else if(seekReady){
    btn.className="seek-btn ready"; btn.disabled=false; lbl.textContent="SEEK";
  } else {
    btn.className="seek-btn"; btn.disabled=true; lbl.textContent="SEEK";
  }

  renderFP(seek.params||[], !!run);
  renderMsnStatus(run, cue, seek);
}

function renderFP(params, dis){
  var body=document.getElementById("fp_body");
  if(!params.length)return;
  var key=params.map(function(p){return p.k+":"+p.v;}).join(",");
  if(body.dataset.key===key && body.dataset.dis===String(dis))return;
  body.dataset.key=key; body.dataset.dis=String(dis);
  body.innerHTML="";
  params.forEach(function(p){
    var row=document.createElement("div"); row.className="fparam";
    row.innerHTML='<label>'+p.t+'</label>'
      +'<input type="number" value="'+p.v+'" step="'+(p.step||1)+'" min="'+(p.lo||0)+'" max="'+(p.hi||999)+'"'
      +(dis?' disabled':'')+' data-pk="'+p.k+'">'
      +'<span class="unit">'+p.u+'</span>';
    body.appendChild(row);
  });
  if(!dis){
    body.querySelectorAll("input").forEach(function(inp){
      inp.addEventListener("change",function(){
        seekCmd("set:SEEK:"+inp.dataset.pk+":"+inp.value);
      });
    });
  }
}

function renderMsnStatus(run, cue, seek){
  var el=document.getElementById("msn_status");
  if(!run){
    var tgt=cue.target;
    if(tgt){
      var info='<div class="msn-box"><span class="c-ok">&#9679;</span> Target: <b>'+tgt.label+'</b><br>'
        +'Lat: '+tgt.lat.toFixed(5)+' Lon: '+tgt.lon.toFixed(5)+'<br>';
      if(tgt.dist_m!=null) info+='Dist: '+tgt.dist_m+' m &nbsp; ';
      if(tgt.speed!=null) info+='Spd: '+tgt.speed.toFixed(1)+' m/s';
      if(seek.ready) info+='<br><span class="c-ok">&#10003; Ready to SEEK</span>';
      else info+='<br><span class="c-warn">&#9679; Target acquired</span>';
      el.innerHTML=info+'</div>';
    } else {
      var n=(cue.tracks||[]).length;
      if(n) el.innerHTML='<div class="msn-box"><span class="c-warn">&#9679;</span> '+n+' track(s), none fresh</div>';
      else el.innerHTML='<div class="msn-box"><span class="c-err">&#9679;</span> No ASTERIX targets</div>';
    }
    return;
  }
  var sc=run.state==="PILOT"||run.state==="ABORTED"?"c-err"
    :run.state==="AI"||run.state==="HOVER"||run.state==="DONE"?"c-ok":"c-warn";
  var h='<div class="msn-box"><b class="'+sc+'">'+run.state+'</b> &mdash; '+(run.why||"")+'<br>';
  if(run.alt!=null) h+='Alt: <b>'+run.alt+' m</b> &nbsp; ';
  if(run.gs!=null) h+='GS: <b>'+run.gs+' m/s</b> &nbsp; ';
  if(run.home_m!=null) h+='Home: <b>'+run.home_m+' m</b><br>';
  if(run.elapsed_s!=null) h+='Time: '+run.elapsed_s+'s &nbsp; ';
  if(run.battery_pct!=null) h+='Batt: '+run.battery_pct+'%';
  h+='</div>';
  el.innerHTML=h;
  if(run.offer){
    var on=run.offer.n;
    if(seekLastOfferN!==on){
      seekLastOfferN=on;
    }
    el.innerHTML+='<div class="offer-box"><div class="offer-text">'+run.offer.text+'</div>'
      +'<div class="offer-btns">'
      +'<button class="offer-no" onclick="seekCmd(\'offer:no:'+on+'\')">No</button>'
      +'<button class="offer-yes" onclick="seekCmd(\'offer:yes:'+on+'\')">Yes</button>'
      +'</div></div>';
  } else { seekLastOfferN=null; }
}

renderMode();
</script></body></html>"""


# ---- HTTP handler -----------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            page = HTML_PAGE.replace("JETSON_PLACEHOLDER", JETSON_IP)
            page = page.replace("LISTEN_PORT_VAL", str(LISTEN_PORT))
            page = page.replace("JETSON_AST_VAL", str(JETSON_AST))
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/drone/state":
            self._proxy_json(f"http://{JETSON_IP}:{JETSON_HTTP}/state")
        elif path == "/drone/cmd":
            qs = urlparse(self.path).query
            self._proxy_json(f"http://{JETSON_IP}:{JETSON_HTTP}/cmd?{qs}")
        elif path == "/api/rfd":
            with _MAV_STATUS_LOCK:
                st = dict(_MAV_STATUS)
            st["enabled"] = _MAV is not None
            self._json(st)
        elif path == "/api/status":
            with LOCK:
                ntgt = len([t for t in STATE["targets"] if t.get("enabled") and (t.get("lat") or t.get("lon"))])
            self._json({"targets": ntgt, "packets": _PKT_COUNT[0]})
        elif path == "/api/elkak":
            with LOCK:
                tracks = list(STATE["elkak_tracks"])
                pkts = STATE["elkak_packets"]
                last = STATE["elkak_last_rx"]
                man = STATE.get("elkak_manual")
            age = round(time.time() - last, 1) if last > 0 else None
            self._json({"packets": pkts, "age": age, "tracks": tracks,
                        "manual_active": bool(man and man.get("enabled"))})
        else:
            self.send_error(404)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/targets":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length)) if length else {}
            incoming = data.get("targets", [])
            with LOCK:
                STATE["targets"] = incoming
            self._json({"ok": True})
        elif path == "/api/mode":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length)) if length else {}
            mode = data.get("mode", "test")
            with LOCK:
                STATE["mode"] = mode
            self._json({"ok": True, "mode": mode})
        elif path == "/api/elkak-manual":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length)) if length else {}
            with LOCK:
                if data.get("enabled", False):
                    STATE["elkak_manual"] = {
                        "lat": data.get("lat", 0),
                        "lon": data.get("lon", 0),
                        "alt": data.get("alt", 50),
                        "enabled": True,
                    }
                else:
                    STATE["elkak_manual"] = None
            self._json({"ok": True})
        else:
            self.send_error(404)

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy_json(self, url):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as r:
                body = r.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self._json({"error": str(e)})


def main():
    global JETSON_IP, JETSON_HTTP, JETSON_AST, LISTEN_PORT
    p = argparse.ArgumentParser(description="Ground planner: TEST + ELKAK ground station")
    p.add_argument("--jetson", default="192.168.205.160", help="Jetson IP (default 192.168.205.160)")
    p.add_argument("--jetson-http", type=int, default=8080, help="Jetson HTTP port (default 8080)")
    p.add_argument("--jetson-ast", type=int, default=8600, help="Jetson ASTERIX UDP port (default 8600)")
    p.add_argument("--listen", type=int, default=8600, help="UDP port to listen for ELKAK ASTERIX (default 8600)")
    p.add_argument("--port", type=int, default=9000, help="Local HTTP port (default 9000)")
    p.add_argument("--mavlink",
                   help="Send targets as ADSB_VEHICLE over this MAVLink connection "
                        "(e.g. /dev/ttyUSB0 for the RFD radio)")
    p.add_argument("--mav-sysid", type=int, default=250)
    args = p.parse_args()
    JETSON_IP = args.jetson
    JETSON_HTTP = args.jetson_http
    JETSON_AST = args.jetson_ast
    LISTEN_PORT = args.listen

    global _MAV
    if args.mavlink:
        from asterix062_bridge import MavlinkSender
        _MAV = MavlinkSender(args.mavlink, args.mav_sysid)
        print(f"[GROUND PLANNER] RFD/MAVLink: sending ADSB_VEHICLE on {args.mavlink}")
        threading.Thread(target=_mav_rx_loop, daemon=True, name="rfd-rx").start()

    threading.Thread(target=_emitter_loop, daemon=True).start()
    threading.Thread(target=_elkak_listener, daemon=True).start()

    import signal, subprocess as _sp
    try:
        out = _sp.check_output(["lsof", "-ti", f":{args.port}"], text=True).strip()
        for pid in out.split():
            pid = int(pid)
            if pid != os.getpid():
                os.kill(pid, signal.SIGTERM)
                print(f"[GROUND PLANNER] Killed old process {pid} on port {args.port}")
                time.sleep(0.3)
    except Exception:
        pass

    import socketserver
    socketserver.TCPServer.allow_reuse_address = True
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[GROUND PLANNER] http://localhost:{args.port}")
    print(f"[GROUND PLANNER] Jetson: {JETSON_IP}:{JETSON_HTTP} (HTTP), :{JETSON_AST} (ASTERIX UDP)")
    if _MAV:
        print(f"[GROUND PLANNER] RFD link: targets also sent as ADSB_VEHICLE via {args.mavlink}")
    print(f"[GROUND PLANNER] ELKAK listener: UDP :{LISTEN_PORT}")
    print(f"[GROUND PLANNER] Tabs: TEST (Circle/Line/Draw), ELKAK (live + manual)")
    print(f"[GROUND PLANNER] TX rate: 2 Hz (per EKETA ICD)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[GROUND PLANNER] Stopped.")


if __name__ == "__main__":
    main()
