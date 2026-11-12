#!/usr/bin/env python3
"""
ground_planner.py — Laptop-side ground station for ASTERIX testing and operations.

Three modes:
  1. Draw (Test)   — draw circles/lines on the map, emits fake ASTERIX to Jetson
  2. Manual (Test)  — type lat/lon/alt for a static target, emits fake ASTERIX
  3. ELKAK (Live)   — listens for real ASTERIX CAT062 from EKETA-EAB, relays to Jetson

    python ground_planner.py --jetson 192.168.205.160

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

# ---- state ------------------------------------------------------------------

STATE = {
    "mode": "draw",          # "draw" | "manual" | "elkak"
    "targets": [],            # mode 1+2: user-defined targets
    "elkak_tracks": [],       # mode 3: live tracks from EKETA
    "elkak_packets": 0,
    "elkak_last_rx": 0,
    "ref": None,
    "hz": 2.0,               # 2 Hz per ICD
}
LOCK = threading.Lock()
JETSON_IP = "192.168.205.160"
JETSON_HTTP = 8080
JETSON_AST = 8600
LISTEN_PORT = 8600
SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_PKT_COUNT = [0]


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
    else:
        return t["lat"], t["lon"], t.get("alt", 50), 0.0, 0.0


# ---- ELKAK listener (Mode 3) -----------------------------------------------

def _elkak_listener():
    """Listen for real ASTERIX CAT062 from EKETA-EAB, decode and relay to Jetson."""
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


# ---- emitter (Mode 1+2) ----------------------------------------------------

def _emitter_loop():
    start = time.time()
    while True:
        with LOCK:
            mode = STATE["mode"]
            tgt_list = [dict(t) for t in STATE["targets"]]
            ref = STATE["ref"]
            hz = STATE["hz"]
        if mode == "elkak":
            time.sleep(0.5)
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
:root{--bg:#0b0f14;--panel:#121820;--text:#e7eef7;--muted:#8ea0b5;--accent:#4fc3f7;
  --ok:#4ade80;--warn:#f59e0b;--err:#f87171;--border:#233041;--op:#e040fb}
*{box-sizing:border-box;margin:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:var(--bg);color:var(--text);display:flex;flex-direction:column;height:100vh}
header{padding:10px 16px;display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--border)}
header h1{font-size:16px;letter-spacing:.3px}
.status{margin-left:auto;font-size:12px;color:var(--muted)}
.status b{color:var(--ok)}
.status b.off{color:var(--err)}
.main{flex:1;display:flex;min-height:0}
.map-wrap{flex:1;min-width:0;position:relative}
#map{width:100%;height:100%;background:#111}
.toolbar{position:absolute;top:10px;left:56px;z-index:1000;display:flex;gap:6px}
.toolbar button{padding:8px 14px;border:none;border-radius:8px;font:600 13px/1 sans-serif;
  cursor:pointer;background:var(--panel);color:var(--text);border:1px solid var(--border);
  transition:all .15s}
.toolbar button:hover{border-color:var(--accent)}
.toolbar button.active{background:var(--accent);color:#04121c;border-color:var(--accent)}
.draw-hint{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);z-index:1000;
  background:rgba(0,0,0,.8);color:var(--warn);padding:8px 16px;border-radius:8px;
  font:600 13px/1 sans-serif;pointer-events:none;display:none}
.sidebar{width:320px;overflow-y:auto;padding:0;display:flex;flex-direction:column;
  border-left:1px solid var(--border);background:var(--panel)}
.mode-tabs{display:flex;border-bottom:1px solid var(--border)}
.mode-tabs button{flex:1;padding:10px 4px;border:none;background:transparent;color:var(--muted);
  font:600 12px/1 sans-serif;cursor:pointer;border-bottom:2px solid transparent;transition:all .15s}
.mode-tabs button.active{color:var(--text);border-bottom-color:var(--accent)}
.mode-tabs button.op-active{color:var(--op);border-bottom-color:var(--op)}
.mode-tabs .test-badge{font-size:9px;background:var(--warn);color:#000;padding:1px 4px;
  border-radius:3px;margin-left:4px;vertical-align:middle}
.mode-tabs .op-badge{font-size:9px;background:var(--op);color:#fff;padding:1px 4px;
  border-radius:3px;margin-left:4px;vertical-align:middle}
.mode-content{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px}
.card{padding:10px 12px;border-radius:10px;border:1px solid var(--border);background:var(--bg)}
.card h3{font-size:13px;margin:0 0 6px;display:flex;align-items:center;gap:8px}
.card h3 .dot{width:10px;height:10px;border-radius:50%;flex:0 0 auto}
.card h3 .spacer{flex:1}
.card .del-btn{background:none;border:none;color:var(--err);cursor:pointer;font-size:16px;
  padding:0 4px;opacity:.6}
.card .del-btn:hover{opacity:1}
.card .info{font-size:11px;color:var(--muted);line-height:1.5}
.row{display:flex;gap:6px;align-items:center;margin:3px 0}
.row label{font-size:11px;color:var(--muted);min-width:46px}
.row input,.row select{flex:1;background:var(--panel);border:1px solid var(--border);color:var(--text);
  border-radius:6px;padding:4px 6px;font-size:12px}
.row .unit{font-size:11px;color:var(--muted);min-width:24px}
.drone-info{font-size:11px;color:var(--muted);line-height:1.5}
.drone-info b{color:var(--text)}
.empty-hint{font-size:12px;color:var(--muted);text-align:center;padding:16px 8px;line-height:1.6}
.add-btn{width:100%;padding:10px;border:1px dashed var(--border);border-radius:8px;
  background:transparent;color:var(--accent);font:600 13px/1 sans-serif;cursor:pointer}
.add-btn:hover{border-color:var(--accent)}
.elkak-status{padding:12px;border-radius:10px;border:1px solid var(--border);background:var(--bg);
  font-size:12px;line-height:1.6}
.elkak-status b{color:var(--op)}
.elkak-track{padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:var(--bg);
  font-size:11px;line-height:1.5}
.elkak-track .tk-label{font-weight:700;font-size:12px}
.elkak-track .tk-classif{font-size:10px;padding:1px 5px;border-radius:3px;margin-left:6px}
.elkak-track .tk-classif.unknown{background:var(--warn);color:#000}
.elkak-track .tk-classif.friendly{background:var(--ok);color:#000}
.elkak-track .tk-classif.not_evaluated{background:var(--muted);color:#000}
/* modal overlay */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:2000;display:flex;
  align-items:center;justify-content:center}
.modal{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:20px 24px;
  min-width:320px;max-width:400px;box-shadow:0 8px 30px rgba(0,0,0,.5)}
.modal h2{font-size:15px;margin:0 0 14px;display:flex;align-items:center;gap:8px}
.modal h2 .badge{font-size:10px;padding:2px 6px;border-radius:4px}
.modal h2 .badge.circle{background:var(--accent);color:#04121c}
.modal h2 .badge.line{background:var(--warn);color:#000}
.modal h2 .badge.static{background:var(--ok);color:#000}
.modal .mrow{display:flex;gap:8px;align-items:center;margin:8px 0}
.modal .mrow label{font-size:12px;color:var(--muted);min-width:56px}
.modal .mrow input{flex:1;background:var(--bg);border:1px solid var(--border);color:var(--text);
  border-radius:6px;padding:6px 8px;font-size:13px}
.modal .mrow .unit{font-size:11px;color:var(--muted);min-width:30px}
.modal .mrow-info{font-size:11px;color:var(--muted);margin:2px 0 2px 64px}
.modal .btn-row{display:flex;gap:10px;margin-top:18px}
.modal .btn-row button{flex:1;padding:10px;border:none;border-radius:8px;font:600 13px/1 sans-serif;
  cursor:pointer;transition:all .15s}
.modal .btn-send{background:var(--accent);color:#04121c}
.modal .btn-send:hover{opacity:.85}
.modal .btn-send:disabled{opacity:.35;cursor:not-allowed}
.modal .btn-cancel{background:var(--bg);color:var(--err);border:1px solid var(--border)}
.modal .btn-cancel:hover{border-color:var(--err)}
/* SEEK button overlay */
.seek-btn{position:absolute;bottom:20px;right:20px;z-index:1000;width:80px;height:80px;
  border-radius:50%;border:3px solid var(--border);background:var(--panel);color:var(--muted);
  font:700 14px/1 sans-serif;cursor:not-allowed;transition:all .3s;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:2px;box-shadow:0 4px 16px rgba(0,0,0,.4);
  user-select:none;opacity:.7}
.seek-btn .seek-icon{font-size:22px}
.seek-btn.ready{background:#1b5e20;border-color:var(--ok);color:var(--ok);cursor:pointer;
  opacity:1;animation:seekPulse 2s ease-in-out infinite}
.seek-btn.ready:hover{background:#2e7d32;transform:scale(1.05)}
.seek-btn.active{background:#7f1d1d;border-color:var(--err);color:var(--err);cursor:pointer;
  opacity:1;animation:none}
.seek-btn.active:hover{background:#991b1b}
@keyframes seekPulse{0%,100%{box-shadow:0 0 8px rgba(74,222,128,.3)}
  50%{box-shadow:0 0 24px rgba(74,222,128,.6)}}
.seek-panel{padding:0 12px 12px;display:flex;flex-direction:column;gap:6px;
  border-top:1px solid var(--border)}
.seek-panel h3{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;
  margin:8px 0 2px;cursor:pointer;user-select:none}
.seek-panel h3:hover{color:var(--accent)}
.fparam{display:flex;gap:6px;align-items:center;margin:2px 0}
.fparam label{font-size:11px;color:var(--muted);min-width:70px}
.fparam input{flex:1;background:var(--bg);border:1px solid var(--border);color:var(--text);
  border-radius:6px;padding:3px 6px;font-size:12px;max-width:80px}
.fparam input:disabled{opacity:.4}
.fparam .unit{font-size:10px;color:var(--muted);min-width:28px}
.msn-box{padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:var(--bg);
  font-size:11px;line-height:1.6}
.msn-box b{color:var(--text)}
.msn-box .c-ok{color:var(--ok)}.msn-box .c-warn{color:var(--warn)}.msn-box .c-err{color:var(--err)}
.offer-box{padding:10px 12px;border-radius:8px;background:#1a1a2e;border:1px solid var(--accent);margin:4px 0}
.offer-box .offer-text{font-size:12px;margin:0 0 8px;line-height:1.4}
.offer-btns{display:flex;gap:8px}
.offer-btns button{flex:1;padding:8px;border:none;border-radius:6px;font:600 12px/1 sans-serif;cursor:pointer}
.offer-yes{background:var(--ok);color:#000}
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
  <span class="status">TX: <b id="tx_st">idle</b> <span id="tx_n"></span></span>
</header>
<div class="main">
  <div class="map-wrap">
    <div id="map"></div>
    <div class="toolbar" id="draw_toolbar">
      <button id="draw_circle">&#9711; Circle</button>
      <button id="draw_line">&#9586; Line</button>
    </div>
    <div class="draw-hint" id="draw_hint"></div>
    <button class="seek-btn" id="seek_btn" disabled><span class="seek-icon">&#9678;</span><span id="seek_label">SEEK</span></button>
  </div>
  <div class="sidebar">
    <div class="mode-tabs">
      <button id="tab_draw" class="active" data-mode="draw">Draw<span class="test-badge">TEST</span></button>
      <button id="tab_manual" data-mode="manual">Manual<span class="test-badge">TEST</span></button>
      <button id="tab_elkak" data-mode="elkak">ELKAK<span class="op-badge">LIVE</span></button>
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

var COLORS = ["#4fc3f7","#f59e0b","#f87171","#a78bfa","#4ade80","#fb923c","#38bdf8","#e879f9"];
var droneMarker = null, centered = false;
var targets = [], elkakMarkers = {};
var drawMode = null, drawStep = null, animStart = Date.now();
var currentMode = "draw";
var modalOpen = false;

function hint(msg){ var el=document.getElementById("draw_hint"); el.textContent=msg||""; el.style.display=msg?"":"none"; }
function tgtColor(i){ return COLORS[i%COLORS.length]; }
function droneIcon(){
  return L.divIcon({className:"",html:'<svg width="28" height="28" viewBox="-14 -14 28 28"><circle r="10" fill="#4ade80" stroke="#fff" stroke-width="2"/><circle r="3" fill="#fff"/></svg>',iconSize:[28,28],iconAnchor:[14,14]});
}
function tgtIcon(color){
  return L.divIcon({className:"",html:'<svg width="20" height="20" viewBox="-10 -10 20 20"><polygon points="0,-8 6,6 -6,6" fill="'+color+'" stroke="#fff" stroke-width="1.5"/></svg>',iconSize:[20,20],iconAnchor:[10,10]});
}
function elkakIcon(){
  return L.divIcon({className:"",html:'<svg width="24" height="24" viewBox="-12 -12 24 24"><polygon points="0,-10 7,7 -7,7" fill="#e040fb" stroke="#fff" stroke-width="1.5"/></svg>',iconSize:[24,24],iconAnchor:[12,12]});
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
  } else {
    fields = '<div class="mrow"><label>Latitude</label><input id="m_lat" type="number" value="" step="0.00001" placeholder="e.g. 37.80700"></div>'
      +'<div class="mrow"><label>Longitude</label><input id="m_lon" type="number" value="" step="0.00001" placeholder="e.g. 23.78500"></div>'
      +'<div class="mrow"><label>Altitude</label><input id="m_alt" type="number" value="50" min="0" step="5"><span class="unit">m</span></div>';
  }
  var badge = '<span class="badge '+type+'">'+type.toUpperCase()+'</span>';
  var title = type === "circle" ? "New Circle Target" : type === "line" ? "New Line Target" : "New Static Target";
  root.innerHTML = '<div class="modal-overlay" id="modal_overlay">'
    +'<div class="modal">'
    +'<h2>'+badge+' '+title+'</h2>'
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
    } else if(type === "line"){
      params.speed = parseFloat(document.getElementById("m_speed").value);
      if(!params.speed || params.speed <= 0) return;
    } else {
      params.lat = parseFloat(document.getElementById("m_lat").value);
      params.lon = parseFloat(document.getElementById("m_lon").value);
      if(!params.lat || !params.lon) return;
    }
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
    document.getElementById("draw_toolbar").style.display = currentMode === "draw" ? "" : "none";
    drawMode = null; drawStep = null; hint("");
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
  if(currentMode === "draw") renderDrawMode(list, mh);
  else if(currentMode === "manual") renderManualMode(list, mh);
  else renderElkakMode(list, mh);
}

// ---- MODE 1: DRAW (read-only cards after confirmation) ----
function renderDrawMode(list, mh){
  if(!targets.length) mh.innerHTML = '<div class="empty-hint">Click <b>Circle</b> or <b>Line</b> on the map toolbar, then click the map to place a target.<br>You\'ll set parameters and confirm before sending.</div>';
  targets.forEach(function(t, idx){
    var c = tgtColor(idx), card = document.createElement("div");
    card.className = "card";
    var typeLabel = t.type === "circle" ? "Circle" : t.type === "line" ? "Line" : "Static";
    var details = "";
    if(t.type === "circle"){
      var period = (2*Math.PI*t.radius / Math.max(0.1,t.speed)).toFixed(1);
      details = 'Radius: '+t.radius+' m &nbsp; Speed: '+t.speed+' m/s &nbsp; Orbit: '+period+' s<br>';
    } else if(t.type === "line"){
      details = 'Speed: '+t.speed+' m/s<br>';
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

// ---- MODE 2: MANUAL (add via modal) ----
function renderManualMode(list, mh){
  targets.forEach(function(t, idx){
    var c = tgtColor(idx), card = document.createElement("div");
    card.className = "card";
    card.innerHTML = '<h3><span class="dot" style="background:'+c+'"></span>T'+t.id+' Static'
      +'<span class="spacer"></span>'
      +'<button class="del-btn" data-delidx="'+idx+'" title="Remove target">&times;</button></h3>'
      +'<div class="info">Lat: '+(t.lat?t.lat.toFixed(6):"--")+' &nbsp; Lon: '+(t.lon?t.lon.toFixed(6):"--")+'<br>Alt: '+t.alt+' m</div>';
    list.appendChild(card);
  });
  var addBtn = document.createElement("button");
  addBtn.className = "add-btn"; addBtn.textContent = "+ Add static target";
  addBtn.onclick = function(){
    showModal("static", {}, function(params){
      commitTarget("static", params.lat, params.lon, 0, 0, params);
    });
  };
  list.appendChild(addBtn);
  list.querySelectorAll("[data-delidx]").forEach(function(btn){
    btn.addEventListener("click",function(){removeTarget(parseInt(btn.dataset.delidx));});
  });
  if(!targets.length) mh.innerHTML = '<div class="empty-hint">Click <b>+ Add static target</b> to enter coordinates and send.</div>';
}

// ---- MODE 3: ELKAK ----
var elkakPollTimer = null;
function renderElkakMode(list, mh){
  mh.innerHTML = '<div class="elkak-status"><b>ELKAK OPERATIONAL MODE</b><br>'
    +'Listening on UDP :LISTEN_PORT_PH for ASTERIX CAT062 from EKETA-EAB.<br>'
    +'Packets are decoded and relayed to the Jetson in real time.<br><br>'
    +'<span id="elkak_st">Waiting for data...</span></div>';
  mh.innerHTML = mh.innerHTML.replace("LISTEN_PORT_PH", "LISTEN_PORT_VAL");
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

// ---- target data model (modes 1+2) ----
function commitTarget(type, lat, lon, lat2, lon2, params){
  var id = targets.length+1;
  targets.push({id:id, enabled:true, type:type, lat:lat, lon:lon, alt:params.alt||50,
    lat2:lat2||0, lon2:lon2||0, speed:params.speed||10, radius:params.radius||200, layers:{}});
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
  if(ly.anim)ly.anim.remove(); targets[idx].layers={};
}
function drawTargetLayers(idx){
  var t=targets[idx],c=tgtColor(idx),ly=t.layers;
  if(ly.m)ly.m.remove(); if(ly.m2)ly.m2.remove();
  if(ly.path)ly.path.remove(); if(ly.circle)ly.circle.remove();
  if(!t.lat&&!t.lon)return;
  ly.m=L.marker([t.lat,t.lon],{icon:tgtIcon(c),draggable:false}).addTo(map);
  if(t.type==="line"&&t.lat2&&t.lon2){
    ly.m2=L.marker([t.lat2,t.lon2],{icon:tgtIcon(c),draggable:false}).addTo(map);
    ly.path=L.polyline([[t.lat,t.lon],[t.lat2,t.lon2]],{color:c,weight:2,dashArray:"6 4"}).addTo(map);
  }
  if(t.type==="circle") ly.circle=L.circle([t.lat,t.lon],{radius:t.radius,color:c,weight:2,fill:false,dashArray:"6 4"}).addTo(map);
}
function redrawAllLayers(){targets.forEach(function(_,i){drawTargetLayers(i);});}

function syncToServer(){
  var data=targets.map(function(t){
    return {id:t.id, enabled:t.enabled, type:t.type, lat:t.lat, lon:t.lon, alt:t.alt,
            lat2:t.lat2, lon2:t.lon2, speed:t.speed, radius:t.radius};
  });
  fetch("/api/targets",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({targets:data})});
}

// ---- drawing (mode 1) — click map then confirm in modal ----
document.getElementById("draw_circle").onclick=function(){
  if(drawMode==="circle"){drawMode=null;drawStep=null;hint("");this.classList.remove("active");return;}
  drawMode="circle";drawStep=null;
  document.getElementById("draw_circle").classList.add("active");
  document.getElementById("draw_line").classList.remove("active");
  hint("Click map to place circle center");
};
document.getElementById("draw_line").onclick=function(){
  if(drawMode==="line"){drawMode=null;drawStep=null;hint("");this.classList.remove("active");return;}
  drawMode="line";drawStep=null;
  document.getElementById("draw_line").classList.add("active");
  document.getElementById("draw_circle").classList.remove("active");
  hint("Click map for line start point");
};

map.on("click",function(e){
  if(modalOpen) return;
  if(currentMode!=="draw"||!drawMode) return;
  if(drawMode==="circle"){
    var clat=e.latlng.lat, clon=e.latlng.lng;
    showModal("circle", {lat:clat,lon:clon}, function(params){
      commitTarget("circle", clat, clon, 0, 0, params);
      hint("Circle sent. Click again for another, or click Circle to stop.");
    });
  } else if(drawMode==="line"){
    if(!drawStep){
      drawStep={lat:e.latlng.lat,lon:e.latlng.lng};
      hint("Click map for line end point");
    } else {
      var lat1=drawStep.lat, lon1=drawStep.lon, lat2=e.latlng.lat, lon2=e.latlng.lng;
      drawStep=null;
      showModal("line", {lat:lat1,lon:lon1,lat2:lat2,lon2:lon2}, function(params){
        commitTarget("line", lat1, lon1, lat2, lon2, params);
        hint("Line sent. Click again for another, or click Line to stop.");
      });
    }
  }
});
document.addEventListener("keydown",function(e){
  if(e.key==="Escape"){
    if(modalOpen){closeModal();return;}
    drawMode=null;drawStep=null;hint("");
    document.getElementById("draw_circle").classList.remove("active");
    document.getElementById("draw_line").classList.remove("active");
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

// ---- animate moving targets (modes 1+2) ----
function animateTargets(){
  if(currentMode==="elkak"){requestAnimationFrame(animateTargets);return;}
  var elapsed=(Date.now()-animStart)/1000;
  targets.forEach(function(t,i){
    if(!t.enabled||(!t.lat&&!t.lon)){if(t.layers.anim){t.layers.anim.remove();t.layers.anim=null;}return;}
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
    }
    if(lat&&lon){
      var c=tgtColor(i);
      if(!t.layers.anim) t.layers.anim=L.circleMarker([lat,lon],{radius:6,color:c,fillColor:c,fillOpacity:.9,weight:2}).addTo(map);
      else t.layers.anim.setLatLng([lat,lon]);
    }
  });
  requestAnimationFrame(animateTargets);
}
requestAnimationFrame(animateTargets);

setInterval(pollDrone,2000); pollDrone();
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
  var phase=(m||{}).phase||"";
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
        elif path == "/api/status":
            with LOCK:
                ntgt = len([t for t in STATE["targets"] if t.get("enabled") and (t.get("lat") or t.get("lon"))])
            self._json({"targets": ntgt, "packets": _PKT_COUNT[0]})
        elif path == "/api/elkak":
            with LOCK:
                tracks = list(STATE["elkak_tracks"])
                pkts = STATE["elkak_packets"]
                last = STATE["elkak_last_rx"]
            age = round(time.time() - last, 1) if last > 0 else None
            self._json({"packets": pkts, "age": age, "tracks": tracks})
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
            mode = data.get("mode", "draw")
            with LOCK:
                STATE["mode"] = mode
            self._json({"ok": True, "mode": mode})
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
    p = argparse.ArgumentParser(description="Ground planner: 3-mode ASTERIX ground station")
    p.add_argument("--jetson", default="192.168.205.160", help="Jetson IP (default 192.168.205.160)")
    p.add_argument("--jetson-http", type=int, default=8080, help="Jetson HTTP port (default 8080)")
    p.add_argument("--jetson-ast", type=int, default=8600, help="Jetson ASTERIX UDP port (default 8600)")
    p.add_argument("--listen", type=int, default=8600, help="UDP port to listen for ELKAK ASTERIX (default 8600)")
    p.add_argument("--port", type=int, default=9000, help="Local HTTP port (default 9000)")
    args = p.parse_args()
    JETSON_IP = args.jetson
    JETSON_HTTP = args.jetson_http
    JETSON_AST = args.jetson_ast
    LISTEN_PORT = args.listen

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
    print(f"[GROUND PLANNER] ELKAK listener: UDP :{LISTEN_PORT}")
    print(f"[GROUND PLANNER] Modes: Draw (test), Manual (test), ELKAK (live)")
    print(f"[GROUND PLANNER] TX rate: 2 Hz (per EKETA ICD)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[GROUND PLANNER] Stopped.")


if __name__ == "__main__":
    main()
