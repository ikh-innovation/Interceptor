#!/usr/bin/env python3
"""
ground_planner.py — Laptop-side ground station with map + fake ASTERIX emitter.

Opens a browser map showing the drone's GPS position (satellite view), lets you
draw target paths (line or circle) for two ASTERIX targets, and streams them as
EKETA CAT062 UDP packets to the Jetson.  Also embeds the live camera feed.

    python ground_planner.py --jetson 192.168.205.160

Then open http://localhost:9000 in a browser.
"""
import argparse
import json
import math
import os
import socket
import sys
import threading
import time
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(__file__) or ".")
from asterix_cue import encode_eketa_cat062

# ---- state ------------------------------------------------------------------

STATE = {
    "playing": False,
    "targets": [
        {"id": 1, "enabled": True, "type": "static", "lat": 0, "lon": 0, "alt": 50,
         "lat2": 0, "lon2": 0, "speed": 10, "radius": 200, "period": 60},
        {"id": 2, "enabled": False, "type": "static", "lat": 0, "lon": 0, "alt": 80,
         "lat2": 0, "lon2": 0, "speed": 10, "radius": 200, "period": 60},
    ],
    "ref": None,
    "hz": 2.0,
}
LOCK = threading.Lock()
JETSON_IP = "192.168.205.160"
JETSON_HTTP = 8080
JETSON_AST = 8600
SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
EMITTER_THREAD = None


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
    if t["type"] == "static":
        return t["lat"], t["lon"], t["alt"], 0.0, 0.0
    elif t["type"] == "circle":
        a = 2 * math.pi * (elapsed / max(1, t["period"]))
        r = t["radius"]
        lat0, lon0 = t["lat"], t["lon"]
        dlat = r * math.cos(a) / 111320.0
        dlon = r * math.sin(a) / (111320.0 * math.cos(math.radians(lat0)))
        w = 2 * math.pi / max(1, t["period"])
        vx = r * w * math.cos(a)
        vy = -r * w * math.sin(a)
        return lat0 + dlat, lon0 + dlon, t["alt"], vx, vy
    elif t["type"] == "line":
        d = math.hypot((t["lat2"] - t["lat"]) * 111320,
                       (t["lon2"] - t["lon"]) * 111320 * math.cos(math.radians(t["lat"])))
        period = (d / max(0.1, t["speed"])) * 2 if d > 0.1 else 1.0
        frac_t = (elapsed % period) / period
        frac = 2 * frac_t if frac_t < 0.5 else 2 * (1 - frac_t)
        lat = t["lat"] + frac * (t["lat2"] - t["lat"])
        lon = t["lon"] + frac * (t["lon2"] - t["lon"])
        alt = t["alt"]
        brg = math.atan2((t["lon2"] - t["lon"]) * math.cos(math.radians(t["lat"])),
                         t["lat2"] - t["lat"])
        sign = 1.0 if frac_t < 0.5 else -1.0
        vx = sign * t["speed"] * math.sin(brg)
        vy = sign * t["speed"] * math.cos(brg)
        return lat, lon, alt, vx, vy


def _emitter_loop():
    start = time.time()
    n = 0
    while True:
        with LOCK:
            playing = STATE["playing"]
            targets = [dict(t) for t in STATE["targets"]]
            ref = STATE["ref"]
            hz = STATE["hz"]
        if not playing:
            time.sleep(0.2)
            continue
        elapsed = time.time() - start
        tracks = []
        for t in targets:
            if not t["enabled"] or t["lat"] == 0 and t["lon"] == 0:
                continue
            lat, lon, alt, vx, vy = _target_pos(t, elapsed)
            if ref is None:
                ref = (lat, lon, alt)
                with LOCK:
                    STATE["ref"] = ref
            polar = _polar(ref[0], ref[1], ref[2], lat, lon, alt)
            tracks.append(dict(lat=lat, lon=lon, h_msl_m=alt, vx=vx, vy=vy,
                               track_no=t["id"], classif="unknown", **polar))
        if tracks:
            data = encode_eketa_cat062(tracks)
            SOCK.sendto(data, (JETSON_IP, JETSON_AST))
            n += 1
        time.sleep(1.0 / hz)


# ---- HTTP handler -----------------------------------------------------------

HTML_PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ground Planner</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root{--bg:#0b0f14;--panel:#121820;--text:#e7eef7;--muted:#8ea0b5;--accent:#4fc3f7;
  --ok:#4ade80;--warn:#f59e0b;--err:#f87171;--border:#233041}
*{box-sizing:border-box;margin:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:var(--bg);color:var(--text);display:flex;flex-direction:column;height:100vh}
header{padding:10px 16px;display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--border)}
header h1{font-size:16px;letter-spacing:.3px}
.status{margin-left:auto;font-size:12px;color:var(--muted)}
.status b{color:var(--ok)}
.status b.off{color:var(--err)}
.main{flex:1;display:flex;min-height:0}
#map{flex:1;min-width:0;background:#111}
.sidebar{width:320px;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px;
  border-left:1px solid var(--border);background:var(--panel)}
.card{padding:10px 12px;border-radius:10px;border:1px solid var(--border);background:var(--bg)}
.card h3{font-size:13px;margin:0 0 8px;display:flex;align-items:center;gap:8px}
.card h3 .dot{width:10px;height:10px;border-radius:50%;flex:0 0 auto}
.t1 .dot{background:var(--accent)}
.t2 .dot{background:var(--warn)}
.row{display:flex;gap:6px;align-items:center;margin:4px 0}
.row label{font-size:11px;color:var(--muted);min-width:50px}
.row input,.row select{flex:1;background:var(--panel);border:1px solid var(--border);color:var(--text);
  border-radius:6px;padding:4px 6px;font-size:12px}
.row input[type=checkbox]{flex:0;width:16px;height:16px}
.btn{padding:8px 16px;border:none;border-radius:8px;font:600 14px/1 sans-serif;cursor:pointer}
.play-btn{width:100%;padding:12px;font-size:16px;letter-spacing:.1em}
.play-btn.on{background:var(--ok);color:#04121c}
.play-btn.off{background:var(--err);color:#fff}
.set-btn{background:var(--accent);color:#04121c;font-size:11px;padding:4px 8px}
.set-btn:hover{opacity:.85}
.set-btn.active{background:var(--warn);color:#1a1200}
.camera{height:200px;border-top:1px solid var(--border);position:relative;overflow:hidden;flex:0 0 auto}
.camera img{width:100%;height:100%;object-fit:contain;background:#000}
.camera .label{position:absolute;top:4px;left:8px;font:600 11px sans-serif;color:var(--muted);
  background:rgba(0,0,0,.6);padding:2px 6px;border-radius:4px}
.drone-info{font-size:11px;color:var(--muted);line-height:1.5}
.drone-info b{color:var(--text)}
.mode-btns{display:flex;gap:4px}
.mode-btns button{flex:1;padding:4px;border:1px solid var(--border);border-radius:6px;
  background:var(--panel);color:var(--text);font-size:11px;cursor:pointer}
.mode-btns button.on{border-color:var(--accent);color:var(--accent)}
@media(max-width:700px){
  .main{flex-direction:column}
  .sidebar{width:100%;max-height:50vh;border-left:none;border-top:1px solid var(--border)}
  #map{min-height:40vh}
}
</style></head><body>
<header>
  <h1>Ground Planner</h1>
  <span class="status">Drone: <b id="drone_st" class="off">no GPS</b></span>
  <span class="status">TX: <b id="tx_st" class="off">stopped</b> <span id="tx_n">0</span> pkt</span>
</header>
<div class="main">
  <div id="map"></div>
  <div class="sidebar">
    <div class="card drone-info" id="drone_card">
      <b>Drone</b><br>
      Lat: <span id="d_lat">--</span> Lon: <span id="d_lon">--</span><br>
      Alt: <span id="d_alt">--</span> m &nbsp; Mode: <span id="d_mode">--</span>
      &nbsp; Armed: <span id="d_armed">--</span>
    </div>

    <div class="card t1">
      <h3><span class="dot"></span>Target 1
        <input type="checkbox" id="t1_en" checked title="Enable">
        <button class="set-btn" id="t1_set" title="Click map to place">Place on map</button></h3>
      <div class="row"><label>Type</label>
        <div class="mode-btns" id="t1_type">
          <button data-v="static" class="on">Static</button>
          <button data-v="circle">Circle</button>
          <button data-v="line">Line</button>
        </div></div>
      <div class="row"><label>Lat</label><input id="t1_lat" type="number" step="0.00001"></div>
      <div class="row"><label>Lon</label><input id="t1_lon" type="number" step="0.00001"></div>
      <div class="row t1line" style="display:none"><label>Lat 2</label><input id="t1_lat2" type="number" step="0.00001"></div>
      <div class="row t1line" style="display:none"><label>Lon 2</label><input id="t1_lon2" type="number" step="0.00001"></div>
      <div class="row"><label>Alt (m)</label><input id="t1_alt" type="number" value="50"></div>
      <div class="row t1motion" style="display:none"><label>Speed</label><input id="t1_speed" type="number" value="10" step="1"> m/s</div>
      <div class="row t1circle" style="display:none"><label>Radius</label><input id="t1_radius" type="number" value="200" step="10"> m</div>
      <div class="row t1circle" style="display:none"><label>Period</label><input id="t1_period" type="number" value="60" step="5"> s</div>
    </div>

    <div class="card t2">
      <h3><span class="dot"></span>Target 2
        <input type="checkbox" id="t2_en" title="Enable">
        <button class="set-btn" id="t2_set" title="Click map to place">Place on map</button></h3>
      <div class="row"><label>Type</label>
        <div class="mode-btns" id="t2_type">
          <button data-v="static" class="on">Static</button>
          <button data-v="circle">Circle</button>
          <button data-v="line">Line</button>
        </div></div>
      <div class="row"><label>Lat</label><input id="t2_lat" type="number" step="0.00001"></div>
      <div class="row"><label>Lon</label><input id="t2_lon" type="number" step="0.00001"></div>
      <div class="row t2line" style="display:none"><label>Lat 2</label><input id="t2_lat2" type="number" step="0.00001"></div>
      <div class="row t2line" style="display:none"><label>Lon 2</label><input id="t2_lon2" type="number" step="0.00001"></div>
      <div class="row"><label>Alt (m)</label><input id="t2_alt" type="number" value="80"></div>
      <div class="row t2motion" style="display:none"><label>Speed</label><input id="t2_speed" type="number" value="10" step="1"> m/s</div>
      <div class="row t2circle" style="display:none"><label>Radius</label><input id="t2_radius" type="number" value="200" step="10"> m</div>
      <div class="row t2circle" style="display:none"><label>Period</label><input id="t2_period" type="number" value="60" step="5"> s</div>
    </div>

    <button class="btn play-btn off" id="play_btn">PLAY</button>
  </div>
</div>
<div class="camera">
  <span class="label">Camera</span>
  <img id="cam" src="" alt="camera feed">
</div>

<script>
var JETSON = "JETSON_PLACEHOLDER";
var map = L.map("map", {zoomControl: true}).setView([37.9, 23.7], 16);
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
  maxZoom: 19, attribution: "Esri"
}).addTo(map);

var droneMarker = null, droneCircle = null;
var markers = [{m: null, m2: null, path: null, circle: null, anim: null},
               {m: null, m2: null, path: null, circle: null, anim: null}];
var colors = ["#4fc3f7", "#f59e0b"];
var placing = null; // {t: 0|1, step: "p1"|"p2"}
var animStart = Date.now();
var centered = false;

function droneIcon() {
  return L.divIcon({className: "", html:
    '<svg width="28" height="28" viewBox="-14 -14 28 28"><circle r="10" fill="#4ade80" stroke="#fff" stroke-width="2"/>' +
    '<circle r="3" fill="#fff"/></svg>', iconSize: [28, 28], iconAnchor: [14, 14]});
}
function tgtIcon(color) {
  return L.divIcon({className: "", html:
    '<svg width="20" height="20" viewBox="-10 -10 20 20"><polygon points="0,-8 6,6 -6,6" fill="' + color + '" stroke="#fff" stroke-width="1.5"/></svg>',
    iconSize: [20, 20], iconAnchor: [10, 10]});
}

// Drone polling
function pollDrone() {
  fetch("/drone/state").then(r => r.ok ? r.json() : null).then(j => {
    if (!j) { document.getElementById("drone_st").className = "off"; document.getElementById("drone_st").textContent = "offline"; return; }
    document.getElementById("d_mode").textContent = j.fc_mode || "--";
    document.getElementById("d_armed").textContent = j.armed ? "YES" : "no";
    document.getElementById("drone_st").className = j.linked ? "" : "off";
    document.getElementById("drone_st").textContent = j.linked ? "linked" : "no FC";
    // position and home are inside j.mission
    var m = j.mission || {};
    var pos = m.position;
    var alt = m.height_m;
    document.getElementById("d_alt").textContent = alt != null ? alt.toFixed(1) : "--";
    if (pos && pos.length === 2 && pos[0]) {
      updateDrone(pos[0], pos[1], alt);
    } else if (!centered && m.home && m.home.length === 2 && m.home[0]) {
      map.setView(m.home, 17); centered = true;
    }
    // ASTERIX cue packet count — seeker snapshot has cue inside mission
    var sk = m.seeker || m;
    var cue = (sk.cue || {});
    if (cue.packets != null) document.getElementById("tx_n").textContent = cue.packets + " rx";
  }).catch(() => {
    document.getElementById("drone_st").className = "off";
    document.getElementById("drone_st").textContent = "offline";
  });
}

function updateDrone(lat, lon, alt) {
  if (!lat || !lon) return;
  document.getElementById("d_lat").textContent = lat.toFixed(6);
  document.getElementById("d_lon").textContent = lon.toFixed(6);
  if (alt != null) document.getElementById("d_alt").textContent = alt.toFixed(1);
  if (!droneMarker) {
    droneMarker = L.marker([lat, lon], {icon: droneIcon(), zIndexOffset: 1000}).addTo(map);
    if (!centered) { map.setView([lat, lon], 17); centered = true; }
  } else droneMarker.setLatLng([lat, lon]);
}

// Target markers
function updateTargetMarker(i) {
  var t = getTarget(i);
  var mk = markers[i], c = colors[i];
  if (mk.m) { mk.m.remove(); mk.m = null; }
  if (mk.m2) { mk.m2.remove(); mk.m2 = null; }
  if (mk.path) { mk.path.remove(); mk.path = null; }
  if (mk.circle) { mk.circle.remove(); mk.circle = null; }
  if (mk.anim) { mk.anim.remove(); mk.anim = null; }
  if (!t.lat && !t.lon) return;
  mk.m = L.marker([t.lat, t.lon], {icon: tgtIcon(c)}).addTo(map);
  if (t.type === "line" && t.lat2 && t.lon2) {
    mk.m2 = L.marker([t.lat2, t.lon2], {icon: tgtIcon(c)}).addTo(map);
    mk.path = L.polyline([[t.lat, t.lon], [t.lat2, t.lon2]], {color: c, weight: 2, dashArray: "6 4"}).addTo(map);
  }
  if (t.type === "circle") {
    mk.circle = L.circle([t.lat, t.lon], {radius: t.radius, color: c, weight: 2, fill: false, dashArray: "6 4"}).addTo(map);
  }
}

function getTarget(i) {
  var p = "t" + (i + 1) + "_";
  return {
    id: i + 1,
    enabled: document.getElementById(p + "en").checked,
    type: document.querySelector("#" + p + "type button.on").dataset.v,
    lat: parseFloat(document.getElementById(p + "lat").value) || 0,
    lon: parseFloat(document.getElementById(p + "lon").value) || 0,
    lat2: parseFloat(document.getElementById(p + "lat2").value) || 0,
    lon2: parseFloat(document.getElementById(p + "lon2").value) || 0,
    alt: parseFloat(document.getElementById(p + "alt").value) || 50,
    speed: parseFloat(document.getElementById(p + "speed").value) || 10,
    radius: parseFloat(document.getElementById(p + "radius").value) || 200,
    period: parseFloat(document.getElementById(p + "period").value) || 60,
  };
}

function syncToServer() {
  var targets = [getTarget(0), getTarget(1)];
  fetch("/api/targets", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({targets: targets})});
}

// Map click
map.on("click", function(e) {
  if (!placing) return;
  var p = "t" + (placing.t + 1) + "_";
  if (placing.step === "p1") {
    document.getElementById(p + "lat").value = e.latlng.lat.toFixed(6);
    document.getElementById(p + "lon").value = e.latlng.lng.toFixed(6);
    var type = getTarget(placing.t).type;
    if (type === "line") {
      placing.step = "p2";
      document.getElementById("t" + (placing.t + 1) + "_set").textContent = "Click end point";
      return;
    }
  } else if (placing.step === "p2") {
    document.getElementById(p + "lat2").value = e.latlng.lat.toFixed(6);
    document.getElementById(p + "lon2").value = e.latlng.lng.toFixed(6);
  }
  placing = null;
  document.querySelectorAll(".set-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("t" + (placing === null ? 1 : placing.t + 1) + "_set").textContent = "Place on map";
  // reset both labels
  document.getElementById("t1_set").textContent = "Place on map";
  document.getElementById("t2_set").textContent = "Place on map";
  updateTargetMarker(0); updateTargetMarker(1);
  syncToServer();
});

// Place buttons
[0, 1].forEach(function(i) {
  var btn = document.getElementById("t" + (i + 1) + "_set");
  btn.onclick = function() {
    if (placing && placing.t === i) {
      placing = null;
      btn.classList.remove("active");
      btn.textContent = "Place on map";
      return;
    }
    document.querySelectorAll(".set-btn").forEach(b => { b.classList.remove("active"); b.textContent = "Place on map"; });
    placing = {t: i, step: "p1"};
    btn.classList.add("active");
    btn.textContent = "Click on map...";
  };
});

// Type buttons
[1, 2].forEach(function(n) {
  document.querySelectorAll("#t" + n + "_type button").forEach(function(b) {
    b.onclick = function() {
      document.querySelectorAll("#t" + n + "_type button").forEach(x => x.classList.remove("on"));
      b.classList.add("on");
      var v = b.dataset.v;
      document.querySelectorAll(".t" + n + "line").forEach(x => x.style.display = v === "line" ? "" : "none");
      document.querySelectorAll(".t" + n + "circle").forEach(x => x.style.display = v === "circle" ? "" : "none");
      document.querySelectorAll(".t" + n + "motion").forEach(x => x.style.display = v === "static" ? "none" : "");
      updateTargetMarker(n - 1);
      syncToServer();
    };
  });
});

// Input changes sync
document.querySelectorAll(".sidebar input").forEach(function(inp) {
  inp.addEventListener("change", function() {
    updateTargetMarker(0); updateTargetMarker(1); syncToServer();
  });
});

// Play button
var playBtn = document.getElementById("play_btn");
var isPlaying = false;
playBtn.onclick = function() {
  isPlaying = !isPlaying;
  playBtn.textContent = isPlaying ? "STOP" : "PLAY";
  playBtn.className = "btn play-btn " + (isPlaying ? "on" : "off");
  document.getElementById("tx_st").className = isPlaying ? "" : "off";
  document.getElementById("tx_st").textContent = isPlaying ? "sending" : "stopped";
  syncToServer();
  fetch("/api/" + (isPlaying ? "play" : "pause"), {method: "POST"});
};

// Animate target positions on map
function animateTargets() {
  if (!isPlaying) { requestAnimationFrame(animateTargets); return; }
  var elapsed = (Date.now() - animStart) / 1000;
  [0, 1].forEach(function(i) {
    var t = getTarget(i);
    if (!t.enabled || (!t.lat && !t.lon)) return;
    var lat, lon;
    if (t.type === "static") { lat = t.lat; lon = t.lon; }
    else if (t.type === "circle") {
      var a = 2 * Math.PI * (elapsed / Math.max(1, t.period));
      var r = t.radius;
      lat = t.lat + r * Math.cos(a) / 111320;
      lon = t.lon + r * Math.sin(a) / (111320 * Math.cos(t.lat * Math.PI / 180));
    } else if (t.type === "line") {
      var d = Math.hypot((t.lat2 - t.lat) * 111320,
                         (t.lon2 - t.lon) * 111320 * Math.cos(t.lat * Math.PI / 180));
      var period = d > 0.1 ? (d / Math.max(0.1, t.speed)) * 2 : 1;
      var ft = (elapsed % period) / period;
      var frac = ft < 0.5 ? 2 * ft : 2 * (1 - ft);
      lat = t.lat + frac * (t.lat2 - t.lat);
      lon = t.lon + frac * (t.lon2 - t.lon);
    }
    if (lat && lon) {
      var mk = markers[i];
      if (!mk.anim) mk.anim = L.circleMarker([lat, lon], {radius: 6, color: colors[i], fillColor: colors[i], fillOpacity: 0.9, weight: 2}).addTo(map);
      else mk.anim.setLatLng([lat, lon]);
    }
  });
  requestAnimationFrame(animateTargets);
}
requestAnimationFrame(animateTargets);

// Camera feed
document.getElementById("cam").src = "/drone/stream";

// Polling
setInterval(pollDrone, 2000);
pollDrone();

// Packet counter
setInterval(function() {
  if (isPlaying) fetch("/api/status").then(r => r.json()).then(j => {
    document.getElementById("tx_n").textContent = j.packets + " pkt";
  }).catch(() => {});
}, 1000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            page = HTML_PAGE.replace("JETSON_PLACEHOLDER", JETSON_IP)
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/drone/state":
            self._proxy_json(f"http://{JETSON_IP}:{JETSON_HTTP}/state")
        elif path == "/drone/stream":
            self._proxy_stream(f"http://{JETSON_IP}:{JETSON_HTTP}/stream.mjpg")
        elif path == "/api/status":
            with LOCK:
                playing = STATE["playing"]
            self._json({"playing": playing, "packets": _PKT_COUNT[0]})
        else:
            self.send_error(404)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/play":
            with LOCK:
                STATE["playing"] = True
            self._json({"ok": True})
        elif path == "/api/pause":
            with LOCK:
                STATE["playing"] = False
            self._json({"ok": True})
        elif path == "/api/targets":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length)) if length else {}
            targets = data.get("targets", [])
            with LOCK:
                for i, t in enumerate(targets[:2]):
                    STATE["targets"][i].update(t)
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

    def _proxy_stream(self, url):
        try:
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=10)
            ct = resp.headers.get("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception:
            self.send_error(502, "camera not reachable")


_PKT_COUNT = [0]


def _emitter_loop_wrapped():
    start = time.time()
    while True:
        with LOCK:
            playing = STATE["playing"]
            targets = [dict(t) for t in STATE["targets"]]
            ref = STATE["ref"]
            hz = STATE["hz"]
        if not playing:
            time.sleep(0.2)
            continue
        elapsed = time.time() - start
        tracks = []
        for t in targets:
            if not t.get("enabled") or (t["lat"] == 0 and t["lon"] == 0):
                continue
            lat, lon, alt, vx, vy = _target_pos(t, elapsed)
            if ref is None:
                ref = (lat, lon, alt)
                with LOCK:
                    STATE["ref"] = ref
            polar = _polar(ref[0], ref[1], ref[2], lat, lon, alt)
            tracks.append(dict(lat=lat, lon=lon, h_msl_m=alt, vx=vx, vy=vy,
                               track_no=t["id"], classif="unknown", **polar))
        if tracks:
            data = encode_eketa_cat062(tracks)
            try:
                SOCK.sendto(data, (JETSON_IP, JETSON_AST))
            except Exception:
                pass
            _PKT_COUNT[0] += 1
        time.sleep(1.0 / hz)


def main():
    global JETSON_IP, JETSON_HTTP, JETSON_AST
    p = argparse.ArgumentParser(description="Ground planner: map + fake ASTERIX emitter")
    p.add_argument("--jetson", default="192.168.205.160", help="Jetson IP (default 192.168.205.160)")
    p.add_argument("--jetson-http", type=int, default=8080, help="Jetson HTTP port (default 8080)")
    p.add_argument("--jetson-ast", type=int, default=8600, help="Jetson ASTERIX UDP port (default 8600)")
    p.add_argument("--port", type=int, default=9000, help="Local HTTP port (default 9000)")
    args = p.parse_args()
    JETSON_IP = args.jetson
    JETSON_HTTP = args.jetson_http
    JETSON_AST = args.jetson_ast

    t = threading.Thread(target=_emitter_loop_wrapped, daemon=True)
    t.start()

    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[GROUND PLANNER] http://localhost:{args.port}")
    print(f"[GROUND PLANNER] Jetson: {JETSON_IP}:{JETSON_HTTP} (HTTP), :{JETSON_AST} (ASTERIX UDP)")
    print(f"[GROUND PLANNER] Camera: http://{JETSON_IP}:{JETSON_HTTP}/stream.mjpg")
    print(f"[GROUND PLANNER] Draw targets on the map, hit PLAY to send ASTERIX packets.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[GROUND PLANNER] Stopped.")


if __name__ == "__main__":
    main()
