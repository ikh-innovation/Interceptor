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
    "playing": True,
    "targets": [],
    "next_id": 1,
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
.sidebar{width:300px;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px;
  border-left:1px solid var(--border);background:var(--panel)}
.card{padding:10px 12px;border-radius:10px;border:1px solid var(--border);background:var(--bg)}
.card h3{font-size:13px;margin:0 0 6px;display:flex;align-items:center;gap:8px}
.card h3 .dot{width:10px;height:10px;border-radius:50%;flex:0 0 auto}
.card h3 .spacer{flex:1}
.card .del-btn{background:none;border:none;color:var(--err);cursor:pointer;font-size:16px;
  padding:0 4px;opacity:.6}
.card .del-btn:hover{opacity:1}
.row{display:flex;gap:6px;align-items:center;margin:3px 0}
.row label{font-size:11px;color:var(--muted);min-width:46px}
.row input,.row select{flex:1;background:var(--panel);border:1px solid var(--border);color:var(--text);
  border-radius:6px;padding:4px 6px;font-size:12px}
.row input[type=checkbox]{flex:0;width:16px;height:16px}
.row .unit{font-size:11px;color:var(--muted);min-width:24px}
.drone-info{font-size:11px;color:var(--muted);line-height:1.5}
.drone-info b{color:var(--text)}
.empty-hint{font-size:12px;color:var(--muted);text-align:center;padding:16px 8px;line-height:1.6}
.go-btn{background:var(--ok);color:#04121c;border:none;border-radius:6px;padding:5px 12px;
  font:700 12px/1 sans-serif;cursor:pointer;letter-spacing:.5px}
.go-btn:hover{opacity:.85}
@media(max-width:700px){
  .main{flex-direction:column}
  .sidebar{width:100%;max-height:50vh;border-left:none;border-top:1px solid var(--border)}
  .map-wrap{min-height:40vh}
}
</style></head><body>
<header>
  <h1>Ground Planner</h1>
  <span class="status">Drone: <b id="drone_st" class="off">no GPS</b></span>
  <span class="status">TX: <b id="tx_st">auto</b> <span id="tx_n">0</span> pkt</span>
</header>
<div class="main">
  <div class="map-wrap">
    <div id="map"></div>
    <div class="toolbar">
      <button id="draw_circle">&#9711; Circle</button>
      <button id="draw_line">&#9586; Line</button>
    </div>
    <div class="draw-hint" id="draw_hint"></div>
  </div>
  <div class="sidebar">
    <div class="card drone-info" id="drone_card">
      <b>Drone</b><br>
      Lat: <span id="d_lat">--</span> Lon: <span id="d_lon">--</span><br>
      Alt: <span id="d_alt">--</span> m &nbsp; Mode: <span id="d_mode">--</span>
      &nbsp; Armed: <span id="d_armed">--</span>
    </div>
    <div id="target_list"></div>
    <div class="empty-hint" id="empty_hint">Click <b>Circle</b> or <b>Line</b> on the map toolbar, then click the map to draw targets.<br>Targets send to the Jetson automatically.</div>
  </div>
</div>

<script>
var JETSON = "JETSON_PLACEHOLDER";
var map = L.map("map", {zoomControl: true}).setView([37.9, 23.7], 16);
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
  maxZoom: 19, attribution: "Esri"
}).addTo(map);

var COLORS = ["#4fc3f7","#f59e0b","#f87171","#a78bfa","#4ade80","#fb923c","#38bdf8","#e879f9"];
var droneMarker = null;
var centered = false;
var targets = [];
var drawMode = null; // "circle" | "line" | null
var drawStep = null; // null | {type, lat, lon}
var animStart = Date.now();

function hint(msg) {
  var el = document.getElementById("draw_hint");
  el.textContent = msg || "";
  el.style.display = msg ? "" : "none";
}

function tgtColor(i) { return COLORS[i % COLORS.length]; }

function droneIcon() {
  return L.divIcon({className:"", html:
    '<svg width="28" height="28" viewBox="-14 -14 28 28"><circle r="10" fill="#4ade80" stroke="#fff" stroke-width="2"/>' +
    '<circle r="3" fill="#fff"/></svg>', iconSize:[28,28], iconAnchor:[14,14]});
}
function tgtIcon(color) {
  return L.divIcon({className:"", html:
    '<svg width="20" height="20" viewBox="-10 -10 20 20"><polygon points="0,-8 6,6 -6,6" fill="'+color+'" stroke="#fff" stroke-width="1.5"/></svg>',
    iconSize:[20,20], iconAnchor:[10,10]});
}

// ---- target data model ----
function addTarget(type, lat, lon, lat2, lon2) {
  var id = targets.length + 1;
  var t = {id:id, enabled:true, type:type, lat:lat, lon:lon, alt:50,
           lat2:lat2||0, lon2:lon2||0, speed:10, radius:200, period:60,
           layers:{}};
  targets.push(t);
  drawTargetLayers(targets.length - 1);
  renderCards();
  syncToServer();
  return t;
}

function removeTarget(idx) {
  clearLayers(idx);
  targets.splice(idx, 1);
  targets.forEach(function(t,i){ t.id = i+1; });
  redrawAllLayers();
  renderCards();
  syncToServer();
}

function clearLayers(idx) {
  var ly = targets[idx].layers;
  if(ly.m) ly.m.remove();
  if(ly.m2) ly.m2.remove();
  if(ly.path) ly.path.remove();
  if(ly.circle) ly.circle.remove();
  if(ly.anim) ly.anim.remove();
  targets[idx].layers = {};
}

function drawTargetLayers(idx) {
  var t = targets[idx], c = tgtColor(idx), ly = t.layers;
  if(ly.m) ly.m.remove();
  if(ly.m2) ly.m2.remove();
  if(ly.path) ly.path.remove();
  if(ly.circle) ly.circle.remove();
  if(!t.lat && !t.lon) return;
  ly.m = L.marker([t.lat,t.lon], {icon:tgtIcon(c), draggable:true}).addTo(map);
  ly.m.on("dragend", function(e){
    t.lat = e.target.getLatLng().lat;
    t.lon = e.target.getLatLng().lng;
    drawTargetLayers(idx);
    renderCards();
    syncToServer();
  });
  if(t.type === "line" && t.lat2 && t.lon2) {
    ly.m2 = L.marker([t.lat2,t.lon2], {icon:tgtIcon(c), draggable:true}).addTo(map);
    ly.m2.on("dragend", function(e){
      t.lat2 = e.target.getLatLng().lat;
      t.lon2 = e.target.getLatLng().lng;
      drawTargetLayers(idx);
      renderCards();
      syncToServer();
    });
    ly.path = L.polyline([[t.lat,t.lon],[t.lat2,t.lon2]], {color:c, weight:2, dashArray:"6 4"}).addTo(map);
  }
  if(t.type === "circle") {
    ly.circle = L.circle([t.lat,t.lon], {radius:t.radius, color:c, weight:2, fill:false, dashArray:"6 4"}).addTo(map);
  }
}

function redrawAllLayers() {
  targets.forEach(function(_,i){ drawTargetLayers(i); });
}

// ---- sidebar cards ----
function renderCards() {
  var list = document.getElementById("target_list");
  list.innerHTML = "";
  document.getElementById("empty_hint").style.display = targets.length ? "none" : "";
  targets.forEach(function(t, idx) {
    var c = tgtColor(idx);
    var card = document.createElement("div");
    card.className = "card";
    var typeLabel = t.type === "circle" ? "Circle" : "Line";
    var extra = "";
    if(t.type === "circle") {
      extra = '<div class="row"><label>Radius</label><input type="number" value="'+t.radius+'" step="10" data-idx="'+idx+'" data-f="radius"><span class="unit">m</span></div>'
            + '<div class="row"><label>Period</label><input type="number" value="'+t.period+'" step="5" data-idx="'+idx+'" data-f="period"><span class="unit">s</span></div>'
            + '<div class="row"><label>Speed</label><input type="number" value="'+t.speed+'" step="1" data-idx="'+idx+'" data-f="speed"><span class="unit">m/s</span></div>';
    } else {
      extra = '<div class="row"><label>Speed</label><input type="number" value="'+t.speed+'" step="1" data-idx="'+idx+'" data-f="speed"><span class="unit">m/s</span></div>';
    }
    card.innerHTML = '<h3><span class="dot" style="background:'+c+'"></span>T'+t.id+' '+typeLabel
      + '<span class="spacer"></span>'
      + '<button class="go-btn" data-goidx="'+idx+'" title="Select this target on the drone">GO</button>'
      + '<input type="checkbox" '+(t.enabled?"checked":"")+ ' data-idx="'+idx+'" data-f="enabled" title="Enable/disable">'
      + '<button class="del-btn" data-delidx="'+idx+'" title="Delete">&times;</button></h3>'
      + '<div class="row"><label>Alt</label><input type="number" value="'+t.alt+'" step="5" data-idx="'+idx+'" data-f="alt"><span class="unit">m</span></div>'
      + extra;
    list.appendChild(card);
  });
  list.querySelectorAll("input[data-f]").forEach(function(inp){
    inp.addEventListener("change", function(){
      var idx = parseInt(inp.dataset.idx);
      var f = inp.dataset.f;
      if(f === "enabled") targets[idx].enabled = inp.checked;
      else targets[idx][f] = parseFloat(inp.value) || 0;
      drawTargetLayers(idx);
      syncToServer();
    });
  });
  list.querySelectorAll("[data-delidx]").forEach(function(btn){
    btn.addEventListener("click", function(){ removeTarget(parseInt(btn.dataset.delidx)); });
  });
  list.querySelectorAll("[data-goidx]").forEach(function(btn){
    btn.addEventListener("click", function(){
      var idx = parseInt(btn.dataset.goidx);
      var t = targets[idx];
      fetch("/drone/cmd?q=msn%3Dcue%3Aselect%3ATRK" + t.id).then(function(r){ return r.json(); }).then(function(j){
        btn.textContent = j.ok === "" ? "SENT" : j.ok || "err";
        setTimeout(function(){ btn.textContent = "GO"; }, 1500);
      }).catch(function(){ btn.textContent = "offline"; setTimeout(function(){ btn.textContent = "GO"; }, 1500); });
    });
  });
}

// ---- sync to server ----
function syncToServer() {
  var data = targets.map(function(t){
    return {id:t.id, enabled:t.enabled, type:t.type, lat:t.lat, lon:t.lon, alt:t.alt,
            lat2:t.lat2, lon2:t.lon2, speed:t.speed, radius:t.radius, period:t.period};
  });
  fetch("/api/targets", {method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({targets:data})});
}

// ---- drawing ----
document.getElementById("draw_circle").onclick = function(){
  if(drawMode === "circle"){ drawMode = null; drawStep = null; hint(""); this.classList.remove("active"); return; }
  drawMode = "circle"; drawStep = null;
  document.getElementById("draw_circle").classList.add("active");
  document.getElementById("draw_line").classList.remove("active");
  hint("Click map to place circle center");
};
document.getElementById("draw_line").onclick = function(){
  if(drawMode === "line"){ drawMode = null; drawStep = null; hint(""); this.classList.remove("active"); return; }
  drawMode = "line"; drawStep = null;
  document.getElementById("draw_line").classList.add("active");
  document.getElementById("draw_circle").classList.remove("active");
  hint("Click map for line start point");
};

map.on("click", function(e) {
  if(!drawMode) return;
  if(drawMode === "circle") {
    addTarget("circle", e.latlng.lat, e.latlng.lng, 0, 0);
    hint("Circle placed. Click again for another, or click Circle to stop.");
  } else if(drawMode === "line") {
    if(!drawStep) {
      drawStep = {type:"line", lat:e.latlng.lat, lon:e.latlng.lng};
      hint("Click map for line end point");
    } else {
      addTarget("line", drawStep.lat, drawStep.lon, e.latlng.lat, e.latlng.lng);
      drawStep = null;
      hint("Line placed. Click again for another, or click Line to stop.");
    }
  }
});

document.addEventListener("keydown", function(e){
  if(e.key === "Escape"){ drawMode = null; drawStep = null; hint("");
    document.getElementById("draw_circle").classList.remove("active");
    document.getElementById("draw_line").classList.remove("active"); }
});

// ---- drone polling ----
function pollDrone() {
  fetch("/drone/state").then(function(r){ return r.ok ? r.json() : null; }).then(function(j) {
    if (!j) { document.getElementById("drone_st").className = "off"; document.getElementById("drone_st").textContent = "offline"; return; }
    document.getElementById("d_mode").textContent = j.fc_mode || "--";
    document.getElementById("d_armed").textContent = j.armed ? "YES" : "no";
    document.getElementById("drone_st").className = j.linked ? "" : "off";
    document.getElementById("drone_st").textContent = j.linked ? "linked" : "no FC";
    var m = j.mission || {};
    var pos = m.position, alt = m.height_m;
    document.getElementById("d_alt").textContent = alt != null ? alt.toFixed(1) : "--";
    if (pos && pos.length === 2 && pos[0]) {
      updateDrone(pos[0], pos[1], alt);
    } else if (!centered && m.home && m.home.length === 2 && m.home[0]) {
      map.setView(m.home, 17); centered = true;
    }
    var sk = m.seeker || m;
    var cue = (sk.cue || {});
    if (cue.packets != null) document.getElementById("tx_n").textContent = cue.packets + " rx";
  }).catch(function(){
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

// ---- animate moving targets on map ----
function animateTargets() {
  var elapsed = (Date.now() - animStart) / 1000;
  targets.forEach(function(t, i) {
    if (!t.enabled || (!t.lat && !t.lon)) { if(t.layers.anim){t.layers.anim.remove();t.layers.anim=null;} return; }
    var lat, lon;
    if (t.type === "circle") {
      var a = 2 * Math.PI * (elapsed / Math.max(1, t.period));
      lat = t.lat + t.radius * Math.cos(a) / 111320;
      lon = t.lon + t.radius * Math.sin(a) / (111320 * Math.cos(t.lat * Math.PI / 180));
    } else if (t.type === "line" && t.lat2 && t.lon2) {
      var d = Math.hypot((t.lat2-t.lat)*111320, (t.lon2-t.lon)*111320*Math.cos(t.lat*Math.PI/180));
      var period = d > 0.1 ? (d / Math.max(0.1, t.speed)) * 2 : 1;
      var ft = (elapsed % period) / period;
      var frac = ft < 0.5 ? 2*ft : 2*(1-ft);
      lat = t.lat + frac*(t.lat2-t.lat);
      lon = t.lon + frac*(t.lon2-t.lon);
    }
    if (lat && lon) {
      var c = tgtColor(i);
      if (!t.layers.anim) t.layers.anim = L.circleMarker([lat,lon], {radius:6, color:c, fillColor:c, fillOpacity:.9, weight:2}).addTo(map);
      else t.layers.anim.setLatLng([lat,lon]);
    }
  });
  requestAnimationFrame(animateTargets);
}
requestAnimationFrame(animateTargets);

setInterval(pollDrone, 2000);
pollDrone();
setInterval(function(){
  fetch("/api/status").then(function(r){return r.json();}).then(function(j){
    document.getElementById("tx_n").textContent = j.packets + " pkt";
    document.getElementById("tx_st").textContent = j.targets > 0 ? "sending" : "idle";
    document.getElementById("tx_st").className = j.targets > 0 ? "" : "off";
  }).catch(function(){});
}, 1000);

renderCards();
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
        elif path == "/drone/cmd":
            qs = urlparse(self.path).query
            self._proxy_json(f"http://{JETSON_IP}:{JETSON_HTTP}/cmd?{qs}")
        elif path == "/api/status":
            with LOCK:
                ntgt = len([t for t in STATE["targets"] if t.get("enabled") and (t.get("lat") or t.get("lon"))])
            self._json({"targets": ntgt, "packets": _PKT_COUNT[0]})
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

_PKT_COUNT = [0]


def _emitter_loop_wrapped():
    start = time.time()
    while True:
        with LOCK:
            tgt_list = [dict(t) for t in STATE["targets"]]
            ref = STATE["ref"]
            hz = STATE["hz"]
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

    # Kill any previous instance on this port
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
    print(f"[GROUND PLANNER] Draw targets on the map — they send automatically.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[GROUND PLANNER] Stopped.")


if __name__ == "__main__":
    main()
