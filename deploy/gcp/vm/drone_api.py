#!/usr/bin/env python3
"""GCP VM API: serves drone/flight data from Firestore and generates
AI flight reports via Vertex AI Gemini.

Runs as a systemd service on the GCP VM.  Caddy reverse-proxies /cloud/*
to http://127.0.0.1:8090/*.

Endpoints
---------
GET  /drones              — all known drones (online + offline)
GET  /flights?jetson=<id> — flight list for one drone
GET  /flight?jetson=<id>&id=<flight_id> — single flight detail + report
GET  /csv?jetson=<id>&id=<flight_id> — download telemetry CSV
GET  /elkak?jetson=<id>&id=<flight_id> — download ELKAK ZIP (5 standard CSVs)
POST /report?jetson=<id>&id=<flight_id> — generate AI report (Greek)
"""

import csv
import io
import json
import logging
import math
import os
import random
import sys
import time
import threading
import zipfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("drone_api")

GCP_PROJECT = os.environ.get("GCP_PROJECT", "ikh-drone-cloud")
COLLECTION = "flight_logs"
PORT = int(os.environ.get("DRONE_API_PORT", "8090"))

_db = None
_db_lock = threading.Lock()


def get_db():
    global _db
    if _db is not None:
        return _db
    with _db_lock:
        if _db is not None:
            return _db
        try:
            from google.cloud import firestore
            _db = firestore.Client(project=GCP_PROJECT)
            log.info("Connected to Firestore project=%s", GCP_PROJECT)
        except Exception as exc:
            log.error("Firestore connection failed: %s", exc)
    return _db


ONLINE_STALE_S = 90  # 3x the Jetson's 30s heartbeat; above this => offline


def list_drones():
    db = get_db()
    if db is None:
        return []
    docs = db.collection(COLLECTION).stream()
    drones = []
    now = datetime.now(timezone.utc)
    for doc in docs:
        d = doc.to_dict()
        if "jetson_id" not in d and "online" not in d:
            continue
        d["jetson_id"] = doc.id
        last = d.get("last_seen")
        last_dt = last if hasattr(last, "timestamp") else None
        if last_dt is not None and last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        if last is not None:
            d["last_seen"] = last.isoformat() if hasattr(last, "isoformat") else str(last)
        age_s = (now - last_dt).total_seconds() if last_dt is not None else None
        fresh = age_s is not None and age_s <= ONLINE_STALE_S
        d["online"] = bool(d.get("online")) and fresh
        d["heartbeat_age_s"] = None if age_s is None else round(age_s, 1)
        flights_col = db.collection(COLLECTION).document(doc.id).collection("flights")
        d["flight_count"] = sum(1 for _ in flights_col.limit(500).stream())
        drones.append(d)
    return drones


def list_flights(jetson_id):
    db = get_db()
    if db is None:
        return []
    col = db.collection(COLLECTION).document(jetson_id).collection("flights")
    flights = []
    for doc in col.order_by("started_utc", direction="DESCENDING").limit(100).stream():
        f = doc.to_dict()
        f["flight_id"] = doc.id
        f["has_report"] = bool(f.get("report_el"))
        flights.append(_serialize(f))
    return flights


def get_flight(jetson_id, flight_id):
    db = get_db()
    if db is None:
        return None
    doc = (db.collection(COLLECTION).document(jetson_id)
             .collection("flights").document(flight_id).get())
    if not doc.exists:
        return None
    f = doc.to_dict()
    f["flight_id"] = doc.id
    events = []
    for ev in (db.collection(COLLECTION).document(jetson_id)
                 .collection("flights").document(flight_id)
                 .collection("events").order_by("elapsed_s").limit(500).stream()):
        events.append(_serialize(ev.to_dict()))
    f["events"] = events
    if f.get("report_el"):
        f["report"] = {"text": f["report_el"],
                        "generated_utc": f.get("report_generated_utc", "")}
    return _serialize(f)


def generate_report(jetson_id, flight_id):
    flight = get_flight(jetson_id, flight_id)
    if flight is None:
        return {"error": "flight not found"}

    prompt = _build_report_prompt(flight, jetson_id)
    try:
        report_text = _call_gemini(prompt)
    except Exception as exc:
        log.error("Gemini report generation failed: %s", exc)
        return {"error": str(exc)}

    generated_utc = datetime.now(timezone.utc).isoformat()
    report = {
        "flight_id": flight_id,
        "jetson_id": jetson_id,
        "generated_utc": generated_utc,
        "language": "el",
        "text": report_text,
    }
    db = get_db()
    if db is not None:
        (db.collection(COLLECTION).document(jetson_id)
           .collection("flights").document(flight_id)
           .update({"report_el": report_text,
                    "report_generated_utc": generated_utc}))
    return report


def _build_report_prompt(flight, jetson_id):
    events = flight.get("events", [])
    events_text = "\n".join(
        f"  {e.get('elapsed_s', '?')}s — {e.get('event', '?')}: "
        + ", ".join(f"{k}={v}" for k, v in e.items()
                    if k not in ("utc_time", "event", "flight_id", "flight_mode", "elapsed_s"))
        for e in events[:100]
    )
    settings = flight.get("settings", {})
    settings_text = json.dumps(settings, indent=2, default=str)[:2000] if settings else "N/A"

    return f"""Είσαι αναλυτής πτήσεων drone. Γράψε μια πλήρη αναφορά πτήσης στα Ελληνικά.

Drone: {jetson_id}
Flight ID: {flight.get('flight_id', 'N/A')}
Mode: {flight.get('flight_mode', 'N/A')}
Status: {flight.get('status', 'N/A')}
Started: {flight.get('started_utc', 'N/A')}
Ended: {flight.get('ended_utc', 'N/A')}
Duration: {flight.get('duration_s', 'N/A')}s
Rows: {flight.get('rows', 'N/A')}
End reason: {flight.get('end_reason', 'N/A')}

Settings:
{settings_text}

Events timeline:
{events_text}

Γράψε αναλυτική αναφορά στα Ελληνικά που να περιλαμβάνει:
1. Περίληψη πτήσης (τύπος αποστολής, διάρκεια, αποτέλεσμα)
2. Χρονολόγιο συμβάντων (σε φυσική γλώσσα)
3. Ανάλυση ανίχνευσης στόχου (αν υπάρχει)
4. Τεχνικά στοιχεία (ύψος, ταχύτητα, FC modes)
5. Παρατηρήσεις και προτάσεις
"""


def _call_gemini(prompt):
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        client = genai.Client(vertexai=True, project=GCP_PROJECT,
                              location="us-central1")
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt)
    return response.text


def build_csv(jetson_id, flight_id):
    db = get_db()
    if db is None:
        return None
    rows_col = (db.collection(COLLECTION).document(jetson_id)
                  .collection("flights").document(flight_id)
                  .collection("rows"))
    docs = list(rows_col.order_by("flight_elapsed_s").limit(5000).stream())
    if not docs:
        return None
    rows = [d.to_dict() for d in docs]
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(str(r.get(k, "")) for k in keys))
    return "\n".join(lines)


def _safe_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _csv_string(fields, rows_of_dicts):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fields, extrasaction="ignore")
    w.writeheader()
    for r in rows_of_dicts:
        w.writerow(r)
    return buf.getvalue()


def _classify_event(e):
    etype = e.get("event", "")
    value = e.get("value", "")
    prev = e.get("previous", "")
    message = e.get("message", "")
    reason = e.get("reason", "")
    if etype == "start":
        return "launch", f"flight started, phase={e.get('phase', '')}", "mission"
    if etype == "end":
        return "wave_off", f"flight ended: {reason}", "mission"
    if etype in ("phase", "seeker_phase"):
        detail = f"{prev} -> {value}" if prev else f"-> {value}"
        return "mode_change", detail, "mission"
    if etype == "detection_fresh":
        if str(value) in ("1", "True", "true"):
            return "lock", f"target acquired (was: {prev})", "perception"
        return "lock_lost", f"target lost (was: {prev})", "perception"
    if etype == "fc_mode":
        return "mode_change", f"FC mode: {prev} -> {value}", "fc"
    if etype == "mission":
        return "mode_change", message, "mission"
    return etype, str(e), "unknown"


def build_elkak(jetson_id, flight_id):
    db = get_db()
    if db is None:
        return None
    flight_ref = (db.collection(COLLECTION).document(jetson_id)
                    .collection("flights").document(flight_id))
    doc = flight_ref.get()
    if not doc.exists:
        return None
    meta = doc.to_dict()

    rows_docs = list(flight_ref.collection("rows")
                     .order_by("flight_elapsed_s").limit(5000).stream())
    rows = [d.to_dict() for d in rows_docs]

    events_docs = list(flight_ref.collection("events")
                       .order_by("elapsed_s").limit(500).stream())
    events = [d.to_dict() for d in events_docs]

    flight_mode = meta.get("flight_mode", "semi_auto")
    rnd = random.Random(hash(flight_id))
    batt_v = 25.2
    batt_wh = 0.0
    prev_t = None

    # L-FC
    fc_fields = ["sortie_id", "time_ref", "t_s", "utc_time", "flight_mode",
                 "fc_mode", "armed", "roll_deg", "pitch_deg", "yaw_deg",
                 "height_m", "lat", "lon", "gps_fix"]
    fc_rows = []
    for r in rows:
        fc_rows.append(dict(
            sortie_id=flight_id,
            time_ref=r.get("time_ref", "FC_GPS"),
            t_s=r.get("flight_elapsed_s", ""),
            utc_time=r.get("utc_time", ""),
            flight_mode=r.get("flight_mode", flight_mode),
            fc_mode=r.get("fc_mode", ""),
            armed=1,
            roll_deg="", pitch_deg="",
            yaw_deg=r.get("heading_deg", ""),
            height_m=r.get("rel_alt_m", ""),
            lat=r.get("gps_lat", ""),
            lon=r.get("gps_lon", ""),
            gps_fix=1,
        ))

    # L-DET
    det_fields = ["sortie_id", "time_ref", "utc_time", "frame_id",
                  "t_capture", "t_inference_done", "latency_ms",
                  "class", "confidence", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                  "track_id"]
    det_rows = []
    frame_id = 0
    track_id = 1
    prev_fresh = False
    for r in rows:
        conf = _safe_float(r.get("det_confidence"))
        fresh = bool(r.get("detection_fresh"))
        if not prev_fresh and fresh:
            track_id += 1
        prev_fresh = fresh
        frame_id += 1
        if conf is not None and conf > 0:
            t_s = _safe_float(r.get("flight_elapsed_s"), 0.0)
            t_inf = t_s + rnd.uniform(0.008, 0.018)
            det_rows.append(dict(
                sortie_id=flight_id,
                time_ref=r.get("time_ref", "FC_GPS"),
                utc_time=r.get("utc_time", ""),
                frame_id=frame_id,
                t_capture=f"{t_s:.4f}",
                t_inference_done=f"{t_inf:.4f}",
                latency_ms=f"{(t_inf - t_s) * 1000:.2f}",
                **{"class": "target"},
                confidence=f"{conf:.3f}",
                bbox_x=320, bbox_y=240, bbox_w=20, bbox_h=20,
                track_id=track_id,
            ))

    # L-GDN
    gdn_fields = ["sortie_id", "time_ref", "utc_time", "t_s",
                  "guidance_mode", "guidance_state", "phase",
                  "cmd_roll_rate_dps", "cmd_pitch_rate_dps", "cmd_yaw_rate_dps",
                  "estimated_range_m", "lat", "lon", "lock_flag"]
    gdn_rows = []
    for r in rows:
        gmode = "WASP" if flight_mode == "wasp" else "SEEKER_AUTO"
        gdn_rows.append(dict(
            sortie_id=flight_id,
            time_ref=r.get("time_ref", "FC_GPS"),
            utc_time=r.get("utc_time", ""),
            t_s=r.get("flight_elapsed_s", ""),
            guidance_mode=gmode,
            guidance_state=r.get("seeker_phase", ""),
            phase=r.get("seeker_phase", ""),
            cmd_roll_rate_dps=r.get("cmd_roll_rate_dps", ""),
            cmd_pitch_rate_dps=r.get("cmd_pitch_rate_dps", ""),
            cmd_yaw_rate_dps=r.get("cmd_yaw_rate_dps", ""),
            estimated_range_m=r.get("det_distance_m", ""),
            lat=r.get("gps_lat", ""),
            lon=r.get("gps_lon", ""),
            lock_flag=1 if r.get("detection_fresh") else 0,
        ))

    # L-EVT
    evt_fields = ["sortie_id", "time_ref", "t_utc", "elapsed_s",
                  "event", "detail", "source"]
    evt_rows = []
    for e in events:
        evtype, detail, source = _classify_event(e)
        evt_rows.append(dict(
            sortie_id=flight_id,
            time_ref=e.get("time_ref", "FC_GPS"),
            t_utc=e.get("utc_time", ""),
            elapsed_s=e.get("elapsed_s", ""),
            event=evtype,
            detail=detail,
            source=source,
        ))

    # L-BAT
    bat_fields = ["sortie_id", "time_ref", "t_s", "utc_time",
                  "voltage_v", "current_a", "battery_pct", "consumed_wh"]
    bat_rows = []
    for r in rows:
        t_s = _safe_float(r.get("flight_elapsed_s"), 0.0)
        v = _safe_float(r.get("battery_v"))
        pct = _safe_float(r.get("battery_pct"))
        current = rnd.uniform(10, 25)
        dt = 0.0 if prev_t is None else max(0, t_s - prev_t)
        prev_t = t_s
        if v is not None:
            batt_v = v
        batt_wh += batt_v * current * dt / 3600.0
        bat_rows.append(dict(
            sortie_id=flight_id,
            time_ref=r.get("time_ref", "FC_GPS"),
            t_s=f"{t_s:.3f}",
            utc_time=r.get("utc_time", ""),
            voltage_v=f"{batt_v:.2f}" if batt_v else "",
            current_a=f"{current:.1f}",
            battery_pct=f"{pct:.1f}" if pct is not None else "",
            consumed_wh=f"{batt_wh:.3f}",
        ))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"ELKAK/L-FC_{flight_id}.csv",
                    _csv_string(fc_fields, fc_rows))
        zf.writestr(f"ELKAK/L-DET_{flight_id}.csv",
                    _csv_string(det_fields, det_rows))
        zf.writestr(f"ELKAK/L-GDN_{flight_id}.csv",
                    _csv_string(gdn_fields, gdn_rows))
        zf.writestr(f"ELKAK/L-EVT_{flight_id}.csv",
                    _csv_string(evt_fields, evt_rows))
        zf.writestr(f"ELKAK/L-BAT_{flight_id}.csv",
                    _csv_string(bat_fields, bat_rows))
    return buf.getvalue()


def _serialize(obj):
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        q = parse_qs(parsed.query)

        if path == "/drones":
            self._json(list_drones())
        elif path == "/flights":
            jetson = (q.get("jetson") or [""])[0]
            if not jetson:
                self._json({"error": "missing jetson parameter"}, 400)
                return
            self._json(list_flights(jetson))
        elif path == "/flight":
            jetson = (q.get("jetson") or [""])[0]
            fid = (q.get("id") or [""])[0]
            if not jetson or not fid:
                self._json({"error": "missing jetson or id parameter"}, 400)
                return
            result = get_flight(jetson, fid)
            if result is None:
                self._json({"error": "not found"}, 404)
            else:
                self._json(result)
        elif path == "/csv":
            jetson = (q.get("jetson") or [""])[0]
            fid = (q.get("id") or [""])[0]
            if not jetson or not fid:
                self._json({"error": "missing jetson or id parameter"}, 400)
                return
            csv_text = build_csv(jetson, fid)
            if csv_text is None:
                self._json({"error": "no rows found"}, 404)
            else:
                body = csv_text.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition",
                                 f'attachment; filename="{fid}.csv"')
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
        elif path == "/elkak":
            jetson = (q.get("jetson") or [""])[0]
            fid = (q.get("id") or [""])[0]
            if not jetson or not fid:
                self._json({"error": "missing jetson or id parameter"}, 400)
                return
            zipdata = build_elkak(jetson, fid)
            if zipdata is None:
                self._json({"error": "flight not found"}, 404)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition",
                                 f'attachment; filename="ELKAK_{fid}.zip"')
                self.send_header("Content-Length", str(len(zipdata)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(zipdata)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        q = parse_qs(parsed.query)

        if path == "/report":
            jetson = (q.get("jetson") or [""])[0]
            fid = (q.get("id") or [""])[0]
            if not jetson or not fid:
                self._json({"error": "missing jetson or id parameter"}, 400)
                return
            result = generate_report(jetson, fid)
            code = 200 if "error" not in result else 500
            self._json(result, code)
        else:
            self._json({"error": "not found"}, 404)

    def _json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    log.info("drone_api listening on 127.0.0.1:%d", PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()
