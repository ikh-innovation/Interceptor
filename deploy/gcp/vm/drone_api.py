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
POST /report?jetson=<id>&id=<flight_id> — generate AI report (Greek)
"""

import json
import logging
import os
import sys
import time
import threading
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


def list_drones():
    db = get_db()
    if db is None:
        return []
    docs = db.collection(COLLECTION).stream()
    drones = []
    for doc in docs:
        d = doc.to_dict()
        if "jetson_id" not in d and "online" not in d:
            continue
        d["jetson_id"] = doc.id
        last = d.get("last_seen")
        if last is not None:
            d["last_seen"] = last.isoformat() if hasattr(last, "isoformat") else str(last)
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
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        vertexai.init(project=GCP_PROJECT, location="europe-west1")
        model = GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text
    except ImportError:
        log.warning("vertexai not installed, trying google-generativeai")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("No Vertex AI SDK and no GEMINI_API_KEY set")
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


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
