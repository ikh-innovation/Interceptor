"""Cloud Function: generate a Greek flight report using Vertex AI.

Triggered by Firestore writes to flight_logs/{jetson_id}/flights/{flight_id}
when status changes to "complete".  Reads event data from the subcollection,
calls Gemini to produce a human-readable Greek report, and writes it back
to the flight document as `report_el`.
"""

import json
import os

from google.cloud import firestore
import vertexai
from vertexai.generative_models import GenerativeModel

PROJECT = os.environ.get("GCP_PROJECT", "ikh-drone-cloud")
REGION = os.environ.get("GCP_REGION", "europe-west1")
MODEL_ID = "gemini-2.0-flash-001"


def generate_flight_report(event, context):
    """Firestore trigger: flight_logs/{jetson_id}/flights/{flight_id}"""
    resource = context.resource
    # resource looks like:
    #   projects/.../databases/(default)/documents/flight_logs/{jetson_id}/flights/{flight_id}
    parts = resource.split("/documents/")[-1].split("/")
    # parts = ["flight_logs", jetson_id, "flights", flight_id]
    if len(parts) < 4:
        return
    jetson_id = parts[1]
    flight_id = parts[3]

    new_value = event.get("value", {}).get("fields", {})
    status = _str_field(new_value.get("status"))
    if status != "complete":
        return

    if _str_field(new_value.get("report_el")):
        return

    db = firestore.Client(project=PROJECT)
    flight_ref = db.collection("flight_logs").document(jetson_id).collection("flights").document(flight_id)
    flight_doc = flight_ref.get()
    if not flight_doc.exists:
        return
    meta = flight_doc.to_dict()

    events_ref = flight_ref.collection("events")
    events = [doc.to_dict() for doc in events_ref.order_by("elapsed_s").stream()]

    rows_ref = flight_ref.collection("rows")
    rows = [doc.to_dict() for doc in rows_ref.order_by("t_s").limit(200).stream()]

    prompt = _build_prompt(jetson_id, flight_id, meta, events, rows)

    vertexai.init(project=PROJECT, location=REGION)
    model = GenerativeModel(MODEL_ID)
    response = model.generate_content(prompt)
    report_text = response.text

    flight_ref.update({"report_el": report_text})


def _str_field(field):
    if field is None:
        return ""
    if isinstance(field, dict):
        return field.get("stringValue", "")
    return str(field)


def _build_prompt(jetson_id, flight_id, meta, events, rows):
    flight_mode = meta.get("flight_mode", "unknown")
    duration = meta.get("duration_s", "?")
    end_reason = meta.get("end_reason", "?")
    started = meta.get("started_utc", "?")
    settings = json.dumps(meta.get("settings", {}), indent=2, ensure_ascii=False, default=str)

    events_text = "\n".join(
        f"  [{e.get('elapsed_s', '?')}s] {e.get('event', '?')}: "
        f"{e.get('value', e.get('message', e.get('reason', '')))}"
        for e in events[:100]
    )

    sample_rows = ""
    if rows:
        keys = ["t_s", "phase", "height_m", "lat", "lon", "detection_fresh",
                "wasp_stage", "seeker_active", "seeker_mission", "seeker_phase",
                "fc_mode"]
        for r in rows[:30]:
            vals = ", ".join(f"{k}={r.get(k, '')}" for k in keys if r.get(k, "") != "")
            sample_rows += f"  {vals}\n"

    return f"""Είσαι ειδικός αναλυτής πτήσεων drone. Γράψε μια λεπτομερή αναφορά πτήσης στα Ελληνικά.

Πληροφορίες πτήσης:
- Drone: {jetson_id}
- Flight ID: {flight_id}
- Τύπος: {flight_mode}
- Ξεκίνησε: {started}
- Διάρκεια: {duration} δευτερόλεπτα
- Λόγος τερματισμού: {end_reason}

Ρυθμίσεις:
{settings}

Γεγονότα (χρονολογικά):
{events_text}

Δείγμα δεδομένων τηλεμετρίας:
{sample_rows}

Γράψε μια πλήρη αναφορά πτήσης που περιλαμβάνει:
1. ΠΕΡΙΛΗΨΗ: Τι τύπος πτήσης ήταν, πόσο κράτησε, και ποιο ήταν το αποτέλεσμα.
2. ΧΡΟΝΟΛΟΓΙΟ: Τα βασικά γεγονότα της πτήσης σε χρονολογική σειρά.
3. ΑΝΙΧΝΕΥΣΕΙΣ: Αν υπήρξαν ανιχνεύσεις στόχου, πότε και με τι confidence.
4. ΦΑΣΕΙΣ ΠΤΗΣΗΣ: Ποιες φάσεις πέρασε η πτήση (IDLE, WASP_FLIGHT, SEEK, κλπ).
5. ΑΞΙΟΛΟΓΗΣΗ: Πώς πήγε η πτήση συνολικά. Υπήρχαν προβλήματα;
6. ΠΑΡΑΤΗΡΗΣΕΙΣ: Οτιδήποτε αξιοσημείωτο.

Κράτα την αναφορά επαγγελματική αλλά κατανοητή. Χρησιμοποίησε τεχνικούς όρους αλλά εξήγησε τι σημαίνουν αν χρειαστεί."""
