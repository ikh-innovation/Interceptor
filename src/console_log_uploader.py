#!/usr/bin/env python3
"""Tail the wasp-tracker console log and ship batches of lines to Firestore.

Path: console_logs/{JETSON_ID}/sessions/{session_id}/batches/{nnn}

Env:
  GCP_PROJECT                    (required)
  JETSON_ID                      drone id, e.g. wasp-03
  GOOGLE_APPLICATION_CREDENTIALS path to service account key
  WASP_CONSOLE_LOG_PATH          file to tail (default ~/wasp-logs/wasp-tracker.log)
  CONSOLE_LOG_BATCH_S            flush interval, seconds (default 5)
  CONSOLE_LOG_MAX_LINES          max lines per batch doc (default 200)
"""
import os
import socket
import sys
import time
import pathlib
import logging
from datetime import datetime, timezone

from google.cloud import firestore

log = logging.getLogger("console-log-uploader")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

PROJECT   = os.environ.get("GCP_PROJECT")
JETSON    = os.environ.get("JETSON_ID", "unknown")
LOG_PATH  = pathlib.Path(os.environ.get(
    "WASP_CONSOLE_LOG_PATH",
    str(pathlib.Path.home() / "wasp-logs" / "wasp-tracker.log")))
BATCH_S   = float(os.environ.get("CONSOLE_LOG_BATCH_S", "5"))
MAX_LINES = int(os.environ.get("CONSOLE_LOG_MAX_LINES", "200"))


def _flush(session_doc, batch_index, lines):
    session_doc.collection("batches").document(f"{batch_index:06d}").set({
        "ts_utc":      firestore.SERVER_TIMESTAMP,
        "batch_index": batch_index,
        "lines":       lines,
        "line_count":  len(lines),
    })
    log.info("flushed batch %d (%d lines)", batch_index, len(lines))


def main():
    if not PROJECT:
        log.error("GCP_PROJECT not set")
        sys.exit(1)

    db = firestore.Client(project=PROJECT)
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_doc = (db.collection("console_logs")
                     .document(JETSON)
                     .collection("sessions")
                     .document(session_id))
    session_doc.set({
        "started_utc": firestore.SERVER_TIMESTAMP,
        "jetson_id":   JETSON,
        "log_path":    str(LOG_PATH),
        "hostname":    socket.gethostname(),
    })
    log.info("session %s/%s created", JETSON, session_id)

    while not LOG_PATH.exists():
        log.info("waiting for %s to exist...", LOG_PATH)
        time.sleep(2)

    with LOG_PATH.open("r") as fh:
        # Start from end — do NOT re-upload historical content on restart.
        fh.seek(0, 2)
        batch = []
        batch_index = 0
        last_flush = time.monotonic()
        while True:
            line = fh.readline()
            if line:
                batch.append(line.rstrip("\n"))
                if len(batch) >= MAX_LINES:
                    try:
                        _flush(session_doc, batch_index, batch)
                        batch_index += 1
                    except Exception as exc:
                        log.error("flush failed, will retry: %s", exc)
                    else:
                        batch = []
                        last_flush = time.monotonic()
            else:
                time.sleep(0.5)
                if batch and time.monotonic() - last_flush >= BATCH_S:
                    try:
                        _flush(session_doc, batch_index, batch)
                        batch_index += 1
                    except Exception as exc:
                        log.error("flush failed, will retry: %s", exc)
                    else:
                        batch = []
                        last_flush = time.monotonic()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
