"""Stream flight logs to GCP Firestore in real-time.

Runs a background thread that pushes new CSV rows and events to Firestore
as they are written locally.  If the Jetson or drone dies mid-flight, every
row that was flushed to disk is also in the cloud.

Usage (inside tracker.py / fc_control.py):

    uploader = LogUploader(project="your-gcp-project",
                           jetson_id="wasp-01")
    uploader.start()

    # On each log row:
    uploader.push_row(flight_id, row_dict)

    # On events:
    uploader.push_event(flight_id, event_dict)

    # On flight start/end:
    uploader.push_metadata(flight_id, meta_dict)

    uploader.stop()

Requires: google-cloud-firestore  (pip install google-cloud-firestore)
The Jetson must have a service account key or workload identity configured.
"""

import json
import logging
import os
import queue
import threading
import time

log = logging.getLogger("log_uploader")

_BATCH_SIZE = 50
_FLUSH_INTERVAL_S = 2.0


class LogUploader:
    def __init__(self, project=None, jetson_id=None, collection_prefix="flight_logs"):
        self.project = project or os.environ.get("GCP_PROJECT")
        self.jetson_id = jetson_id or os.environ.get("JETSON_ID", "unknown")
        self.collection_prefix = collection_prefix
        self._q = queue.Queue(maxsize=10000)
        self._thread = None
        self._stop = threading.Event()
        self._db = None
        self.error = None
        self.rows_uploaded = 0

    def start(self):
        if not self.project:
            log.warning("LogUploader: no GCP_PROJECT set, uploads disabled")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="log-upload")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _connect(self):
        try:
            from google.cloud import firestore
            self._db = firestore.Client(project=self.project)
            log.info("LogUploader: connected to Firestore project=%s", self.project)
        except Exception as exc:
            self.error = str(exc)
            log.error("LogUploader: Firestore connection failed: %s", exc)

    def _run(self):
        self._connect()
        if self._db is None:
            return
        batch = []
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=_FLUSH_INTERVAL_S)
                batch.append(item)
            except queue.Empty:
                pass
            while not self._q.empty() and len(batch) < _BATCH_SIZE:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            if batch:
                self._flush(batch)
                batch = []
        while not self._q.empty():
            try:
                batch.append(self._q.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._flush(batch)

    def _flush(self, items):
        try:
            fb = self._db.batch()
            for item in items:
                kind = item["kind"]
                flight_id = item["flight_id"]
                col = f"{self.collection_prefix}/{self.jetson_id}/flights/{flight_id}/{kind}"
                doc_ref = self._db.collection(col).document()
                fb.set(doc_ref, item["data"])
            fb.commit()
            self.rows_uploaded += len(items)
        except Exception as exc:
            self.error = str(exc)
            log.error("LogUploader: flush failed (%d items): %s", len(items), exc)

    def push_row(self, flight_id, row_dict):
        self._enqueue("rows", flight_id, row_dict)

    def push_event(self, flight_id, event_dict):
        self._enqueue("events", flight_id, event_dict)

    def push_metadata(self, flight_id, meta_dict):
        if self._db is None:
            return
        try:
            col = f"{self.collection_prefix}/{self.jetson_id}/flights"
            self._db.collection(col).document(flight_id).set(meta_dict, merge=True)
        except Exception as exc:
            self.error = str(exc)

    def _enqueue(self, kind, flight_id, data):
        if self._db is None:
            return
        try:
            safe = {}
            for k, v in data.items():
                if isinstance(v, float) and (v != v):
                    safe[k] = None
                else:
                    safe[k] = v
            self._q.put_nowait(dict(kind=kind, flight_id=flight_id, data=safe))
        except queue.Full:
            pass

    def snapshot(self):
        return dict(
            enabled=self._db is not None,
            project=self.project,
            jetson_id=self.jetson_id,
            queue_size=self._q.qsize(),
            rows_uploaded=self.rows_uploaded,
            error=self.error,
        )
