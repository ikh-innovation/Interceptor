import ast
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tracker
import log_uploader


class _FakeDoc:
    def __init__(self, store, path):
        self.store, self.path = store, path

    def set(self, data, merge=False):
        self.store.append((self.path, dict(data), merge))


class _FakeCol:
    def __init__(self, store, path):
        self.store, self.path, self._n = store, path, 0

    def document(self, doc_id=None):
        if doc_id is None:
            self._n += 1
            doc_id = f"auto{self._n}"
        return _FakeDoc(self.store, f"{self.path}/{doc_id}")


class _FakeBatch:
    def __init__(self, store):
        self.store, self._ops = store, []

    def set(self, ref, data, merge=False):
        self._ops.append((ref.path, dict(data), merge))

    def commit(self):
        self.store.extend(self._ops)


class _FakeDB:
    def __init__(self):
        self.store = []

    def collection(self, path):
        return _FakeCol(self.store, path)

    def batch(self):
        return _FakeBatch(self.store)


class LogsOnlyTests(unittest.TestCase):
    def test_service_entrypoint_records_video_locally(self):
        tree = ast.parse(Path(tracker.__file__).read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "DroneTracker"]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        record_dir = keywords["record_dir"]
        # Video recording is ON by default now, env-gated, and written to a
        # LOCAL directory -- not the constant None that used to disable it.
        self.assertNotIsInstance(record_dir, ast.Constant)
        source = ast.unparse(record_dir)
        self.assertIn("RECORD_VIDEO", source)   # toggle to disable it
        self.assertIn("RECORD_DIR", source)     # optional location override
        self.assertIn("_DATA_DIR", source)      # default is local, under data/
        self.assertIn("video", source)

    def test_video_is_never_uploaded(self):
        # Videos must stay on the Jetson: the cloud uploader only writes
        # Firestore log documents and must not touch video files or storage.
        uploader = (Path(tracker.__file__).resolve().parent
                    / "log_uploader.py").read_text().lower()
        for token in (".mp4", "storage", "bucket", "upload_from", "blob("):
            self.assertNotIn(token, uploader)


class LogUploaderTests(unittest.TestCase):
    def _uploader(self):
        u = log_uploader.LogUploader(project="p", jetson_id="jet")
        u._db = _FakeDB()          # bypass the real Firestore connect
        return u

    def test_push_metadata_is_queued_not_synchronous(self):
        # The write must NOT happen on the caller's (steering) thread: after
        # push_metadata returns, nothing is written yet -- it is queued for the
        # upload thread, which only writes on _flush.
        u = self._uploader()
        u.push_metadata("flight_A", {"status": "recording"})
        self.assertEqual(u._db.store, [])          # no synchronous write
        self.assertEqual(u._q.qsize(), 1)

    def test_flush_writes_metadata_rows_events_to_correct_paths(self):
        u = self._uploader()
        u.push_metadata("flight_A", {"status": "recording"})
        u.push_event("flight_A", {"event": "start"})
        u.push_row("flight_A", {"rel_alt_m": 12.5, "det_distance_m": float("nan")})
        batch = []
        while not u._q.empty():
            batch.append(u._q.get_nowait())
        u._flush(batch)

        writes = {path: (data, merge) for path, data, merge in u._db.store}
        # Metadata: fixed per-flight doc, merged.
        self.assertIn("flight_logs/jet/flights/flight_A", writes)
        self.assertTrue(writes["flight_logs/jet/flights/flight_A"][1])   # merge=True
        # Rows and events: auto-id docs in their subcollections.
        self.assertTrue(any(p.startswith("flight_logs/jet/flights/flight_A/events/")
                            for p in writes))
        row_path = next(p for p in writes if p.startswith(
            "flight_logs/jet/flights/flight_A/rows/"))
        # NaN is sanitized to None before upload.
        self.assertIsNone(writes[row_path][0]["det_distance_m"])
        self.assertEqual(writes[row_path][0]["rel_alt_m"], 12.5)

    def test_pushes_are_dropped_while_disconnected(self):
        u = log_uploader.LogUploader(project="p", jetson_id="jet")  # _db is None
        u.push_metadata("f", {"a": 1})
        u.push_row("f", {"a": 1})
        u.push_event("f", {"a": 1})
        self.assertEqual(u._q.qsize(), 0)

    def test_logs_only_run_does_not_open_a_video_writer(self):
        t = tracker.DroneTracker.__new__(tracker.DroneTracker)
        t.ap = Mock()
        cap = Mock()
        t._open_capture = Mock(return_value=(cap, 60, "test camera"))
        t.record_dir, t.record_pad_sec = None, 1.5
        t.stream_port, t.headless = None, True
        t.skip_frames, t._rec_on = 0, False
        t._rec_finish = Mock()
        t._loop = Mock()
        with patch.object(tracker.cv2, "VideoWriter") as writer, redirect_stdout(io.StringIO()):
            t._run()
        writer.assert_not_called()
        self.assertFalse(t._loop.call_args.args[3])
        self.assertIsNone(t._loop.call_args.args[6])
        cap.release.assert_called_once()
        t.ap.connect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
