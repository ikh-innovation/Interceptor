import ast
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tracker


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
