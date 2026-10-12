"""Synthetic camera motion and live-loop integration; no CSI camera or FC access."""

from collections import deque
from itertools import product
import contextlib
import io
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from tracker import CameraConfig, DroneTracker
from mjpeg_stream import MJPEGServer
from video_stabilization import VideoStabilizer, PreviewWorker, camera_gyro


def texture(w=640, h=480):
    rng = np.random.default_rng(42)
    gray = cv2.GaussianBlur(rng.integers(20, 230, (h, w), np.uint8), (3, 3), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def preview_tracker():
    t = DroneTracker.__new__(DroneTracker)
    t.stabilizer = VideoStabilizer()
    t.video_stabilization = False
    t._stabilization_lock = threading.Lock()
    t._stabilization_generation = t._stabilization_seen = 0
    t._preview_geometry = None
    t._stabilization_status = dict(source="off", ms=0)
    t._preview_worker = t._preview_frame = None
    t.camera_config = CameraConfig()
    t.cam_mode = 30
    t.ap = SimpleNamespace(mission=None, backend=None)
    return t


class StabilizationTests(unittest.TestCase):
    def test_reduces_translation_and_rotation_jitter_at_30_and_60_fps(self):
        base = texture()
        for fps in (30, 60):
            s = VideoStabilizer()
            raw, stabilized = [], []
            for i in range(80):
                dx, dy = 7*np.sin(i*1.7), 5*np.cos(i*1.4)
                angle = 1.2*np.sin(i*1.5)
                motion = cv2.getRotationMatrix2D((319.5, 239.5), angle, 1.0)
                motion[:, 2] += (dx, dy)
                frame = cv2.warpAffine(base, motion, (640, 480))
                result = s.update(frame, 100+i/fps)
                composed = result.matrix[:, :2] @ motion[:, :2]
                center = motion @ np.array([319.5, 239.5, 1])
                displayed = result.matrix @ np.array([*center, 1])
                raw.append((dx, dy, np.radians(angle)))
                stabilized.append((*displayed, np.arctan2(composed[1, 0], composed[0, 0])))
            before, after = np.std(raw[10:], axis=0), np.std(stabilized[10:], axis=0)
            self.assertTrue(np.all(after < before*0.5), (fps, before, after))
            self.assertEqual(s.source, "optical flow")

    def test_entire_frame_fits_at_every_correction_limit_without_scaling(self):
        s = VideoStabilizer()
        for w, h in ((640, 640), (500, 374), (1280, 960), (1280, 720)):
            corners = np.array([[0, 0, 1], [w-1, 0, 1], [0, h-1, 1], [w-1, h-1, 1]])
            sizes = set()
            for signs in product((-1, 1), repeat=3):
                correction = np.array(signs)*[w*s.MAX_SHIFT, h*s.MAX_SHIFT, s.MAX_ANGLE]
                result = s._canvas((w, h), correction)
                sizes.add(result.size)
                points = corners @ result.matrix.T
                self.assertTrue(np.all(points >= 0))
                self.assertTrue(np.all(points < result.size))
                np.testing.assert_allclose(result.matrix[:, :2].T @ result.matrix[:, :2], np.eye(2), atol=1e-12)
            self.assertEqual(len(sizes), 1)
        white = np.full((480, 640, 3), 255, np.uint8)
        result = s.update(white, 100)
        preview = s.render(white, result)
        self.assertTrue(np.all(preview[0, 0] == 0))
        self.assertTrue(np.all(preview[preview.shape[0]//2, preview.shape[1]//2] == 255))
        self.assertEqual(result.size, (preview.shape[1], preview.shape[0]))
        # Picking a point on the padded, transformed preview still maps to its
        # original detector/test-target coordinate.
        transform = s._canvas((640, 480), (20, -10, .03))
        point = np.array([100., 200., 1.])
        displayed = transform.matrix @ point
        inverse = np.array(transform.geometry((640, 480))["inverse"])
        np.testing.assert_allclose(inverse @ np.array([*displayed, 1]), point[:2], atol=1e-10)

    def test_feature_loss_gap_geometry_and_context_changes_reset_correction(self):
        s, base = VideoStabilizer(), texture()
        s.update(base, 100)
        shifted = cv2.warpAffine(base, np.float32([[1, 0, 8], [0, 1, -4]]), (640, 480))
        s.update(shifted, 100+1/30)
        self.assertGreater(np.linalg.norm(s.smooth), 1)
        blank = np.zeros_like(base)
        result = s.update(blank, 100+2/30)
        self.assertEqual(s.source, "waiting for features")
        np.testing.assert_array_equal(s.smooth, 0)
        s.update(base, 101)
        self.assertEqual(s.source, "warming up")
        s.update(base, 101+1/30, context="new mount")
        self.assertEqual(s.source, "warming up")
        result = s.update(cv2.resize(base, (500, 374)), 101+2/30)
        self.assertEqual(s.source, "warming up")
        self.assertEqual(s.gray.shape[1], 320)
        self.assertGreater(result.size[0], 500)

    def test_gyro_projection_freshness_and_flow_fallback(self):
        att = dict(age=.03, rollspeed=1, pitchspeed=2, yawspeed=3)
        np.testing.assert_allclose(camera_gyro(att, 0), (2, 3, 1))
        np.testing.assert_allclose(camera_gyro(att, 90), (2, 1, -3))
        np.testing.assert_allclose(camera_gyro(att, 90, 90), (1, -2, -3))
        for bad in (None, {}, dict(att, age=.5), dict(att, yawspeed=float("nan"))):
            self.assertIsNone(camera_gyro(bad, 90))
        s, base = VideoStabilizer(), texture()
        s.update(base, 100, gyro=(0, 0, 0))
        s.update(base, 100+1/30, gyro=(0, 0, 0))
        self.assertEqual(s.source, "gyro + optical flow")
        s.update(np.zeros_like(base), 100+2/30, gyro=(0, 0, .02))
        self.assertEqual(s.source, "gyro")
        # Short gyro bridging ends instead of integrating drift indefinitely.
        for i in range(3, 12):
            s.update(np.zeros_like(base), 100+i/30, gyro=(0, 0, .02))
        self.assertEqual(s.source, "waiting for features")
        s.update(base, 100+12/30, gyro=None)
        s.update(base, 100+13/30, gyro=None)
        self.assertEqual(s.source, "optical flow")


class LivePreviewTests(unittest.TestCase):
    def test_toggle_resets_on_camera_thread_and_off_performs_no_cv_work(self):
        t, base = preview_tracker(), texture()
        with patch.object(t.stabilizer, "update", wraps=t.stabilizer.update) as update:
            self.assertIsNone(t._preview_transform(base, 100))
            update.assert_not_called()
            self.assertTrue(t.set_stabilization("invalid"))
            self.assertEqual(t._command("stabilization", "on"), "")
            t._preview_transform(base, 100)
            self.assertEqual(t._stabilization_status["source"], "warming up")
            t._preview_transform(base, 100+1/30)
            self.assertEqual(t._stabilization_status["source"], "optical flow")
            self.assertEqual(t.set_stabilization("off"), "")
            calls = update.call_count
            self.assertIsNone(t._preview_transform(base, 100+2/30))
            self.assertEqual(update.call_count, calls)
            self.assertIsNone(t._preview_geometry)
            self.assertEqual(t.set_stabilization("on"), "")
            t._preview_transform(base, 100+3/30)
            self.assertEqual(t._stabilization_status["source"], "warming up")

    def test_unsupported_or_stale_telemetry_uses_flow_and_cv_errors_recover(self):
        t, base = preview_tracker(), texture()
        t.ap.mission = SimpleNamespace(camera_mount_deg=90, wasp=SimpleNamespace(enabled=False))
        t.ap.backend = SimpleNamespace(attitude=lambda max_age: None)
        t.set_stabilization("on")
        self.assertIsNotNone(t._preview_transform(base, 100))
        self.assertIsNotNone(t._preview_transform(base, 100+1/30))
        self.assertEqual(t._stabilization_status["source"], "optical flow")
        with patch.object(t.stabilizer, "update", side_effect=cv2.error("test")):
            self.assertIsNone(t._preview_transform(base, 100+2/30))
        self.assertIsNotNone(t._preview_transform(base, 100+3/30))

    def test_running_stream_toggle_preserves_detection_recording_and_capture(self):
        t, base = preview_tracker(), texture()
        raw, published = [], []
        modes = ("off", "on", "on", "off", "on")
        delivered = threading.Event()

        class Capture:
            i = 0
            release = Mock()
            def isOpened(self):
                return self.i < len(modes)
            def read(self):
                if self.i:
                    self_outer.assertTrue(delivered.wait(2), "preview worker did not publish")
                    delivered.clear()
                t.set_stabilization(modes[self.i])
                self.i += 1
                return True, base.copy()

        def process(full, display, capture_t):
            raw.append(display.copy())
            return None, 0, 0, None, False

        cap = Capture()
        self_outer = self
        t._cam_req = t._t_ui_prev = None
        t._ui_fps = t._frame_idx = 0
        t.display_size, t.fw, t.fh = (640, 480), 640, 480
        t.PROFILE = False
        t.headless = True
        t.max_frames = len(modes)
        t._last_cmd = {}
        t.ap.steering = False
        t.ap.neutral_command = lambda: {}
        t.ap.update = t.ap.close = Mock()
        t._process = process
        t._update_log_context = t._prof_report = t._rec_start_thread = Mock()
        t._lap = lambda name, previous: previous
        t._draw = lambda frame, *args: frame.__setitem__((10, 10), (255, 0, 255))
        def publish(frame):
            published.append(frame.copy())
            delivered.set()
        t.stream = SimpleNamespace(publish=publish)
        recorded = deque(maxlen=len(modes))
        try:
            t._loop(cap, 30, 1, True, 2, recorded, Path("unused.mp4"), None)
            self.assertTrue(delivered.wait(2))
        finally:
            if t._preview_worker is not None:
                t._preview_worker.close()
        cap.release.assert_not_called()
        self.assertIs(t._cap, cap)
        self.assertEqual(len(published), len(modes))
        for original in raw:
            np.testing.assert_array_equal(original, base)
        for (_, frame) in recorded:
            self.assertEqual(frame.shape, base.shape)
            np.testing.assert_array_equal(frame[10, 10], (255, 0, 255))
        for mode, frame in zip(modes, published):
            if mode == "off":
                self.assertEqual(frame.shape, base.shape)
                np.testing.assert_array_equal(frame, recorded[0][1])
            else:
                self.assertGreater(frame.shape[0], base.shape[0])
                self.assertGreater(frame.shape[1], base.shape[1])

    def test_off_during_in_flight_stabilization_cannot_publish_a_stale_on_frame(self):
        t, base = preview_tracker(), texture()
        t.stream = SimpleNamespace(publish=Mock())
        t.set_stabilization("on")
        started, resume = threading.Event(), threading.Event()
        update = t.stabilizer.update
        def blocked_update(*args, **kwargs):
            started.set()
            self.assertTrue(resume.wait(2))
            return update(*args, **kwargs)
        with patch.object(t.stabilizer, "update", side_effect=blocked_update):
            worker = PreviewWorker(t._stabilize_preview)
            try:
                worker.submit(base.copy(), base, 100, t._stabilization_generation)
                self.assertTrue(started.wait(2))
                t.set_stabilization("off")
                resume.set()
            finally:
                worker.close()
        t.stream.publish.assert_not_called()
        self.assertIsNone(t._preview_geometry)
        self.assertEqual(t._stabilization_status["source"], "off")

    def test_preview_worker_replaces_queued_frames(self):
        entered, resume, finished = threading.Event(), threading.Event(), threading.Event()
        frames = []
        def process(raw, annotated, capture_t, generation):
            frames.append(capture_t)
            if capture_t == 1:
                entered.set()
                resume.wait(2)
            else:
                finished.set()
        worker, frame = PreviewWorker(process), texture()
        try:
            worker.submit(frame.copy(), frame, 1, 0)
            self.assertTrue(entered.wait(2))
            worker.submit(frame.copy(), frame, 2, 0)
            worker.submit(frame.copy(), frame, 3, 0)
            resume.set()
            self.assertTrue(finished.wait(2))
        finally:
            worker.close()
        self.assertEqual(frames, [1, 3])

    def test_http_toggle_uses_existing_command_endpoint(self):
        t = preview_tracker()
        server = MJPEGServer(host="127.0.0.1", port=0,
            state_get=lambda: dict(video_stabilization=t.video_stabilization), command=t._command)
        with contextlib.redirect_stdout(io.StringIO()):
            server.start()
        try:
            url = f"http://127.0.0.1:{server._httpd.server_port}/cmd?c=test&stabilization="
            for value, expected in (("on", True), ("off", False)):
                with urlopen(url+value, timeout=2) as response:
                    state = json.load(response)
                self.assertEqual(state["applied"], ["stabilization"])
                self.assertFalse(state["refused"])
                self.assertEqual(state["video_stabilization"], expected)
            with urlopen(url+"invalid", timeout=2) as response:
                self.assertTrue(json.load(response)["refused"])
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
