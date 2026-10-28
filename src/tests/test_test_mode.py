import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission import Mission, phone_command
from tracker import DroneTracker
from test_wasp import Mapper, ATT, observe, WaspGuidance


class TestTargetTests(unittest.TestCase):
    def setUp(self):
        self.mapper = Mapper()
        self.m = Mission(self.mapper)
        self.assertEqual(self.m.set_wasp_enabled(True), "")
        self.m.wasp.params["launch_s"] = 3.0  # exercise the countdown path
        self.assertEqual(self.m.confirm_mount(90), "")

    def test_fixed_box_size_edges_and_disable(self):
        self.assertTrue(self.m.set_test_target(".5:.5"))
        self.assertEqual(phone_command(self.m, "testmode", "on", 100), "")
        self.assertEqual(phone_command(self.m, "testtarget", ".75:.25", 100), "")
        self.assertEqual(self.m.test_target_box(640, 480), (480, 120, 448, 96, 64, 48))
        self.assertEqual(self.m.test_target_box(1280, 960), (960, 240, 896, 192, 128, 96))
        for invalid in ("nan:.5", "inf:.5", "-.1:.5", "1.1:.5", "x:y", ".5", ".5:.5:.5"):
            self.assertTrue(self.m.set_test_target(invalid))
        self.assertEqual(self.m.set_test_target("0:1"), "")
        self.assertEqual(self.m.test_target_box(640, 480), (32, 456, 0, 432, 64, 48))
        self.assertEqual(self.m.set_wasp_enabled(False, selection=True), "")
        self.assertFalse(self.m.test_mode)
        self.assertIsNone(self.m.test_target_box(640, 480))

    def test_test_launch_keeps_countdown_and_settings_and_locks_target(self):
        self.assertEqual(self.m.set_wasp_param("profile", "extra"), "")
        self.assertEqual(self.m.set_test_mode(True), "")
        self.mapper.armed = True
        ctx = dict(gate_ok=True, att=ATT, height=0, ai_cmd=None, det_t=None, det_conf=None)
        self.m.cycle(100, ctx)
        self.assertTrue(self.m.request_engage(), "test launch needs an explicit image target")
        self.assertEqual(self.m.set_test_target(".6:.4"), "")
        self.assertEqual(self.m.request_engage(), "")
        self.assertEqual(self.m.cycle(100, ctx), {})
        self.assertEqual(self.m.wasp.progress(101)["launch_countdown_s"], 2)
        self.assertTrue(self.m.set_test_target(".2:.2"))
        self.assertTrue(self.m.set_test_mode(False))
        self.assertEqual(self.m.cycle(102.99, ctx), {})
        ctx.update(ai_cmd=observe(WaspGuidance(), x=64, y=-48, t=103), det_t=103, det_conf=1)
        self.assertEqual(self.m.cycle(103, ctx)["throttle"], 1000)   # 0.1 s launch ramp starts at zero
        self.assertEqual(self.m.cycle(103.1, ctx)["throttle"], 2000)
        self.assertEqual(self.m.wasp.params["rate"], 120)
        self.assertEqual(self.m.wasp.params["gain"], 6)
        self.assertEqual(self.m.wasp.params["blend"], .25)
        self.assertTrue(self.m.set_test_target(".2:.2"))

    def test_test_frames_skip_models_sky_and_follower(self):
        self.m.set_test_mode(True)
        tracker = DroneTracker.__new__(DroneTracker)
        tracker.ap = SimpleNamespace(mission=self.m)
        tracker.fw, tracker.fh = 640, 480
        tracker.follower = Mock()
        tracker.follower.track.side_effect = AssertionError("follower must be skipped")
        tracker.sky = Mock()
        tracker.sky.mask.side_effect = AssertionError("real detector path")
        tracker.model = Mock(side_effect=AssertionError("model must be skipped"))
        self.assertEqual(tracker._process(None, None), (None, 0, 0, None, False))
        self.m.set_test_target(".75:.25")
        for _ in range(5):
            self.assertEqual(tracker._process(None, None), ((480, 120), 0, 0, None, True))
            self.assertEqual(tracker.last_bbox, (448, 96, 64, 48))
            self.assertEqual(tracker.conf, 1)
        tracker.model.assert_not_called()
        tracker.sky.mask.assert_not_called()
        tracker.follower.track.assert_not_called()
        self.m.set_test_mode(False)
        tracker.use_sky_mask = True
        with self.assertRaisesRegex(AssertionError, "real detector path"):
            tracker._process(None, None)
        self.assertIsNone(tracker.last_bbox)

    def start_centring(self, seconds=4):
        self.assertEqual(self.m.set_wasp_param("centre_timer", seconds), "")
        self.assertEqual(self.m.set_test_mode(True), "")
        self.assertEqual(self.m.set_test_target(".8:.2"), "")
        self.mapper.armed = True
        ctx = dict(gate_ok=True, att=ATT, height=0, ai_cmd=None, det_t=None, det_conf=None)
        self.m.cycle(100, ctx)
        self.assertEqual(self.m.request_engage(), "")
        self.assertEqual(self.m.cycle(100, ctx), {})
        return ctx

    def test_centre_timer_validation_and_editing_after_arming(self):
        self.assertEqual(self.m.wasp.params["centre_timer"], 3)
        for invalid in ("bad", "nan", "inf", "-1", "60.01"):
            self.assertTrue(self.m.set_wasp_param("centre_timer", invalid))
        self.mapper.armed = True
        self.m.cycle(100, dict(gate_ok=True, att=ATT))
        for value in (0, 3.5, 60):
            self.assertEqual(phone_command(self.m, "waspp", f"centre_timer:{value}", 100), "")
            self.assertEqual(self.m.wasp.params["centre_timer"], value)

    def test_centre_timer_starts_at_throttle_and_holds_center(self):
        ctx = self.start_centring()
        for now in (100, 101, 102.99):
            self.assertEqual(self.m.snapshot(now)["test_target"], dict(x=.8, y=.2))
            self.assertIsNone(self.m.snapshot(now)["test_centre_remaining_s"])
            self.assertEqual(self.m.cycle(now, ctx), {})
        self.assertEqual(self.m.cycle(103, ctx)["throttle"], 1000)   # 0.1 s launch ramp starts at zero
        self.assertEqual(self.m.cycle(103.1, ctx)["throttle"], 2000)
        self.assertEqual(self.m._test_target_started, self.m.wasp.started)
        self.assertEqual(self.m.snapshot(103)["test_centre_remaining_s"], 4)
        self.assertTrue(self.m.set_wasp_param("centre_timer", 2))
        positions = [self.m.snapshot(103+i*.1)["test_target"] for i in range(41)]
        self.assertTrue(all(a["x"] >= b["x"] and a["y"] <= b["y"]
                            for a, b in zip(positions, positions[1:])))
        self.assertEqual(self.m.snapshot(105)["test_target"], dict(x=.65, y=.35))
        for now in (107, 110, 160):
            self.assertEqual(self.m.snapshot(now)["test_target"], dict(x=.5, y=.5))
            self.assertEqual(self.m.test_target_box(640, 480, now), (320, 240, 288, 216, 64, 48))
            self.assertEqual(self.m.snapshot(now)["test_centre_remaining_s"], 0)

    def test_abort_freezes_motion_and_countdown_abort_never_starts_it(self):
        ctx = self.start_centring()
        self.m.request_abort()
        self.m.cycle(101, ctx)
        self.assertIsNone(self.m._test_target_started)
        self.assertEqual(self.m.snapshot(120)["test_target"], dict(x=.8, y=.2))
        self.setUp()
        ctx = self.start_centring()
        self.m.cycle(103, ctx)
        self.m.request_abort()
        self.m.cycle(105, ctx)
        self.assertIsNone(self.m._test_target_started)
        self.assertEqual(self.m.snapshot(150)["test_target"], dict(x=.65, y=.35))

    def test_gate_loss_cancels_launch_and_freezes_motion(self):
        ctx = self.start_centring()
        self.m.cycle(102, dict(ctx, gate_ok=False))
        self.assertIsNone(self.m.wasp.launch_t)
        self.assertIsNone(self.m._test_target_started)
        self.assertEqual(self.m.snapshot(120)["test_target"], dict(x=.8, y=.2))
        self.setUp()
        ctx = self.start_centring()
        self.m.cycle(103, ctx)
        self.m.cycle(105, dict(ctx, gate_ok=False))
        self.assertIsNone(self.m.wasp.started)
        self.assertEqual(self.m.snapshot(150)["test_target"], dict(x=.65, y=.35))

    def test_zero_centre_timer_and_moving_frames_bypass_models(self):
        ctx = self.start_centring(seconds=0)
        self.assertEqual(self.m.snapshot(102.99)["test_target"], dict(x=.8, y=.2))
        self.m.cycle(103, ctx)
        self.assertEqual(self.m.snapshot(103)["test_target"], dict(x=.5, y=.5))
        self.setUp()
        ctx = self.start_centring()
        self.m.cycle(103, ctx)
        tracker = DroneTracker.__new__(DroneTracker)
        tracker.ap = SimpleNamespace(mission=self.m)
        tracker.fw, tracker.fh = 640, 480
        tracker.follower, tracker.model, tracker.sky = Mock(), Mock(), Mock()
        with patch("mission.perf_counter", return_value=105):
            self.assertEqual(tracker._process(None, None), ((416, 168), 0, 0, None, True))
        with patch("mission.perf_counter", return_value=108):
            self.assertEqual(tracker._process(None, None), ((320, 240), 0, 0, None, True))
        tracker.model.assert_not_called()
        tracker.sky.mask.assert_not_called()
        tracker.follower.track.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
