import csv
import io
import json
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fc_control import AcroCommander, ArduPilotAcro, InterceptLaw
from mission import Mission
from wasp import WaspGuidance


class FlightLoggingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.c = AcroCommander(conn="none", law=InterceptLaw(), mission=True,
                               log_dir=self.tmp.name, verbose=False)
        self.addCleanup(self.c.close)
        self.b = ArduPilotAcro("unused", axes=("roll", "pitch", "yaw", "throttle"), verbose=False)
        self.c.backend = self.b
        self.b.linked = self.b.armed = True
        self.b.mode_name = "ALT_HOLD"
        self.b.angle_max = math.radians(30)
        self.b.max_climb_mps, self.b.max_descent_mps = 25, 4
        self.b.pilot_yaw_rate = math.radians(202.5)
        self.b.params = dict(ACRO_TRAINER=0, ACRO_RP_RATE=360,
                             ACRO_Y_RATE=202.5, ACRO_RP_EXPO=0, ACRO_Y_EXPO=0,
                             RC_OVERRIDE_TIME=1)
        for ch, axis in enumerate(self.b.axes, 1):
            self.b.rc[axis] = dict(chan=ch, min=1100, trim=1500, max=1900, dz=20,
                                  reversed=False, from_fc=True, wasp_calibrated=True)
        self.b._send_override = lambda *_: True
        self.att = dict(roll=0.0, pitch=0.0, yaw=.2,
                        rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0)
        self.b.attitude = lambda: dict(self.att)
        self.b.height = lambda: (2.0, 0.0)
        self.b.vibration = lambda: None
        self.c.idle_request_alt_hold = lambda: True
        self.c.mission = Mission(self.c, log=lambda *_: None,
                                 idle_path=str(self.directory / "idle_settings.json"))
        self.ctx = dict(gate_ok=True, gate_why="", att=self.att, height=2.0,
                        climb=0.0, fc_mode="ALT_HOLD", idle_ok=True, airborne=True,
                        ai_cmd=None, ai_t=None, det_t=None, det_conf=None)
        self.t = perf_counter()
        self.c.update_log_context(cam_fps=60, processing_fps=60.2,
                                  detector="TensorRT", video_recording=False)
        with patch.dict(os.environ, WASP_RUN_STAMP="test_run"), redirect_stdout(io.StringIO()):
            self.c._open_log()
        self.tick()

    def tick(self, dt=.02):
        self.t += dt
        self.c._mission_out = self.c.mission.cycle(self.t, self.ctx)
        cmd = self.c.law.neutral()
        if self.ctx.get("ai_cmd") is not None:
            cmd.update(self.ctx["ai_cmd"])
        self.c._det_t = self.ctx.get("det_t")
        self.c._det_conf = self.ctx.get("det_conf")
        self.c._det_bbox = (100, 100, 12, 10)
        self.c._cap_age = .04
        with redirect_stdout(io.StringIO()):
            self.c._log_row(self.t, cmd, (10, -5) if self.c._det_t else None,
                            bool(self.c._mission_out), bool(self.c._mission_out))
        self.assertIsNone(self.c._log_error)

    def start(self, kind):
        m = self.c.mission
        if kind == "wasp":
            self.b.armed = False
            self.assertEqual(m.set_wasp_enabled(True, selection=True), "")
            self.assertEqual(m.confirm_mount(90), "")
            self.b.mode_name = self.ctx["fc_mode"] = "ACRO"
            self.b.armed = True
            self.tick()
        if kind == "auto":
            self.assertEqual(m.set_auto_enabled(True), "")
            self.assertEqual(m.request_auto(True), "")
        else:
            self.assertEqual(m.request_engage(), "")
        self.tick()
        self.assertEqual(self.c._flight_logs.active["flight_mode"], kind)
        return Path(self.c._flight_logs.active["csv"])

    def rows(self, path):
        with path.open() as handle:
            return list(csv.DictReader(handle))

    def test_idle_retains_each_mode_and_timer_after_mission_reset(self):
        for kind in ("auto", "semi_auto", "wasp"):
            with self.subTest(kind=kind):
                path = self.start(kind)
                self.tick(10)
                self.assertEqual(self.c.mission.phase, "IDLE")
                self.assertIsNone(self.c.mission._run_t)
                self.assertFalse(self.c.mission.auto_run)
                self.b.mode_name = self.ctx["fc_mode"] = "ALT_HOLD"
                self.tick()
                rows = self.rows(path)
                self.assertEqual({r["flight_mode"] for r in rows}, {kind})
                self.assertEqual(rows[-1]["phase"], "IDLE")
                self.assertGreaterEqual(float(rows[-1]["flight_elapsed_s"]), 10)
                self.assertEqual(float(rows[-1]["idle_remaining_s"]), 0)
                self.assertEqual(float(rows[-1]["idle_height_target_m"]), 3.5)
                self.assertTrue(self.c._flight_logs.snapshot()["active"])
                self.b.armed = False
                self.ctx["gate_ok"] = False
                self.ctx["gate_why"] = "disarmed"
                self.tick()
                self.assertFalse(self.c._flight_logs.snapshot()["active"])
                metadata = json.loads(path.with_suffix(".json").read_text())
                self.assertEqual(metadata["end_reason"], "disarmed")
                self.assertEqual(metadata["status"], "complete")
                self.ctx.update(gate_ok=True, gate_why="")
                self.assertEqual(self.c.mission.set_wasp_enabled(False, selection=True), "")
                self.b.armed = True
                self.tick()

    def test_standby_does_not_create_a_flight_and_headers_are_unique(self):
        self.assertFalse(list(self.directory.glob("flight_*")))
        with Path(self.c.log_path).open() as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(len(reader.fieldnames), len(set(reader.fieldnames)))
            rows = list(reader)
        self.assertEqual(rows[-1]["flight_mode"], "standby")
        self.assertNotIn(None, rows[-1])
        self.assertEqual(float(rows[-1]["cam_fps"]), 60)
        self.assertEqual(float(rows[-1]["processing_fps"]), 60.2)

    def test_wasp_detection_loss_and_reacquisition_events_and_rates(self):
        path = self.start("wasp")
        guide = WaspGuidance()
        def detection():
            t = self.t + .02
            cmd = guide.observe(50, -30, 640, 480, math.tan(.5), math.tan(.4), self.att, t)
            cmd["camera_roll"] = 0
            self.ctx.update(ai_cmd=cmd, ai_t=t, det_t=t, det_conf=.9)
        detection()
        self.tick()
        self.tick(1.1)
        detection()
        self.tick()
        for _ in range(30):
            detection()
            self.tick()
        rows = self.rows(path)
        self.assertEqual(rows[-1]["wasp_stage"], "GUIDE")
        self.assertNotEqual(float(rows[-1]["cmd_roll_rate_dps"]), 0)
        self.assertEqual(rows[-1]["detection_fresh"], "1")
        self.tick(.3)
        self.assertEqual(self.rows(path)[-1]["wasp_stage"], "SEARCH")
        detection()
        self.tick()
        events = [json.loads(line) for line in path.with_suffix(".events.jsonl").read_text().splitlines()]
        freshness = [e["value"] for e in events if e["event"] == "detection_fresh"]
        self.assertEqual(freshness[-3:], [1, 0, 1])
        meta = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual(meta["settings"]["runtime"]["detector"], "TensorRT")
        self.assertEqual(meta["settings"]["mission"]["wasp_params"]["throttle"], 1)
        self.assertEqual(meta["settings"]["fc_params"]["RC_OVERRIDE_TIME"], 1)

    def test_abort_idle_and_pilot_takeover_close_same_auto_file(self):
        path = self.start("auto")
        self.c.mission.request_abort()
        self.tick()
        self.assertEqual(self.c.mission.phase, "IDLE")
        self.ctx.update(gate_ok=False, gate_why="pilot changed mode, STABILIZE")
        self.b.mode_name = "STABILIZE"
        self.c._mission_why = self.ctx["gate_why"]
        self.tick()
        rows = self.rows(path)
        self.assertEqual(rows[-1]["flight_mode"], "auto")
        self.assertEqual(rows[-1]["phase"], "PILOT")
        self.assertNotIn(None, rows[-1])
        events = path.with_suffix(".events.jsonl").read_text()
        self.assertIn("ABORT", events)
        self.assertIn("pilot changed mode", events)

    def test_service_shutdown_finalizes_logs_and_second_close_is_safe(self):
        path = self.start("semi_auto")
        self.c.close()
        self.c.close()
        self.assertEqual(json.loads(path.with_suffix(".json").read_text())["end_reason"], "service stopped")

    def test_per_flight_disk_failure_does_not_stop_continuous_log(self):
        self.start("semi_auto")
        with patch.object(self.c._flight_logs, "_changes", side_effect=OSError("disk write failure")):
            self.tick()
        self.assertEqual(self.c._flight_logs.error, "disk write failure")
        rows = self.rows(Path(self.c.log_path))
        before = len(rows)
        self.tick()
        self.assertEqual(len(self.rows(Path(self.c.log_path))), before+1)
        self.assertEqual(self.rows(Path(self.c.log_path))[-1]["flight_mode"], "semi_auto")


if __name__ == "__main__":
    unittest.main()
