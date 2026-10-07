"""HTTP disappears after launch; real MAVLink continues to a simulated FC.

Run with .env/yolo/bin/python src/tests/test_wifi_autonomy.py. No hardware,
camera, model, or changes to the Jetson's network interfaces are used.
"""
import contextlib
import io
import json
import math
from pathlib import Path
import socket
import sys
import tempfile
import threading
from time import perf_counter, sleep
import unittest
from unittest.mock import patch
from urllib.parse import urlencode
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fc_control import AcroCommander, InterceptLaw
from fake_fc import FakeArduPilot
from mission import phone_command
from mjpeg_stream import MJPEGServer


def until(predicate, timeout=3):
    deadline = perf_counter() + timeout
    while perf_counter() < deadline:
        if predicate():
            return
        sleep(.02)
    raise AssertionError("condition timed out")


class WifiAutonomyTests(unittest.TestCase):
    def setUp(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.fc = FakeArduPilot(port, armed=False, mode=0)
        self.fc.params.update(ACRO_RP_EXPO=0, ACRO_Y_EXPO=0, RC_OVERRIDE_TIME=1,
                              PILOT_SPEED_DN=250, PILOT_ACCEL_Z=500, PSC_JERK_Z=5,
                              ATC_INPUT_TC=.2, ATC_ACCEL_R_MAX=100000, ATC_ACCEL_P_MAX=100000,
                              LOG_BACKEND_TYPE=0, LOG_BITMASK=0, INS_LOG_BAT_MASK=0, FS_GCS_ENABLE=0)
        for ch in range(1, 5):
            self.fc.params.update({f"RC{ch}_MIN": 1100, f"RC{ch}_MAX": 1900,
                                   f"RC{ch}_TRIM": 1500, f"RC{ch}_DZ": 20})
        self.fc.vfr["alt"] = 8
        self.fc.start()
        self.logs = tempfile.TemporaryDirectory(prefix="wasp-wifi-test-")
        self.c = AcroCommander(conn=f"udpin:127.0.0.1:{port}", transport="mavlink",
                               axes=("roll", "pitch", "yaw", "throttle"), law=InterceptLaw(cam_tilt_deg=30),
                               send_hz=50, mission=True, verbose=False,
                               steer_modes={"ACRO", "ALT_HOLD"}, log_dir=self.logs.name)
        with contextlib.redirect_stdout(io.StringIO()), patch("mission._read_json", return_value={}):
            self.c.connect()
        self.m = self.c.mission
        self.m.log = lambda *_: None
        self.sent = []
        send = self.c.backend._send_override

        def record(channels):
            self.sent.append((perf_counter(), tuple(channels)))
            return send(channels)

        self.c.backend._send_override = record
        self.target = dict(on=False, x=80, y=0)
        self.stop = threading.Event()

        def detections():
            while not self.stop.is_set():
                if self.target["on"]:
                    x, y = self.target["x"], self.target["y"]
                    cmd = self.c.compute(x, y, "LOCK", 640, 640, bbox=(0, 0, 20, 20),
                                         capture_t=perf_counter()-.03)
                    self.c.update(cmd, (x, y), True, detected=True, conf=.9,
                                  bbox=(0, 0, 20, 20))
                sleep(.03)

        self.worker = threading.Thread(target=detections, daemon=True)
        self.worker.start()
        self.ui = None
        self.connect_phone()

    def tearDown(self):
        if self.ui is not None:
            self.ui.stop()
        self.stop.set()
        self.worker.join(timeout=2)
        self.c.close()
        self.fc.shutdown()
        self.fc.m.close()
        self.logs.cleanup()

    def connect_phone(self):
        self.ui = MJPEGServer(port=0, host="127.0.0.1", state_get=lambda: dict(mission=self.c.mission_state()),
                              command=lambda key, value: phone_command(self.m, key, value, perf_counter()))
        with contextlib.redirect_stdout(io.StringIO()):
            self.ui.start()
        self.url = f"http://127.0.0.1:{self.ui._httpd.server_port}"

    def command(self, key, value):
        with urlopen(self.url+"/cmd?"+urlencode({"c": "phone", key: value}), timeout=3) as response:
            state = json.load(response)
        self.assertFalse(state.get("refused"), state.get("refused"))
        self.assertEqual(state["control"], "you")
        return state

    def disconnect_phone(self):
        # Make every HTTP endpoint unavailable, without touching the FC link or
        # mission sender. This also removes all browser requests/keepalives.
        ui = self.ui
        disconnected = perf_counter()
        ui.stop()
        self.ui = None
        with self.assertRaises(OSError):
            urlopen(self.url+"/state", timeout=.2)
        until(lambda: perf_counter()-disconnected > MJPEGServer.LEASE_S+.1)
        self.assertEqual(ui._lease_status("phone"), "free")
        self.assertGreater(perf_counter()-disconnected, self.fc.params["RC_OVERRIDE_TIME"])
        return disconnected

    def assert_transmitting(self, since):
        recent = [t for t, channels in self.sent if t >= since and any(channels)]
        self.assertGreater(len(recent), 30)
        self.assertLess(perf_counter()-recent[-1], .15)
        self.assertTrue(self.fc.overrides)

    def seeker_ready(self):
        self.command("flightmode", "normal")
        self.command("mount", "30")
        self.fc.mode = 2
        self.fc.armed = True
        until(lambda: self.c.armed)
        self.fc.vfr["alt"] = 16  # Eight metres above the captured arming height.
        until(lambda: self.m.phase == "HOVER")

    def wasp_ready(self, no_det=5):
        self.command("flightmode", "wasp")
        self.command("waspp", f"no_det:{no_det}")
        self.command("mount", "90")
        self.fc.mode = 1
        self.fc.armed = True
        until(lambda: self.m.phase == "WASP_READY")

    def test_seeker_engage_continues_and_steers_with_no_http(self):
        self.seeker_ready()
        self.command("engage", "1")
        until(lambda: self.m.phase == "PUSH" and len(self.fc.overrides) == 4)
        offline = self.disconnect_phone()
        self.assertEqual(self.m.phase, "PUSH")
        self.assert_transmitting(offline)
        self.target["on"] = True
        yaw_ch = self.c.backend.rc["yaw"]["chan"]
        until(lambda: self.m.phase == "AI" and self.fc.overrides.get(yaw_ch, 0) > 1500)
        self.target["x"] = -80
        until(lambda: self.fc.overrides.get(yaw_ch, 2000) < 1500)
        self.connect_phone()
        with urlopen(self.url+"/state?c=reconnected", timeout=3) as response:
            state = json.load(response)
        self.assertEqual(state["mission"]["phase"], "AI")
        self.fc.mode = 0
        until(lambda: self.m.phase == "PILOT" and not self.fc.overrides)

    def test_seeker_auto_climbs_acquires_and_guides_with_no_http(self):
        self.seeker_ready()
        self.command("autoen", "on")
        self.command("auto", "on")
        until(lambda: self.m.phase == "AUTO_CLIMB" and self.fc.overrides)
        offline = self.disconnect_phone()
        self.assertEqual(self.m.phase, "AUTO_CLIMB")
        self.assert_transmitting(offline)
        self.fc.vfr["alt"] = 8+self.m.AUTO_HEIGHT_M
        until(lambda: self.m.phase == "AUTO_HOLD")
        self.target.update(on=True, x=0,
                           y=320*math.tan(math.radians(-self.m.auto_aim_el()))/self.c.law._tan_v)
        until(lambda: self.m.phase in ("PUSH", "AI"))
        until(lambda: self.m.phase == "AI")
        self.assertTrue(self.m.auto_run)
        self.target["x"] = 80
        yaw_ch = self.c.backend.rc["yaw"]["chan"]
        until(lambda: self.fc.overrides.get(yaw_ch, 0) > 1500)
        self.fc.mode = 0
        until(lambda: self.m.phase == "PILOT" and not self.fc.overrides)

    def test_wasp_acquires_and_steers_offline_then_releases_on_target_loss(self):
        self.wasp_ready()
        self.command("engage", "1")
        until(lambda: self.m.phase == "WASP_FLIGHT" and len(self.fc.overrides) == 4)
        offline = self.disconnect_phone()
        self.assertEqual(self.m.wasp.progress(perf_counter())["stage"], "SEARCH")
        self.assert_transmitting(offline)
        self.target["on"] = True
        roll_ch = self.c.backend.rc["roll"]["chan"]
        until(lambda: self.m.wasp.progress(perf_counter())["stage"] == "GUIDE"
              and self.fc.overrides.get(roll_ch, 0) > 1500)
        self.target["x"] = -80
        until(lambda: self.fc.overrides.get(roll_ch, 2000) < 1500)
        self.target["on"] = False
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides, timeout=1)
        self.assertTrue(self.m.request_engage())

    def test_wasp_no_detection_deadline_still_releases_offline(self):
        self.wasp_ready(no_det=3)
        self.command("engage", "1")
        until(lambda: self.m.phase == "WASP_FLIGHT" and self.fc.overrides)
        offline = self.disconnect_phone()
        self.assertEqual(self.m.phase, "WASP_FLIGHT")
        self.assert_transmitting(offline)
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides, timeout=2)
        self.assertIn("no detection time limit", self.m.event)

    def test_manual_held_move_expires_when_http_disappears(self):
        self.seeker_ready()
        self.command("hold", "up:manual")
        until(lambda: self.fc.overrides)
        self.disconnect_phone()
        until(lambda: self.m.phase == "HOVER" and not self.fc.overrides)


if __name__ == "__main__":
    unittest.main(verbosity=2)
