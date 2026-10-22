
import math
from time import perf_counter, sleep
import unittest
from pymavlink import mavutil
from test_wifi_autonomy import WifiFixture, until


class RecoveryMavlinkTests(WifiFixture):
    def test_test_launch_countdown_centre_and_abort(self):
        self.wasp_ready(idle_s=8, launch_s=.3)
        self.command("waspp", "straight:0")
        self.command("waspp", "centre_timer:1")
        self.command("waspp", "throttle:0.85")
        self.command("testmode", "on")
        self.command("testtarget", "0.65:0.35")
        self.command("engage", "1")
        sleep(.1)
        self.assertFalse(self.fc.overrides)
        until(lambda: self.m.phase == "WASP_FLIGHT" and self.fc.overrides.get(3) == 1780)
        until(lambda: self.c._cmd.get("wasp") is True)
        until(lambda: self.m.snapshot(perf_counter())["test_target"] == dict(x=.5,y=.5))
        until(lambda: abs(self.fc.overrides.get(1,0)-1500)<=21
              and abs(self.fc.overrides.get(2,0)-1500)<=21)
        self.assertEqual(self.fc.overrides[3],1780)
        self.command("abort","1")
        self.assert_released_without_idle()

    def test_armed_test_launch_centres_target_then_abort_releases_to_pilot(self):
        self.wasp_ready(idle_s=8)
        self.command("waspp", "profile:extra")

        self.assertTrue(self.c.armed)
        self.assertEqual(self.c.mode_name, "ACRO")
        self.command("waspp", "launch_s:3")
        self.command("waspp", "centre_timer:2")
        self.command("waspp", "throttle:0.85")
        self.command("testmode", "on")
        self.command("testtarget", "0.75:0.3")
        self.command("engage", "1")
        until(lambda: self.m.wasp.launch_t is not None)
        sleep(.2)
        self.assertFalse(self.fc.overrides)
        self.assertIsNone(self.m._test_target_started)
        self.assertEqual(self.m.snapshot(perf_counter())["test_target"], dict(x=.75, y=.3))
        until(lambda: self.m.phase == "WASP_FLIGHT", timeout=4)
        self.assertEqual(self.m._test_target_started, self.m.wasp.started)
        until(lambda: self.fc.overrides.get(3) == 1780)
        self.assertEqual(self.fc.overrides[3], 1780)
        self.assertTrue(self.m.set_wasp_param("centre_timer", 3))
        until(lambda: .5 < self.m.snapshot(perf_counter())["test_target"]["x"] < .75)
        until(lambda: self.m.snapshot(perf_counter())["test_target"] == dict(x=.5, y=.5))
        until(lambda: abs(self.fc.overrides.get(1, 0)-1500) <= 21
              and abs(self.fc.overrides.get(2, 0)-1500) <= 21)
        self.assertEqual(self.m.phase, "WASP_FLIGHT")
        self.assertEqual(self.fc.overrides[3], 1780)
        self.assertEqual(self.m.wasp.params["rate"], 120)
        self.command("abort", "1")
        self.assert_released_without_idle()
        self.assertIsNone(self.m._test_target_started)
        self.assertEqual(self.m.snapshot(perf_counter())["test_target"], dict(x=.5, y=.5))

    def test_fixed_test_target_launches_with_selected_steering_and_aborts(self):
        self.wasp_ready(idle_s=5)
        self.command("waspp", "profile:extra")
        self.command("waspp", "launch_s:0.2")
        self.command("testmode", "on")
        self.assertTrue(self.m.request_engage())
        self.command("testtarget", "0.65:0.4")
        self.command("engage", "1")
        until(lambda: self.m.phase == "WASP_FLIGHT")
        until(lambda: self.m.wasp.progress(perf_counter())["stage"] == "GUIDE")
        self.assertEqual(self.m.wasp.params["rate"], 120)
        self.assertEqual(self.fc.overrides[3], 1900)
        self.assertGreater(self.fc.overrides[1], 1500)
        self.assertEqual(self.m.test_target, dict(x=.65, y=.4))
        self.assertTrue(self.m.set_test_target(".2:.2"))
        self.command("abort", "1")
        self.assert_released_without_idle()

    def test_default_countdown_waits_before_throttle_and_abort_cancels(self):
        self.wasp_ready(launch_s=3)
        self.command("engage", "1")
        until(lambda: self.m.wasp.launch_t is not None)
        start = self.m.wasp.launch_t - 3
        sleep(.15)
        self.assertIsNone(self.m.wasp.started)
        self.assertIsNone(self.m._run_t)
        self.assertFalse(self.fc.overrides)
        until(lambda: self.m.wasp.started is not None, timeout=4)
        self.assertAlmostEqual(self.m.wasp.started-start, 3, delta=.1)
        until(lambda: self.fc.overrides.get(3) == 1900)
        self.assertEqual(self.fc.overrides[3], 1900)
        self.command("abort", "1")
        self.assert_released_without_idle()
        self.fc.mode = 0; self.fc.armed = False
        until(lambda: not self.c.armed and self.m.phase == "PILOT")

        self.wasp_ready(launch_s=3)
        self.command("engage", "1")
        until(lambda: self.m.wasp.launch_t is not None)
        self.command("abort", "1")
        until(lambda: self.m.wasp.launch_t is None)
        self.assertIsNone(self.m.wasp.started)
        self.assertIn("WASP ABORT", self.m.event)
        sleep(3.2)
        # The countdown was cancelled: never launched, ready for a fresh ENGAGE.
        self.assertIsNone(self.m.wasp.started)
        self.assertIsNone(self.m.wasp.launch_t)
        self.assertEqual(self.m.phase, "WASP_READY")
        self.assertFalse(self.fc.overrides)
        self.assertFalse(self.fc.mode_requests)
        self.assertFalse(self.m.idle.active)

    def test_wasp_crosses_ceiling_and_guides_past_idle_setting_until_stabilize(self):
        self.wasp_ready(idle_s=3)
        self.target["on"] = True
        self.command("engage", "1")
        until(lambda: self.m.wasp.started is not None)
        launch = self.m.wasp.started
        self.fc.vfr["alt"] = 8 + self.m.ceiling_m + 20
        until(lambda: self.c.backend.height()[0] > self.m.ceiling_m)
        until(lambda: self.m.wasp.progress(perf_counter())["stage"] == "GUIDE")
        self.assertEqual(self.m.phase, "WASP_FLIGHT")
        self.assertEqual(self.fc.overrides[3], 1900)
        until(lambda: perf_counter()-launch > 4.5, timeout=6)
        self.assertEqual(self.m.phase, "WASP_FLIGHT")
        self.assertEqual(self.m.wasp.progress(perf_counter())["stage"], "GUIDE")
        self.assertEqual(self.fc.overrides[3], 1900)
        self.assertIsNone(self.m._run_t)
        self.assertFalse(self.m.idle.active)
        self.assertFalse(self.fc.mode_requests)
        self.assertNotIn("Idle time limit", self.m.event)
        self.fc.mode = 0
        until(lambda: self.c.mode_name == "STABILIZE" and self.m.phase == "WASP_DONE"
              and not self.fc.overrides)
        self.assertFalse(self.fc.mode_requests)

    def test_wasp_platform_angles_launch_loss_reacquisition_and_stabilize_takeover(self):
        for tilt in (0, 45, 90):
            with self.subTest(tilt=tilt):
                self.fc.att.update(pitch=-math.radians(tilt), roll=0, yaw=.7)
                self.wasp_ready(idle_s=1)
                self.command("engage", "1")
                until(lambda: self.fc.overrides.get(3) == 1900)
                launch = self.m.wasp.started
                self.assertIsNone(self.m._run_t)
                self.assertEqual(self.fc.overrides[1], 1500)
                self.assertEqual(self.fc.overrides[2], 1500)
                self.target.update(on=True, x=80, y=-20)
                until(lambda: self.m.wasp.progress(perf_counter())["stage"] == "GUIDE")
                until(lambda: self.fc.overrides.get(1, 0) > 1500)
                self.target["on"] = False
                until(lambda: self.m.wasp.acquired_t is None and self.fc.overrides.get(1) == 1500)
                self.assertEqual(self.m.phase, "WASP_FLIGHT")
                self.assertEqual(self.fc.overrides[3], 1900)
                self.target["on"] = True
                until(lambda: self.m.wasp.acquired_t is not None)
                until(lambda: perf_counter()-launch > 1.5)
                # The 1 s Idle setting does not end WASP.
                until(lambda: self.m.wasp.progress(perf_counter())["stage"] == "GUIDE")
                self.assertEqual(self.m.phase, "WASP_FLIGHT")
                self.assertEqual(self.fc.overrides[3], 1900)
                self.assertEqual(self.fc.mode, 1)
                self.assertFalse(self.fc.mode_requests)
                self.assertFalse(self.m.idle.active)
                self.assertIsNone(self.m._run_t)
                self.fc.mode = 0
                until(lambda: self.c.mode_name == "STABILIZE"
                      and self.m.phase == "WASP_DONE" and not self.fc.overrides)
                count = len(self.fc.mode_requests)
                sleep(.3)
                self.assertEqual(len(self.fc.mode_requests), count, "never fight RC mode takeover")
                self.fc.armed = False; self.target["on"] = False
                until(lambda: not self.c.armed and not self.m.wasp.used)
                self.fc.vfr["alt"] = 8
                sleep(.3)

    def test_abort_enters_idle_for_seeker_without_restarting_height(self):
        for wasp in (False,):
            self.seeker_ready()
            self.command("engage", "1")
            until(lambda: len(self.fc.overrides) == 4)
            self.command("abort", "1")
            until(lambda: self.m.phase == "IDLE" and self.m.idle.confirmed)
            height, start = self.m.idle.height, self.m.idle.started
            self.command("abort", "1"); sleep(.2)
            self.assertEqual(self.m.idle.height, height)
            self.assertEqual(self.m.idle.started, start)
            self.assertEqual(self.fc.mode, 2)
            self.fc.mode = 0
            until(lambda: self.c.mode_name == "STABILIZE" and not self.fc.overrides
                  and self.m.phase in (("PILOT", "WASP_DONE") if wasp else ("PILOT",)))
            self.fc.armed = False
            until(lambda: not self.c.armed)

    def test_seeker_target_loss_enters_idle_and_disarm_releases(self):
        self.seeker_ready(); self.target["on"] = True
        self.command("engage", "1")
        until(lambda: self.m.phase == "AI")
        self.target["on"] = False
        until(lambda: self.m.phase == "IDLE", timeout=1)
        self.assertIn("target lost", self.m.event)
        self.assertFalse(self.fc.mode_requests, "ALT_HOLD needs no mode command")
        self.fc.armed = False
        until(lambda: not self.fc.overrides and not self.m.idle.active)

    def test_wasp_abort_releases_every_channel_without_idle(self):
        self.wasp_ready()
        self.command("engage", "1")
        until(lambda: len(self.fc.overrides) == 4)
        self.command("abort", "1")
        self.assert_released_without_idle()
        self.command("abort", "1"); sleep(.2)
        self.assertEqual(self.m.phase, "WASP_DONE")
        self.assertFalse(self.fc.overrides)
        self.assertFalse(self.fc.mode_requests)
        self.assertEqual(self.fc.mode, 1)
        self.assertTrue(self.m.request_engage())
        self.fc.mode = 0
        until(lambda: self.c.mode_name == "STABILIZE" and not self.fc.overrides)
        self.fc.armed = False
        until(lambda: not self.c.armed and self.m.phase == "PILOT")

    def test_wasp_never_requests_alt_hold_and_flies_past_idle_setting(self):
        self.fc.mode_ack = mavutil.mavlink.MAV_RESULT_DENIED
        self.wasp_ready(idle_s=1)
        self.command("engage", "1")
        until(lambda: self.m.phase == "WASP_FLIGHT" and self.fc.overrides.get(3) == 1900)
        launch = self.m.wasp.started
        until(lambda: perf_counter()-launch > 2.5, timeout=4)
        self.assertEqual(self.m.phase, "WASP_FLIGHT")
        self.assert_transmitting(launch+1.5)
        self.assertEqual(self.fc.overrides[3], 1900)
        self.assertFalse(self.fc.mode_requests)
        self.assertFalse(self.m.idle.active)
        self.assertIsNone(self.m.snapshot(perf_counter())["idle_remaining_s"])

    def test_wasp_disarm_mid_flight_releases(self):
        self.wasp_ready(idle_s=1)
        self.target["on"] = True
        self.command("engage", "1")
        until(lambda: self.m.wasp.progress(perf_counter())["stage"] == "GUIDE"
              and self.fc.overrides.get(3) == 1900)
        self.fc.armed = False
        until(lambda: not self.c.armed and not self.fc.overrides
              and self.m.phase in ("PILOT", "WASP_DONE"))
        self.assertIsNone(self.m.wasp.started)
        self.assertFalse(self.fc.mode_requests)
        self.assertFalse(self.m.idle.active)
        sleep(.3)
        self.assertFalse(self.fc.overrides)

    def test_wasp_stale_attitude_mid_flight_keeps_throttle_with_zero_rates(self):
        self.wasp_ready(idle_s=1)
        self.target["on"] = True
        self.command("engage", "1")
        roll_ch = self.c.backend.rc["roll"]["chan"]
        until(lambda: self.m.wasp.progress(perf_counter())["stage"] == "GUIDE"
              and self.fc.overrides.get(roll_ch, 0) > 1500)
        self.fc.attitude = False
        until(lambda: self.m.wasp.acquired_t is None and self.fc.overrides.get(roll_ch) == 1500, timeout=1)
        stale = perf_counter()
        sleep(1.2)
        self.assertEqual(self.m.phase, "WASP_FLIGHT")
        self.assert_transmitting(stale)
        self.assertEqual(self.fc.overrides.get(roll_ch), 1500)
        self.assertEqual(self.fc.overrides.get(self.c.backend.rc["pitch"]["chan"]), 1500)
        self.assertEqual(self.fc.overrides.get(3), 1900)
        self.assertIsNone(self.m.wasp.acquired_t)
        self.fc.attitude = True
        until(lambda: self.m.wasp.acquired_t is not None
              and self.fc.overrides.get(roll_ch, 0) > 1500, timeout=3)
        self.assertEqual(self.m.phase, "WASP_FLIGHT")
        self.assertFalse(self.fc.mode_requests)
        self.fc.mode = 0
        until(lambda: self.c.mode_name == "STABILIZE" and self.m.phase == "WASP_DONE"
              and not self.fc.overrides)

    def assert_released_without_idle(self):
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides)
        self.assertIn("WASP ABORT", self.m.event)
        self.assertFalse(self.m.idle.active)
        self.assertIsNone(self.m.wasp.started)
        self.assertIsNone(self.m._run_t)
        self.assertFalse(self.fc.mode_requests)
        sleep(.2)
        self.assertEqual(self.m.phase, "WASP_DONE")
        self.assertFalse(self.fc.overrides)


if __name__ == "__main__":
    unittest.main(verbosity=2)
