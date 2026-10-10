
import math
from time import perf_counter, sleep
import unittest
from pymavlink import mavutil
from test_wifi_autonomy import WifiFixture, until


class RecoveryMavlinkTests(WifiFixture):
    def test_motion_assisted_test_launch_countdown_centre_and_abort(self):
        self.wasp_ready(idle_s=8, launch_s=.3)
        self.command("waspp", "motion_assist:1")
        self.command("waspp", "straight:0")
        self.command("waspp", "centre_timer:1")
        self.command("waspp", "throttle:0.85")
        self.command("testmode", "on")
        self.command("testtarget", "0.65:0.35")
        self.command("engage", "1")
        sleep(.1)
        self.assertFalse(self.fc.overrides)
        until(lambda: self.m.phase == "WASP_FLIGHT" and self.fc.overrides.get(3) == 1780)
        until(lambda: self.c._cmd.get("motion_assist") is True)
        self.assertTrue(self.m.set_wasp_param("motion_assist", 0))
        until(lambda: self.m.snapshot(perf_counter())["test_target"] == dict(x=.5,y=.5))
        until(lambda: abs(self.fc.overrides.get(1,0)-1500)<=21
              and abs(self.fc.overrides.get(2,0)-1500)<=21)
        self.assertEqual(self.fc.overrides[3],1780)
        self.command("abort","1")
        until(lambda: self.m.phase == "IDLE" and self.m.idle.confirmed)
        self.assertNotEqual(self.fc.overrides.get(3),1780)

    def test_armed_test_launch_centres_target_then_abort_enters_idle(self):
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
        until(lambda: self.m.phase == "IDLE" and self.m.idle.confirmed)
        self.assertIsNone(self.m._test_target_started)
        self.assertEqual(self.m.snapshot(perf_counter())["test_target"], dict(x=.5, y=.5))
        self.assertNotEqual(self.fc.overrides.get(3), 1780)

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
        until(lambda: self.m.phase == "IDLE" and self.m.idle.confirmed)
        self.assertNotEqual(self.fc.overrides.get(3), 1900)

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
        until(lambda: self.m.phase == "IDLE" and self.m.idle.confirmed)
        self.fc.mode = 0; self.fc.armed = False
        until(lambda: not self.c.armed and self.m.phase == "PILOT")

        self.wasp_ready(launch_s=3)
        self.command("engage", "1")
        until(lambda: self.m.wasp.launch_t is not None)
        self.command("abort", "1")
        until(lambda: self.m.phase == "IDLE")
        self.assertIsNone(self.m.wasp.launch_t)
        self.assertIsNone(self.m.wasp.started)
        self.assertNotEqual(self.fc.overrides.get(3), 1900)

    def test_wasp_crosses_ceiling_and_guides_until_idle_deadline(self):
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
        until(lambda: self.m.phase == "IDLE" and self.m.idle.confirmed)
        self.assertAlmostEqual(self.m.idle.started-launch, 3, delta=.1)
        self.assertGreater(self.m.idle.height, self.m.ceiling_m)
        self.assertIn("Idle time limit", self.m.event)

    def test_wasp_platform_angles_launch_loss_reacquisition_and_idle(self):
        for tilt in (0, 45, 90):
            with self.subTest(tilt=tilt):
                self.fc.att.update(pitch=-math.radians(tilt), roll=0, yaw=.7)
                self.wasp_ready(idle_s=5)
                self.command("engage", "1")
                until(lambda: self.fc.overrides.get(3) == 1900)
                launch = self.m.wasp.started
                self.assertEqual(self.m._run_t, launch)
                self.assertEqual(self.fc.overrides[1], 1500)
                self.assertEqual(self.fc.overrides[2], 1500)
                self.target.update(on=True, x=80, y=-20)
                until(lambda: self.m.wasp.progress(perf_counter())["stage"] == "GUIDE")
                until(lambda: self.fc.overrides.get(1, 0) > 1500)
                self.target["on"] = False
                until(lambda: self.m.wasp.acquired_t is None and self.fc.overrides.get(1) == 1500)
                self.assertEqual(self.m.phase, "WASP_FLIGHT")
                self.assertEqual(self.m._run_t, launch)
                self.target["on"] = True
                until(lambda: self.m.wasp.acquired_t is not None)
                self.assertEqual(self.m._run_t, launch)
                until(lambda: self.m.phase == "IDLE" and self.m.idle.confirmed, timeout=5)
                self.assertAlmostEqual(self.m.idle.started-launch, 5, delta=.1)
                self.assertEqual(self.fc.mode, 2)
                self.assertEqual(len(self.fc.mode_requests), tilt//45+1)
                self.assertEqual(self.m.idle.height, 1.5)
                self.assertAlmostEqual(self.m.idle.heading, .7, places=5)
                self.assertEqual(self.fc.overrides[1], 1500)
                self.assertEqual(self.fc.overrides[2], 1500)
                self.assertNotEqual(self.fc.overrides[3], 1900)
                if tilt:
                    self.assertEqual(self.m.idle.stage, "LEVEL")
                    self.assertEqual(self.fc.overrides[3], 1500)
                self.fc.att.update(pitch=0, roll=0)
                until(lambda: self.fc.overrides.get(3, 0) > 1500)
                self.fc.vfr["alt"] = 9.5
                until(lambda: self.m.idle.stage == "HOLD")
                until(lambda: self.fc.overrides.get(3) == 1500)
                self.fc.mode = 0


                until(lambda: self.c.mode_name == "STABILIZE"
                      and self.m.phase in ("PILOT", "WASP_DONE") and not self.fc.overrides)
                count = len(self.fc.mode_requests)
                sleep(.3)
                self.assertEqual(len(self.fc.mode_requests), count, "never fight RC mode takeover")
                self.fc.armed = False; self.target["on"] = False
                until(lambda: not self.c.armed and not self.m.wasp.used)
                self.fc.vfr["alt"] = 8
                sleep(.3)

    def test_abort_enters_idle_in_both_modes_without_restarting_height(self):
        for wasp in (False, True):
            if wasp:
                self.wasp_ready()
            else:
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

    def test_rejected_mode_command_releases_and_latches(self):
        self.fc.mode_ack = mavutil.mavlink.MAV_RESULT_DENIED
        self.wasp_ready(idle_s=1)
        self.command("engage", "1")
        until(lambda: self.m.phase == "IDLE_FAILED" and not self.fc.overrides)
        self.assertIn("rejected", self.m.event)
        self.assertEqual(len(self.fc.mode_requests), 1)
        self.target["on"] = True; sleep(.3)
        self.assertFalse(self.fc.overrides)
        self.assertTrue(self.m.request_engage())

    def test_accepted_ack_without_mode_confirmation_releases(self):
        self.fc.apply_mode = False
        self.wasp_ready(idle_s=1)
        self.command("engage", "1")
        until(lambda: self.m.phase == "IDLE_FAILED" and not self.fc.overrides, timeout=4)
        self.assertIn("not confirmed", self.m.event)
        self.assertEqual(len(self.fc.mode_requests), 1)

    def test_idle_stale_attitude_releases_without_mode_retry(self):
        self.wasp_ready(idle_s=1); self.command("engage", "1")
        until(lambda: self.m.phase == "IDLE" and self.m.idle.confirmed)
        self.fc.attitude = False
        until(lambda: not self.fc.overrides and not self.m.idle.active, timeout=1)
        self.fc.attitude = True; sleep(.4)
        self.assertFalse(self.fc.overrides)
        self.assertEqual(len(self.fc.mode_requests), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
