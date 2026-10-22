
import math
from time import perf_counter, sleep
import unittest
from pymavlink import mavutil
from test_wifi_autonomy import WifiFixture, until  # sets sys.path
from idle import Idle

ALT_HOLD, STABILIZE, ACRO, LOITER, LAND = 2, 0, 1, 5, 9


class RecoveryMavlinkTests(WifiFixture):
    def test_test_launch_countdown_centre_and_abort_recovers_in_alt_hold(self):
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
        self.assert_recovers()
        self.assertIsNone(self.m._test_target_started)
        self.assert_climbs()
        self.release_by_pilot()

    def test_armed_test_launch_centres_target_then_abort_recovers_in_alt_hold(self):
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
        self.assert_recovers()
        self.assertIsNone(self.m._test_target_started)
        self.assertEqual(self.m.snapshot(perf_counter())["test_target"], dict(x=.5, y=.5))
        self.assert_climbs()
        self.release_by_pilot()

    def test_fixed_test_target_launches_with_selected_steering_and_abort_recovers(self):
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
        self.assert_recovers()
        self.assert_climbs()
        self.release_by_pilot()

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
        # Still at the arming height and inside RECOVERY_GROUND_S: cancelled.
        self.command("abort", "1")
        self.assert_cancelled_on_ground()
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

    def test_wasp_crosses_ceiling_and_guides_past_idle_setting_until_stabilize_recovers_without_climbing(self):
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
        self.fc.mode = STABILIZE
        self.assert_recovers()
        self.assertIn("STABILIZE", self.m.event)
        # Above the ceiling already: the target is the current height, so it
        # holds there instead of climbing further.
        self.assertAlmostEqual(self.m.idle.height, self.m.ceiling_m + 20, delta=.5)
        until(lambda: self.m.idle.stage == "HOLD" and self.fc.overrides.get(3) == 1500)
        sleep(.3)
        self.assertEqual(self.fc.overrides.get(3), 1500)
        self.release_by_pilot()

    def test_wasp_platform_angles_launch_loss_reacquisition_and_stabilize_takeover_recovers(self):
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
                self.fc.mode = STABILIZE
                self.assert_recovers()
                if tilt:
                    # Not level yet: ALT_HOLD with zero climb until within 10 degrees.
                    until(lambda: self.m.idle.stage == "LEVEL" and self.fc.overrides.get(3) == 1500)
                    sleep(.3)
                    self.assertEqual(self.fc.overrides.get(3), 1500)
                    self.fc.att.update(pitch=0)
                self.assert_climbs()
                self.release_by_pilot()
                self.fc.armed = False; self.target["on"] = False
                self.fc.att.update(pitch=-math.radians(tilt))
                self.fc.mode_requests.clear()
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

    def test_wasp_abort_on_the_ground_right_after_launch_cancels_without_recovery(self):
        self.wasp_ready()
        self.command("engage", "1")
        until(lambda: len(self.fc.overrides) == 4)
        self.assertFalse(self.c.backend.airborne()[0])
        self.command("abort", "1")
        self.assert_cancelled_on_ground()
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
        self.assert_mode_change_releases(LOITER)

    def assert_mode_change_releases(self, mode):
        self.fc.mode = mode
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides)
        self.assertFalse(self.m.idle.active)
        self.assertFalse(self.m.idle.recovery)
        self.assertFalse(self.fc.mode_requests)
        self.assertTrue(self.fc.armed and self.c.armed)
        sleep(.3)
        self.assertEqual(self.m.phase, "WASP_DONE")
        self.assertFalse(self.fc.overrides)
        self.assertFalse(self.fc.mode_requests)

    def fly(self, after=.7, **params):
        """Arm and launch WASP; return the launch time once `after` s in flight."""
        self.wasp_ready(idle_s=1)
        for key, value in params.items():
            self.command("waspp", f"{key}:{value}")
        self.command("engage", "1")
        until(lambda: self.m.phase == "WASP_FLIGHT" and len(self.fc.overrides) == 4)
        launch = self.m.wasp.started
        until(lambda: perf_counter()-launch > after)
        self.assertEqual(self.m.phase, "WASP_FLIGHT")
        return launch

    def assert_recovers(self):
        until(lambda: self.m.phase == "IDLE" and self.m.idle.recovery)
        self.assertIsNone(self.m.wasp.started)
        self.assertIsNone(self.m.snapshot(perf_counter())["idle_remaining_s"])
        until(lambda: [mode for _t, mode in self.fc.mode_requests] == [ALT_HOLD])
        until(lambda: self.c.mode_name == "ALT_HOLD" and self.m.idle.confirmed)
        self.assertEqual(self.m.phase, "IDLE")
        self.assertTrue(self.fc.overrides)

    def assert_climbs(self):
        # ALT_HOLD climb sticks: level, throttle above mid, below full stick.
        until(lambda: self.m.idle.stage == "CLIMB" and 1600 < self.fc.overrides.get(3, 0) < 1900)
        self.assertEqual((self.fc.overrides[1], self.fc.overrides[2]), (1500, 1500))
        self.assertLess(abs(self.fc.overrides[4]-1500), 25)
        self.assertEqual(len(self.fc.mode_requests), 1)

    def assert_holds_at_target(self):
        start = self.m.idle.height - Idle.RECOVERY_RISE_M
        self.assertAlmostEqual(start, self.c.backend.height()[0], delta=.5)
        self.fc.vfr["alt"] = 8 + self.m.idle.height
        until(lambda: self.m.idle.stage == "HOLD" and self.fc.overrides.get(3) == 1500, timeout=4)
        sleep(.3)
        self.assertEqual(self.m.phase, "IDLE")
        self.assertEqual(self.fc.overrides.get(3), 1500)

    def release_by_pilot(self, mode=STABILIZE):
        count = len(self.fc.mode_requests)
        self.fc.mode = mode
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides)
        self.assertFalse(self.m.idle.active)
        self.assertTrue(self.fc.armed and self.c.armed)
        self.assertTrue(self.m.request_engage())
        sleep(.3)
        self.assertEqual(len(self.fc.mode_requests), count, "never fight the pilot's mode change")
        self.assertEqual(self.m.phase, "WASP_DONE")
        self.assertFalse(self.fc.overrides)
        self.assertIsNone(self.m.wasp.started)

    def assert_cancelled_on_ground(self):
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides)
        self.assertIn("WASP ABORT", self.m.event)
        self.assertIn("on the ground", self.m.event)
        self.assertFalse(self.m.idle.active)
        self.assertIsNone(self.m.wasp.started)
        self.assertIsNone(self.m._run_t)
        self.assertFalse(self.fc.mode_requests)
        sleep(.2)
        self.assertEqual(self.m.phase, "WASP_DONE")
        self.assertFalse(self.fc.overrides)

    def watch(self, seconds):
        """Samples of the FC's override channels for `seconds`."""
        samples, end = [], perf_counter()+seconds
        while perf_counter() < end:
            samples.append(dict(self.fc.overrides))
            sleep(.01)
        return samples

    def assert_level_at_launch_throttle_until_ack(self, throttle_pwm):
        self.fc.ack_delay = .5
        until(lambda: self.fc.mode_requests)
        requested = self.fc.mode_requests[0][0]
        self.assertEqual(self.fc.mode_requests[0][1], ALT_HOLD)
        samples = self.watch(requested+.4-perf_counter())
        self.assertGreater(len(samples), 10)
        for sample in samples:
            self.assertEqual(sample, {1: 1500, 2: 1500, 3: throttle_pwm, 4: 1500})
        self.assertEqual((self.m.phase, self.m.idle.stage), ("IDLE", "SWITCHING"))
        self.assertNotEqual(self.c.mode_name, "ALT_HOLD")

    def test_wasp_abort_in_flight_levels_at_launch_throttle_then_climbs_and_holds_in_alt_hold(self):
        self.fc.ack_delay = .5
        self.fly(throttle=.85)
        self.assertEqual(self.fc.overrides[3], 1780)
        self.command("abort", "1")
        until(lambda: self.m.phase == "IDLE" and self.m.idle.recovery)
        self.assertIn("RECOVERY: ABORT", self.m.event)
        self.assert_level_at_launch_throttle_until_ack(1780)
        self.assert_recovers()
        self.assertAlmostEqual(self.m.idle.height, Idle.RECOVERY_RISE_M, delta=.2)
        self.assert_climbs()
        self.assert_holds_at_target()
        # No time limit in recovery either.
        sleep(1.2)
        self.assertEqual((self.m.phase, self.m.idle.stage), ("IDLE", "HOLD"))
        self.assert_transmitting(perf_counter()-1)
        self.release_by_pilot()

    def test_wasp_stabilize_takeover_in_flight_recovers_in_alt_hold(self):
        self.fc.ack_delay = .5
        self.fly(after=1)
        self.fc.mode = STABILIZE
        until(lambda: self.m.phase == "IDLE" and self.m.idle.recovery)
        self.assertEqual(self.m.idle.source_mode, "STABILIZE")
        self.assertIn("STABILIZE", self.m.event)
        self.assert_level_at_launch_throttle_until_ack(1900)
        self.assert_recovers()
        self.assert_climbs()
        self.assert_holds_at_target()
        self.release_by_pilot(LOITER)

    def test_wasp_recovery_fails_to_the_pilot_when_fc_rejects_alt_hold(self):
        self.fc.mode_ack = mavutil.mavlink.MAV_RESULT_DENIED
        self.fly()
        self.command("abort", "1")
        until(lambda: self.m.phase == "IDLE_FAILED" and not self.fc.overrides)
        self.assertIn("rejected ALT_HOLD", self.m.event)
        self.assertFalse(self.m.idle.active)
        self.assertEqual([mode for _t, mode in self.fc.mode_requests], [ALT_HOLD])
        self.assertEqual(self.c.mode_name, "ACRO")
        self.assertTrue(self.fc.armed and self.c.armed)
        self.assertTrue(self.m.request_engage())
        sleep(.5)
        self.assertFalse(self.fc.overrides)
        self.assertIsNone(self.m.wasp.started)
        self.assertEqual(len(self.fc.mode_requests), 1)
        self.fc.mode = STABILIZE
        until(lambda: self.m.phase == "WASP_DONE")
        self.assertFalse(self.fc.overrides)

    def test_wasp_recovery_fails_when_alt_hold_never_shows_in_the_heartbeat(self):
        self.fc.apply_mode = False
        self.fly()
        self.command("abort", "1")
        until(lambda: self.m.phase == "IDLE" and self.m.idle.accepted)
        began = self.m.idle.started
        until(lambda: self.m.phase == "IDLE_FAILED" and not self.fc.overrides, timeout=3)
        self.assertGreaterEqual(perf_counter()-began, Idle.MODE_TIMEOUT_S)
        self.assertIn("not confirmed", self.m.event)
        self.assertTrue(self.fc.armed and self.c.armed)

    def test_wasp_recovery_keeps_going_through_a_late_alt_hold_heartbeat(self):
        self.fc.apply_mode = False
        self.fly()
        self.command("abort", "1")
        until(lambda: self.fc.mode_requests)
        requested = self.fc.mode_requests[0][0]
        until(lambda: self.m.idle.accepted)
        # ACKed, heartbeat still ACRO: ALT_HOLD climb sticks already, no release.
        until(lambda: 1500 < self.fc.overrides.get(3, 0) < 1900, timeout=1)
        self.assertEqual(self.c.mode_name, "ACRO")
        self.assertFalse(self.m.idle.confirmed)
        samples = self.watch(requested+.9-perf_counter())
        self.assertTrue(all(len(x) == 4 and x[1] == x[2] == 1500 for x in samples))
        self.assertEqual(self.m.phase, "IDLE")
        self.assertEqual(self.c.mode_name, "ACRO")
        self.fc.mode = ALT_HOLD
        until(lambda: self.c.mode_name == "ALT_HOLD" and self.m.idle.confirmed)
        self.assertEqual(self.m.phase, "IDLE")
        self.assert_climbs()
        self.release_by_pilot()

    def test_wasp_recovery_centres_sticks_on_stale_attitude_height_or_heartbeat(self):
        for stale in ("attitude", "vfr_on", "heartbeat"):
            with self.subTest(stale=stale):
                self.fly()
                self.command("abort", "1")
                self.assert_recovers()
                self.assert_climbs()
                setattr(self.fc, stale, False)
                until(lambda: self.fc.overrides == {1: 1500, 2: 1500, 3: 1500, 4: 1500}, timeout=2.5)
                since = perf_counter()
                samples = self.watch(1.2)
                self.assertTrue(all(x == {1: 1500, 2: 1500, 3: 1500, 4: 1500} for x in samples))
                self.assertEqual((self.m.phase, self.m.idle.stage), ("IDLE", "HOLD"))
                self.assert_transmitting(since)
                setattr(self.fc, stale, True)
                self.assert_climbs()
                self.release_by_pilot()
                self.fc.armed = False
                until(lambda: not self.c.armed and not self.m.wasp.used)
                self.fc.mode_requests.clear()

    def test_wasp_takeover_on_the_ground_right_after_launch_cancels(self):
        self.wasp_ready(idle_s=1)
        self.command("engage", "1")
        until(lambda: self.m.phase == "WASP_FLIGHT" and self.fc.overrides)
        self.assertFalse(self.c.backend.airborne()[0])
        self.fc.mode = STABILIZE
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides)
        self.assertIn("on the ground", self.m.event)
        sleep(.3)
        self.assertFalse(self.fc.mode_requests)
        self.assertFalse(self.m.idle.active)

    def test_wasp_mode_change_other_than_stabilize_releases_without_recovery(self):
        for mode in (LOITER, LAND):
            with self.subTest(mode=mode):
                self.fly()
                self.assert_mode_change_releases(mode)
                self.fc.armed = False
                until(lambda: not self.c.armed and not self.m.wasp.used)

    def test_wasp_disarm_during_recovery_releases(self):
        self.fly()
        self.command("abort", "1")
        self.assert_recovers()
        self.assert_climbs()
        self.fc.armed = False
        until(lambda: not self.c.armed and not self.fc.overrides and not self.m.idle.active)
        sleep(.3)
        self.assertFalse(self.fc.overrides)
        self.assertEqual(len(self.fc.mode_requests), 1)
        self.assertIn(self.m.phase, ("PILOT", "WASP_READY"))


class ModeSwitchMavlinkTests(WifiFixture):
    # FLTMODE_CH 5: position 0 ACRO, 1 LOITER, 2 LAND, 3 STABILIZE, 4 POSHOLD, 5 ALT_HOLD.
    fake_params = dict(FLTMODE_CH=5, FLTMODE1=ACRO, FLTMODE2=LOITER, FLTMODE3=LAND,
                       FLTMODE4=STABILIZE, FLTMODE5=16, FLTMODE6=ALT_HOLD)
    PWM = dict(ACRO=1165, LOITER=1295, LAND=1425, STABILIZE=1555, POSHOLD=1685, ALT_HOLD=1815)
    fake_radio = {5: PWM["ACRO"]}

    fly = RecoveryMavlinkTests.fly
    assert_recovers = RecoveryMavlinkTests.assert_recovers
    assert_climbs = RecoveryMavlinkTests.assert_climbs

    def setUp(self):
        super().setUp()
        self.assertEqual(self.c.backend.flight_modes,
                         ["ACRO", "LOITER", "LAND", "STABILIZE", "POSHOLD", "ALT_HOLD"])
        self.assertEqual(self.c.backend.pilot_mode_switch(), (0, "ACRO"))
        # A 1 Hz heartbeat, like a real FC: the switch is read from RC_CHANNELS.
        self.fc.hb_period = 1.0

    def switch(self, mode):
        self.fc.radio[5] = self.PWM[mode]
        until(lambda: self.c.backend.pilot_mode_switch() == (list(self.PWM).index(mode), mode))
        return perf_counter()

    def test_switch_to_stabilize_starts_recovery_before_the_heartbeat_shows_it(self):
        self.fly()
        self.fc.apply_mode = False  # the heartbeat keeps saying ACRO
        moved = self.switch("STABILIZE")
        until(lambda: self.m.phase == "IDLE" and self.m.idle.recovery, timeout=.5)
        self.assertLess(perf_counter()-moved, .3)
        self.assertEqual(self.c.mode_name, "ACRO")
        self.assertIn("STABILIZE", self.m.event)
        until(lambda: [mode for _t, mode in self.fc.mode_requests] == [ALT_HOLD])
        self.fc.mode = ALT_HOLD
        until(lambda: self.c.mode_name == "ALT_HOLD" and self.m.idle.confirmed, timeout=2)
        self.assert_climbs()
        # The switch stays in STABILIZE: recovery holds on.
        sleep(.5)
        self.assertEqual(self.m.phase, "IDLE")

    def test_switch_to_alt_hold_during_recovery_releases_with_the_fc_armed(self):
        self.fly()
        self.command("abort", "1")
        self.assert_recovers()
        self.assert_climbs()
        sleep(.3)
        self.assertEqual(self.m.phase, "IDLE")
        self.switch("ALT_HOLD")
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides)
        self.assertIn("moved the mode switch", self.m.event)
        self.assertFalse(self.m.idle.active)
        self.assertEqual(self.fc.mode, ALT_HOLD)
        self.assertTrue(self.fc.armed and self.c.armed)
        sleep(.3)
        self.assertFalse(self.fc.overrides)
        self.assertEqual(len(self.fc.mode_requests), 1)

    def test_switch_to_loiter_in_flight_releases_without_recovery(self):
        self.fly()
        self.switch("LOITER")
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides, timeout=.5)
        self.assertEqual(self.c.mode_name, "ACRO")
        self.assertFalse(self.m.idle.active)
        sleep(.3)
        self.assertFalse(self.fc.mode_requests)
        self.assertFalse(self.fc.overrides)
        self.assertEqual(self.m.phase, "WASP_DONE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
