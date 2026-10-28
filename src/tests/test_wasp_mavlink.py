
import math
from time import perf_counter, sleep
import unittest
from pymavlink import mavutil
from test_wifi_autonomy import WifiFixture, until  # sets sys.path

ALT_HOLD, STABILIZE, ACRO, LOITER, RTL, LAND = 2, 0, 1, 5, 6, 9


class SafeDownMavlinkTests(WifiFixture):
    def test_test_launch_countdown_centre_and_safedown_lands(self):
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
        self.command("safedown","1")
        self.assert_safedown()
        self.assertIsNone(self.m._test_target_started)
        self.assert_holds()
        self.assert_lands()

    def test_armed_test_launch_centres_target_then_safedown_holds_in_alt_hold(self):
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
        self.command("safedown", "1")
        self.assert_safedown()
        self.assertIsNone(self.m._test_target_started)
        self.assertEqual(self.m.snapshot(perf_counter())["test_target"], dict(x=.5, y=.5))
        self.assert_holds()
        self.release_by_pilot()

    def test_fixed_test_target_launches_with_selected_steering_and_safedown_holds(self):
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
        self.command("safedown", "1")
        self.assert_safedown()
        self.assert_holds()
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
        # Still at the arming height and inside SAFEDOWN_GROUND_S: cancelled.
        self.command("safedown", "1")
        self.assert_cancelled_on_ground()
        self.fc.mode = 0; self.fc.armed = False
        until(lambda: not self.c.armed and self.m.phase == "PILOT")

        self.wasp_ready(launch_s=3)
        self.command("engage", "1")
        until(lambda: self.m.wasp.launch_t is not None)
        self.command("safedown", "1")
        until(lambda: self.m.wasp.launch_t is None)
        self.assertIsNone(self.m.wasp.started)
        self.assertIn("WASP SAFEDOWN", self.m.event)
        sleep(3.2)
        # The countdown was cancelled: never launched, ready for a fresh ENGAGE.
        self.assertIsNone(self.m.wasp.started)
        self.assertIsNone(self.m.wasp.launch_t)
        self.assertEqual(self.m.phase, "WASP_READY")
        self.assertFalse(self.fc.overrides)
        self.assertFalse(self.fc.mode_requests)
        self.assertFalse(self.m.safedown.active)

    def test_wasp_crosses_ceiling_and_guides_past_idle_setting_until_stabilize_lands_it(self):
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
        self.fc.mode = ALT_HOLD  # the pilot's switch up
        self.assert_safedown()
        self.assertIn("ALT_HOLD", self.m.event)
        # Above the ceiling already: centred ALT_HOLD sticks hold it there,
        # never a climb, and then it lands.
        self.assert_holds()
        sleep(.3)
        self.assertEqual(self.fc.overrides.get(3), 1500)
        self.assert_lands()

    def test_wasp_platform_angles_launch_loss_reacquisition_and_stabilize_takeover_levels(self):
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
                self.fc.mode = ALT_HOLD  # the pilot's switch up
                self.assert_safedown()
                if tilt:
                    # Not level yet: ALT_HOLD with centred sticks, no LAND.
                    until(lambda: self.m.safedown.stage == "LEVEL" and self.fc.overrides == self.CENTRED)
                    sleep(2.5)
                    self.assertEqual(self.m.safedown.stage, "LEVEL")
                    self.assertNotIn(LAND, [m for _t, m in self.fc.mode_requests])
                    self.fc.att.update(pitch=0)
                self.assert_holds()
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

    def test_wasp_safedown_on_the_ground_right_after_launch_cancels(self):
        self.wasp_ready()
        self.command("engage", "1")
        until(lambda: len(self.fc.overrides) == 4)
        self.assertFalse(self.c.backend.airborne()[0])
        self.command("safedown", "1")
        self.assert_cancelled_on_ground()
        self.command("safedown", "1"); sleep(.2)
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
        self.assertFalse(self.m.safedown.active)
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
        self.assertFalse(self.m.safedown.active)
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
        self.assertFalse(self.m.safedown.active)
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

    CENTRED = {1: 1500, 2: 1500, 3: 1500, 4: 1500}

    def assert_safedown(self):
        until(lambda: self.m.phase == "SAFEDOWN" and self.m.safedown.active)
        self.assertIsNone(self.m.wasp.started)
        self.assertIsNone(self.m.snapshot(perf_counter())["idle_remaining_s"])
        until(lambda: self.c.mode_name == "ALT_HOLD")
        self.assertLessEqual({mode for _t, mode in self.fc.mode_requests}, {ALT_HOLD})
        self.assertEqual(self.m.phase, "SAFEDOWN")
        self.assertTrue(self.fc.overrides)

    def assert_holds(self):
        # ALT_HOLD with centred sticks: level, heading hold, no climb.
        until(lambda: self.m.safedown.stage in ("LEVEL", "STOP") and self.fc.overrides == self.CENTRED)
        self.assertEqual(self.c.mode_name, "ALT_HOLD")

    def assert_lands(self):
        """LAND after the stop, the FC's own touchdown disarm, then the pilot."""
        until(lambda: self.c.mode_name == "LAND", timeout=12)
        self.assertEqual(self.m.safedown.stage, "DESCEND")
        self.assertEqual(self.fc.mode_requests[-1][1], LAND)
        until(lambda: self.fc.overrides == self.CENTRED)
        until(lambda: not self.fc.armed and not self.c.armed and not self.m.safedown.active, timeout=4)
        self.assertIn("SAFEDOWN complete: landed", self.m.event)
        until(lambda: not self.fc.overrides)
        self.assertIn(self.m.phase, ("PILOT", "WASP_READY"))

    def release_by_pilot(self, mode=LOITER):
        count = len(self.fc.mode_requests)
        self.fc.mode = mode
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides)
        self.assertFalse(self.m.safedown.active)
        self.assertTrue(self.fc.armed and self.c.armed)
        self.assertTrue(self.m.request_engage())
        sleep(.3)
        self.assertEqual(len(self.fc.mode_requests), count, "never fight the pilot's mode change")
        self.assertEqual(self.m.phase, "WASP_DONE")
        self.assertFalse(self.fc.overrides)
        self.assertIsNone(self.m.wasp.started)

    def assert_cancelled_on_ground(self):
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides)
        self.assertIn("WASP SAFEDOWN", self.m.event)
        self.assertIn("on the ground", self.m.event)
        self.assertFalse(self.m.safedown.active)
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
        self.assertEqual((self.m.phase, self.m.safedown.stage), ("SAFEDOWN", "SWITCHING"))
        self.assertNotEqual(self.c.mode_name, "ALT_HOLD")

    def test_wasp_safedown_in_flight_bridges_at_launch_throttle_then_levels_and_lands(self):
        self.fc.ack_delay = .5
        self.fly(throttle=.85)
        self.assertEqual(self.fc.overrides[3], 1780)
        self.command("safedown", "1")
        until(lambda: self.m.phase == "SAFEDOWN")
        self.assertIn("SAFEDOWN: SAFEDOWN button", self.m.event)
        self.assert_level_at_launch_throttle_until_ack(1780)
        self.assert_safedown()
        self.assert_holds()
        self.assert_transmitting(perf_counter()-1)
        self.assert_lands()
        self.assertFalse(self.fc.disarm_requests, "LAND disarms by itself")

    def test_wasp_switch_up_in_flight_runs_the_same_safedown(self):
        self.fly(after=1)
        self.fc.mode = ALT_HOLD
        until(lambda: self.m.phase == "SAFEDOWN")
        self.assertEqual(self.m.safedown.source_mode, "ALT_HOLD")
        self.assertIn("ALT_HOLD", self.m.event)
        # ALT_HOLD reads the launch throttle as a climb: centred at once.
        until(lambda: self.fc.overrides == self.CENTRED, timeout=.3)
        self.assert_safedown()
        self.assert_holds()
        self.assert_lands()

    def test_the_old_abort_command_is_safedown_too(self):
        self.fly()
        self.command("abort", "1")
        self.assert_safedown()
        self.release_by_pilot()

    def test_alt_hold_refused_flies_stabilize_with_the_app_throttle_then_lands(self):
        self.fc.deny_modes.add(ALT_HOLD)
        self.fc.landed_state = 2
        self.fly()
        self.command("safedown", "1")
        until(lambda: self.c.mode_name == "STABILIZE" and self.m.safedown.fallback == "STABILIZE")
        self.assertIn("ALT_HOLD refused", self.m.event)
        # STABILIZE levels; the throttle is the app's: near the FC's mid-stick
        # hover, not the launch throttle and not centred ALT_HOLD sticks.
        until(lambda: self.m.safedown.stage in ("LEVEL", "STOP"))
        sleep(.6)
        self.assertEqual((self.fc.overrides[1], self.fc.overrides[2], self.fc.overrides[4]), (1500, 1500, 1500))
        self.assertLess(abs(self.fc.overrides[3]-1500), 60)
        until(lambda: sum(1 for _t, m in self.fc.mode_requests if m == ALT_HOLD) > 1)  # still offered
        self.assert_lands()

    def test_alt_hold_and_stabilize_refused_fail_to_the_pilot(self):
        self.fc.deny_modes.update((ALT_HOLD, STABILIZE))
        self.fly()
        self.command("safedown", "1")
        until(lambda: self.m.phase == "SAFEDOWN_FAILED" and not self.fc.overrides, timeout=4)
        self.assertIn("SAFEDOWN FAILED", self.m.event)
        self.assertFalse(self.m.safedown.active)
        self.assertEqual(self.c.mode_name, "ACRO")
        self.assertTrue(self.fc.armed and self.c.armed)
        self.assertTrue(self.m.request_engage())
        sleep(.5)
        self.assertFalse(self.fc.overrides)
        self.assertIsNone(self.m.wasp.started)

    def test_land_refused_descends_in_alt_hold_and_disarms_on_touchdown(self):
        self.fc.deny_modes.add(LAND)
        self.fc.landed_state = 2
        self.fly()
        self.command("safedown", "1")
        self.assert_safedown()
        until(lambda: self.m.safedown.fallback == "ALT_HOLD", timeout=12)
        self.assertIn("LAND refused", self.m.event)
        throttle_ch = self.c.backend.rc["throttle"]["chan"]
        until(lambda: self.fc.overrides.get(throttle_ch, 1500) < 1450)
        self.assertEqual(self.c.mode_name, "ALT_HOLD")
        self.assertFalse(self.fc.disarm_requests, "no disarm in the air")
        self.fc.landed_state = 1
        until(lambda: self.fc.disarm_requests, timeout=3)
        self.assertTrue(self.fc.disarm_requests[-1][1])
        until(lambda: not self.c.armed and not self.fc.overrides)
        self.assertIn("SAFEDOWN complete: landed", self.m.event)

    def test_acked_mode_that_never_shows_in_a_fresh_heartbeat_is_not_trusted(self):
        self.fc.apply_mode = False
        self.fly()
        self.command("safedown", "1")
        until(lambda: self.m.phase == "SAFEDOWN_FAILED" and not self.fc.overrides, timeout=5)
        self.assertIn("not confirmed", self.m.event)
        self.assertGreater(len(self.fc.mode_requests), 3, "retried before giving up")
        self.assertTrue(self.fc.armed and self.c.armed)

    def test_safedown_keeps_going_through_a_late_alt_hold(self):
        self.fc.apply_mode = False
        self.fly()
        self.command("safedown", "1")
        until(lambda: self.fc.mode_requests)
        sleep(.5)
        self.assertEqual(self.m.safedown.stage, "SWITCHING")
        self.assertGreater(len(self.fc.mode_requests), 1, "retried")
        self.fc.mode = ALT_HOLD
        until(lambda: self.c.mode_name == "ALT_HOLD" and self.m.safedown.stage in ("LEVEL", "STOP"))
        self.assertEqual(self.m.phase, "SAFEDOWN")
        self.fc.apply_mode = True
        self.assert_holds()
        self.assert_lands()

    def test_safedown_centres_sticks_on_stale_attitude_height_or_heartbeat(self):
        for stale in ("attitude", "vfr_on", "heartbeat"):
            with self.subTest(stale=stale):
                self.fly()
                self.command("safedown", "1")
                self.assert_safedown()
                self.assert_holds()
                setattr(self.fc, stale, False)
                since = perf_counter()
                samples = self.watch(1.2)
                self.assertTrue(all(x == self.CENTRED for x in samples))
                self.assertEqual(self.m.phase, "SAFEDOWN")
                self.assert_transmitting(since)
                setattr(self.fc, stale, True)
                self.assert_lands()
                self.fc.mode_requests.clear()

    def test_wasp_takeover_on_the_ground_right_after_launch_cancels(self):
        self.wasp_ready(idle_s=1)
        self.command("engage", "1")
        until(lambda: self.m.phase == "WASP_FLIGHT" and self.fc.overrides)
        self.assertFalse(self.c.backend.airborne()[0])
        self.fc.mode = ALT_HOLD
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides)
        self.assertIn("on the ground", self.m.event)
        sleep(.3)
        self.assertFalse(self.fc.mode_requests)
        self.assertFalse(self.m.safedown.active)

    def test_wasp_mode_change_other_than_alt_hold_releases_without_safedown(self):
        for mode in (LOITER, LAND):
            with self.subTest(mode=mode):
                self.fly()
                self.assert_mode_change_releases(mode)
                self.fc.armed = False
                until(lambda: not self.c.armed and not self.m.wasp.used)

    def test_wasp_disarm_during_safedown_releases(self):
        self.fly()
        self.command("safedown", "1")
        self.assert_safedown()
        self.fc.armed = False
        until(lambda: not self.c.armed and not self.fc.overrides and not self.m.safedown.active)
        sleep(.3)
        self.assertFalse(self.fc.overrides)
        self.assertIn("SAFEDOWN ended: disarmed", self.m.event)
        self.assertIn(self.m.phase, ("PILOT", "WASP_READY"))

    def test_safedown_after_the_pilot_took_over_in_stabilize(self):
        self.fly()
        self.fc.vfr["alt"] += 5  # well clear of the arming height
        self.fc.mode = STABILIZE  # the pilot's own flight, after the catch
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides, timeout=5)
        self.command("safedown", "1")
        until(lambda: self.m.phase == "SAFEDOWN")
        self.assert_holds()
        self.assert_lands()


class GpsReturnMavlinkTests(WifiFixture):
    fake_gps = True
    fly = SafeDownMavlinkTests.fly
    CENTRED = SafeDownMavlinkTests.CENTRED

    def assert_returns_and_lands(self):
        until(lambda: self.c.mode_name == "RTL", timeout=4)
        self.assertEqual(self.m.safedown.stage, "RETURN")
        self.assertIn("RTL, flying back to the launch point", self.m.event)
        self.assertIsNotNone(self.m.snapshot(perf_counter())["safedown"]["home_m"])
        until(lambda: self.fc.overrides == self.CENTRED)
        until(lambda: not self.fc.armed and not self.c.armed and not self.m.safedown.active, timeout=5)
        self.assertIn("SAFEDOWN complete: landed", self.m.event)
        self.assertNotIn(LAND, [m for _t, m in self.fc.mode_requests], "home, not here")

    def test_no_go_without_a_gps_position(self):
        self.wasp_ready(idle_s=1)
        self.fc.gps = False
        until(lambda: "waiting for GPS" in (self.c._mission_gate()[1] or ""), timeout=4)
        self.m.request_engage()
        sleep(.5)
        self.assertIsNone(self.m.wasp.started, "no launch without a way home")
        self.assertFalse(self.fc.overrides)
        self.fc.gps = True
        until(lambda: self.m.phase == "WASP_READY", timeout=4)

    def test_safedown_button_flies_home_by_rtl_and_lands_there(self):
        self.fly()
        self.command("safedown", "1")
        self.assert_returns_and_lands()

    def test_switch_up_flies_home_too(self):
        self.fly()
        self.fc.mode = ALT_HOLD
        until(lambda: self.m.phase == "SAFEDOWN")
        self.assertEqual(self.m.safedown.reason, "pilot switched to ALT_HOLD")
        self.assert_returns_and_lands()

    def test_rtl_refused_lands_here(self):
        self.fc.deny_modes.add(RTL)
        self.fly()
        self.command("safedown", "1")
        until(lambda: self.c.mode_name == "LAND", timeout=12)
        self.assertIn(RTL, [m for _t, m in self.fc.mode_requests], "RTL was asked for first")
        until(lambda: "SAFEDOWN complete: landed" in self.m.event, timeout=5)

    def test_losing_the_gps_on_the_way_home_lands_here(self):
        self.fc.rtl_s = 60
        self.fly()
        self.command("safedown", "1")
        until(lambda: self.c.mode_name == "RTL", timeout=4)
        self.fc.gps = False
        until(lambda: self.c.mode_name == "LAND", timeout=5)
        self.assertIn("GPS position lost", self.m.event)

    def test_the_pilots_stabilize_takes_it_back_on_the_way(self):
        self.fc.rtl_s = 60
        self.fly()
        self.command("safedown", "1")
        until(lambda: self.c.mode_name == "RTL", timeout=4)
        self.fc.radio[5] = 1100  # the pilot moves the mode switch
        self.fc.mode = STABILIZE
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides, timeout=5)
        self.assertIn("SAFEDOWN ended", self.m.event)


class ArmMavlinkTests(WifiFixture):
    """The phone's ARM: the app arms the FC through the arm switch's function."""
    fake_gps = True

    def ready_disarmed(self):
        self.command("flightmode", "wasp")
        self.command("mount", "90")
        self.fc.mode, self.fc.armed = ACRO, False
        until(lambda: self.c.mode_name == "ACRO" and not self.c.armed)
        until(lambda: self.c.mission_state()["wasp_arm_why"] == "", timeout=4)

    def test_arm_then_launch_wasp_after_two_seconds(self):
        self.ready_disarmed()
        st = self.c.mission_state()
        self.assertTrue(st["compass_ok"] and st["gps"]["ok"] and st["position"])
        self.command("arm", "1")
        until(lambda: self.c.armed, timeout=3)
        self.assertEqual([(f, pos) for _t, f, pos in self.fc.arm_requests], [(154, 2)],
                         "the arm switch's own function, switch HIGH")
        until(lambda: self.m.phase == "WASP_READY")
        with urlopen_cmd(self, "engage") as refused:
            self.assertIn("after 2.5 s", refused)
        until(lambda: (self.c.mission_state()["armed_s"] or 0) > 2.55, timeout=4)
        self.command("engage", "1")
        until(lambda: self.m.phase == "WASP_FLIGHT" and len(self.fc.overrides) == 4, timeout=2)

    def test_arm_refused_in_another_mode_and_by_the_fc(self):
        self.command("flightmode", "wasp")
        self.command("mount", "90")
        self.fc.mode, self.fc.armed = STABILIZE, False
        until(lambda: self.c.mode_name == "STABILIZE" and not self.c.armed)
        with urlopen_cmd(self, "arm") as refused:
            self.assertIn("ACRO", refused)
        self.assertFalse(self.fc.arm_requests)
        self.fc.mode, self.fc.arm_refuse = ACRO, True
        until(lambda: self.c.mission_state()["wasp_arm_why"] == "", timeout=4)
        self.command("arm", "1")
        until(lambda: "ARM refused by the FC" in self.m.refusal or "ARM refused" in self.m.event, timeout=4)
        self.assertFalse(self.c.armed)

    def test_no_compass_shows_in_the_state(self):
        self.fc.compass = False
        until(lambda: self.c.mission_state()["compass_ok"] is False, timeout=4)
        self.fc.compass = True
        until(lambda: self.c.mission_state()["compass_ok"], timeout=4)


class urlopen_cmd:
    """A command the app refuses: the refusal text."""
    def __init__(self, test, key, value="1"):
        self.test, self.key, self.value = test, key, value

    def __enter__(self):
        import json
        from urllib.parse import urlencode
        from urllib.request import urlopen
        with urlopen(self.test.url+"/cmd?"+urlencode({"c": "phone", self.key: self.value}), timeout=3) as r:
            st = json.load(r)
        self.test.assertTrue(st.get("refused"), f"{self.key} was not refused")
        return st["refused"]

    def __exit__(self, *_exc):
        return False


class ModeSwitchMavlinkTests(WifiFixture):
    # FLTMODE_CH 5: position 0 ACRO, 1 LOITER, 2 LAND, 3 STABILIZE, 4 POSHOLD, 5 ALT_HOLD.
    fake_params = dict(FLTMODE_CH=5, FLTMODE1=ACRO, FLTMODE2=LOITER, FLTMODE3=LAND,
                       FLTMODE4=STABILIZE, FLTMODE5=16, FLTMODE6=ALT_HOLD)
    PWM = dict(ACRO=1165, LOITER=1295, LAND=1425, STABILIZE=1555, POSHOLD=1685, ALT_HOLD=1815)
    fake_radio = {5: PWM["ACRO"]}

    fly = SafeDownMavlinkTests.fly
    CENTRED = SafeDownMavlinkTests.CENTRED
    assert_safedown = SafeDownMavlinkTests.assert_safedown
    assert_holds = SafeDownMavlinkTests.assert_holds
    assert_lands = SafeDownMavlinkTests.assert_lands

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

    def test_switch_up_starts_safedown_before_the_heartbeat_shows_it(self):
        self.fly()
        moved = self.switch("ALT_HOLD")
        until(lambda: self.m.phase == "SAFEDOWN", timeout=.5)
        self.assertLess(perf_counter()-moved, .3)
        self.assertIn("ALT_HOLD", self.m.event)
        until(lambda: self.c.mode_name == "ALT_HOLD", timeout=2)
        self.assert_holds()
        # The switch stays up: SafeDown carries on and lands it.
        self.assert_lands()

    def test_switch_to_stabilize_in_flight_catches_then_hands_over(self):
        self.fly()
        self.switch("STABILIZE")
        self.fc.mode = STABILIZE
        until(lambda: self.m.phase == "SAFEDOWN" and self.m.safedown.catch, timeout=.5)
        self.assertTrue(self.fc.overrides, "the app steadies it first")
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides, timeout=5)
        self.assertIn("CATCH done", self.m.event)
        self.assertFalse(self.fc.mode_requests, "no mode changes in a catch")
        self.assertTrue(self.fc.armed and self.c.armed)

    def test_switch_up_after_the_button_carries_on(self):
        self.fly()
        self.command("safedown", "1")
        self.assert_safedown()
        self.switch("ALT_HOLD")
        self.fc.mode = ALT_HOLD  # the FC follows the switch
        sleep(.5)
        self.assertEqual(self.m.phase, "SAFEDOWN")
        self.assertIn("SafeDown carries on", self.m.event)
        self.assert_lands()

    def test_switch_to_stabilize_during_safedown_catches_then_is_the_pilots(self):
        self.fly()
        self.command("safedown", "1")
        self.assert_safedown()
        self.assert_holds()
        self.switch("STABILIZE")
        self.fc.mode = STABILIZE
        until(lambda: self.m.safedown.catch, timeout=1)
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides, timeout=5)
        self.assertIn("CATCH done", self.m.event)
        self.assertFalse(self.m.safedown.active)
        self.assertTrue(self.fc.armed and self.c.armed)
        sleep(.3)
        self.assertFalse(self.fc.overrides)

    def test_switch_back_up_after_the_pilots_stabilize_lands_it(self):
        self.fly()
        self.fc.vfr["alt"] += 5
        self.switch("STABILIZE")
        self.fc.mode = STABILIZE
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides, timeout=5)
        self.switch("ALT_HOLD")
        self.fc.mode = ALT_HOLD
        until(lambda: self.m.phase == "SAFEDOWN")
        self.assert_lands()

    def test_switch_to_loiter_in_flight_releases_without_safedown(self):
        self.fly()
        self.switch("LOITER")
        until(lambda: self.m.phase == "WASP_DONE" and not self.fc.overrides, timeout=.5)
        self.assertEqual(self.c.mode_name, "ACRO")
        self.assertFalse(self.m.safedown.active)
        sleep(.3)
        self.assertFalse(self.fc.mode_requests)
        self.assertFalse(self.fc.overrides)
        self.assertEqual(self.m.phase, "WASP_DONE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
