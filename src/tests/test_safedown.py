import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission import Mission
from safedown import SafeDown

LEVEL = dict(roll=0.0, pitch=0.0, yaw=1.0, rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0)


class Mapper:
    camera_mount_deg = 90
    hover_thrust = .125

    def __init__(self):
        self.armed = False
        self.requests = []
        self.disarms = 0
        self.forced = 0

    def wasp_pwm(self, rates, throttle):
        return dict(kind="manual", rates=tuple(rates), throttle=throttle)

    def idle_pwm(self, yaw, climb):
        return dict(kind="alt", yaw=yaw, climb=climb)

    def request_mode(self, name):
        self.requests.append(name)
        return True

    def request_disarm(self, force=False):
        self.disarms += 1
        self.forced = self.forced + 1 if force else getattr(self, "forced", 0)
        return True


class FC:
    """Just enough FC for SafeDown: ACKs, a slow heartbeat, refusals."""

    def __init__(self, mapper, mode="ACRO"):
        self.mapper = mapper
        self.mode = self.heartbeat = mode
        self.refuse = set()
        self.silent = set()
        self.ack = None
        self.ack_delay = .03
        self.hb_delay = .5
        self.pending = []
        self.seen = 0
        self.landed_state = 2
        self.att = dict(LEVEL)
        self.height, self.climb = 6.0, 0.0
        self.accel = (0.0, 0.0, -9.8)

    def step(self, now):
        for name in self.mapper.requests[self.seen:]:
            self.pending.append((now+self.ack_delay, name))
        self.seen = len(self.mapper.requests)
        for due, name in list(self.pending):
            if due <= now:
                self.pending.remove((due, name))
                if name in self.silent:
                    continue
                accepted = name not in self.refuse
                self.ack = (name, accepted, due)
                if accepted:
                    self.set_mode(name, now)
        if self._hb_due is not None and now >= self._hb_due:
            self.heartbeat, self._hb_due = self.mode, None

    _hb_due = None

    def set_mode(self, name, now):
        self.mode = name
        self._hb_due = now + self.hb_delay

    def ctx(self, **extra):
        ctx = dict(gate_ok=self.mapper.armed, gate_why="" if self.mapper.armed else "disarmed",
                   att=self.att, height=self.height, climb=self.climb, accel=self.accel,
                   fc_mode=self.heartbeat, mode_ack=self.ack, landed_state=self.landed_state,
                   airborne=True, idle_ok=True, pilot_takeover=False, pilot_safedown=False,
                   safedown_ok=self.mapper.armed, ai_cmd=None, det_t=None, det_conf=None)
        ctx.update(extra)
        return ctx


class SafeDownFlight(unittest.TestCase):
    def setUp(self):
        self.mapper = Mapper()
        self.m = Mission(self.mapper, log=lambda *_: None)
        self.assertEqual(self.m.set_wasp_enabled(True), "")
        self.assertEqual(self.m.confirm_mount(90), "")
        self.m.wasp.params.update(ramp=0.0, straight=1.0, launch_s=0.0)
        self.mapper.armed = True
        self.fc = FC(self.mapper)
        self.now = 97.0
        self.out = self.m.cycle(self.now, self.fc.ctx())
        self.assertEqual(self.m.phase, "WASP_READY")

    def launch(self):
        self.assertEqual(self.m.request_engage(), "")
        self.run_for(.5)
        self.assertEqual(self.m.phase, "WASP_FLIGHT")

    def tick(self, dt=.02, **extra):
        self.now = round(self.now + dt, 6)
        self.fc.step(self.now)
        self.out = self.m.cycle(self.now, self.fc.ctx(**extra))
        return self.out

    def run_for(self, seconds, until=None, **extra):
        end = self.now + seconds
        while self.now < end - 1e-9:
            self.tick(**extra)
            if until is not None and until():
                return True
        return False

    def abort(self):
        self.launch()
        self.m.request_abort()
        self.tick()
        self.assertEqual(self.m.phase, "SAFEDOWN")
        self.assertTrue(self.m.wasp.used)
        self.assertIsNone(self.m.wasp.started)

    @property
    def r(self):
        return self.m.safedown

    def assert_land_and_disarm(self):
        self.assertTrue(self.run_for(12, lambda: self.r.stage == "DESCEND"))
        self.assertTrue(self.run_for(3, lambda: self.fc.mode == "LAND"))
        self.run_for(.5)
        self.assertEqual(self.out, dict(kind="alt", yaw=0.0, climb=0.0))
        self.fc.landed_state = 1
        self.mapper.armed = False
        self.tick()
        self.assertFalse(self.r.active)
        self.assertIn("landed, FC disarmed", self.m.event)
        self.assertEqual(self.out, {})


class ShapeOfSafeDown(SafeDownFlight):
    def test_abort_levels_stops_then_lands_and_ends_when_land_disarms(self):
        self.abort()
        self.assertEqual(self.mapper.requests, ["ALT_HOLD"])
        self.assertIn("level, stop, then fly home and land", self.m.event)
        # Before the ACK: ACRO, launch throttle, levelling rates (level: none).
        self.assertEqual(self.out, dict(kind="manual", rates=(0.0, 0.0, 0.0), throttle=1.0))
        self.assertTrue(self.run_for(.1, lambda: self.out["kind"] == "alt"))
        self.assertEqual(self.out, dict(kind="alt", yaw=0.0, climb=0.0))
        self.assertEqual(self.fc.heartbeat, "ACRO", "the ACK alone already means ALT_HOLD")
        self.assertIn(self.r.stage, ("LEVEL", "STOP"))
        # Level and steady from the start: STOP for at least MIN_STOP_S.
        began = self.now
        self.assertTrue(self.run_for(4, lambda: self.r.stage == "DESCEND"))
        self.assertAlmostEqual(self.now - began, SafeDown.MIN_STOP_S, delta=.1)
        self.assertEqual(self.mapper.requests[-1], "LAND")
        self.assert_land_and_disarm()
        self.assertEqual(self.mapper.requests, ["ALT_HOLD", "LAND"])
        self.assertEqual(self.mapper.disarms, 0, "LAND disarms by itself")
        self.tick()
        self.assertEqual(self.m.phase, "PILOT")
        self.assertFalse(self.m.wasp.used, "disarmed: ready for the next launch once armed")

    def test_pilot_switch_up_in_flight_runs_the_same_safedown(self):
        self.launch()
        # The switch up puts the FC in ALT_HOLD itself: centred ALT_HOLD sticks
        # at once, never the launch throttle (which ALT_HOLD reads as a climb).
        self.fc.mode = self.fc.heartbeat = "ALT_HOLD"
        self.tick(gate_ok=False, pilot_takeover=True, pilot_safedown=True)
        self.assertEqual(self.m.phase, "SAFEDOWN")
        self.assertIn("pilot switched to ALT_HOLD", self.m.event)
        self.assertEqual(self.out, dict(kind="alt", yaw=0.0, climb=0.0))
        self.assert_land_and_disarm()

    def test_the_pilots_stabilize_is_his_own_flight(self):
        self.launch()
        self.fc.mode = self.fc.heartbeat = "STABILIZE"
        self.tick(gate_ok=False, gate_why="pilot switched to STABILIZE")
        self.tick(gate_ok=False, gate_why="pilot switched to STABILIZE")
        self.assertFalse(self.r.active)
        self.assertEqual(self.out, {})
        self.assertEqual(self.m.phase, "WASP_DONE")

    def test_levelling_waits_for_level_steady_and_drag_to_fade(self):
        self.abort()
        tilted = dict(LEVEL, roll=math.radians(25))
        self.fc.att = tilted
        self.run_for(1)
        self.assertEqual(self.r.stage, "LEVEL")
        self.fc.att = dict(LEVEL)
        self.fc.accel = (3.0, 0.0, -9.8)  # level, but still moving fast: drag
        self.run_for(1.5)
        self.assertEqual(self.r.stage, "STOP")
        self.fc.climb = -.9
        self.fc.accel = (0.4, 0.3, -9.8)
        self.run_for(1.5)
        self.assertEqual(self.r.stage, "STOP", "still sinking")
        self.fc.climb = 0.0
        self.fc.accel = (0.4, 0.3, -9.8)
        self.assertTrue(self.run_for(3, lambda: self.r.stage == "DESCEND"))
        self.assertIn("level and stopped", self.m.event)

    def test_never_settling_still_lands_after_the_stop_limit(self):
        self.abort()
        self.fc.climb = 1.2
        began = self.now
        self.assertTrue(self.run_for(12, lambda: self.r.stage == "DESCEND"))
        self.assertAlmostEqual(self.now - began, SafeDown.MAX_STOP_S, delta=.15)
        self.assertIn("LAND anyway", self.m.event)

    def test_stale_telemetry_centres_the_sticks_and_still_lands(self):
        self.abort()
        self.run_for(.2)
        for i in range(int(SafeDown.MAX_STOP_S/.02) + 10):
            self.tick(att=None, height=None)
            if self.r.stage == "DESCEND":
                break
            self.assertEqual(self.out, dict(kind="alt", yaw=0.0, climb=0.0))
        self.assertEqual(self.r.stage, "DESCEND")
        self.assert_land_and_disarm()

    def test_a_failsafe_land_skips_straight_to_the_descent(self):
        self.abort()
        self.run_for(.3)
        self.fc.set_mode("LAND", self.now)
        self.fc.heartbeat = "LAND"
        self.tick()
        self.assertEqual(self.r.stage, "DESCEND")
        self.assertIn("FC switched to LAND itself", self.m.event)
        self.run_for(1)
        self.assertEqual(self.mapper.requests, ["ALT_HOLD"])

    def test_on_the_ground_right_after_go_cancels_instead(self):
        self.assertEqual(self.m.request_engage(), "")
        self.tick()
        self.m.request_abort()
        self.tick(airborne=False)
        self.assertEqual(self.m.phase, "WASP_DONE")
        self.assertIn("launch cancelled", self.m.event)
        self.assertFalse(self.r.active)
        self.assertEqual(self.mapper.requests, [])


class WeirdAngles(SafeDownFlight):
    def test_acro_levelling_rates_turn_the_thrust_axis_up(self):
        r = SafeDown()
        roll, pitch, yaw = r._acro_level_rates(dict(roll=math.radians(30), pitch=0.0))
        self.assertLess(roll, 0)
        self.assertAlmostEqual(pitch, 0.0)
        self.assertEqual(yaw, 0.0)
        roll, pitch, _ = r._acro_level_rates(dict(roll=0.0, pitch=math.radians(-40)))
        self.assertGreater(pitch, 0)
        self.assertAlmostEqual(roll, 0.0)
        # Upside down: the rate is capped and still turns it over.
        rates = r._acro_level_rates(dict(roll=math.radians(179.9), pitch=0.0))
        self.assertAlmostEqual(math.hypot(*rates), math.radians(SafeDown.ACRO_LEVEL_MAX_DPS))
        rates = r._acro_level_rates(dict(roll=0.0, pitch=math.radians(180)))
        self.assertAlmostEqual(math.hypot(*rates), math.radians(SafeDown.ACRO_LEVEL_MAX_DPS))

    def test_tumbling_at_abort_gets_levelling_rates_and_little_thrust(self):
        self.launch()
        self.fc.att = dict(LEVEL, roll=math.radians(130), rollspeed=4.0)
        self.fc.ack_delay = .4
        self.m.request_abort()
        self.tick()
        self.assertEqual(self.out["kind"], "manual")
        self.assertLess(self.out["rates"][0], 0)
        self.assertLess(self.out["throttle"], .5, "no launch throttle pointing at the ground")
        self.fc.att = dict(LEVEL, roll=math.radians(40))
        self.tick()
        self.assertEqual(self.out["throttle"], 1.0, "under 60 degrees the launch throttle bridges")

    def test_launch_throttle_bridges_only_half_a_second_without_alt_hold(self):
        self.launch()
        self.fc.silent.add("ALT_HOLD")
        self.m.request_abort()
        self.tick()
        self.assertEqual(self.out["throttle"], 1.0)
        self.run_for(SafeDown.HOLD_THROTTLE_S + .05)
        self.assertEqual(self.out["kind"], "manual")
        self.assertLess(self.out["throttle"], .9)
        # Retried every MODE_RETRY_S, then STABILIZE once ALT_HOLD is overdue.
        self.assertGreaterEqual(self.mapper.requests.count("ALT_HOLD"), 2)
        self.assertTrue(self.run_for(2, lambda: "STABILIZE" in self.mapper.requests))
        self.assertIn("ALT_HOLD not confirmed", self.m.event)


class FastVertical(SafeDownFlight):
    def test_a_fast_climb_is_braked_in_stabilize_with_little_thrust_first(self):
        self.launch()
        self.fc.climb = 25.0
        self.m.request_abort()
        self.tick()
        self.assertEqual(self.r.stage, "ARREST")
        self.assertEqual(self.mapper.requests, ["STABILIZE"], "no ALT_HOLD at 25 m/s")
        self.assertIn("climbing at 25 m/s", self.m.event)
        self.run_for(.2)
        self.assertEqual(self.fc.mode, "STABILIZE")
        self.assertLess(self.out["throttle"], .3, "gravity brakes it: thrust near zero")
        self.fc.climb = 4.0
        self.run_for(.2)
        self.assertEqual(self.r.stage, "ARREST")
        self.fc.climb = 1.0
        self.assertTrue(self.run_for(.5, lambda: self.fc.mode == "ALT_HOLD"))
        self.assertEqual(self.mapper.requests[-1], "ALT_HOLD")
        self.fc.climb = 0.0
        self.assert_land_and_disarm()

    def test_a_fast_dive_gets_most_of_the_thrust(self):
        self.launch()
        self.fc.climb = -20.0
        self.m.request_abort()
        self.run_for(.3)
        self.assertEqual(self.r.stage, "ARREST")
        self.assertIn("diving at 20 m/s", self.m.event)
        self.assertGreater(self.out["throttle"], .75, "full-range throttle against the dive")
        # Upside down mid-dive: right it first with little thrust.
        self.fc.att = dict(LEVEL, roll=math.radians(150))
        self.tick()
        self.assertLess(self.out["throttle"], .5)

    def test_arrest_falls_back_to_alt_hold_when_stabilize_is_refused_or_blind(self):
        for case in ("refused", "stale"):
            with self.subTest(case=case):
                self.setUp()
                if case == "refused":
                    self.fc.refuse.add("STABILIZE")
                self.launch()
                self.fc.climb = 15.0
                self.m.request_abort()
                self.tick()
                self.assertEqual(self.r.stage, "ARREST")
                if case == "stale":
                    self.tick(att=None)
                else:
                    self.run_for(.2)
                self.assertIn("ALT_HOLD", self.mapper.requests)
                self.assertNotEqual(self.r.stage, "ARREST")
                self.run_for(.5)
                self.assertEqual(self.fc.mode, "ALT_HOLD")
                self.assertIn(self.r.stage, ("LEVEL", "STOP"))

    def test_arrest_happens_once_and_never_outlasts_its_timeout(self):
        self.launch()
        self.fc.climb = 30.0
        self.m.request_abort()
        self.tick()
        began = self.now
        self.assertTrue(self.run_for(8, lambda: self.r.stage != "ARREST"))
        self.assertAlmostEqual(self.now - began, SafeDown.ARREST_TIMEOUT_S, delta=.1)
        self.run_for(1)
        self.assertNotEqual(self.r.stage, "ARREST")


class BrokenEkf(SafeDownFlight):
    def test_ekf_climb_against_the_barometer_switches_to_the_app_height_hold(self):
        self.abort()
        self.run_for(1, baro_climb=0.0)
        self.assertEqual(self.fc.mode, "ALT_HOLD")
        # The EKF says level flight; the barometer says it is falling.
        self.fc.climb = 0.2
        self.run_for(SafeDown.EKF_BAD_S + .1, baro_climb=-6.0)
        self.assertTrue(self.r.ekf_bad)
        self.assertIn("barometer", self.m.event)
        self.run_for(.3, baro_climb=-6.0)
        self.assertEqual(self.fc.mode, "STABILIZE")
        self.assertGreater(self.out["throttle"], .6, "arrests the real fall, not the EKF's")
        asked = len(self.mapper.requests)
        self.run_for(2, baro_climb=-6.0)
        self.assertEqual(len(self.mapper.requests), asked, "no ALT_HOLD offers while the EKF is wrong")
        # They agree again for EKF_GOOD_S: the FC's own height hold comes back.
        self.run_for(SafeDown.EKF_GOOD_S + .2, baro_climb=0.0)
        self.assertFalse(self.r.ekf_bad)
        self.assertTrue(self.run_for(2, lambda: self.fc.mode in ("ALT_HOLD", "LAND"), baro_climb=0.0))

    def test_a_hard_deceleration_is_not_a_broken_ekf(self):
        # The barometer's slope lags the EKF by half its window: braking a
        # 30 m/s climb at 15 m/s^2 must not look like a disagreement.
        self.abort()
        t0 = self.now
        while self.now < t0 + 3:
            climb = max(0.0, 30 - 15*(self.now - t0))
            lagged = max(0.0, 30 - 15*(self.now - t0 - SafeDown.BARO_WINDOW_S/2))
            self.fc.climb = climb
            self.tick(baro_climb=min(30.0, lagged))
        self.assertFalse(self.r.ekf_bad)

    def test_short_or_small_disagreements_change_nothing(self):
        self.abort()
        self.run_for(1, baro_climb=0.0)
        self.run_for(SafeDown.EKF_BAD_S - .1, baro_climb=-5.0)
        self.run_for(.5, baro_climb=-2.5)
        self.assertFalse(self.r.ekf_bad)
        self.assertIsNone(self.r.fallback)
        self.assertIn(self.fc.mode, ("ALT_HOLD", "LAND"))

    def test_a_broken_ekf_in_the_descent_still_lands_on_the_barometer(self):
        self.abort()
        self.assertTrue(self.run_for(4, lambda: self.fc.mode == "LAND", baro_climb=0.0))
        self.fc.climb = 0.0
        self.run_for(SafeDown.EKF_BAD_S + .2, baro_climb=-5.0)
        self.assertEqual(self.r.fallback, "STABILIZE")
        self.assertEqual(self.r.stage, "DESCEND")
        self.fc.landed_state = 2  # the FC's land detector is blind too
        self.fc.height = .1
        self.run_for(.5, baro_climb=-.5)
        self.run_for(SafeDown.TOUCHDOWN_S + .2, baro_climb=0.0)
        self.assertTrue(self.r.landed)
        self.assertEqual(self.out["throttle"], 0.0)
        self.assertGreaterEqual(self.mapper.disarms, 1)


class ForcedDisarm(SafeDownFlight):
    def landed_blind(self):
        self.fc.refuse.update(("ALT_HOLD", "LAND"))
        self.fc.landed_state = None  # no landed state the app could use
        self.abort()
        self.assertTrue(self.run_for(12, lambda: self.r.stage == "DESCEND", baro_climb=0.0))
        self.fc.climb, self.fc.height = -.5, 1.0
        self.run_for(1, baro_climb=-.5)
        self.fc.climb, self.fc.height = 0.0, .05

    def test_forced_only_after_seconds_of_proof_the_ground_holds_it(self):
        self.landed_blind()
        self.run_for(SafeDown.TOUCHDOWN_S + .1, baro_climb=0.0)
        self.assertTrue(self.r.landed)
        self.assertEqual(self.mapper.forced, 0)
        self.run_for(SafeDown.FORCE_DISARM_S + 1.2, baro_climb=0.0)
        self.assertGreaterEqual(self.mapper.forced, 1)
        self.assertIn("forcing it", self.m.event)

    def test_a_jittery_barometer_at_rest_still_allows_it(self):
        self.landed_blind()
        self.run_for(SafeDown.TOUCHDOWN_S + .1, baro_climb=0.0)
        for i in range(int((SafeDown.FORCE_DISARM_S + 1.5)/.02)):
            self.tick(baro_climb=(.5, -.5, .3, -.4)[i % 4])
        self.assertGreaterEqual(self.mapper.forced, 1)

    def test_never_forced_while_the_imu_feels_less_than_1_g(self):
        self.landed_blind()
        self.fc.accel = (0.0, 0.0, -3.0)  # thrust near zero, falling
        self.run_for(SafeDown.TOUCHDOWN_S + SafeDown.FORCE_DISARM_S + 2, baro_climb=0.0)
        self.assertEqual(self.mapper.forced, 0)

    def test_never_forced_while_the_barometer_moves(self):
        self.landed_blind()
        self.run_for(SafeDown.TOUCHDOWN_S + SafeDown.FORCE_DISARM_S + 2, baro_climb=-.8)
        self.assertEqual(self.mapper.forced, 0)


class AirBrake(SafeDownFlight):
    def setUp(self):
        super().setUp()
        self.leans = []
        self.mapper.lean_pwm = lambda roll, pitch: self.leans.append((roll, pitch)) or dict(lean=(roll, pitch))

    def test_leans_against_the_drag_and_holds_the_lean_that_stops_it(self):
        self.abort()
        self.run_for(.3)
        # Drifting forward through the air: drag pushes back (-x).
        self.fc.accel = (-2.0, 0.0, -9.8)
        self.run_for(1)
        roll, pitch = self.out["lean"]
        self.assertGreater(pitch, 0, "nose up against a forward slide")
        self.assertAlmostEqual(roll, 0.0)
        self.assertLessEqual(pitch, SafeDown.BRAKE_MAX_DEG)
        # Sliding right: drag pushes left (-y): lean left.
        self.fc.accel = (0.0, -2.0, -9.8)
        self.run_for(1)
        self.assertLess(self.out["lean"][0], 0)
        # Still relative to the air: the integral keeps the lean that undoes
        # a level bias instead of dropping back to the FC's wrong "level".
        self.fc.accel = (0.0, 0.0, -9.8)
        self.run_for(.6)
        self.assertLess(self.out["lean"][0], -1.0)
        self.assertIn("brake_deg", self.m.snapshot(self.now)["safedown"])

    def test_no_brake_when_the_fc_has_a_gps_position(self):
        self.abort()
        self.fc.accel = (-2.0, 1.0, -9.8)
        self.run_for(1, position_ok=True)
        self.assertNotIn("lean", self.out)
        self.run_for(.5)
        self.assertIn("lean", self.out)

    def test_tiny_drag_is_ignored_and_the_lean_is_capped(self):
        self.abort()
        self.fc.accel = (.1, -.1, -9.8)
        self.run_for(2)
        self.assertEqual(self.out["lean"], (0.0, 0.0))
        self.fc.accel = (-30.0, 30.0, -9.8)
        self.run_for(5)
        self.assertEqual(self.out["lean"], (SafeDown.BRAKE_MAX_DEG, SafeDown.BRAKE_MAX_DEG))

    def test_no_brake_in_acro_while_switching_or_after_touchdown(self):
        self.launch()
        self.fc.ack_delay = .4
        self.fc.accel = (-3.0, 0.0, -9.8)
        self.m.request_abort()
        self.run_for(.2)
        self.assertEqual(self.r.stage, "SWITCHING")
        self.assertNotIn("lean", self.out)
        self.fc.refuse.update(("LAND",))
        self.assertTrue(self.run_for(12, lambda: self.r.fallback == "ALT_HOLD"))
        self.assertIn("lean", self.out)
        self.fc.landed_state = 1
        self.run_for(SafeDown.TOUCHDOWN_S + .1)
        self.assertTrue(self.r.landed)
        self.assertNotIn("lean", self.out)


class GpsReturn(SafeDownFlight):
    GPS = dict(position_ok=True, home_ok=True, home_m=120.0)

    def test_level_then_rtl_without_waiting_for_the_stop(self):
        self.launch()
        self.m.request_abort()
        self.tick(**self.GPS)
        self.assertTrue(self.run_for(1, lambda: self.r.stage == "RETURN", **self.GPS))
        self.assertLess(self.now - self.r.started, 1.0, "no 2-8 s stop wait with GPS")
        self.assertIn("RTL, flying back to the launch point (120 m)", self.m.event)
        self.assertTrue(self.run_for(1, lambda: self.fc.mode == "RTL", **self.GPS))
        self.run_for(.3, **self.GPS)
        self.assertEqual(self.out, dict(kind="alt", yaw=0.0, climb=0.0))
        self.assertEqual(self.mapper.requests, ["ALT_HOLD", "RTL"])
        self.assertEqual(self.m.snapshot(self.now)["safedown"]["home_m"], 120)
        self.mapper.armed = False
        self.tick(**self.GPS)
        self.assertIn("SAFEDOWN complete: landed", self.m.event)

    def test_a_fast_climb_is_braked_before_the_return(self):
        self.launch()
        self.fc.climb = 25.0
        self.m.request_abort()
        self.tick(**self.GPS)
        self.assertEqual(self.r.stage, "ARREST")
        self.run_for(.5, **self.GPS)
        self.assertNotIn("RTL", self.mapper.requests)
        self.fc.climb = 0.5
        self.assertTrue(self.run_for(2, lambda: self.fc.mode == "RTL", **self.GPS))

    def test_rtl_refused_lands_here(self):
        self.fc.refuse.add("RTL")
        self.abort()
        self.assertTrue(self.run_for(3, lambda: "RTL refused" in self.m.event, **self.GPS))
        self.assertTrue(self.run_for(10, lambda: self.fc.mode == "LAND", **self.GPS))
        self.assertNotEqual(self.r.stage, "RETURN")

    def test_losing_the_gps_on_the_way_home_lands_here(self):
        self.abort()
        self.assertTrue(self.run_for(2, lambda: self.fc.mode == "RTL", **self.GPS))
        self.run_for(SafeDown.POSITION_LOST_S - .2, position_ok=False, home_ok=True)
        self.assertEqual(self.r.stage, "RETURN", "a short dropout is ridden out")
        self.run_for(.5, **self.GPS)
        self.run_for(SafeDown.POSITION_LOST_S + .3, position_ok=False, home_ok=True)
        self.assertIn("GPS position lost", self.m.event)
        self.assertTrue(self.run_for(1, lambda: self.fc.mode == "LAND", position_ok=False))

    def test_a_failsafe_land_on_the_way_home_is_accepted(self):
        self.abort()
        self.assertTrue(self.run_for(2, lambda: self.fc.mode == "RTL", **self.GPS))
        self.fc.set_mode("LAND", self.now)
        self.fc.heartbeat = "LAND"
        self.tick(**self.GPS)
        self.assertEqual(self.r.stage, "DESCEND")
        self.run_for(1, **self.GPS)
        self.assertEqual(self.mapper.requests[-1], "RTL", "no fight with the FC's LAND")

    def test_a_position_found_on_the_way_down_still_flies_it_home(self):
        self.abort()
        # No compass: no position yet, so it stops and starts down in LAND.
        self.assertTrue(self.run_for(10, lambda: self.fc.mode == "LAND", position_ok=False, home_ok=True))
        self.fc.height = 30.0
        self.tick(**self.GPS)
        self.assertEqual(self.r.stage, "RETURN")
        self.assertIn("GPS position found on the way down", self.m.event)
        self.assertTrue(self.run_for(1, lambda: self.fc.mode == "RTL", **self.GPS))

    def test_too_low_to_turn_back(self):
        self.abort()
        self.assertTrue(self.run_for(10, lambda: self.fc.mode == "LAND", position_ok=False, home_ok=True))
        self.fc.height = 3.0
        self.run_for(1, **self.GPS)
        self.assertEqual(self.r.stage, "DESCEND")

    def test_no_home_no_return(self):
        self.abort()
        self.run_for(4, position_ok=True, home_ok=False)
        self.assertNotIn("RTL", self.mapper.requests)
        self.assertTrue(self.run_for(8, lambda: self.fc.mode == "LAND", position_ok=True, home_ok=False))


class Catch(SafeDownFlight):
    def stab(self, **extra):
        self.fc.mode = self.fc.heartbeat = "STABILIZE"
        return self.tick(gate_ok=False, catch_takeover=True, pilot_catch=True, **extra)

    def test_stabilize_steadies_it_then_hands_it_to_the_pilot(self):
        self.launch()
        self.stab()
        self.assertEqual((self.m.phase, self.r.stage), ("SAFEDOWN", "CATCH"))
        self.assertIn("CATCH: pilot switched to STABILIZE", self.m.event)
        self.assertEqual(self.out["kind"], "manual")
        self.assertEqual(self.out["rates"], (0.0, 0.0, 0.0), "STABILIZE levels it: centred sticks")
        began = self.now
        self.assertTrue(self.run_for(4, lambda: not self.r.active, pilot_catch=True))
        self.assertAlmostEqual(self.now - began, SafeDown.CATCH_MIN_S, delta=.1)
        self.assertIn("CATCH done: caught: level and steady", self.m.event)
        self.assertEqual(self.mapper.requests, [], "never a mode change")
        self.tick(gate_ok=False, pilot_catch=True)
        self.assertEqual(self.out, {})
        self.assertEqual(self.m.phase, "WASP_DONE")

    def test_a_tumbling_or_climbing_drone_is_held_up_to_the_limit(self):
        self.launch()
        self.fc.climb = 12.0
        self.stab()
        self.assertLess(self.out["throttle"], .3, "full range: little thrust against a climb")
        began = self.now
        self.assertTrue(self.run_for(5, lambda: not self.r.active, pilot_catch=True))
        self.assertAlmostEqual(self.now - began, SafeDown.CATCH_MAX_S, delta=.1)
        self.assertIn(f"caught for {SafeDown.CATCH_MAX_S:g} s", self.m.event)

    def test_a_dive_gets_the_thrust_and_it_leans_against_the_drag(self):
        self.mapper.lean_pwm = lambda roll, pitch: dict(lean=(roll, pitch))
        self.launch()
        self.fc.climb = -15.0
        self.fc.accel = (-3.0, 0.0, -9.8)
        self.stab()
        self.run_for(.5, pilot_catch=True, position_ok=True)
        self.assertGreater(self.out["throttle"], .7)
        self.assertGreater(self.out["lean"][1], 0, "air brake in STABILIZE, GPS or not")

    def test_switch_up_from_a_catch_goes_on_to_the_return(self):
        self.launch()
        self.stab(mode_switch=2, mode_switch_mode="STABILIZE")
        self.fc.set_mode("ALT_HOLD", self.now)
        self.fc.heartbeat = "ALT_HOLD"
        self.tick(mode_switch=5, mode_switch_mode="ALT_HOLD")
        self.assertFalse(self.r.catch)
        self.assertIn("fly home and land", self.m.event)
        self.assert_land_and_disarm()

    def test_switch_to_stabilize_during_safedown_becomes_a_catch(self):
        self.launch()
        self.m.request_abort()
        self.tick(mode_switch=0, mode_switch_mode="ACRO")
        self.run_for(.5, mode_switch=0, mode_switch_mode="ACRO")
        self.fc.set_mode("STABILIZE", self.now)
        self.fc.heartbeat = "STABILIZE"
        self.tick(mode_switch=2, mode_switch_mode="STABILIZE")
        self.assertTrue(self.r.catch)
        self.assertIn("CATCH: pilot switched to STABILIZE", self.m.event)
        self.assertTrue(self.run_for(4, lambda: not self.r.active, mode_switch=2, mode_switch_mode="STABILIZE"))
        self.assertIn("CATCH done", self.m.event)

    def test_any_other_mode_ends_the_catch(self):
        self.launch()
        self.stab()
        self.fc.mode = self.fc.heartbeat = "LOITER"
        self.tick(pilot_catch=False)
        self.assertFalse(self.r.active)
        self.assertEqual(self.out, {})


class Fallbacks(SafeDownFlight):
    def test_alt_hold_refused_flies_stabilize_holding_height_and_still_lands(self):
        self.fc.refuse.add("ALT_HOLD")
        self.abort()
        self.run_for(.2)
        self.assertEqual(self.fc.mode, "STABILIZE")
        self.assertEqual(self.r.fallback, "STABILIZE")
        self.assertIn("STABILIZE levels", self.m.event)
        self.run_for(.6)
        hover = self.out["throttle"]
        self.assertAlmostEqual(hover, .5, delta=.05)  # mid stick is the FC's hover
        self.fc.climb = -1.5
        self.run_for(.3)
        self.assertGreater(self.out["throttle"], hover + .05)
        self.fc.climb = 1.5
        self.run_for(.6)
        self.assertLess(self.out["throttle"], hover)
        self.fc.climb = 0
        self.assertGreater(self.mapper.requests.count("ALT_HOLD"), 1, "ALT_HOLD keeps being offered")
        self.assert_land_and_disarm()

    def test_alt_hold_offer_accepted_later_returns_to_the_fc_height_hold(self):
        self.fc.refuse.add("ALT_HOLD")
        self.abort()
        self.run_for(.5)
        self.assertEqual(self.r.fallback, "STABILIZE")
        self.fc.refuse.clear()
        self.assertTrue(self.run_for(2, lambda: self.r.fallback is None))
        self.run_for(.1)
        self.assertEqual(self.out["kind"], "alt")
        self.assertIn("ALT_HOLD accepted", self.m.event)

    def test_stabilize_refused_too_fails_to_the_pilot(self):
        self.fc.refuse.update(("ALT_HOLD", "STABILIZE"))
        self.abort()
        self.assertTrue(self.run_for(3, lambda: self.m.phase == "SAFEDOWN_FAILED"))
        self.assertEqual(self.out, {})
        self.assertIn("SAFEDOWN FAILED", self.m.event)
        self.assertFalse(self.r.active)
        # No restart loop while the pilot's switch stays where it is.
        self.run_for(1, pilot_safedown=True)
        self.assertEqual(self.m.phase, "SAFEDOWN_FAILED")

    def test_land_refused_descends_in_alt_hold_then_disarms_on_touchdown(self):
        self.fc.refuse.add("LAND")
        self.abort()
        self.assertTrue(self.run_for(12, lambda: self.r.fallback == "ALT_HOLD"))
        self.assertIn("LAND refused", self.m.event)
        self.run_for(1.5)
        self.assertEqual(self.out["kind"], "alt")
        self.assertAlmostEqual(self.out["climb"], -SafeDown.DESCENT_MPS)
        self.fc.height = 14
        self.run_for(1.5)
        self.assertAlmostEqual(self.out["climb"], -SafeDown.HIGH_DESCENT_MPS)
        self.fc.height = .2
        self.fc.landed_state = 1
        self.run_for(SafeDown.TOUCHDOWN_S + .1)
        self.assertTrue(self.r.landed)
        self.assertLess(self.out["climb"], -100, "throttle stick to the bottom")
        self.assertEqual(self.mapper.disarms, 1)
        self.run_for(1.1)
        self.assertEqual(self.mapper.disarms, 2)
        self.assertGreater(self.mapper.requests.count("LAND"), 2, "LAND keeps being offered")
        self.mapper.armed = False
        self.tick()
        self.assertIn("landed, FC disarmed", self.m.event)

    def test_stabilize_descent_touchdown_needs_a_low_stopped_drone(self):
        self.fc.refuse.update(("ALT_HOLD", "LAND"))
        self.fc.landed_state = None
        self.abort()
        self.assertTrue(self.run_for(12, lambda: self.r.stage == "DESCEND"))
        # Not descending yet at 6 m: never taken for a touchdown.
        self.run_for(3)
        self.assertFalse(self.r.landed)
        self.assertGreater(self.out["throttle"], .3)
        self.fc.climb = -.5
        self.run_for(1)
        self.fc.climb, self.fc.height = 0.0, 4.0
        self.run_for(2)
        self.assertFalse(self.r.landed, "stopped but still high")
        self.fc.height = .1
        self.run_for(SafeDown.TOUCHDOWN_S + .1)
        self.assertTrue(self.r.landed)
        self.assertEqual(self.out["throttle"], 0.0)
        self.assertGreaterEqual(self.mapper.disarms, 1)

    def test_hover_thrust_is_learnt_from_the_imu(self):
        self.fc.refuse.update(("ALT_HOLD", "LAND"))
        self.abort()
        self.run_for(1)
        # Thrust commanded ~0.125 only gives 0.4 g: the real hover is ~0.31.
        for _ in range(150):
            thrust = self.r._thrust
            self.fc.accel = (0.0, 0.0, -9.80665*thrust/.31)
            self.tick()
        self.assertAlmostEqual(self.r._hover, .31, delta=.02)

    def test_stick_curve_inverse_matches_the_fc(self):
        for hover in (.125, .2, .35, .5, .7):
            e = max(-.5, min(1.0, -(hover-.5)/.375))
            for thrust in (.05, .125, .3, .6, .9):
                stick = SafeDown.stabilize_stick(thrust, hover)
                self.assertAlmostEqual(stick*(1-e) + e*stick**3, thrust, places=6)
            if hover <= .6875:  # the FC clamps its expo above that
                self.assertAlmostEqual(SafeDown.stabilize_stick(hover, hover), .5, places=6)


class PilotSwitch(SafeDownFlight):
    def start(self, band=0):
        self.launch()
        self.m.request_abort()
        self.tick(mode_switch=band, mode_switch_mode="ACRO")
        self.run_for(.5, mode_switch=band, mode_switch_mode="ACRO")
        self.assertEqual(self.m.phase, "SAFEDOWN")

    def test_switch_up_after_the_button_carries_on(self):
        self.start()
        self.fc.set_mode("ALT_HOLD", self.now)
        self.fc.heartbeat = "ALT_HOLD"
        self.tick(mode_switch=5, mode_switch_mode="ALT_HOLD")
        self.assertEqual(self.m.phase, "SAFEDOWN")
        self.assertIn("SafeDown carries on", self.m.event)
        self.run_for(.2, mode_switch=5, mode_switch_mode="ALT_HOLD")
        self.assertEqual(self.out["kind"], "alt")
        self.assertIsNone(self.r.fallback)

    def test_switch_up_while_landing_reasks_for_land(self):
        self.start()
        self.assertTrue(self.run_for(4, lambda: self.fc.mode == "LAND", mode_switch=0,
                                     mode_switch_mode="ACRO"))
        self.fc.set_mode("ALT_HOLD", self.now)
        self.fc.heartbeat = "ALT_HOLD"
        self.tick(mode_switch=5, mode_switch_mode="ALT_HOLD")
        self.run_for(.2, mode_switch=5, mode_switch_mode="ALT_HOLD")
        self.assertEqual(self.mapper.requests[-1], "LAND")
        self.assertEqual(self.fc.mode, "LAND")

    def test_any_other_switch_position_is_plan_b(self):
        # STABILIZE catches first (Catch tests); ACRO is the pilot at once.
        for band, mode in ((0, "ACRO"),):
            with self.subTest(mode=mode):
                self.setUp()
                self.start(band=3)
                self.tick(mode_switch=band, mode_switch_mode=mode)
                self.assertFalse(self.r.active)
                self.assertEqual(self.out, {})
                self.assertIn(f"pilot switched to {mode}", self.m.event)
                self.tick(mode_switch=band, mode_switch_mode=mode)
                self.assertEqual(self.m.phase, "WASP_DONE")

    def test_unknown_switch_modes_hand_over_on_any_move(self):
        self.start()
        self.tick(mode_switch=3, mode_switch_mode=None)
        self.assertFalse(self.r.active)
        self.assertIn("another mode", self.m.event)

    def test_fc_mode_changes_hand_over(self):
        for mode in ("LOITER", "ACRO", "POSHOLD"):
            with self.subTest(mode=mode):
                self.setUp()
                self.start()
                self.run_for(.5)
                self.fc.mode = self.fc.heartbeat = mode
                self.tick()
                self.assertFalse(self.r.active)
                self.assertIn(f"FC in {mode}" if mode != "ACRO" else "back in ACRO", self.m.event)

    def test_a_stale_ack_never_hides_a_later_mode_change(self):
        self.start()
        self.run_for(1)
        self.assertEqual(self.fc.heartbeat, "ALT_HOLD")
        self.fc.mode = self.fc.heartbeat = "LOITER"
        self.tick()
        self.assertFalse(self.r.active)

    def test_a_heartbeat_sent_after_the_ack_is_the_truth(self):
        self.launch()
        self.fc.hb_delay = 99  # this FC ACKs but never changes the heartbeat
        self.m.request_abort()
        self.run_for(.1, fc_mode_t=None)
        self.assertEqual(self.out["kind"], "alt", "ACK first: ALT_HOLD sticks")
        asked = self.mapper.requests.count("ALT_HOLD")
        for _ in range(30):  # a fresh heartbeat every cycle, still ACRO
            self.tick(fc_mode_t=self.now + .02)
        self.assertNotEqual(self.r.stage, "DESCEND")
        self.assertEqual(self.out["kind"], "manual", "a fresh ACRO heartbeat wins")
        self.assertGreater(self.mapper.requests.count("ALT_HOLD"), asked, "asked again")
        self.assertTrue(self.r.active)

    def test_disarm_in_the_air_ends_it_without_claiming_a_landing(self):
        self.start()
        self.mapper.armed = False
        self.tick()
        self.assertFalse(self.r.active)
        self.assertIn("SAFEDOWN ended: disarmed", self.m.event)


class TriggersAfterHandover(SafeDownFlight):
    def handed_over(self):
        self.launch()
        self.fc.mode = self.fc.heartbeat = "STABILIZE"
        # Pilot flipped to STABILIZE: his own flight, every channel back.
        self.tick(gate_ok=False, gate_why="pilot switched to STABILIZE")
        self.tick(gate_ok=False, gate_why="pilot switched to STABILIZE")
        self.assertEqual(self.m.phase, "WASP_DONE")
        self.assertEqual(self.out, {})

    def test_abort_in_the_air_after_a_handover_lands_it(self):
        self.handed_over()
        self.m.request_abort()
        self.tick(gate_ok=False)
        self.assertEqual(self.m.phase, "SAFEDOWN")
        self.assertEqual(self.mapper.requests, ["ALT_HOLD"])
        self.assert_land_and_disarm()

    def test_switch_up_after_a_handover_starts_it_again(self):
        self.handed_over()
        self.fc.mode = self.fc.heartbeat = "ALT_HOLD"
        self.tick(gate_ok=False, pilot_safedown=True)
        self.assertEqual(self.m.phase, "SAFEDOWN")
        self.assertIn("pilot switched to ALT_HOLD", self.m.event)

    def test_safedown_during_the_countdown_cancels_it(self):
        self.m.wasp.params["launch_s"] = 3.0
        self.assertEqual(self.m.request_engage(), "")
        self.tick()
        self.assertIsNotNone(self.m.wasp.launch_t)
        self.m.request_abort()
        self.tick()
        self.assertIsNone(self.m.wasp.launch_t)
        self.assertFalse(self.r.active)
        self.run_for(3.5)
        self.assertIsNone(self.m.wasp.started, "never launched")

    def test_nothing_starts_on_the_ground(self):
        self.handed_over()
        self.m.request_abort()
        self.tick(gate_ok=False, airborne=False)
        self.tick(gate_ok=False, airborne=False, pilot_safedown=True)
        self.assertEqual(self.m.phase, "WASP_DONE")
        self.assertFalse(self.r.active)


if __name__ == "__main__":
    unittest.main(verbosity=2)
