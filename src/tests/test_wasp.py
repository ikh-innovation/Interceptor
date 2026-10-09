"""Run with python3 src/tests/test_wasp.py. No FC, camera or model required."""
import math
import os
import sys
import unittest
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wasp import Wasp, WaspGuidance, camera_ray, rotation, rotate, unit, cross, interpolate_rotation
from mission import Mission, phone_command
from fc_control import ArduPilotAcro, AcroCommander, InterceptLaw

TH, TV = math.tan(math.radians(31.1)), math.tan(math.radians(24.4))
ATT = dict(roll=0.0, pitch=0.0, yaw=0.0)


def observe(g, x=0, y=0, att=None, t=100.0, camera_roll=0):
    c = g.observe(x, y, 640, 480, TH, TV, att or ATT, t, camera_roll)
    c["camera_roll"] = camera_roll
    return c


def project(world, att, camera_roll=0):
    x, y, z = rotate(rotation(att), world, inverse=True)
    right, down = y/-z, x/-z
    angle = math.radians(camera_roll)
    return ((right*math.cos(angle)+down*math.sin(angle))/TH*320,
            (-right*math.sin(angle)+down*math.cos(angle))/TV*240)


def backend():
    b = ArduPilotAcro("unused", axes=("roll", "pitch", "yaw", "throttle"), verbose=False)
    b.linked = b.armed = True
    b.mode_name = "ACRO"
    b.params = dict(ACRO_TRAINER=0, ACRO_RP_RATE=360, ACRO_Y_RATE=202.5,
                    ACRO_RP_EXPO=0.3, ACRO_Y_EXPO=0.2, RC_OVERRIDE_TIME=1)
    for ch, axis in enumerate(b.axes, 1):
        b.rc[axis] = dict(chan=ch, min=1100, trim=1500, max=1900, dz=20,
                          reversed=False, from_fc=True, wasp_calibrated=True)
    now = perf_counter()
    b._heartbeat_t = now
    b._att_hist = [(now-.04, 0, 0, 0, 0, 0, 0), (now, 0, 0, 0, 0, 0, 0)]
    b._attitude_t = now
    b.sent = []
    b._send_override = lambda chans: b.sent.append(chans) or True
    return b


class GuidanceTests(unittest.TestCase):
    def test_signs_and_full_attitude_at_each_platform_angle(self):
        for tilt in (0, 20, 40, 60, 90):
            for yaw in (0, 89, 179, -179):
                att = dict(roll=0.2, pitch=-math.radians(tilt), yaw=math.radians(yaw))
                for x, y in ((0, 0), (50, 0), (-50, 0), (0, 50), (0, -50)):
                    c = observe(WaspGuidance(), x, y, att)
                    r, p, z = WaspGuidance.rates(c, att, 100, lead_s=0)
                    self.assertAlmostEqual(z, 0)
                    if x:
                        self.assertGreater(r*x, 0)
                    else:
                        self.assertAlmostEqual(r, 0)
                    if y:
                        self.assertLess(p*y, 0)
                    else:
                        self.assertAlmostEqual(p, 0)

    def test_ego_rotation_is_not_target_velocity(self):
        world = unit((0.15, 0.1, -1))
        for camera_roll in (0, 90, 180, 270):
            g = WaspGuidance()
            # Changing motion on all three axes, with a rigid mounted camera.
            for i in range(60):
                att = dict(roll=.15*math.sin(i*.13), pitch=.12*math.cos(i*.09),
                           yaw=.6*math.sin(i*.07))
                x, y = project(world, att, camera_roll)
                c = observe(g, x, y, att, 100+i*.02, camera_roll)
                for a, b in zip(c["bearing"], world):
                    self.assertAlmostEqual(a, b, places=10)
                self.assertLess(math.sqrt(sum(v*v for v in c["bearing_rate"])), 1e-10)

    def test_capture_latency_reprojects_into_current_body(self):
        c = observe(WaspGuidance())
        now_att = dict(roll=0.1, pitch=-0.1, yaw=1)
        r, p, _ = WaspGuidance.rates(c, now_att, 100.05)
        self.assertLess(r, 0)
        self.assertGreater(p, 0)

    def test_moving_target_lead_and_rate_cap(self):
        g = WaspGuidance()
        observe(g, x=0)
        c = observe(g, x=25, t=100.03)
        a = WaspGuidance.rates(c, ATT, 100.03, lead_s=0, max_rate_dps=180)
        b = WaspGuidance.rates(c, ATT, 100.03, lead_s=.15, max_rate_dps=180)
        self.assertGreater(b[0], a[0])
        bounded = WaspGuidance.rates(c, ATT, 100.03, gain=100, max_rate_dps=30)
        self.assertAlmostEqual(math.hypot(*bounded[:2]), math.radians(30))

    def test_camera_roll_and_crop(self):
        for angle in (0, 90, 180, 270):
            a = camera_ray(70, -35, 640, 480, TH, TV, angle)
            b = camera_ray(70, -35, 320, 240, TH/2, TV/2, angle)
            self.assertEqual(a, b)
        c = observe(WaspGuidance(), x=50, camera_roll=90)
        r, p, _ = WaspGuidance.rates(c, ATT, 100)
        self.assertAlmostEqual(r, 0)
        self.assertLess(p, 0)

    def test_wrap_and_vertical_slerp(self):
        # These two Euler representations describe the same vertical attitude.
        a, b = (0, math.pi/2, 0), (math.pi, math.pi/2, math.pi)
        mat = interpolate_rotation(a, b, .5)
        for actual, expected in zip(rotate(mat, (0, 0, -1)), (-1, 0, 0)):
            self.assertAlmostEqual(actual, expected)
        mat = interpolate_rotation((0, 0, math.radians(179)), (0, 0, math.radians(-179)), .5)
        self.assertAlmostEqual(rotate(mat, (1, 0, 0))[0], -1)

    def test_closed_loop_converges_across_platform_angles(self):
        for tilt in (0, 20, 40, 60, 90):
            start = rotation(dict(roll=.15, pitch=-math.radians(tilt), yaw=.7))
            target = rotate(start, unit((.13, .12, -1)))
            mat = start
            g = WaspGuidance()
            # The gentler default gain of 2 gets two simulated seconds to settle.
            for i in range(200):
                att = {"rotation": mat}
                x, y = project(target, att)
                c = observe(g, x, y, att, 100+i*.01)
                omega = WaspGuidance.rates(c, att, 100+i*.01)
                # Integrate the commanded body rates with Rodrigues' formula.
                angle = math.sqrt(sum(v*v for v in omega))*.01
                if angle > 1e-12:
                    axis = unit(omega)
                    def turn(v):
                        dot = sum(a*b for a, b in zip(axis, v))
                        return tuple(v[j]*math.cos(angle)+cross(axis, v)[j]*math.sin(angle)
                                     + axis[j]*dot*(1-math.cos(angle)) for j in range(3))
                    cols = [rotate(mat, turn(v)) for v in ((1,0,0), (0,1,0), (0,0,1))]
                    mat = tuple(zip(*cols))
            direction = rotate(mat, (0, 0, -1))
            self.assertLess(math.acos(min(1, sum(a*b for a,b in zip(direction,target)))), .01)

    def test_stale_and_invalid_observations(self):
        c = observe(WaspGuidance())
        for t in (99.9, 100.26, float("nan")):
            with self.assertRaises(ValueError):
                WaspGuidance.rates(c, ATT, t)
        with self.assertRaises(ValueError):
            observe(WaspGuidance(), x=float("nan"))


class Mapper:
    armed = False
    camera_mount_deg = 90

    def wasp_pwm(self, rates, throttle):
        return dict(roll=1500+int(rates[0]*100), pitch=1500+int(rates[1]*100),
                    yaw=1500, throttle=1000+int(throttle*1000))


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.mapper = Mapper()
        self.m = Mission(self.mapper, log=lambda *_: None)
        self.assertEqual(self.m.set_wasp_enabled(True), "")
        self.assertEqual(self.m.confirm_mount(90), "")
        self.mapper.armed = True
        self.ctx = dict(gate_ok=True, att=ATT, height=0, ai_cmd=observe(WaspGuidance()),
                        det_t=100, det_conf=.9)
        self.assertEqual(self.m.cycle(100, self.ctx), {})
        self.assertEqual(self.m.phase, "WASP_READY")

    def launch(self):
        self.assertEqual(self.m.request_engage(), "")
        return self.m.cycle(100, self.ctx)

    def test_launch_has_no_hover_or_ramp(self):
        out = self.launch()
        self.assertEqual(out["throttle"], 2000)
        self.assertEqual(self.m.phase, "WASP_FLIGHT")
        self.assertEqual(set(out), {"roll", "pitch", "yaw", "throttle"})
        self.assertTrue(self.m.request_hold("up", "x", 100))
        self.assertTrue(self.m.request_lock(True))

    def test_straight_then_smooth_live_target_handover(self):
        sent = []
        self.mapper.wasp_pwm = lambda rates, throttle: sent.append((rates, throttle)) or {"throttle": throttle}
        self.launch()
        for i in range(1, 181):
            now = 100 + i*.01
            # The target moves to the opposite side before guidance starts.
            x = 50 if i < 90 else -50
            cmd = observe(WaspGuidance(), x=x, t=now)
            self.ctx.update(ai_cmd=cmd, det_t=now)
            self.m.cycle(now, self.ctx)
            rates, throttle = sent[-1]
            self.assertEqual(throttle, 1)
            if i <= 100:
                self.assertEqual(rates, (0, 0, 0))
            else:
                self.assertLess(rates[0], 0)
                full = WaspGuidance.rates(cmd, ATT, now, max_rate_dps=60)
                blend = self.m.wasp.progress(now)["steering_blend"]
                self.assertAlmostEqual(rates[0], full[0]*blend)
            change = math.dist(rates, sent[-2][0])
            self.assertLessEqual(change, 2*math.radians(60)/.5*.01+1e-10)
        self.assertEqual(self.m.wasp.progress(100.5)["stage"], "STRAIGHT")
        self.assertEqual(self.m.wasp.progress(101.25)["stage"], "BLEND")
        self.assertEqual(self.m.wasp.progress(101.5)["stage"], "GUIDE")

    def test_guidance_changes_slew_limited_and_new_launch_resets(self):
        w = self.m.wasp
        self.mapper.wasp_pwm = lambda rates, throttle: rates
        w.begin(100)
        previous = (0, 0, 0)
        for i in range(181):
            now = 100+i*.01
            self.ctx.update(ai_cmd=observe(WaspGuidance(), x=200 if i < 170 else -200, t=now))
            rates = w.output(self.mapper, now, self.ctx)
            self.assertLessEqual(math.dist(previous, rates), 2*math.radians(60)/.5*.01+1e-10)
            previous = rates
        w.begin(102)
        self.ctx.update(ai_cmd=observe(WaspGuidance(), x=200, t=102))
        self.assertEqual(w.output(self.mapper, 102, self.ctx), (0, 0, 0))

    def test_abort_and_target_loss_in_each_stage(self):
        for offset in (.5, 1.25, 1.75):
            for abort in (False, True):
                self.setUp()
                self.launch()
                now = 100+offset
                self.ctx.update(ai_cmd=observe(WaspGuidance(), x=50, t=now), det_t=now)
                self.assertTrue(self.m.cycle(now, self.ctx))
                if abort:
                    self.m.request_abort()
                else:
                    self.ctx["det_t"] = now-.26
                out = self.m.cycle(now+.01, self.ctx)
                self.assertEqual(self.m.phase, "WASP_DONE" if abort else "WASP_FLIGHT")
                self.assertEqual(out, {} if abort else self.mapper.wasp_pwm((0, 0, 0), 1))

    def test_launch_throttle_choices_and_timing_validation(self):
        for throttle in (.25, .5, 1):
            self.setUp()
            self.mapper.armed = False
            self.assertEqual(self.m.set_wasp_param("throttle", throttle), "")
            self.mapper.armed = True
            out = self.launch()
            self.assertEqual(out["throttle"], 1000+int(throttle*1000))
        self.setUp()
        self.mapper.armed = False
        self.assertTrue(self.m.set_wasp_param("throttle", .75))
        self.assertTrue(self.m.set_wasp_param("duration", 1))
        self.assertNotIn("duration", self.m.wasp.params)
        self.assertTrue(self.m.set_wasp_param("no_det", 10))
        self.assertEqual(self.m.set_wasp_param("straight", 0), "")
        self.assertEqual(self.m.set_wasp_param("blend", 1), "")
        self.assertEqual(self.m.set_wasp_param("straight", 2), "")
        self.assertEqual(self.m.set_wasp_param("blend", 1), "")

    def test_45_degree_choice_limits_combined_steering(self):
        self.mapper.armed = False
        self.assertEqual(self.m.set_wasp_param("rate", 45), "")
        self.mapper.armed = True
        self.mapper.wasp_pwm = lambda rates, throttle: rates
        self.launch()
        for i in range(1, 201):
            now = 100+i*.01
            self.ctx.update(ai_cmd=observe(WaspGuidance(), x=1000, y=700, t=now), det_t=now)
            rates = self.m.cycle(now, self.ctx)
            self.assertLessEqual(math.hypot(*rates[:2]), math.radians(45)+1e-10)
        self.assertAlmostEqual(math.hypot(*rates[:2]), math.radians(45))

    def test_every_timing_choice_and_zero_delay(self):
        self.mapper.wasp_pwm = lambda rates, throttle: rates
        for straight in Wasp.SETTINGS["straight"]:
            for blend in Wasp.SETTINGS["blend"]:
                w = Wasp()
                w.params.update(straight=straight, blend=blend)
                w.begin(100)
                for i in range(round((straight+blend+.1)*100)+1):
                    now = 100+i*.01
                    self.ctx.update(ai_cmd=observe(WaspGuidance(), x=50, t=now))
                    rates = w.output(self.mapper, now, self.ctx)
                    if i*.01 <= straight:
                        self.assertEqual(rates, (0, 0, 0))
                self.assertGreater(rates[0], 0)
                self.assertEqual(w.progress(now)["stage"], "GUIDE")

    def test_mutual_exclusion_is_enforced_in_backend(self):
        self.assertTrue(self.m.set_auto_enabled(True))
        self.assertTrue(self.m.request_auto(True))
        m = Mission(Mapper(), log=lambda *_: None)
        m.set_auto_enabled(True)
        self.assertTrue(m.set_wasp_enabled(True))
        self.assertFalse(m.wasp.enabled)

    def test_mode_chooser_is_atomic_and_refresh_is_non_mutating(self):
        m = Mission(Mapper(), log=lambda *_: None)
        m.set_auto_enabled(True)
        self.assertEqual(phone_command(m, "flightmode", "wasp", 100), "")
        self.assertFalse(m.auto_enabled)
        self.assertTrue(m.wasp.enabled)
        m.confirm_mount(90)
        m.mapper.armed = True
        m.phase = "WASP_FLIGHT"
        m.wasp.started = 100
        m.wasp.used = True
        self.assertEqual(phone_command(m, "flightmode", "wasp", 100), "")
        self.assertEqual(m.wasp.started, 100)
        self.assertTrue(m.wasp.camera_confirmed)
        self.assertTrue(phone_command(m, "flightmode", "normal", 100))
        self.assertTrue(phone_command(m, "flightmode", "typo", 100))

    def test_mount_and_parameters_frozen_while_armed(self):
        for key, value in (("waspen", "off"), ("mount", "30"), ("waspp", "throttle:0.5")):
            self.assertTrue(phone_command(self.m, key, value, 100))
        for value in ("nan", "inf", "0.99"):
            self.assertTrue(self.m.set_wasp_param("throttle", value))

    def test_abort_wins_over_launch(self):
        self.m.request_engage()
        self.m.request_abort()
        self.assertEqual(self.m.cycle(100, self.ctx), {})
        self.assertIsNone(self.m.wasp.started)

    def test_detection_gap_continues_straight_and_reacquires(self):
        self.launch()
        self.assertEqual(self.m.cycle(100.26, self.ctx), self.mapper.wasp_pwm((0, 0, 0), 1))
        self.assertEqual(self.m.phase, "WASP_FLIGHT")
        self.assertTrue(self.m.request_engage())
        self.ctx.update(ai_cmd=observe(WaspGuidance(), t=100.3), det_t=100.3)
        self.assertTrue(self.m.cycle(100.3, self.ctx))
        self.assertEqual(self.m.wasp.acquired_t, 100.3)
        self.assertEqual(self.m._run_t, 100)

    def test_gate_loss_and_abort_release_all_channels(self):
        self.launch()
        self.ctx["gate_ok"] = False
        self.assertEqual(self.m.cycle(100.01, self.ctx), {})
        self.ctx["gate_ok"] = True
        self.assertEqual(self.m.cycle(100.02, self.ctx), {})
        self.mapper.armed = False
        self.ctx["gate_ok"] = False
        self.m.cycle(100.03, self.ctx)
        self.assertFalse(self.m.wasp.used)

    def test_pilot_takeover_resets_idle_clock(self):
        self.mapper.armed = False
        self.assertEqual(self.m.set_idle_seconds(160, save=False), "")
        self.mapper.armed = True
        self.launch()
        self.assertEqual(self.m.snapshot(150)["idle_remaining_s"], 110)
        self.ctx["gate_ok"] = False
        self.m.cycle(150, self.ctx)
        self.ctx["gate_ok"] = True
        self.m.cycle(155, self.ctx)
        self.assertIsNone(self.m._run_t)
        self.assertIsNone(self.m.snapshot(155)["idle_remaining_s"])

    def test_timeout_releases_with_live_detection(self):
        self.launch()
        for now in (100.5, 101.25, 101.75, 102, 105, 109.99):
            self.ctx.update(ai_cmd=observe(WaspGuidance(), t=now), det_t=now)
            self.assertTrue(self.m.cycle(now, self.ctx))
            self.assertEqual(self.m.phase, "WASP_FLIGHT")
        self.ctx.update(ai_cmd=observe(WaspGuidance(), t=110), det_t=110)
        self.assertEqual(self.m.cycle(110, self.ctx), {})
        self.assertIn("time limit", self.m.event)

    def test_launch_requires_attitude_and_confirmation(self):
        self.m.request_engage()
        self.assertEqual(self.m.cycle(100, dict(self.ctx, att=None)), {})
        self.assertFalse(self.m.wasp.used)
        self.m.wasp.camera_confirmed = False
        self.assertEqual(self.launch(), {})

    def test_no_detection_choices_hold_zero_rates_then_release_and_latch(self):
        for limit in (3, 5, 7, 10):
            for tilt in (0, 45, 90):
                with self.subTest(limit=limit, tilt=tilt):
                    self.setUp()
                    self.mapper.armed = False
                    self.assertEqual(self.m.set_idle_seconds(limit), "")
                    self.mapper.armed = True
                    self.ctx.update(ai_cmd=None, det_t=None, det_conf=None,
                                    att=dict(roll=.2, pitch=math.radians(tilt), yaw=1))
                    sent = []
                    self.mapper.wasp_pwm = lambda rates, throttle: sent.append((rates, throttle)) or {"throttle": throttle}
                    self.assertTrue(self.launch())
                    for offset in (0, .5, 1.25, limit-.01):
                        self.assertTrue(self.m.cycle(100+offset, self.ctx))
                        self.assertEqual(sent[-1], ((0, 0, 0), 1))
                        progress = self.m.wasp.progress(100+offset)
                        self.assertEqual(progress["stage"], "SEARCH")
                        self.assertAlmostEqual(self.m.snapshot(100+offset)["idle_remaining_s"], limit-offset)
                    self.assertEqual(self.m.cycle(100+limit, self.ctx), {})
                    self.assertEqual(self.m.phase, "WASP_DONE")
                    self.assertTrue(self.m.request_engage())
                    self.ctx.update(ai_cmd=observe(WaspGuidance(), t=100+limit+.01),
                                    det_t=100+limit+.01, det_conf=.9)
                    self.assertEqual(self.m.cycle(100+limit+.01, self.ctx), {})

    def test_first_detection_starts_blend_after_remaining_straight_launch(self):
        for acquired in (.5, 2, 2.99):
            with self.subTest(acquired=acquired):
                self.setUp()
                self.ctx.update(ai_cmd=None, det_t=None, det_conf=None)
                self.mapper.wasp_pwm = lambda rates, throttle: {"rates": rates, "throttle": throttle}
                self.launch()
                now = 100+acquired
                self.ctx.update(ai_cmd=observe(WaspGuidance(), x=80, t=now), det_t=now, det_conf=.9)
                out = self.m.cycle(now, self.ctx)
                self.assertEqual(out["rates"], (0, 0, 0))
                self.assertEqual(self.m.wasp.acquired_t, now)
                blend_start = 100+max(1, acquired)
                for i in range(1, 61):
                    now = blend_start+i*.01
                    self.ctx.update(ai_cmd=observe(WaspGuidance(), x=80, t=now), det_t=now)
                    out = self.m.cycle(now, self.ctx)
                    full = WaspGuidance.rates(self.ctx["ai_cmd"], ATT, now, max_rate_dps=60)
                    blend = self.m.wasp.progress(now)["steering_blend"]
                    self.assertAlmostEqual(out["rates"][0], full[0]*blend)
                    self.assertEqual(out["throttle"], 1)
                self.assertEqual(self.m.wasp.progress(now)["stage"], "GUIDE")
                # Acquisition timeout no longer applies; total launch time still does.
                self.ctx.update(ai_cmd=observe(WaspGuidance(), t=109.99), det_t=109.99)
                self.assertTrue(self.m.cycle(109.99, self.ctx))
                self.ctx.update(ai_cmd=observe(WaspGuidance(), t=110), det_t=110)
                self.assertEqual(self.m.cycle(110, self.ctx), {})

    def test_invalid_initial_detections_do_not_acquire_or_steer(self):
        cmd = observe(WaspGuidance(), x=80)
        cases = [dict(det_conf=.01), dict(det_conf=float("nan")), dict(det_t=99.74),
                 dict(det_t=100.1), dict(ai_cmd={}), dict(ai_cmd=dict(cmd, camera_roll=90)),
                 dict(ai_cmd=dict(cmd, capture_t=99.74)),
                 dict(ai_cmd=dict(cmd, bearing=(0, 0, 1))),
                 dict(ai_cmd=dict(cmd, bearing=(float("nan"), 0, -1)))]
        for values in cases:
            with self.subTest(values=values):
                self.setUp()
                self.ctx.update(values)
                self.assertTrue(self.launch())
                self.assertIsNone(self.m.wasp.acquired_t)
                self.assertEqual(self.m.wasp.progress(100)["stage"], "SEARCH")
                now = 100.1
                self.ctx.update(ai_cmd=observe(WaspGuidance(), t=now), det_t=now, det_conf=.9)
                self.assertTrue(self.m.cycle(now, self.ctx))
                self.assertEqual(self.m.wasp.acquired_t, now)

    def test_acquisition_abort_gate_attitude_and_ceiling_release(self):
        for failure in ("abort", "gate", "attitude", "ceiling"):
            self.setUp()
            self.ctx.update(ai_cmd=None, det_t=None, det_conf=None)
            self.assertTrue(self.launch())
            if failure == "abort":
                self.m.request_abort()
            elif failure == "gate":
                self.ctx["gate_ok"] = False
            elif failure == "attitude":
                self.ctx["att"] = None
            else:
                self.ctx["height"] = self.m.ceiling_m
            self.assertEqual(self.m.cycle(100.01, self.ctx), {})
            self.assertEqual(self.m.phase, "WASP_DONE")

    def test_no_detection_deadline_wins_over_new_detection(self):
        self.ctx.update(ai_cmd=None, det_t=None, det_conf=None)
        self.launch()
        self.ctx.update(ai_cmd=observe(WaspGuidance(), t=110), det_t=110, det_conf=.9)
        self.assertEqual(self.m.cycle(110, self.ctx), {})
        self.assertIsNone(self.m.wasp.acquired_t)
        self.assertIn("Idle time limit", self.m.event)

    def test_total_flight_limit_bounds_acquisition_and_new_parameter_validation(self):
        self.mapper.armed = False
        self.assertEqual(self.m.set_wasp_param("straight", 0), "")
        self.assertEqual(self.m.set_idle_seconds(1), "")
        for value in (0, 601, float("nan"), float("inf")):
            self.assertTrue(self.m.set_idle_seconds(value))
        self.mapper.armed = True
        self.assertTrue(self.m.set_idle_seconds(3))
        self.ctx.update(ai_cmd=None, det_t=None, det_conf=None)
        self.launch()
        self.assertEqual(self.m.snapshot(100)["idle_remaining_s"], 1)
        self.assertTrue(self.m.cycle(100.99, self.ctx))
        self.assertEqual(self.m.cycle(101, self.ctx), {})
        self.assertIn("Idle time limit", self.m.event)


class BackendTests(unittest.TestCase):
    def test_throttle_mapping_does_not_change_existing_acro_behaviour(self):
        b = backend()
        out = b.wasp_pwm((0, 0, 0), 1)
        self.assertEqual(out["throttle"], 1900)
        b.steer_pwm(out)
        self.assertEqual(b.sent[-1][3], 0)  # legacy ACRO never overrides throttle
        self.assertEqual(b.wasp_ready(), "")
        b.steer_wasp_pwm(out)
        self.assertEqual(b.sent[-1][3], 1900)
        b.rc["throttle"]["reversed"] = True
        self.assertEqual(b.wasp_pwm((0, 0, 0), 1)["throttle"], 1100)
        b.mode_name = "ALT_HOLD"
        b.steer_wasp_pwm(out)
        self.assertEqual(b.sent[-1], [0]*8)

    def test_expo_is_inverted_with_rc_deadzone_and_reversal(self):
        b = backend()
        for expo in (0, .3, 1):
            b.params["ACRO_RP_EXPO"] = expo
            for reversed_ in (False, True):
                b.rc["roll"]["reversed"] = reversed_
                pwm = b.wasp_pwm((math.radians(30), 0, 0), 1)["roll"]
                n = b._exact_norm("roll", pwm)
                actual = 360*((1-expo)*n+expo*n**3)
                self.assertAlmostEqual(actual, 30, delta=1.5)

    def test_gate_rejects_missing_calibration_stale_telemetry_and_trainer(self):
        b = backend()
        for key, bad in (("ACRO_TRAINER", 1), ("RC_OVERRIDE_TIME", -1),
                         ("ACRO_RP_EXPO", None), ("ACRO_RP_RATE", float("nan"))):
            old = b.params[key]
            b.params[key] = bad
            self.assertTrue(b.wasp_ready())
            b.params[key] = old
        b._heartbeat_t -= 2
        self.assertTrue(b.wasp_ready())
        b._heartbeat_t += 2
        b.rc["pitch"]["wasp_calibrated"] = False
        self.assertTrue(b.wasp_ready())

    def test_attitude_history_bounds_and_slerp(self):
        b = backend()
        now = perf_counter()
        b._att_hist = [(now-.04, 0, math.pi/2, 0, 0, 0, 0),
                       (now, math.pi, math.pi/2, math.pi, 0, 0, 0)]
        a = b.wasp_attitude(now-.02)
        self.assertAlmostEqual(rotate(rotation(a), (0,0,-1))[0], -1)
        self.assertIsNone(b.wasp_attitude(now-1))
        self.assertIsNone(b.wasp_attitude(now+.1))
        with patch("fc_control.perf_counter", return_value=now+.2):
            self.assertIsNone(b.wasp_attitude())
        b._att_hist[0] = (now-.04, float("nan"), 0, 0, 0, 0, 0)
        self.assertIsNone(b.wasp_attitude(now-.02))

    def test_commander_mode_and_mount_gates(self):
        b = backend()
        c = AcroCommander(law=InterceptLaw(), mission=True)
        c.backend = b
        c.mission = Mission(c, log=lambda *_: None)
        c.mission.camera_mount_deg = 90
        self.assertIn("30 degree", c._mission_gate()[1])
        c.mission.wasp.enabled = True
        self.assertIn("confirm", c._mission_gate()[1])
        c.mission.wasp.camera_confirmed = True
        self.assertEqual(c._mission_gate(), (True, ""))
        # Platform launch bypasses the hover mission's airborne requirement.
        b.airborne = lambda: (False, "landed")
        self.assertEqual(c._mission_gate(), (True, ""))
        c.commit_chan = 7
        b._commit = 0
        self.assertIn("commit", c._mission_gate()[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
