
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from idle import Idle
from mission import Mission, load_idle_settings
from test_wasp import Mapper, observe, WaspGuidance, backend


class IdleMapper(Mapper):
    def idle_pwm(self, yaw, climb):
        return dict(roll=1500, pitch=1500, yaw=yaw, throttle=climb)

    def idle_request_alt_hold(self):
        return True

    def pilot_pwm(self, axis):
        return 1500

    def pwm_to_phys(self, axis, value):
        return 0

    def phys_to_pwm(self, axis, value):
        return 1500 + int(value)


def telemetry(mode="ALT_HOLD", height=8, pitch=0, yaw=0):
    return dict(gate_ok=True, fc_mode=mode, idle_ok=True, height=height, climb=0,
                att=dict(roll=0, pitch=pitch, yaw=yaw), ai_cmd=None,
                det_t=None, ai_t=None, det_conf=None)


class IdleTests(unittest.TestCase):
    def test_level_first_height_and_wrapped_heading_closed_loop(self):
        idle, mp, ctx = Idle(), IdleMapper(), telemetry(pitch=math.pi/2, yaw=math.radians(179))
        idle.begin(100, ctx, 30)
        self.assertEqual(idle.height, 9.5)
        for i in range(1, 51):
            out = idle.output(mp, 100+i*.02, ctx)
            self.assertEqual(out["throttle"], 0, "level before asking to climb")
            self.assertEqual(out["roll"], 1500)
            self.assertEqual(out["pitch"], 1500)
        ctx["att"].update(pitch=0, yaw=math.radians(-179))
        out = idle.output(mp, 101.02, ctx)
        self.assertLess(out["yaw"], 0, "take short direction across north")
        self.assertLessEqual(abs(out["yaw"]), 15)

        for i in range(2, 1201):
            out = idle.output(mp, 101+i*.02, ctx)
            self.assertLessEqual(abs(out["throttle"]), 1)
            ctx["height"] += out["throttle"]*.02
            ctx["climb"] = out["throttle"]
            ctx["att"]["yaw"] += math.radians(out["yaw"])*.02
        self.assertAlmostEqual(ctx["height"], 9.5, places=3)
        self.assertEqual(idle.stage, "HOLD")
        self.assertFalse(idle.snapshot()["horizontal_hold"])
        self.assertTrue(idle.active, "no timeout during Idle hold")

    def test_ceiling_does_not_request_extra_climb_or_forced_descent(self):
        for h, expected in ((29, 30), (31, 31)):
            idle = Idle(); idle.begin(100, telemetry(height=h), 30)
            self.assertEqual(idle.height, expected)

    def test_acro_mode_confirmation_and_faults(self):
        ctx = telemetry("ACRO")
        idle = Idle(); idle.begin(100, ctx, 30)
        out = idle.output(IdleMapper(), 101, ctx)
        self.assertEqual(out["throttle"], 0)
        with self.assertRaisesRegex(ValueError, "not confirmed"):
            idle.output(IdleMapper(), 102, ctx)
        with self.assertRaisesRegex(ValueError, "rejected"):
            idle.output(IdleMapper(), 101, dict(ctx, idle_mode_denied=True))
        ctx["fc_mode"] = "ALT_HOLD"
        idle.output(IdleMapper(), 101, ctx)
        self.assertTrue(idle.confirmed)
        with self.assertRaisesRegex(ValueError, "pilot changed"):
            idle.output(IdleMapper(), 101.1, dict(ctx, fc_mode="ACRO"))
        for change in (dict(idle_ok=False), dict(att=None), dict(height=float("nan")), dict(climb=None)):
            with self.assertRaises(ValueError):
                idle.output(IdleMapper(), 101.1, dict(ctx, **change))

    def test_pwm_uses_alt_hold_midpoint_with_low_throttle_trim(self):
        b = backend(); b.mode_name = "ALT_HOLD"
        b.rc["throttle"].update(min=988, trim=990, max=2010, dz=30)
        b.pilot_yaw_rate = math.radians(202.5)
        b.max_climb_mps, b.max_descent_mps = 25, 4
        out = b.idle_pwm(0, 0)
        self.assertEqual(out["throttle"], 1499)
        self.assertEqual(out["roll"], 1500)
        self.assertGreater(b.idle_pwm(0, 1)["throttle"], 1499)
        self.assertLess(b.idle_pwm(0, -1)["throttle"], 1499)
        b.rc["throttle"]["reversed"] = True
        self.assertLess(b.idle_pwm(0, 1)["throttle"], 1499)

    def test_idle_gate_requires_calibration_fresh_height_and_heartbeat(self):
        from time import perf_counter
        b = backend(); b.mode_name = "ALT_HOLD"
        b._vfr = (perf_counter(), 8, 0, 0)
        self.assertFalse(b.idle_ready())
        b.rc["roll"]["dz"] = 400
        self.assertIn("calibration", b.idle_ready())
        b.rc["roll"]["dz"] = 20
        b._vfr = (perf_counter()-1, 8, 0, 0)
        self.assertIn("height", b.idle_ready())
        b._vfr = (perf_counter(), 8, 0, 0)
        b._heartbeat_t = perf_counter()-2
        self.assertIn("heartbeat", b.idle_ready())

    def mission(self, wasp=False):
        mp = IdleMapper(); m = Mission(mp, log=lambda *_: None)
        if wasp:
            self.assertFalse(m.set_wasp_enabled(True)); self.assertFalse(m.confirm_mount(90))
        else:
            self.assertFalse(m.confirm_mount(30))
        mp.armed = True
        ctx = telemetry("ACRO" if wasp else "ALT_HOLD")
        m.cycle(100, ctx)
        return m, mp, ctx

    def test_wasp_gaps_reacquisition_and_hard_deadline(self):
        m, mp, ctx = self.mission(True)
        self.assertFalse(m.request_engage()); m.cycle(101, ctx)
        for now, seen in ((103, True), (104, False), (105, True), (109, False), (110.99, True)):
            ctx.update(ai_cmd=observe(WaspGuidance(), x=50, t=now) if seen else None,
                       det_t=now if seen else None, det_conf=.9 if seen else None)
            out = m.cycle(now, ctx)
            self.assertEqual(m.phase, "WASP_FLIGHT")
            self.assertEqual(m._run_t, 101)
            self.assertEqual(out["throttle"], 2000)
            if not seen:
                self.assertEqual(out["roll"], 1500)
                self.assertIsNone(m.wasp.acquired_t)
        ctx.update(ai_cmd=observe(WaspGuidance(), t=111), det_t=111, det_conf=.9)
        m.cycle(111, ctx)
        self.assertEqual(m.phase, "IDLE", "live detection cannot extend deadline")
        self.assertEqual(m.idle.started, 111)
        ctx["fc_mode"] = "ALT_HOLD"; m.cycle(111.1, ctx)
        original = (m.idle.started, m.idle.height, m.idle.heading)
        m.request_abort(); m.cycle(111.2, ctx)
        self.assertEqual((m.idle.started, m.idle.height, m.idle.heading), original)
        m.cycle(1000, ctx); self.assertTrue(m.idle.active)
        self.assertTrue(m.request_engage())
        ctx.update(gate_ok=False, fc_mode="STABILIZE")
        self.assertEqual(m.cycle(1001, ctx), {})
        self.assertFalse(m.idle.active)

    def test_wasp_above_ceiling_enters_idle_only_at_configured_deadline(self):
        for detected in (False, True):
            with self.subTest(detected=detected):
                m, mp, ctx = self.mission(True)
                mp.armed = False
                self.assertEqual(m.set_idle_seconds(5, save=False), "")
                mp.armed = True
                self.assertEqual(m.request_engage(), "")
                m.cycle(101, ctx)
                for offset in (.5, 1, 1.25, 1.5, 3, 4.99):
                    now = 101 + offset
                    ctx.update(height=m.ceiling_m + 20*offset,
                               ai_cmd=observe(WaspGuidance(), x=50, t=now) if detected else None,
                               det_t=now if detected else None,
                               det_conf=.9 if detected else None)
                    out = m.cycle(now, ctx)
                    self.assertEqual(m.phase, "WASP_FLIGHT")
                    self.assertFalse(m.idle.active)
                    self.assertEqual(out["throttle"], 2000)
                    self.assertEqual(m._run_t, 101)
                    self.assertAlmostEqual(m.snapshot(now)["idle_remaining_s"], 5-offset)
                    if detected and offset >= 1.5:
                        self.assertEqual(m.wasp.progress(now)["stage"], "GUIDE")
                ctx.update(ai_cmd=observe(WaspGuidance(), t=106) if detected else None,
                           det_t=106 if detected else None)
                m.cycle(106, ctx)
                self.assertEqual(m.phase, "IDLE")
                self.assertEqual(m.idle.started, 106)
                self.assertIn("Idle time limit", m.event)
                self.assertEqual(m.idle.height, ctx["height"])

    def test_seeker_engage_and_auto_use_total_command_start(self):
        for auto in (False, True):
            m, mp, ctx = self.mission()
            if auto:
                m.set_auto_enabled(True); self.assertFalse(m.request_auto(True))
            else:
                self.assertFalse(m.request_engage())
            m.cycle(101, ctx)
            self.assertEqual(m._run_t, 101)
            if auto:

                self.assertFalse(m.request_engage()); m.cycle(105, ctx)
                self.assertEqual(m._run_t, 101)
            m.cycle(110.99, ctx); self.assertFalse(m.idle.active)
            m.cycle(111, ctx); self.assertEqual(m.phase, "IDLE")

    def test_abort_and_seeker_loss_enter_idle(self):
        for wasp in (False, True):
            m, mp, ctx = self.mission(wasp)
            m.request_engage(); m.cycle(101, ctx)
            m.request_abort(); out = m.cycle(101.1, ctx)
            self.assertEqual(m.phase, "IDLE")
            self.assertEqual(out["roll"], 1500)
        m, mp, ctx = self.mission()
        m.phase = "AI"; m._run_t = 101
        m.cycle(102, ctx); self.assertEqual(m.phase, "IDLE")

    def test_abort_before_ground_launch_never_commands_takeoff(self):
        m, mp, ctx = self.mission(True)
        ctx.update(height=0, airborne=False)
        m.request_engage(); m.request_abort()
        self.assertEqual(m.cycle(101, ctx), {})
        self.assertFalse(m.idle.active)
        self.assertIsNone(m.wasp.started)

    def test_idle_persists_custom_value_and_rejects_bad_or_armed_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"idle.json")
            m = Mission(IdleMapper(), idle_path=path, log=lambda *_: None)
            self.assertEqual(m.idle_s, 10)
            self.assertFalse(m.set_idle_seconds("6.5"))
            other = Mission(IdleMapper(), idle_path=path, log=lambda *_: None)
            other.mapper.armed = True
            self.assertFalse(load_idle_settings(other)); self.assertEqual(other.idle_s, 6.5)
            self.assertFalse(other.idle.active)
            for value in (0, 601, "", "nan", "inf", "invalid"):
                self.assertTrue(m.set_idle_seconds(value)); self.assertEqual(m.idle_s, 6.5)
            with patch("mission._write_json", return_value=False):
                self.assertTrue(m.set_idle_seconds(12)); self.assertEqual(m.idle_s, 6.5)
            m.mapper.armed = True
            self.assertTrue(m.set_idle_seconds(12)); self.assertEqual(m.idle_s, 6.5)
            self.assertNotIn("duration", m.wasp.params)
            self.assertNotIn("no_det", m.wasp.params)
            self.assertNotIn("search", m.level)


if __name__ == "__main__":
    unittest.main(verbosity=2)
