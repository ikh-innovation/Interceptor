#!/usr/bin/env python3
"""ACRO as a second control mode, and the wind-rejection fix.

Two things pinned down here, both driven by a real field failure:

  1. THE ACRO-RATE TRANSLATION.  InterceptLaw always decides a physical bank
     and lean ANGLE (proportional navigation, closing-speed law).  In ALT_HOLD
     that angle IS the stick, normalised by ANGLE_MAX, and the FC's own
     ~400 Hz attitude controller holds it.  ACRO has no such controller, so an
     outer P loop here converts the SAME angle target into the RATE that would
     reach it — the "same as ALT_HOLD" the user asked for, at a bandwidth we
     are honest about being lower.  Throttle is never driven in ACRO: a stick
     there is raw thrust with nothing underneath it, so the law commands 0 and
     a second, independent guard at the backend leaves the channel with the
     pilot regardless of what axes are configured.

  2. WIND.  Flight-in-the-field drift is not a bug in the guidance law's
     tracking — it is the "calm at acquisition" bank ceiling (bank_far_deg)
     running out of authority against a real crosswind while the target is
     still far/small in frame.  Raising it (12 -> 24 deg) costs nothing in
     still air (verified: identical result at 0 m/s) and is the difference
     between a hit and a miss at 12 m/s of crosswind (measured below).

    python3 tests/test_acro_dual.py
"""
import io
import math
import os
import sys
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from fc_control import InterceptLaw, ArduPilotAcro, ANGLE_MODES   # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def law(**kw):
    base = dict(h_fov_deg=62.2, v_fov_deg=48.8, cam_tilt_deg=30.0,
                roll_pn_gain=45.0, max_bank_deg=30.0, max_lean_deg=20.0,
                acro_rp_rate=math.radians(360.0), acro_yaw_rate=math.radians(202.5),
                max_yaw_rate=math.radians(202.5), angle_max=math.radians(30.0))
    base.update(kw)
    return InterceptLaw(**base)


print("=== 1. _angle_to_stick: the outer P loop that stands in for ACRO ===")
L = law(att_kp=6.0)
# ANGLE mode: pass-through, exactly as before -- must be a bit-for-bit no-op.
check("ANGLE mode is a pure angle/angle_max pass-through, measured angle ignored",
      L._angle_to_stick(math.radians(15.0), math.radians(99.0), False)
      == L._angle_to_stick(math.radians(15.0), None, False)
      == 15.0 / 30.0)

# RATE mode: outer P loop against measured angle, clamped, then rate-normalised.
target, measured = math.radians(15.0), math.radians(5.0)
want_rate = 6.0 * math.radians(10.0)                    # att_kp * error
want_stick = want_rate / math.radians(360.0)
got = L._angle_to_stick(target, measured, True)
check("RATE mode outputs att_kp * angle_error, normalised by ACRO_RP_RATE",
      abs(got - want_stick) < 1e-9, f"{got:.4f} vs {want_stick:.4f}")

check("RATE mode clamps to the ACRO rate ceiling, not the raw P output",
      abs(L._angle_to_stick(math.radians(80.0), 0.0, True) - 1.0) < 1e-9)

check("no measured angle -> 0, not a crash or a guess",
      L._angle_to_stick(math.radians(15.0), None, True) == 0.0)

L2 = law(acro_rp_rate=None)
check("no known ACRO_RP_RATE (never connected) -> 0, not a ZeroDivisionError",
      L2._angle_to_stick(math.radians(15.0), 0.0, True) == 0.0)


print("\n=== 2. yaw scale switches with the mode, roll/pitch/climb follow suit ===")
att = dict(roll=0.0, pitch=0.0, yaw=0.0, yawspeed=0.0, pitchspeed=0.0)
L = law(kp_yaw=4.5, acro_yaw_rate=math.radians(400.0), max_yaw_rate=math.radians(200.0))
c_angle = L.compute(30.0, 0.0, 640, 640, dt=0.05, att=att, rate_mode=False)
c_rate = L.compute(30.0, 0.0, 640, 640, dt=0.05, att=att, rate_mode=True)
check("ANGLE mode yaw stick uses max_yaw_rate (PILOT_Y_RATE)",
      abs(c_angle["yaw_rate"] - c_angle["yaw_stick"] * L.max_yaw_rate) < 1e-9)
check("...and it differs from the RATE-mode stick for the SAME pixel error",
      c_angle["yaw_stick"] != c_rate["yaw_stick"],
      f"angle {c_angle['yaw_stick']:.4f}  rate {c_rate['yaw_stick']:.4f}")

print("\n=== 3. throttle is NEVER driven in ACRO ===")
L = law()
for _ in range(30):     # a few frames so climb has something to command
    cmd = L.compute(0.0, -60.0, 640, 640, dt=0.05,
                    att=dict(roll=0.0, pitch=0.0, yaw=0.0,
                             yawspeed=0.0, pitchspeed=0.0),
                    commit=1.0, rate_mode=True)
check("throttle_stick is exactly 0 in rate_mode, whatever the elevation error",
      cmd["throttle_stick"] == 0.0 and cmd["climb_mps"] == 0.0,
      f"throttle_stick={cmd['throttle_stick']} climb_mps={cmd['climb_mps']}")
cmd_angle = L.compute(0.0, -60.0, 640, 640, dt=0.05,
                     att=dict(roll=0.0, pitch=0.0, yaw=0.0,
                              yawspeed=0.0, pitchspeed=0.0),
                     commit=1.0, rate_mode=False)
check("...but ALT_HOLD still climbs on the same input (throttle is mode-gated, "
      "not broken)", cmd_angle["climb_mps"] > 0.1, f"{cmd_angle['climb_mps']:.2f} m/s")


print("\n=== 4. the backend-level guard: throttle channel untouched outside an "
      "angle mode ===")


def make_backend(mode_name):
    b = ArduPilotAcro.__new__(ArduPilotAcro)          # skip __init__ (no serial)
    b.axes = ("roll", "pitch", "yaw", "throttle")
    b.mode_name = mode_name
    b.thr_deadzone = 100
    b.rc = {
        ax: dict(chan=i + 1, min=1000, max=2000, trim=1500, reversed=False)
        for i, ax in enumerate(("roll", "pitch", "yaw", "throttle"))
    }
    b._lock = __import__("threading").Lock()
    b._sent_pwm = {}
    b._sent_hist = {}
    b._steer_since = None
    b.ECHO_HISTORY_S = 1.0
    b._send_override = lambda chans: True   # no real link to write to
    return b


cmd = dict(roll_stick=0.3, pitch_stick=-0.2, yaw_stick=0.1, throttle_stick=0.5)
b_alt = make_backend("ALT_HOLD")
b_alt.steer(cmd)
check("ALT_HOLD: throttle channel IS written (angle mode = climb-rate assist)",
      4 in b_alt._sent_pwm)

b_acro = make_backend("ACRO")
b_acro.steer(cmd)
check("ACRO: throttle channel is NEVER written, whatever the axes list says",
      4 not in b_acro._sent_pwm)
check("...the other three axes still drive normally in ACRO",
      set(b_acro._sent_pwm) == {1, 2, 3})


print("\n=== 5. the reversal-flag invariant holds in BOTH modes ===")
# The exact property test_intercept.py locks for ALT_HOLD: an invert flag must
# change ONLY the wire value, never the reported physical angle.  Re-checked
# here for rate_mode too, since a unit-mismatched formula there is exactly the
# class of bug this file exists to catch (it bit the ALT_HOLD side once
# already, in the same session that added this feature).
for rate_mode in (False, True):
    L1 = law(invert_roll=False)
    L2 = law(invert_roll=True)
    yaw = 0.0
    c1 = c2 = None
    for _ in range(60):
        yaw += math.radians(8.0) * 0.05
        a = dict(roll=0.0, pitch=0.0, yaw=yaw, yawspeed=0.0, pitchspeed=0.0)
        px = (640 / 2.0) * math.tan(math.radians(3.0)) / math.tan(math.radians(31.1))
        c1 = L1.compute(px, 0.0, 640, 640, dt=0.05, att=a, rate_mode=rate_mode)
        c2 = L2.compute(px, 0.0, 640, 640, dt=0.05, att=a, rate_mode=rate_mode)
    check(f"roll_stick flips sign with ROLL_REVERSED (rate_mode={rate_mode})",
          abs(c1["roll_stick"] + c2["roll_stick"]) < 1e-9
          and c1["roll_stick"] != 0.0)
    check(f"...but bank_deg (the physical report) does not (rate_mode={rate_mode})",
          abs(c1["bank_deg"] - c2["bank_deg"]) < 1e-6,
          f"{c1['bank_deg']:.3f} vs {c2['bank_deg']:.3f}")


print("\n=== 6. WIND: the actual field fix, measured ===")
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
        import test_intercept as T
except SystemExit:
    pass

G = 9.80665


class WindSim(T.Sim):
    """T.Sim, but drag opposes AIRSPEED (ground velocity minus a constant wind
    vector) instead of ground velocity.  An attitude-hold vehicle actively
    holds whatever lean/bank is commanded regardless of wind -- that IS what
    an attitude controller does -- so wind does not change the angle achieved,
    it changes the GROUND velocity that angle produces.  A hovering airframe
    (lean=bank=0) therefore drifts at exactly the wind speed."""

    def __init__(self, *a, wind=(0.0, 0.0), **kw):
        super().__init__(*a, **kw)
        self.wind = wind

    def step(self, lean_c, bank_c, yawrate_c, climb_c, dt):
        for name, cmd, tau in (("lean", lean_c, self.TAU_ATT),
                               ("bank", bank_c, self.TAU_ATT),
                               ("yaw_rate", yawrate_c, self.TAU_YAW),
                               ("climb", climb_c, self.TAU_CLIMB)):
            cur = getattr(self, name)
            setattr(self, name, cur + (dt / (tau + dt)) * (cmd - cur))
        self.yaw += self.yaw_rate * dt
        a_f = G * math.tan(-self.lean)
        a_r = G * math.tan(self.bank)
        f, r = self.fwd(), self.right()
        ax = a_f * f[0] + a_r * r[0]
        ay = a_f * f[1] + a_r * r[1]
        avx, avy = self.v[0] - self.wind[0], self.v[1] - self.wind[1]
        asp = math.hypot(avx, avy)
        ax -= self.DRAG * asp * avx
        ay -= self.DRAG * asp * avy
        self.v[0] += ax * dt
        self.v[1] += ay * dt
        self.v[2] = self.climb
        for i in range(3):
            self.p[i] += self.v[i] * dt


def stationary(dist, height):
    def tgt(t, _d=dist, _h=height):
        return (_d, 0.0, _h)
    return tgt


SKY = dict(sens=0.5, pn_near_mult=1.4, lock_ramp_s=0.5, lean_el_guard=0.0,
           edge_guard=0.55, stick_limit_throttle=1.0, max_climb_mps=25.0,
           max_descent_mps=4.0, slew_per_s=14.0, rate_tau=0.13,
           deadband_frac=0.040, climb_los=True, climb_trim=8.0,
           cruise_mps=15.0, cruise_lean_deg=20.0)


def wind_run(bank_far, wind):
    kw = dict(SKY); kw["bank_far_deg"] = bank_far
    class L(InterceptLaw):
        def __init__(self, **k): k.update(kw); super().__init__(**k)
    T.InterceptLaw = L
    T.Sim = lambda *a, **k: WindSim(*a, wind=wind, **k)
    return T.run(45.0, stationary(20.0, 10.0), cam_tilt_deg=30.0,
                max_lean_deg=22.0, climb_gain=65.0, dur=25.0)


try:
    r0_old = wind_run(12.0, (0.0, 0.0))
    r0_new = wind_run(24.0, (0.0, 0.0))
    check("raising bank_far_deg costs NOTHING in still air",
          abs(r0_old["best"] - r0_new["best"]) < 0.05
          and r0_old["t_hit"] is not None and r0_new["t_hit"] is not None,
          f"12 deg: {r0_old['best']:.2f} m   24 deg: {r0_new['best']:.2f} m")

    r12_old = wind_run(12.0, (0.0, 12.0))
    r12_new = wind_run(24.0, (0.0, 12.0))
    check("the OLD ceiling (12 deg) MISSES a 12 m/s crosswind",
          r12_old["t_hit"] is None,
          f"closest {r12_old['best']:.2f} m, in frame {r12_old['inframe']*100:.0f}%")
    check("the NEW ceiling (24 deg) HITS the same crosswind",
          r12_new["t_hit"] is not None and r12_new["best"] < 1.0,
          f"hit at {r12_new['t_hit']:.2f} s, closest {r12_new['best']:.2f} m, "
          f"in frame {r12_new['inframe']*100:.0f}%")
finally:
    T.InterceptLaw = InterceptLaw          # do not leak the monkeypatch


print("\n" + "=" * 62)
if FAILED:
    print(f"{len(FAILED)} ACRO/WIND CHECK(S) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("ALL ACRO/WIND CHECKS PASSED")
