#!/usr/bin/env python3
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fc_control import InterceptLaw

FAILED = []


def check(label, ok, detail=""):

    if not ok:
        FAILED.append(label)


def law(**kw):
    base = dict(h_fov_deg=62.2, v_fov_deg=48.8, cam_tilt_deg=30.0,
                roll_pn_gain=35.0, max_bank_deg=30.0, max_lean_deg=12.0)
    base.update(kw)
    return InterceptLaw(**base)


def drive(L, fill, n=60, dt=0.05):
    fw = 640.0
    for _ in range(n):
        L._fill += (dt / (L.fill_tau + dt)) * (fill - L._fill)
        L._lock_t = dt if L._lock_t is None else L._lock_t + dt
        L._schedule()
    return L.sens_state()



L = law()
check("bank_far defaults to max_bank",
      abs(L.bank_far - L.max_bank) < 1e-9,
      f"{math.degrees(L.bank_far):.0f} == {math.degrees(L.max_bank):.0f} deg")
s = drive(L, 0.50)
check("gain is untouched at any range with the schedule off",
      abs(s["gain_eff"] - 35.0) < 1e-6, f"gain_eff {s['gain_eff']:.3f}")
check("bank ceiling is untouched at any range with the schedule off",
      abs(s["bank_cap_deg"] - 30.0) < 1e-6, f"cap {s['bank_cap_deg']:.3f} deg")
check("sensitivity defaults to neutral", abs(L.sens - 0.5) < 1e-9, f"{L.sens}")


worst = None
for sv in [i / 20.0 for i in range(11)]:
    st = drive(law(**dict(bank_far_deg=12.0, sens=sv, pn_near_mult=1.4)), 0.001)
    worst = st["gain_eff"] if worst is None else min(worst, st["gain_eff"])
check("no sensitivity at or below neutral reduces the gain",
      worst >= 35.0 - 1e-6, f"lowest gain seen across sens 0..0.5: {worst:.3f}")
st = drive(law(bank_far_deg=12.0, sens=0.0, pn_near_mult=1.4), 0.001)
check("...it takes away BANK instead",
      st["bank_cap_deg"] < 12.0,
      f"cap {st['bank_cap_deg']:.1f} deg at sens 0.0 (far ceiling 12)")


hi = drive(law(bank_far_deg=12.0, sens=1.0, sens_max_mult=2.0), 0.001)
check("sens 1.0 doubles the gain", abs(hi["gain_eff"] - 70.0) < 1e-6,
      f"{hi['gain_eff']:.1f} m/s from a 35 floor")
mid = drive(law(bank_far_deg=12.0, sens=0.75, sens_max_mult=2.0), 0.001)
check("sens 0.75 is halfway there", abs(mid["gain_eff"] - 52.5) < 1e-6,
      f"{mid['gain_eff']:.1f} m/s")


L = law(bank_far_deg=12.0, sens=0.5, pn_near_mult=1.4,
        sched_lo_frac=0.02, sched_hi_frac=0.35)
far = drive(law(bank_far_deg=12.0, sens=0.5, pn_near_mult=1.4), 0.02)
near = drive(law(bank_far_deg=12.0, sens=0.5, pn_near_mult=1.4), 0.35)
check("bank ceiling rises from the far value to the hard limit",
      abs(far["bank_cap_deg"] - 12.0) < 0.5 and abs(near["bank_cap_deg"] - 30.0) < 0.5,
      f"{far['bank_cap_deg']:.1f} -> {near['bank_cap_deg']:.1f} deg")
check("gain rises by PN_NEAR_MULT over the same span",
      abs(far["gain_eff"] - 35.0) < 0.5 and abs(near["gain_eff"] - 49.0) < 0.5,
      f"{far['gain_eff']:.1f} -> {near['gain_eff']:.1f} m/s")
check("the schedule is monotonic in fill", all(
      drive(law(bank_far_deg=12.0), f)["bank_cap_deg"]
      <= drive(law(bank_far_deg=12.0), f + 0.02)["bank_cap_deg"] + 1e-9
      for f in [0.02 * k for k in range(1, 17)]))


L = law(bank_far_deg=12.0, sens=0.5, lock_ramp_s=0.5)
L._fill = 0.02
L._lock_t = None
L._schedule()
first = L.sens_state()
L._lock_t = 0.0
check("the first frame after lock gets a quarter of the ceiling",
      first["lock_ramp"] <= 0.30, f"ramp {first['lock_ramp'] * 100:.0f}%")
for _ in range(20):
    L._lock_t += 0.05
    L._schedule()
check("...and the full ceiling once the ramp has run",
      L.sens_state()["lock_ramp"] >= 0.999,
      f"ramp {L.sens_state()['lock_ramp'] * 100:.0f}% after 1.0 s")
check("neutral() gives the ramp back, so the next lock re-earns it",
      (L.neutral() is not None) and L._lock_t is None and L._ramp <= 0.25 + 1e-9)


L = law(bank_far_deg=12.0)
check("above 1 clamps to 1", L.set_sens(4.2) == 1.0)
check("below 0 clamps to 0", L.set_sens(-3.0) == 0.0)
L.set_sens(0.37)
check("a good value is kept exactly", abs(L.sens - 0.37) < 1e-9)
check("None is ignored, not fatal", L.set_sens(None) == 0.37)
check("a non-number is ignored, not fatal", L.set_sens("banana") == 0.37)
check("sens_state reports what the loop will use",
      set(L.sens_state()) >= {"sens", "gain_eff", "gain_floor",
                              "bank_cap_deg", "fill_frac", "lock_ramp"})


L = law(bank_far_deg=12.0, sens=0.2, pn_near_mult=1.4, lock_ramp_s=0.5)
att = dict(roll=0.0, pitch=0.0, yaw=0.0, yawspeed=0.0, pitchspeed=0.0)
cmd = None
for _ in range(40):
    cmd = L.compute(40.0, 0.0, 640, 640, dt=0.05, bbox_w=60, att=att)
check("the command carries sens, gain and cap for the HUD",
      all(k in cmd for k in ("sens", "pn_gain_eff", "bank_cap_deg", "fill_frac")),
      f"sens {cmd['sens']:.2f} gain {cmd['pn_gain_eff']:.1f} "
      f"cap {cmd['bank_cap_deg']:.1f} deg fill {cmd['fill_frac'] * 100:.1f}%")
check("the reported bank never exceeds the reported ceiling",
      abs(cmd["bank_deg"]) <= cmd["bank_cap_deg"] + 1e-6,
      f"|{cmd['bank_deg']:.2f}| <= {cmd['bank_cap_deg']:.2f} deg")
check("neutral() reports the schedule too, so the HUD sees no hole",
      all(k in L.neutral() for k in ("sens", "pn_gain_eff", "bank_cap_deg")))


if FAILED:

    for f in FAILED:
        pass
    sys.exit(1)
