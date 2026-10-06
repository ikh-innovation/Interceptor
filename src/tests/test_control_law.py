#!/usr/bin/env python3
"""Offline verification of fc_control: signs, limits, plumbing, convergence."""
import math
import os
import sys
import tempfile
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fc_control import (AcroCommander, ArduPilotAcro, ControlLaw, _deadband,
                        _msp_frame, commander_from_env)

FW = FH = 640
fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


print("\n=== 1. sign conventions ===")
law = ControlLaw(h_fov_deg=62.2, v_fov_deg=48.8, deadband_frac=0.03, slew_per_s=1e9)

c = law.compute(+160, 0, FW, FH)      # target RIGHT of centre
check("target right -> yaw stick positive (yaw right)", c["yaw_stick"] > 0,
      f"yaw_stick={c['yaw_stick']:+.3f}")
law.neutral()
c = law.compute(-160, 0, FW, FH)
check("target left  -> yaw stick negative", c["yaw_stick"] < 0,
      f"yaw_stick={c['yaw_stick']:+.3f}")
law.neutral()
c = law.compute(0, +160, FW, FH)      # target BELOW centre
check("target below -> pitch stick negative (nose DOWN)", c["pitch_stick"] < 0,
      f"pitch_stick={c['pitch_stick']:+.3f}")
law.neutral()
c = law.compute(0, -160, FW, FH)
check("target above -> pitch stick positive (nose UP)", c["pitch_stick"] > 0,
      f"pitch_stick={c['pitch_stick']:+.3f}")

print("\n=== 2. deadband / saturation / limits ===")
law.neutral()
c = law.compute(4, 4, FW, FH)         # 4px of 320 = 1.25% < 3% deadband
check("inside deadband -> exactly zero", c["yaw_stick"] == 0.0 and c["pitch_stick"] == 0.0,
      f"yaw={c['yaw_stick']} pitch={c['pitch_stick']}")
check("deadband rescale is continuous", abs(_deadband(0.03000001, 0.03)) < 1e-5)
check("deadband reaches 1.0 at the edge", abs(_deadband(1.0, 0.03) - 1.0) < 1e-9)

law2 = ControlLaw(kp_yaw=50.0, kp_pitch=50.0, stick_limit=0.55, slew_per_s=1e9)
c = law2.compute(319, 319, FW, FH)
check("saturates at stick_limit", abs(c["yaw_stick"]) <= 0.55 + 1e-9
      and abs(abs(c["yaw_stick"]) - 0.55) < 1e-9, f"yaw_stick={c['yaw_stick']:+.4f}")
check("reported rate matches the clamped stick",
      abs(c["yaw_rate"] - 0.55 * law2.max_yaw_rate) < 1e-6,
      f"{math.degrees(c['yaw_rate']):.1f} deg/s")

print("\n=== 3. slew limiting ===")
law3 = ControlLaw(kp_yaw=50.0, stick_limit=1.0, slew_per_s=2.0, deadband_frac=0.0)
import time
law3.compute(0, 0, FW, FH, dt=0.05)   # seed
c = law3.compute(319, 0, FW, FH, dt=0.1)   # demand full stick, 0.1 s elapsed
check("slew caps the step (<=0.25 after 0.1s at 2/s)", 0.0 < c["yaw_stick"] <= 0.30,
      f"yaw_stick={c['yaw_stick']:+.3f}")

print("\n=== 4. inversion flags ===")
lawi = ControlLaw(invert_yaw=True, invert_pitch=True, slew_per_s=1e9)
ci = lawi.compute(+160, +160, FW, FH)
check("YAW_REVERSED flips the yaw stick", ci["yaw_stick"] < 0, f"{ci['yaw_stick']:+.3f}")
check("PITCH_REVERSED flips the pitch stick", ci["pitch_stick"] > 0,
      f"{ci['pitch_stick']:+.3f}")

print("\n=== 5. closed-loop convergence (kinematic sim) ===")


def simulate(kp, seconds=6.0, dt=0.05, tgt_az_deg=25.0, tgt_el_deg=-15.0,
             hfov=62.2, vfov=48.8, latency_steps=2, kp_pitch=None,
             stick_limit=0.55, slew=3.0, kd=0.0, deadband=0.03):
    """Camera boresight integrates the commanded body rates.  If the sign
    convention is right the pixel error must decay to inside the deadband."""
    L = ControlLaw(h_fov_deg=hfov, v_fov_deg=vfov, kp_yaw=kp,
                   kp_pitch=(kp if kp_pitch is None else kp_pitch),
                   kd_yaw=kd, kd_pitch=kd,
                   deadband_frac=deadband, stick_limit=stick_limit,
                   slew_per_s=slew)
    psi = 0.0                     # heading (rad)
    theta = 0.0                   # pitch   (rad)
    tgt_az = math.radians(tgt_az_deg)
    tgt_el = math.radians(tgt_el_deg)
    tan_h, tan_v = math.tan(math.radians(hfov) / 2), math.tan(math.radians(vfov) / 2)
    pipe = [(0.0, 0.0)] * latency_steps        # detection latency
    hist = []
    for _ in range(int(seconds / dt)):
        az_rel = tgt_az - psi                  # >0 target to the right
        el_rel = tgt_el - theta                # >0 target above boresight
        if abs(az_rel) > math.radians(80) or abs(el_rel) > math.radians(80):
            return None, "boresight ran away"
        ex = (math.tan(az_rel) / tan_h) * (FW / 2)
        ey = -(math.tan(el_rel) / tan_v) * (FH / 2)   # below centre -> +ey
        pipe.append((ex, ey))
        mex, mey = pipe.pop(0)
        c = L.compute(mex, mey, FW, FH, dt=dt)
        psi += c["yaw_rate"] * dt
        theta += c["pitch_rate"] * dt
        hist.append((math.degrees(az_rel), math.degrees(el_rel)))
    return hist, None


for kp in (1.0, 2.0, 3.0):
    hist, err = simulate(kp)
    if err:
        check(f"kp={kp}: converges", False, err)
        continue
    az0, el0 = hist[0]
    azf, elf = hist[-1]
    peak_over = max(0.0, -min(a for a, _ in hist))   # overshoot past 0 in azimuth
    check(f"kp={kp}: azimuth error {az0:+.1f} -> {azf:+.1f} deg (converges)",
          abs(azf) < abs(az0) * 0.25,
          f"overshoot {peak_over:.1f} deg")
    check(f"kp={kp}: elevation error {el0:+.1f} -> {elf:+.1f} deg (converges)",
          abs(elf) < abs(el0) * 0.25)

print("\n=== 6. a genuinely reversed axis: diverges, and the flag fixes it ===")


def simulate_reversed(invert):
    """Model an airframe/TX whose yaw axis responds BACKWARDS to the stick: the
    plant applies -stick*max_rate.  Without YAW_REVERSED the loop must run away
    (that is the symptom on the gimbal); setting it must recover convergence."""
    L = ControlLaw(kp_yaw=2.0, slew_per_s=3.0, deadband_frac=0.03,
                   invert_yaw=invert)
    psi, tgt = 0.0, math.radians(25.0)
    tan_h = math.tan(math.radians(62.2) / 2)
    worst = 0.0
    for _ in range(160):
        az_rel = tgt - psi
        if abs(az_rel) > math.radians(85):
            return math.degrees(az_rel)
        ex = (math.tan(az_rel) / tan_h) * (FW / 2)
        c = L.compute(ex, 0, FW, FH, dt=0.05)
        psi += -(c["yaw_stick"] * L.max_yaw_rate) * 0.05   # REVERSED plant
        worst = max(worst, abs(math.degrees(tgt - psi)))
    return math.degrees(tgt - psi)


bad = simulate_reversed(invert=False)
good = simulate_reversed(invert=True)
check("reversed axis without the flag DIVERGES (so a polarity error is caught)",
      abs(bad) > 25.0, f"final error {bad:+.1f} deg")
check("YAW_REVERSED=1 recovers convergence on that same axis",
      abs(good) < 6.0, f"final error {good:+.1f} deg")

print("\n=== 7. RC PWM mapping ===")
be = ArduPilotAcro.__new__(ArduPilotAcro)          # no link needed
be.rc = {"yaw": dict(chan=4, min=1100, max=1900, trim=1500, reversed=False),
         "pitch": dict(chan=2, min=1000, max=2000, trim=1450, reversed=True)}
check("centre stick -> trim", be._axis_pwm("yaw", 0.0) == 1500)
check("full right -> max", be._axis_pwm("yaw", 1.0) == 1900)
check("full left  -> min", be._axis_pwm("yaw", -1.0) == 1100)
check("half right honours asymmetric trim",
      be._axis_pwm("pitch", 0.5) == int(round(1450 - 0.5 * (1450 - 1000))),
      f"reversed axis -> {be._axis_pwm('pitch', 0.5)}")
check("clamped inside [min,max]", 1100 <= be._axis_pwm("yaw", 5.0) <= 1900)

print("\n=== 8. MSP framing ===")
f = _msp_frame(105)
check("MSP request header", f[:3] == b"$M<", repr(f))
check("MSP zero-payload crc = len^cmd", f[-1] == (0 ^ 105), f"crc={f[-1]}")
f2 = _msp_frame(200, bytes([0x01, 0x02]))
check("MSP crc over payload", f2[-1] == (2 ^ 200 ^ 1 ^ 2), f"crc={f2[-1]}")

print("\n=== 9. offline commander end-to-end ===")
tmp = tempfile.mkdtemp()
os.environ["MAV_CONN"] = "none"
cmdr = commander_from_env(log_dir=tmp, verbose=False)
cmdr.connect()
check("offline backend links=False", cmdr.linked is False)
check("no-crash compute+update", True)
for i in range(10):
    c = cmdr.compute(100, -50, "LOCK", FW, FH)
    cmdr.update(c, (100, -50), have_target=True)
    time.sleep(0.02)
line = cmdr.format_command("LOCK", c, (100, -50), False, cmdr.linked, True)
check("format_command mentions SIM when offline", "SIM" in line, line)
c0 = cmdr.compute(0, 0, "SEARCH", FW, FH)
check("SEARCH state -> neutral sticks",
      c0["yaw_stick"] == 0.0 and c0["pitch_stick"] == 0.0)
cmdr.close()
csv = list(Path(tmp).glob("control_*.csv"))
check("control CSV written", len(csv) == 1, str(csv))
if csv:
    rows = csv[0].read_text().strip().splitlines()
    check("CSV has a header + rows", len(rows) > 1, f"{len(rows)} lines")
    check("CSV header names sticks not velocities",
          "yaw_stick" in rows[0] and "vz" not in rows[0], rows[0])
cmdr.close()                                    # idempotent
check("close() is idempotent", True)

print("\n=== 10. adaptive law: snap on far, ease in near ===")

# The adaptive SHAPE is still supported by fc_control (CONTROL_SHAPE=adaptive) but
# is no longer a run.py preset — run.py ships exactly one preset, see section 11.
# These checks stay because the shape code is still reachable and its sign
# conventions must keep matching the linear law's.
AD = dict(shape="adaptive", shape_gamma=1.6, shape_sat=0.42,
          kd_yaw=0.18, kd_pitch=0.18, stick_limit=0.60,
          stick_limit_yaw=0.60, stick_limit_pitch=0.18,
          slew_per_s=12.0, deadband_frac=0.03)
YLIM, PLIM, SAT, GAMMA = 0.60, 0.18, 0.42, 1.6


def ad_law(**over):
    kw = dict(AD)
    kw.update(over)
    return ControlLaw(**kw)


def stick_at(frac, axis="yaw"):
    """Deflection commanded for a target `frac` of the half-frame off centre.
    Slew history is reset each time so we read the law, not the ramp."""
    L = ad_law()
    px = frac * (FW / 2)
    last = None
    for _ in range(40):                       # let the slew limiter settle
        c = L.compute(px if axis == "yaw" else 0.0,
                      0.0 if axis == "yaw" else px, FW, FH, dt=0.05)
        last = c
    return last["yaw_stick"] if axis == "yaw" else last["pitch_stick"]


# -- signs must match the linear law exactly; only the magnitude curve differs --
L = ad_law()
c = L.compute(+200, 0, FW, FH, dt=0.05)
check("adaptive: target RIGHT -> yaw positive", c["yaw_stick"] > 0,
      f"{c['yaw_stick']:+.3f}")
L = ad_law()
c = L.compute(-200, 0, FW, FH, dt=0.05)
check("adaptive: target LEFT -> yaw negative", c["yaw_stick"] < 0,
      f"{c['yaw_stick']:+.3f}")
L = ad_law()
c = L.compute(0, +200, FW, FH, dt=0.05)
check("adaptive: target BELOW -> pitch negative (nose down)",
      c["pitch_stick"] < 0, f"{c['pitch_stick']:+.3f}")
L = ad_law()
c = L.compute(0, -200, FW, FH, dt=0.05)
check("adaptive: target ABOVE -> pitch positive (nose up)",
      c["pitch_stick"] > 0, f"{c['pitch_stick']:+.3f}")

# -- saturation.  SHAPE_SAT is measured on the offset AFTER the deadband rescale,
#    so the raw offset that saturates is slightly larger than SHAPE_SAT itself. --
raw_sat = SAT * (1.0 - 0.03) + 0.03
s_sat = stick_at(raw_sat + 0.01)
s75 = stick_at(0.75)
s99 = stick_at(0.99)
check(f"adaptive: yaw hits its ceiling by {(raw_sat + 0.01) * 100:.0f}% off-centre",
      abs(s_sat - YLIM) < 1e-6, f"{s_sat:.3f} (ceiling {YLIM})")
check("adaptive: still exactly the ceiling at 75% (saturated, not growing)",
      abs(s75 - YLIM) < 1e-6, f"{s75:.3f}")
check("adaptive: still exactly the ceiling at 99%", abs(s99 - YLIM) < 1e-6,
      f"{s99:.3f}")
check("adaptive: pitch has its OWN, lower ceiling",
      abs(abs(stick_at(0.75, "pitch")) - PLIM) < 1e-6,
      f"pitch {abs(stick_at(0.75, 'pitch')):.3f} vs yaw {s75:.3f}")

# -- the near field must collapse, and follow the documented curve --------------
s30, s20, s10, s05 = (stick_at(0.30), stick_at(0.20), stick_at(0.10),
                      stick_at(0.05))
check("adaptive: monotonic in offset (05 < 10 < 20 < 30)",
      s05 < s10 < s20 < s30, f"{s05:.3f} {s10:.3f} {s20:.3f} {s30:.3f}")
for frac, got in ((0.30, s30), (0.20, s20), (0.10, s10), (0.05, s05)):
    # The deadband rescales the offset before shaping, so predict through it too.
    n = (frac - 0.03) / (1.0 - 0.03)
    want = min(1.0, n / SAT) ** GAMMA * YLIM
    check(f"adaptive: yaw stick at {frac * 100:.0f}% off-centre matches the curve",
          abs(got - want) < 0.02, f"got {got:.3f}, curve says {want:.3f}")

# -- and it must actually be FASTER far out than the linear preset it replaces --
LIN = ControlLaw(kp_yaw=3.5, kp_pitch=6.0, kd_yaw=0.15, kd_pitch=0.15,
                 stick_limit=0.35, slew_per_s=3.0, deadband_frac=0.03)
lin_last = None
for _ in range(40):
    lin_last = LIN.compute(0.40 * (FW / 2), 0, FW, FH, dt=0.05)
check("adaptive beats the --tuned linear preset at 40% off-centre",
      stick_at(0.40) > lin_last["yaw_stick"] * 1.5,
      f"adaptive {stick_at(0.40):.3f} vs tuned {lin_last['yaw_stick']:.3f}")

# -- reversal flags still work in this mode ------------------------------------
LR = ad_law(invert_yaw=True)
c = LR.compute(+200, 0, FW, FH, dt=0.05)
check("adaptive: YAW_REVERSED flips the stick", c["yaw_stick"] < 0,
      f"{c['yaw_stick']:+.3f}")
check("adaptive: reported yaw_rate stays the PHYSICAL intent under inversion",
      c["yaw_rate"] > 0, f"{math.degrees(c['yaw_rate']):+.1f} d/s")


def simulate_adaptive(seconds=6.0, dt=0.05, tgt_az_deg=25.0, tgt_el_deg=-15.0,
                      hfov=62.2, vfov=48.8, latency_steps=2):
    """Same kinematic sim as section 5, driving the adaptive law.  A law that
    snaps to full stick is exactly the kind that can limit-cycle, so convergence
    here is not a formality."""
    L = ad_law(h_fov_deg=hfov, v_fov_deg=vfov)
    psi = theta = 0.0
    tgt_az, tgt_el = math.radians(tgt_az_deg), math.radians(tgt_el_deg)
    tan_h = math.tan(math.radians(hfov) / 2)
    tan_v = math.tan(math.radians(vfov) / 2)
    pipe = [(0.0, 0.0)] * latency_steps
    hist = []
    for _ in range(int(seconds / dt)):
        az_rel, el_rel = tgt_az - psi, tgt_el - theta
        if abs(az_rel) > math.radians(80) or abs(el_rel) > math.radians(80):
            return None, "boresight ran away"
        ex = (math.tan(az_rel) / tan_h) * (FW / 2)
        ey = -(math.tan(el_rel) / tan_v) * (FH / 2)
        pipe.append((ex, ey))
        mex, mey = pipe.pop(0)
        c = L.compute(mex, mey, FW, FH, dt=dt)
        psi += c["yaw_rate"] * dt
        theta += c["pitch_rate"] * dt
        hist.append((math.degrees(az_rel), math.degrees(el_rel)))
    return hist, None


def settle_of(hist, tol=2.0, dt=0.05):
    """Seconds until BOTH axes are inside tol and stay there.  None = never."""
    if hist is None:
        return None
    for i, (a, e) in enumerate(hist):
        if (abs(a) < tol and abs(e) < tol
                and all(abs(x) < tol and abs(y) < tol for x, y in hist[i:])):
            return i * dt
    return None


def crossings_of(hist):
    return (sum(1 for i in range(1, len(hist)) if hist[i][0] * hist[i - 1][0] < 0)
            + sum(1 for i in range(1, len(hist)) if hist[i][1] * hist[i - 1][1] < 0))


def simulate_kw(law_kw, tgt_az_deg, tgt_el_deg, seconds=8.0, dt=0.05, lat=2,
                noise=0.0, seed=7, hfov=62.2, vfov=48.8):
    """The section-5 kinematic sim, but taking a full ControlLaw kwargs dict and
    able to inject detection jitter.

    `noise` is peak pixel error added to the measurement, from a deterministic LCG
    so a failure is reproducible.  Jitter matters because the derivative term is
    computed from the detection position: a law that looks fine on clean input can
    still chase noise on real input.
    """
    L = ControlLaw(h_fov_deg=hfov, v_fov_deg=vfov, **law_kw)
    psi = theta = 0.0
    tgt_az, tgt_el = math.radians(tgt_az_deg), math.radians(tgt_el_deg)
    tan_h = math.tan(math.radians(hfov) / 2)
    tan_v = math.tan(math.radians(vfov) / 2)
    pipe = [(0.0, 0.0)] * lat
    hist = []
    state = seed
    for _ in range(int(seconds / dt)):
        az_rel, el_rel = tgt_az - psi, tgt_el - theta
        if abs(az_rel) > math.radians(80) or abs(el_rel) > math.radians(80):
            return None
        ex = (math.tan(az_rel) / tan_h) * (FW / 2)
        ey = -(math.tan(el_rel) / tan_v) * (FH / 2)
        if noise:
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            ex += ((state / 0x7FFFFFFF) - 0.5) * 2 * noise
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            ey += ((state / 0x7FFFFFFF) - 0.5) * 2 * noise
        pipe.append((ex, ey))
        mex, mey = pipe.pop(0)
        c = L.compute(mex, mey, FW, FH, dt=dt)
        psi += c["yaw_rate"] * dt
        theta += c["pitch_rate"] * dt
        hist.append((math.degrees(az_rel), math.degrees(el_rel)))
    return hist


# Six starting geometries, both signs on both axes.  A preset that only works
# from one corner is not a preset.
GEOMS = [(25.0, -15.0), (8.0, -5.0), (-30.0, 12.0),
         (45.0, -25.0), (15.0, -30.0), (-10.0, 20.0)]

worst_ad, worst_cross_ad, diverged = 0.0, 0, []
for az, el in GEOMS:
    h = simulate_adaptive(tgt_az_deg=az, tgt_el_deg=el)[0]
    s = settle_of(h)
    if s is None:
        diverged.append((az, el))
        continue
    worst_ad = max(worst_ad, s)
    worst_cross_ad = max(worst_cross_ad, crossings_of(h))

check("adaptive: converges from every starting geometry", not diverged,
      f"diverged at {diverged}" if diverged else f"worst {worst_ad:.2f}s")
check("adaptive: does NOT limit-cycle on any geometry", worst_cross_ad <= 3,
      f"max {worst_cross_ad} zero crossings")

print("\n=== 11. the shipped TUNED preset: the only thing that flies ===")

# These MUST equal run.py's TUNED dict.  run.py's comment block makes specific
# numerical claims about this preset; this section is what stops those claims from
# quietly becoming false.
TUNED_KW = dict(kp_yaw=4.5, kp_pitch=6.0, kd_yaw=0.10, kd_pitch=0.10,
                stick_limit=0.65, stick_limit_yaw=0.65, stick_limit_pitch=0.22,
                slew_per_s=11.0, deadband_frac=0.03)

# Eight geometries, both signs on both axes.  Six was not enough: a candidate that
# passed six went unstable on the seventh during tuning.
GEOMS8 = GEOMS + [(35.0, 5.0), (-20.0, -8.0)]


def sim_tuned(az, el, lat=2, noise=0.0, seed=7):
    return simulate_kw(TUNED_KW, az, el, lat=lat, noise=noise, seed=seed)


worst2 = 0.0
bad = []
for az, el in GEOMS8:
    h = sim_tuned(az, el, lat=2)
    s = settle_of(h)
    if s is None or (h and crossings_of(h) > 4):
        bad.append((az, el))
        continue
    worst2 = max(worst2, s)
check("TUNED converges from all 8 geometries at 0.1s detection latency",
      not bad, f"failed at {bad}" if bad else f"worst {worst2:.2f}s")
check("TUNED settles within 0.6s (run.py claims 0.50s)", worst2 <= 0.60,
      f"{worst2:.2f}s")

worst3 = 0.0
bad3 = []
for az, el in GEOMS8:
    h = sim_tuned(az, el, lat=3)
    s = settle_of(h)
    if s is None or (h and crossings_of(h) > 4):
        bad3.append((az, el))
        continue
    worst3 = max(worst3, s)
check("TUNED still converges at 1.5x the detection latency", not bad3,
      f"failed at {bad3}" if bad3 else f"worst {worst3:.2f}s")

# Jitter robustness on several noise seeds.  The PREVIOUS tuned managed 3 of 5
# here, which is the concrete sense in which this preset is not just faster.
seeds_ok = 0
for seed in (1, 7, 13, 29, 101):
    good = True
    for az, el in GEOMS8:
        h = sim_tuned(az, el, lat=3, noise=8.0, seed=seed)
        if settle_of(h) is None or (h and crossings_of(h) > 4):
            good = False
            break
    seeds_ok += good
check("TUNED survives 8px detection jitter on all 5 noise seeds",
      seeds_ok == 5, f"{seeds_ok}/5")

# And it must beat what it replaced, on the same bar.
prev = dict(kp_yaw=3.5, kp_pitch=6.0, kd_yaw=0.15, kd_pitch=0.15,
            stick_limit=0.35, slew_per_s=3.0, deadband_frac=0.03)
prev_worst = 0.0
for az, el in GEOMS8:
    s = settle_of(simulate_kw(prev, az, el, lat=2))
    prev_worst = max(prev_worst, s if s is not None else 99.0)
check("TUNED is faster than the preset it replaced",
      worst2 < prev_worst, f"{worst2:.2f}s vs previous {prev_worst:.2f}s")

# Pitch authority is deliberately BELOW yaw.  run.py explains why; assert it so
# nobody "fixes" the asymmetry by raising pitch and reintroducing the limit cycle.
LT = ControlLaw(**TUNED_KW)
check("TUNED: pitch ceiling is lower than yaw (ACRO_RP_RATE 360 vs 202)",
      LT.stick_limit_pitch < LT.stick_limit_yaw,
      f"pitch {LT.stick_limit_pitch} < yaw {LT.stick_limit_yaw}")
# Pitch is at its ceiling in the sense that matters: raising KP_PITCH one step
# destabilises outright.  This is the check that stops "make pitch more
# aggressive" from being applied without re-running the search.
# One step hotter does not fail by refusing to settle — it fails by HUNTING:
# extra zero crossings on clean input, and total collapse once the detection is
# noisy.  Hunting around the target is exactly the "imprecise" symptom reported
# from the field, so this is the failure mode that matters, and it is what a
# naive "turn the pitch gain up" would reintroduce.
hotter_pitch = dict(TUNED_KW, kp_pitch=8.0)
hot_hunts = False
for az, el in GEOMS8:
    h = simulate_kw(hotter_pitch, az, el, lat=3)
    if h is None or crossings_of(h) > 4:
        hot_hunts = True
        break
hot_seeds = 0
for seed in (1, 7, 13, 29, 101):
    good = True
    for az, el in GEOMS8:
        h = simulate_kw(hotter_pitch, az, el, lat=3, noise=8.0, seed=seed)
        if h is None or settle_of(h) is None or crossings_of(h) > 4:
            good = False
            break
    hot_seeds += good
check("KP_PITCH=8.0 hunts and/or dies on jitter — pitch is already maxed",
      hot_hunts or hot_seeds < 5,
      f"hunts={hot_hunts}, jitter {hot_seeds}/5 (TUNED gets 5/5)")

# The pitch CEILING is not what limits the steady-state command (KP_PITCH gets
# nowhere near 0.22 on a normal error); its job is clipping jitter transients.
# Loosening it measurably costs noise robustness, which is why it is not "spare".
loose_ceiling = dict(TUNED_KW, stick_limit_pitch=0.40)
loose_ok = 0
for seed in (1, 7, 13, 29, 101):
    good = True
    for az, el in GEOMS8:
        h = simulate_kw(loose_ceiling, az, el, lat=3, noise=8.0, seed=seed)
        if settle_of(h) is None or (h and crossings_of(h) > 4):
            good = False
            break
    loose_ok += good
check("the pitch ceiling earns its keep: loosening it costs jitter robustness",
      loose_ok < 5, f"0.40 ceiling -> {loose_ok}/5 seeds vs 0.22 -> 5/5")

print("\n=== 12. approach mode: forward whatever the target's elevation ===")

# The field bug, in one sentence: the pilot flew BELOW the target, so the camera
# had to look UP, so pitch leaned the airframe BACK, so it flew away.  On a free
# multirotor pitch attitude IS direction of travel, so aiming and approaching are
# opposite commands on one axis.  Approach mode takes pitch off aiming entirely.

APPROACH_KW = dict(TUNED_KW, pitch_mode="approach", approach_deg=12.0,
                   approach_kp=2.5, approach_thr_deadband=0.05)


def fly(law_kw, tgt_az_deg, tgt_el_deg, throttle=1.0, seconds=6.0, dt=0.05,
        lat=2, hfov=62.2, vfov=48.8):
    """Kinematic sim WITH translation.

    Adds what section 5's sim leaves out and what the field bug lives in: the
    airframe's pitch attitude integrates the commanded pitch rate, and a leaned
    multirotor accelerates along the direction it leans.  `fwd` is displacement
    towards the target: positive = closing, negative = retreating.
    """
    L = ControlLaw(h_fov_deg=hfov, v_fov_deg=vfov, **law_kw)
    psi = 0.0            # heading error integrator
    theta = 0.0          # pitch attitude, positive NOSE-UP
    vel = 0.0            # forward velocity, m/s
    fwd = 0.0            # forward displacement, m
    tgt_az, tgt_el = math.radians(tgt_az_deg), math.radians(tgt_el_deg)
    tan_h = math.tan(math.radians(hfov) / 2)
    tan_v = math.tan(math.radians(vfov) / 2)
    pipe = [(0.0, 0.0)] * lat
    leans = []
    for _ in range(int(seconds / dt)):
        az_rel = tgt_az - psi
        # The camera is body-fixed, so leaning the airframe moves the target in
        # frame.  This coupling is the whole reason aim mode misbehaves in flight.
        el_rel = tgt_el - theta
        if abs(az_rel) > math.radians(80) or abs(el_rel) > math.radians(80):
            return None
        ex = (math.tan(az_rel) / tan_h) * (FW / 2)
        ey = -(math.tan(el_rel) / tan_v) * (FH / 2)
        pipe.append((ex, ey))
        mex, mey = pipe.pop(0)
        c = L.compute(mex, mey, FW, FH, dt=dt,
                      measured_pitch_rad=theta, throttle_frac=throttle)
        psi += c["yaw_rate"] * dt
        theta += c["pitch_rate"] * dt
        # Nose-down (negative theta) accelerates forward.  9.81*tan(lean), lightly
        # damped so velocity does not grow without bound in a 6 s window.
        vel += (9.81 * math.tan(-theta) - 0.25 * vel) * dt
        fwd += vel * dt
        leans.append(math.degrees(theta))
    return dict(fwd=fwd, lean_final=leans[-1], leans=leans,
                az_final=math.degrees(tgt_az - psi))


# -- THE regression.  Target above, below, left, right: always forward. ---------
for name, (az, el) in (("target ABOVE (the field case)", (0.0, 20.0)),
                       ("target BELOW", (0.0, -20.0)),
                       ("target LEFT", (-25.0, 0.0)),
                       ("target RIGHT", (25.0, 0.0)),
                       ("target ABOVE and LEFT", (-20.0, 15.0))):
    r = fly(APPROACH_KW, az, el)
    check(f"approach: {name} -> moves FORWARD",
          r is not None and r["fwd"] > 0.5,
          "diverged" if r is None else f"{r['fwd']:+.1f} m in 6 s")

# -- and prove the bug is real in aim mode, so this test means something --------
r_aim = fly(dict(TUNED_KW, pitch_mode="aim"), 0.0, 20.0)
check("aim mode with the target ABOVE really does RETREAT (the reported bug)",
      r_aim is not None and r_aim["fwd"] < 0,
      "diverged" if r_aim is None else f"{r_aim['fwd']:+.1f} m — flies backwards")

# -- the lean is HELD, not integrated away.  This is what ACRO cannot do open-loop.
r = fly(APPROACH_KW, 0.0, 20.0, seconds=8.0)
check("approach: lean settles at the commanded nose-down angle",
      r is not None and abs(r["lean_final"] - (-12.0)) < 2.0,
      f"{r['lean_final']:+.1f} deg (target -12)")
check("approach: lean never runs away past the clamp",
      r is not None and min(r["leans"]) > -20.0,
      f"most nose-down {min(r['leans']):+.1f} deg")

# -- yaw must still do its job while pitch is busy approaching ------------------
r = fly(APPROACH_KW, 30.0, 15.0, seconds=8.0)
check("approach: yaw still centres the target horizontally",
      r is not None and abs(r["az_final"]) < 3.0,
      f"azimuth {r['az_final']:+.1f} deg")

# -- throttle is the pilot's go/stop, and it must mean it ----------------------
r_idle = fly(APPROACH_KW, 0.0, 20.0, throttle=0.0)
check("approach: throttle at mid commands NO lean and no motion",
      r_idle is not None and abs(r_idle["fwd"]) < 0.05
      and abs(r_idle["lean_final"]) < 0.5,
      f"{r_idle['fwd']:+.3f} m, lean {r_idle['lean_final']:+.2f} deg")
r_half = fly(APPROACH_KW, 0.0, 20.0, throttle=0.5)
check("approach: half throttle leans less than full throttle",
      r_half is not None and r_idle is not None
      and 0.0 < r_half["fwd"] < r["fwd"],
      f"half {r_half['fwd']:+.1f} m")

# -- and the two hard safety properties ---------------------------------------
LA = ControlLaw(**APPROACH_KW)
check("approach: a missing/stale attitude commands ZERO pitch, never a guess",
      LA.approach_pitch_stick(None, 1.0) == 0.0)
worst_up = max(LA.approach_pitch_stick(math.radians(d), 1.0)
               for d in range(-12, 1))
check("approach: never commands nose-up while at or above the target lean",
      worst_up <= 1e-9, f"max positive stick {worst_up:+.4f}")
for el in (-40, -20, 0, 20, 40):
    c = LA.compute(0, -el * 8, FW, FH, dt=0.05, measured_pitch_rad=0.0,
                   throttle_frac=1.0)
    if c["pitch_stick"] > 0:
        check(f"approach: elevation error {el} never yields nose-up", False,
              f"{c['pitch_stick']:+.3f}")
        break
else:
    check("approach: no elevation error anywhere yields nose-up", True,
          "checked -40..+40 deg")

print("\n" + "=" * 60)
if fails:
    print(f"{len(fails)} FAILURE(S): " + ", ".join(fails))
    sys.exit(1)
print("ALL CHECKS PASSED")
