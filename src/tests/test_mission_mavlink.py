#!/usr/bin/env python3
"""The mission app end to end, over real MAVLink, against the fake ArduPilot
configured as THIS airframe's FC (read off it on 2026-09-25):

    throttle ch1, roll ch2, pitch ch3, yaw ch4, all 1100/1500/1900
    RC1_DZ 30, RC2-4_DZ 20, THR_DZ 100
    PILOT_SPEED_UP 2500, PILOT_SPEED_DN 250, ANGLE_MAX 4000, PILOT_Y_RATE 202.5

It asserts the actual PWM on each channel per phase, and that the axes that
must NOT move are NOT overridden (the pilot keeps them).

    ../.env/yolo/bin/python tests/test_mission_mavlink.py
"""
import contextlib
import csv
import io
import math
import os
import sys
import tempfile
from time import perf_counter, sleep

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fake_fc import FakeArduPilot                                  # noqa: E402
from fc_control import AcroCommander, InterceptLaw                 # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


# ArduPilot 4.6's own decode, integer maths as on the FC (see test_mission B).
def ap_climb(pwm, lo=1100, hi=1900, dz=30, thr_dz=100, up=25.0, dn=2.5):
    trim_low = lo + dz
    c = (1000 * (pwm - trim_low)) // (hi - trim_low) if pwm > trim_low else 0
    mid = (1000 * ((lo + hi) // 2 - trim_low)) // (hi - trim_low)
    top, bot = mid + thr_dz, mid - thr_dz
    return up * (c - top) / (1000.0 - top) if c > top else (
        dn * (c - bot) / bot if c < bot else 0.0)


def ap_norm(pwm, tr=1500, dz=20, lo=1100, hi=1900):
    if pwm < tr - dz:
        return (pwm - (tr - dz)) / float((tr - dz) - lo)
    if pwm > tr + dz:
        return (pwm - (tr + dz)) / float(hi - (tr + dz))
    return 0.0


PORT = 14861
fc = FakeArduPilot(PORT, sysid_mygcs=255, armed=True, mode=0)       # STABILIZE
fc.params.update({"RCMAP_THROTTLE": 1.0, "RCMAP_ROLL": 2.0, "RCMAP_PITCH": 3.0,
                  "RCMAP_YAW": 4.0, "ANGLE_MAX": 4000.0, "PILOT_SPEED_UP": 2500.0,
                  "PILOT_SPEED_DN": 250.0, "PILOT_ACCEL_Z": 500.0, "PSC_JERK_Z": 5.0,
                  "PILOT_Y_RATE": 202.5, "THR_DZ": 100.0, "RC_OVERRIDE_TIME": 3.0,
                  "ATC_INPUT_TC": 0.15})
for ch in (1, 2, 3, 4):
    fc.params.update({f"RC{ch}_MIN": 1100.0, f"RC{ch}_MAX": 1900.0,
                      f"RC{ch}_TRIM": 1500.0, f"RC{ch}_REVERSED": 0.0,
                      f"RC{ch}_DZ": 30.0 if ch == 1 else 20.0})
fc.vfr.update(alt=2.0)
fc.start()
sleep(0.4)
LOGDIR = tempfile.mkdtemp()
law = InterceptLaw(cam_tilt_deg=30.0, stick_limit=0.90)
c = AcroCommander(conn=f"udpin:127.0.0.1:{PORT}", transport="mavlink",
                  axes=("roll", "pitch", "yaw", "throttle"), law=law, send_hz=50.0,
                  steer_modes=frozenset({"ALT_HOLD"}), mission=True,
                  log_dir=LOGDIR, verbose=False)
with contextlib.redirect_stdout(io.StringIO()):
    c.connect()
M = c.mission


def hold(kind, secs, session="s"):
    t_end = perf_counter() + secs
    while perf_counter() < t_end:
        M.request_hold(kind, session, perf_counter())
        sleep(0.1)


try:
    print("=== 1. the pilot has it ===")
    sleep(0.6)
    check("STABILIZE: PILOT, nothing overridden", M.phase == "PILOT" and not fc.overrides,
          f"{M.phase} {fc.overrides}")
    check("a ▲ press is refused while the pilot has it",
          M.request_hold("up", "p", perf_counter()) != "")
    fc.vfr.update(alt=0.2)                                        # on the ground
    fc.mode = 2                                                   # ALT_HOLD
    sleep(0.6)
    check("armed in ALT_HOLD but ON THE GROUND: still PILOT (no phone takeoff)",
          M.phase == "PILOT" and "ground" in c._mission_why and not fc.overrides,
          f"{M.phase} / {c._mission_why}")
    fc.vfr.update(alt=2.0)                                        # took off
    sleep(0.6)
    check("ALT_HOLD + armed + airborne: HOVER, still nothing overridden",
          M.phase == "HOVER" and not fc.overrides, f"{M.phase} {fc.overrides}")
    st = c.mission_state()
    check("instruments: height 2.0 m and vibration read off the FC",
          st["height_m"] == 2.0 and st["vibe"] and st["vibe"]["z"] == 12.0, str(st.get("vibe")))

    print("=== 2. ▲ SLOW: throttle ONLY, exact 1 m/s ===")
    M.set_level("vert", "SLOW")
    hold("up", 0.9)
    o = dict(fc.overrides)
    check("only ch1 (throttle) is overridden; roll/pitch/yaw stay the pilot's",
          set(o) == {1}, str(o))
    check("ch1 decodes to exactly the SLOW climb, 1.0 m/s",
          1 in o and abs(ap_climb(o[1]) - 1.0) < 0.05, f"ch1={o.get(1)} -> "
          f"{ap_climb(o[1]) if 1 in o else 0:.2f} m/s")
    sleep(1.2)                                                   # let go
    check("let go: dead-man + ramp-out, then ch1 handed back",
          not fc.overrides and M.phase == "HOVER", str(fc.overrides))

    print("=== 3. ⟳ MEDIUM: yaw ONLY, exact 20 deg/s ===")
    M.set_level("yaw", "MEDIUM")
    hold("yawr", 0.8)
    o = dict(fc.overrides)
    check("only ch4 (yaw) is overridden", set(o) == {4}, str(o))
    check("ch4 decodes to 20 deg/s", 4 in o and abs(ap_norm(o[4]) * 202.5 - 20.0) < 0.6,
          f"ch4={o.get(4)} -> {ap_norm(o.get(4, 1500)) * 202.5:.1f} deg/s")
    sleep(0.9)
    check("let go: yaw handed back", not fc.overrides, str(fc.overrides))

    print("=== 4. bumpless: the pilot's own stick is the starting point ===")
    fc.radio[1] = 1690                  # pilot's thumb: ~+8 m/s of climb stick
    sleep(0.5)
    M.set_level("vert", "SLOW")
    M.request_hold("up", "b", perf_counter())
    sleep(0.03)
    first = fc.overrides.get(1)
    check("first ch1 override starts at the pilot's 1690, not at mid",
          first is not None and abs(first - 1690) <= 25, f"first ch1={first}")
    hold("up", 0.7, session="b")
    fc.radio[1] = 1500
    sleep(1.2)

    print("=== 5. ENGAGE: PUSH on all four, exact 30 deg ===")
    M.set_level("push", "MEDIUM")
    check("ENGAGE accepted in HOVER", M.request_engage() == "")
    sleep(1.5)
    o = dict(fc.overrides)
    check("PUSH overrides all four channels", set(o) == {1, 2, 3, 4}, str(o))
    check("ch3 decodes to 30 deg nose DOWN", 3 in o and abs(ap_norm(o[3]) * 40.0 + 30.0) < 0.2,
          f"ch3={o.get(3)} -> {ap_norm(o.get(3, 1500)) * 40:.1f} deg")
    check("heading and altitude held: ch4 1500, ch1 1500 (exact mid), roll 1500",
          o.get(4) == 1500 and o.get(1) == 1500 and o.get(2) == 1500, str(o))
    check("the FC echo confirms the override", c.override_confirmed is True)
    check("holds are refused during PUSH", M.request_hold("up", "z", perf_counter()) != "")

    print("=== 6. the detector confirms: AI, blended in ===")
    HALF_H = math.tan(math.radians(62.2 / 2))
    ex = 320.0 * math.tan(math.radians(8.0)) / HALF_H
    ch4 = []
    t_end = perf_counter() + 2.2
    while perf_counter() < t_end:
        cmd = c.compute(ex, 0.0, "LOCK", 640, 640, bbox=(0, 0, 20, 20),
                        capture_t=perf_counter() - 0.05)
        c.update(cmd, (int(ex), 0), have_face=True, detected=True, conf=0.55,
                 bbox=(0, 0, 20, 20))
        ch4.append(fc.overrides.get(4))
        sleep(0.03)
    check("handed over to the AI", M.phase == "AI", M.phase)
    s4 = [v for v in ch4 if v is not None]
    jumps = [abs(b - a) for a, b in zip(s4, s4[1:])]
    check("yaw glides from 1500 toward the target (right), no step > 15 us",
          s4 and s4[-1] > 1520 and max(jumps) <= 15,
          f"final ch4={s4[-1] if s4 else None}, max step {max(jumps) if jumps else 0} us")

    print("=== 7. ABORT, and the mode switch ===")
    M.request_abort()
    sleep(0.1)
    check("ABORT: every channel released within 0.1 s, HOVER",
          not fc.overrides and M.phase == "HOVER", f"{M.phase} {fc.overrides}")
    hold("up", 0.5, session="m")
    check("climbing again after ABORT (a new press)", set(fc.overrides) == {1})
    fc.mode = 0                                                   # STABILIZE
    sleep(0.5)
    check("pilot flips to STABILIZE mid-climb: all released, PILOT",
          not fc.overrides and M.phase == "PILOT", f"{M.phase} {fc.overrides}")

    print("=== 8. the checklist and the log ===")
    rows = c.mission_checklist()
    check("checklist flags RC_OVERRIDE_TIME 3 and ATC_INPUT_TC 0.15 for change",
          any(r.startswith("!! RC_OVERRIDE_TIME") for r in rows)
          and any(r.startswith("!! ATC_INPUT_TC") for r in rows)
          and any(r.startswith("ok PILOT_SPEED_UP") for r in rows), "\n     " + "\n     ".join(rows))
finally:
    with contextlib.redirect_stdout(io.StringIO()):
        c.close()
    fc.shutdown()

rows = list(csv.DictReader(open(c.log_path)))
phases = {r["phase"] for r in rows}
check("CSV: every row has exactly the header's columns", all(None not in r for r in rows)
      and all(r.get("cap_age_s") is not None for r in rows), f"{len(rows)} rows")
check("CSV: every phase flown is logged", {"PILOT", "HOVER", "CLIMB", "SCOUT", "PUSH", "AI"} <= phases,
      str(sorted(phases)))
cl = [r for r in rows if r["phase"] == "CLIMB"]
check("CSV: CLIMB rows show ch1 only, plus height and vibration",
      cl and all(r["pwm_roll"] == "0" and r["pwm_pitch"] == "0" for r in cl)
      and cl[-1]["height_m"] == "2.00" and cl[-1]["vibe_z"] == "12.00")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED:")
    for f in FAILED:
        print("  - " + f)
    sys.exit(1)
print("all passed")
