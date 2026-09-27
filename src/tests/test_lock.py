#!/usr/bin/env python3
"""LOCK robustness: the cases a clean bench never shows.

    ../.env/yolo/bin/python tests/test_lock.py

Uses the real mission.Mission against sim_lock_wind's model of ArduPilot's
yaw hold (heading target, error limit, rate loop, finite yaw authority).
The end-to-end MAVLink path is in test_mission_mavlink.py section 7b; the
odds against gusts are in sim_lock_wind.py.
"""
import contextlib
import io
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import mission as M                                                # noqa: E402
import sim_lock_wind as SL                                         # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def fly(secs, psi0=0.0, lat=0.02, noise=0.3, seed=1, dist=None, blind=None, lock=True):
    """Hover with LOCK; returns (errors after 2 s, yaw commands, fc, mission).
    dist(t) -> deg/s^2 of yaw disturbance; blind(t) -> True while no ATTITUDE."""
    rng = random.Random(seed)
    fc = SL.FcYaw(psi0)
    mp = SL.Mapper()
    m = M.Mission(mp, log=lambda *_a: None)
    ctx = dict(gate_ok=True, gate_why="", att=dict(yaw=math.radians(psi0)), height=6.0,
               climb=0.0, ai_cmd=None, ai_t=None, det_t=None, det_conf=None)
    m.cycle(0.0, ctx)
    if lock:
        m.request_lock(True)
    hist, out, errs, cmds, t, nxt = [], {}, [], [], 0.0, 0.0
    lock_at = None
    while t < secs:
        hist.append((t, fc.psi))
        while hist and hist[0][0] < t - lat:
            hist.pop(0)
        if t >= nxt:
            nxt += 0.02
            if blind and blind(t):
                ctx["att"] = None
            else:
                ctx["att"] = dict(yaw=math.radians(hist[0][1] + rng.gauss(0, noise)))
            out = m.cycle(t, ctx)
            if lock_at is None and m.lock_hdg is not None:
                lock_at = m.lock_hdg
            cmds.append(SL.Mapper().pwm_to_phys("yaw", out["yaw"]) if "yaw" in out else 0.0)
        stick = mp.pwm_to_phys("yaw", out["yaw"]) if "yaw" in out else 0.0
        fc.step(stick, dist(t) if dist else 0.0, SL.DT)
        if t > 2.0 and int(t / SL.DT) % 25 == 0:
            errs.append(SL.wrap180(fc.psi - (lock_at if lock_at is not None else psi0)))
        t += SL.DT
    return errs, cmds, fc, m


def reversals(cmds):
    nz = [c for c in cmds if abs(c) > 0.5]
    return sum(1 for a, b in zip(nz, nz[1:]) if (a > 0) != (b > 0))


print("=== 1. a bad heading signal: 2 deg of noise, 150 ms late ===")
errs, cmds, fc, m = fly(60.0, lat=0.15, noise=2.0, seed=3)
sd = math.sqrt(sum(e * e for e in errs) / len(errs))
within = sum(1 for e in errs if abs(e) < 3.0) / len(errs)
check("no hunting: heading RMS under 1.5 deg, within 3 deg 99% of the time",
      sd < 1.5 and within > 0.99, f"RMS {sd:.2f} deg, within 3 deg {100 * within:.1f}%")
check("the yaw stick does not chatter (under 1 direction change per second)",
      reversals(cmds) < 60, f"{reversals(cmds)} reversals in 60 s")

print("=== 2. across north: locked at 359.8, noise straddling 0/360 ===")
errs, cmds, fc, m = fly(30.0, psi0=359.8, noise=1.0, seed=5)
check("never takes the long way round (worst error < 3 deg, peak command small)",
      max(abs(e) for e in errs) < 3.0 and max(abs(c) for c in cmds) < 10.0,
      f"worst {max(abs(e) for e in errs):.2f} deg, peak cmd {max(abs(c) for c in cmds):.1f} deg/s")
for start, kick in ((1.0, 40.0), (359.0, -40.0)):
    def d(t, k=kick):
        return k * 25.0 if 3.0 <= t < 3.2 else 0.0           # a sharp gust: ~40 deg in 0.2 s
    errs, cmds, fc, m = fly(15.0, psi0=start, dist=d, seed=7)
    after = [abs(e) for e in errs[-40:]]
    check(f"locked at {start:.0f}, a gust across north ({'+' if kick > 0 else '-'}): back within 1 deg",
          max(after) < 1.0, f"end error {max(after):.2f} deg")

print("=== 3. the heading lost for 5 s mid-LOCK, while a steady torque turns it ===")
errs, cmds, fc, m = fly(20.0, dist=lambda t: 25.0 if 4.0 <= t < 9.0 else 0.0,
                       blind=lambda t: 4.0 <= t < 9.0, seed=9)
blind_cmds = cmds[int(4.1 / 0.02):int(8.9 / 0.02)]
check("while blind: the yaw command is EXACTLY zero (never corrects on a stale heading)",
      all(c == 0.0 for c in blind_cmds), f"max |cmd| {max(abs(c) for c in blind_cmds):.2f}")
check("heading back: it returns to the lock and settles within 1 deg",
      max(abs(e) for e in errs[-40:]) < 1.0 and m.lock_hdg is not None,
      f"end error {max(abs(e) for e in errs[-40:]):.2f} deg")

print("=== 4. ten minutes locked in gusty Bf 3 (medium sensitivity) ===")
res = [SL.run((3, "medium", True, s, 600.0)) for s in (11, 12, 13)]
check("10 min: never walks away, ends within 2 deg on every run",
      all(r["final"] < 2.0 for r in res), str([round(r["final"], 2) for r in res]))
check("10 min: within 3 deg at least 98% of the time",
      all(r["in3"] > 0.98 for r in res), str([round(100 * r["in3"], 1) for r in res]))

print("=== 5. weaker yaw motors (80 deg/s^2 instead of 150), Bf 3 high sensitivity ===")
old = SL.ACC_MAX
SL.ACC_MAX = 80.0
fco = [SL.run((3, "high", False, s, 60.0)) for s in range(40, 52)]
lck = [SL.run((3, "high", True, s, 60.0)) for s in range(40, 52)]
SL.ACC_MAX = old
fo = sum(1 for r in fco if r["final"] > 10.0)
lo = sum(1 for r in lck if r["final"] > 10.0)
check("LOCK ends >10 deg off in fewer runs than the FC alone",
      lo < fo, f"LOCK {lo}/12 vs FC only {fo}/12")

print("=== 6. the phone's command path ===")
with contextlib.redirect_stdout(io.StringIO()):
    import tracker
    import mjpeg_stream


class _Stub:
    pass


st = _Stub()
st.ap = _Stub()
st.ap.mission = M.Mission(SL.Mapper(), log=lambda *_a: None)
st.MODES = tracker.DroneTracker.MODES
st.ap.mission.cycle(0.0, dict(gate_ok=True, att=dict(yaw=0.0), height=6.0))
r_on = tracker.DroneTracker._mission_command(st, "lock", "on")
st.ap.mission.cycle(0.02, dict(gate_ok=True, att=dict(yaw=0.5), height=6.0))
on_hdg = st.ap.mission.lock_hdg
r_on2 = tracker.DroneTracker._mission_command(st, "lock", "on")
st.ap.mission.cycle(0.04, dict(gate_ok=True, att=dict(yaw=0.9), height=6.0))
check("lock=on reaches the mission and locks; a repeated lock=on does NOT move it",
      r_on == "" and r_on2 == "" and on_hdg is not None
      and abs(st.ap.mission.lock_hdg - on_hdg) < 1e-6,
      f"{on_hdg} -> {st.ap.mission.lock_hdg}")
tracker.DroneTracker._mission_command(st, "lock", "off")
st.ap.mission.cycle(0.06, dict(gate_ok=True, att=dict(yaw=0.9), height=6.0))
check("lock=off unlocks", st.ap.mission.lock_hdg is None)
check("lock needs the controller lease and the PIN, like every move command",
      "lock" in mjpeg_stream.MjpegServer.CONTROL_KEYS if hasattr(mjpeg_stream, "MjpegServer")
      else any("lock" in getattr(v, "CONTROL_KEYS", ()) for v in vars(mjpeg_stream).values()))

print()
if FAILED:
    print(f"{len(FAILED)} FAILED:")
    for x in FAILED:
        print("  - " + x)
    sys.exit(1)
print("all passed")
