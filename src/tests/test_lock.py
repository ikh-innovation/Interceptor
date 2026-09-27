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
    errs, cmds, fc, m = fly(25.0, psi0=start, dist=d, seed=7)
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

print("=== 6. locked at 313: wind turns it, hands turn it, it comes back to 313 ===")


def fly313(secs, script, dist=None, seed=21):
    """FC model + Mission, locked at 313 from 300 via the degree scroll.
    script(t, m) is called every TX cycle (holds, taps); returns the heading
    trace [(t, psi)] and the mission."""
    rng = random.Random(seed)
    fc = SL.FcYaw(300.0)
    mp = SL.Mapper()
    m = M.Mission(mp, log=lambda *_a: None)
    ctx = dict(gate_ok=True, gate_why="", att=dict(yaw=math.radians(300.0)), height=6.0,
               climb=0.0, ai_cmd=None, ai_t=None, det_t=None, det_conf=None)
    m.cycle(0.0, ctx)
    m.request_lock(True, 313)
    hist, out, tr, t, nxt = [], {}, [], 0.0, 0.0
    while t < secs:
        hist.append((t, fc.psi))
        while hist and hist[0][0] < t - 0.02:
            hist.pop(0)
        if t >= nxt:
            nxt += 0.02
            script(t, m)
            ctx["att"] = dict(yaw=math.radians(hist[0][1] + rng.gauss(0, 0.3)))
            out = m.cycle(t, ctx)
            tr.append((t, fc.psi))
        stick = mp.pwm_to_phys("yaw", out["yaw"]) if "yaw" in out else 0.0
        fc.step(stick, dist(t) if dist else 0.0, SL.DT)
        t += SL.DT
    return tr, m


def at(tr, t0, t1):
    return [abs(SL.wrap180(p - 313.0)) for t, p in tr if t0 <= t < t1]


tr, m = fly313(8.0, lambda t, m: None)
check("from 300, lock at 313 (the scroll): there within 3 s, then within 1 deg",
      max(at(tr, 3.0, 8.0)) < 1.0 and abs(m.lock_hdg - 313.0) < 1e-6, f"worst after 3 s {max(at(tr, 3.0, 8.0)):.2f}")


# a gust stronger than the motors (the FC's own target is dragged): 60+ deg
def gust(t):
    return 400.0 if 5.0 <= t < 5.4 else 0.0


tr, m = fly313(20.0, lambda t, m: None, dist=gust)
check("a gust stronger than the motors swings it far off 313...", max(at(tr, 5.0, 7.0)) > 25.0,
      f"peak {max(at(tr, 5.0, 7.0)):.0f} deg")
check("...and LOCK brings it back to 313 within 1 deg, and it stays (no drift)",
      max(at(tr, 16.0, 20.0)) < 1.0, f"worst 16-20 s {max(at(tr, 16.0, 20.0)):.2f}")
fo = SL.FcYaw(313.0)
for i in range(int(15.0 / SL.DT)):
    fo.step(0.0, gust(i * SL.DT), SL.DT)
check("(the FC alone would stay wherever that gust left it)", abs(SL.wrap180(fo.psi - 313.0)) > 10.0,
      f"FC only ends {SL.wrap180(fo.psi - 313.0):+.0f} deg")


def steady(t):
    return 30.0 * math.sin(2 * math.pi * t / 7.0) + (60.0 if 4.0 <= t < 20.0 else 0.0)


tr, m = fly313(30.0, lambda t, m: None, dist=steady)
check("16 s of steady torque + a slow swell: stays within 3 deg, ends within 1 deg of 313",
      max(at(tr, 4.0, 30.0)) < 3.0 and max(at(tr, 26.0, 30.0)) < 1.0,
      f"worst {max(at(tr, 4.0, 30.0)):.2f}, end {max(at(tr, 26.0, 30.0)):.2f}")


def hands(t, m):
    if 4.0 <= t < 6.0:
        m.request_hold("yawl", "h1", t)             # ⟲ held 2 s
    elif 6.0 <= t < 6.02:
        m.request_release("h1")
    if 12.0 <= t < 12.02:
        m.request_step(+1)                          # a tap ⟳
    if 16.0 <= t < 18.0:
        m.request_hold("up", "h2", t)               # ▲ held 2 s
    elif 18.0 <= t < 18.02:
        m.request_release("h2")


tr, m = fly313(24.0, hands, dist=lambda t: 40.0 * math.sin(t * 1.3))
check("⟲ held 2 s while locked (gusty): it really turns away (> 20 deg)",
      max(at(tr, 5.5, 6.5)) > 20.0, f"{max(at(tr, 5.5, 6.5)):.0f} deg")
check("...let go: back at 313 within 1 deg by 5 s later, lock never moved",
      max(at(tr, 11.0, 12.0)) < 1.0 and abs(m.lock_hdg - 313.0) < 1e-6, f"{max(at(tr, 11.0, 12.0)):.2f}")
# the gusts here (40 deg/s^2, steady swing) wobble even the FC's own hold by
# ~2 deg; "back at 313" = back inside that wobble
check("a tap ⟳ while locked: back at 313 (inside the wind's own ±2.5 deg)",
      max(at(tr, 14.5, 16.0)) < 2.5, f"{max(at(tr, 14.5, 16.0)):.2f}")
check("▲ held while locked: the heading stays at 313 all the way up (±2.5)", max(at(tr, 16.0, 24.0)) < 2.5,
      f"{max(at(tr, 16.0, 24.0)):.2f}")


# the operator's rule: LOCK itself never steers quickly
def worst_cmd_step(cmds):
    return max(abs(b - a) for a, b in zip(cmds, cmds[1:]))


errs, cmds, fc, m = fly(25.0, psi0=1.0, dist=lambda t: 1000.0 if 3.0 <= t < 3.2 else 0.0, seed=7)
check(f"gentle: LOCK never commands more than {M.Mission.LOCK_MAX_DPS:.0f} deg/s, even 100 deg off",
      max(abs(c) for c in cmds) <= M.Mission.LOCK_MAX_DPS + 0.5, f"peak {max(abs(c) for c in cmds):.1f} deg/s")
check(f"gentle: its command never changes faster than {M.Mission.LOCK_ACCEL:.0f} deg/s^2 (no jerks)",
      # + two PWM steps (1 PWM = 0.5 deg/s here) for the rounding and a 22 ms cycle
      worst_cmd_step(cmds) <= M.Mission.LOCK_ACCEL * 0.022 + 2 * SL.RATE_MAX / 400.0,
      f"worst step {worst_cmd_step(cmds):.2f} deg/s per 20 ms")

print("=== 7. the phone's command path ===")
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
r_deg = tracker.DroneTracker._mission_command(st, "lock", "313")
st.ap.mission.cycle(0.08, dict(gate_ok=True, att=dict(yaw=0.9), height=6.0))
check("lock=313 (the degree scroll) locks at 313", r_deg == "" and st.ap.mission.lock_hdg is not None
      and abs(st.ap.mission.lock_hdg - 313.0) < 1e-6, f"{r_deg!r} {st.ap.mission.lock_hdg}")
check("lock=<junk> is refused, lock unchanged",
      tracker.DroneTracker._mission_command(st, "lock", "abc") != ""
      and abs(st.ap.mission.lock_hdg - 313.0) < 1e-6)
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
