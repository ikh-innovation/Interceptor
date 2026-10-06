#!/usr/bin/env python3
"""AUTO against enemy drones ABOVE us: coming at us, or going away.

    ../.env/yolo/bin/python tests/sim_auto_scenarios.py              # every scenario
    ../.env/yolo/bin/python tests/sim_auto_scenarios.py --seeds 60   # more runs each
    ../.env/yolo/bin/python tests/sim_auto_scenarios.py --only come_straight,go_straight
    ../.env/yolo/bin/python tests/sim_auto_scenarios.py --list       # what each one is
    ../.env/yolo/bin/python tests/sim_auto_scenarios.py --json out.json --traces 2

Not pass/fail: a campaign that prints the odds, scenario by scenario, and WHY
each miss missed.  Every enemy starts ABOVE us (6-45 m higher) and either
COMES at us (straight, past our side, angled, overhead, diving, climbing,
weaving, homing on us, stopping short) or GOES away (straight, angled,
climbing, turning, weaving, fleeing once we close in).  AUTO is pressed at
t = 0, just off the ground (1.5 m) or from a hover (5-10 m), and the run ends
at a hit, a crash, or when AUTO has handed back to the pilot.

REAL and MODELLED are as in tests/sim_sky_wind.py, whose airframe (total tilt
held to ANGLE_MAX), wind (Bf 0-3), camera (30 deg up, 62.2 x 48.8 deg),
detector (25 Hz, 120 ms late, misses that rise as the target shrinks) and
barometer this reuses.  The Mission (AUTO_CLIMB -> AUTO_HOLD -> AUTO_ALIGN ->
PUSH -> AI -> STOP) and the InterceptLaw, built as start-wasp.sh builds it,
with the tracker's AI levels, are the code that flies.  A HIT is a closest
approach under 0.5 m between centres (TOUCH_M).

Why a miss missed (the first that fits):
  crash        it hit the ground
  never_seen   no confirmed detection, the whole run
  no_go        seen, but AUTO never flew the second line
  lost_line    flew the line, but the AI never got to steer
  lost_chase   the AI steered, then lost it for good (> 3 m away)
  near_miss    the AI steered and passed within 3 m, but not 0.5 m
"""
import argparse
import contextlib
import io
import json
import math
import os
import random
import sys
import types
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

# The law is built from tracker.DroneTracker's AI levels (MODES / set_mode),
# which never need the detector.  Importing ultralytics costs ~6 s and ~400 MB
# per process on the Jetson (next to the live tracker), so it is stubbed here.
if "ultralytics" not in sys.modules:
    _stub = types.ModuleType("ultralytics")
    _stub.YOLO = None
    sys.modules["ultralytics"] = _stub

import sim_sky_wind as S      # noqa: E402  the airframe, wind, camera, law

FW, FH, HFOV, VFOV = S.FW, S.FH, S.HFOV, S.VFOV_LAW
FOCAL = (FW / 2) / math.tan(math.radians(HFOV / 2))
T_END = 50.0
TRACE_DT = 0.1
TARGET_TIMEOUT_S = 0.5        # AcroCommander(target_timeout=0.5)

# Stress, for the robustness pass (all off by default, so the yardstick is the
# yardstick).  SIM_DET_SCALE: the detector finds it this much less often.
# SIM_LAT: detection latency, s.  SIM_MOUNT_ERR: the REAL camera tilt minus the
# 30 deg the law is told, deg.  SIM_TARGET_W: the REAL enemy width, m (the law
# assumes 0.30).  SIM_ANGLE_MAX: the FC's ANGLE_MAX, deg (this FC read 30 on
# 2026-10-06; the checklist asks for 40).  SIM_BF: every run in this Beaufort
# (4 = 5.5-7.9 m/s, gusting).
DET_SCALE = float(os.environ.get("SIM_DET_SCALE", "1.0"))
DET_LAT = float(os.environ.get("SIM_LAT", "0.12"))
FORCE_BF = os.environ.get("SIM_BF", "")
S.BEAUFORT.setdefault(4, (6.7, 1.8))


def apply_stress():
    """Module-level stress knobs, set in the parent before the law is built
    and the workers fork."""
    if os.environ.get("SIM_MOUNT_ERR"):
        S.TILT = 30.0 + float(os.environ["SIM_MOUNT_ERR"])
    if os.environ.get("SIM_TARGET_W"):
        S.TARGET_W = float(os.environ["SIM_TARGET_W"])
    if os.environ.get("SIM_ANGLE_MAX"):
        S.ANGLE_MAX = math.radians(float(os.environ["SIM_ANGLE_MAX"]))


# ── our airframe ─────────────────────────────────────────────────────────────
class Airframe(S.Drone):
    """sim_sky_wind's airframe, plus ALT_HOLD's PILOT_ACCEL_Z: the FC moves
    its climb-rate target at most this fast (1500 cm/s^2 on this FC, read at
    connect on 2026-10-06), so a climb ordered at once is not flown at once.
    Climbing hard is most of what an intercept above us asks for; without
    this the sim would credit climbs the FC never gives."""
    ACCEL_Z = float(os.environ.get("SIM_ACCEL_Z", "15.0"))

    def step(self, lean_c, bank_c, yr_c, climb_c, wind, dt, rng):
        tgt = getattr(self, "_climb_tgt", 0.0)
        d = self.ACCEL_Z * dt
        self._climb_tgt = tgt + max(-d, min(d, climb_c - tgt))
        super().step(lean_c, bank_c, yr_c, self._climb_tgt, wind, dt, rng)


# ── the enemy ────────────────────────────────────────────────────────────────
class Enemy:
    """A point flying a velocity law, integrated as the sim asks for it.
    vel(t, p, own, st) -> (vx, vy, vz); `own` is OUR position (for the ones
    that home on us or run once we are close), `st` a dict it may keep."""

    def __init__(self, p0, vel):
        self.p = list(p0)
        self.t = 0.0
        self.vel = vel
        self.st = {}

    def at(self, t, own):
        while self.t < t - 1e-9:
            h = min(0.01, t - self.t)
            v = self.vel(self.t, self.p, own, self.st)
            for i in range(3):
                self.p[i] += v[i] * h
            self.t += h
        return tuple(self.p)


def _bob_v(t):
    """A hovering drone is never quite still: the velocity of a ~0.3 m wobble."""
    return -0.21 * math.sin(0.7 * t), -0.15 * math.cos(0.5 * t), -0.165 * math.sin(1.1 * t)


def v_straight(heading, speed, vz=0.0, z_lo=None, z_hi=None):
    def vel(t, p, own, st):
        w = vz
        if (z_lo is not None and p[2] <= z_lo and w < 0) or (z_hi is not None and p[2] >= z_hi and w > 0):
            w = 0.0
        return speed * math.cos(heading), speed * math.sin(heading), w
    return vel


def v_weave(heading, speed, swing, period, vz=0.0):
    """S-turns at a constant speed: the heading swings +/- `swing` rad."""
    w = 2 * math.pi / period

    def vel(t, p, own, st):
        h = heading + swing * math.sin(w * t)
        return speed * math.cos(h), speed * math.sin(h), vz
    return vel


def v_turn(heading, speed, t_turn, turn, rate=math.radians(30.0), vz=0.0):
    """Straight, then a steady turn of `turn` rad at `rate`, then straight."""
    dur = abs(turn) / rate

    def vel(t, p, own, st):
        h = heading + turn * max(0.0, min(1.0, (t - t_turn) / dur))
        return speed * math.cos(h), speed * math.sin(h), vz
    return vel


def v_homing(speed):
    """Flies at where we ARE now, at its own height: an attacker that passes
    over us if nothing stops it."""
    def vel(t, p, own, st):
        h = math.atan2(own[1] - p[1], own[0] - p[0])
        if math.hypot(own[0] - p[0], own[1] - p[1]) < 2.0:
            h = st.get("h", h)                  # overhead: carry straight on
        st["h"] = h
        return speed * math.cos(h), speed * math.sin(h), 0.0
    return vel


def v_stop(heading, speed, stop_after, decel=3.0):
    """Flies in, brakes after `stop_after` m and hovers there."""
    def vel(t, p, own, st):
        d = st.setdefault("d", 0.0)
        v = st.setdefault("v", speed)
        if d >= stop_after:
            v = max(0.0, v - decel * 0.01)
        st["d"] = d + v * 0.01
        st["v"] = v
        if v <= 0.0:
            return _bob_v(t)
        return v * math.cos(heading), v * math.sin(heading), 0.0
    return vel


def v_flee(trigger_m, speed, accel=4.0, vz=1.0):
    """Hovers until we come within `trigger_m`, then runs straight away from
    us (climbing a little), up to `speed`."""
    def vel(t, p, own, st):
        if "h" not in st:
            if math.dist(own, p) > trigger_m:
                return _bob_v(t)
            st["h"] = math.atan2(p[1] - own[1], p[0] - own[0])
            st["v"] = 0.0
        st["v"] = min(speed, st["v"] + accel * 0.01)
        return st["v"] * math.cos(st["h"]), st["v"] * math.sin(st["h"]), vz
    return vel


def v_hover():
    def vel(t, p, own, st):
        return _bob_v(t)
    return vel


# ── the scenarios ────────────────────────────────────────────────────────────
# Each picks, from its own random stream: where the enemy starts (horizontal
# range r, bearing off our nose az, height ABOVE our start dz) and how it flies.
# `to_us` is the bearing from the enemy to us, `away` its reverse.
def _sc(family, desc, place, motion, speeds):
    return dict(family=family, desc=desc, place=place, motion=motion, speeds=speeds)


SCENARIOS = {
    # ── COMING AT US ─────────────────────────────────────────────────────────
    "come_straight": _sc(
        "come", "flies straight at us, level, 8-40 m above, from 55-85 m",
        lambda r: dict(r=r.uniform(55, 85), az=r.uniform(-15, 15), dz=r.uniform(8, 40)),
        lambda r, sp, to_us: v_straight(to_us, sp), (5.0, 10.0, 15.0)),
    "come_beside": _sc(
        "come", "flies in on a line 8-25 m to our side (a fly-by), 8-35 m above",
        lambda r: dict(r=r.uniform(55, 85), az=r.uniform(-12, 12), dz=r.uniform(8, 35),
                       offset=r.uniform(8, 25) * r.choice((-1, 1))),
        lambda r, sp, to_us: v_straight(to_us, sp), (5.0, 10.0, 15.0)),
    "come_angled": _sc(
        "come", "flies in 30-60 deg off the line to us, 8-35 m above",
        lambda r: dict(r=r.uniform(50, 75), az=r.uniform(-15, 15), dz=r.uniform(8, 35)),
        lambda r, sp, to_us: v_straight(to_us + math.radians(r.uniform(30, 60)) * r.choice((-1, 1)), sp),
        (5.0, 10.0)),
    "come_overhead": _sc(
        "come", "flies straight over us, HIGH: 30-45 m above",
        lambda r: dict(r=r.uniform(55, 85), az=r.uniform(-15, 15), dz=r.uniform(30, 45)),
        lambda r, sp, to_us: v_straight(to_us, sp), (5.0, 10.0)),
    "come_dive": _sc(
        "come", "comes at us descending 2-4 m/s, levels off 4-8 m above our start",
        lambda r: dict(r=r.uniform(50, 75), az=r.uniform(-15, 15), dz=r.uniform(18, 35),
                       level=r.uniform(4, 8)),
        None, (8.0, 12.0)),
    "come_climb": _sc(
        "come", "comes at us climbing 1-2.5 m/s, starting 8-25 m above",
        lambda r: dict(r=r.uniform(55, 80), az=r.uniform(-15, 15), dz=r.uniform(8, 25)),
        lambda r, sp, to_us: v_straight(to_us, sp, vz=r.uniform(1.0, 2.5)), (6.0, 10.0)),
    "come_weave": _sc(
        "come", "comes at us in S-turns (+/-30-50 deg, every 3-5 s), 8-35 m above",
        lambda r: dict(r=r.uniform(50, 75), az=r.uniform(-15, 15), dz=r.uniform(8, 35)),
        lambda r, sp, to_us: v_weave(to_us, sp, math.radians(r.uniform(30, 50)), r.uniform(3, 5)),
        (6.0, 10.0)),
    "come_homing": _sc(
        "come", "homes on where we are now, level, 8-35 m above (passes over us)",
        lambda r: dict(r=r.uniform(55, 80), az=r.uniform(-15, 15), dz=r.uniform(8, 35)),
        lambda r, sp, to_us: v_homing(sp), (6.0, 10.0, 14.0)),
    "come_stop": _sc(
        "come", "flies in at 6-10 m/s and stops 20-40 m short, hovering, 8-35 m above",
        lambda r: dict(r=r.uniform(55, 80), az=r.uniform(-15, 15), dz=r.uniform(8, 35),
                       stop=r.uniform(20, 40)),
        None, (6.0, 10.0)),
    # ── GOING AWAY ───────────────────────────────────────────────────────────
    "go_straight": _sc(
        "go", "flies straight away, level, 15-40 m out, 6-30 m above",
        lambda r: dict(r=r.uniform(15, 40), az=r.uniform(-15, 15), dz=r.uniform(6, 30)),
        lambda r, sp, to_us: v_straight(to_us + math.pi, sp), (3.0, 6.0, 9.0, 12.0)),
    "go_angled": _sc(
        "go", "flies away 30-60 deg off straight away, 15-40 m out, 6-30 m above",
        lambda r: dict(r=r.uniform(15, 40), az=r.uniform(-15, 15), dz=r.uniform(6, 30)),
        lambda r, sp, to_us: v_straight(to_us + math.pi + math.radians(r.uniform(30, 60)) * r.choice((-1, 1)), sp),
        (5.0, 9.0)),
    "go_climb": _sc(
        "go", "flies away climbing 1-3 m/s, 15-35 m out, 6-25 m above",
        lambda r: dict(r=r.uniform(15, 35), az=r.uniform(-15, 15), dz=r.uniform(6, 25)),
        lambda r, sp, to_us: v_straight(to_us + math.pi, sp, vz=r.uniform(1.0, 3.0)), (5.0, 8.0)),
    "go_turn": _sc(
        "go", "flies away, then turns 60-120 deg after 2-5 s, 15-35 m out, 6-25 m above",
        lambda r: dict(r=r.uniform(15, 35), az=r.uniform(-15, 15), dz=r.uniform(6, 25)),
        lambda r, sp, to_us: v_turn(to_us + math.pi, sp, r.uniform(2, 5),
                                    math.radians(r.uniform(60, 120)) * r.choice((-1, 1))),
        (6.0, 9.0)),
    "go_weave": _sc(
        "go", "flies away in S-turns (+/-30-50 deg, every 3-5 s), 15-35 m out",
        lambda r: dict(r=r.uniform(15, 35), az=r.uniform(-15, 15), dz=r.uniform(6, 25)),
        lambda r, sp, to_us: v_weave(to_us + math.pi, sp, math.radians(r.uniform(30, 50)), r.uniform(3, 5)),
        (5.0, 8.0)),
    "go_flee": _sc(
        "go", "hovers 25-45 m out, then RUNS (to 8-12 m/s) once we are within 15-25 m",
        lambda r: dict(r=r.uniform(25, 45), az=r.uniform(-15, 15), dz=r.uniform(6, 30),
                       trigger=r.uniform(15, 25)),
        None, (8.0, 12.0)),
    # ── for reference ────────────────────────────────────────────────────────
    "hover_above": _sc(
        "other", "hovers 20-50 m out, 6-35 m above",
        lambda r: dict(r=r.uniform(20, 50), az=r.uniform(-15, 15), dz=r.uniform(6, 35)),
        lambda r, sp, to_us: v_hover(), (0.0,)),
    "cross_above": _sc(
        "other", "crosses in front, 30-60 m out, 8-30 m above",
        lambda r: dict(r=r.uniform(30, 60), az=r.uniform(-15, 15), dz=r.uniform(8, 30)),
        lambda r, sp, to_us: v_straight(to_us + math.pi / 2 * r.choice((-1, 1)), sp), (4.0, 8.0)),
    "side_come": _sc(
        "other", "as come_straight at 5-10 m/s, but starting 45-110 deg off our nose",
        lambda r: dict(r=r.uniform(50, 75), az=r.uniform(45, 110) * r.choice((-1, 1)), dz=r.uniform(8, 35)),
        lambda r, sp, to_us: v_straight(to_us, sp), (5.0, 10.0)),
    "side_go": _sc(
        "other", "as go_straight at 3-6 m/s, but starting 45-110 deg off our nose",
        lambda r: dict(r=r.uniform(15, 35), az=r.uniform(45, 110) * r.choice((-1, 1)), dz=r.uniform(6, 25)),
        lambda r, sp, to_us: v_straight(to_us + math.pi, sp), (3.0, 6.0)),
}


def build_enemy(name, speed, r, p0, own_p0, pl):
    sc = SCENARIOS[name]
    to_us = math.atan2(own_p0[1] - p0[1], own_p0[0] - p0[0])
    if name == "come_dive":
        vel = v_straight(to_us, speed, vz=-r.uniform(2.0, 4.0), z_lo=own_p0[2] + pl["level"])
    elif name == "come_stop":
        vel = v_stop(to_us, speed, max(0.0, pl["r"] - pl["stop"]))
    elif name == "go_flee":
        vel = v_flee(pl["trigger"], speed)
    else:
        vel = sc["motion"](r, speed, to_us)
    return Enemy(p0, vel)


# ── one engagement ───────────────────────────────────────────────────────────
def run_one(job):
    """job = (scenario, speed, start, bf, seed, trace).  Mirrors
    sim_sky_wind.engage with AUTO pressed at t = 0."""
    name, speed, start, bf, seed, want_trace = job
    bf = int(FORCE_BF) if FORCE_BF else bf
    import mission as M
    rng = random.Random(seed)
    law = S.build_law_cached()
    mp = S.Mapper()
    wind = S.Wind(bf, rng)
    m = M.Mission(mp, log=lambda *_a: None, accel_z=2.5, jerk_z=5.0)
    lv_law = "MEDIUM"
    m.set_level("ai", lv_law)
    for kv in filter(None, os.environ.get("SIM_AUTOP", "").split(",")):
        k, _, v = kv.partition("=")
        assert m.set_auto_param(k, v) == "", kv

    # pilot hovering in ALT_HOLD, trimmed into the MEAN wind (as sim_sky_wind)
    wx, wy = wind.mean * math.cos(wind.dir), wind.mean * math.sin(wind.dir)
    yaw0 = rng.uniform(-math.pi, math.pi)
    trim_tilt = math.atan(S.Drone.DRAG * wind.mean * wind.mean / S.G)
    f = (math.cos(yaw0), math.sin(yaw0))
    rr = (-math.sin(yaw0), math.cos(yaw0))
    if wind.mean > 0:
        ux, uy = -wx / wind.mean, -wy / wind.mean
        lean0 = -trim_tilt * (ux * f[0] + uy * f[1])
        bank0 = trim_tilt * (ux * rr[0] + uy * rr[1])
    else:
        lean0 = bank0 = 0.0
    lean0 += math.radians(rng.gauss(0, 1.0))
    bank0 += math.radians(rng.gauss(0, 1.0))
    h0 = 1.5 if start == "ground" else rng.uniform(5.0, 10.0)
    dr = Airframe((0.0, 0.0, h0), yaw0, lean0, bank0, v0=(rng.gauss(0, 0.3), rng.gauss(0, 0.3)))
    mp.pilot = {"pitch": mp.phys_to_pwm("pitch", math.degrees(lean0)),
                "roll": mp.phys_to_pwm("roll", math.degrees(bank0)),
                "yaw": 1500, "throttle": 1500}

    # the enemy: r m out (horizontally), az deg off our nose, dz m above us
    pl = SCENARIOS[name]["place"](rng)
    a = yaw0 + math.radians(pl["az"])
    p0 = [pl["r"] * math.cos(a), pl["r"] * math.sin(a), h0 + pl["dz"]]
    if "offset" in pl:              # its line passes `offset` m to our side
        p0[0] += pl["offset"] * -math.sin(a)
        p0[1] += pl["offset"] * math.cos(a)
    own0 = (0.0, 0.0, h0)
    if "offset" in pl:
        aim = (pl["offset"] * -math.sin(a), pl["offset"] * math.cos(a), h0)
        enemy = build_enemy(name, speed, rng, p0, aim, pl)
    else:
        enemy = build_enemy(name, speed, rng, p0, own0, pl)

    dt = 0.005
    det_period, det_lat = 1 / 25.0, DET_LAT
    pending = []
    ctx = dict(gate_ok=True, gate_why="", att=dict(roll=dr.bank, pitch=dr.lean, yaw=dr.yaw),
               height=h0, climb=0.0, ai_cmd=None, ai_t=None, det_t=None, det_conf=None)
    next_det, next_tx, t_last_det = 0.0, 0.0, None
    best, t_best, crashed = 1e9, None, False
    rel_best = (0.0, 0.0, 0.0)
    m.cycle(0.0, ctx)                               # PILOT -> HOVER
    m.set_auto_enabled(True)
    assert m.request_auto(True) == ""
    out = {}
    t = 0.0
    ev = dict(t_seen=None, t_align=None, t_go=None, t_ai=None, t_stop=None)
    n_align = n_go = 0
    phase_prev = None
    det_after_go = 0
    frames_after_go = in_after_go = 0
    h_max, h_min = h0, h0
    trace = [] if want_trace else None
    next_trace = 0.0
    fresh_det = False
    last_cmd = None
    tgt = enemy.at(0.0, dr.p)
    while t < T_END:
        tgt = enemy.at(t, dr.p)
        d = math.dist(tgt, dr.p)
        if d < best:
            best, t_best = d, t
            fw_, rt_ = dr.fwd(), dr.right()
            dx, dy, dz = (tgt[0] - dr.p[0], tgt[1] - dr.p[1], tgt[2] - dr.p[2])
            rel_best = (dx * fw_[0] + dy * fw_[1], dx * rt_[0] + dy * rt_[1], dz)
        if best < S.TOUCH_M:
            break
        if dr.p[2] < 0.3:
            crashed = True
            break
        h_max, h_min = max(h_max, dr.p[2]), min(h_min, dr.p[2])
        # camera frame
        if t >= next_det:
            next_det += det_period
            meas = dr.measure(tgt, VFOV, HFOV)
            att = dict(roll=dr.bank, pitch=dr.lean, yaw=dr.yaw, yawspeed=dr.yaw_rate,
                       pitchspeed=0.0, rollspeed=0.0)
            inside = False
            if meas is not None:
                ex, ey, rng_m = meas
                inside = abs(ex) <= FW / 2 and abs(ey) <= FH / 2
                if inside:
                    bw = FOCAL * S.TARGET_W / max(0.2, rng_m)
                    p_det = 0.93 * max(0.0, min(1.0, (bw - 4.0) / 10.0)) * DET_SCALE
                    if bw > 0.6 * FH:
                        p_det *= 0.2
                    if rng.random() < p_det:
                        conf = min(0.95, 0.25 + 0.03 * bw) * rng.uniform(0.8, 1.0)
                        nx = rng.gauss(0, 2.0 + 0.04 * bw)
                        ny = rng.gauss(0, 2.0 + 0.04 * bw)
                        bwn = bw * rng.uniform(0.85, 1.15)
                        pending.append((t + det_lat + rng.uniform(-0.02, 0.03),
                                        t, ex + nx, ey + ny, bwn, conf, att))
            if ev["t_go"] is not None and d > 3.0:
                frames_after_go += 1
                in_after_go += inside
        while pending and pending[0][0] <= t:
            arr, cap, ex, ey, bwn, conf, att = pending.pop(0)
            ddt = det_period if t_last_det is None else max(1e-3, arr - t_last_det)
            t_last_det = arr
            last_cmd = law.compute(ex, ey, FW, FH, dt=ddt, bbox_w=bwn, bbox_h=bwn,
                                   att=att, capture_age=arr - cap)
            ctx.update(det_t=t, det_conf=conf)
            if ev["t_go"] is not None:
                det_after_go += 1
        # TX thread, 50 Hz
        if t >= next_tx:
            next_tx += 0.02
            # exactly as AcroCommander._mission_tx: the AI command only while
            # a REAL detection is younger than target_timeout, else None
            fresh_ai = ctx["det_t"] is not None and t - ctx["det_t"] <= TARGET_TIMEOUT_S
            ctx.update(ai_cmd=last_cmd if fresh_ai else None,
                       ai_t=ctx["det_t"] if fresh_ai else None)
            ctx["height"] = dr.p[2] + rng.gauss(0, 0.25)
            ctx["climb"] = dr.v[2] + rng.gauss(0, 0.2)
            ctx["att"] = dict(roll=dr.bank, pitch=dr.lean, yaw=dr.yaw + rng.gauss(0, 0.01))
            if ev["t_seen"] is None and m._confirmed(t, ctx):
                ev["t_seen"] = t
            fresh_det = ctx["det_t"] is not None and t - ctx["det_t"] <= 0.25
            out = m.cycle(t, ctx)
            ph = m.phase
            if ph != phase_prev:
                if ph == "AUTO_ALIGN":
                    n_align += 1
                    ev["t_align"] = ev["t_align"] if ev["t_align"] is not None else t
                if ph == "PUSH" and phase_prev != "AI":
                    n_go += 1               # a line flown (not a re-acquire)
                if ph == "PUSH" and ev["t_go"] is None:
                    ev["t_go"] = t
                if ph == "STOP" and ev["t_stop"] is None:
                    ev["t_stop"] = t
                phase_prev = ph
            if ev["t_ai"] is None and (m._push_steer or ph == "AI"):
                ev["t_ai"] = t
            lv = m.ai_level_now()
            if lv != lv_law:                        # the tracker's per-frame sync
                S.set_level(law, lv)
                lv_law = lv
            if ph == "HOVER" and t > 0.5:
                break                               # AUTO handed back: over
        wv = wind.step(t, dt)
        # a channel AUTO does not hold stays the pilot's (the trim into the wind)
        lean = math.radians(mp.pwm_to_phys("pitch", out.get("pitch", mp.pilot["pitch"])))
        bank = math.radians(mp.pwm_to_phys("roll", out.get("roll", mp.pilot["roll"])))
        yr = math.radians(mp.pwm_to_phys("yaw", out.get("yaw", 1500)))
        cl = mp.pwm_to_phys("throttle", out.get("throttle", 1500))
        if trace is not None and t >= next_trace:
            next_trace += TRACE_DT
            fw_, rt_ = dr.fwd(), dr.right()
            dx, dy, dz = (tgt[0] - dr.p[0], tgt[1] - dr.p[1], tgt[2] - dr.p[2])
            hz = math.hypot(dx, dy)
            az_t = math.degrees(math.atan2(dx * rt_[0] + dy * rt_[1], dx * fw_[0] + dy * fw_[1]))
            el_c = math.degrees(math.atan2(dz, max(1e-6, hz))) - math.degrees(dr.lean) - S.TILT
            trace.append([round(t, 2), round(dr.p[0], 2), round(dr.p[1], 2), round(dr.p[2], 2),
                          round(math.degrees(dr.yaw), 1), round(math.degrees(dr.lean), 1),
                          round(math.degrees(dr.bank), 1), round(tgt[0], 2), round(tgt[1], 2),
                          round(tgt[2], 2), M.PHASES.index(m.phase), int(fresh_det),
                          round(az_t, 1), round(el_c, 1), round(math.hypot(dr.v[0], dr.v[1]), 1)])
        dr.step(lean, bank, yr, cl, wv, dt, rng)
        t += dt
    hit = best < S.TOUCH_M
    if hit:
        why = "hit"
    elif crashed:
        why = "crash"
    elif ev["t_seen"] is None:
        why = "never_seen"
    elif ev["t_go"] is None:
        why = "no_go"
    elif ev["t_ai"] is None:
        why = "lost_line"
    elif best < 3.0:
        why = "near_miss"
    else:
        why = "lost_chase"
    res = dict(scenario=name, family=SCENARIOS[name]["family"], speed=speed, start=start, bf=bf,
               seed=seed, hit=hit, why=why, best=round(best, 3), t_best=t_best,
               rel_best=[round(x, 2) for x in rel_best], crashed=crashed, end_phase=m.phase,
               t_end=round(t, 2), n_align=n_align, n_go=n_go, det_after_go=det_after_go,
               in_frame_after_go=(in_after_go / frames_after_go) if frames_after_go else None,
               h0=round(h0, 2), h_max=round(h_max, 1), h_min=round(h_min, 1),
               r0=round(pl["r"], 1), az0=round(pl["az"], 1), dz0=round(pl["dz"], 1),
               **{k: (None if v is None else round(v, 2)) for k, v in ev.items()})
    if trace is not None:
        res["trace"] = trace
    return res


# ── the campaign ─────────────────────────────────────────────────────────────
def jobs(names, n_seeds, n_traces, master_seed=2026):
    out = []
    for name in names:
        sc = SCENARIOS[name]
        rng = random.Random(f"{master_seed}:{name}")
        for i in range(n_seeds):
            for sp in sc["speeds"]:
                # every wind from the ground, then every wind from a hover
                start = "ground" if (i // 4) % 2 == 0 else "hover"
                out.append((name, sp, start, i % 4, rng.randrange(10 ** 9), i < n_traces))
    return out


def pct(xs):
    xs = list(xs)
    return 100.0 * sum(xs) / max(1, len(xs))


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else float("nan")


WHYS = ("never_seen", "no_go", "lost_line", "lost_chase", "near_miss", "crash")


def report(res):
    print(f"\n=== AUTO vs drones ABOVE us: {len(res)} engagements, hit = closest approach "
          f"< {S.TOUCH_M} m ===")
    hdr = f"  {'scenario':<15s}{'m/s':>5s}{'n':>5s}{'HIT':>6s}{'<1.5m':>7s}{'miss':>6s}{'t hit':>7s}  " \
          + "".join(f"{w:>11s}" for w in WHYS)
    print(hdr)
    names = []
    for r in res:
        if r["scenario"] not in names:
            names.append(r["scenario"])
    fam_rows = {}
    for name in names:
        rs = [r for r in res if r["scenario"] == name]
        fam_rows.setdefault(rs[0]["family"], []).extend(rs)
        for sp in sorted({r["speed"] for r in rs}):
            rr = [r for r in rs if r["speed"] == sp]
            print(f"  {name:<15s}{sp:5.0f}{len(rr):5d}{pct(r['hit'] for r in rr):5.0f}%"
                  f"{pct(r['best'] < 1.5 for r in rr):6.0f}%{med(r['best'] for r in rr):6.2f}"
                  f"{med(r['t_best'] for r in rr if r['hit']):6.1f}s  "
                  + "".join(f"{pct(r['why'] == w for r in rr):10.0f}%" for w in WHYS))
    print("  " + "-" * (len(hdr) - 2))
    for fam in ("come", "go", "other"):
        rs = fam_rows.get(fam, [])
        if rs:
            print(f"  {fam.upper() + ' (all)':<20s}{len(rs):5d}{pct(r['hit'] for r in rs):5.0f}%"
                  f"{pct(r['best'] < 1.5 for r in rs):6.0f}%{med(r['best'] for r in rs):6.2f}"
                  f"{med(r['t_best'] for r in rs if r['hit']):6.1f}s  "
                  + "".join(f"{pct(r['why'] == w for r in rs):10.0f}%" for w in WHYS))
    cg = [r for r in res if r["family"] in ("come", "go")]
    print(f"\n  COME + GO: HIT {pct(r['hit'] for r in cg):.1f}%   ALL: HIT {pct(r['hit'] for r in res):.1f}%  "
          f"(n {len(cg)} / {len(res)})")
    for st in ("ground", "hover"):
        rs = [r for r in cg if r["start"] == st]
        if rs:
            print(f"    from the {st:<6s}: HIT {pct(r['hit'] for r in rs):5.1f}%  (n {len(rs)})")
    for b in (0, 1, 2, 3):
        rs = [r for r in cg if r["bf"] == b]
        if rs:
            print(f"    Bf{b}: HIT {pct(r['hit'] for r in rs):5.1f}%  (n {len(rs)})", end="")
    print()
    print(f"    crashed: {sum(r['crashed'] for r in res)} of {len(res)}")


def compare(old, new):
    """Hit rate per scenario and speed, before -> after (same jobs: same seeds)."""
    print("\n=== CHANGE vs --compare (hit %) ===")
    keys = []
    for r in new:
        if (r["scenario"], r["speed"]) not in keys:
            keys.append((r["scenario"], r["speed"]))
    for k in keys:
        a = [r["hit"] for r in old if (r["scenario"], r["speed"]) == k]
        b = [r["hit"] for r in new if (r["scenario"], r["speed"]) == k]
        if a:
            d = pct(b) - pct(a)
            print(f"  {k[0]:<15s}{k[1]:4.0f} m/s  {pct(a):5.0f}% -> {pct(b):5.0f}%  {d:+5.0f}"
                  + ("   <-- worse" if d <= -10 else ""))
    print(f"  SCORE {score(old):.1f} -> {score(new):.1f}")


def score(res):
    """One number to compare versions: the hit rate over COME + GO."""
    cg = [r for r in res if r["family"] in ("come", "go")]
    return pct(r["hit"] for r in cg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24,
                    help="runs per scenario and speed (a multiple of 8 balances wind x start)")
    ap.add_argument("--quick", action="store_true", help="8 runs per scenario and speed")
    ap.add_argument("--only", default="", help="comma-separated scenario names (or come / go / other)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", default="", help="write every result here")
    ap.add_argument("--traces", type=int, default=0,
                    help="record a 10 Hz trace of the first N runs of each scenario and speed")
    ap.add_argument("--set", action="append", default=[],
                    help="Mission.ATTR=value before the campaign (experiments), e.g. --set AUTO_STEER_S=0")
    ap.add_argument("--law", action="append", default=[],
                    help="InterceptLaw.attr=value on the law as built (experiments); an AI level "
                         "switch still re-applies tracker MODES on top")
    ap.add_argument("--master-seed", type=int, default=2026,
                    help="another value = a fresh, held-out set of engagements")
    ap.add_argument("--compare", default="", help="a --json from an earlier run: print the change")
    ap.add_argument("--procs", type=int, default=0)
    a = ap.parse_args()
    if a.list:
        for k, sc in SCENARIOS.items():
            print(f"  {k:<15s}[{sc['family']:<5s}] {', '.join(f'{s:g}' for s in sc['speeds'])} m/s: {sc['desc']}")
        return
    names = list(SCENARIOS)
    if a.only:
        want = [w.strip() for w in a.only.split(",") if w.strip()]
        names = [n for n in names if n in want or SCENARIOS[n]["family"] in want]
    import mission as M
    for kv in a.set:
        k, _, v = kv.partition("=")
        cur = getattr(M.Mission, k)
        setattr(M.Mission, k, type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true", "on"))
    try:
        os.nice(10)                     # the live tracker on this Jetson keeps priority
    except OSError:
        pass
    apply_stress()
    S.build_law_cached()                # forked workers inherit it
    for kv in a.law:                    # InterceptLaw.attr=value (experiments)
        k, _, v = kv.partition("=")
        setattr(S._LAW._pristine, k, float(v))
    js = jobs(names, 8 if a.quick else a.seeds, a.traces, a.master_seed)
    with Pool(a.procs or None) as p:
        res = p.map(run_one, js, chunksize=2)
    report(res)
    print(f"\n  SCORE (hit %, COME + GO): {score(res):.1f}")
    if a.compare:
        compare(json.load(open(a.compare))["results"], res)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(score=score(res), settings=[x for x in a.set], results=res), f)
        print(f"  wrote {a.json}")


if __name__ == "__main__":
    main()
