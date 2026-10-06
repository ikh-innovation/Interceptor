#!/usr/bin/env python3
"""SKY MODE before the field: a moving drone target, gusty wind Bf 0-3.

    ../.env/yolo/bin/python tests/sim_sky_wind.py            # full campaign
    ../.env/yolo/bin/python tests/sim_sky_wind.py --quick    # ~1/4 of the runs

Not a pass/fail test -- a campaign that prints the odds.  What is REAL here and
what is modelled:

  REAL      the InterceptLaw, built exactly as commander_from_env builds it
            from flight_config.py's sky env on top of run.py's INTERCEPT preset,
            then the MEDIUM speed profile applied by tracker.DroneTracker's own
            set_mode.  The Mission state machine (PUSH -> AI -> STOP, blends,
            floor, re-acquire, terminal grace), stepped at 50 Hz like the TX
            thread.  ENGAGE pressed once the target is in frame.
  MODELLED  the airframe (point mass in ALT_HOLD: attitude lag, yaw-accel limit,
            quadratic drag on AIRSPEED so wind pushes it), the wind (mean +
            filtered gusts + direction wander + vertical gusts, ALT_HOLD
            rejecting only part of the vertical), the camera (1280x960 px, mode 3,
            30 deg tilt, body-fixed, 30 fps), the detector (25 Hz, 120 ms
            latency, pixel noise, misses that rise as the target shrinks), the
            barometer (noise), and the pilot's pre-ENGAGE trim into the wind.

A TOUCH is a closest approach under 0.5 m between centres (two small quads).
"""
import argparse
import contextlib
import io
import math
import os
import random
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

G = 9.80665
FW, FH = 1280, 960                    # mode 3, full sensor, 4:3
HFOV = 62.2
VFOV_LAW = 48.8                       # what the law is told (CAM_VFOV_DEG default)
TILT = 30.0
ANGLE_MAX = math.radians(40.0)        # this FC
CLIMB_UP, CLIMB_DN = 25.0, 2.5        # PILOT_SPEED_UP / _DN on this FC, m/s
YAW_FULL = math.radians(202.5)
YAW_ACC = math.radians(297.0)         # ATC_ACCEL_Y_MAX
TOUCH_M = 0.5
AUTO = os.environ.get("SIM_AUTO") == "1"      # press AUTO instead of ENGAGE
GROUND = os.environ.get("SIM_START") == "ground"
HORIZON = os.environ.get("SIM_START") == "horizon"
# The pilot's own flow, for comparing with AUTO: ENGAGE pressed this many
# seconds after the target is first confirmed on screen ("" = at once).
ENGAGE_ON_DET = os.environ.get("SIM_ENGAGE_ON_DET", "")
H0 = os.environ.get("SIM_H0", "")                # start height override, m
TARGET_W = 0.30                       # the enemy, motor tip to tip

# Beaufort -> mean wind, m/s (middle of each band) and gust sigma
BEAUFORT = {0: (0.0, 0.0), 1: (1.0, 0.4), 2: (2.5, 0.8), 3: (4.4, 1.3)}


# ── the law, exactly as flown ────────────────────────────────────────────────
def build_law():
    with contextlib.redirect_stdout(io.StringIO()):
        import flight_config
        import run as R
        import fc_control as F
        import tracker
    env = dict(R.INTERCEPT)
    for line in _flight_env(flight_config):
        k, v = line
        env[k] = v                    # start-wasp.sh exports these; they win
    old = dict(os.environ)
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cmdr_law = _law_only(F)
    finally:
        os.environ.clear()
        os.environ.update(old)
    law = cmdr_law
    law.angle_max = ANGLE_MAX
    law.max_climb_mps, law.max_descent_mps = CLIMB_UP, CLIMB_DN

    class _Stub:
        pass
    stub = _Stub()
    stub.ap = _Stub()
    stub.ap.law = law
    stub.mode_name = None
    stub.MODES = tracker.DroneTracker.MODES
    stub._SPEED_KEYS = tracker.DroneTracker._SPEED_KEYS
    stub._STAB_KEYS = tracker.DroneTracker._STAB_KEYS
    with contextlib.redirect_stdout(io.StringIO()):
        tracker.DroneTracker.set_mode(stub, os.environ.get("SIM_SPEED")
                                      or env.get("WASP_MODE_SPEED", "MEDIUM"))
    return law


def _flight_env(fc):
    """flight_config.py --shell, parsed: what start-wasp.sh exports."""
    out = []
    for ln in fc.as_shell(fc.resolve()).splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            out.append((k.strip(), v.strip().strip("'")))
    return out


def _law_only(F):
    """commander_from_env's InterceptLaw, without opening any link."""
    src = F.commander_from_env.__code__
    del src
    captured = {}
    real = F.AcroCommander

    class _Grab:
        def __init__(self, *a, law=None, **k):
            captured["law"] = law
    F.AcroCommander = _Grab
    try:
        os.environ["MAV_CONN"] = "none"
        F.commander_from_env(verbose=False)
    finally:
        F.AcroCommander = real
    return captured["law"]


# ── the mapper the mission flies through (physical <-> PWM, linear) ─────────
class Mapper:
    def __init__(self):
        self.pilot = {}

    def _span(self, axis, v):
        if axis == "throttle":
            return v / CLIMB_UP if v >= 0 else v / CLIMB_DN
        if axis in ("roll", "pitch"):
            return math.radians(v) / ANGLE_MAX
        return math.radians(v) / YAW_FULL

    def phys_to_pwm(self, axis, v):
        n = max(-1.0, min(1.0, self._span(axis, float(v))))
        return int(round(1500 + 400 * n))

    def pwm_to_phys(self, axis, pwm):
        n = (pwm - 1500) / 400.0
        if axis == "throttle":
            return n * (CLIMB_UP if n >= 0 else CLIMB_DN)
        if axis in ("roll", "pitch"):
            return math.degrees(n * ANGLE_MAX)
        return math.degrees(n * YAW_FULL)

    def pilot_pwm(self, axis):
        return self.pilot.get(axis)

    def ai_pwm(self, cmd):
        return {"roll": self.phys_to_pwm("roll", cmd["bank_deg"]),
                "pitch": self.phys_to_pwm("pitch", cmd["lean_deg"]),
                "yaw": self.phys_to_pwm("yaw", math.degrees(cmd["yaw_rate"])),
                "throttle": self.phys_to_pwm("throttle", cmd["climb_mps"])}


# ── wind ─────────────────────────────────────────────────────────────────────
class Wind:
    """Mean + first-order gusts (tau ~2 s) on speed, direction wander, and
    discrete gust hits of 1.3-1.7x the mean.  Vertical gusts +/-0.4*sigma."""

    def __init__(self, bf, rng):
        self.mean, self.sig = BEAUFORT[bf]
        self.dir = rng.uniform(0, 2 * math.pi)
        self.rng = rng
        self.g = 0.0
        self.gd = 0.0
        self.gz = 0.0
        self.hit_t, self.hit_amp = -99.0, 0.0

    def step(self, t, dt):
        if self.mean <= 0:
            return 0.0, 0.0, 0.0
        r = self.rng
        a = dt / 2.0
        self.g += -a * self.g + self.sig * math.sqrt(2 * a) * r.gauss(0, 1)
        self.gd += -a * self.gd + math.radians(15) * math.sqrt(2 * a) * r.gauss(0, 1)
        self.gz += -(dt / 1.0) * self.gz + 0.4 * self.sig * math.sqrt(2 * dt) * r.gauss(0, 1)
        if t - self.hit_t > 2.0 and r.random() < dt / 4.0:     # a gust every ~4 s
            self.hit_t, self.hit_amp = t, self.mean * r.uniform(0.3, 0.7)
        hit = self.hit_amp * math.sin(math.pi * min(1.0, (t - self.hit_t) / 1.5)) \
            if t - self.hit_t < 1.5 else 0.0
        s = max(0.0, self.mean + self.g + hit)
        d = self.dir + self.gd
        return s * math.cos(d), s * math.sin(d), self.gz


# ── the airframe ─────────────────────────────────────────────────────────────
# ArduCopter's ALT_HOLD (rc_input_to_roll_pitch) limits the TOTAL tilt, not each
# axis: roll and pitch become one horizontal-thrust vector whose length is held
# to tan(ANGLE_MAX).  Without this, 35 deg of lean plus 36 of bank flies here as
# a 46 deg tilt the FC would never give.  SIM_TILT_LIMIT=0 = the old per-axis sim.
TILT_LIMIT = os.environ.get("SIM_TILT_LIMIT", "1") == "1"


def tilt_limit(lean, bank):
    tx, ty = math.tan(lean), math.tan(bank)
    n, cap = math.hypot(tx, ty), math.tan(ANGLE_MAX)
    if n > cap:
        tx, ty = tx * cap / n, ty * cap / n
    # back as per-axis angles whose tangents ARE that vector, because step()
    # turns each angle into g * tan(angle) of horizontal acceleration
    return math.atan(tx), math.atan(ty)


class Drone:
    TAU_ATT, TAU_YAW, TAU_CLIMB = 0.15, 0.12, 0.40
    DRAG = 0.0159                     # ~15 m/s at 20 deg of lean

    def __init__(self, p, yaw, lean0=0.0, bank0=0.0, v0=(0.0, 0.0)):
        self.p = list(p)
        self.v = [v0[0], v0[1], 0.0]
        self.yaw, self.yaw_rate = yaw, 0.0
        self.lean, self.bank, self.climb = lean0, bank0, 0.0

    def fwd(self):
        return math.cos(self.yaw), math.sin(self.yaw)

    def right(self):
        return -math.sin(self.yaw), math.cos(self.yaw)

    def step(self, lean_c, bank_c, yr_c, climb_c, wind, dt, rng):
        if TILT_LIMIT:
            lean_c, bank_c = tilt_limit(lean_c, bank_c)
        k = dt / (self.TAU_ATT + dt)
        # a gust also knocks the attitude a little before the FC rejects it
        self.lean += k * (lean_c - self.lean) + rng.gauss(0, 0.002) * (1 + abs(wind[0]) + abs(wind[1]))
        self.bank += k * (bank_c - self.bank) + rng.gauss(0, 0.002) * (1 + abs(wind[0]) + abs(wind[1]))
        want = self.yaw_rate + (dt / (self.TAU_YAW + dt)) * (yr_c - self.yaw_rate)
        self.yaw_rate += max(-YAW_ACC * dt, min(YAW_ACC * dt, want - self.yaw_rate))
        self.yaw += self.yaw_rate * dt
        self.climb += (dt / (self.TAU_CLIMB + dt)) * (climb_c - self.climb)
        a_f, a_r = G * math.tan(-self.lean), G * math.tan(self.bank)
        f, r = self.fwd(), self.right()
        ax, ay = a_f * f[0] + a_r * r[0], a_f * f[1] + a_r * r[1]
        avx, avy = self.v[0] - wind[0], self.v[1] - wind[1]
        asp = math.hypot(avx, avy)
        self.v[0] += (ax - self.DRAG * asp * avx) * dt
        self.v[1] += (ay - self.DRAG * asp * avy) * dt
        # ALT_HOLD rejects most of a vertical gust, not all of it
        self.v[2] = self.climb + 0.35 * wind[2]
        for i in range(3):
            self.p[i] += self.v[i] * dt

    def measure(self, tgt, vfov_real, hfov_real=HFOV):
        d = [tgt[i] - self.p[i] for i in range(3)]
        f, r = self.fwd(), self.right()
        fw = d[0] * f[0] + d[1] * f[1]
        rt = d[0] * r[0] + d[1] * r[1]
        h = math.hypot(fw, rt)
        if fw <= 0.05 or h < 1e-6:
            return None
        az = math.atan2(rt, fw)
        el_cam = math.atan2(d[2], h) - (self.lean + math.radians(TILT))
        if abs(el_cam) > math.radians(80):
            return None
        exl = (FW / 2) * math.tan(az) / math.tan(math.radians(hfov_real / 2))
        eyl = -(FH / 2) * math.tan(el_cam) / math.tan(math.radians(vfov_real / 2))
        c, s = math.cos(self.bank), math.sin(self.bank)
        rng_m = math.sqrt(h * h + d[2] * d[2])
        return exl * c + eyl * s, -exl * s + eyl * c, rng_m


# ── the enemy ────────────────────────────────────────────────────────────────
def make_target(kind, speed, p0, own_p, rng):
    """kind: hover | toward | away | cross | zigzag | climb_cross |
    angle_in (crossing at 45 deg while closing) | angle_out (45 deg while opening) |
    come<deg> (flying in, <deg> off the line to us: 0 = straight at us, 90 =
    crossing) | go<deg> (flying away, <deg> off straight away)."""
    dx, dy = own_p[0] - p0[0], own_p[1] - p0[1]
    to_us = math.atan2(dy, dx)
    if kind.startswith(("come", "go")):
        off = math.radians(float(kind.lstrip("comego"))) * rng.choice((-1, 1))
        heading = to_us + off if kind.startswith("come") else to_us + math.pi + off

        def straight(t):
            return (p0[0] + speed * t * math.cos(heading), p0[1] + speed * t * math.sin(heading), p0[2])
        return straight
    heading = {"hover": 0.0, "toward": to_us, "away": to_us + math.pi,
               "cross": to_us + math.pi / 2 * rng.choice((-1, 1)),
               "zigzag": to_us + math.pi / 2 * rng.choice((-1, 1)),
               "climb_cross": to_us + math.pi / 2 * rng.choice((-1, 1)),
               "angle_in": to_us + math.pi / 4 * rng.choice((-1, 1)),
               "angle_out": to_us + math.pi - math.pi / 4 * rng.choice((-1, 1))}[kind]
    period = rng.uniform(1.5, 3.0)
    vz = 1.5 if kind == "climb_cross" else 0.0

    def pos(t):
        if kind == "hover" or speed == 0:
            return (p0[0] + 0.3 * math.sin(0.7 * t), p0[1] + 0.3 * math.cos(0.5 * t), p0[2])
        if kind == "zigzag":
            # sideways at up to `speed`, reversing every half period, while
            # drifting toward us at 30% of it
            w = 2 * math.pi / period
            side = (speed / w) * math.sin(w * t)
            return (p0[0] + side * math.cos(heading) + 0.3 * speed * t * math.cos(to_us),
                    p0[1] + side * math.sin(heading) + 0.3 * speed * t * math.sin(to_us),
                    p0[2])
        return (p0[0] + speed * t * math.cos(heading),
                p0[1] + speed * t * math.sin(heading), p0[2] + vz * t)
    return pos


# ── one engagement ───────────────────────────────────────────────────────────
def engage(args):
    (bf, kind, speed, rng0, el0, az0, h0, seed, vfov_real, hfov_real) = args
    import mission as M
    rng = random.Random(seed)
    law = build_law_cached()
    law.reset() if hasattr(law, "reset") else None
    mp = Mapper()
    wind = Wind(bf, rng)
    m = M.Mission(mp, log=lambda *_a: None, accel_z=2.5, jerk_z=5.0)
    if os.environ.get("SIM_PUSH"):                  # default: the app's own (SLOW, 20 deg)
        m.set_level("push", os.environ["SIM_PUSH"])
    # the mission's AI level is the one the law was built with; an AUTO run may
    # ask for its own (ai_level_now), applied below as the tracker does
    lv_law = os.environ.get("SIM_SPEED") or "MEDIUM"
    m.set_level("ai", lv_law)
    for kv in filter(None, os.environ.get("SIM_AUTOP", "").split(",")):
        k, _, v = kv.partition("=")
        assert m.set_auto_param(k, v) == "", kv

    # pilot hovering in ALT_HOLD, trimmed into the MEAN wind (no GPS: by hand)
    wx, wy = wind.mean * math.cos(wind.dir), wind.mean * math.sin(wind.dir)
    yaw0 = rng.uniform(-math.pi, math.pi)
    trim_tilt = math.atan(Drone.DRAG * wind.mean * wind.mean / G)
    # lean/bank that push AGAINST the wind in body axes
    f = (math.cos(yaw0), math.sin(yaw0))
    r = (-math.sin(yaw0), math.cos(yaw0))
    if wind.mean > 0:
        ux, uy = -wx / wind.mean, -wy / wind.mean
        lean0 = -trim_tilt * (ux * f[0] + uy * f[1])
        bank0 = trim_tilt * (ux * r[0] + uy * r[1])
    else:
        lean0 = bank0 = 0.0
    lean0 += math.radians(rng.gauss(0, 1.0))        # the pilot is not perfect
    bank0 += math.radians(rng.gauss(0, 1.0))
    dr = Drone((0.0, 0.0, h0), yaw0, lean0, bank0, v0=(rng.gauss(0, 0.3), rng.gauss(0, 0.3)))
    mp.pilot = {"pitch": mp.phys_to_pwm("pitch", math.degrees(lean0)),
                "roll": mp.phys_to_pwm("roll", math.degrees(bank0)),
                "yaw": 1500, "throttle": 1500}

    # target: range / elevation / azimuth relative to where the camera looks
    a = yaw0 + math.radians(az0)
    hz = rng0 * math.cos(math.radians(el0))
    p0 = (hz * math.cos(a), hz * math.sin(a), h0 + rng0 * math.sin(math.radians(el0)))
    if H0:                                          # same target, another start height
        h0 = float(H0)
        dr.p[2] = h0
    tgt_at = make_target(kind, speed, p0, dr.p, rng)

    dt = 0.005
    det_period, det_lat = 1 / 25.0, 0.12
    hist = []                                     # (t, measurement, att)
    pending = []                                  # detections in the pipe
    ctx = dict(gate_ok=True, gate_why="", att=None, height=h0, climb=0.0,
               ai_cmd=None, ai_t=None, det_t=None, det_conf=None)
    next_det, next_tx, t_last_det = 0.0, 0.0, None
    best, t_best, crashed = 1e9, None, False
    frames_in, frames = 0, 0
    max_bank = 0.0
    engaged_at = 0.0
    ctx["height"] = h0
    m.cycle(0.0, ctx)                             # PILOT -> HOVER
    pressed = True
    if AUTO:
        ctx["att"] = dict(roll=dr.bank, pitch=dr.lean, yaw=dr.yaw)
        m.set_auto_enabled(True)
        m.request_auto(True)                      # AUTO: climb, find, align, go
    elif ENGAGE_ON_DET:
        pressed = False                           # the pilot waits for it on screen
    else:
        m.request_engage()
    t_seen = None                                 # first confirmed detection
    out = {}
    T_END = 45.0 if HORIZON else 30.0 if AUTO else 16.0
    t = 0.0
    phases = set()
    t_push = None           # when the push began (AUTO: the second line)
    # ── the turn metrics (the field complaint: it turns PAST the target) ──
    t_ai0 = None            # when the AI first steered
    az_ai0 = 0.0            # target bearing off the nose at that instant, deg
    overshoot = 0.0         # deg the nose swung PAST the target, other side
    peak_yr = 0.0           # deg/s, the airframe
    sweep = 0.0             # total heading swept after takeover, deg
    lost_after = 0          # frames with the target OUT of the picture, post takeover
    frames_after = 0
    bank_hist = []          # (t, commanded bank deg)
    bank_jump = 0.0         # biggest change of commanded bank inside 0.2 s
    passed_latch = False
    while t < T_END:
        tgt = tgt_at(t)
        d = math.dist(tgt, dr.p)
        if d < best:
            best, t_best = d, t
        if best < TOUCH_M:
            break
        if dr.p[2] < 0.3:
            crashed = True
            break
        # true bearing of the target off the nose
        dxy = (tgt[0] - dr.p[0], tgt[1] - dr.p[1])
        az_true = math.degrees(math.atan2(-math.sin(dr.yaw) * dxy[0] + math.cos(dr.yaw) * dxy[1],
                                          math.cos(dr.yaw) * dxy[0] + math.sin(dr.yaw) * dxy[1]))
        az_pre = az_true
        if t_ai0 is None and (m._push_steer or m.phase == "AI"):
            t_ai0, az_ai0 = t, az_true
        # once it has gone PAST the target (range opening again after a close
        # approach), what follows is the pass, not the turn: stop scoring it
        passed = (best < 10.0 and d > best + 2.0) or passed_latch \
            or (d < 8.0 and abs(az_pre) > 75.0)
        passed_latch = passed
        if t_ai0 is not None and d > 5.0 and not passed:
            if abs(az_ai0) > 2.0:
                overshoot = max(overshoot, -math.copysign(1, az_ai0) * az_true)
            else:
                overshoot = max(overshoot, abs(az_true) - 2.0)
            peak_yr = max(peak_yr, abs(math.degrees(dr.yaw_rate)))
            sweep += abs(math.degrees(dr.yaw_rate)) * dt
        # camera frame
        if t >= next_det:
            next_det += det_period
            meas = dr.measure(tgt, vfov_real, hfov_real)
            frames += 1
            att = dict(roll=dr.bank, pitch=dr.lean, yaw=dr.yaw, yawspeed=dr.yaw_rate,
                       pitchspeed=0.0, rollspeed=0.0)
            if meas is None and t_ai0 is not None and d > 3.0 and not passed:
                frames_after += 1
                lost_after += 1
            if meas is not None:
                ex, ey, rng_m = meas
                inside = abs(ex) <= FW / 2 and abs(ey) <= FH / 2
                if t_ai0 is not None and rng_m > 3.0 and not passed:
                    frames_after += 1
                    lost_after += 0 if inside else 1
                if inside:
                    frames_in += 1
                    focal = (FW / 2) / math.tan(math.radians(hfov_real / 2))
                    bw = focal * TARGET_W / max(0.2, rng_m)
                    # detection odds: poor below ~6 px, good above ~14 px
                    p_det = 0.93 * max(0.0, min(1.0, (bw - 4.0) / 10.0))
                    if bw > 0.6 * FH:
                        p_det *= 0.2                 # fills the frame: drops out
                    if rng.random() < p_det:
                        conf = min(0.95, 0.25 + 0.03 * bw) * rng.uniform(0.8, 1.0)
                        nx = rng.gauss(0, 2.0 + 0.04 * bw)
                        ny = rng.gauss(0, 2.0 + 0.04 * bw)
                        bwn = bw * rng.uniform(0.85, 1.15)
                        pending.append((t + det_lat + rng.uniform(-0.02, 0.03),
                                        t, ex + nx, ey + ny, bwn, conf, att))
        # detections arriving at the Jetson's control code
        while pending and pending[0][0] <= t:
            arr, cap, ex, ey, bwn, conf, att = pending.pop(0)
            ddt = det_period if t_last_det is None else max(1e-3, arr - t_last_det)
            t_last_det = arr
            # attitude interpolated to capture time, ~10 ms residual error
            cmd = law.compute(ex, ey, FW, FH, dt=ddt, bbox_w=bwn, bbox_h=bwn,
                              att=att, capture_age=arr - cap)
            ctx.update(ai_cmd=cmd, ai_t=t, det_t=t, det_conf=conf)
        # TX thread, 50 Hz
        if t >= next_tx:
            next_tx += 0.02
            # As AcroCommander._mission_tx: the AI command reaches the mission
            # only while a real detection is younger than target_timeout
            # (0.5 s).  Keeping the last one in ctx for ever let a stale
            # TERMINAL command renew the mission's terminal hold every cycle:
            # the sim then flew the frozen sticks for many seconds after a
            # pass, often into the ground -- a crash the real code cannot make.
            if ctx["det_t"] is not None and t - ctx["det_t"] > 0.5:
                ctx["ai_cmd"] = ctx["ai_t"] = None
            ctx["height"] = dr.p[2] + rng.gauss(0, 0.25)
            ctx["climb"] = dr.v[2] + rng.gauss(0, 0.2)
            ctx["att"] = dict(roll=dr.bank, pitch=dr.lean, yaw=dr.yaw + rng.gauss(0, 0.01))
            if t_seen is None and m._confirmed(t, ctx):
                t_seen = t
            if not pressed and t_seen is not None and t >= t_seen + float(ENGAGE_ON_DET):
                pressed = m.request_engage() == "" or pressed
            out = m.cycle(t, ctx)
            phases.add(m.phase)
            lv = m.ai_level_now()
            if lv != lv_law:                       # the tracker's per-frame sync
                set_level(law, lv)
                lv_law = lv
            if t_push is None and m.phase == "PUSH":
                t_push = t
            if m.phase in ("HOVER",) and t > 0.5 and pressed:
                break                                   # the attempt is over
        wv = wind.step(t, dt)
        if out:
            lean = math.radians(mp.pwm_to_phys("pitch", out.get("pitch", 1500)))
            bank = math.radians(mp.pwm_to_phys("roll", out.get("roll", 1500)))
            if t_ai0 is not None and d > 3.0 and not passed:
                bd = math.degrees(bank)
                bank_hist.append((t, bd))
                while bank_hist and t - bank_hist[0][0] > 0.2:
                    bank_hist.pop(0)
                bank_jump = max(bank_jump, max(abs(bd - b) for _, b in bank_hist))
            yr = math.radians(mp.pwm_to_phys("yaw", out.get("yaw", 1500)))
            cl = mp.pwm_to_phys("throttle", out.get("throttle", 1500))
        else:                                           # released: pilot's trim
            lean, bank, yr, cl = lean0, bank0, 0.0, 0.0
        max_bank = max(max_bank, abs(math.degrees(dr.bank)))
        dr.step(lean, bank, yr, cl, wv, dt, rng)
        t += dt
    return dict(bf=bf, kind=kind, speed=speed, rng0=rng0, el0=el0, best=best,
                touch=best < TOUCH_M, t=t_best, crashed=crashed,
                in_frame=frames_in / max(1, frames), max_bank=max_bank,
                ai="AI" in phases, vfov=vfov_real, h_end=dr.p[2] - h0,
                steered=t_ai0 is not None, az_ai0=az_ai0, overshoot=max(0.0, overshoot),
                peak_yr=peak_yr, sweep=sweep, bank_jump=bank_jump,
                lost_frac=lost_after / max(1, frames_after), az0=az0,
                t_push=t_push, aligned="AUTO_ALIGN" in phases, t_seen=t_seen,
                tgt_alt=p0[2])


_LAW = None


def set_level(law, level):
    """An AI level onto a live law, exactly as tracker.DroneTracker.set_mode."""
    import tracker

    class _S:
        pass
    st = _S()
    st.ap = _S()
    st.ap.law = law
    st.mode_name = None
    st.MODES = tracker.DroneTracker.MODES
    st._SPEED_KEYS = tracker.DroneTracker._SPEED_KEYS
    st._STAB_KEYS = tracker.DroneTracker._STAB_KEYS
    with contextlib.redirect_stdout(io.StringIO()):
        tracker.DroneTracker.set_mode(st, level)


def build_law_cached():
    global _LAW
    if _LAW is None:
        _LAW = build_law()
        if os.environ.get("SIM_OLD_SEEKER") == "1":
            _LAW.kp_yaw, _LAW.stick_limit_yaw = 4.5, 0.90
        import copy
        _LAW._pristine = copy.deepcopy(_LAW)
    import copy
    return copy.deepcopy(_LAW._pristine)


# ── the campaign ─────────────────────────────────────────────────────────────
def scenarios(n_seeds, vfov_real=VFOV_LAW, hfov_real=HFOV):
    if HORIZON:
        return scenarios_horizon(n_seeds, vfov_real, hfov_real)
    if GROUND:
        return scenarios_ground(n_seeds, vfov_real, hfov_real)
    rng = random.Random(1234)
    rows = []
    motions = [("hover", 0.0), ("toward", 5.0), ("toward", 10.0), ("away", 3.0),
               ("cross", 3.0), ("cross", 6.0), ("cross", 10.0),
               ("zigzag", 5.0), ("climb_cross", 5.0),
               ("angle_in", 5.0), ("angle_in", 10.0), ("angle_out", 5.0)]
    for bf in (0, 1, 2, 3):
        for kind, sp in motions:
            for s in range(n_seeds):
                rows.append((bf, kind, sp, rng.uniform(20, 40), rng.uniform(5, 25),
                             rng.uniform(-28, 28), rng.uniform(4, 8),
                             rng.randrange(10 ** 9), vfov_real, hfov_real))
    return rows


def scenarios_ground(n_seeds, vfov_real=VFOV_LAW, hfov_real=HFOV):
    """SIM_START=ground: just off the ground (1.5 m), the enemy 25-50 m out
    and 6-25 m up, within 25 deg of the nose -- often above the frame or too
    small to detect yet, as when AUTO is pressed in the field."""
    rng = random.Random(4321)
    rows = []
    motions = [("hover", 0.0), ("toward", 5.0), ("toward", 10.0), ("away", 3.0),
               ("cross", 3.0), ("cross", 6.0), ("cross", 10.0),
               ("zigzag", 5.0), ("climb_cross", 5.0),
               ("angle_in", 5.0), ("angle_in", 10.0), ("angle_out", 5.0)]
    for bf in (0, 1, 2, 3):
        for kind, sp in motions:
            for s in range(n_seeds):
                hz, dz = rng.uniform(25, 50), rng.uniform(6, 25)
                rows.append((bf, kind, sp, math.hypot(hz, dz), math.degrees(math.atan2(dz, hz)),
                             rng.uniform(-25, 25), 1.5,
                             rng.randrange(10 ** 9), vfov_real, hfov_real))
    return rows


HORIZON_ALTS = {"low": (5.0, 12.0), "mid": (15.0, 25.0), "high": (30.0, 45.0)}


def scenarios_horizon(n_seeds, vfov_real=VFOV_LAW, hfov_real=HFOV):
    """SIM_START=horizon: from 1.5 m (SIM_H0 to change), the enemy within 15
    deg of the nose, at a low / mid / high altitude band above the ground.
    COMING: it appears 60-80 m out and flies in, 0 / 30 / 60 / 90 deg off the
    line to us, at 5 / 10 / 15 m/s.  GOING: 20-40 m out, flying away, 0 / 30
    / 60 deg off straight away, at 5 / 10 m/s."""
    rng = random.Random(777)
    rows = []
    paths = [(f"come{a}", v, (60, 80)) for a in (0, 30, 60, 90) for v in (5.0, 10.0, 15.0)]
    paths += [(f"go{a}", v, (20, 40)) for a in (0, 30, 60) for v in (5.0, 10.0)]
    for bf in (0, 1, 2, 3):
        for kind, sp, (r0, r1) in paths:
            for band, (z0, z1) in HORIZON_ALTS.items():
                for s in range(n_seeds):
                    hz, alt = rng.uniform(r0, r1), rng.uniform(z0, z1)
                    dz = alt - 1.5
                    rows.append((bf, kind, sp, math.hypot(hz, dz), math.degrees(math.atan2(dz, hz)),
                                 rng.uniform(-15, 15), 1.5, rng.randrange(10 ** 9), vfov_real, hfov_real))
    return rows


def pct(xs):
    return 100.0 * sum(xs) / max(1, len(xs))


def report(res, title):
    print(f"\n=== {title} ===")
    print(f"{'':22s}" + "".join(f"{'Bf' + str(b):>10s}" for b in (0, 1, 2, 3)))
    keys = []
    for r in res:
        k = (r["kind"], r["speed"])
        if k not in keys:
            keys.append(k)
    for k in keys:
        cells = []
        for b in (0, 1, 2, 3):
            rs = [r for r in res if (r["kind"], r["speed"]) == k and r["bf"] == b]
            cells.append(f"{pct([r['touch'] for r in rs]):8.0f}% ")
        print(f"  {k[0]:<12s}{k[1]:4.0f} m/s " + "".join(cells))
    print("  " + "-" * 60)
    for label, f in (("TOUCH  (< 0.5 m)", lambda r: r["touch"]),
                     ("NEAR   (< 1.5 m)", lambda r: r["best"] < 1.5),
                     ("AI took over", lambda r: r["ai"]),
                     ("ground / crash", lambda r: r["crashed"])):
        cells = "".join(f"{pct([f(r) for r in res if r['bf'] == b]):8.0f}% " for b in (0, 1, 2, 3))
        print(f"  {label:<22s}" + cells)
    for b in (0, 1, 2, 3):
        rs = sorted(r["best"] for r in res if r["bf"] == b)
        ts = sorted(r["t"] for r in res if r["bf"] == b and r["touch"])
        med_t = ts[len(ts) // 2] if ts else float("nan")
        bank = sorted(r["max_bank"] for r in res if r["bf"] == b)
        print(f"  Bf{b}: median miss {rs[len(rs) // 2]:.2f} m, 90% of runs within "
              f"{rs[int(0.9 * (len(rs) - 1))]:.1f} m, median time to touch "
              f"{med_t:.1f} s, peak bank p95 {bank[int(0.95 * (len(bank) - 1))]:.0f} deg")


def q(xs, f):
    xs = sorted(xs)
    return xs[int(f * (len(xs) - 1))] if xs else float("nan")


def report_turns(res):
    print("\n=== THE TURN AT TAKEOVER: does it swing PAST the target? ===")
    print("  (only runs where the AI steered; the last 3 m are excluded -- that is the pass)")
    st = [r for r in res if r["steered"]]
    print(f"{'':34s}" + "".join(f"{'Bf' + str(b):>9s}" for b in (0, 1, 2, 3)))
    rows = (
        ("overshoot past target, median", lambda rs: f"{q([r['overshoot'] for r in rs], .5):6.1f} deg"),
        ("overshoot past target, p95", lambda rs: f"{q([r['overshoot'] for r in rs], .95):6.1f} deg"),
        ("overshoot, worst", lambda rs: f"{max(r['overshoot'] for r in rs):6.1f} deg"),
        ("runs swinging > 15 deg past", lambda rs: f"{pct([r['overshoot'] > 15 for r in rs]):6.1f} %  "),
        ("peak yaw rate, worst", lambda rs: f"{max(r['peak_yr'] for r in rs):6.1f} d/s"),
        ("heading swept > 270 (circle)", lambda rs: f"{pct([r['sweep'] > 270 for r in rs]):6.1f} %  "),
        ("bank jump in 0.2 s, p95", lambda rs: f"{q([r['bank_jump'] for r in rs], .95):6.1f} deg"),
        ("bank jump in 0.2 s, worst", lambda rs: f"{max(r['bank_jump'] for r in rs):6.1f} deg"),
        ("target out of frame, median", lambda rs: f"{100 * q([r['lost_frac'] for r in rs], .5):6.1f} %  "),
        ("target out of frame, p90", lambda rs: f"{100 * q([r['lost_frac'] for r in rs], .9):6.1f} %  "),
    )
    for label, f in rows:
        print(f"  {label:<32s}" + "".join(f"{f([r for r in st if r['bf'] == b]):>9s}" for b in (0, 1, 2, 3)))
    print("\n  by where the target was in the frame at ENGAGE (all winds):")
    for lo, hi in ((0, 10), (10, 20), (20, 29)):
        rs = [r for r in st if lo <= abs(r["az0"]) < hi]
        if rs:
            print(f"    {lo:2d}-{hi:2d} deg off-centre: overshoot p95 {q([r['overshoot'] for r in rs], .95):5.1f} deg, "
                  f"worst {max(r['overshoot'] for r in rs):5.1f}, peak yaw {max(r['peak_yr'] for r in rs):4.0f} deg/s, "
                  f"touch {pct([r['touch'] for r in rs]):3.0f}%  (n={len(rs)})")
    print("\n  by target motion (all winds):")
    kinds = []
    for r in st:
        if (r["kind"], r["speed"]) not in kinds:
            kinds.append((r["kind"], r["speed"]))
    for k in kinds:
        rs = [r for r in st if (r["kind"], r["speed"]) == k]
        print(f"    {k[0]:<12s}{k[1]:4.0f} m/s: past target median {q([r['overshoot'] for r in rs], .5):4.1f}, "
              f"p95 {q([r['overshoot'] for r in rs], .95):5.1f}, worst {max(r['overshoot'] for r in rs):5.1f} deg; "
              f"> 15 deg in {pct([r['overshoot'] > 15 for r in rs]):3.0f}% of runs")
    worst = sorted(st, key=lambda r: -r["overshoot"])[:5]
    print("\n  the 5 worst swings:")
    for r in worst:
        print(f"    Bf{r['bf']} {r['kind']:<11s}{r['speed']:4.0f} m/s  target {r['az0']:+5.1f} deg at ENGAGE -> "
              f"{r['overshoot']:5.1f} deg past, peak yaw {r['peak_yr']:4.0f} d/s, miss {r['best']:.2f} m")


def compare_speeds(allres):
    """One table: every target motion x wind, SLOW / MEDIUM / FAST side by side."""
    speeds = [sp for sp in ("SLOW", "MEDIUM", "FAST", "MAX") if sp in allres]
    print("\n=== AI SPEED COMPARISON: touch rate (< 0.5 m), PUSH at %s ===" % (os.environ.get("SIM_PUSH") or "SLOW (the default, 20 deg)"))
    print(f"  {'target':<22s}" + "".join(f"{'Bf' + str(b):>16s}" for b in (0, 1, 2, 3)))
    print(f"  {'':<22s}" + "".join(f"{' / '.join(s[0] for s in speeds):>16s}" for _ in range(4)))
    keys = []
    for r in allres[speeds[0]]:
        if (r["kind"], r["speed"]) not in keys:
            keys.append((r["kind"], r["speed"]))
    for k in keys:
        cells = []
        for b in (0, 1, 2, 3):
            vals = [pct([r["touch"] for r in allres[sp] if (r["kind"], r["speed"]) == k and r["bf"] == b])
                    for sp in speeds]
            cells.append(" / ".join(f"{v:3.0f}" for v in vals))
        print(f"  {k[0]:<12s}{k[1]:4.0f} m/s   " + "".join(f"{c:>16s}" for c in cells))
    print("  " + "-" * 86)
    for label, f in (("TOUCH (< 0.5 m)", lambda r: r["touch"]), ("NEAR (< 1.5 m)", lambda r: r["best"] < 1.5),
                     ("ground / crash", lambda r: r["crashed"])):
        cells = []
        for b in (0, 1, 2, 3):
            vals = [pct([f(r) for r in allres[sp] if r["bf"] == b]) for sp in speeds]
            cells.append(" / ".join(f"{v:3.0f}" for v in vals))
        print(f"  {label:<22s}" + "".join(f"{c:>16s}" for c in cells))
    print()
    for sp in speeds:
        rs = allres[sp]
        ts = sorted(r["t"] for r in rs if r["touch"])
        st = [r for r in rs if r["steered"]]
        print(f"  {sp:<6s}: touch {pct([r['touch'] for r in rs]):4.0f}%  near {pct([r['best'] < 1.5 for r in rs]):4.0f}%  "
              f"median time to touch {ts[len(ts) // 2] if ts else float('nan'):4.1f} s  "
              f"peak bank p95 {q([r['max_bank'] for r in rs], .95):3.0f} deg  "
              f"bank jump/0.2 s p95 {q([r['bank_jump'] for r in st], .95):3.0f} deg  "
              f"swing past target p95 {q([r['overshoot'] for r in st if r['kind'] != 'zigzag'], .95):4.1f} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--old-seeker", action="store_true",
                    help="the seeker as flown before 2026-09-25: kp_yaw 4.5, yaw up to 182 deg/s")
    ap.add_argument("--speed", default=None, choices=("SLOW", "MEDIUM", "FAST", "MAX", "all"),
                    help="AI speed profile (tracker.DroneTracker.MODES); 'all' compares the four")
    ap.add_argument("--push", default=None, choices=("SLOW", "MEDIUM", "FAST"),
                    help="PUSH lean level; default = the app's default (SLOW, 20 deg)")
    ap.add_argument("--hfov", type=float, default=None,
                    help="the REAL horizontal FOV, if it differs from the law's 62.2")
    ap.add_argument("--vfov", type=float, default=None,
                    help="the REAL vertical FOV, if it differs from the law's 48.8")
    a = ap.parse_args()
    n = a.seeds or (10 if a.quick else 40)
    if a.old_seeker:
        os.environ["SIM_OLD_SEEKER"] = "1"
    if a.push:
        os.environ["SIM_PUSH"] = a.push
    if a.speed == "all":
        global _LAW
        allres = {}
        for sp in ("SLOW", "MEDIUM", "FAST", "MAX"):
            os.environ["SIM_SPEED"] = sp
            _LAW = None
            build_law_cached()                       # forked workers inherit it
            rows = scenarios(n, a.vfov or VFOV_LAW, a.hfov or HFOV)
            with Pool() as p:
                allres[sp] = p.map(engage, rows, chunksize=4)
        compare_speeds(allres)
        return
    if a.speed:
        os.environ["SIM_SPEED"] = a.speed
    law = build_law_cached()
    print("law as flown: tilt %.0f, lean %.0f, PN %.0f, bank %.0f..%.0f, kp_yaw %.1f, "
          "yaw cap %.0f deg/s, climb_gain %.0f, climb_los %s, slew %.1f" % (
              math.degrees(law.cam_tilt),
              math.degrees(law.max_lean), law.roll_pn_gain,
              math.degrees(law.bank_far) if law.bank_far else float("nan"),
              math.degrees(law.max_bank), law.kp_yaw,
              math.degrees(law.stick_limit_yaw * YAW_FULL), law.climb_gain,
              law.climb_los, law.slew_per_s))
    rows = scenarios(n, a.vfov or VFOV_LAW, a.hfov or HFOV)
    with Pool() as p:
        res = p.map(engage, rows, chunksize=4)
    report(res, f"SKY MODE, {len(res)} engagements, touch rate by target motion x wind"
           + (f"  (real FOV {a.hfov or HFOV} x {a.vfov or VFOV_LAW})" if (a.vfov or a.hfov) else ""))
    report_turns(res)


if __name__ == "__main__":
    main()
