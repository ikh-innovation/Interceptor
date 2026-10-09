#!/usr/bin/env python3
import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from fc_control import InterceptLaw

with contextlib.redirect_stdout(io.StringIO()):
    import sim_model as T

FAILED = []


def check(label, ok, detail=""):

    if not ok:
        FAILED.append(label)


G = 9.80665
YAW_ACC = math.radians(297.0)
TILT = 30.0


class FcSim(T.Sim):

    def __init__(self, wind=(0.0, 0.0), **kw):
        super().__init__(**kw)
        self.wind = wind
        self.v = [wind[0], wind[1], 0.0]

    def step(self, lean_c, bank_c, yawrate_c, climb_c, dt):
        for name, cmd, tau in (("lean", lean_c, self.TAU_ATT), ("bank", bank_c, self.TAU_ATT),
                               ("climb", climb_c, self.TAU_CLIMB)):
            c = getattr(self, name)
            setattr(self, name, c + (dt / (tau + dt)) * (cmd - c))
        cur = self.yaw_rate
        want = cur + (dt / (self.TAU_YAW + dt)) * (yawrate_c - cur)
        self.yaw_rate = cur + max(-YAW_ACC * dt, min(YAW_ACC * dt, want - cur))
        self.yaw += self.yaw_rate * dt
        a_f, a_r = G * math.tan(-self.lean), G * math.tan(self.bank)
        f, r = self.fwd(), self.right()
        ax, ay = a_f * f[0] + a_r * r[0], a_f * f[1] + a_r * r[1]
        avx, avy = self.v[0] - self.wind[0], self.v[1] - self.wind[1]
        asp = math.hypot(avx, avy)
        self.v[0] += (ax - self.DRAG * asp * avx) * dt
        self.v[1] += (ay - self.DRAG * asp * avy) * dt
        self.v[2] = self.climb
        for i in range(3):
            self.p[i] += self.v[i] * dt


def ss(a):
    a = min(1.0, max(0.0, a))
    return a * a * (3 - 2 * a)


def engage(kp, cap_dps, az_off, el_off, tgt_v, wind, seed, att_late=0.08):
    L = InterceptLaw(h_fov_deg=T.HFOV, v_fov_deg=T.VFOV, cam_tilt_deg=TILT, kp_yaw=kp,
                     stick_limit=0.90, stick_limit_yaw=min(0.90, cap_dps / 202.5),
                     roll_pn_gain=45.0, max_bank_deg=30.0, bank_far_deg=24.0, max_lean_deg=22.0,
                     climb_limit_mps=25.0, sens=0.65, pn_near_mult=1.4, lock_ramp_s=1.0,
                     edge_guard=0.55, climb_los=True, climb_trim=8.0, cruise_mps=15.0,
                     cruise_lean_deg=20.0, stick_limit_throttle=1.0, max_climb_mps=25.0,
                     max_descent_mps=2.5, angle_max=math.radians(40.0), slew_per_s=6.0,
                     rate_tau=0.22, deadband_frac=0.05, latency_s=0.15, terminal_bbox_frac=0.35,
                     terminal_range_m=2.5, terminal_hold_s=1.0, target_width_m=0.30)
    sim = FcSim(wind=wind, drone=(0.0, 0.0, 10.0))
    a = math.radians(az_off)
    d = 40.0
    tgt0 = (d * math.cos(a), d * math.sin(a), 10.0 + d * math.tan(math.radians(10.0 + el_off)))
    dt, lag, late = 0.005, int(0.15 / 0.005), int(att_late / 0.005)
    rng = np.random.default_rng(seed)
    hist, cmd, t_seen, best = [], L.neutral(), -1e9, 1e9
    src = {k: "push" for k in ("lean_deg", "bank_deg", "yaw_rate", "climb_mps")}
    mix, last, t_ai, yaws = {}, dict(lean_deg=0.0, bank_deg=0.0, yaw_rate=0.0, climb_mps=0.0), None, []
    max_rate = 0.0
    for i in range(int(14.0 / dt)):
        t = i * dt
        tgt = tuple(tgt0[k] + tgt_v[k] * t for k in range(3))
        best = min(best, math.dist(tgt, sim.p))
        if best < 0.5:
            break
        hist.append((sim.measure(tgt, math.radians(TILT)), sim.lean, sim.bank, sim.yaw, sim.yaw_rate, 0))
        det = False
        if i % 20 == 0 and len(hist) > lag:
            meas = hist[-1 - lag][0]
            _, lm, bm, ym, yr, _ = hist[min(len(hist) - 1, len(hist) - 1 - lag + late)]
            if meas:
                ex, ey = meas[0] + rng.uniform(-8, 8), meas[1] + rng.uniform(-8, 8)
                if abs(ex) <= T.FW / 2 and abs(ey) <= T.FH / 2:
                    t_seen, det = t, True
                    rm = math.dist(tgt, sim.p)
                    bw = ((T.FW / 2) / math.tan(math.radians(T.HFOV / 2))) * 0.30 / rm
                    cmd = L.compute(ex, ey, T.FW, T.FH, dt=0.1, bbox_w=bw, bbox_h=bw,
                                    att=dict(roll=bm, pitch=lm, yaw=ym, yawspeed=yr,
                                             pitchspeed=0.0, rollspeed=0.0))
        if (t - t_seen) > 0.5 and t_ai is not None:
            break
        push = dict(lean_deg=-30.0 * ss(t / 1.2), bank_deg=0.0, yaw_rate=0.0, climb_mps=0.0)

        def switch(k, s_):
            if src[k] != s_:
                mix[k] = (last[k], t)
                src[k] = s_
        if det and t_ai is None:
            t_ai = t
            for k in ("bank_deg", "yaw_rate", "climb_mps"):
                switch(k, "ai")
        if t_ai is not None and t >= 1.5:
            switch("lean_deg", "ai")
        out = {}
        for k in src:
            v = cmd.get(k, 0.0) if src[k] == "ai" else push[k]
            if k in mix:
                w = ss((t - mix[k][1]) / 1.5)
                v = v if w >= 1.0 else (1 - w) * mix[k][0] + w * v
            out[k] = v
        last = out
        if i % 4 == 0:
            yaws.append(math.degrees(sim.yaw))
        max_rate = max(max_rate, abs(math.degrees(sim.yaw_rate)))
        sim.step(math.radians(out["lean_deg"]), math.radians(out["bank_deg"]),
                 out["yaw_rate"], out["climb_mps"], dt)
    return best < 0.5, best, max_rate


CASES = [(24, 8, (0, 0, 0)), (24, 8, (0, 5, 0)), (28, 0, (0, 5, 0)), (0, 0, (0, 5, 0))]
WINDS = ((0.0, 0.0), (0.0, 5.0))


def sweep(kp, cap):
    rows = [engage(kp, cap, az, el, mv, w, seed)
            for az, el, mv in CASES for w in WINDS for seed in (1, 2)]
    return (sum(h for h, _, _ in rows), len(rows), max(b for _, b, _ in rows),
            max(r for _, _, r in rows))



h0, n, w0, r0 = sweep(4.5, 182.0)
h1, _, w1, r1 = sweep(2.0, 40.0)


check("OLD seeker spins at > 100 deg/s from a frame-edge first detection (the field)", r0 > 100,
      f"{r0:.0f} deg/s")
check("NEW seeker never exceeds its 40 deg/s cap", r1 <= 41.0, f"{r1:.1f} deg/s")
check("...and touches at least as often, worst miss under 1 m",
      h1 >= h0 and w1 < 1.0, f"{h1} vs {h0}, worst {w1:.2f} m")

with contextlib.redirect_stdout(io.StringIO()):
    import tracker
check("the shipped profiles carry the new seeker (kp 2.0, 40 deg/s) on every speed",
      all(p.get("kp_yaw") == 2.0 and p.get("yaw_cap_dps") == 40.0
          for p in tracker.DroneTracker.MODES.values()))


if FAILED:

    for f in FAILED:
        pass
    sys.exit(1)
