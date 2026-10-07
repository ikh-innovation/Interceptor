#!/usr/bin/env python3
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fc_control import InterceptLaw

FW = FH = 640
HFOV, VFOV = 62.2, 48.8
G = 9.80665


class Sim:

    TAU_ATT = 0.15
    TAU_YAW = 0.12
    TAU_CLIMB = 0.40
    DRAG = 0.0159

    def __init__(self, drone=(0.0, 0.0, 2.0), yaw=0.0):
        self.p = list(drone)
        self.v = [0.0, 0.0, 0.0]
        self.yaw = yaw
        self.lean = 0.0
        self.bank = 0.0
        self.yaw_rate = 0.0
        self.climb = 0.0

    def fwd(self):
        return (math.cos(self.yaw), math.sin(self.yaw), 0.0)

    def right(self):
        return (-math.sin(self.yaw), math.cos(self.yaw), 0.0)

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
        sp = math.hypot(self.v[0], self.v[1])
        ax -= self.DRAG * sp * self.v[0]
        ay -= self.DRAG * sp * self.v[1]
        self.v[0] += ax * dt
        self.v[1] += ay * dt
        self.v[2] = self.climb
        for i in range(3):
            self.p[i] += self.v[i] * dt

    def measure(self, tgt, cam_tilt):
        d = [tgt[i] - self.p[i] for i in range(3)]
        f, r = self.fwd(), self.right()
        fwd = d[0] * f[0] + d[1] * f[1]
        rgt = d[0] * r[0] + d[1] * r[1]
        horiz = math.hypot(fwd, rgt)
        if horiz < 1e-6:
            return None
        az = math.atan2(rgt, fwd)
        el_world = math.atan2(d[2], horiz)
        el_cam = el_world - (self.lean + cam_tilt)
        exl = (FW / 2.0) * math.tan(az) / math.tan(math.radians(HFOV / 2))
        eyl = -(FH / 2.0) * math.tan(el_cam) / math.tan(math.radians(VFOV / 2))
        c, s = math.cos(self.bank), math.sin(self.bank)
        return (exl * c + eyl * s, -exl * s + eyl * c,
                math.hypot(fwd, rgt, d[2]) if hasattr(math, "hypot") else 0.0)


def run(pn_gain, target, dur=12.0, latency=0.10, det_hz=20.0, climb_gain=8.0,
        cam_tilt_deg=15.0, max_lean_deg=20.0, jitter=0.0, seed=1, terminal=False,
        target_width_m=0.30):
    L = InterceptLaw(
        h_fov_deg=HFOV, v_fov_deg=VFOV, cam_tilt_deg=cam_tilt_deg, kp_yaw=4.5,
        roll_pn_gain=pn_gain, max_bank_deg=30.0, max_lean_deg=max_lean_deg,
        climb_gain=climb_gain, climb_lead=3.0, stick_limit=0.90, slew_per_s=11.0,
        deadband_frac=0.02, latency_s=latency, max_climb_mps=2.5,
        terminal_bbox_frac=0.35 if terminal else 2.0,
        terminal_range_m=2.5 if terminal else 0.0, terminal_hold_s=1.0)
    sim = Sim()
    dt = 0.005
    det_every = max(1, int(round((1.0 / det_hz) / dt)))
    lag = max(0, int(round(latency / dt)))
    hist, cmd = [], L.neutral()
    best, oof, seen, rsum, rn = 1e9, 0, 0, 0.0, 0
    t_seen, t_hit, maxbank, yaw0, yawspan = 0.0, None, 0.0, None, 0.0
    rnd = seed * 12345
    for i in range(int(dur / dt)):
        t = i * dt
        tgt = target(t)
        rng = math.sqrt(sum((tgt[k] - sim.p[k]) ** 2 for k in range(3)))
        best = min(best, rng)
        rsum += rng
        rn += 1
        maxbank = max(maxbank, abs(math.degrees(sim.bank)))
        yaw0 = sim.yaw if yaw0 is None else yaw0
        yawspan = max(yawspan, abs(math.degrees(sim.yaw - yaw0)))
        if rng < 0.6:
            t_hit = t
            break
        hist.append((sim.measure(tgt, math.radians(cam_tilt_deg)), sim.lean,
                     sim.bank, sim.yaw, sim.yaw_rate, rng))
        if i % det_every == 0 and len(hist) > lag:
            meas, lean_m, bank_m, yaw_m, yr_m, rng_m = hist[-1 - lag]
            ex, ey = (meas[0], meas[1]) if meas else (1e9, 1e9)
            if jitter and meas:
                rnd = (1103515245 * rnd + 12345) % (2 ** 31)
                ex += ((rnd / 2 ** 31) - 0.5) * 2 * jitter
                rnd = (1103515245 * rnd + 12345) % (2 ** 31)
                ey += ((rnd / 2 ** 31) - 0.5) * 2 * jitter
            if abs(ex) > FW / 2 or abs(ey) > FH / 2:
                oof += 1
            else:
                seen += 1
                t_seen = t
                bw = ((FW / 2.0) / math.tan(math.radians(HFOV / 2))) * target_width_m / rng_m
                cmd = L.compute(ex, ey, FW, FH, dt=1.0 / det_hz, bbox_w=bw,
                                bbox_h=bw,
                                att=dict(roll=bank_m, pitch=lean_m, yaw=yaw_m,
                                         yawspeed=yr_m, pitchspeed=0.0,
                                         rollspeed=0.0))
        if (t - t_seen) > 0.5:
            cmd = L.neutral()
        sim.step(math.radians(cmd.get("lean_deg", 0.0)),
                 math.radians(cmd.get("bank_deg", 0.0)),
                 cmd.get("yaw_rate", 0.0), cmd.get("climb_mps", 0.0), dt)
    return dict(best=best, mean=rsum / max(1, rn), t_hit=t_hit, maxbank=maxbank,
                yawspan=yawspan, inframe=seen / max(1, seen + oof))
