#!/usr/bin/env python3
"""LOCK against gusts that try to turn the drone: does the heading stay put?

    ../.env/yolo/bin/python tests/sim_lock_wind.py            # full campaign
    ../.env/yolo/bin/python tests/sim_lock_wind.py --quick

Hovering in ALT_HOLD, 60 s per run, gusty wind Bf 0-3 (the same wind model as
sim_sky_wind.py).  Two ways to fly the same gusts:

  FC ONLY   the phone's LOCK off: yaw stick centred, ArduPilot's own heading
            hold is all there is.
  LOCK      the real mission.Mission with LOCK on, closing the loop on the
            FC's ATTITUDE.yaw at 50 Hz, exactly as on the Jetson.

REAL      Mission (LOCK loop, ramps, rate limits), the PWM round trip.
MODELLED  ArduPilot's yaw controller in ALT_HOLD:
            * the yaw stick is a RATE; the FC integrates it into its heading
              TARGET and holds that target (ATC_ANG_YAW_P = 4.5);
            * a rate loop with an integrator, and a hard ceiling on the yaw
              acceleration the motors can make (yaw is a quad's weakest axis);
            * the attitude-error LIMIT: ArduPilot keeps the yaw error within
              AC_ATTITUDE_ACCEL_Y_CONTROLLER_MAX (120 deg/s^2) / ATC_ANG_YAW_P
              = ~27 deg; a gust that pushes it further DRAGS the target along,
              and the FC then holds the NEW heading.  That is how a gust
              permanently turns a drone that nothing is correcting.
          The gust torque: weathervaning (the airframe tends to turn its nose
          into the wind, stronger with airspeed^2 and with sideslip) plus
          turbulence.  How strongly an airframe weathervanes is not known for
          this one, so three sensitivities are run.
          The heading read by the Jetson: 20 ms late, 0.3 deg of noise.

Assumes the FC's heading ESTIMATE is right (a compass).  Neither mode can fix
an estimate that drifts: LOCK holds what the FC believes.
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
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import mission as M                                                # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    import sim_sky_wind as W                                       # noqa: E402

DT = 0.002
# ArduPilot yaw, ALT_HOLD
ANG_P = 4.5
ERR_MAX = 120.0 / ANG_P             # deg: beyond this the target is dragged
RATE_MAX = 202.5                    # deg/s
RATE_BW = 8.0                       # 1/s, the rate loop
RATE_I = 4.0                        # 1/s^2
I_MAX = 60.0                        # deg/s^2 the integrator may supply
ACC_MAX = 150.0                     # deg/s^2 the motors can make in yaw
# yaw disturbance at 5 m/s airspeed, broadside, deg/s^2
SENS = {"low": 15.0, "medium": 40.0, "high": 90.0}


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


class FcYaw:
    def __init__(self, psi0):
        self.psi = psi0                 # true heading, deg
        self.r = 0.0                    # yaw rate, deg/s
        self.tgt = psi0                 # the FC's heading target
        self.i = 0.0
        self.drags = 0
        self._dragging = False

    def step(self, stick_rate, dist_acc, dt):
        self.tgt = (self.tgt + stick_rate * dt) % 360.0
        e = wrap180(self.tgt - self.psi)
        if abs(e) > ERR_MAX:            # the error limit: the target is dragged
            self.tgt = (self.psi + math.copysign(ERR_MAX, e)) % 360.0
            e = math.copysign(ERR_MAX, e)
            if not self._dragging:
                self.drags += 1
            self._dragging = True
        else:
            self._dragging = False
        rate_des = max(-RATE_MAX, min(RATE_MAX, ANG_P * e + stick_rate))
        er = rate_des - self.r
        self.i = max(-I_MAX, min(I_MAX, self.i + RATE_I * er * dt))
        acc = max(-ACC_MAX, min(ACC_MAX, RATE_BW * er + self.i))
        self.r += (acc + dist_acc) * dt
        self.psi = (self.psi + self.r * dt) % 360.0


class Mapper:
    def phys_to_pwm(self, axis, v):
        return int(round(1500 + 400 * max(-1.0, min(1.0, v / RATE_MAX)))) if axis == "yaw" \
            else int(round(1500 + 10 * v))

    def pwm_to_phys(self, axis, pwm):
        return (pwm - 1500) / 400.0 * RATE_MAX if axis == "yaw" else (pwm - 1500) / 10.0

    def pilot_pwm(self, axis):
        return 1500

    def ai_pwm(self, cmd):
        return {}


def run(args):
    bf, sens, lock, seed, dur = args
    rng = random.Random(seed)
    wind = W.Wind(bf, rng)
    psi0 = rng.uniform(0, 360)
    fc = FcYaw(psi0)
    mp = Mapper()
    m = M.Mission(mp, log=lambda *_a: None)
    hist = []                                       # (t, psi) for the 20 ms lag
    ctx = dict(gate_ok=True, gate_why="", att=dict(yaw=math.radians(psi0)), height=6.0,
               climb=0.0, ai_cmd=None, ai_t=None, det_t=None, det_conf=None)
    m.cycle(0.0, ctx)
    if lock:
        m.request_lock(True)
    turb = 0.0
    out, next_tx, t = {}, 0.0, 0.0
    errs, peak_cmd = [], 0.0
    while t < dur:
        hist.append((t, fc.psi))
        while hist and hist[0][0] < t - 0.020:
            hist.pop(0)
        if t >= next_tx:
            next_tx += 0.02
            meas = hist[0][1] + rng.gauss(0, 0.3)
            ctx["att"] = dict(yaw=math.radians(meas))
            out = m.cycle(t, ctx)
        stick = mp.pwm_to_phys("yaw", out["yaw"]) if "yaw" in out else 0.0
        if lock and t > 1.0:
            peak_cmd = max(peak_cmd, abs(stick))
        wx, wy, _ = wind.step(t, DT)
        v = math.hypot(wx, wy)
        if v > 0.05:
            upwind = math.degrees(math.atan2(-wy, -wx))       # where the wind comes FROM
            beta = wrap180(upwind - fc.psi)
            k = SENS[sens] * (v / 5.0) ** 2
            a = DT / 0.5
            turb += -a * turb + math.sqrt(2 * a) * rng.gauss(0, 1)
            dist = k * math.sin(math.radians(beta)) + 0.35 * k * turb
        else:
            dist = 0.0
        fc.step(stick, dist, DT)
        if t > 2.0 and int(t / DT) % 10 == 0:
            errs.append(abs(wrap180(fc.psi - psi0)))
        t += DT
    errs.sort()
    n = len(errs)
    return dict(bf=bf, sens=sens, lock=lock,
                rms=math.sqrt(sum(e * e for e in errs) / n), p95=errs[int(0.95 * (n - 1))],
                mx=errs[-1], final=abs(wrap180(fc.psi - psi0)),
                in3=sum(1 for e in errs if e < 3.0) / n, drags=fc.drags, peak_cmd=peak_cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lock-max", type=float, default=None, help="try another LOCK_MAX_DPS")
    a = ap.parse_args()
    if a.lock_max:
        M.Mission.LOCK_MAX_DPS = a.lock_max
    seeds = 8 if a.quick else 25
    dur = 30.0 if a.quick else 60.0
    rows = [(bf, s, lk, 1000 * bf + 97 * i + (7 if s == "medium" else 13 if s == "high" else 0), dur)
            for bf in (0, 1, 2, 3) for s in SENS for lk in (False, True) for i in range(seeds)]
    with Pool() as p:
        res = p.map(run, rows, chunksize=2)
    print(f"=== LOCK vs gusts: {len(res)} hovers of {dur:.0f} s, heading error from where it was locked ===")
    print("    airframe yaw sensitivity = how hard a 5 m/s broadside gust tries to turn it")
    for sens in SENS:
        print(f"\n  sensitivity {sens.upper()} ({SENS[sens]:.0f} deg/s^2 at 5 m/s broadside)")
        print(f"    {'':10s}{'':9s}{'typical':>9s}{'p95':>8s}{'worst':>8s}{'at end':>9s}{'< 3 deg':>9s}{'ends >10°':>11s}")
        for bf in (0, 1, 2, 3):
            for lk in (False, True):
                rs = [r for r in res if r["bf"] == bf and r["sens"] == sens and r["lock"] == lk]
                med = sorted(r["rms"] for r in rs)[len(rs) // 2]
                p95 = sorted(r["p95"] for r in rs)[int(0.95 * (len(rs) - 1))]
                mx = max(r["mx"] for r in rs)
                fin = sorted(r["final"] for r in rs)[int(0.95 * (len(rs) - 1))]
                in3 = 100 * sum(r["in3"] for r in rs) / len(rs)
                lost = sum(1 for r in rs if r["final"] > 10.0)
                print(f"    Bf{bf} {'LOCK' if lk else 'FC only':<9s}{med:7.1f}°{p95:7.1f}°{mx:7.1f}°{fin:8.1f}°"
                      f"{in3:8.0f}%{lost:5d}/{len(rs)}")
    lk = [r for r in res if r["lock"]]
    print(f"\n  LOCK's own yaw command, worst of all runs: {max(r['peak_cmd'] for r in lk):.0f} deg/s "
          f"(cap {M.Mission.LOCK_MAX_DPS:.0f})")
    fc = [r for r in res if not r["lock"]]
    print(f"  FC only: runs where a gust dragged the FC's own heading target: "
          f"{sum(1 for r in fc if r['drags'])}/{len(fc)} (it then HOLDS the wrong heading)")
    print("  typical = RMS heading error; at end = 95th percentile of where it ended up;")
    print("  ends >10° = runs that finished more than 10 deg from the locked heading")


if __name__ == "__main__":
    main()
