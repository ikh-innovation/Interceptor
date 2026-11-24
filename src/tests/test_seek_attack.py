"""SEEK's ALT_HOLD attack (seek_attack.py): the stick logic, and a small ALT_HOLD
simulation (climb limited by PILOT_SPEED_UP / PILOT_ACCEL_Z, lean with a lag,
a camera fixed at 90 degrees that sees only inside its field of view) in which
the drone, 10 m under the target, has to reach it.

    python src/tests/test_seek_attack.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seek_attack import SeekAttack

FAILED = []


def check(label, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(label)


class Backend:
    max_climb_mps = 25.0           # PILOT_SPEED_UP 2500 on the FC


class Mapper:
    """Returns the physical stick values instead of PWM, for the simulation."""
    backend = Backend()

    def idle_pwm(self, yaw_dps, climb_mps):
        return dict(roll=0.0, pitch=0.0, yaw=yaw_dps, throttle=climb_mps / self.backend.max_climb_mps)

    def lean_pwm(self, roll_deg, pitch_deg):
        return dict(roll=roll_deg, pitch=pitch_deg)


def obs(n, e, d, t, rate=(0.0, 0.0, 0.0)):
    k = math.sqrt(n*n + e*e + d*d)
    return dict(bearing=(n/k, e/k, d/k), bearing_rate=rate, capture_t=t)


M = Mapper()
ATT0 = dict(roll=0.0, pitch=0.0, yaw=0.0)

# ---- the sticks ---------------------------------------------------------------
a = SeekAttack()
check("not attacking: no sticks", a.output(M, 0.0, {}) == {})
a.begin(0.0)
out = a.output(M, 0.1, dict(ai_cmd=obs(0, 0, -10, 0.05), att=ATT0))
check("target straight overhead: full climb, level", out["throttle"] == 1.0 and abs(out["roll"]) < 1e-6
      and abs(out["pitch"]) < 1e-6 and out["yaw"] == 0.0, out)
out = a.output(M, 0.2, dict(ai_cmd=obs(10*math.tan(math.radians(20)), 0, -10, 0.15), att=ATT0))
check("target 20° north, nose north: nose down (pitch < 0), no roll, full climb",
      out["pitch"] < -15 and abs(out["roll"]) < 1e-6 and out["throttle"] == 1.0, out)
out = a.output(M, 0.3, dict(ai_cmd=obs(10*math.tan(math.radians(20)), 0, -10, 0.25),
                            att=dict(roll=0.0, pitch=0.0, yaw=math.radians(90))))
check("same target, nose east: roll left (roll < 0), no pitch", out["roll"] < -15 and abs(out["pitch"]) < 1e-6, out)
out = a.output(M, 0.38, dict(ai_cmd=obs(0, 10*math.tan(math.radians(28)), -10, 0.33), att=ATT0))
check("target 28° off: climb reduced while it leans under it", 0.0 < out["throttle"] < 1.0, out)
out = a.output(M, 0.4, dict(ai_cmd=obs(0, 10*math.tan(math.radians(45)), -10, 0.35), att=ATT0))
check("target 45° off: lean capped at 35°, no climb until it is under it", abs(out["roll"]) <= 35.0 + 1e-6
      and out["throttle"] == 0.0, out)
out = a.output(M, 0.5, dict(ai_cmd=obs(0, 0, -10, 0.0), att=ATT0))
check("a stale detection (0.5 s old): sticks centred (ALT_HOLD holds)", out["throttle"] == 0.0
      and out["roll"] == 0.0 and out["pitch"] == 0.0, out)
a.output(M, 0.45 + SeekAttack.LOST_S + 0.1, {})
check(f"no detection for {SeekAttack.LOST_S:g} s: the attack is over", a.done is not None and "not seen" in a.done, a.done)
out = a.output(M, 3.0, dict(ai_cmd=obs(0, 0, -10, 2.95), att=ATT0))
check("over: sticks stay centred (hover) even if it sees it again", out["throttle"] == 0.0 and out["pitch"] == 0.0)
a.begin(10.0)
a.output(M, 10.1, dict(ai_cmd=obs(10, 0, 0.5, 10.05), att=ATT0))
check("target no longer above (passed): the attack is over", a.done is not None and "passed" in a.done, a.done)
a.begin(20.0)
for i in range(1, 400):
    a.output(M, 20.0 + i*0.05, dict(ai_cmd=obs(0, 0, -10, 20.0 + i*0.05 - 0.02), att=ATT0))
check(f"capped at {SeekAttack.MAX_S:g} s", a.done is not None and "longer" in a.done, a.done)
a.stop()
check("stopped (ABORT / RTL / takeover): no sticks", a.output(M, 50.0, {}) == {})


# ---- closed loop: ALT_HOLD simulation -------------------------------------------
G = 9.81
HFOV_HALF = math.radians(31.0)      # IMX219: 62 degrees across


def fly(tgt_vel, tgt_off=(3.0, 2.0), below=10.0, dt=0.01, latency=0.05, hit_m=1.0):
    a = SeekAttack()
    a.begin(0.0)
    p = [0.0, 0.0, 30.0]                     # north, east, up
    v = [0.0, 0.0, 0.0]
    tp = [tgt_off[0], tgt_off[1], 30.0 + below]
    roll = pitch = 0.0
    yaw = 0.0
    hist = []                                # (t, bearing) for latency and the bearing rate
    t, best = 0.0, 1e9
    cmd = dict(roll=0.0, pitch=0.0, throttle=0.0)
    while t < 12.0:
        rel = [tp[i] - p[i] for i in range(3)]
        dist = math.sqrt(sum(x*x for x in rel))
        best = min(best, dist)
        if dist < hit_m:
            return True, best, t, a.done
        # Camera along body up: sees the target only inside its field of view.
        up_body = (math.sin(-pitch) * 0 + math.sin(math.radians(0)), 0, 0)   # placeholder, replaced below
        cr, sr, cp, sp = math.cos(math.radians(roll)), math.sin(math.radians(roll)), math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        # body z-up axis in world (n, e, up) for yaw 0: tilting nose down (pitch<0) points it north.
        axis = (-sp * cr, sr, cp * cr)
        cosang = sum(axis[i] * rel[i] for i in range(3)) / max(dist, 1e-9)
        seen = cosang > math.cos(HFOV_HALF) and dist < 80
        bearing = (rel[0]/dist, rel[1]/dist, -rel[2]/dist)
        hist.append((t, bearing, seen))
        ctx = dict(att=dict(roll=math.radians(roll), pitch=math.radians(pitch), yaw=yaw))
        old = [h for h in hist if h[0] <= t - latency]
        if old and old[-1][2]:
            t0, b0, _ = old[-1]
            prev = [h for h in hist if h[0] <= t0 - 0.05]
            rate = (0.0, 0.0, 0.0)
            if prev:
                t1, b1, _ = prev[-1]
                rate = tuple((x - y) / max(t0 - t1, 1e-3) for x, y in zip(b0, b1))
            ctx["ai_cmd"] = dict(bearing=b0, bearing_rate=rate, capture_t=t0)
        cmd = a.output(M, t, ctx)
        # ALT_HOLD: lean follows the stick with a lag, climb rate with PILOT_ACCEL_Z.
        roll += (cmd["roll"] - roll) * min(1.0, dt / 0.15)
        pitch += (cmd["pitch"] - pitch) * min(1.0, dt / 0.15)
        want_vz = cmd["throttle"] * 25.0 if cmd["throttle"] > 0 else 0.0
        v[2] += max(-5.0 * dt, min(5.0 * dt, want_vz - v[2]))
        an = G * math.tan(math.radians(-pitch)) - 0.3 * v[0]
        ae = G * math.tan(math.radians(roll)) - 0.3 * v[1]
        v[0] += an * dt
        v[1] += ae * dt
        for i in range(3):
            p[i] += v[i] * dt
            tp[i] += tgt_vel[i] * dt
        t += dt
    return False, best, t, a.done


SCENARIOS = [((0, 0, 0), "standing"), ((0, 5, 0), "5 m/s sideways"), ((-8, 0, 0), "8 m/s"),
             ((0, 10, 0), "10 m/s sideways"), ((3, 3, 1), "moving and climbing"), ((0, -7, 0), "7 m/s the other way"),
             ((6, 6, 0), "8.5 m/s diagonal"), ((0, 12, 0), "12 m/s"), ((-10, 4, 0), "10.8 m/s")]
OFFSETS = [(3, 2), (-4, 1), (0, -5), (1, 1)]
for vel, label in SCENARIOS:
    res = [fly(vel, tgt_off=o) for o in OFFSETS]
    worst = max(r[1] for r in res)
    check(f"reaches a target 10 m above, {label}, from {len(OFFSETS)} starting offsets",
          all(r[0] for r in res), f"worst closest approach {worst:.2f} m, longest {max(r[2] for r in res):.1f} s")
res = [fly(v, tgt_off=o, latency=0.1) for v, _ in SCENARIOS for o in OFFSETS]
check("still reaches them with twice the camera delay (0.1 s)", all(r[0] for r in res),
      f"{sum(r[0] for r in res)}/{len(res)}")

print("\nALL PASS" if not FAILED else f"\n{len(FAILED)} FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
