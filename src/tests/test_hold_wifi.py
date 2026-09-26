#!/usr/bin/env python3
"""A HELD ▲ / ▼ / ⟲ / ⟳ over field WiFi: does it move smoothly, or stutter?

    ../.env/yolo/bin/python tests/test_hold_wifi.py

The page (mission_ui.html holdButton) sends one keepalive every 100 ms while
the finger is down, at most INFLIGHT at once.  Mission drops the hold 0.3 s
(DEADMAN_S) after the last keepalive ARRIVED and ramps the axis back to zero,
so a WiFi stall that starves it for ~0.2 s is a stutter in a hold the operator
never let go of.  With INFLIGHT = 1 (the page before 2026-09-26) one slow
request held back every keepalive behind it; now only a stall hitting three
requests at once can.

This replays that exact page logic against the real Mission, at 50 Hz, over
three link models, and counts how often a continuous 5 s hold sagged.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mission as M                                                # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


class Mapper:
    K = {"throttle": 10.0, "pitch": 10.0, "roll": 10.0, "yaw": 2.0}

    def phys_to_pwm(self, axis, v):
        return int(round(1500 + self.K[axis] * v))

    def pwm_to_phys(self, axis, pwm):
        return (pwm - 1500) / self.K[axis]

    def pilot_pwm(self, axis):
        return 1500

    def ai_pwm(self, cmd):
        return {}


# One HTTP/1.0 request = a fresh TCP connection: handshake + request + reply,
# i.e. ~2 round trips before the reply lands.  `arrive` is when the server
# (and so Mission) sees it; `done` is when the page may send the next one.
LINKS = {
    "good (phone next to the Jetson)": dict(rtt=0.008, jit=0.5, p_spike=0.005, spike=(0.1, 0.25)),
    "field hotspot, 30-80 m":          dict(rtt=0.030, jit=0.6, p_spike=0.03, spike=(0.15, 0.5)),
    "weak (edge of range)":            dict(rtt=0.070, jit=0.7, p_spike=0.08, spike=(0.2, 0.9)),
}


def request(link, rng):
    rtt = link["rtt"] * math.exp(rng.gauss(0, link["jit"]))
    extra = rng.uniform(*link["spike"]) if rng.random() < link["p_spike"] else 0.0
    arrive = 1.5 * rtt + extra * rng.random()        # the stall can hit either leg
    done = 2.0 * rtt + extra + 0.002
    return arrive, max(done, arrive + 0.5 * rtt)


def hold_run(kind, link, seed, hold_s=5.0, inflight_max=3):
    rng = random.Random(seed)
    mp = Mapper()
    m = M.Mission(mp, log=lambda *_a: None)
    ctx = dict(gate_ok=True, gate_why="", att=dict(yaw=0.0), height=8.0, climb=0.0,
               ai_cmd=None, ai_t=None, det_t=None, det_conf=None)
    axis = "throttle" if kind in ("up", "down") else "yaw"
    want = {"up": m.climb_rate(), "down": -m.descent_rate(),
            "yawl": -m.yaw_rate(), "yawr": m.yaw_rate()}[kind]
    t, dt = 0.0, 0.02
    m.cycle(t, ctx)                                  # PILOT -> HOVER
    arrivals = []                                    # keepalives in flight: arrival times
    inflight = []                                    # when each open request completes
    rel_at = None
    next_tick = 0.0
    sags, sagging, low_time, samples = 0, False, 0.0, 0
    while t < hold_s + 2.0:
        # the page: setInterval(send, 100), skip while one is in flight
        if t < hold_s and t >= next_tick:
            next_tick += 0.1
            inflight = [x for x in inflight if x > t]
            if len(inflight) < inflight_max:
                a, d = request(link, rng)
                arrivals.append(t + a)
                inflight.append(t + d)
        if rel_at is None and t >= hold_s:           # finger up: the release, over the same WiFi
            rel_at = t + request(link, rng)[0]
        for at in sorted(x for x in arrivals if x <= t):
            arrivals.remove(at)
            m.request_hold(kind, "s1", at)
        if rel_at is not None and t >= rel_at:
            m.request_release("s1")
            rel_at = float("inf")
        out = m.cycle(t, ctx)
        v = mp.pwm_to_phys(axis, out[axis]) if axis in out else 0.0
        if 0.8 <= t <= hold_s:                       # after the ramp-in
            samples += 1
            low = abs(v) < 0.9 * abs(want)
            if low:
                low_time += dt
            if low and not sagging:
                sags += 1
            sagging = low
        if axis == "yaw" and "yaw" in out:
            ctx["att"]["yaw"] += math.radians(v) * dt
        t += dt
    return sags, low_time / max(1e-9, hold_s - 0.8), abs(v) < 0.05 * abs(want)


print("=== BEFORE: one keepalive in flight (the page until 2026-09-26) ===")
for name, link in LINKS.items():
    rows = [hold_run(k, link, s, inflight_max=1) for k in ("up", "down", "yawl", "yawr") for s in range(50)]
    print(f"  {name:<34s} holds with a stutter: {sum(1 for r in rows if r[0] > 0):3d}/200   "
          f"time below 90%: {100 * sum(r[1] for r in rows) / len(rows):4.1f}%")

print("\n=== NOW: up to three in flight, late keepalives refused after release ===")
results = {}
for name, link in LINKS.items():
    rows = [hold_run(k, link, s) for k in ("up", "down", "yawl", "yawr") for s in range(50)]
    stutter = sum(1 for r in rows if r[0] > 0)
    low = sum(r[1] for r in rows) / len(rows)
    stopped = all(r[2] for r in rows)
    results[name] = (stutter, low, stopped)
    print(f"  {name:<34s} holds with a stutter: {stutter:3d}/200   time below 90%: {100 * low:4.1f}%"
          f"   stopped after release: {'yes' if stopped else 'NO'}")

print()
g = results["good (phone next to the Jetson)"]
f = results["field hotspot, 30-80 m"]
check("good link: a held button never stutters", g[0] == 0, f"{g[0]}/200")
check("field hotspot: under 2% of holds stutter, under 0.5% of the time sagging",
      f[0] <= 4 and f[1] < 0.005, f"{f[0]}/200, {100 * f[1]:.2f}%")
check("every link: letting go stops the axis (the dead-man still works)",
      all(r[2] for r in results.values()))

print()
if FAILED:
    print(f"{len(FAILED)} FAILED:")
    for x in FAILED:
        print("  - " + x)
    sys.exit(1)
print("all passed")
