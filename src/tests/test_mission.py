#!/usr/bin/env python3
"""Hover -> Scout -> Engage: the mission state machine, and the exact PWM map.

    ../.env/yolo/bin/python tests/test_mission.py

Part A drives mission.Mission with a fake clock and a fake mapper, so every
transition is checked without hardware.  Part B checks ArduPilotAcro's exact
PWM map against an independent model of ArduPilot 4.6's own stick decode,
with THIS FC's calibration.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mission as M                                                # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


class FakeMapper:
    """Linear stand-in: 10 us per m/s of climb, per deg of lean/bank, 2 us per
    deg/s of yaw.  Exactly invertible, so a test can read physical values off
    the PWM the mission produced."""
    K = {"throttle": 10.0, "pitch": 10.0, "roll": 10.0, "yaw": 2.0}

    def __init__(self):
        self.pilot = {}

    def phys_to_pwm(self, axis, v):
        return int(round(1500 + self.K[axis] * v))

    def pwm_to_phys(self, axis, pwm):
        return (pwm - 1500) / self.K[axis]

    def pilot_pwm(self, axis):
        return self.pilot.get(axis)

    def ai_pwm(self, cmd):
        return dict(cmd["pwm"])


def phys(mp, out, axis):
    return None if axis not in out else mp.pwm_to_phys(axis, out[axis])


class Rig:
    """A mission, a clock and a context, stepped at 50 Hz."""

    def __init__(self, **kw):
        self.mp = FakeMapper()
        self.m = M.Mission(self.mp, log=lambda *_a: None, **kw)
        self.t = 100.0
        self.ctx = dict(gate_ok=True, gate_why="", att=dict(yaw=0.0),
                        height=2.0, climb=0.0, ai_cmd=None, ai_t=None,
                        det_t=None, det_conf=None)
        self.out = {}

    def step(self, n=1, hold=None, session="s1"):
        for _ in range(n):
            if hold:
                self.m.request_hold(hold, session, self.t)
            self.out = self.m.cycle(self.t, self.ctx)
            # a trivial yaw plant, so steps can be closed
            if "yaw" in self.out:
                y = self.ctx["att"]["yaw"]
                self.ctx["att"]["yaw"] = y + math.radians(
                    self.mp.pwm_to_phys("yaw", self.out["yaw"])) * 0.02
            self.t += 0.02
        return self.out

    def run_until(self, secs, **kw):
        return self.step(int(round(secs / 0.02)), **kw)


print("=== A1. the pilot's gate ===")
r = Rig()
r.ctx["gate_ok"] = False
check("not armed / not ALT_HOLD: PILOT, nothing driven",
      r.step() == {} and r.m.phase == "PILOT")
r.ctx["gate_ok"] = True
check("armed in ALT_HOLD: HOVER, still nothing driven",
      r.step() == {} and r.m.phase == "HOVER")
check("a hold is refused while the pilot has it",
      Rig().m.request_hold("up", "x", 0.0) != "")

print("=== A2. CLIMB: throttle only, ramped, dead-man ===")
r = Rig()
r.step()
r.m.set_level("vert", "SLOW")
o = r.step(hold="up")
check("▲ held: ONLY the throttle channel is overridden", set(o) == {"throttle"}, str(o))
first = phys(r.mp, o, "throttle")
r.run_until(0.25, hold="up")
mid = phys(r.mp, r.out, "throttle")
r.run_until(0.40, hold="up")
full = phys(r.mp, r.out, "throttle")
check("ramps IN (no step): starts near 0, half-way at ~0.25 s, full by 0.5 s",
      first < 0.05 and 0.3 < mid < 0.7 and abs(full - 1.0) < 1e-6,
      f"{first:.2f} -> {mid:.2f} -> {full:.2f} m/s")
check("phase CLIMB", r.m.phase == "CLIMB")
r.run_until(0.28)                       # keepalives stop: still inside dead-man
still = phys(r.mp, r.out, "throttle")
r.run_until(0.22)
check("keepalives stop: holds for up to 0.3 s, then starts ramping out",
      abs(still - 1.0) < 1e-6 and phys(r.mp, r.out, "throttle") < 0.95,
      f"{still:.2f} then {phys(r.mp, r.out, 'throttle'):.2f}")
r.run_until(0.40)
check("...lands on zero and hands the throttle back to the pilot",
      r.out == {} and r.m.phase == "HOVER")

r = Rig()
r.step()
r.run_until(0.6, hold="up")
r.m.request_release("s1")
r.run_until(0.12)
check("explicit release starts the ramp-out at once (no 0.3 s wait)",
      phys(r.mp, r.out, "throttle") < 0.95, f"{phys(r.mp, r.out, 'throttle'):.2f}")

r = Rig()
r.step()
r.m.set_level("vert", "MEDIUM")
r.run_until(0.7, hold="down")
check("▼ MEDIUM: -1.5 m/s, throttle only, DESCEND",
      abs(phys(r.mp, r.out, "throttle") + 1.5) < 1e-6 and set(r.out) == {"throttle"}
      and r.m.phase == "DESCEND")

for lv, up, dn in (("SLOW", 1.0, 0.5), ("MEDIUM", 3.0, 1.5), ("FAST", 8.0, 2.5)):
    r = Rig()
    r.step()
    r.m.set_level("vert", lv)
    r.run_until(0.7, hold="up")
    u = phys(r.mp, r.out, "throttle")
    r = Rig()
    r.ctx["height"] = 10.0                 # FAST descent needs ~2.7 m to stop
    r.step()
    r.m.set_level("vert", lv)
    r.run_until(0.7, hold="down")
    d = phys(r.mp, r.out, "throttle")
    check(f"vertical {lv}: up {up} / down {dn} m/s", abs(u - up) < 1e-6 and abs(d + dn) < 1e-6,
          f"{u:.1f} / {d:.1f}")

print("=== A3. bumpless: the Jetson starts from the pilot's thumb ===")
r = Rig()
r.step()
r.mp.pilot["throttle"] = r.mp.phys_to_pwm("throttle", 0.4)
o = r.step(hold="up")
check("pilot at +0.4 m/s: first climb command is 0.4, not 0",
      abs(phys(r.mp, o, "throttle") - 0.4) < 0.02, f"{phys(r.mp, o, 'throttle'):.2f}")

print("=== A4. ceiling, floor, unknown height ===")
r = Rig(ceiling_m=30.0)
r.ctx["height"] = 29.5
r.step()
r.run_until(0.3, hold="up")
check("at the ceiling: ▲ refused, nothing driven", r.out == {} and "ceiling" in r.m.event)
r = Rig(ceiling_m=30.0)
r.step()
r.m.set_level("vert", "FAST")
r.run_until(0.7, hold="up")
r.ctx.update(height=20.0, climb=8.0)       # climbing fast, 10 m short
r.run_until(0.1, hold="up")
check("FAST climb 10 m under the ceiling: already easing off (stop distance "
      f"{M.stop_distance(8.0, 5.0, 5.0, 0.5):.1f} m)",
      phys(r.mp, r.out, "throttle") < 8.0)
r = Rig(floor_m=0.5)
r.ctx["height"] = 0.8
r.step()
r.run_until(0.3, hold="down")
check("near the floor: ▼ refused", r.out == {} and "floor" in r.m.event)
r = Rig(floor_m=0.5)
r.step()
r.m.set_level("vert", "FAST")
r.run_until(0.3, hold="down")
check("FAST descent at 2 m: refused (stop distance "
      f"{M.stop_distance(2.5, 5.0, 5.0, 0.5):.1f} m + margin > 1.5 m of room)",
      r.out == {} and "floor" in r.m.event)
r = Rig()
r.ctx["height"] = None
r.step()
r.run_until(0.3, hold="up")
check("height unknown (no VFR_HUD): no climb at all", r.out == {} and "height" in r.m.event)

print("=== A5. SCOUT: yaw only; hold and tap ===")
r = Rig()
r.step()
r.m.set_level("yaw", "MEDIUM")
r.run_until(0.5, hold="yawr")
check("⟳ held: ONLY yaw, at +20 deg/s, SCOUT",
      set(r.out) == {"yaw"} and abs(phys(r.mp, r.out, "yaw") - 20.0) < 1e-6
      and r.m.phase == "SCOUT", str(r.out))
r.run_until(0.7)
check("released: ramps out and hands yaw back", r.out == {} and r.m.phase == "HOVER")
for lv, deg in (("SLOW", 2.0), ("MEDIUM", 5.0), ("FAST", 10.0)):
    r = Rig()
    r.step()
    r.m.set_level("yaw", lv)
    y0 = math.degrees(r.ctx["att"]["yaw"])
    r.m.request_step(-1)
    r.run_until(3.5)
    got = math.degrees(r.ctx["att"]["yaw"]) - y0
    check(f"tap ⟲ {lv}: turns {deg:.0f} deg left and stops (closed on ATTITUDE yaw)",
          abs(got + deg) < 0.6 and r.out == {}, f"{got:+.2f} deg")

print("=== A6. ENGAGE -> PUSH ===")
r = Rig()
r.step()
r.run_until(0.3, hold="up")
check("ENGAGE refused while climbing", r.m.request_engage() != "")
r.run_until(1.0)
r.mp.pilot.update(pitch=r.mp.phys_to_pwm("pitch", -2.0),
                  roll=r.mp.phys_to_pwm("roll", 3.0))
check("the shipped default push is SLOW = 20 deg (lower closing speed, more frames)",
      r.m.level["push"] == "SLOW" and r.m.push_deg() == 20.0)
r.m.set_level("push", "MEDIUM")
check("ENGAGE accepted from HOVER", r.m.request_engage() == "")
o = r.step()
check("PUSH drives all four channels", set(o) == {"roll", "pitch", "yaw", "throttle"}, str(o))
check("...BUMPLESS: pitch and roll start from the pilot's -2 / +3 deg",
      abs(phys(r.mp, o, "pitch") + 2.0) < 0.2 and abs(phys(r.mp, o, "roll") - 3.0) < 0.2,
      f"pitch {phys(r.mp, o, 'pitch'):+.1f} roll {phys(r.mp, o, 'roll'):+.1f}")
leans = []
for _ in range(70):
    leans.append(phys(r.mp, r.step(), "pitch"))
steps = [abs(b - a) for a, b in zip(leans, leans[1:])]
check("lean eases in to -30 deg (MEDIUM) by 1.2 s, largest 20 ms step < 1 deg",
      abs(leans[-1] + 30.0) < 1e-6 and max(steps) < 1.0,
      f"max step {max(steps):.2f} deg")
check("heading and altitude held: yaw 0, climb 0, roll 0",
      phys(r.mp, r.out, "yaw") == 0.0 and phys(r.mp, r.out, "throttle") == 0.0
      and phys(r.mp, r.out, "roll") == 0.0)
for lv, deg in (("SLOW", 20.0), ("FAST", 40.0)):
    q = Rig()
    q.step()
    q.m.set_level("push", lv)
    q.m.request_engage()
    q.run_until(1.4)
    check(f"PUSH {lv}: {deg:.0f} deg", abs(phys(q.mp, q.out, "pitch") + deg) < 1e-6)

print("=== A7. PUSH -> AI: not early, not on junk, smooth ===")
AI = {"pwm": {"roll": 1700, "pitch": 1350, "yaw": 1560, "throttle": 1520}}


def push_rig():
    q = Rig(min_conf=0.20)
    q.step()
    q.m.set_level("push", "MEDIUM")           # these checks are written for 30 deg
    q.m.request_engage()
    q.step()
    return q


def see(q, conf):
    q.ctx.update(det_t=q.t, det_conf=conf, ai_cmd=AI, ai_t=q.t)


q = push_rig()
q.run_until(0.3)
p_before = dict(q.out)
for _ in range(45):                       # detections from 0.3 s to 1.2 s
    see(q, 0.9)
    q.step()
check("a detection early in PUSH: the AI already steers yaw / roll / throttle",
      q.m.phase == "PUSH" and q.m.snapshot(q.t)["ai_steering"]
      and q.out["roll"] > p_before["roll"] + 50 and q.out["yaw"] > 1520,
      f"roll {p_before['roll']}->{q.out['roll']}  yaw {q.out['yaw']}")
check("...but the lean is still the push angle until 1.5 s",
      abs(phys(q.mp, q.out, "pitch") + 30.0) < 0.2, f"{phys(q.mp, q.out, 'pitch'):.1f} deg")
for _ in range(20):
    see(q, 0.9)
    q.step()
check("at 1.5 s the AI takes the lean too: full AI", q.m.phase == "AI")
q = push_rig()
for _ in range(100):
    see(q, 0.15)
    q.step()
check("conf 0.15 < 0.20: junk-level, never steers anything",
      q.m.phase == "PUSH" and q.out["roll"] == 1500 and q.out["yaw"] == 1500
      and not q.m.snapshot(q.t)["ai_steering"])
q = push_rig()
q.run_until(0.5)
for _ in range(30):
    see(q, 0.6)
    q.step()
q.ctx.update(ai_cmd=None, ai_t=None, det_t=None)      # lost mid-push
rolls_back = []
for _ in range(int(M.Mission.BLEND_S / 0.02) + 5):
    rolls_back.append(q.step()["roll"])
_glide = int(200 * 1.5 * 0.02 / M.Mission.BLEND_S) + 3
check("target lost mid-PUSH: steering glides back to heading / height hold, "
      "still pushing (no soft stop yet)",
      q.m.phase == "PUSH" and rolls_back[-1] == 1500
      and max(abs(b - a) for a, b in zip(rolls_back, rolls_back[1:])) <= _glide,
      f"roll back to {rolls_back[-1]}")
q = push_rig()
q.run_until(1.52)
push_out = dict(q.out)
see(q, 0.35)
o = q.step()
check("conf 0.35 after 1.5 s: AI takes over", q.m.phase == "AI")
check("...first AI output == the PUSH output (no step)",
      all(abs(o[a] - push_out[a]) <= 2 for a in o), f"{o} vs {push_out}")
rolls = [o["roll"]]
for _ in range(int(M.Mission.BLEND_S / 0.02) + 5):
    see(q, 0.35)
    rolls.append(q.step()["roll"])
d = [b - a for a, b in zip(rolls, rolls[1:])]
_lim = int(200 * 1.5 * 0.02 / M.Mission.BLEND_S) + 3     # smoothstep's peak slope
check(f"roll glides monotonically to the AI's value, max step <= {_lim} us per 20 ms",
      all(x >= -1 for x in d) and max(d) <= _lim and rolls[-1] == 1700,
      f"max {max(d)} us, final {rolls[-1]}")

print("=== A8. soft stops ===")
q = push_rig()
q.run_until(4.1)
check("PUSH with no target for 4 s: STOP (soft)", q.m.phase == "STOP")
p0 = phys(q.mp, q.out, "pitch")
q.run_until(0.5)
p1 = phys(q.mp, q.out, "pitch")
q.run_until(0.6)
check("...eases the lean back to level over 1 s, then releases",
      p0 < p1 < 0.0 and q.out == {} and q.m.phase == "HOVER",
      f"{p0:.1f} -> {p1:.1f} -> released")
q = push_rig()
q.run_until(1.52)
see(q, 0.5)
q.step()
q.run_until(0.6)                           # detections stop -> stale
check("AI loses the target: NOT a stop -- back to the straight PUSH, looking again",
      q.m.phase == "PUSH", q.m.phase)
q.run_until(1.6)
check("...gliding back to straight: roll 1500, yaw 1500, lean = the push angle",
      q.out["roll"] == 1500 and q.out["yaw"] == 1500 and abs(phys(q.mp, q.out, "pitch") + 30.0) < 0.2,
      str(q.out))
q.run_until(1.2)                           # 2.5 s of re-acquire window gone
q.run_until(1.2)
check("...nothing found in 2.5 s: soft stop, released", q.out == {} and q.m.phase == "HOVER",
      q.m.phase)
q = push_rig()
q.run_until(1.52)
see(q, 0.5)
q.step()
q.run_until(0.8)                           # lost -> back to PUSH
was = q.m.phase
see(q, 0.5)
q.step()
check("lost, then SEEN AGAIN within 2.5 s: straight back to the AI", was == "PUSH" and q.m.phase == "AI",
      f"{was} -> {q.m.phase}")

q = push_rig()
q.run_until(1.52)
see(q, 0.5)
q.step()
for _ in range(60):                         # blend done, AI flying, target in view
    see(q, 0.5)
    q.step()
TERM = {"pwm": {"roll": 1520, "pitch": 1250, "yaw": 1500, "throttle": 1560},
        "terminal": True}
q.ctx.update(ai_cmd=TERM, ai_t=q.t, det_t=q.t, det_conf=0.6)
q.step()
q.ctx.update(ai_cmd=None, ai_t=None)       # the target fills the frame: lost
q.run_until(1.0)
check("TERMINAL commit: detection lost in the last metre -> keeps flying the "
      "frozen command (no soft stop before the touch)",
      q.m.phase == "AI" and q.out.get("pitch") == 1250, f"{q.m.phase} {q.out}")
q.run_until(0.4)
check("...for 1.2 s at most, then the soft stop", q.m.phase in ("STOP", "HOVER"))

print("=== A9. ABORT and the mode switch, from EVERY phase ===")


def in_phase(name):
    q = Rig()
    q.step()
    if name == "HOVER":
        return q, None
    if name == "CLIMB":
        q.run_until(0.3, hold="up")
        return q, "up"
    if name == "DESCEND":
        q.run_until(0.3, hold="down")
        return q, "down"
    if name == "SCOUT":
        q.run_until(0.3, hold="yawl")
        return q, "yawl"
    q.m.request_engage()
    q.step()
    if name == "PUSH":
        return q, None
    q.run_until(1.52)
    see(q, 0.5)
    q.step()
    if name == "AI":
        return q, None
    q.run_until(0.7)                       # target lost -> back to PUSH
    q.run_until(2.6)                       # nothing re-acquired -> STOP
    return q, None


for name in ("HOVER", "CLIMB", "DESCEND", "SCOUT", "PUSH", "AI", "STOP"):
    q, hold = in_phase(name)
    ok_in = q.m.phase == name
    q.m.request_abort()
    o = q.step(hold=hold)                  # a finger still on the button
    check(f"ABORT from {name}: released at once, HOVER, a held button stays dead",
          ok_in and o == {} and q.m.phase == "HOVER"
          and q.step(hold=hold) == {}, f"was {q.m.phase if not ok_in else name}")
    q, hold = in_phase(name)
    q.ctx["gate_ok"] = False
    o = q.step(hold=hold)
    check(f"pilot leaves ALT_HOLD in {name}: released at once, PILOT",
          o == {} and q.m.phase == "PILOT")
q, _ = in_phase("HOVER")
q.m.request_abort()
q.step()
check("after ABORT a NEW press works again",
      set(q.run_until(0.3, hold="up", session="s2")) == {"throttle"})

print("=== A8b. a gap in REAL detections: stop turning, then soft stop ===")
q = push_rig()
q.run_until(1.52)
for _ in range(90):                       # AI established, target in view
    see(q, 0.6)
    q.step()
y0, r0 = q.out["yaw"], q.out["roll"]
t_lost = q.t
q.ctx.update(det_t=t_lost - 0.02, ai_t=t_lost - 0.02)     # detections stop; the cmd stays
yaws, rolls = [], []
for _ in range(40):                       # until the soft stop takes over
    o = q.step()
    if q.m.phase != "AI":
        break
    yaws.append(o["yaw"]); rolls.append(o["roll"])
check("the gap: yaw eases from the AI's turn back to 1500 before the stop (stops turning)",
      y0 > 1540 and abs(yaws[-1] - 1500) <= 3 and all(b <= a + 1 for a, b in zip(yaws, yaws[1:])),
      f"{y0} -> {yaws[-1]} over {len(yaws) * 0.02:.2f} s")
check("...while roll holds the last REAL command", all(r == r0 for r in rolls), f"{r0}")
q.run_until(0.1)
check("0.5 s with no real detection: back to the straight push (not a spin, not a stop)",
      q.m.phase == "PUSH", q.m.phase)

print("=== A9b. one attempt per pilot mode cycle (ENGAGE red -> green -> reset) ===")
q = push_rig()
check("before the AI steers: not engaged (ENGAGE stays red)", not q.m.snapshot(q.t)["engaged"])
q.run_until(0.3)
see(q, 0.6)
q.step()
check("the AI takes a channel: engaged (ENGAGE turns green)", q.m.snapshot(q.t)["engaged"])
q.ctx.update(ai_cmd=None, ai_t=None, det_t=None)
q.run_until(4.5)                               # lost -> back to hold -> timeout -> STOP -> HOVER
q.run_until(1.2)
check("after the attempt, back in HOVER: still engaged, a 2nd ENGAGE is refused",
      q.m.phase == "HOVER" and q.m.snapshot(q.t)["engaged"] and "STAB" in q.m.request_engage())
q.ctx["gate_ok"] = False                       # pilot -> STAB
q.step()
q.ctx["gate_ok"] = True                        # ... -> ALT_HOLD
q.step()
check("pilot STAB -> ALT_HOLD: reset (red), ENGAGE accepted again",
      not q.m.snapshot(q.t)["engaged"] and q.m.request_engage() == "")
q = push_rig()
q.run_until(4.2)                               # PUSH timed out, the AI never steered
q.run_until(1.2)
check("a push that never found a target does NOT latch: ENGAGE again at once",
      q.m.phase == "HOVER" and not q.m.snapshot(q.t)["engaged"] and q.m.request_engage() == "")

print("=== A10. refused during PUSH / AI ===")
q, _ = in_phase("PUSH")
check("holds refused in PUSH", q.m.request_hold("up", "z", q.t) != "")
check("yaw step refused in PUSH", q.m.request_step(1) != "")
check("ENGAGE refused in PUSH", q.m.request_engage() != "")
q, _ = in_phase("AI")
check("holds refused in AI", q.m.request_hold("yawl", "z", q.t) != "")

print("=== B. exact PWM vs ArduPilot 4.6's own decode, this FC ===")
from fc_control import ArduPilotAcro                               # noqa: E402


def ap_climb(pwm, lo=1100, hi=1900, dz=30, thr_dz=100, up=25.0, dn=2.5):
    trim_low = lo + dz
    c = (1000 * (pwm - trim_low)) // (hi - trim_low) if pwm > trim_low else 0
    mid = (1000 * ((lo + hi) // 2 - trim_low)) // (hi - trim_low)
    top, bot = mid + thr_dz, mid - thr_dz
    if c > top:
        return up * (c - top) / (1000.0 - top)
    if c < bot:
        return dn * (c - bot) / bot
    return 0.0


def ap_norm(pwm, lo=1100, hi=1900, tr=1500, dz=20):
    dmin, dmax = tr - dz, tr + dz
    if pwm < dmin:
        n = (pwm - dmin) / float(dmin - lo)
    elif pwm > dmax:
        n = (pwm - dmax) / float(hi - dmax)
    else:
        n = 0.0
    return max(-1.0, min(1.0, n))


bk = ArduPilotAcro.__new__(ArduPilotAcro)
bk.thr_deadzone = 100
bk.rc = {"throttle": dict(chan=1, min=1100, max=1900, trim=1500, reversed=False, dz=30),
         "roll": dict(chan=2, min=1100, max=1900, trim=1500, reversed=False, dz=20),
         "pitch": dict(chan=3, min=1100, max=1900, trim=1500, reversed=False, dz=20),
         "yaw": dict(chan=4, min=1100, max=1900, trim=1500, reversed=False, dz=20)}
for v in (1.0, 3.0, 8.0, 15.0, 25.0):
    pwm = bk._exact_pwm("throttle", v / 25.0)
    got = ap_climb(pwm)
    check(f"climb {v:4.1f} m/s -> ch1 {pwm} -> FC targets {got:.2f}", abs(got - v) < 0.1)
for v in (0.5, 1.5, 2.5):
    pwm = bk._exact_pwm("throttle", -v / 2.5)
    got = ap_climb(pwm)
    check(f"descent {v} m/s -> ch1 {pwm} -> FC targets {got:.2f}", abs(got + v) < 0.03)
for deg in (20.0, 30.0, 40.0):
    pwm = bk._exact_pwm("pitch", -deg / 40.0)
    got = ap_norm(pwm) * 40.0
    check(f"lean {deg:.0f} deg -> ch3 {pwm} -> FC targets {-got:.2f} deg", abs(got + deg) < 0.15)
for rate in (10.0, 20.0, 45.0):
    pwm = bk._exact_pwm("yaw", rate / 202.5)
    got = ap_norm(pwm) * 202.5
    check(f"yaw {rate:.0f} deg/s -> ch4 {pwm} -> FC targets {got:.1f}", abs(got - rate) < 0.6)
check("0 -> exact mid / trim (hold altitude, level, hold heading)",
      all(bk._exact_pwm(a, 0.0) == 1500 for a in bk.rc))
rt = [(a, n) for a in ("pitch", "roll", "yaw") for n in (-0.8, -0.3, 0.25, 0.7)]
rt += [("throttle", n) for n in (-0.6, 0.12, 0.5, 1.0)]
check("decode(encode(x)) == x on every axis (bumpless reads back what was sent)",
      all(abs(bk._exact_norm(a, bk._exact_pwm(a, n)) - n) < 0.01 for a, n in rt))
check("the pilot's centred stick decodes to exactly 0 (inside the dead zones)",
      all(bk._exact_norm(a, 1500 + d) == 0.0 for a in bk.rc for d in (-15, 0, 15)))
bk.rc["pitch"]["reversed"] = True
check("RCn_REVERSED honoured both ways",
      bk._exact_pwm("pitch", -0.5) > 1500 and abs(bk._exact_norm("pitch", bk._exact_pwm("pitch", -0.5)) + 0.5) < 0.01)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED:")
    for f in FAILED:
        print("  - " + f)
    sys.exit(1)
print("all passed")
