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
r.m.set_level("vert", "SLOW")              # written for the 1 m/s climb
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

print("=== A4. ceiling, NO ground check, unknown height ===")
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
for h in (0.8, 0.0, -0.5):
    r = Rig()
    r.ctx["height"] = h
    r.step()
    r.run_until(0.6, hold="down")
    check(f"no ground check: ▼ at {h:+.1f} m still descends (the operator lowers it as far as they like)",
          "throttle" in r.out and phys(r.mp, r.out, "throttle") < -0.5 and "floor" not in r.m.event,
          f"{r.out} {r.m.event}")
r = Rig()
r.step()
r.m.set_level("vert", "FAST")
r.run_until(0.6, hold="down")
check("FAST descent at 2 m: allowed too", "throttle" in r.out and phys(r.mp, r.out, "throttle") < -2.0)
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
r.ctx["height"] = 10.0
r.step()
r.run_until(0.3, hold="up")
check("ENGAGE refused while climbing", r.m.request_engage() != "")
r.run_until(1.0)
r.mp.pilot.update(pitch=r.mp.phys_to_pwm("pitch", -2.0),
                  roll=r.mp.phys_to_pwm("roll", 3.0))
check("the shipped defaults are all MEDIUM: PUSH 30 deg, search 6 s, height 3 m/s, turn 20 deg/s",
      all(v == "MEDIUM" for v in r.m.level.values()) and r.m.push_deg() == 30.0 and r.m.search_s() == 6.0
      and r.m.climb_rate() == 3.0 and r.m.yaw_rate() == 20.0, str(r.m.level))
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
    q.ctx["height"] = 10.0                    # the real flow engages at ~10 m
    q.step()
    q.m.set_level("push", "MEDIUM")           # these checks are written for 30 deg
    q.m.set_level("search", "SLOW")           # ...and for the 4 s search
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
q = Rig(min_conf=0.20); q.ctx["height"] = 10.0; q.step()
q.m.request_engage(); q.step()                # the shipped default search
q.run_until(5.8)
check("PUSH with no target: still pushing at 5.8 s (the default search is 6 s)", q.m.phase == "PUSH")
q.run_until(0.3)
check("...soft stop at 6 s", q.m.phase == "STOP")
for lv, secs in (("SLOW", 4.0), ("FAST", 8.0)):
    qs = Rig(min_conf=0.20); qs.ctx["height"] = 10.0; qs.step()
    check(f"search {lv} accepted", qs.m.set_level("search", lv) == "")
    qs.m.request_engage(); qs.step()
    qs.run_until(secs - 0.2)
    still = qs.m.phase
    qs.run_until(0.3)
    check(f"search {lv}: still pushing at {secs - 0.2:.1f} s, soft stop at {secs:.0f} s",
          still == "PUSH" and qs.m.phase == "STOP", f"{still} -> {qs.m.phase}")
check("the shipped default search is 6 s", Rig().m.search_s() == 6.0)
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

print("=== A8c. no AI floor: the AI may descend toward a low target at any height ===")
DOWN = {"pwm": {"roll": 1500, "pitch": 1350, "yaw": 1500, "throttle": 1450}}   # AI wants -5 m/s
for h, vz, want, label in ((12.0, 0.0, "free", "12 m: the AI's descent passes"),
                           (3.8, -2.5, "free", "3.8 m descending at 2.5 m/s: still passes"),
                           (2.5, 0.0, "free", "2.5 m (the old floor): passes"),
                           (1.5, 0.0, "free", "1.5 m: passes (no climb forced)")):
    q = push_rig()
    q.ctx.update(height=h, climb=vz)
    q.run_until(1.52)
    for _ in range(40):
        q.ctx.update(det_t=q.t, det_conf=0.6, ai_cmd=DOWN, ai_t=q.t)
        q.step()
    c = phys(q.mp, q.out, "throttle")
    ok = (c < -4.0) if want == "free" else (c >= 0.0 and c < 0.5) if want == "hold" else (c >= 0.9)
    check(label, q.m.phase == "AI" and ok, f"climb command {c:+.1f} m/s")

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

print("=== A11. LOCK: hold the heading where it was pressed ===")


def hdg(r):
    return math.degrees(r.ctx["att"]["yaw"]) % 360.0


def herr(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


r = Rig()
r.step()                                             # HOVER
r.ctx["att"]["yaw"] = math.radians(0.0)              # facing north
check("LOCK accepted while hovering", r.m.request_lock(True) == "")
r.step()
check("LOCK on: heading captured, phase still HOVER, yaw channel held at centre",
      r.m.lock_hdg is not None and abs(herr(r.m.lock_hdg, 0.0)) < 0.1 and r.m.phase == "HOVER"
      and "yaw" in r.out and abs(phys(r.mp, r.out, "yaw")) < 0.5, f"{r.m.lock_hdg} {r.out}")
r.ctx["att"]["yaw"] = math.radians(25.0)             # a gust knocks it 25 deg right
r.step(1)
turned = phys(r.mp, r.out, "yaw")
r.run_until(6.0)
check("knocked 25 deg: it turns back on its own (negative rate) and settles within 1 deg",
      abs(herr(hdg(r), 0.0)) < 1.0, f"now {hdg(r):.1f} deg")
peak = []
r.ctx["att"]["yaw"] = math.radians(-90.0)
for _ in range(450):                                 # 1 s settle wait + the turn back
    r.step()
    peak.append(abs(phys(r.mp, r.out, "yaw")))
check(f"knocked 90 deg: back without a spin (rate <= {r.m.LOCK_MAX_DPS:.0f} deg/s)",
      max(peak) <= r.m.LOCK_MAX_DPS + 0.5 and abs(herr(hdg(r), 0.0)) < 1.0,
      f"peak {max(peak):.1f} deg/s, now {hdg(r):.1f}")
r.run_until(20.0)
check("no timeout: 20 s later it still holds the yaw channel (a step gives up at 3 s)",
      r.m.lock_hdg is not None and "yaw" in r.out)
r.run_until(1.0, hold="up")
check("▲ while locked: climbs AND keeps the heading", "throttle" in r.out and "yaw" in r.out
      and r.m.lock_hdg is not None)
r.m.request_release("s1"); r.run_until(1.5)
# a held ⟳ turns it away; let go and it comes BACK to the locked heading
held_lock = r.m.lock_hdg
r.run_until(1.0, hold="yawr", session="s2")
away = hdg(r)
check("hold ⟳ while locked: it really turns (SCOUT), the lock itself stays put",
      abs(herr(away, held_lock)) > 10.0 and r.m.phase == "SCOUT"
      and abs(herr(r.m.lock_hdg, held_lock)) < 1e-6, f"heading {away:.1f}, lock {r.m.lock_hdg:.1f}")
r.m.request_release("s2"); r.run_until(4.0)
check("...let go: turns back to the LOCKED heading on its own and holds it",
      abs(herr(r.m.lock_hdg, held_lock)) < 1e-6 and abs(herr(hdg(r), held_lock)) < 1.0
      and r.m.phase == "HOVER", f"lock {r.m.lock_hdg:.1f}, heading {hdg(r):.1f}")
r.m.request_step(-1); r.run_until(0.3)
tapped = hdg(r)
r.run_until(4.0)
check("a tap ⟲ while locked: a small turn, then back to the locked heading (lock unchanged)",
      herr(tapped, held_lock) < -1.0 and abs(herr(r.m.lock_hdg, held_lock)) < 1e-6
      and abs(herr(hdg(r), held_lock)) < 1.0, f"tap {tapped:.1f}, now {hdg(r):.1f}, lock {r.m.lock_hdg:.1f}")
r.run_until(1.0, hold="down", session="s3")
check("▼ while locked: descends AND keeps the heading", "throttle" in r.out and "yaw" in r.out
      and abs(herr(hdg(r), held_lock)) < 1.0 and abs(herr(r.m.lock_hdg, held_lock)) < 1e-6)
r.m.request_release("s3"); r.run_until(1.5)
r.m.request_lock(True); r.step()
check("LOCK on again while locked: nothing moves (idempotent)", abs(herr(r.m.lock_hdg, held_lock)) < 1e-6)

print("=== A11b. LOCK at a picked heading (the degree scroll) ===")
q = Rig(); q.step()
q.ctx["att"]["yaw"] = math.radians(300.0)
check("lock at 313 accepted", q.m.request_lock(True, 313) == "")
q.step()
check("lock_hdg = 313 at once, the drone turning toward it",
      abs(herr(q.m.lock_hdg, 313.0)) < 1e-6 and phys(q.mp, q.out, "yaw") is not None)
q.run_until(3.0)
check("...reaches 313 and holds it", abs(herr(hdg(q), 313.0)) < 1.0, f"now {hdg(q):.1f}")
q.ctx["att"]["yaw"] = math.radians(340.0)            # a gust: 27 deg right
peak = []
for _ in range(250):
    q.step(); peak.append(abs(phys(q.mp, q.out, "yaw")))
check("a gust turns it to 340: back to 313, no spin, and it STAYS",
      abs(herr(hdg(q), 313.0)) < 1.0 and max(peak) <= q.m.LOCK_MAX_DPS + 0.5, f"now {hdg(q):.1f}")
q.run_until(20.0)
check("20 s later: still at 313 (no drift)", abs(herr(hdg(q), 313.0)) < 1.0, f"now {hdg(q):.1f}")
check("pick another heading while locked (200): the lock moves there, the long turn capped",
      q.m.request_lock(True, 200) == "")
peak = []
for _ in range(600):
    q.step(); peak.append(abs(phys(q.mp, q.out, "yaw")))
check("...turns 113 deg the short way to 200 at <= the LOCK cap and holds",
      abs(herr(hdg(q), 200.0)) < 1.0 and max(peak) <= q.m.LOCK_MAX_DPS + 0.5, f"now {hdg(q):.1f}")
q.m.request_lock(True, 355); q.run_until(15.0)
check("across north: lock at 5 from 355 accepted", q.m.request_lock(True, 5) == "")
path = []
for _ in range(250):
    q.step(); path.append(herr(hdg(q), 355.0))
check("...turns +10 the short way (never heads off to the left), and holds 5",
      min(path) > -1.0 and abs(herr(hdg(q), 5.0)) < 1.0, f"min {min(path):.1f}, now {hdg(q):.1f}")
check("a bad heading is refused", q.m.request_lock(True, "x") != "" and q.m.request_lock(True, float("nan")) != "")
check("lock=360 is north", q.m.request_lock(True, 360) == "" and (q.step() or True) and abs(q.m.lock_hdg) < 1e-6)
q.m.request_lock(False); q.run_until(1.0)
check("LOCK off: released, the yaw goes back to the pilot", q.m.lock_hdg is None and "yaw" not in q.out)
qq = Rig(); qq.step(); qq.m.request_engage(); qq.step()
check("a picked heading is refused during PUSH", qq.m.request_lock(True, 90) != "")

r.m.request_lock(True); r.step()
r.m.request_lock(False); r.run_until(1.0)
check("LOCK pressed again: off, the yaw channel goes back to the pilot",
      r.m.lock_hdg is None and "yaw" not in r.out, str(r.out))
r.m.request_lock(True); r.step()
r.m.request_engage(); r.step()
check("ENGAGE clears LOCK (the AI takes the yaw); PUSH starts", r.m.lock_hdg is None and r.m.phase == "PUSH")
check("LOCK refused during PUSH", r.m.request_lock(True) != "")
r = Rig(); r.step(); r.m.request_lock(True); r.step()
r.m.request_abort(); r.step()
check("ABORT clears LOCK", r.m.lock_hdg is None and r.out == {})
r.m.request_lock(True); r.step()
r.ctx["gate_ok"] = False; r.step()
check("the pilot's mode switch clears LOCK", r.m.lock_hdg is None and r.m.phase == "PILOT")
r.ctx["gate_ok"] = True; r.step()
check("...and it does not come back on its own", r.m.lock_hdg is None)
r = Rig(); r.step(); r.ctx["att"] = None; r.m.request_lock(True); r.step()
check("no attitude: LOCK refused, nothing driven", r.m.lock_hdg is None and r.out == {})

print("=== A12. AUTO: up, find, line up, go ===")


def auto_rig(h=2.0):
    q = Rig(min_conf=0.20)
    q.ctx["height"] = h
    q.step()
    q.m.set_level("push", "MEDIUM")           # aim mark -20 deg, push 30 deg
    q.m.set_auto_enabled(True)
    # these checks are written for 10 m, the PUSH lean and the AI steering
    # at once; the operator's defaults are checked in A13
    for k, v in (("height", 10), ("line", 0), ("ai", "SAME"), ("steer", 0)):
        q.m.set_auto_param(k, v)
    return q


def fly(q, secs, det=None):
    """Step with a trivial height plant; `det` = a target seen every frame."""
    for _ in range(int(round(secs / 0.02))):
        if det is not None:
            q.ctx.update(det_t=q.t, det_conf=0.8, ai_cmd=det, ai_t=q.t)
        q.step()
        q.ctx["height"] += phys(q.mp, q.out, "throttle") * 0.02 if "throttle" in q.out else 0.0


def tgt(az=0.0, el=-20.0, rate=0.0, rng=40.0, closing=0.0):
    return dict(AI, az_deg=az, el_cam_deg=el, los_az_dps=rate, range_m=rng, closing_mps=closing)


q = auto_rig()
check("AUTO accepted from HOVER", q.m.request_auto(True) == "")
q.step()
check("AUTO_CLIMB: throttle and yaw only, heading locked where it was",
      q.m.phase == "AUTO_CLIMB" and set(q.out) == {"throttle", "yaw"}
      and q.m.lock_hdg is not None, f"{q.m.phase} {q.out}")
first = phys(q.mp, q.out, "throttle")
fly(q, 1.0)
check("climb ramps in (no step) to the MEDIUM 3 m/s",
      first < 0.2 and abs(phys(q.mp, q.out, "throttle") - 3.0) < 1e-6,
      f"{first:.2f} -> {phys(q.mp, q.out, 'throttle'):.2f}")
fly(q, 6.0)
check("stops at ~10 m without overshooting, AUTO_HOLD",
      q.m.phase == "AUTO_HOLD" and 9.0 < q.ctx["height"] <= 10.3, f"{q.m.phase} h {q.ctx['height']:.1f}")
fly(q, 20.0)
check("no detection: still holding 20 s later, heading locked, nothing else driven",
      q.m.phase == "AUTO_HOLD" and set(q.out) == {"throttle", "yaw"}
      and abs(phys(q.mp, q.out, "throttle")) < 1e-6 and q.m.lock_hdg is not None)
check("▲ / ▼ refused during AUTO", q.m.request_hold("up", "x", q.t) != "")
hd0 = q.m.lock_hdg
for _ in range(50):                            # ⟳ held 1 s while watching
    q.m.request_hold("yawr", "y1", q.t); fly(q, 0.02)
check("⟳ held while watching: it turns, still AUTO_HOLD", q.m.phase == "AUTO_HOLD"
      and phys(q.mp, q.out, "yaw") > 5.0, f"{q.m.phase} {phys(q.mp, q.out, 'yaw'):.1f}")
q.m.request_release("y1"); fly(q, 1.0)
stop_hdg = math.degrees(q.ctx["att"]["yaw"]) % 360.0
fly(q, 3.0)
check("...let go: it stops and LOCKS where it stopped (no turning back)",
      abs(M.wrap180(q.m.lock_hdg - stop_hdg)) < 1.5 and abs(M.wrap180(q.m.lock_hdg - hd0)) > 10.0
      and abs(phys(q.mp, q.out, "yaw")) < 0.5,
      f"lock {q.m.lock_hdg:.1f}, stopped {stop_hdg:.1f}, was {hd0:.1f}")
h1 = q.m.lock_hdg
check("a ⟲ tap while watching: accepted", q.m.request_step(-1) == "")
fly(q, 3.0)
check("...turns one step and locks there", abs(M.wrap180(q.m.lock_hdg - (h1 - q.m.step_deg()))) < 1.5,
      f"{h1:.1f} -> {q.m.lock_hdg:.1f}")
check("LOCK refused during AUTO", q.m.request_lock(True) != "")
h0 = q.ctx["height"]
fly(q, 0.5, det=tgt(az=10.0, el=-5.0))
check("first detection -> AUTO_ALIGN: turns toward it, climbs to bring it down to the mark",
      q.m.phase == "AUTO_ALIGN" and phys(q.mp, q.out, "yaw") > 0
      and phys(q.mp, q.out, "throttle") > 0 and q.m.lock_hdg is None, f"{q.m.phase} {q.out}")
yaws = []
for _ in range(25):
    fly(q, 0.02, det=tgt(az=10.0, el=-5.0)); yaws.append(phys(q.mp, q.out, "yaw"))
check("ALIGN's yaw is rate-limited and capped (never a snap)",
      max(abs(b - a) for a, b in zip(yaws, yaws[1:])) <= q.m.AUTO_YAW_ACCEL * 0.02 + 0.5
      and max(yaws) <= q.m.AUTO_YAW_MAX + 1e-6, f"max {max(yaws):.1f}")
check("⟲ / ⟳ refused while lining up", q.m.request_hold("yawl", "y2", q.t) != "" and q.m.request_step(1) != "")
fly(q, 0.3, det=tgt(az=4.0, el=-12.0))
check("not on the mark yet (8 deg above it): still aligning", q.m.phase == "AUTO_ALIGN")
before = dict(q.out)
fly(q, 0.32, det=tgt(az=2.0, el=-18.0))
check("on the mark for AUTO_READY_S -> the second line: PUSH",
      q.m.phase == "PUSH", q.m.phase)
check("...from the same yaw / throttle it was flying (bumpless)",
      all(abs(q.out[a] - before[a]) <= 30 for a in ("yaw", "throttle")), f"{before} -> {q.out}")
leans = []
for _ in range(40):
    fly(q, 0.02, det=tgt()); leans.append(phys(q.mp, q.out, "pitch"))
check("...the lean ramps in, no jump", max(abs(b - a) for a, b in zip(leans, leans[1:])) < 1.5
      and leans[-1] < -5.0, f"{leans[0]:.1f} .. {leans[-1]:.1f}")
fly(q, 1.5, det=tgt())
check("...then the AI has it, as after ENGAGE", q.m.phase == "AI" and q.m.engaged)

q = auto_rig(h=12.0)
q.m.request_auto(True); q.step()
check("AUTO above 10 m: holds where it is", q.m.phase == "AUTO_HOLD")
fly(q, 0.3, det=tgt(az=8.0))
q.ctx.update(ai_cmd=None, ai_t=None)
fly(q, 0.8)
check("target lost while aligning: back to holding, heading locked again",
      q.m.phase == "AUTO_HOLD" and q.m.lock_hdg is not None, q.m.phase)
q.m.request_auto(False); fly(q, 0.1)
check("AUTO pressed again: soft stop", q.m.phase == "STOP")
fly(q, 1.2)
check("...then HOVER, every channel back to the pilot", q.m.phase == "HOVER" and q.out == {})

q = auto_rig(); q.m.request_auto(True); fly(q, 1.0)
q.m.request_abort(); q.step()
check("ABORT in AUTO: instant, every channel to the pilot", q.m.phase == "HOVER" and q.out == {})
q = auto_rig(); q.m.request_auto(True); fly(q, 1.0)
q.ctx["gate_ok"] = False; q.step()
check("the pilot's mode switch in AUTO: PILOT at once", q.m.phase == "PILOT" and q.out == {})
q = auto_rig(); q.m.request_auto(True); fly(q, 1.0)
q.m.request_engage(); q.step()
check("ENGAGE during AUTO: go now, PUSH", q.m.phase == "PUSH")

q = auto_rig(h=3.2); q.m.request_auto(True); q.step()
fly(q, 1.0, det=tgt(az=0.0, el=-30.0))        # below the mark: would descend
check("ALIGN never descends to the AUTO floor",
      q.ctx["height"] > q.m.AUTO_FLOOR_M and phys(q.mp, q.out, "throttle") >= -1e-6,
      f"h {q.ctx['height']:.2f}")
q = auto_rig(h=12.0); q.m.request_auto(True); q.step()
fly(q, 1.0, det=tgt(az=12.0, el=-20.0, rate=-24.0))   # keeps sliding: never on the mark
check("not lined up after AUTO_ALIGN_MAX_S but near the nose: goes anyway",
      q.m.phase == "AUTO_ALIGN")
fly(q, 1.3, det=tgt(az=12.0, el=-20.0, rate=-24.0))
check("...PUSH", q.m.phase == "PUSH", q.m.phase)
q = auto_rig(h=12.0); q.m.request_auto(True); q.step()
fly(q, 0.3, det=tgt(az=12.0, rng=20.0, closing=10.0))
check("closing fast (2 s to go): not on the first noisy frames", q.m.phase == "AUTO_ALIGN")
fly(q, 0.4, det=tgt(az=12.0, rng=20.0, closing=10.0))
check("...but held 0.5 s: goes at once, no time to line up", q.m.phase == "PUSH", q.m.phase)

q = auto_rig(); q.ctx["height"] = None
q.m.request_auto(True); q.step()
check("no height: AUTO refused, nothing driven", q.m.phase == "HOVER" and q.out == {})
q = auto_rig(); q.ctx["att"] = None
q.m.request_auto(True); q.step()
check("no attitude: AUTO refused", q.m.phase == "HOVER" and q.out == {})
q = auto_rig(); q.m.engaged = True
check("AUTO refused once engaged (one attempt per mode cycle)", q.m.request_auto(True) != "")
q = Rig(); q.ctx["height"] = 2.0; q.step()
check("AUTO OFF at power-up: refused until switched on in PARAMETERS",
      not q.m.auto_enabled and q.m.request_auto(True) != "" and (q.step() or True) and q.m.phase == "HOVER")
check("phone: autoen=on enables it", M.phone_command(q.m, "autoen", "on", q.t) == ""
      and q.m.request_auto(True) == "")
q.step()
check("...and AUTO runs", q.m.phase == "AUTO_CLIMB")
M.phone_command(q.m, "autoen", "off", q.t); q.step()
check("switched OFF while AUTO flies: AUTO stops (soft)", q.m.phase == "STOP")
check("a bad switch value is refused", M.phone_command(q.m, "autoen", "x", q.t) != "")
q = auto_rig()
check("phone: auto=on / auto=off", M.phone_command(q.m, "auto", "on", q.t) == ""
      and (q.step() or True) and q.m.phase == "AUTO_CLIMB"
      and M.phone_command(q.m, "auto", "off", q.t) == "")
print("=== A13. AUTO settings: from the phone, checked, saved ===")
q = auto_rig()
check("the operator's defaults: 15 m, a 30 deg line, AI MAX, the AI steering after 0.6 s",
      M.Mission(FakeMapper(), log=lambda *_: None).auto_params()
      == dict(height=15.0, climb=3.0, turn=30.0, adjust=5.0, window=5.0, ready=0.25, giveup=2.0,
              closing=3.0, floor=3.0, line=30.0, ai="MAX", steer=0.6))
check("every default is one of its own choices",
      all((getattr(M.Mission, a[0]) in c) for _k, a, c in M.Mission.AUTO_SETTINGS))
check("autop=height:15 accepted", M.phone_command(q.m, "autop", "height:15", q.t) == ""
      and q.m.AUTO_HEIGHT_M == 15.0)
check("a value off the list is refused", M.phone_command(q.m, "autop", "height:12", q.t) != ""
      and q.m.AUTO_HEIGHT_M == 15.0)
check("an unknown setting is refused", q.m.set_auto_param("warp", 9) != "")
check("window sets BOTH the left/right and the up/down tolerance",
      q.m.set_auto_param("window", 8) == "" and q.m.AUTO_AZ_OK == 8.0 and q.m.AUTO_EL_OK == 8.0)
check("ai: MAX accepted, 'turbo' refused", q.m.set_auto_param("ai", "max") == ""
      and q.m.AUTO_AI == "MAX" and q.m.set_auto_param("ai", "turbo") != "")
check("one Mission's settings never touch another's / the class defaults",
      M.Mission.AUTO_HEIGHT_M == 15.0 and M.Mission(FakeMapper(), log=lambda *_: None).AUTO_HEIGHT_M == 15.0)
q.m.set_auto_param("line", 40); q.m.set_auto_param("height", 10)
check("outside an AUTO run the AI level is the AI setting", q.m.ai_level_now() == q.m.level["ai"])
q.m.request_auto(True); fly(q, 1.0)
check("AUTO run: the law is told to fly the run's own AI level (MAX)", q.m.ai_level_now() == "MAX")
check("settings refused while AUTO flies", q.m.set_auto_param("height", 5) != ""
      and q.m.reset_auto_params() != "")
q.ctx["height"] = 12.0
fly(q, 0.1, det=tgt()); fly(q, 0.4, det=tgt())
check("the straight line leans the run's own 40 deg (PUSH is 30)", q.m.phase == "PUSH"
      and abs(q.m._line_deg - 40.0) < 1e-6, f"{q.m.phase} {q.m._line_deg}")
q.m.request_abort(); q.step()
check("after the run: back to the AI setting", q.m.ai_level_now() == q.m.level["ai"] and q.m.ai_override is None)
check("autoreset: every setting back to its default", M.phone_command(q.m, "autoreset", "1", q.t) == ""
      and q.m.auto_params() == M.Mission(FakeMapper(), log=lambda *_: None).auto_params())
import tempfile
tf = os.path.join(tempfile.mkdtemp(), "wasp", "auto.json")
q.m.set_auto_param("height", 20); q.m.set_auto_param("ai", "FAST")
check("saved", M.save_auto_params(q.m, tf) and os.path.exists(tf))
q2 = auto_rig()
got = M.load_auto_params(q2.m, tf)
check("...and loaded into a fresh mission (a reboot)", q2.m.AUTO_HEIGHT_M == 20.0 and q2.m.AUTO_AI == "FAST"
      and got.get("height") == 20.0, str(got))
open(tf, "w").write('{"height": 999, "ai": "MAX", "junk": 1}')
q3 = auto_rig()
check("a bad saved value is skipped, the rest applied, nothing breaks",
      M.load_auto_params(q3.m, tf) == {"ai": "MAX"} and q3.m.AUTO_HEIGHT_M == 10.0)  # the rig's 10
check("no saved file: defaults, no error", M.load_auto_params(auto_rig().m, tf + ".none") == {})
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
