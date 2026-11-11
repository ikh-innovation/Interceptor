#!/usr/bin/env python3
"""Write example WASP and Seeker flights through the real logging code.

Nothing here flies.  A fake FC backend and scripted targets feed
AcroCommander, Mission and FlightLogs exactly as on the Jetson, so the
output files match what a real sortie writes:

    control_<stamp>.csv                       whole-session log
    flight_<stamp>_NNN_<mode>.csv             the flight only
    flight_<stamp>_NNN_<mode>.events.jsonl    phase / stage / detection changes
    flight_<stamp>_NNN_<mode>.json            settings and end reason

Two flights are produced:
  1. WASP  – launch, search, guide, detection dropout, reacquisition,
             pilot takeover (STABILIZE), land, disarm.
  2. Seeker (auto) – hover, AUTO climb, hold, align on target,
             PUSH, AI (full guidance), target lost briefly, STOP, disarm.

Usage: python src/tests/make_example_flight.py [out_dir]
"""
import io
import math
import os
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fc_control import AcroCommander, ArduPilotAcro, InterceptLaw
from mission import Mission
from wasp import WaspGuidance

WASP_STAMP = "20261110_101500"
SEEKER_STAMP = "20261110_103000"
DT = 0.02
W, H = 640, 480
HFOV, VFOV = math.tan(.5), math.tan(.4)


# ── helpers ──────────────────────────────────────────────────────────

def _make_backend(rnd):
    """Build a fake ArduPilotAcro with stubs for everything the logger needs."""
    b = ArduPilotAcro("unused",
                      axes=("roll", "pitch", "yaw", "throttle"),
                      verbose=False)
    b.linked = b.armed = True
    b.mode_name = "ALT_HOLD"
    b.angle_max = math.radians(30)
    b.max_climb_mps, b.max_descent_mps = 25, 4
    b.pilot_yaw_rate = math.radians(202.5)
    b.params = dict(ACRO_TRAINER=0, ACRO_RP_RATE=360, ACRO_Y_RATE=202.5,
                    ACRO_RP_EXPO=0, ACRO_Y_EXPO=0, RC_OVERRIDE_TIME=1)
    for ch, axis in enumerate(b.axes, 1):
        b.rc[axis] = dict(chan=ch, min=1100, trim=1500, max=1900, dz=20,
                          reversed=False, from_fc=True, wasp_calibrated=True)
    b._send_override = lambda *_: True
    return b


def _make_commander(out, stamp, rnd, is_wasp):
    """Build an AcroCommander wired up with fake FC and drone-physics state."""
    c = AcroCommander(conn="none", law=InterceptLaw(), mission=True,
                      log_dir=str(out), verbose=False)
    b = _make_backend(rnd)
    c.backend = b
    b.linked, b.armed, b.mode_name = True, False, "ALT_HOLD"

    # Minimal drone physics state
    s = dict(roll=0.0, pitch=0.0, yaw=math.radians(110), rollspeed=0.0,
             pitchspeed=0.0, yawspeed=0.0, h=0.0, climb=0.0)
    b.attitude = lambda: {k: s[k] for k in ("roll", "pitch", "yaw",
                                             "rollspeed", "pitchspeed", "yawspeed")}
    b.height = lambda: (s["h"], s["climb"])
    b.vibration = lambda: dict(x=rnd.uniform(8, 14), y=rnd.uniform(8, 14),
                               z=rnd.uniform(12, 22), clip=0)
    # Fake GPS: 3D fix, 12 sats, position near test site
    gps_rng = random.Random(42)
    gps = dict(lat=32.0654800, lon=34.7766200)
    b._gps = (perf_counter(), 3, 12, gps["lat"], gps["lon"], 50.0)
    b._gpos = (perf_counter(), gps["lat"], gps["lon"], 50.0)
    b._ekf = (perf_counter(), 0x10)  # EKF_POS_HORIZ_ABS, no CONST_POS

    # Fake battery: 6S LiPo (25.2 V full), 15 A typical draw.
    # _battery is (timestamp, voltage_mV, current_cA, remaining_pct).
    batt = dict(v=25200, i=1500, pct=100)
    b._battery = (0.0, batt["v"], batt["i"], batt["pct"])

    def _update_battery(phase, h):
        """Shift fake battery and GPS each tick so logs look realistic."""
        if phase in ("PUSH", "AI", "WASP_FLIGHT"):
            draw = rnd.randint(2200, 2800)
        elif h > 0.5:
            draw = rnd.randint(1200, 1800)
        else:
            draw = rnd.randint(100, 300)
        batt["v"] = max(19500, batt["v"] - rnd.randint(0, 2))
        batt["i"] = draw
        batt["pct"] = max(0, batt["pct"] - rnd.randint(0, 1) * int(rnd.random() < 0.02))
        now = perf_counter()
        b._battery = (now, batt["v"], batt["i"], batt["pct"])
        gps["lat"] += gps_rng.gauss(0, 1e-7)
        gps["lon"] += gps_rng.gauss(0, 1e-7)
        b._gps = (now, 3, 12, gps["lat"], gps["lon"], 50.0 + h)
        b._gpos = (now, gps["lat"], gps["lon"], 50.0 + h)
        b._ekf = (now, 0x10)

    c._update_battery = _update_battery
    c.idle_request_alt_hold = lambda: True
    c.mission = Mission(c, log=lambda *_: None,
                        idle_path=str(out / f"idle_{stamp}.json"))
    c.update_log_context(cam_fps=60, processing_fps=58.7,
                         detector="TensorRT", engine="best.engine",
                         video_recording=False)
    os.environ["WASP_RUN_STAMP"] = stamp
    return c, b, s


# ── WASP flight ──────────────────────────────────────────────────────

def wasp_flight(out):
    """WASP: launch, search, guide, dropout, reacquire, pilot takeover."""
    rnd = random.Random(7)
    c, b, s = _make_commander(out, WASP_STAMP, rnd, is_wasp=True)
    m = c.mission
    t = c._t0
    guide = WaspGuidance()

    ctx = dict(gate_ok=True, gate_why="", att=b.attitude(), height=0.0,
               climb=0.0, fc_mode="ALT_HOLD", idle_ok=True, airborne=False,
               ai_cmd=None, ai_t=None, det_t=None, det_conf=None)

    def tick(dt=DT, target=None):
        nonlocal t
        t += dt
        ctx.update(att=b.attitude(), height=s["h"], climb=s["climb"],
                   airborne=s["h"] > .5)
        if target is not None:
            ex, ey, conf, bw, bh = target
            cmd = guide.observe(ex, ey, W, H, HFOV, VFOV, b.attitude(), t)
            cmd["camera_roll"] = 0
            ctx.update(ai_cmd=cmd, ai_t=t, det_t=t, det_conf=conf)
            c._det_bbox = (int(W/2+ex), int(H/2+ey), bw, bh)
        else:
            ctx.update(ai_cmd=None, ai_t=None, det_t=None, det_conf=None)
            c._det_bbox = None
        c._mission_out = m.cycle(t, ctx)
        cmd = c.law.neutral()
        if ctx.get("ai_cmd") is not None:
            cmd.update(ctx["ai_cmd"])
        c._det_t, c._det_conf = ctx.get("det_t"), ctx.get("det_conf")
        c._cap_age = rnd.uniform(.03, .05)
        rates = m.wasp._rates if m.phase == "WASP_FLIGHT" else (None, None, None)
        for k, r in zip(("rollspeed", "pitchspeed", "yawspeed"), rates):
            want = 0.0 if r is None else r
            s[k] += (want - s[k]) * min(1, dt/.08) + rnd.gauss(0, .01)
        for a, k in (("roll", "rollspeed"), ("pitch", "pitchspeed"),
                     ("yaw", "yawspeed")):
            s[a] += s[k] * dt
        s["roll"] *= .97 if rates[0] is None else 1
        s["pitch"] *= .97 if rates[1] is None else 1
        s["roll"] = max(-.9, min(.9, s["roll"]))
        s["pitch"] = max(-.9, min(.9, s["pitch"]))
        c._update_battery(m.phase, s["h"])
        with redirect_stdout(io.StringIO()):
            c._log_row(t, cmd, None if target is None else target[:2],
                       bool(c._mission_out), bool(c._mission_out))
        if c._log_error:
            raise RuntimeError(c._log_error)

    def run(seconds):
        for _ in range(int(round(seconds / DT))):
            tick()

    # 1. Standby on the ground
    run(2.0)

    # 2. Select WASP, confirm 90-deg mount, arm in ACRO
    assert m.set_wasp_enabled(True, selection=True) == ""
    assert m.confirm_mount(90) == ""
    m.wasp.params["launch_s"] = 3.0
    b.mode_name = ctx["fc_mode"] = "ACRO"
    b.armed = True
    run(1.0)

    # 3. LAUNCH WASP: countdown then flight starts
    assert m.request_engage() == ""
    run(m.wasp.params["launch_s"] + 2 * DT)
    assert m.phase == "WASP_FLIGHT", m.phase

    # 4. Climb-out and search (no target yet)
    pos = dict(x=180.0, y=-120.0, size=6.0)

    def target(conf_lo=.55, conf_hi=.9):
        pos["x"] += -pos["x"] * .04 + rnd.gauss(0, 3)
        pos["y"] += -pos["y"] * .04 + rnd.gauss(0, 3)
        pos["size"] = min(140, pos["size"] * 1.012)
        sz = int(pos["size"])
        return (pos["x"], pos["y"], rnd.uniform(conf_lo, conf_hi),
                sz, int(sz * .7))

    for _ in range(int(2.5 / DT)):
        s["climb"] = min(6.0, s["climb"] + .2)
        s["h"] += s["climb"] * DT
        tick()

    # 5. Target acquired: WASP guides
    for _ in range(int(3.0 / DT)):
        s["h"] += s["climb"] * DT
        tick(target=target())

    # 6. Detection dropout, then reacquisition
    run(0.9)
    pos["x"], pos["y"] = 60.0, 40.0
    for _ in range(int(2.5 / DT)):
        s["h"] += s["climb"] * DT * .3
        tick(target=target(.7, .95))

    # 7. Pilot flips to STABILIZE
    b.mode_name = ctx["fc_mode"] = "STABILIZE"
    ctx.update(gate_ok=False, gate_why="pilot switched to STABILIZE",
               ai_cmd=None, det_t=None, det_conf=None)
    c._mission_why = ctx["gate_why"]
    tick()

    # 8. Pilot descends and disarms
    for _ in range(int(4.0 / DT)):
        s["climb"] = -2.0 if s["h"] > .1 else 0.0
        s["h"] = max(0.0, s["h"] + s["climb"] * DT)
        tick()
    b.armed = False
    ctx["gate_why"] = c._mission_why = "disarmed"
    run(1.0)
    c.close()
    return WASP_STAMP


# ── Seeker (auto) flight ─────────────────────────────────────────────

def seeker_flight(out):
    """Seeker: hover, AUTO climb+hold, align, PUSH, AI, STOP, disarm."""
    rnd = random.Random(42)
    c, b, s = _make_commander(out, SEEKER_STAMP, rnd, is_wasp=False)
    m = c.mission
    law = c.law
    t = c._t0

    ctx = dict(gate_ok=True, gate_why="", att=b.attitude(), height=0.0,
               climb=0.0, fc_mode="ALT_HOLD", idle_ok=True, airborne=False,
               ai_cmd=None, ai_t=None, det_t=None, det_conf=None)

    def tick(dt=DT, target=None):
        nonlocal t
        t += dt
        ctx.update(att=b.attitude(), height=s["h"], climb=s["climb"],
                   airborne=s["h"] > .5)
        if target is not None:
            ex, ey, conf, bw, bh = target
            # Use InterceptLaw.compute to get the full Seeker ai_cmd
            cmd = law.compute(ex, ey, W, H, dt=dt, bbox_w=bw, bbox_h=bh,
                              att=b.attitude(), commit=1.0, capture_age=.04,
                              rate_mode=True)
            ctx.update(ai_cmd=cmd, ai_t=t, det_t=t, det_conf=conf)
            c._det_bbox = (int(W/2+ex), int(H/2+ey), bw, bh)
        else:
            ctx.update(ai_cmd=None, ai_t=None, det_t=None, det_conf=None)
            c._det_bbox = None
        c._mission_out = m.cycle(t, ctx)
        cmd = law.neutral()
        if ctx.get("ai_cmd") is not None:
            cmd.update(ctx["ai_cmd"])
        c._det_t, c._det_conf = ctx.get("det_t"), ctx.get("det_conf")
        c._cap_age = rnd.uniform(.03, .05)
        # Seeker doesn't use wasp rates; simple attitude model
        for k in ("rollspeed", "pitchspeed", "yawspeed"):
            s[k] *= .9
            s[k] += rnd.gauss(0, .005)
        for a, k in (("roll", "rollspeed"), ("pitch", "pitchspeed"),
                     ("yaw", "yawspeed")):
            s[a] += s[k] * dt
        s["roll"] = max(-.6, min(.6, s["roll"] * .98))
        s["pitch"] = max(-.6, min(.6, s["pitch"] * .98))
        c._update_battery(m.phase, s["h"])
        with redirect_stdout(io.StringIO()):
            c._log_row(t, cmd, None if target is None else target[:2],
                       bool(c._mission_out), bool(c._mission_out))
        if c._log_error:
            raise RuntimeError(c._log_error)

    def run(seconds):
        for _ in range(int(round(seconds / DT))):
            tick()

    # Approaching target: starts far right & above, drifts to centre
    pos = dict(x=100.0, y=-60.0, size=8.0)

    def target(conf_lo=.6, conf_hi=.95):
        pos["x"] += -pos["x"] * .02 + rnd.gauss(0, 2)
        pos["y"] += -pos["y"] * .02 + rnd.gauss(0, 2)
        pos["size"] = min(200, pos["size"] * 1.008)
        sz = int(pos["size"])
        return (pos["x"], pos["y"], rnd.uniform(conf_lo, conf_hi),
                sz, int(sz * .8))

    # 1. Ground standby (disarmed)
    ctx["gate_ok"] = False
    ctx["gate_why"] = "disarmed"
    run(1.0)

    # 2. Arm, climb to a hover
    b.armed = True
    b.mode_name = ctx["fc_mode"] = "ALT_HOLD"
    ctx.update(gate_ok=True, gate_why="")
    # First tick with gate_ok transitions PILOT -> HOVER
    tick()
    for _ in range(int(3.0 / DT)):
        s["climb"] = min(3.0, s["climb"] + .15)
        s["h"] += s["climb"] * DT
        tick()
    s["climb"] = 0.0
    run(1.0)
    assert m.phase == "HOVER", f"expected HOVER, got {m.phase}"

    # 3. Enable AUTO and engage
    assert m.set_auto_enabled(True) == "", m.set_auto_enabled(True)
    err = m.request_auto(True)
    assert err == "", f"request_auto failed: {err}"
    # One tick to let cycle() process the request
    tick()
    assert m.phase in ("AUTO_CLIMB", "AUTO_HOLD"), m.phase

    # 4. AUTO_CLIMB to the target altitude
    for _ in range(int(4.0 / DT)):
        s["climb"] = min(3.0, s["climb"] + .1)
        s["h"] += s["climb"] * DT
        tick()

    # 5. AUTO_HOLD: scanning for the target
    s["climb"] = 0.0
    run(2.0)

    # 6. First detection: AUTO_ALIGN starts
    for _ in range(int(3.0 / DT)):
        tick(target=target())

    # 7. PUSH and AI: the intercept law steers toward the target
    for _ in range(int(6.0 / DT)):
        # Target grows and centres as range closes
        s["pitch"] -= .002  # nose down slightly
        tick(target=target(.75, .98))

    # 8. Brief target loss during AI, then the law stops
    for _ in range(int(1.5 / DT)):
        tick()   # no target

    # 9. Pilot flips to STABILIZE
    b.mode_name = ctx["fc_mode"] = "STABILIZE"
    ctx.update(gate_ok=False, gate_why="pilot switched to STABILIZE")
    c._mission_why = ctx["gate_why"]
    tick()

    # 10. Descent and disarm
    for _ in range(int(4.0 / DT)):
        s["climb"] = -2.0 if s["h"] > .1 else 0.0
        s["h"] = max(0.0, s["h"] + s["climb"] * DT)
        tick()
    b.armed = False
    ctx["gate_why"] = c._mission_why = "disarmed"
    run(1.0)
    c.close()
    return SEEKER_STAMP


# ── main ─────────────────────────────────────────────────────────────

def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Clean previous runs
    for stamp in (WASP_STAMP, SEEKER_STAMP):
        for old in out.glob(f"*{stamp}*"):
            old.unlink()

    print("=== WASP flight ===")
    wasp_flight(out)
    for p in sorted(out.glob(f"*{WASP_STAMP}*")):
        print(f"  {p.stat().st_size:>9}  {p.name}")

    print("\n=== Seeker (auto) flight ===")
    seeker_flight(out)
    for p in sorted(out.glob(f"*{SEEKER_STAMP}*")):
        print(f"  {p.stat().st_size:>9}  {p.name}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[2]
    main(sys.argv[1] if len(sys.argv) > 1 else here / "examples" / "flight_logs")
