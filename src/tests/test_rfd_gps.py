#!/usr/bin/env python3
"""RFD line format and the Seeker GO path: climb, fly, line up, lean."""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mission as M
from flight_logging import FlightLogs
from rfd_gps import RfdLink, checksum, format_line, ned_m, parse_line

FAILED = []


def check(label, ok, detail=""):
    if not ok:
        FAILED.append(f"{label} {detail}".rstrip())


class FakeMapper:
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
    def __init__(self):
        self.mp = FakeMapper()
        self.m = M.Mission(self.mp, log=lambda *_a: None)
        self.t = 100.0
        self.ctx = dict(gate_ok=True, gate_why="", att=dict(yaw=0.0),
                        height=2.0, climb=0.0, ai_cmd=None, ai_t=None,
                        det_t=None, det_conf=None,
                        gpos=dict(lat=37.9, lon=23.7, alt_m=100.0, rel_m=2.0))
        self.out = {}

    def step(self, n=1):
        for _ in range(n):
            self.out = self.m.cycle(self.t, self.ctx)
            self.t += 0.02
        return self.out

    def run_until(self, secs):
        return self.step(int(round(secs / 0.02)))


def fresh(lat, lon, alt=10.0, frame="REL"):
    return dict(lat=lat, lon=lon, alt_m=alt, frame=frame, age_s=0.0, fresh=True)


def see(r, el=-20.0, az=0.0):
    r.ctx["det_t"] = r.t
    r.ctx["ai_t"] = r.t
    r.ctx["det_conf"] = 0.8
    r.ctx["ai_cmd"] = dict(az_deg=az, el_cam_deg=el, los_az_dps=0.0,
                           pwm=dict(roll=1500, pitch=1500, yaw=1500, throttle=1500))


line = format_line(37.9834123, 23.7271000, 12.5, "REL")
fix = parse_line(line)
check("a good line round-trips",
      fix == dict(lat=37.9834123, lon=23.7271000, alt_m=12.5, frame="REL", other=[]),
      str(fix))
check("a bad checksum is dropped", parse_line(line[:-2] + "00") is None)
check("junk is dropped", parse_line("not a fix") is None and parse_line("$RFD,1,2*00") is None)
body = "RFD,91,0,1,REL"
check("latitude out of range is dropped", parse_line(f"${body}*{checksum(body)}") is None)
body = "RFD,1.0,2.0,3.00,NOPE"
check("a bad frame is dropped", parse_line(f"${body}*{checksum(body)}") is None)
body = "RFD,1.0000000,2.0000000,3.00,MSL,note"
got = parse_line(f"${body}*{checksum(body)}")
check("extra words are kept", got and got["frame"] == "MSL" and got["other"] == ["note"], str(got))
north, east = ned_m(0.0, 0.0, 1.0, 1.0)
check("one degree north and east at the equator",
      math.isclose(north, math.radians(1.0) * 6378137.0, rel_tol=1e-9)
      and math.isclose(east, north, rel_tol=1e-9), f"{north} {east}")

link = RfdLink("/dev/does-not-matter")
check("ingest rejects a bad line", link.ingest("nope", now=10.0) is False and link.snapshot(10.0) is None)
check("ingest keeps a fresh fix", link.ingest(line, now=10.0) and link.snapshot(11.0)["fresh"] is True)
check("an old fix is stale", link.snapshot(13.0)["fresh"] is False)

r = Rig()
r.step()
check("GO with no radio", r.m.request_go() == "no fresh RFD GPS: press CHECK", r.m.request_go())
r.m.rfd = dict(lat=1.0, lon=2.0, alt_m=3.0, frame="REL", age_s=4.0, fresh=False)
check("GO with a stale fix", "stale" in r.m.rfd_status() and r.m.request_go().startswith("no fresh"),
      r.m.rfd_status())
r.m.rfd = fresh(37.91, 23.7)
check("CHECK names the fresh fix", r.m.rfd_status().startswith("RFD 37.91000, 23.70000"), r.m.rfd_status())


class Backend:
    def __init__(self):
        self.armed = False
        self.mode_name = "ALT_HOLD"

    def position_ok(self):
        return self.mode_name == "ALT_HOLD" and self.armed


r.m.mapper.backend = Backend()
check("GO while disarmed", r.m.request_go() == "arm in ALT_HOLD before GO", r.m.request_go())
r.m.mapper.backend.armed = True
r.m.mapper.backend.mode_name = "STABILIZE"
check("GO outside ALT_HOLD", "not ALT_HOLD" in r.m.request_go(), r.m.request_go())
r.m.mapper.backend.mode_name = "ALT_HOLD"
r.m.wasp.enabled = True
check("GO is refused in WASP", "WASP" in r.m.request_go(), r.m.request_go())
r.m.wasp.enabled = False
r.m.mapper.backend = None

r = Rig()
r.step()
r.ctx["gpos"] = None
r.m.rfd = fresh(37.91, 23.7)
check("GO is accepted before the cycle sees position", r.m.request_go() == "")
check("no flight-controller position: stay in HOVER", r.step() == {} and r.m.phase == "HOVER", r.m.phase)

r = Rig()
r.step()
r.m.set_level("push", "SLOW")
r.m.rfd = fresh(37.9 + 0.001, 23.7, alt=4.0)
check("GO accepted", r.m.request_go() == "" and r.m.gps_pending)
o = r.step()
check("from 2 m it climbs", r.m.phase == "GPS_CLIMB" and r.m.gps_run and not r.m.auto_run, r.m.phase)
check("climb throttle is above hover and the nose stays level",
      o.get("throttle", 0) > 1500 and abs(o.get("pitch", 0) - 1500) <= 1 and o.get("roll") == 1500,
      str(o))
r.run_until(1.2)
check("the climb settles near 3 m/s", abs(phys(r.mp, r.out, "throttle") - 3.0) < 0.15
      and r.m.phase == "GPS_CLIMB", f"{phys(r.mp, r.out, 'throttle')} {r.m.phase}")
r.ctx["height"] = 5.0
r.step()
check("at 5 m it flies to the point", r.m.phase == "GPS_GOTO", r.m.phase)
r.run_until(0.7)
check("nose down toward a point still ahead", phys(r.mp, r.out, "pitch") < -10.0, str(r.out))
check("the lean is capped", phys(r.mp, r.out, "pitch") >= -r.m.GPS_LEAN_MAX - 0.2, str(r.out))

r = Rig()
r.step()
r.ctx["height"] = 8.0
r.m.rfd = fresh(37.9, 23.7 + 0.001, alt=8.0)
check("GO toward a point to the east", r.m.request_go() == "")
r.step()
check("already high enough: go straight there", r.m.phase == "GPS_GOTO", r.m.phase)
r.run_until(0.8)
check("yaw is to the east", phys(r.mp, r.out, "yaw") > 20.0, str(r.out))
check("roll stays neutral", r.out.get("roll") == 1500, str(r.out))
r.m.rfd = None
r.run_until(2.5)
check("a short gap keeps the GO alive", r.m.phase == "GPS_GOTO", r.m.phase)
r.run_until(0.7)
check("three seconds without RFD is a soft stop", r.m.phase == "STOP", r.m.phase)

r = Rig()
r.step()
r.ctx["height"] = 8.0
r.m.set_level("push", "SLOW")
r.m.rfd = fresh(37.9, 23.7, alt=8.0)
see(r)
check("GO on the point", r.m.request_go() == "")
r.step()
check("on the lower mark it lines up before the lean", r.m.phase == "GPS_ALIGN", r.m.phase)
for _ in range(20):
    see(r)
    r.step()
    if r.m.phase == "PUSH":
        break
check("lined up for a quarter second: PUSH", r.m.phase == "PUSH" and r.m.gps_run and not r.m.auto_run,
      r.m.phase)
lean = None
for _ in range(90):
    see(r)
    r.step()
    if r.m.phase == "PUSH" and r.t - r.m._push_t >= 1.30:
        lean = dict(r.out)
        break
check("the lean is the 30 degree line, not the 20 degree stick setting",
      lean is not None and abs(phys(r.mp, lean, "pitch") + 30.0) < 0.6, str(lean))
for _ in range(40):
    see(r)
    r.step()
    if r.m.phase == "AI":
        break
check("the AI then takes the run", r.m.phase == "AI" and r.m.engaged, r.m.phase)

r = Rig()
r.step()
r.ctx["height"] = 8.0
r.m.rfd = fresh(37.9, 23.7, alt=8.0)
r.m.request_go()
r.step()
check("STOP while going there", r.m.phase == "GPS_GOTO")
r.m.request_go(False)
r.step()
check("GO off is a soft stop", r.m.phase == "STOP", r.m.phase)

home = dict(alt_m=100.0, rel_m=8.0)
check("MSL becomes height above home",
      abs(r.m._gps_target_height(dict(frame="MSL", alt_m=120.0), home) - 28.0) < 1e-6)
check("REL above the ceiling is capped",
      abs(r.m._gps_target_height(dict(frame="REL", alt_m=80.0), home) - (r.m.ceiling_m - 1.0)) < 1e-6)

with tempfile.TemporaryDirectory() as d:
    logs = FlightLogs(d, "stamp", "phase")
    logs._start(1.0, dict(wasp_enabled=False, auto_run=False, gps_run=True,
                          phase="GPS_GOTO", t_s=1.0), lambda: {})
    check("a GO flight is logged as gps", logs.active["flight_mode"] == "gps", str(logs.active))
    logs._csv.close()
    logs._events.close()

if FAILED:
    for item in FAILED:
        print("FAIL", item)
    sys.exit(1)
print("ok", "rfd")
