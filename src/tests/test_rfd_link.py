"""The laptop <-> drone RFD link end to end, and the EKETA-EAB gateway messages.

A pseudo-terminal pair stands in for the two RFD radios: the Jetson's real
AsterixCue + SeekerMissions on one end, the laptop's real RfdLink on the other.
A SEEK:FOLLOW is flown on the simulated copter of test_seeker_missions.py with
only the radio in between: target lock, follow 10 m below, no AUTO ENGAGE on a
camera lock, a new target offered and swapped to, RTL.

    python src/tests/test_rfd_link.py
"""
import ast
import json
import os
import pty
import select
import sys
import tempfile
import threading
import time
import tty

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import asterix_cue as A
import seeker_missions as S
import rfd_ground as G

FAILED = []


def check(label, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(label)


# The simulated copter, mapper and mission of test_seeker_missions.py, without its tests.
_src = open(os.path.join(HERE, "test_seeker_missions.py")).read()
_tree = ast.parse(_src)
_tree.body = [n for n in _tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
              or (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets))]
H = {"__file__": os.path.join(HERE, "test_seeker_missions.py"), "__name__": "seeker_harness"}
exec(compile(_tree, "test_seeker_missions.py", "exec"), H)


class PtyEnd:
    """The laptop's end of the 'radio' (the pty master) with a serial-like API."""

    def __init__(self, fd):
        self.fd = fd

    def read(self, n):
        r, _, _ = select.select([self.fd], [], [], 0.2)
        return os.read(self.fd, n) if r else b""

    def write(self, data):
        return os.write(self.fd, data)

    def close(self):
        pass


# ---- protocol units ----------------------------------------------------------

line = A.rfd_msg_line("RFDTEL", A.rfd_json({"why": "a*b$c", "n": 1}))
check("status line round-trips with '*' and '$' in the text",
      json.loads(A.parse_rfd_msg(line, "RFDTEL")) == {"why": "a*b$c", "n": 1}, line)
check("a corrupted status line is rejected", A.parse_rfd_msg(line.replace("1", "2"), "RFDTEL") is None)
check("EKETA CRC check value (ICD table 6)", G.eketa_crc(b"123456789") == "cbf43926")
ex = bytes.fromhex("3E00319B0D010102000158784000 6B859C0043AD8A001C001C04D280154348014034BC01F4000186A00000220C0000359B".replace(" ", ""))
check("EKETA CRC of the ICD example datagram (10.2)", len(ex) == 49 and G.eketa_crc(ex) == "42ed60b3", G.eketa_crc(ex))
rec = G.eketa_record(dict(p=[37.80412, 23.78861], h=102.4, ms=152.4, hd=87.0, co=92.0, gs=10.1389,
                          vz=1.2, bat=74, fc=[True, True, "GUIDED"]), "INT-01")
check("EKETA telemetry record (ICD 10.1)",
      rec == dict(droneId="INT-01", latitude=37.80412, longitude=23.78861, relativeAltitude=102.4, altitude=152.4,
                  heading=87.0, course=92.0, speed=36.5, verticalSpeed=1.2, battery=74, inAir=True), rec)
rec = G.eketa_record(dict(p=[37.8, 23.7], h=0.2, fc=[True, False, "STABILIZE"]), "INT-01")
check("unknown values are left out, not sent as 0", rec == dict(droneId="INT-01", latitude=37.8, longitude=23.7,
                                                                relativeAltitude=0.2, inAir=False), rec)
check("no telemetry without a position", G.eketa_record(dict(p=None, h=3.0), "X") is None)

# ---- the radio, end to end -----------------------------------------------------

master, slave = pty.openpty()
tty.setraw(slave)
tty.setraw(master)
os.environ["RFD_DEV"] = os.ttyname(slave)

k, fc, mapper, m, sm = H["setup"]()
m.min_conf = 0.3
m.camera_mount_deg = 90.0
A.perf_counter = lambda: k.t            # the cue's timestamps on the simulated clock
link = G.RfdLink("pty", serial_factory=lambda: PtyEnd(master), hz=5.0)
HOME = H["HOME"]
DLAT = 1 / 111320.0


def at(north, east):
    return HOME[0] + north * DLAT, HOME[1] + east * DLAT / __import__("math").cos(__import__("math").radians(HOME[0]))


ctx = {}


def tick(n=1, dt=0.05, pace=0.01):
    for _ in range(n):
        k.t += dt
        fc.step(dt)
        mapper.frame_t = k.t
        m._fence_state = m.fence.check(fc.pos())
        sm.cycle(k.t, mapper, m, ctx)
        time.sleep(pace)


def until(cond, sim_s, **kw):
    t_end = k.t + sim_s
    while k.t < t_end:
        if cond():
            return True
        tick(**kw)
    return cond()


def in_bg(fn, *a):
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("r", fn(*a)), daemon=True)
    th.start()
    return th, box


tick(5)
check("Jetson opened its RFD", until(lambda: sm.cue.rfd_open, 10), sm.cue.rfd_error)

t1 = at(150, 60)
link.update_tracks([dict(tid=1, lat=t1[0], lon=t1[1], alt_m=40.0, frame="REL")])
check("the laptop's target reaches the Jetson as RFD_1",
      until(lambda: any(x["key"] == "RFD_1" and x["fresh"] for x in sm.cue.tracks(k.t)), 10))
tr = next(x for x in sm.cue.tracks(k.t) if x["key"] == "RFD_1")
check("a drawn target's altitude is height above home (REL)", tr["alt_rel"] == 40.0, tr["alt_rel"])
check("drone status reaches the laptop over the RFD", until(lambda: link.drone_state() is not None, 10))
check("the laptop sees the target in the drone's cue",
      until(lambda: any(t["key"] == "RFD_1" for t in link.drone_state()["mission"]["seeker"]["cue"]["tracks"]), 5))

runs = []
orig_cb = sm.command
sm.cue.cmd_callback = lambda v: (runs.append(v), orig_cb(v))[1]
th, box = in_bg(link.command, "cue:select:RFD_1")
until(lambda: "r" in box, 20)
check("TARGET LOCK over the RFD is acknowledged", box.get("r") == (True, ""), box.get("r"))
raw = A.rfd_cmd_line("#424242,view:SEEK")
link._write(raw)                      # through the link's lock, as a resend would go
link._write(raw)
until(lambda: "view:SEEK" in runs, 10)
tick(20)
check("a resent command runs once", runs.count("view:SEEK") == 1, runs)
th, box = in_bg(link.command, "start:BOGUS")
until(lambda: "r" in box, 20)
check("a refused command comes back with the drone's reason", box.get("r") == (False, "unknown mission"), box.get("r"))

until(lambda: link.drone_state()["mission"]["seeker"]["seek"]["ready"], 30)
st = link.drone_state()["mission"]["seeker"]["seek"]
check("SEEK pre-flight ready, seen on the laptop", st["ready"], st["why"])
check("SEEK settings reach the laptop", until(lambda: len(link.drone_state()["mission"]["seeker"]["seek"]["params"]) > 5, 30))

th, box = in_bg(link.command, "start:SEEK:FOLLOW")
until(lambda: "r" in box, 20)
check("START SEEK:FOLLOW over the RFD", box.get("r") == (True, ""), box.get("r"))
check("SEEK runs in FOLLOW-only", until(lambda: sm.run is not None and sm.run.get("follow_only"), 10))

# The camera 'locks' the whole time: FOLLOW-only must not hand over to WASP.
ctx.update(det_t=None, det_conf=0.99)
following = lambda: sm.run is not None and sm.run.get("seek") == "FOLLOW" and fc.alt > 25
check("climbs and follows the target", until(following, 120, dt=0.05), f"alt {fc.alt:.1f}")


def keep_locked():
    ctx["det_t"] = k.t


eng = False
t_end = k.t + 60
while k.t < t_end:
    keep_locked()
    tick()
    eng = eng or bool(m._req_engage) or (sm.run is not None and sm.run.get("state") == "AI")
pos = fc.pos()
d = S._dist(pos, t1)
check("FOLLOW-only: a steady camera lock does not AUTO ENGAGE", not eng and sm.run is not None
      and sm.run.get("seek") == "FOLLOW", f"state {sm.run and sm.run.get('state')}")
check("directly over the target", d < 15, f"{d:.1f} m off")
check("10 m below it (40 m target -> 30 m)", abs(fc.alt - 30.0) < 2.0 and sm.run["target_alt"] == 30.0,
      f"alt {fc.alt:.1f}, target_alt {sm.run['target_alt']}")
ds = link.drone_state()
run = ds["mission"]["seeker"]["run"]
check("the laptop sees the run: FOLLOW-only, target height, position",
      run and run["follow_only"] and run["target_alt"] == 30.0 and ds["mission"]["position"] is not None, run)

ctx["det_t"] = None
t2 = at(-120, -80)
link.update_tracks([dict(tid=1, lat=t1[0], lon=t1[1], alt_m=40.0, frame="REL"),
                    dict(tid=2, lat=t2[0], lon=t2[1], alt_m=50.0, frame="REL")])
check("a new target is offered on the laptop",
      until(lambda: (link.drone_state()["mission"]["seeker"]["run"] or {}).get("offer") is not None, 20))
of = link.drone_state()["mission"]["seeker"]["run"]["offer"]
th, box = in_bg(link.command, f"offer:yes:{of['n']}")
until(lambda: "r" in box, 20)
check("YES to the swap over the RFD", box.get("r") == (True, ""), box.get("r"))
check("TARGET SWAP: now following RFD_2", until(lambda: sm.cue.selected == "RFD_2", 10), sm.cue.selected)
until(lambda: S._dist(fc.pos(), t2) < 15 and abs(fc.alt - 40.0) < 2.0, 120)
check("over the new target, 10 m below it (50 -> 40 m)", S._dist(fc.pos(), t2) < 15 and abs(fc.alt - 40.0) < 2.0,
      f"{S._dist(fc.pos(), t2):.1f} m off, alt {fc.alt:.1f}")

th, box = in_bg(link.command, "after:RTL")
until(lambda: "r" in box, 20)
check("RTL over the RFD", box.get("r") == (True, ""), box.get("r"))
check("the drone flies home (RTL)", until(lambda: fc.mode_name == "RTL", 20), fc.mode_name)

# ---- the EKETA gateway, against a local WebSocket server -------------------------

try:
    from websockets.sync.server import serve
except ImportError:
    serve = None
if serve is None:
    check("websockets installed for the gateway test", False)
else:
    got = []

    def handler(ws):
        for msg in ws:
            got.append(json.loads(msg))
            t = got[-1].get("type")
            ws.send(json.dumps({"status": 200, "message": f"{t}_success"}))

    srv = serve(handler, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"ws://127.0.0.1:{srv.socket.getsockname()[1]}/gcs"
    gw = G.EketaGateway(url, "WASP-01")
    t_end = time.time() + 5
    while not gw.connected and time.time() < t_end:
        time.sleep(0.05)
    check("gateway connects", gw.connected, gw.last_error)
    gw.ack(ex)
    gw.telemetry(dict(p=[37.8, 23.7], h=31.5, gs=5.0, fc=[True, True, "GUIDED"]))
    t_end = time.time() + 5
    while len(got) < 2 and time.time() < t_end:
        time.sleep(0.05)
    check("ACK sent as {type: ack, crc}", got and got[0] == {"type": "ack", "crc": "42ed60b3"}, got[:1])
    check("telemetry sent as {type: telemetry, data: [...]}", len(got) > 1 and got[1]["type"] == "telemetry"
          and got[1]["data"][0]["droneId"] == "WASP-01" and got[1]["data"][0]["speed"] == 18.0, got[1:2])
    time.sleep(0.3)
    check("gateway replies counted", gw.replies.get("200 ack_success") == 1, gw.replies)
    srv.shutdown()

print("\nALL PASS" if not FAILED else f"\n{len(FAILED)} FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
