#!/usr/bin/env python3
"""SafeDown against real ArduCopter: the app flies SITL, launched from the stand.

Each scenario boots ArduCopter SITL (sitl_launch.py) with the user's FC
parameters and the JSON physics (sitl_physics.py): mass, CG offset, wind and
gusts, the tilted 4-leg stand.  The real app (AcroCommander + Mission over
MAVLink, through the fault-injecting sitl_proxy.py) arms in ACRO, launches
WASP with GO, then SafeDown is triggered by the phone's button or the pilot's
switch, optionally after an upset or with a fault, and is watched to the
ground.  Metrics come from the physics' true state, not from the FC.

    python sitl_safedown.py              # every scenario, 3 at a time
    python sitl_safedown.py -k wind -j 2 # only names containing "wind"
    python sitl_safedown.py --list

Results: a table on stdout and sitl_safedown_results.json in --out.
"""
import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from concurrent.futures import as_completed
from dataclasses import dataclass, field, asdict
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ACRO, STABILIZE, ALT_HOLD, LAND = 1, 0, 2, 9
SWITCH = dict(down=1000, middle=1500, up=1900)


@dataclass
class Scenario:
    name: str
    trigger: str = "button"          # "button" (phone) or "switch" (ch6 middle)
    after_s: float = 1.5             # launch -> trigger
    mass_kg: float = None            # None: the physics default
    cg_offset_m: tuple = None        # None: the physics default
    wind_ned: tuple = (0.0, 0.0, 0.0)
    gust_sigma: float = 0.0
    wind_on_stand: bool = False      # False: the wind starts once it is off the stand
    stand_pitch_deg: float = 45.0
    stand_roll_deg: float = 0.0
    upset: tuple = None              # body torque (N m, 3) for upset_s at the trigger
    upset_s: float = 0.15
    pre_upset: tuple = None          # (torque N m, seconds before the trigger, duration s): a dive
    deny: tuple = ()                 # modes the proxy refuses: "ALT_HOLD", "LAND", "STABILIZE"
    freeze_at: str = None            # "level" / "descend": the Jetson hangs then
    weak_motor: tuple = None         # (motor 1..4, thrust scale)
    plan_b_after_s: float = None     # pilot: throttle mid, switch UP this long after the trigger
    steer: bool = False              # WASP steering at a test target before the trigger
    expect: str = "landed"           # "landed", "pilot", "failed"
    extra_params: dict = field(default_factory=dict)


def _wind(speed, deg):
    return (speed*math.cos(math.radians(deg)), speed*math.sin(math.radians(deg)), 0.0)


def scenarios():
    out = []
    add = out.append
    for trigger in ("button", "switch"):
        add(Scenario(f"nominal-{trigger}", trigger=trigger))
    for after in (0.6, 1.0, 3.0, 5.0):
        add(Scenario(f"trigger-{after:g}s", after_s=after))
    for pitch in (15, 30, 60, 70):
        add(Scenario(f"stand-{pitch}deg", stand_pitch_deg=pitch))
    add(Scenario("stand-45deg-rolled-15", stand_roll_deg=15))
    for mass in (1.3, 1.8):
        add(Scenario(f"mass-{mass:g}kg", mass_kg=mass))
        add(Scenario(f"mass-{mass:g}kg-switch", mass_kg=mass, trigger="switch"))
    # More than ~1.4 cm forward tips the drone off a 45 degree stand before GO.
    for name, cg, stand in (("cg-fwd-1.2cm", (.012, 0, .01), 45), ("cg-fwd-2cm-stand30", (.02, 0, .01), 30),
                            ("cg-aft-2cm", (-.02, 0, .01), 45), ("cg-right-2.5cm", (0, .025, .01), 45),
                            ("cg-aft-left-low", (-.015, -.015, .03), 45)):
        add(Scenario(name, cg_offset_m=cg, stand_pitch_deg=stand))
    add(Scenario("heavy-cg-right-low", mass_kg=1.6, cg_offset_m=(.0, .02, .03)))
    for speed, deg, gust in ((5, 0, 0), (5, 180, 0), (8, 90, 2.0), (10, 45, 3.0)):
        add(Scenario(f"wind-{speed}ms-{deg}deg" + (f"-gust{gust:g}" if gust else ""),
                     wind_ned=_wind(speed, deg), gust_sigma=gust))
    add(Scenario("wind-8ms-gust-heavy-cg", wind_ned=_wind(8, 270), gust_sigma=2.5,
                 mass_kg=1.5, cg_offset_m=(.015, -.02, .02), trigger="switch"))
    # Launch findings: a tail wind on the 45 degree stand, before GO.
    add(Scenario("stand-tailwind-3ms", wind_ned=_wind(3, 0), wind_on_stand=True))
    add(Scenario("stand-tailwind-5ms", wind_ned=_wind(5, 0), wind_on_stand=True))
    # A hit or a prop strike at the trigger: about 1500 deg/s peaks, inside
    # a real gyro's 2000 deg/s, and the FC fighting it.
    for name, torque, secs in (("upset-roll", (2.2, 0, 0), .15), ("upset-pitch-flip", (0, -2.2, 0), .2),
                               ("upset-tumble", (1.6, 1.6, .4), .2)):
        add(Scenario(name, upset=torque, upset_s=secs))
    add(Scenario("upset-roll-low", upset=(2.2, 0, 0), upset_s=.15, after_s=.7))
    add(Scenario("upset-tumble-wind-heavy", upset=(1.6, 1.6, .4), upset_s=.2,
                 wind_ned=_wind(5, 30), gust_sigma=1.5, mass_kg=1.2, trigger="switch"))
    # Far past any real gyro (5000+ deg/s): the EKF's vertical velocity breaks.
    add(Scenario("extreme-tumble-ekf-breaks", after_s=3.0, pre_upset=((0, -6.0, 0), 1.2, .3)))
    # WASP holds any attitude at full throttle in ACRO: a nose-over before the
    # trigger is a full-power dive.
    add(Scenario("nose-over-low", after_s=1.2, pre_upset=((0, -4.0, 0), .45, .2)))
    add(Scenario("dive-steep", after_s=3.0, pre_upset=((0, -2.2, 0), 1.2, .2)))
    add(Scenario("dive-inverted", after_s=3.0, pre_upset=((0, -2.2, 0), 1.0, .3)))
    add(Scenario("dive-heavy-wind-switch", after_s=3.0, pre_upset=((0, -2.2, 0), 1.2, .2), mass_kg=1.4,
                 wind_ned=_wind(6, 120), gust_sigma=1.5, trigger="switch"))
    add(Scenario("steering-then-safedown", steer=True, after_s=2.5))
    add(Scenario("weak-motor-15pct", weak_motor=(1, .85)))
    add(Scenario("fault-alt-hold-refused", deny=("ALT_HOLD",)))
    add(Scenario("fault-land-refused", deny=("LAND",)))
    add(Scenario("fault-alt-hold-and-land-refused", deny=("ALT_HOLD", "LAND")))
    add(Scenario("fault-alt-hold-and-land-refused-heavy-wind", deny=("ALT_HOLD", "LAND"),
                 mass_kg=1.5, wind_ned=_wind(6, 0), gust_sigma=1.5))
    add(Scenario("fault-everything-refused", deny=("ALT_HOLD", "LAND", "STABILIZE"), expect="failed"))
    add(Scenario("fault-jetson-hangs-in-descent", freeze_at="descend"))
    add(Scenario("fault-jetson-hangs-while-levelling", freeze_at="level", expect="any"))
    add(Scenario("plan-b-pilot-takes-over", plan_b_after_s=3.0, expect="pilot"))
    # The same drone with its GPS used by the EKF (EK3_SRC1_POSXY/VELXY = 3):
    # attitude stays honest and LAND holds position against the wind.
    # The FC has COMPASS_ENABLE 0: without a heading the EKF cannot use GPS.
    gps = dict(EK3_SRC1_POSXY=3, EK3_SRC1_VELXY=3, COMPASS_ENABLE=1)
    add(Scenario("gps-nominal", extra_params=gps))
    add(Scenario("gps-wind-8ms-90deg-gust2", wind_ned=_wind(8, 90), gust_sigma=2.0, extra_params=gps))
    add(Scenario("gps-wind-10ms-45deg-gust3", wind_ned=_wind(10, 45), gust_sigma=3.0, extra_params=gps))
    add(Scenario("gps-extreme-tumble-ekf-breaks", after_s=3.0, pre_upset=((0, -6.0, 0), 1.2, .3), extra_params=gps))
    return out


# -- one run --------------------------------------------------------------------

class Run:
    def __init__(self, sc, instance, out_dir):
        self.sc, self.instance, self.out_dir = sc, instance, Path(out_dir)
        self.events = []
        self.samples = []

    def note(self, text):
        self.events.append((round(time.perf_counter()-self.t0, 2), text))

    def wait(self, predicate, timeout, what):
        end = time.perf_counter()+timeout
        while time.perf_counter() < end:
            if predicate():
                return True
            self.sample()
            time.sleep(.02)
        raise TimeoutError(what)

    def sample(self):
        st = self.sitl.physics.state()
        m = self.m
        b = self.c.backend
        att, hgt, acc = b.wasp_attitude(), b.height(max_age=.5), b.body_accel()
        self.samples.append(dict(
            st, physics_t=st["t"], t=time.perf_counter()-self.t0, phase=m.phase,
            stage=m.safedown.stage if m.safedown.active else "",
            fallback=m.safedown.fallback if m.safedown.active else None,
            mode=self.c.mode_name, armed=self.c.armed,
            fc_tilt=None if att is None else math.degrees(math.acos(max(-1, min(1, math.cos(att["roll"])*math.cos(att["pitch"]))))),
            fc_h=None if hgt is None else hgt[0], fc_climb=None if hgt is None else hgt[1],
            fc_accel_h=None if acc is None else math.hypot(acc[0], acc[1])))
        return st

    def run(self):
        from sitl_launch import Sitl
        from sitl_physics import PhysicsConfig
        from sitl_proxy import MavProxy
        from fc_control import AcroCommander, InterceptLaw
        from mission import phone_command
        sc = self.sc
        opt = {k: v for k, v in (("mass_kg", sc.mass_kg), ("cg_offset_m", sc.cg_offset_m)) if v is not None}
        calm = not sc.wind_on_stand
        cfg = PhysicsConfig(instance=self.instance, stand_pitch_deg=sc.stand_pitch_deg,
                            stand_roll_deg=sc.stand_roll_deg,
                            wind_ned=(0.0, 0.0, 0.0) if calm else sc.wind_ned,
                            gust_sigma=0.0 if calm else sc.gust_sigma,
                            seed=sum(map(ord, sc.name)), **opt)
        self.sitl = Sitl(self.instance, cfg, extra_params=sc.extra_params or None)
        logs = tempfile.TemporaryDirectory(prefix="sitl-safedown-")
        try:
            self.sitl.start()
            self.sitl.wait_ready()
            self.proxy = MavProxy(int(self.sitl.mavlink_url.rsplit(":", 1)[1]))
            self.proxy.deny_modes = {dict(ALT_HOLD=ALT_HOLD, LAND=LAND, STABILIZE=STABILIZE)[d]
                                     for d in sc.deny}
            self.c = AcroCommander(conn=self.proxy.url, transport="mavlink",
                                   axes=("roll", "pitch", "yaw", "throttle"),
                                   law=InterceptLaw(cam_tilt_deg=30), send_hz=50, mission=True,
                                   verbose=False, steer_modes={"ACRO", "ALT_HOLD"}, log_dir=logs.name)
            with contextlib.redirect_stdout(io.StringIO()), patch("mission._read_json", return_value={}):
                self.c.connect()
            self.m = m = self.c.mission
            for attr in ("idle_path", "wasp_path", "auto_path", "levels_path", "flight_path"):
                setattr(m, attr, str(Path(logs.name)/f"{attr}.json"))
            m.log = lambda text: self.note(text)
            self.t0 = time.perf_counter()
            cmd = lambda k, v: phone_command(m, k, v, time.perf_counter())
            for k, v in (("flightmode", "wasp"), ("mount", "90"), ("waspp", "launch_s:0"),
                         ("waspp", "throttle:1"), ("waspp", "ramp:0.1"), ("waspp", "straight:0.5")):
                why = cmd(k, v)
                if why:
                    raise RuntimeError(f"{k}={v}: {why}")
            if m.confirm_mount(90):
                raise RuntimeError("camera mount")
            self.sitl.arm()
            self.wait(lambda: self.c.armed and self.c.mode_name == "ACRO" and m.phase == "WASP_READY",
                      20, f"WASP_READY (gate: {self.c._mission_why})")
            # ArduCopter holds the motors for 2 s after arming; GO comes later.
            armed = time.perf_counter()
            self.wait(lambda: time.perf_counter()-armed > 3, 4, "arming delay")
            if sc.steer:
                self._start_detections()
            if sc.weak_motor:
                self.sitl.physics.event("motor_scale", motor=sc.weak_motor[0], scale=sc.weak_motor[1])
            why = cmd("engage", "1")
            if why:
                raise RuntimeError(f"GO refused: {why}")
            self.wait(lambda: m.phase == "WASP_FLIGHT", 3, "launch")
            launched = time.perf_counter()
            if not sc.wind_on_stand and (any(sc.wind_ned) or sc.gust_sigma):
                with contextlib.suppress(TimeoutError):
                    self.wait(lambda: self.sample().get("left_stand"), 2, "liftoff")
                self.sitl.physics.event("wind", ned=sc.wind_ned)
                if sc.gust_sigma:
                    self.sitl.physics.event("gusts", sigma=sc.gust_sigma)
                self.note("WIND ON")
            if sc.pre_upset:
                torque, lead, duration = sc.pre_upset
                self.wait(lambda: time.perf_counter()-launched >= sc.after_s-lead, sc.after_s+1, "flight")
                self.sitl.physics.event("torque", nm=torque, duration=duration)
                self.note("PRE-UPSET")
            self.wait(lambda: time.perf_counter()-launched >= sc.after_s, sc.after_s+1, "flight")
            st = self.sample()
            at_trigger = dict(st)
            self.lifted = any(x.get("left_stand") for x in self.samples)
            if sc.upset:
                self.sitl.physics.event("torque", nm=sc.upset, duration=sc.upset_s)
            if sc.trigger == "switch":
                self.sitl.rc.set(6, SWITCH["middle"])
            else:
                if cmd("safedown", "1"):
                    raise RuntimeError("SAFEDOWN refused")
            self.trigger_t = time.perf_counter()
            self._watch(at_trigger)
        finally:
            self._stop_detections()
            for closer in (getattr(self, "c", None), getattr(self, "proxy", None)):
                with contextlib.suppress(Exception):
                    closer and closer.close()
            with contextlib.suppress(Exception):
                self.sitl.stop()
            logs.cleanup()
        return self.metrics(at_trigger)

    def _watch(self, at_trigger):
        sc, m = self.sc, self.m
        frozen = plan_b = False
        end = self.trigger_t + 120
        while time.perf_counter() < end:
            st = self.sample()
            since = time.perf_counter() - self.trigger_t
            if sc.freeze_at and not frozen and (
                    (sc.freeze_at == "level" and m.safedown.stage in ("LEVEL", "STOP"))
                    or (sc.freeze_at == "descend" and m.safedown.stage == "DESCEND" and self.c.mode_name == "LAND")):
                self.proxy.frozen = frozen = True
                self.note(f"JETSON FROZEN in {m.safedown.stage}")
            if sc.plan_b_after_s is not None and not plan_b and since >= sc.plan_b_after_s:
                # The pilot's Plan B: throttle stick to mid first, then switch up.
                self.sitl.rc.set_stick("throttle", .5)
                time.sleep(.2)
                self.sitl.rc.set(6, SWITCH["up"])
                plan_b, self.plan_b_t = True, time.perf_counter()
                self.note("PILOT: throttle mid, switch UP")
            if plan_b and time.perf_counter() - self.plan_b_t > 8:
                break
            if not self.c.armed and (st.get("on_ground") or st.get("crashed")):
                for _ in range(25):
                    self.sample()
                    time.sleep(.02)
                break
            if st.get("crashed"):
                break
            time.sleep(.02)

    # Fake detections for the "steering" scenario: a target off to one side.
    def _start_detections(self):
        import threading
        self._det_stop = threading.Event()

        def loop():
            while not self._det_stop.is_set():
                x, y = 120, -40
                cmd = self.c.compute(x, y, "LOCK", 640, 640, bbox=(0, 0, 24, 24),
                                     capture_t=time.perf_counter()-.03)
                self.c.update(cmd, (x, y), True, detected=True, conf=.9, bbox=(0, 0, 24, 24))
                time.sleep(.03)
        threading.Thread(target=loop, daemon=True).start()
        self.note("detections on")

    def _stop_detections(self):
        if getattr(self, "_det_stop", None):
            self._det_stop.set()

    def metrics(self, at_trigger):
        s = [x for x in self.samples if x["t"] >= self.trigger_t - self.t0]
        tilt = lambda x: x["tilt"]
        h = lambda x: -x["pos"][2]
        vh = lambda x: math.hypot(x["vel"][0], x["vel"][1])
        level_t = next((x["t"] for x in s if tilt(x) < 10 and x["t"] > s[0]["t"]+.05), None)
        descend = next((x for x in s if x["stage"] == "DESCEND"), None)
        touch = next((x for x in s if x.get("on_ground")), None)
        after_level = [x for x in s if level_t is not None and x["t"] >= level_t + 1.0 and not x.get("on_ground")]
        yaw_hold = None
        if descend:
            yaws = [x["euler"][2] for x in s if x["t"] >= descend["t"] and not x.get("on_ground")]
            if yaws:
                yaw_hold = max(abs((y - yaws[0] + 180) % 360 - 180) for y in yaws)
        events = [e for _t, e in self.events]
        landed = any("SAFEDOWN complete: landed" in e for e in events)
        final = s[-1] if s else {}
        crashed = any(x.get("crashed") for x in s)
        tipped = bool(final) and final.get("on_ground") and tilt(final) > 45
        ok = {"landed": landed and not crashed and not tipped and self.lifted,
              "pilot": not crashed and any("SAFEDOWN ended: pilot switched" in e for e in events),
              "failed": any("SAFEDOWN FAILED" in e for e in events),
              "any": not crashed and not tipped}[self.sc.expect]
        if not self.lifted:
            ok = None  # never left the stand: a launch finding, SafeDown never flew
        return dict(
            name=self.sc.name, ok=ok, expect=self.sc.expect, lifted=self.lifted,
            trigger_h=round(h(at_trigger), 1), trigger_speed=round(vh(at_trigger), 1),
            trigger_vz=round(-at_trigger["vel"][2], 1),
            trigger_tilt=round(tilt(at_trigger)), max_tilt=round(max(tilt(x) for x in s)),
            max_rate_dps=round(max(max(abs(r) for r in x["rates"]) for x in self.samples)),
            level_s=None if level_t is None else round(level_t - s[0]["t"], 2),
            max_tilt_after_level=round(max((tilt(x) for x in after_level), default=0)),
            min_h=round(min(h(x) for x in s if not x.get("on_ground")) if any(not x.get("on_ground") for x in s) else 0, 1),
            max_h=round(max(h(x) for x in s), 1),
            descend_s=None if descend is None else round(descend["t"] - s[0]["t"], 1),
            descend_h=None if descend is None else round(h(descend), 1),
            descend_speed=None if descend is None else round(vh(descend), 1),
            yaw_drift_deg=None if yaw_hold is None else round(yaw_hold),
            touchdown_s=None if touch is None else round(touch["t"] - s[0]["t"], 1),
            touchdown_vz=round(final.get("max_td_vz") or 0, 2) if touch else None,
            touchdown_vh=round(final.get("max_td_vh") or 0, 2) if touch else None,
            hard_landing=bool(final.get("hard_landings")) if final else None,
            drift_m=round(math.hypot(final["pos"][0]-at_trigger["pos"][0], final["pos"][1]-at_trigger["pos"][1]), 1) if final else None,
            crashed=crashed, crash_reason=next((x.get("crash_reason") for x in s if x.get("crashed")), None),
            tipped=tipped, disarmed=bool(final) and not final["armed"],
            fallbacks=sorted({x["fallback"] for x in s if x["fallback"]}),
            modes=_collapse([x["mode"] for x in s]),
            events=[e for e in events if "SAFEDOWN" in e or "PILOT" in e or "JETSON" in e][:14],
            series=[{k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()
                     if k in ("t", "height", "tilt", "fc_tilt", "fc_h", "fc_climb", "fc_accel_h", "stage", "mode", "airspeed")}
                    | dict(vz=round(-x["vel"][2], 2), vh=round(vh(x), 2))
                    for x in s[::10]])


def _collapse(seq):
    out = []
    for v in seq:
        if not out or out[-1] != v:
            out.append(v)
    return out


def _one(name, instance, out_dir):
    sc = next(s for s in scenarios() if s.name == name)
    try:
        result = Run(sc, instance, out_dir).run()
    except Exception as exc:
        result = dict(name=sc.name, ok=False, error=f"{type(exc).__name__}: {exc}",
                      trace=traceback.format_exc()[-1500:])
    return result


def _spawn(name, instance, out_dir):
    import subprocess
    path = Path(out_dir)/f"{name}.json"
    cmd = [sys.executable, __file__, "--one", name, "--instance", str(instance), "--out", str(out_dir)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=400)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return dict(name=name, ok=False, error=f"exit {proc.returncode}", trace=proc.stdout[-1500:])


COLS = ("name", "ok", "trigger_h", "trigger_vz", "trigger_speed", "max_tilt", "max_rate_dps", "level_s", "min_h",
        "descend_s", "descend_speed", "touchdown_vz", "touchdown_vh", "drift_m", "fallbacks")


def table(results):
    rows = [[str(r.get(c, "")) if c in r else r.get("error", "")[:60] if c == "fallbacks" else ""
             for c in COLS] for r in results]
    widths = [max(len(c), *(len(row[i]) for row in rows)) for i, c in enumerate(COLS)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(COLS, widths))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="", help="only scenarios whose name contains one of these (comma list)")
    ap.add_argument("-j", type=int, default=3, help="scenarios in parallel")
    ap.add_argument("--out", default=os.environ.get("WASP_SITL_OUT", tempfile.gettempdir()))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--instance", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out)/f"{a.one}.json").write_text(json.dumps(_one(a.one, a.instance, a.out), default=str))
        return 0
    keys = a.k.split(",")
    chosen = [s for s in scenarios() if any(k in s.name for k in keys)]
    if a.list:
        print("\n".join(s.name for s in chosen))
        return 0
    from sitl_launch import find_binary
    if find_binary() is None:
        print("SKIP: no ArduCopter SITL binary (set WASP_SITL_BIN)")
        return 0
    results = []
    import queue
    from concurrent.futures import ThreadPoolExecutor
    free = queue.Queue()
    for i in range(a.j):
        free.put(i)

    def job(name):
        instance = free.get()
        try:
            return _spawn(name, instance, a.out)
        finally:
            free.put(instance)
    with ThreadPoolExecutor(max_workers=a.j) as pool:
        futures = [pool.submit(job, s.name) for s in chosen]
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            tag = "NO-LAUNCH " if r.get("ok") is None else "PASS " if r["ok"] else "FAIL "
            print(tag + r["name"] + ("" if r.get("ok") is not False else f"  {r.get('error', '')}"), flush=True)
    results.sort(key=lambda r: [s.name for s in chosen].index(r["name"]))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out)/"sitl_safedown_results.json").write_text(json.dumps(results, indent=1, default=str))
    print(table(results))
    flew = [r for r in results if r.get("ok") is not None]
    print(f"SafeDown: {sum(bool(r['ok']) for r in flew)}/{len(flew)} passed; "
          f"{len(results)-len(flew)} never left the stand (launch findings)")
    return 0 if all(r["ok"] for r in flew) else 1


if __name__ == "__main__":
    sys.exit(main())
