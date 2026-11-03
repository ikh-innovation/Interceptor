#!/usr/bin/env python3
"""Seeker test missions against real ArduCopter SITL, the way the phone runs them.

Each scenario boots SITL (sitl_launch.py) with the WASP FC's parameter file,
GPS in the EKF, a flat take-off spot, and the pilot's transmitter (mode switch
on channel 6).  The real app (AcroCommander + Mission + SeekerMissions over
MAVLink) gets a geofence around the take-off point and starts the mission with
the phone's own command (msn=start:Mx); fake camera frames stand in for the
video.  The run is watched to its result.

    python sitl_seeker_missions.py                 # every scenario, 3 at a time
    python sitl_seeker_missions.py -k M3 -j 1      # names containing "M3"
    python sitl_seeker_missions.py -k M3 --serve 8091   # also serve the phone UI

Results: a table on stdout, sitl_seeker_results.json and the mission CSVs in --out.
"""
import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import queue
import shutil
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sitl_launch import Sitl  # noqa: E402
from sitl_physics import PhysicsConfig  # noqa: E402
from seeker_missions import _dist as _d  # noqa: E402

GPS_PARAMS = dict(EK3_SRC1_POSXY=3, EK3_SRC1_VELXY=3, COMPASS_ENABLE=1,
                  # The FC's battery failsafe (the FC file's values; SITL drops BATT_*).
                  BATT_MONITOR=4, BATT_CAPACITY=3300, BATT_LOW_VOLT=10.5, BATT_FS_LOW_ACT=1, BATT_LOW_TIMER=10)
SWITCH = dict(down=1000, middle=1500, up=1900)


@dataclass
class Scenario:
    name: str
    mission: str
    fence_m: float = 1400.0             # square around the take-off point
    params: dict = field(default_factory=dict)       # mission settings (msn=set)
    limits: dict = field(default_factory=dict)
    abort_at_m: float = None            # the operator presses ABORT at this height
    abort_at_home_m: float = None       # ... or this far from home
    takeover_at_m: float = None         # the pilot: throttle mid, switch UP -> MIDDLE (STABILIZE) here
    answers: tuple = ()                 # SEEK: the operator's answers in order (then YES)
    second_target_s: float = None       # SEEK: a second track appears this long after the start
    pilot_after_ai_s: float = None      # SEEK: the pilot takes over (MIDDLE) this long after the AI engages
    answer_after_s: float = 0.0         # SEEK: the operator waits this long before answering
    expect: str = "PASS"
    timeout_s: float = 900.0
    extra_params: dict = field(default_factory=dict)
    wind_ned: tuple = (0.0, 0.0, 0.0)
    gust_sigma: float = 0.0
    cg_offset_m: tuple = (0.0, 0.0, 0.01)


def scenarios():
    return [
        Scenario("M1-startup-shutdown", "M1"),
        Scenario("M2-link-check", "M2"),
        Scenario("M3-altitude-ladder", "M3"),
        # 6 m/s side wind with gusts and the CG 3 cm off to one side: the lines must stay straight.
        Scenario("M3-ladder-wind-heavy-side", "M3", params=dict(alt1=20, alt2=40, hold1=5, hold2=5),
                 wind_ned=(0.0, -6.0, 0.0), gust_sigma=1.5, cg_offset_m=(0.0, 0.03, 0.01)),
        Scenario("M4-route-level", "M4"),
        Scenario("M4-route-45deg-wind", "M4", params=dict(angle=45, heading=90, slant=120),
                 wind_ned=(4.0, 0.0, 0.0), gust_sigma=1.0),
        Scenario("M5-abort-hover", "M5"),
        Scenario("M6-geofence", "M6", fence_m=300, params=dict(alt=20, speed=10)),
        Scenario("M6-geofence-fast", "M6", fence_m=500, params=dict(alt=20, speed=15, heading=45)),
        Scenario("M6-geofence-climbing", "M6", fence_m=400, params=dict(alt=20, speed=10, heading=270, angle=20)),
        Scenario("M7-app-restart", "M7", params=dict(alt=20, angle=45, dist=40)),
        Scenario("M7-app-restart-up-only", "M7", params=dict(alt=15, angle=90, dist=15)),
        Scenario("ABORT-button-M3-rtl", "M3", abort_at_m=25, expect="FAIL"),
        Scenario("ABORT-button-M4-rtl", "M4", abort_at_m=None, abort_at_home_m=60, expect="FAIL"),
        Scenario("RC-takeover-M3", "M3", takeover_at_m=25, expect="INCOMPLETE"),
        Scenario("SEEK-asterix-to-AI", "SEEK", expect="HANDOVER", timeout_s=300),
        Scenario("SEEK-asterix-fast-target", "SEEK", expect="HANDOVER", timeout_s=300,
                 params=dict(standoff=30)),
        Scenario("SEEK-field-switch-target-ai-pilot", "SEEK", expect="PILOT", timeout_s=420,
                 answers=("yes",), second_target_s=25, pilot_after_ai_s=3),
        Scenario("blocked-low-battery", "M3", expect="FAIL", extra_params=dict(SIM_BATT_VOLTAGE=10.6)),
        Scenario("blocked-M6-no-room-to-stop", "M6", fence_m=60, params=dict(speed=25), expect="FAIL"),
    ]


class Run:
    def __init__(self, instance, sc, out, serve=None):
        self.instance, self.sc, self.out, self.serve = instance, sc, Path(out), serve
        self.notes = []
        self.t0 = time.perf_counter()

    def note(self, text):
        self.notes.append((round(time.perf_counter() - self.t0, 2), text))

    def wait(self, pred, timeout, what):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if pred():
                return
            time.sleep(0.05)
        raise TimeoutError(what)

    def run(self):
        sc = self.sc
        from fc_control import AcroCommander, InterceptLaw
        from mission import phone_command
        cfg = PhysicsConfig(instance=self.instance, stand_pitch_deg=0.0, seed=sum(map(ord, sc.name)),
                            wind_ned=sc.wind_ned, gust_sigma=sc.gust_sigma, cg_offset_m=sc.cg_offset_m)
        self.sitl = sitl = Sitl(self.instance, cfg, extra_params=dict(GPS_PARAMS, **sc.extra_params))
        work = Path(tempfile.mkdtemp(prefix="sitl-seeker-"))
        stop = threading.Event()
        res = dict(scenario=sc.name, mission=sc.mission, expect=sc.expect)
        srv = None
        try:
            sitl.start()
            sitl.wait_ready()
            # The field procedure: mode switch UP (ALT_HOLD), throttle stick DOWN for ARM;
            # the pilot raises it to MIDDLE once it is in the air (_drive).
            sitl.rc.set(6, SWITCH["up"])
            sitl.rc.set_stick("throttle", 0)
            self.c = c = AcroCommander(conn=sitl.mavlink_url, transport="mavlink",
                                       axes=("roll", "pitch", "yaw", "throttle"),
                                       law=InterceptLaw(cam_tilt_deg=30), send_hz=20, mission=True,
                                       verbose=False, steer_modes={"ACRO", "ALT_HOLD"}, log_dir=str(work))
            with contextlib.redirect_stdout(io.StringIO()), patch("mission._read_json", return_value={}):
                c.connect()
            m = c.mission
            for attr in ("idle_path", "wasp_path", "auto_path", "levels_path", "flight_path", "fence_path"):
                setattr(m, attr, str(work/f"{attr}.json"))
            m.seeker.path = str(work/"seeker.json")
            m.seeker.log_dir = str(self.out)
            # Not the Jetson's own saved choices (the live app may have WASP selected).
            import seeker_missions as SM
            m.seeker.lim = {x["k"]: x["d"] for x in SM.LIMITS}
            m.seeker.params = {mm["id"]: {x["k"]: x["d"] for x in mm["params"]} for mm in SM.BY_ID.values()}
            m.seeker.restore, m.seeker.resume = {}, None
            m.wasp.enabled = False

            def restart():
                # M7's RESTART APP: this "process" is silent for 20 s (the systemd
                # restart), then a fresh SeekerMissions starts from the saved file.
                old = m.seeker
                self.note("APP RESTART: silent for 20 s")

                def back():
                    time.sleep(20)
                    old.cue.stop()
                    new = SM.SeekerMissions(path=old.path, log_dir=old.log_dir, log=old.log)
                    new.restart_hook = restart
                    c.backend.gcs_muted = False
                    m.seeker = new
                    self.note("APP BACK")
                threading.Thread(target=back, daemon=True).start()
            m.seeker.restart_hook = restart
            m.camera_mount_deg = 30.0
            m.log = self.note
            cmd = lambda k, v: phone_command(m, k, v, time.perf_counter())

            def frames():
                # Camera frames for the video checks: the tracker calls update() per frame.
                while not stop.wait(1/30):
                    # One report per frame, as the tracker sends: the camera
                    # thread reports the frames that hold the target.
                    if not getattr(self, "in_view", False):
                        c.update(c.neutral_command(), None, have_target=False)
            threading.Thread(target=frames, daemon=True).start()
            if self.serve:
                srv = self._serve(c, m)
            self.wait(lambda: c.backend.position_ok() and c.backend.gps_fix_ok(), 180, "EKF GPS position")
            lat, lon = c.backend.position()
            half = sc.fence_m/2
            dlat, dlon = half/111320, half/(111320*math.cos(math.radians(lat)))
            why = cmd("fence", "set:" + json.dumps([[lat-dlat, lon-dlon], [lat+dlat, lon-dlon],
                                                    [lat+dlat, lon+dlon], [lat-dlat, lon+dlon]]))
            if why:
                raise RuntimeError(f"geofence: {why}")
            for k, v in sc.limits.items():
                why = cmd("msn", f"set:limits:{k}:{v}")
                if why:
                    raise RuntimeError(f"limit {k}: {why}")
            for k, v in sc.params.items():
                why = cmd("msn", f"set:{sc.mission}:{k}:{v}")
                if why:
                    raise RuntimeError(f"{k}={v}: {why}")
            if sc.mission == "SEEK":
                self._asterix(c, m, cmd, stop)
            cmd("msn", f"view:{sc.mission}")
            # The phone's START is enabled once the checks pass (or the run shows why not).
            try:
                self.wait(lambda: m.seeker.snapshot(time.perf_counter())["ready"], 40, "checks")
            except TimeoutError:
                self.note("checks not ready: " + m.seeker.snapshot(time.perf_counter())["ready_why"])
            why = cmd("msn", f"start:{sc.mission}")
            if why:
                raise RuntimeError(f"start refused: {why}")
            self.wait(lambda: m.seeker.active, 5, "mission start")
            self._drive(m, cmd)
            if sc.mission == "SEEK" and m.seeker.result is None and m.seeker.handover is not None or \
                    sc.mission == "SEEK" and (m.seeker.run or {}).get("state") == "AI":
                # Handed over: the AI should be flying it in ALT_HOLD now.
                self.wait(lambda: m.phase in ("PUSH", "AI", "AUTO_CLIMB", "AUTO_HOLD", "AUTO_ALIGN"), 4, "the AI engaging")
                # Watch the AI fly it for a few seconds (the phases it goes through).
                seen, t_ai = [], time.perf_counter()
                while time.perf_counter() - t_ai < 6.0:
                    if not seen or seen[-1] != m.phase:
                        seen.append(m.phase)
                    time.sleep(0.05)
                res["ai_phases"] = seen
                self.note(f"the AI: {' -> '.join(seen)}")
                res.update(verdict="HANDOVER" if c.mode_name == "ALT_HOLD" else "NO-AI",
                           why=f"FC {c.mode_name}, mission phase {m.phase}",
                           closest_m=round(self.closest, 1), metrics={})
                self.note(f"after the handover: FC {c.mode_name}, phase {m.phase}")
                res["asked"] = getattr(self, "asked", [])
                ca = getattr(self, "cam_at_answer", None)
                res["cam_at_answer"] = None if ca is None else [round(x, 1) for x in ca]
                if sc.mission == "SEEK" and any(a[0] == "engage" for a in res["asked"]):
                    res["verdict"] = "ASKED-ON-DETECTION"
                if sc.pilot_after_ai_s is not None:
                    time.sleep(sc.pilot_after_ai_s)
                    self.note("PILOT: switch UP -> MIDDLE (STABILIZE)")
                    sitl.rc.set(6, SWITCH["middle"])
                    self.wait(lambda: c.mode_name == "STABILIZE" and m.phase == "PILOT" and not c.steering, 5,
                              "the pilot taking over from the AI")
                    res["verdict"] = "PILOT"
                    res["why"] += f"; then FC {c.mode_name}, phase {m.phase}, app steering {c.steering}"
                    sitl.rc.set_stick("throttle", .5)
                cmd("abort", "1")
                time.sleep(0.5)
                sitl.set_mode("LAND")
                self.wait(lambda: not c.armed, 240, "landing after the handover")
                return self._done(res)
            r = m.seeker.result
            res.update(verdict=r["verdict"], why=r["why"], duration_s=r["duration_s"],
                       checks=[(x["name"], x["ok"], x["detail"]) for x in r["checks"]],
                       metrics=r["metrics"], csv=r["csv"], fc_armed_after=c.armed,
                       fs_gcs_after=c.backend.params.get("FS_GCS_ENABLE"))
        except Exception as exc:
            res.update(verdict="ERROR", why=f"{exc!r}", trace=traceback.format_exc(limit=4))
        finally:
            stop.set()
            if srv is not None:
                with contextlib.suppress(Exception):
                    srv.stop()
            with contextlib.suppress(Exception):
                self.c.close()
            with contextlib.suppress(Exception):
                sitl.stop()
            shutil.rmtree(work, ignore_errors=True)
        return self._done(res)

    def _done(self, res):
        res["notes"] = self.notes[-60:]
        res["ok"] = res.get("verdict") == self.sc.expect and not res.get("fc_armed_after") \
            and res.get("fs_gcs_after") in (None, 0.0)
        return res

    def _asterix(self, c, m, cmd, stop):
        """A target track over real UDP CAT062 to the app's cue port, 450 m
        north-east and 60 m up, moving; camera detections once within 120 m."""
        import socket as _s
        from asterix_cue import encode_cat062
        port = 18600 + self.instance
        why = cmd("msn", f"cue:udp:{port}::")
        if why:
            raise RuntimeError(f"ASTERIX port: {why}")
        home = c.backend.position()
        amsl = c.backend.telemetry()["amsl"]
        fast = "fast" in self.sc.name
        t0 = time.perf_counter()
        self.closest = 1e9

        def pos():
            dt = time.perf_counter() - t0
            # The fast one crosses the area east to west at 7 m/s.
            n, e = (350.0, 340.0 - 7.0*dt) if fast else (300.0 + 1.5*dt, 340.0)
            return (home[0] + math.degrees(n/6371000.0),
                    home[1] + math.degrees(e/(6371000.0*math.cos(math.radians(home[0])))))

        def pos2():
            return (home[0] + math.degrees(-250.0/6371000.0),
                    home[1] + math.degrees(150.0/(6371000.0*math.cos(math.radians(home[0])))))

        def run():
            sock = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
            while not stop.wait(0.5):
                lat, lon = pos()
                sock.sendto(encode_cat062([dict(icao=0x3C0A55, track_no=7, lat=lat, lon=lon,
                                                alt_ft=(amsl + 60.0)/0.3048, vx=0, vy=1.5),
                                           dict(icao=0xF00099, track_no=99, lat=lat + 0.01, lon=lon + 0.01,
                                                alt_ft=(amsl + 900)/0.3048)]), ("127.0.0.1", port))
                if self.sc.second_target_s is not None and time.perf_counter() - t0 >= self.sc.second_target_s:
                    lat2, lon2 = pos2()
                    sock.sendto(encode_cat062([dict(icao=0x4B1A2B, track_no=55, lat=lat2, lon=lon2,
                                                    alt_ft=(amsl + 50.0)/0.3048)]), ("127.0.0.1", port))
                cue = (m.seeker.run or {}).get("cue")
                if cue:
                    lat, lon = cue["lat"], cue["lon"]
                p = c.backend.position()
                if p is not None:
                    from seeker_missions import _dist
                    d = math.hypot(_dist(p, (lat, lon)), (c.backend.telemetry()["rel_alt"] or 0) - 60.0)
                    self.closest = min(self.closest, d)
                    self.in_view = d < 120.0
                    self.view_target = (lat, lon, 60.0)

        def camera():
            # Detections at the camera's rate while the target is in range, with
            # the angles the AI's law reports: camera 30 degrees up, on the nose.
            from seeker_missions import _bearing, _dist as _d
            while not stop.wait(1/30):
                if not self.in_view:
                    continue
                cmd = c.neutral_command()
                t = c.backend.telemetry()
                tgt = self.view_target
                if t["position"] is not None and t["heading"] is not None and tgt is not None:
                    d = max(1.0, _d(t["position"], tgt[:2]))
                    att = c.backend.attitude() or {}
                    pitch = math.degrees(att.get("pitch", 0.0)) if isinstance(att, dict) else 0.0
                    cmd.update(az_deg=(_bearing(t["position"], tgt[:2]) - t["heading"] + 180) % 360 - 180,
                               el_cam_deg=math.degrees(math.atan2(tgt[2] - (t["rel_alt"] or 0), d)) - 30.0 - pitch)
                    self.cam_now = (cmd["az_deg"], cmd["el_cam_deg"], d)
                c.update(cmd, (0.0, 0.0), have_target=True, detected=True, conf=0.8)
        self.in_view, self.view_target, self.cam_now = False, None, None
        threading.Thread(target=run, daemon=True).start()
        threading.Thread(target=camera, daemon=True).start()
        self.wait(lambda: m.seeker.cue.target(time.perf_counter()) is not None, 10, "ASTERIX track on UDP")
        cmd("msn", "cue:select:3C0A55")

    def _drive(self, m, cmd):
        sc, sitl = self.sc, self.sitl
        deadline = time.perf_counter() + sc.timeout_s
        aborted = took = False
        while (m.seeker.active or m.seeker.resume is not None) and not (m.seeker.run or {}).get("state") == "AI":
            if time.perf_counter() > deadline:
                cmd("abort", "1")
                raise TimeoutError(f"mission still running after {sc.timeout_s:g} s")
            r = m.seeker.run
            h = (r or {}).get("tel", {}).get("rel_alt") if r else None
            if h is not None and h > 3.0 and not getattr(self, "_thr_mid", False):
                self._thr_mid = True
                self.note(f"PILOT: throttle to MIDDLE at {h:.1f} m")
                sitl.rc.set_stick("throttle", .5)
            hm = None
            if r and r.get("home") and r["tel"].get("position"):
                hm = _d(r["home"], r["tel"]["position"])
            if not aborted and ((sc.abort_at_m and h is not None and h >= sc.abort_at_m)
                                or (sc.abort_at_home_m and hm is not None and hm >= sc.abort_at_home_m)):
                aborted = True
                self.note(f"OPERATOR: ABORT at {h:.1f} m" + (f", {hm:.0f} m from home" if hm is not None else ""))
                cmd("abort", "1")
            if sc.takeover_at_m and not took and h is not None and h >= sc.takeover_at_m:
                took = True
                self.note(f"PILOT: switch UP -> MIDDLE (STABILIZE) at {h:.1f} m")
                sitl.rc.set(6, SWITCH["middle"])
                took_t = time.perf_counter()
            if took and time.perf_counter() - took_t > 15 and sitl.mode != "LAND" and self.c.armed:
                self.note("PILOT: lands (LAND)")
                sitl.set_mode("LAND")
            if r and r.get("wait_restart") and not getattr(self, "_restarted", False):
                self._restarted = True
                time.sleep(2.0)
                self.note("OPERATOR: RESTART APP")
                cmd("msn", "restart")
            o = r and r.get("offer")
            if o:
                self.__dict__.setdefault("_offer_t", {}).setdefault(o["n"], time.perf_counter())
                if time.perf_counter() - self._offer_t[o["n"]] < sc.answer_after_s:
                    o = None
            if o and o["n"] != getattr(self, "_answered", None):
                self._answered = o["n"]
                if o["kind"] == "engage":
                    self.cam_at_answer = getattr(self, "cam_now", None)
                answers = self.__dict__.setdefault("_answers", list(sc.answers))
                ans = answers.pop(0) if answers else "yes"
                self.asked = self.__dict__.get("asked", []) + [(o["kind"], o.get("key"), ans)]
                self.note(f"OPERATOR: {ans.upper()} to '{o['text']}'")
                cmd("msn", f"offer:{ans}:{o['n']}")
            a = m.seeker.alert
            if a and a.get("shown_t") is None and time.perf_counter() - a["t"] > 0.5:
                cmd("msn", f"alertshown:{a['n']}")
            time.sleep(0.1)

    def _serve(self, c, m):
        """The phone UI, fed by the real mission state, for screenshots."""
        import mjpeg_stream
        from mission import phone_command

        def state():
            return dict(linked=c.linked, armed=c.armed, fc_mode=c.mode_name, detector="TensorRT", fps=30.0,
                        lat_ms=12, tilt_deg=30.0, target="DRONE", console_seq=0, cam_aspect=4/3,
                        mission=c.mission_state())
        with contextlib.redirect_stdout(io.StringIO()):
            srv = mjpeg_stream.MJPEGServer(port=self.serve, state_get=state,
                                           command=lambda k, v: phone_command(m, k, v, time.perf_counter()))
            srv.start()
        self.note(f"UI on http://127.0.0.1:{self.serve}/")
        return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="", help="only scenarios whose name contains this")
    ap.add_argument("-j", type=int, default=3, help="scenarios at a time")
    ap.add_argument("--out", default=str(Path(tempfile.gettempdir())/"sitl-seeker-out"))
    ap.add_argument("--serve", type=int, default=None, help="serve the phone UI on this port (with -j 1)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    todo = [s for s in scenarios() if a.k in s.name]
    if a.list:
        print("\n".join(s.name for s in todo))
        return 0
    Path(a.out).mkdir(parents=True, exist_ok=True)
    results = []
    instances = queue.Queue()
    for i in range(max(1, a.j)):
        instances.put(i + 1)

    def one(s):
        n = instances.get()
        try:
            return Run(n, s, a.out, a.serve if a.j == 1 else None).run()
        finally:
            instances.put(n)
    with ThreadPoolExecutor(max_workers=max(1, a.j)) as ex:
        futs = {ex.submit(one, s): s for s in todo}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            fails = [f"{n}: {d}" for n, ok, d in r.get("checks", []) if ok is False]
            print(f"{'ok  ' if r['ok'] else 'FAIL'} {r['scenario']:<30} {r.get('verdict')!s:<11} "
                  f"expect {r['expect']:<10} {r.get('duration_s', 0)!s:>6} s  {r.get('why', '')[:90]}"
                  + ("".join(f"\n       - {x[:150]}" for x in fails[:6])), flush=True)
    Path(a.out, "sitl_seeker_results.json").write_text(json.dumps(results, indent=1, default=str))
    bad = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(bad)}/{len(results)} as expected; details in {a.out}/sitl_seeker_results.json")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
