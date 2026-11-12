"""Seeker test missions: one-click scripted flights for the client test reports.

Each mission (M1-M7) is a script the phone starts from the MISSIONS button.
It flies in GUIDED (ArduCopter): the climbs and the lines are flown with
MAVLink velocity targets along a straight line, corrected towards it by GPS
(wind, a heavy side), the holds with GPS position targets; it returns with
RTL and lands at the take-off point.  ABORT is always RTL.  The pilot's mode
switch takes over at any moment: the mission then never sends another command.

Safety that no mission screen can switch off:
  * the geofence (mission.py's Geofence): a drone heading out of it, closer
    than the fence margin, or outside it, is sent home (RTL) with an alert;
  * the altitude limit (the NOTAM limit): targets are capped below it and a
    drone above it is brought down to it, then home;
  * pre-flight checks: GPS, battery, FC link, pilot radio, geofence, route
    inside the fence, FC settings.  Any failure blocks ARM, with the reason.

Every run writes one CSV (kind = sample / event / check / result) to
<log_dir>/seeker_missions/ for the KPI reports.
"""
from __future__ import annotations

import csv
import json
import math
import os
import socket
import threading
from datetime import datetime, timezone

from asterix_cue import AsterixCue

EARTH_R = 6371000.0

GROUND = ("M1", "M2")
FAILSAFE_MODES = ("RTL", "LAND", "SMART_RTL", "AUTO_RTL")
FENCED = ("M6", "SEEK")         # only these use the geofence; the other missions ignore it completely
HOVER_MODE = "BRAKE"            # holds position and ignores the sticks (LOITER would follow a low throttle down)
ABORT_MODES = {"HOVER": HOVER_MODE, "RTL": "RTL", "LAND": "LAND"}
BATT_FS_MODES = {1: "LAND", 2: "RTL", 3: "SMART_RTL", 4: "SMART_RTL"}
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_REQUEST_MESSAGE = 512
MSG_AUTOPILOT_VERSION = 148

# Pass limits of the checkpoints.
HOLD_ALT_RMS_M = 1.0
HOLD_DRIFT_M = 3.0
TAKEOFF_TILT_DEG = 30.0        # wind and gusts lean it at take-off; more than this is not wind
RETURN_ACCURACY_M = 5.0
SPEED_RATIO = 0.8
FENCE_BREACH_M = 0.0
LINE_ERR_M = 3.0                # a straight line: never further than this off it
TAKEOFF_M = 3.0                 # NAV_TAKEOFF leaves the ground; the velocity line climbs from there
M7_SPEED = 8.0                  # M7 flies its line at this speed (m/s)
M7_CLIMB = 5.0                  # and climbs at this one

FC_PARAMS = ("FS_GCS_ENABLE", "FS_GCS_TIMEOUT", "BATT_FS_LOW_ACT", "BATT_LOW_VOLT", "WPNAV_SPEED",
             "WPNAV_SPEED_UP", "WPNAV_SPEED_DN", "WPNAV_ACCEL", "WPNAV_ACCEL_Z", "RTL_ALT", "FENCE_ENABLE",
             "FENCE_ALT_MAX", "FENCE_TYPE", "DISARM_DELAY", "SIM_RATE_HZ")


def _num(k, label, unit, default, lo, hi, step=1):
    return dict(k=k, t=label, u=unit, d=default, lo=lo, hi=hi, step=step)


def _choice(k, label, default, choices):
    return dict(k=k, t=label, d=default, ch=list(choices))


LIMITS = [
    _num("alt_limit", "Altitude limit (NOTAM)", "m", 120, 10, 500, 5),
    _num("fence_margin", "Geofence margin", "m", 10, 3, 50, 1),
    _num("min_battery", "Minimum battery to start", "%", 60, 20, 100, 5),
]
CLIMB = lambda d=6: _num("climb", "Climb speed (straight up)", "m/s", d, 1, 15, 0.5)
CATALOG = [
    dict(id="M1", name="Start-up & shut-down check", short="Ground: ARM by itself, props spin, stop",
         params=[_num("arm_s", "Props spinning", "s", 5, 1, 30)]),
    dict(id="M2", name="GCS & link check", short="Ground: telemetry, commands, RC, video",
         params=[_num("window_s", "Measuring time", "s", 10, 5, 60)]),
    dict(id="M3", name="Altitude ladder", short="Straight up to 30 m, hold, 60 m, hold, RTL",
         params=[_num("alt1", "First height", "m", 30, 5, 500), _num("hold1", "First hold", "s", 10, 0, 600),
                 _num("alt2", "Second height", "m", 60, 5, 500), _num("hold2", "Second hold", "s", 10, 0, 600),
                 CLIMB()]),
    dict(id="M4", name="Automatic route", short="Straight up, then a straight line out to the slant distance, RTL",
         params=[_num("alt", "Climb straight up to", "m", 30, 5, 500),
                 _num("angle", "Line angle (0 = level, 90 = up)", "°", 0, 0, 90),
                 _num("heading", "Heading", "°", 0, 0, 359),
                 _num("slant", "Stop at slant distance from home", "m", 200, 20, 5000, 10),
                 _num("speed", "Line speed", "m/s", 15, 1, 25, 0.5), CLIMB()]),
    dict(id="M5", name="Abort test", short="Straight up towards 200 m, auto ABORT at 100 m, hover, RTL",
         params=[_num("target", "Climb towards", "m", 200, 10, 500), _num("abort_at", "Auto ABORT at", "m", 100, 5, 500),
                 _num("hover_s", "Hover after the ABORT", "s", 5, 0, 120), CLIMB()]),
    dict(id="M6", name="Geofence test", short="Straight up, then a straight line at the fence: stop, RTL",
         params=[_num("alt", "Climb straight up to", "m", 30, 5, 500), _num("heading", "Heading to the fence", "°", 0, 0, 359),
                 _num("angle", "Line climb angle (0 = level)", "°", 0, 0, 60),
                 _num("speed", "Line speed", "m/s", 8, 1, 25, 0.5), _num("max_dist", "Give up after", "m", 1000, 50, 5000, 10),
                 CLIMB()]),
    dict(id="M7", name="Link-loss test", short="Up, a line, the Jetson goes silent: the FC flies home (RTL)",
         params=[_num("alt", "Climb straight up to", "m", 30, 5, 500),
                 _num("angle", "Line angle (90 = only up)", "°", 45, 0, 90),
                 _num("dist", "Line length", "m", 50, 0, 2000, 10)]),
]
# The Seeker's own job (not a test): fly the ASTERIX cue in GUIDED at the FC's
# top speed until the camera sees the target, then ALT_HOLD and the AI.
SEEK = dict(id="SEEK", name="Seek the ASTERIX target", short="Fly to the cue, the AI takes over on sight",
            params=[_num("speed", "Speed (FC max ≈ WPNAV_SPEED)", "m/s", 20, 2, 25, 0.5),
                    _num("below", "Fly this far below the target", "m", 15, 0, 100),
                    _num("standoff", "Stop short of the target", "m", 50, 0, 1000, 5),
                    _num("min_alt", "Minimum height", "m", 20, 5, 500),
                    _num("cruise_alt", "Height when the track has none", "m", 40, 5, 500),
                    _num("det_s", "Camera lock before the AI takes over", "s", 0.5, 0, 5, 0.1)])
BY_ID = {m["id"]: m for m in CATALOG + [SEEK]}


class MissionFail(Exception):
    pass


def _offset(pos, north, east):
    lat, lon = pos
    return (lat + math.degrees(north/EARTH_R),
            lon + math.degrees(east/(EARTH_R*math.cos(math.radians(lat)))))


def _toward(pos, heading_deg, dist):
    h = math.radians(heading_deg)
    return _offset(pos, dist*math.cos(h), dist*math.sin(h))


def _dist(a, b):
    lat0 = math.radians(a[0])
    return math.hypot(math.radians(b[1]-a[1])*math.cos(lat0)*EARTH_R, math.radians(b[0]-a[0])*EARTH_R)


def _bearing(a, b):
    lat0 = math.radians(a[0])
    return math.degrees(math.atan2(math.radians(b[1]-a[1])*math.cos(lat0), math.radians(b[0]-a[0]))) % 360


def _fmt(v):
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def line_geometry(mid, p, alt_limit=None):
    """M4 / M6 / M7: (length along the line, horizontal reach, top height) of
    the line that starts at p['alt'] straight above home; None if it has none.
    M6's line ends 5 m under the altitude limit (or at its give-up distance)."""
    if mid not in ("M4", "M6", "M7"):
        return None
    el = math.radians(p["angle"])
    x = p["alt"]
    if mid == "M6":
        s = p["max_dist"]
        if alt_limit is not None and math.sin(el) > 1e-3:
            s = max(0.0, min(s, (alt_limit - 5.0 - x)/math.sin(el)))
    elif mid == "M4":
        # Out along the line until the slant distance from home: |(0, x) + s(cos, sin)| = slant.
        disc = p["slant"]**2 - (x*math.cos(el))**2
        s = max(0.0, -x*math.sin(el) + math.sqrt(disc)) if disc > 0 else 0.0
    else:
        s = p["dist"]
    return s, s*math.cos(el), x + s*math.sin(el)


def describe(mid, p, lim):
    """The steps of a mission with these parameters, as the phone lists them."""
    top = lim["alt_limit"]
    start = ["Pre-flight checks", "GUIDED + ARM (by itself)", "Wait 3 s, take off"]
    up = lambda h: f"Straight up to {_fmt(h)} m at {_fmt(p['climb'])} m/s (GPS holds the line)"
    if mid == "M1":
        return ["Pre-flight checks", "ARM-block self-test (injected low battery, no GPS)", "GUIDED + ARM (by itself)",
                f"Props spinning {_fmt(p['arm_s'])} s on the ground", "Disarm: props stop"]
    if mid == "M2":
        return ["FC link", f"Measure {_fmt(p['window_s'])} s: telemetry, commands, RC, video", "Report"]
    if mid == "M3":
        return start + [up(p["alt1"]), f"Hold {_fmt(p['hold1'])} s (GPS)", up(p["alt2"]),
                        f"Hold {_fmt(p['hold2'])} s (GPS)", "Return and land at the take-off point (RTL)"]
    if mid == "M4":
        s, h, z = line_geometry(mid, p)
        line = ("Straight on up" if p["angle"] >= 90 else
                f"Straight line {_fmt(p['heading'])}°, {_fmt(p['angle'])}° up" if p["angle"] else
                f"Straight level line {_fmt(p['heading'])}°")
        return start + [up(p["alt"]), f"{line} at {_fmt(p['speed'])} m/s until {_fmt(p['slant'])} m slant from home "
                        f"({s:.0f} m, ends {h:.0f} m out at {z:.0f} m)", "Return and land at the take-off point (RTL)"]
    if mid == "M5":
        tgt = min(p["target"], top - 2)
        return start + [f"Straight up towards {_fmt(tgt)} m at {_fmt(p['climb'])} m/s",
                        f"Automatic ABORT at {_fmt(p['abort_at'])} m: stop ({HOVER_MODE})",
                        f"Hover {_fmt(p['hover_s'])} s", "Return and land (RTL)"]
    if mid == "M6":
        line = f"Straight line {_fmt(p['heading'])}°, {_fmt(p['angle'])}° up" if p["angle"] else \
            f"Straight level line {_fmt(p['heading'])}°"
        return start + [up(p["alt"]), f"{line} at {_fmt(p['speed'])} m/s towards the geofence",
                        "Geofence: stop, alert, return and land (RTL)"]
    if mid == "M7":
        s, h, z = line_geometry(mid, p)
        line = "Straight on up" if p["angle"] >= 90 else f"Line {_fmt(p['angle'])}° up where the nose points"
        return start + [f"Straight up to {_fmt(p['alt'])} m", f"{line}, {_fmt(p['dist'])} m (ends at {z:.0f} m)",
                        "Hold; the FC's link-loss failsafe on (RTL)",
                        "Press RESTART APP: the whole app goes down in flight",
                        "App down: the FC must fly home by itself (RTL)",
                        "App back: the FC setting restored, it lands at the take-off point"]
    if mid == "SEEK":
        return ["Pre-flight checks", "GUIDED + ARM", "Wait 3 s", "Take off to 5 m",
                        f"Climb and turn to the ASTERIX target, follow it at {_fmt(p['speed'])} m/s, "
                        f"{_fmt(p['below'])} m under it, {_fmt(p['standoff'])} m short, nose on it "
                        "(a new target on the feed: asks to switch)",
                        f"Camera on the target {_fmt(p['det_s'])} s: asks ENGAGE? "
                        "YES → ALT_HOLD, the AI takes it. NO → hold, wait for another target. "
                        "ABORT or the geofence: RTL"]
    return []


def summary(mid, p, lim):
    """One plain sentence: what the drone will do."""
    if mid == "M1":
        return f"Stays on the ground: arms by itself, the props spin {_fmt(p['arm_s'])} s, then it disarms."
    if mid == "M2":
        return f"Stays on the ground: measures the FC link, commands, the pilot radio and video for {_fmt(p['window_s'])} s."
    if mid == "M3":
        return (f"Straight up to {_fmt(p['alt1'])} m, holds {_fmt(p['hold1'])} s, straight to {_fmt(p['alt2'])} m, "
                f"holds {_fmt(p['hold2'])} s, flies home and lands.")
    if mid == "M4":
        s, h, z = line_geometry(mid, p)
        return (f"Straight up to {_fmt(p['alt'])} m, then a straight line, heading {_fmt(p['heading'])}°, "
                f"{'level' if not p['angle'] else 'climbing ' + str(_fmt(p['angle'])) + '°'}, at {_fmt(p['speed'])} m/s; "
                f"stops {_fmt(p['slant'])} m from home ({h:.0f} m out, {z:.0f} m high), flies home and lands.")
    if mid == "M5":
        return (f"Straight up towards {_fmt(min(p['target'], lim['alt_limit'] - 2))} m; at {_fmt(p['abort_at'])} m the app "
                f"presses ABORT by itself: it stops, hovers {_fmt(p['hover_s'])} s, flies home and lands.")
    if mid == "M6":
        return (f"Straight up to {_fmt(p['alt'])} m, then a straight line, heading {_fmt(p['heading'])}°, "
                f"{'level' if not p['angle'] else 'climbing ' + str(_fmt(p['angle'])) + '°'}, at {_fmt(p['speed'])} m/s at the geofence: "
                "it must stop before it, sound the alarm, fly home and land.")
    if mid == "M7":
        return (f"Straight up to {_fmt(p['alt'])} m, a {_fmt(p['dist'])} m line {_fmt(p['angle'])}° up where the nose "
                "points, then holds. You press RESTART APP: the app goes down in flight, the FC must fly home by "
                "itself; the app comes back and shows the result.")
    return ""


def profile(mid, p, lim):
    """The planned flight seen from the side, [[metres from home, height], ...],
    and marks on it, for the mission screen's picture."""
    if mid in ("M1", "M2", "SEEK") or mid not in BY_ID:
        return [], []
    top = lim["alt_limit"]
    if mid == "M3":
        return [[0, 0], [0, p["alt1"]], [0, p["alt2"]]], [
            dict(d=0, h=p["alt1"], t=f"{_fmt(p['alt1'])} m · hold {_fmt(p['hold1'])} s"),
            dict(d=0, h=p["alt2"], t=f"{_fmt(p['alt2'])} m · hold {_fmt(p['hold2'])} s")]
    if mid == "M5":
        tgt = min(p["target"], top - 2)
        return [[0, 0], [0, tgt]], [dict(d=0, h=p["abort_at"], t=f"{_fmt(p['abort_at'])} m · auto ABORT", hot=True)]
    s, h, z = line_geometry(mid, p, top)
    pts = [[0, 0], [0, p["alt"]], [round(h, 1), round(z, 1)]]
    mark = {"M4": lambda: f"{_fmt(p['slant'])} m slant: stop", "M6": lambda: "give up here",
            "M7": lambda: "RESTART APP here"}[mid]()
    return pts, [dict(d=round(h, 1), h=round(z, 1), t=mark, hot=mid == "M7")]


class RunLog:
    COLUMNS = ("utc", "t_s", "kind", "mission", "step", "text", "result", "mode", "armed",
               "rel_alt_m", "target_alt_m", "gs_mps", "vz_mps", "lat", "lon", "home_m", "slant_m",
               "battery_v", "battery_pct", "fc_hb_age_s", "rc_age_s", "video_age_s",
               "mav_lost", "mav_rx", "rejected", "fence_edge_m", "fence_inside", "tilt_deg")

    def __init__(self, directory, mid, name, append=None):
        self.path = None
        self.error = ""
        self._f = self._w = None
        if append:
            # The same run after an app restart (M7): carry on in its file.
            try:
                self.path = append
                self._f = open(append, "a", newline="")
                self._w = csv.DictWriter(self._f, fieldnames=self.COLUMNS, extrasaction="ignore")
            except OSError as exc:
                self.error = str(exc)
            return
        try:
            os.makedirs(directory, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
            n = 1
            self.path = os.path.join(directory, f"{stamp}_{mid}_{slug}.csv")
            while os.path.exists(self.path):
                n += 1
                self.path = os.path.join(directory, f"{stamp}_{mid}_{slug}_{n}.csv")
            self._f = open(self.path, "x", newline="")
            self._w = csv.DictWriter(self._f, fieldnames=self.COLUMNS, extrasaction="ignore")
            self._w.writeheader()
        except OSError as exc:
            self.error = str(exc)

    def row(self, **kw):
        if self._w is None:
            return
        kw.setdefault("utc", datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
        try:
            self._w.writerow({k: ("" if v is None else round(v, 3) if isinstance(v, float) else v)
                              for k, v in kw.items()})
            self._f.flush()
        except (OSError, ValueError):
            pass

    def close(self):
        if self._f is not None:
            try:
                self._f.close()
            except OSError:
                pass
            self._f = self._w = None


class SeekerMissions:
    VERSION = 2
    RESUME_MAX_S = 900.0     # an M7 restart this old is not picked up again
    SAMPLE_S = 0.5
    CHECK_S = 0.5
    RESEND_S = 1.0
    BRAKE_DECEL = 4.0

    def __init__(self, path=None, log_dir=None, log=(lambda *_a: None)):
        self.path = path
        self.log_dir = log_dir
        self.log = log
        self._lock = threading.Lock()
        self._reqs = []
        saved = self._read()
        self.lim = {x["k"]: x["d"] for x in LIMITS}
        self.params = {m["id"]: {x["k"]: x["d"] for x in m["params"]} for m in BY_ID.values()}
        for x in LIMITS:
            self._apply(self.lim, x, saved.get("limits", {}).get(x["k"]))
        for m in BY_ID.values():
            if m["id"] != "SEEK" and saved.get("version", 1) < self.VERSION:
                continue            # mission settings from before the M1-M7 rework: start from the new defaults
            for x in m["params"]:
                self._apply(self.params[m["id"]], x, saved.get(m["id"], {}).get(x["k"]))
        r = saved.get("resume")
        self.resume = r if isinstance(r, dict) and r.get("id") == "M7" else None
        self.restart_hook = None     # tests: stands in for the systemd restart
        self.dead = False            # restarting: this app sends nothing more
        self.restore = {k: float(v) for k, v in (saved.get("restore") or {}).items()
                        if isinstance(v, (int, float))}
        cue = saved.get("cue") if isinstance(saved.get("cue"), dict) else {}
        try:
            port = int(cue.get("port", os.environ.get("ASTERIX_UDP_PORT", 8600)))
        except (TypeError, ValueError):
            port = 8600
        self.cue = AsterixCue(port, cue.get("mcast", os.environ.get("ASTERIX_MCAST", "")),
                              cue.get("iface", os.environ.get("ASTERIX_IFACE", "0.0.0.0")))
        self.handover = None
        self.seek_pre = []
        self.ctx = {}
        self._cue_started = False
        self._offer_n = 0
        self._asked_batt = False
        self.viewing = None
        self.sim = False
        self.pre = []
        self._pre_t = -1e9
        self._asked_t = -1e9
        self._restore_t = -1e9
        self.alert = None
        self._alert_n = 0
        self.result = None
        self.run = None
        self.now = 0.0
        self.b = self.c = None

    # ---- settings ----------------------------------------------------------

    def _read(self):
        if not self.path:
            return {}
        try:
            with open(self.path) as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self):
        if not self.path:
            return True
        d = dict(version=self.VERSION, limits=self.lim, restore=self.restore, resume=self.resume,
                 cue=dict(port=self.cue.port, mcast=self.cue.mcast, iface=self.cue.iface), **self.params)
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(d, f, indent=1)
            os.replace(tmp, self.path)
            return True
        except OSError:
            return False

    @staticmethod
    def _apply(store, spec, value):
        """Set store[spec.k] from value if it is valid; '' or the reason."""
        if value is None:
            return ""
        if "ch" in spec:
            v = str(value).strip().upper()
            if v not in spec["ch"]:
                return f"{spec['t']}: one of {', '.join(spec['ch'])}"
            store[spec["k"]] = v
            return ""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return f"{spec['t']}: a number"
        if not math.isfinite(v) or not spec["lo"] <= v <= spec["hi"]:
            return f"{spec['t']}: {spec['lo']:g} to {spec['hi']:g} {spec['u']}"
        store[spec["k"]] = int(v) if float(v).is_integer() else v
        return ""

    # ---- requests from the phone (HTTP thread) -----------------------------

    @property
    def active(self):
        # Only a run in this process: a saved M7 restart waiting to be picked up
        # must never take WASP's ABORT or settings.
        return self.run is not None

    def command(self, value):
        """msn=<verb>[:args]: view:M3, set:M3:alt1:60, set:limits:alt_limit:120,
        start:M3, next, abort, after:RTL|LAND, alertshown:N, alertok:N, dismiss."""
        verb, _, rest = str(value).strip().partition(":")
        verb = verb.lower()
        if verb == "view":
            mid = rest.strip().upper()
            if mid and mid not in BY_ID:
                return "unknown mission"
            with self._lock:
                self.viewing = mid or None
                self._pre_t = -1e9
                self._asked_t = -1e9
            return ""
        if verb == "set":
            mid, _, kv = rest.partition(":")
            key, _, val = kv.partition(":")
            mid, key = mid.strip(), key.strip()
            with self._lock:
                if self.run is not None:
                    return "change mission settings after the run"
                if mid == "limits":
                    spec, store = next((x for x in LIMITS if x["k"] == key), None), self.lim
                else:
                    m = BY_ID.get(mid.upper())
                    spec = m and next((x for x in m["params"] if x["k"] == key), None)
                    store = m and self.params[m["id"]]
                if not spec:
                    return "unknown mission setting"
                why = self._apply(store, spec, val)
                self._pre_t = -1e9
                if why:
                    return why
                return "" if self._save() else "could not save on the Jetson"
        if verb == "cue":
            what, _, arg = rest.partition(":")
            what = what.strip().lower()
            if what == "select":
                return self.cue.select(arg)
            if what == "nearest":
                return self.cue.select("")
            if what == "udp":
                port, _, more = arg.partition(":")
                mcast, _, iface = more.partition(":")
                try:
                    port = int(port)
                    if not 1 <= port <= 65535:
                        raise ValueError
                    for a in (mcast, iface):
                        if a:
                            socket.inet_aton(a)
                except (ValueError, OSError):
                    return "ASTERIX: port 1-65535, multicast and interface as IPv4 addresses (or empty)"
                with self._lock:
                    if self.run is not None:
                        return "change the ASTERIX input on the ground"
                self.cue.configure(port, mcast.strip(), iface.strip() or "0.0.0.0")
                if not self._save():
                    return "could not save on the Jetson"
                return self.cue.error
            return "cue: select:<track>, nearest or udp:<port>:<multicast>:<interface>"
        if verb == "offer":
            ans, _, n = rest.partition(":")
            if ans.strip().lower() not in ("yes", "no") or not n.strip().isdigit():
                return "offer: yes:<n> or no:<n>"
            with self._lock:
                if self.run is None:
                    return "no mission running"
                self._reqs.append(("offer", f"{ans.strip().lower()}:{n.strip()}"))
            return ""
        if verb in ("start", "next", "abort", "after", "dismiss", "alertshown", "alertok", "restart"):
            with self._lock:
                if verb == "start":
                    if self.run is not None:
                        return "a mission is already running"
                    if rest.strip().upper() not in BY_ID:
                        return "unknown mission"
                elif verb in ("next", "after") and self.run is None:
                    return "no mission running"
                elif verb == "restart" and not (self.run and self.run.get("wait_restart")):
                    return "RESTART APP only at that step of M7"
                self._reqs.append((verb, rest.strip().upper()))
            return ""
        return "missions: view, set, start, next, abort, after, dismiss"

    def request_abort(self):
        with self._lock:
            self._reqs.append(("abort", ""))

    # ---- pre-flight checks -------------------------------------------------

    def checks(self, mid, mission, inject=None):
        """[(name, ok, detail, blocking)] for starting mission mid now.
        inject: 'battery' / 'gps' fakes that failure (the M1 self-test)."""
        b = self.b
        out = []

        def add(name, ok, detail="", blocking=True):
            out.append(dict(name=name, ok=bool(ok), detail=detail, blocking=blocking))
        hb = None if b is None else b.telemetry().get("hb_age")
        add("FC link", b is not None and b.linked and hb is not None and hb < 1.5,
            "no ArduPilot MAVLink" if b is None else f"heartbeat {hb:.1f} s ago" if hb is not None else "no heartbeat")
        if b is None:
            return out
        if mid == "M2":
            return out
        if mission.wasp.enabled:
            add("Seeker mode", False, "WASP is selected")
        add("Disarmed", not b.armed, "disarm first" if b.armed else "")
        g = b.gps_status()
        gps = b.gps_fix_ok() and b.position_ok() and inject != "gps"
        add("GPS", gps, "injected failure" if inject == "gps" else
            f"{g['sats']} sat, fix {g['fix']}" + ("" if b.position_ok() else ", EKF has no position"))
        t = b.telemetry()
        pct, volt = t["battery_pct"], t["battery_v"]
        low = b.params.get("BATT_LOW_VOLT")
        if inject == "battery":
            add("Battery", False, "injected failure: 5 %")
        elif pct is None and volt is None:
            add("Battery", False, "no battery reading from the FC")
        else:
            # The percentage (capacity used) and the voltage against the FC's
            # own low level: a sagging pack can still read 100 %.
            ok = (pct is None or pct >= self.lim["min_battery"]) and (volt is None or not low or volt >= low + 0.3)
            add("Battery", ok and (pct is not None or bool(low)),
                ", ".join(x for x in (f"{pct} % (min {self.lim['min_battery']:g} %)" if pct is not None else "",
                                      f"{volt:.1f} V" + (f" (FC low {low:.1f} V)" if low else "") if volt else "") if x))
        sw = b.pilot_mode_switch()
        add("Pilot radio", sw is not None, f"mode switch: {sw[1] or sw[0]}" if sw else
            "no RC from the transmitter: the pilot could not take over")
        if sw is not None and mid not in GROUND:
            # Switch UP (ALT_HOLD) before the flight: then UP -> MIDDLE (STABILIZE)
            # takes over in every phase, the AI's ALT_HOLD included.
            up = sw[1] == "ALT_HOLD"
            add("Mode switch UP (ALT_HOLD)", up, "takeover: switch to MIDDLE (throttle at middle by then)" if up else
                f"switch is on {sw[1] or sw[0]}: put it UP", blocking=mid == "SEEK")
        if mid != "M2":
            # ArduCopter refuses ARM ("Throttle not neutral") with the stick up, even in GUIDED.
            thr = self._throttle()
            add("Throttle stick DOWN", thr is not None and thr < 0.1,
                "down: raise it to MIDDLE once it is in the air" if thr is not None and thr < 0.1 else
                "no throttle reading from the transmitter" if thr is None else
                f"at {100*thr:.0f} %: the FC will refuse ARM; put it DOWN")
        f, (inside, edge) = mission.fence, mission._fence_state
        if mid not in FENCED:
            pass
        elif not f.armed:
            add("Geofence", False, "no geofence: draw one (Geofence button)")
        else:
            add("Geofence", inside is True, f"inside, {edge:.0f} m to the edge" if inside else
                "drone outside the geofence" if inside is False else "no GPS position to check it")
        fence_on = int(b.params.get("FENCE_ENABLE") or 0)
        fc_top = b.params.get("FENCE_ALT_MAX")
        if mid not in GROUND:
            top = self._top(mid)
            add("Altitude limit", top <= self.lim["alt_limit"],
                f"mission {top:g} m, limit {self.lim['alt_limit']:g} m")
            if fence_on and fc_top and int(b.params.get("FENCE_TYPE") or 0) & 1:
                add("FC altitude fence", top <= fc_top - 2, f"FC FENCE_ALT_MAX {fc_top:g} m")
            why = self._route_check(mid, mission, b.position())
            if why is not None:
                add("Route", not why, why or ("reaches the geofence" if mid == "M6" else "inside the geofence"))
        p = self.params[mid]
        if mid == "M1":
            # On the ground with the throttle stick down the FC disarms by itself after DISARM_DELAY.
            dd = b.params.get("DISARM_DELAY")
            add("Props time under the FC auto-disarm", dd is not None and (dd == 0 or p["arm_s"] <= dd - 1),
                "DISARM_DELAY not read from the FC yet" if dd is None else
                f"{p['arm_s']:g} s; the FC disarms on the ground after {dd:g} s (DISARM_DELAY)" if dd else
                "FC auto-disarm off")
        if mid == "M4":
            s = line_geometry(mid, p)[0]
            add("Line length", s >= 10, f"{s:.0f} m along the line" if s >= 10 else
                f"the slant distance {p['slant']:g} m must be well beyond the climb ({p['alt']:g} m)")
        if mid == "M5":
            add("Abort height", p["abort_at"] < min(p["target"], self.lim["alt_limit"]) - 2,
                f"ABORT at {p['abort_at']:g} m, under the climb target and the limit")
        if mid == "M7":
            fs = b.params.get("FS_GCS_ENABLE")
            add("GCS failsafe setting", fs is not None, "FS_GCS_ENABLE read" if fs is not None else
                "FS_GCS_ENABLE not read from the FC yet")
            unit = self._unit()
            add("App can restart itself", bool(unit or self.restart_hook),
                f"systemd {unit}: comes back by itself" if unit else "test restart" if self.restart_hook else
                "the app is not running as the systemd service: it would not come back")
        if mid == "SEEK":
            law = getattr(self.c, "law", None)
            tilt = math.degrees(getattr(law, "cam_tilt", math.radians(30)))
            cam_ok = abs(mission.camera_mount_deg - 30) < 0.1 and abs(tilt - 30) < 0.1
            add("Camera 30°", cam_ok, "Seeker camera at 30°" if cam_ok else
                f"camera set to {mission.camera_mount_deg:g}° / the AI expects {tilt:.0f}°")
            pos = b.position()
            tgt = self.cue.target(self.now, pos, self._home_amsl(t))
            st = self.cue.status(self.now)
            add("ASTERIX target", tgt is not None,
                f"{tgt['label']}: {tgt['dist_m']} m at {tgt['bearing']}°" + (
                    f", {tgt['alt_rel']:+.0f} m" if tgt.get("alt_rel") is not None else "") if tgt else
                st["error"] or (f"no fresh track on UDP {st['port']}" + (f" ({self.cue.selected} selected)"
                                                                          if self.cue.selected else "")))
            if tgt and f.armed:
                inside_t = f.check((tgt["lat"], tgt["lon"]))[0]
                add("Target inside the geofence", inside_t is True,
                    "yes" if inside_t else "no: the Seeker will stop at the geofence", blocking=False)
        video = self.c is not None and self.c.frame_t is not None and self.now - self.c.frame_t < 1.0
        add("Video", video, "camera frames" if video else "no camera frames", blocking=False)
        return out

    def _top(self, mid):
        p = self.params[mid]
        if mid in GROUND:
            return 0
        if mid == "M3":
            return max(p["alt1"], p["alt2"])
        if mid in ("M4", "M7"):
            return round(line_geometry(mid, p)[2], 1)
        if mid == "M5":
            # Where it stops after the ABORT: the reaction plus braking at the FC's vertical acceleration.
            v, a = p["climb"], self._accel_z()
            return round(p["abort_at"] + 0.5*v + v*v/(2*a), 1)
        if mid == "SEEK":
            return max(p["min_alt"], p["cruise_alt"])
        return p["alt"]

    def _accel_z(self):
        a = (self.b.params.get("WPNAV_ACCEL_Z") if self.b is not None else None) or 100.0
        return max(0.5, a/100.0)

    def _accel_xy(self):
        a = (self.b.params.get("WPNAV_ACCEL") if self.b is not None else None) or 250.0
        return max(0.5, a/100.0)

    def _throttle(self):
        """The pilot's throttle stick, 0 (down) .. 1, or None without a reading."""
        b = self.b
        cfg = getattr(b, "rc", {}).get("throttle")
        pwm = b.pilot_pwm("throttle") if callable(getattr(b, "pilot_pwm", None)) else None
        if not cfg or pwm is None:
            return None
        f = (pwm - cfg["min"]) / max(1.0, cfg["max"] - cfg["min"])
        return max(0.0, min(1.0, 1.0 - f if cfg.get("reversed") else f))

    @staticmethod
    def _home_amsl(t):
        return None if t.get("amsl") is None or t.get("rel_alt") is None else t["amsl"] - t["rel_alt"]

    def _route_check(self, mid, mission, pos):
        """'' if the route stays inside the fence (by its margin), the reason
        if not, None when the mission has no route."""
        p, f = self.params[mid], mission.fence
        if mid != "M6" or not f.armed:
            return None
        if pos is None:
            return "no GPS position to check the route"
        margin = self.lim["fence_margin"]
        if mid == "M6":
            reach = line_geometry(mid, p, self.lim["alt_limit"])[1]
            for d in range(0, int(reach) + 1, 5):
                inside, edge = f.check(_toward(pos, p["heading"], d))
                if inside is False or edge < margin:
                    # Room to reach line speed and still stop before the margin.
                    v = p["speed"]*math.cos(math.radians(p["angle"]))
                    need = v*v/(2*self._accel_xy()) + 0.8*v + v*v/(2*self.BRAKE_DECEL)
                    return "" if d >= need else (f"the geofence is {d} m away at {p['heading']:g}°: "
                                                 f"{need:.0f} m needed at {v:g} m/s (lower the speed)")
            return (f"heading {p['heading']:g}° does not reach the geofence within {reach:.0f} m"
                    + (" (the climbing line meets the altitude limit first)" if reach < p["max_dist"] - 1 else ""))
        dist = self._route_dist(mid)
        pts = [_toward(pos, self._heading(mid), d) for d in list(range(0, int(dist), 10)) + [dist]]
        for q in pts:
            inside, edge = f.check(q)
            if inside is not True or edge < margin:
                return f"the route comes within {margin:g} m of the geofence: change it or the mission"
        return ""

    def _heading(self, mid):
        """The line's heading: M7 has none of its own, it flies where the nose points."""
        if mid != "M7":
            return self.params[mid]["heading"]
        r = self.run
        if r is not None and r["id"] == "M7" and r.get("heading") is not None:
            return r["heading"]
        h = self.b.telemetry().get("heading") if self.b is not None else None
        return 0.0 if h is None else h

    def _route_dist(self, mid):
        return line_geometry(mid, self.params[mid])[1]

    def route(self, mid, pos):
        """The planned track over the ground from pos, for the phone's map."""
        if mid not in BY_ID or pos is None:
            return []
        p = self.params[mid]
        if mid in ("M4", "M7"):
            return [list(pos), list(_toward(pos, self._heading(mid), self._route_dist(mid)))]
        if mid == "M6":
            # As far as the geofence (where it must stop), not the give-up distance.
            reach = line_geometry(mid, p, self.lim["alt_limit"])[1]
            f = getattr(self, "_fence", None)
            if f is not None and f.armed:
                for d in range(0, int(reach) + 1, 5):
                    if f.check(_toward(pos, p["heading"], d))[0] is not True:
                        reach = d
                        break
            return [list(pos), list(_toward(pos, p["heading"], reach))]
        return [list(pos)]

    # ---- the cycle (MAVLink TX thread) ---------------------------------------

    def cycle(self, now, mapper, mission, ctx):
        self.now = now
        self._fence = mission.fence
        if self.dead:
            return
        self.c = mapper
        b = getattr(mapper, "backend", None)
        self.b = b if callable(getattr(b, "guided_goto", None)) else None
        if self.run is None and self.resume is not None and self.b is not None and self.b.linked \
                and getattr(self.b, "mode_name", None):
            self._pick_up(mission)
        with self._lock:
            reqs, self._reqs = self._reqs, []
        for verb, arg in reqs:
            self._request(verb, arg, mission)
        if self.b is not None:
            self._background(mission)
        self.ctx = ctx
        if not self._cue_started:
            self._cue_started = True
            self.cue.start()
        if self.run is None:
            if now - self._pre_t >= self.CHECK_S:
                self._pre_t = now
                pre = self.checks(self.viewing, mission) if self.viewing else []
                seek = self.checks("SEEK", mission)
                with self._lock:
                    self.pre, self.seek_pre = pre, seek
            return
        r = self.run
        r["tel"] = self.b.telemetry()
        r["fence"] = mission._fence_state
        if self.b.armed:
            r["armed_once"] = True
        if r["pos_last"] is None or r["tel"]["position"] is not None:
            r["pos_last"] = r["tel"]["position"] or r["pos_last"]
        if r["tel"]["position"] is not None and now - r["trail_t"] >= 1.0:
            r["trail_t"] = now
            r["trail"].append([round(x, 7) for x in r["tel"]["position"]])
            del r["trail"][:-900]
            if r["home"] is not None and r["tel"]["rel_alt"] is not None:
                r.setdefault("side", []).append([round(_dist(r["home"], r["tel"]["position"]), 1),
                                                 round(r["tel"]["rel_alt"], 1)])
                del r["side"][:-900]
        if not r["hands_off"]:
            self._monitor(mission)
        if r["armed_once"] and not self.b.armed and not r["expect_disarm"]:
            self._event("disarmed")
            return self._finish()
        try:
            next(r["gen"])
        except StopIteration:
            return self._finish()
        except MissionFail as exc:
            self.check(r["steps"][r["step_i"]] if r["step_i"] < len(r["steps"]) else "Mission", False, str(exc))
            self._event(f"FAIL: {exc}")
            self._interrupt(f"{exc}", self._abort_gen(r["abort"], str(exc)))
        if now - r["sample_t"] >= self.SAMPLE_S:
            r["sample_t"] = now
            self._row("sample")

    def _request(self, verb, arg, mission):
        r = self.run
        if verb == "start":
            if r is None:
                self._start(arg, mission)
        elif verb == "next" and r is not None:
            r["next"] = True
        elif verb == "restart" and r is not None and r.get("wait_restart") and not r["hands_off"]:
            r["restart"] = True
        elif verb == "abort":
            if r is not None and r["hands_off"]:
                self._event("ABORT ignored: the pilot has control")
            elif r is not None and not r["aborted"]:
                self._event("ABORT button")
                self._interrupt("ABORT button", self._abort_gen(r["abort"], "ABORT button"))
        elif verb == "after" and r is not None and arg in ("RTL", "LAND") and not r["hands_off"]:
            self._event(f"operator: {arg}")
            self._interrupt(f"operator {arg}", self._mode_then_watch(arg))
        elif verb == "offer" and r is not None:
            ans, _, n = arg.lower().partition(":")
            r["answer"] = (int(n), ans == "yes")
        elif verb == "dismiss":
            self.result = None
        elif verb in ("alertshown", "alertok") and self.alert and str(self.alert["n"]) == arg:
            key = "shown_t" if verb == "alertshown" else "ok_t"
            if self.alert.get(key) is None:
                self.alert[key] = self.now
                self._event(f"alert {'shown on the phone' if verb == 'alertshown' else 'acknowledged'}")
            if verb == "alertok":
                self.alert["open"] = False

    def _background(self, mission):
        b = self.b
        if b.linked and self.now - self._asked_t > 20.0:
            self._asked_t = self.now
            b.request_params(FC_PARAMS)
        self.sim = "SIM_RATE_HZ" in b.params
        self.keep_fc(self.now, self.c)

    def keep_fc(self, now, mapper):
        """Put back an FC setting a link-loss test changed (also after a crash
        mid-test), in either flight mode; nothing to do almost always.  Also
        asks once for the FC's low battery voltage, for the battery indicator."""
        b = getattr(mapper, "backend", None)
        if not callable(getattr(b, "set_param", None)):
            return
        if not self._asked_batt and b.linked:
            self._asked_batt = True
            b.request_params(("BATT_LOW_VOLT",))
        if not self.restore:
            return
        self.now = now
        if b.linked and not b.gcs_muted and (self.run is None or self.run["id"] != "M7"):
            if all(b.params.get(k) == v for k, v in self.restore.items()):
                self.restore = {}
                self._save()
            elif self.now - self._restore_t > 2.0:
                self._restore_t = self.now
                for k, v in self.restore.items():
                    b.set_param(k, v)

    def _start(self, mid, mission):
        m = BY_ID[mid]
        p = dict(self.params[mid])
        steps = describe(mid, p, self.lim)
        log = RunLog(os.path.join(self.log_dir or ".", "seeker_missions"), mid, m["name"])
        self.result = None
        self.alert = None
        self.run = r = self._new_run(mid, p, dict(self.lim), steps, log, mission)
        tel = r["tel"]
        r["heading"] = tel.get("heading") if mid == "M7" else None
        r["route"] = self.route(mid, tel.get("position"))
        r["gen"] = self._script(mid, mission)
        self._event(f"start {mid} {m['name']}: " + ", ".join(f"{k}={_fmt(v)}" for k, v in p.items())
                    + f"; limit {self.lim['alt_limit']:g} m; sim={self.sim}")
        if log.error:
            self._event(f"CSV log unavailable: {log.error}")

    def _new_run(self, mid, p, lim, steps, log, mission):
        return dict(
            id=mid, name=BY_ID[mid]["name"], p=p, lim=lim, steps=steps, step_i=0, step_t=self.now,
            t0=self.now, state="RUNNING", why="", abort="RTL", trail=[], trail_t=-1e9, line=None,
            log=log, checks=[], events=[], next=False, wait_next=False, armed_once=False,
            expect_disarm=False, expect_fs=False, hands_off=False, aborted=False, home=None,
            target_alt=None, cmd_mode=None, cmd_t=-1e9, switch0=None, sample_t=-1e9,
            tel=self.b.telemetry() if self.b else {}, fence=mission._fence_state, edge_hist=[],
            pos_last=None, breach=None, abort_info=None, fs=None, metrics={}, heading=None, route=[],
            wait_restart=False, restart=False)

    # ---- M7: the app restarts in flight, and picks its run up again ---------

    @staticmethod
    def _unit():
        """The systemd unit this app runs in (it restarts the app), or None."""
        if not os.environ.get("INVOCATION_ID"):
            return None
        try:
            with open("/proc/self/cgroup") as f:
                path = f.read().strip().splitlines()[0]
        except (OSError, IndexError):
            return None
        # The innermost unit (".../user@1000.service/app.slice/wasp-tracker.service"):
        # never the user manager itself, restarting that would end the whole session.
        unit = path.rstrip("/").rsplit("/", 1)[-1]
        return unit if unit.endswith(".service") and not unit.startswith("user@") else None

    def restart_app(self):
        """Take the whole app down and up again (systemd starts it again)."""
        if self.restart_hook is not None:
            return self.restart_hook()
        unit = self._unit()
        if unit:
            import subprocess
            subprocess.Popen(["systemctl", "--user", "--no-block", "restart", unit], start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _save_resume(self):
        r = self.run
        self.resume = dict(id=r["id"], wall=datetime.now(timezone.utc).timestamp(), elapsed=self.now - r["t0"],
                           p=r["p"], lim=r["lim"], steps=r["steps"], csv=r["log"].path, home=r["home"],
                           heading=r["heading"], route=r["route"], trail=r["trail"], side=r.get("side", []),
                           checks=r["checks"],
                           metrics=r["metrics"], events=r["events"], switch0=r["switch0"],
                           arm_wall=None if r.get("arm_t") is None else
                           datetime.now(timezone.utc).timestamp() - (self.now - r["arm_t"]),
                           fs_old=self.restore.get("FS_GCS_ENABLE"))
        return self._save()

    def _pick_up(self, mission):
        """Back after the restart: the M7 run again, from its saved state."""
        z, self.resume = self.resume, None
        self._save()
        age = datetime.now(timezone.utc).timestamp() - float(z.get("wall") or 0)
        if not 0 <= age <= self.RESUME_MAX_S:
            self.log(f"[SEEKER] an M7 restart {age:.0f} s old: not picked up")
            return
        log = RunLog(os.path.join(self.log_dir or ".", "seeker_missions"), "M7", BY_ID["M7"]["name"], append=z.get("csv"))
        self.run = r = self._new_run("M7", z["p"], z["lim"], z["steps"], log, mission)
        r.update(t0=self.now - float(z.get("elapsed") or 0) - age, home=z.get("home"), heading=z.get("heading"),
                 route=z.get("route") or [], trail=z.get("trail") or [], side=z.get("side") or [],
                 checks=z.get("checks") or [],
                 metrics=z.get("metrics") or {}, events=[tuple(e) for e in z.get("events") or []],
                 switch0=z.get("switch0"), armed_once=True, expect_disarm=True, expect_fs=True)
        if z.get("arm_wall"):
            r["arm_t"] = self.now - (datetime.now(timezone.utc).timestamp() - z["arm_wall"])
        r["gen"] = self._m7_back(z, age)

    def _m7_back(self, z, age):
        r, b = self.run, self.b
        self._at(8)
        t0 = self.now
        while b.mode_name is None and self.now - t0 < 5.0:
            yield
        mode, armed = b.mode_name, b.armed
        r["cmd_mode"], r["cmd_t"] = mode, self.now
        self._event(f"APP BACK {age:.0f} s after the restart: FC in {mode}, {'armed' if armed else 'disarmed'}")
        self.check("App down: the FC flew home by itself", mode in FAILSAFE_MODES,
                   f"FC in {mode} when the app came back, {age:.0f} s after the restart")
        old = z.get("fs_old")
        if old is not None:
            yield from self._set_param("FS_GCS_ENABLE", float(old), 8.0)
            restored = b.params.get("FS_GCS_ENABLE") == float(old)
            if restored:
                self.restore = {}
                self._save()
            self.check("FC setting restored", restored, f"FS_GCS_ENABLE back to {float(old):g}"
                       if restored else "not confirmed yet: the app keeps retrying")
        if armed:
            yield from self._land_watch("Failsafe RTL", 900)
        else:
            self._landed("Failsafe RTL")

    def _script(self, mid, mission):
        if self.b is None:
            raise MissionFail("Seeker missions need ArduPilot MAVLink")
        return getattr(self, "_" + mid.lower())(mission)

    # ---- safety monitors -------------------------------------------------------

    def _monitor(self, mission):
        r, b = self.run, self.b
        if not b.armed:
            return
        # The pilot's switch against its position at ARM, or a mode nobody
        # here asked for (and no FC failsafe): the pilot has it.
        sw = b.pilot_mode_switch()
        moved = sw is not None and r["switch0"] is not None and sw[0] != r["switch0"]
        mode = b.mode_name
        if (not r.get("hands_off") and mode == "ACRO"
                and r["cmd_mode"] and mode != r["cmd_mode"]):
            thr = self._throttle()
            ch8 = callable(getattr(b, "pilot_throttle_wants_takeover", None)) and b.pilot_throttle_wants_takeover()
            if ch8 or (thr is not None and thr >= 0.70):
                reason = "CH8 button" if ch8 else f"throttle {thr:.0%}"
                r["hands_off"] = True
                self._event(f"PILOT TAKEOVER (ACRO + {reason}): full manual control")
                self.check("ACRO takeover", None, reason)
                r["state"], r["why"] = "PILOT", "the pilot has control"
                self._replace(self._watch())
                return
        late = self.now - r["cmd_t"] > 2.5
        if moved or (r["cmd_mode"] and mode != r["cmd_mode"] and late and mode not in FAILSAFE_MODES):
            r["hands_off"] = True
            why = f"pilot switch to {sw[1] or sw[0]}" if moved else f"FC in {mode}"
            self._event(f"PILOT TAKEOVER ({why}): the mission sends nothing more")
            self.check("RC takeover", None, why)
            r["state"], r["why"] = "PILOT", "the pilot has control"
            self._replace(self._watch())
            return
        if (b.armed and r["cmd_mode"] and mode != r["cmd_mode"] and late and mode in FAILSAFE_MODES
                and not r["expect_fs"] and not r["aborted"]):
            self._event(f"FC failsafe: {mode}. " + self._fc_text())
            r["fs"] = (self.now, mode)
            self._interrupt(f"FC failsafe {mode}", self._watch(), mode=mode)
            return
        if not b.armed or r["aborted"]:
            return
        t = r["tel"]
        alt = t["rel_alt"]
        if alt is not None and alt > r["lim"]["alt_limit"] + 1.0:
            self._raise_alert("ALTITUDE", f"ALTITUDE LIMIT: {alt:.0f} m > {r['lim']['alt_limit']:g} m")
            r["breach"] = dict(kind="altitude", t=self.now, value=alt - r["lim"]["alt_limit"])
            self._interrupt("altitude limit", self._alt_recover())
            return
        inside, edge = r["fence"]
        if r["id"] not in FENCED or not mission.fence.armed or inside is None or alt is None or alt < 1.0:
            return
        hist = r["edge_hist"]
        hist.append((self.now, edge if inside else -edge))
        while hist and hist[0][0] < self.now - 1.0:
            hist.pop(0)
        closing = len(hist) > 2 and hist[-1][1] < hist[0][1] - 0.2 and self.now - hist[0][0] > 0.4
        # Stop in time: the margin plus what it covers before BRAKE bites and
        # while it brakes (ArduCopter's BRAKE: about 7.5 m/s², taken as 4).
        v = t["gs"] or 0.0
        need = r["lim"]["fence_margin"] + 0.8*v + v*v/(2*self.BRAKE_DECEL)
        if inside is False or (edge < need and closing and v > 0.5):
            self._raise_alert("FENCE", "GEOFENCE: " + (f"{edge:.0f} m OUTSIDE" if inside is False
                                                        else f"{edge:.0f} m from the edge") + ": returning home")
            r["breach"] = dict(kind="fence", t=self.now, pos=t["position"], inside=inside, edge=edge,
                               worst=(-edge if inside is False else edge))
            self._interrupt("geofence", self._mode_then_watch("RTL", fence=True))

    def _replace(self, gen):
        # Close the script it replaces: its finally blocks (the link-loss
        # test's mute) run now, not whenever it is collected.
        old, self.run["gen"] = self.run.get("gen"), gen
        if old is not None:
            try:
                old.close()
            except (ValueError, RuntimeError):
                pass
        if self.b is not None and self.b.gcs_muted:
            self.b.gcs_muted = False

    def _raise_alert(self, kind, text):
        self._alert_n += 1
        self.alert = dict(n=self._alert_n, kind=kind, text=text, t=self.now, open=True, shown_t=None, ok_t=None)
        self._event(f"ALERT {kind}: {text}")

    def _interrupt(self, why, gen, mode=None):
        r = self.run
        r["aborted"] = True
        r["state"], r["why"] = "ABORTED", why
        r["wait_next"] = False
        self._replace(gen)
        if mode:
            r["cmd_mode"], r["cmd_t"] = mode, self.now

    # ---- helpers the scripts yield from -------------------------------------

    def _at(self, i):
        r = self.run
        r["step_i"], r["step_t"], r["wait_next"] = i, self.now, False
        self._event(f"step {i+1}: {r['steps'][i]}")

    def _mode(self, name, timeout=4.0):
        r, b = self.run, self.b
        r["cmd_mode"], r["cmd_t"] = name, self.now
        t0, sent = self.now, -1e9
        while b.mode_name != name:
            if r["hands_off"]:
                return
            if self.now - sent >= self.RESEND_S:
                sent = self.now
                r["cmd_t"] = self.now
                b.request_mode(name, allow_disarmed=True)
            if self.now - t0 > timeout:
                raise MissionFail(f"the FC did not switch to {name}. " + self._fc_text())
            yield
        r["metrics"].setdefault("mode_s", {})[name] = round(self.now - t0, 2)

    def _arm(self):
        r, b = self.run, self.b
        # GUIDED arms only from a GCS command, not from the arm switch's function.
        if not b.request_arm(mavlink=True):
            raise MissionFail("ARM: the FC link refused the command")
        t0 = self.now
        while not b.armed:
            ack = b.arm_ack()
            if ack is not None and not ack[0] and ack[1] >= t0 - 0.05 and self.now - t0 > 0.5:
                raise MissionFail("ARM refused by the FC. " + self._fc_text())
            if self.now - t0 > 5.0:
                raise MissionFail("the FC did not arm within 5 s. " + self._fc_text())
            yield
        r["home"] = r["tel"]["position"] or (b.home() if callable(getattr(b, "home", None)) else None)
        r["arm_t"] = self.now
        sw = b.pilot_mode_switch()
        r["switch0"] = sw[0] if sw else None
        self.check("ARM", True, f"armed in {self.now - t0:.1f} s")

    def _wait(self, secs, step_mode=False):
        r = self.run
        t0 = self.now
        if step_mode:
            r["wait_next"], r["next"] = True, False
            while not r["next"]:
                yield
            r["wait_next"] = False
            return
        while self.now - t0 < secs:
            yield

    def _takeoff(self, alt):
        r, b = self.run, self.b
        r["target_alt"] = alt
        t0 = self.now
        sent = -1e9
        tilt = 0.0
        while True:
            t = r["tel"]
            h = t["rel_alt"]
            if t["tilt_deg"] is not None:
                tilt = max(tilt, t["tilt_deg"])
            if h is not None and h >= alt - max(0.5, 0.03*alt):
                break
            if self.now - sent >= 3.0 and (h is None or h < 0.5):
                sent = self.now
                b.guided_takeoff(alt)
            ack = b.command_ack(MAV_CMD_NAV_TAKEOFF)
            if ack and ack[0] >= t0 and ack[1] not in (0, 5):
                raise MissionFail(f"take-off refused by the FC (result {ack[1]}). " + self._fc_text())
            if self.now - t0 > alt/0.5 + 30:
                raise MissionFail(f"did not reach {alt:g} m")
            yield
        took = self.now - t0
        peak = r["tel"]["rel_alt"] or 0.0
        t1 = self.now
        while self.now - t1 < 3.0:
            peak = max(peak, r["tel"]["rel_alt"] or 0.0)
            if r["tel"]["tilt_deg"] is not None:
                tilt = max(tilt, r["tel"]["tilt_deg"])
            yield
        r["metrics"]["takeoff"] = dict(s=round(took, 1), max_tilt=round(tilt, 1), overshoot=round(peak-alt, 2))
        self.check(f"Take-off to {alt:g} m", tilt <= TAKEOFF_TILT_DEG,
                   f"{took:.1f} s, max tilt {tilt:.1f}° (limit {TAKEOFF_TILT_DEG:g}°), overshoot {peak-alt:+.1f} m")

    def _goto(self, point, alt, done=None, tol=3.0, timeout=None, sample=None):
        """Fly to point at alt (above home) in GUIDED, resending the target;
        ends on arrival or when done() is true."""
        r, b = self.run, self.b
        alt = min(alt, r["lim"]["alt_limit"] - 1.0)
        r["target_alt"] = alt
        t0, sent = self.now, -1e9
        while True:
            t = r["tel"]
            pos, h = t["position"], t["rel_alt"]
            if sample:
                sample(t)
            if done is not None and done(t):
                return
            if pos is not None and h is not None and _dist(pos, point) <= tol and abs(h - alt) <= 1.5:
                return
            if self.now - sent >= self.RESEND_S:
                sent = self.now
                b.guided_goto(point[0], point[1], alt)
            if timeout and self.now - t0 > timeout:
                raise MissionFail(f"did not reach the target in {timeout:.0f} s")
            yield

    def _hold(self, secs, label, step_mode=False):
        r, b = self.run, self.b
        point, alt = r["tel"]["position"] or r["pos_last"], r["target_alt"]
        if point is None:
            raise MissionFail("no GPS position to hold")
        b.guided_goto(point[0], point[1], alt)
        errs, drift = [], 0.0
        t0, sent = self.now, self.now
        r["wait_next"], r["next"] = step_mode, False
        while (not r["next"]) if step_mode else self.now - t0 < secs:
            t = r["tel"]
            if t["rel_alt"] is not None:
                errs.append(t["rel_alt"] - alt)
            if t["position"] is not None:
                drift = max(drift, _dist(point, t["position"]))
            if self.now - sent >= self.RESEND_S:
                sent = self.now
                b.guided_goto(point[0], point[1], alt)
            yield
        r["wait_next"] = False
        if errs:
            rms = math.sqrt(sum(e*e for e in errs)/len(errs))
            worst = max(abs(e) for e in errs)
            r["metrics"].setdefault("holds", []).append(dict(alt=alt, s=round(self.now-t0, 1), rms=round(rms, 2),
                                                             max=round(worst, 2), drift=round(drift, 2)))
            self.check(f"{label}: height", rms <= HOLD_ALT_RMS_M,
                       f"RMS {rms:.2f} m, max {worst:.2f} m (limit RMS {HOLD_ALT_RMS_M:g} m)")
            self.check(f"{label}: position", drift <= HOLD_DRIFT_M,
                       f"drift {drift:.1f} m (limit {HOLD_DRIFT_M:g} m)")

    LINE_HZ = 10.0
    LINE_GAIN = 0.8          # m/s back towards the line per metre off it (GPS: wind, a heavy side)
    LINE_CORR_MAX = 3.0
    LINE_BRAKE = 0.7         # of the FC's acceleration limit, for slowing down to the end

    DESCENT_MAX = 2.5        # m/s straight down (faster sinks into its own downwash)

    def _up(self, alt, label, climb=None):
        """Straight up (or down) from here to alt at the mission's climb speed."""
        r = self.run
        here = r["tel"]["position"] or r["pos_last"]
        if here is None:
            raise MissionFail("no GPS position")
        h0 = r["tel"]["rel_alt"] or 0.0
        v = climb or r["p"]["climb"]
        if abs(alt - h0) <= 0.5:
            return
        if alt < h0:
            v = min(v, self.DESCENT_MAX)
        yield from self._vline(label, here, h0, 0.0, 90.0 if alt > h0 else -90.0, abs(alt - h0), v, v)

    def _vline(self, label, origin, alt0, heading, el_deg, length, speed, climb, done=None, yaw=None):
        """Fly the straight line from origin (lat, lon) at alt0 above home, heading
        and el_deg (0 level, 90 straight up) its direction, length metres long:
        velocity targets along it at speed (its vertical part at most climb),
        slowing down in time for its end, plus a GPS correction back onto it; at
        the end a position target holds it there.  Ends there, or on done(t)."""
        r, b = self.run, self.b
        el, hd = math.radians(el_deg), math.radians(heading)
        ce, se = math.cos(el), math.sin(el)
        u = (ce*math.cos(hd), ce*math.sin(hd), se)                 # north, east, up
        az = abs(se)
        v_max = min(speed if ce > 1e-3 else climb, climb/az if az > 1e-3 else 1e9)
        # The FC's own acceleration limits (WPNAV_ACCEL, WPNAV_ACCEL_Z) along this line.
        acc = self.LINE_BRAKE*min(self._accel_xy()/ce if ce > 1e-3 else 1e9, self._accel_z()/az if az > 1e-3 else 1e9)
        end = _toward(origin, heading, length*ce)
        end_alt = min(alt0 + length*se, r["lim"]["alt_limit"] - 1.0)
        coslat = math.cos(math.radians(origin[0]))
        r["line"] = dict(origin=list(origin), end=list(end), alt0=round(alt0, 1), end_alt=round(end_alt, 1))
        r["target_alt"] = end_alt
        t0, sent = self.now, -1e9
        worst, speeds, final = 0.0, [], False
        timeout = length/max(0.5, 0.3*v_max) + 60
        while True:
            t = r["tel"]
            pos, h = t["position"], t["rel_alt"]
            if done is not None and done(t):
                break
            if pos is None or h is None:
                raise MissionFail("lost the GPS position on the line")
            n = math.radians(pos[0] - origin[0])*EARTH_R
            e = math.radians(pos[1] - origin[1])*EARTH_R*coslat
            z = h - alt0
            s = n*u[0] + e*u[1] + z*u[2]
            off = (n - s*u[0], e - s*u[1], z - s*u[2])
            err = math.sqrt(sum(c*c for c in off))
            rem = length - s
            if not final and rem < 1.5:
                final = True
                sent = -1e9
            if final:
                if self.now - sent >= self.RESEND_S:
                    sent = self.now
                    b.guided_goto(end[0], end[1], end_alt, yaw_deg=yaw)
                if _dist(pos, end) <= 2.0 and abs(h - end_alt) <= 1.0 and (t["gs"] or 0) < 1.0 \
                        and abs(t["vd"] or 0) < 0.5:
                    break
            else:
                if 2.0 < s and rem > 2.0:
                    worst = max(worst, err)
                vel = (t["vn"], t["ve"], None if t["vd"] is None else -t["vd"])
                if None not in vel and 0.15*length < s < 0.85*length:
                    speeds.append(vel[0]*u[0] + vel[1]*u[1] + vel[2]*u[2])
                va = max(0.5, min(v_max, math.sqrt(2*acc*max(0.0, rem - 1.0))))
                g = min(self.LINE_GAIN, self.LINE_CORR_MAX/err) if err > 1e-6 else 0.0
                vn, ve, vu = (va*u[i] - g*off[i] for i in range(3))
                if self.now - sent >= 1.0/self.LINE_HZ:
                    sent = self.now
                    b.guided_velocity(vn, ve, -vu, yaw_deg=yaw)
            if self.now - t0 > timeout:
                raise MissionFail(f"{label}: not at the end of the line in {timeout:.0f} s")
            yield
        m = r["metrics"].setdefault("lines", [])
        m.append(dict(label=label, length=round(length, 1), set=round(v_max, 2), max_off=round(worst, 2),
                      mean=round(sum(speeds)/len(speeds), 2) if speeds else None,
                      max=round(max(speeds), 2) if speeds else None, s=round(self.now - t0, 1)))
        self.check(f"{label}: straight", worst <= LINE_ERR_M,
                   f"at most {worst:.1f} m off the line (limit {LINE_ERR_M:g} m)")
        if speeds:
            peak = max(speeds)
            # Short lines never reach the set speed: judge the speed it could reach on this length.
            reach = min(v_max, math.sqrt(acc*length))
            self.check(f"{label}: speed", peak >= SPEED_RATIO*reach,
                       f"max {peak:.1f} m/s, mean {sum(speeds)/len(speeds):.1f} m/s, set {v_max:.1f} m/s"
                       + (f" (this length allows about {reach:.1f})" if reach < v_max - 0.1 else ""))

    def _climb_cap(self):
        up = self.b.params.get("WPNAV_SPEED_UP")
        return up/100.0 if up else None

    def _rtl(self, label="Return and land"):
        r = self.run
        r["expect_disarm"] = True
        yield from self._mode("RTL")
        yield from self._land_watch(label, 900)

    def _land_watch(self, label, timeout, at_home=True):
        r, b = self.run, self.b
        r["expect_disarm"] = True
        t0, st = self.now, {}
        while b.armed:
            if self.now - t0 > timeout:
                raise MissionFail(f"not landed within {timeout:g} s")
            self._ground_disarm(st)
            yield
        self._landed(label, at_home)

    def _landed(self, label, at_home=True):
        r = self.run
        pos, home = r["pos_last"], r["home"]
        if pos is not None and home is not None:
            d = _dist(home, pos)
            r["metrics"]["return_m"] = round(d, 2)
            if at_home:
                self.check(f"{label}: at the take-off point", d <= RETURN_ACCURACY_M,
                           f"{d:.1f} m from the take-off point (limit {RETURN_ACCURACY_M:g} m)")
            else:
                self.check(f"{label}: landed where it was", None, f"{d:.1f} m from the take-off point")
        else:
            self.check(f"{label}: at the take-off point", None, "no GPS position")
        if r.get("arm_t") is not None:
            r["metrics"]["flight_s"] = round(self.now - r["arm_t"], 1)

    def _watch(self):
        r = self.run
        r["expect_disarm"] = True
        st = {}
        while self.b.armed or not r["armed_once"]:
            if not r["armed_once"]:
                return
            self._ground_disarm(st)
            yield
        self._landed("Landed")

    GROUND_S = 6.0           # down and still this long in RTL / LAND but armed: the app disarms

    def _ground_disarm(self, st):
        """An automatic landing that ends on the ground without the FC disarming
        (tipped over in a gust, the land detector never trips): disarm it, then
        force it, so the props do not keep spinning on the ground."""
        r, b = self.run, self.b
        t = r["tel"]
        down = (b.armed and not r["hands_off"] and b.mode_name in ("RTL", "LAND", "SMART_RTL")
                and t["rel_alt"] is not None and t["rel_alt"] < 1.0
                and abs(t["vd"] or 0.0) < 0.3 and (t["gs"] or 0.0) < 0.5)
        if not down:
            st.clear()
            return
        st.setdefault("t", self.now)
        if self.now - st["t"] >= self.GROUND_S and "sent" not in st:
            st["sent"] = self.now
            self._event(f"on the ground {self.GROUND_S:g} s but still armed (tilt {t['tilt_deg'] or 0:.0f}°): disarm")
            b.request_disarm()
        elif "sent" in st and self.now - st["sent"] >= 3.0 and "forced" not in st:
            st["forced"] = True
            self._event("the FC did not disarm: forced disarm")
            b.request_disarm(force=True)

    def _mode_then_watch(self, mode, fence=False):
        r = self.run
        r["expect_disarm"] = True
        if fence:
            # Geofence: BRAKE (the hardest stop), then home.
            yield from self._mode(HOVER_MODE, timeout=3.0)
            t0 = self.now
            while self.now - t0 < 8.0 and ((r["tel"]["gs"] or 0) > 0.5 or self.now - t0 < 1.0):
                self._fence_worst()
                yield
        yield from self._mode(mode, timeout=6.0)
        if fence:
            yield from self._fence_watch()
        yield from self._watch()

    def _fence_worst(self):
        inside, edge = self.run["fence"]
        br = self.run["breach"]
        if inside is not None:
            br["worst"] = min(br["worst"], -edge if inside is False else edge)

    def _fence_watch(self):
        """After a fence stop: how far past the edge it went, and that it turned back."""
        r = self.run
        br, t0 = r["breach"], self.now
        home = r["home"]
        far = d0 = None
        while self.now - t0 < 8.0:
            self._fence_worst()
            pos = r["tel"]["position"]
            if pos is not None and home is not None:
                d = _dist(home, pos)
                d0 = d if d0 is None else d0
                far = max(far or d, d)
            yield
        pos = r["tel"]["position"]
        br["turned_back"] = bool(pos is not None and home is not None and far is not None
                                 and _dist(home, pos) < far - 2.0)
        br["stop_pos"] = pos

    def _alt_recover(self):
        r, b = self.run, self.b
        r["expect_disarm"] = True
        point = r["tel"]["position"] or r["pos_last"]
        if point is not None and b.mode_name in ("GUIDED",) + (HOVER_MODE,):
            yield from self._mode("GUIDED")
            yield from self._goto(point, r["lim"]["alt_limit"] - 5.0, tol=50.0, timeout=120)
        yield from self._mode("RTL")
        yield from self._watch()

    def _abort_gen(self, action, why):
        """ABORT (button, automatic or a failed step): HOVER / RTL / LAND, the
        reaction timed; on the ground it disarms instead."""
        r, b = self.run, self.b
        r["expect_disarm"] = True
        if not b.armed:
            return iter(())
        t = r["tel"]
        if (t["rel_alt"] is None or t["rel_alt"] < 1.0) and not r.get("took_off"):
            return self._disarm_gen()
        return self._abort_air(action, why)

    def _disarm_gen(self):
        self.b.request_disarm()
        t0 = self.now
        while self.b.armed and self.now - t0 < 3.0:
            yield
        if self.b.armed:
            self.b.request_disarm(force=True)
        yield from self._watch()

    def _abort_air(self, action, why):
        yield from self._abort_react(action, why)
        if action == "HOVER" and self.run["id"] != "M5":
            self.run["state"] = "HOVER"
        yield from self._watch()

    def _abort_react(self, action, why):
        r, b = self.run, self.b
        mode = ABORT_MODES[action]
        t0 = self.now
        climb0 = -(r["tel"]["vd"] or 0.0)
        info = r["abort_info"] = dict(why=why, action=action, mode=mode, alt=r["tel"]["rel_alt"], climb=climb0,
                                      mode_s=None, stop_s=None, end_alt=None)
        r["cmd_mode"], r["cmd_t"] = mode, self.now
        sent = -1e9
        while self.now - t0 < 15.0:
            t = r["tel"]
            if b.mode_name != mode and self.now - sent >= self.RESEND_S:
                sent = self.now
                b.request_mode(mode)
            if b.mode_name == mode and info["mode_s"] is None:
                info["mode_s"] = round(self.now - t0, 2)
                self._event(f"ABORT reaction: FC in {mode} after {info['mode_s']:.2f} s")
            if info["stop_s"] is None and t["vd"] is not None and abs(t["vd"]) < 0.5 and action != "LAND":
                info["stop_s"] = round(self.now - t0, 2)
            if info["mode_s"] is not None and (info["stop_s"] is not None or action == "LAND") \
                    and self.now - t0 > 5.0:
                break
            if not b.armed:
                break
            yield
        info["end_alt"] = r["tel"]["rel_alt"]
        if info["mode_s"] is None:
            raise MissionFail(f"ABORT: the FC did not switch to {mode}")

    # ---- the missions ------------------------------------------------------

    def _preflight(self, mission, inject=None):
        res = self.checks(self.run["id"], mission, inject)
        bad = [c for c in res if c["blocking"] and not c["ok"]]
        if inject is None:
            for c in res:
                self.check(f"Pre-flight: {c['name']}", c["ok"] if c["blocking"] else (c["ok"] or None), c["detail"])
            if bad:
                raise MissionFail("ARM blocked: " + "; ".join(f"{c['name']}: {c['detail']}" for c in bad))
        return bad

    def _start_flight(self, mission, takeoff_alt, fold=False):
        """Checks, GUIDED, ARM, 3 s, take-off.  fold: the take-off is part of
        step 3 (the missions); SEEK lists it as its own step 4."""
        r = self.run
        self._at(0)
        self._preflight(mission)
        yield
        self._at(1)
        yield from self._mode("GUIDED")
        yield from self._arm()
        self._at(2)
        yield from self._wait(3.0)
        if self.b.mode_name != "GUIDED":
            raise MissionFail(f"left GUIDED before take-off ({self.b.mode_name})")
        if not fold:
            self._at(3)
        r["took_off"] = True
        yield from self._takeoff(takeoff_alt)

    def _m1(self, mission):
        r, b = self.run, self.b
        p = r["p"]
        self._at(0)
        self._preflight(mission)
        yield
        self._at(1)
        for inject in ("battery", "gps"):
            bad = self._preflight(mission, inject)
            self.check(f"Failed check blocks ARM ({inject})", any(c["name"] in ("Battery", "GPS") for c in bad),
                       "ARM refused: " + "; ".join(c["name"] for c in bad) if bad else "ARM was NOT blocked")
            yield
        self._at(2)
        yield from self._mode("GUIDED")
        yield from self._arm()
        self._at(3)
        t0 = self.now
        while self.now - t0 < p["arm_s"]:
            if not b.armed:
                raise MissionFail("disarmed early. " + self._fc_text())
            if b.mode_name != "GUIDED":
                raise MissionFail(f"left GUIDED on the ground ({b.mode_name})")
            yield
        h = r["tel"]["rel_alt"]
        self.check("Props spinning, on the ground", b.armed and (h is None or h < 0.5),
                   f"{p['arm_s']:g} s armed" + (f", height {h:.2f} m" if h is not None else ""))
        self._at(4)
        r["expect_disarm"] = True
        b.request_disarm()
        t0, again = self.now, False
        while b.armed:
            if self.now - t0 > 1.5 and not again:
                again = True
                b.request_disarm()
            if self.now - t0 > 3.0:
                raise MissionFail("the FC did not disarm. " + self._fc_text())
            yield
        self.check("Disarm: props stopped", True, f"disarmed in {self.now - t0:.1f} s")

    def _m2(self, mission):
        r, b, c = self.run, self.b, self.c
        p = r["p"]
        self._at(0)
        if not b.linked:
            raise MissionFail("no FC link")
        try:
            with open("/proc/uptime") as f:
                boot = self.now - float(f.read().split()[0])
        except (OSError, ValueError, IndexError):
            boot = None
        app0 = getattr(c, "_t0", None)
        link_t, video_t = getattr(b, "link_t", None), getattr(c, "frame_first_t", None)

        def since(t, ref):
            return None if t is None or ref is None else round(t - ref, 1)
        times = dict(fc_after_app_s=since(link_t, app0), fc_after_boot_s=since(link_t, boot),
                     video_after_app_s=since(video_t, app0), video_after_boot_s=since(video_t, boot))
        r["metrics"]["ready"] = times
        self.check("Control ready (FC MAVLink)", link_t is not None,
                   f"{times['fc_after_app_s']} s after app start, {times['fc_after_boot_s']} s after Jetson boot")
        self.check("Video ready", video_t is not None,
                   f"{times['video_after_app_s']} s after app start, {times['video_after_boot_s']} s after boot"
                   if video_t is not None else "no camera frame yet")
        self._at(1)
        s0, t0 = b.link_stats(), self.now
        acks, pending, sent = [], None, -1e9
        rc_gap = video_gap = 0.0
        while self.now - t0 < p["window_s"]:
            if pending is None and self.now - sent >= 2.0:
                sent = pending = self.now
                b._command(MAV_CMD_REQUEST_MESSAGE, MSG_AUTOPILOT_VERSION)
            if pending is not None:
                ack = b.command_ack(MAV_CMD_REQUEST_MESSAGE)
                if ack and ack[0] >= pending:
                    acks.append((self.now - pending, ack[1]))
                    pending = None
                elif self.now - pending > 2.0:
                    acks.append((None, None))
                    pending = None
            t = r["tel"]
            rc_gap = max(rc_gap, t["rc_age"] if t["rc_age"] is not None else 99.0)
            video_gap = max(video_gap, self.now - c.frame_t if getattr(c, "frame_t", None) else 99.0)
            yield
        self._at(2)
        s1, span = b.link_stats(), self.now - t0
        rate = lambda k: (s1["counts"].get(k, 0) - s0["counts"].get(k, 0)) / span
        hb, att, pos = rate("HEARTBEAT"), rate("ATTITUDE"), rate("GLOBAL_POSITION_INT")
        lost, rx = s1["lost"] - s0["lost"], s1["received"] - s0["received"]
        rej = s1["rejected"] - s0["rejected"]
        loss = 100.0*lost/max(1, lost + rx)
        r["metrics"].update(heartbeat_hz=round(hb, 2), attitude_hz=round(att, 1), position_hz=round(pos, 1),
                            lost=lost, received=rx, loss_pct=round(loss, 2), rejected=rej,
                            ack_ms=[None if a is None else round(1000*a) for a, _ in acks],
                            rc_gap_s=round(rc_gap, 2), video_gap_s=round(video_gap, 2))
        self.check("Telemetry", hb >= 0.8 and att >= 10 and pos >= 2,
                   f"heartbeat {hb:.1f} Hz, attitude {att:.0f} Hz, position {pos:.1f} Hz")
        ok_acks = [a for a, res in acks if a is not None and res == 0]
        self.check("Commands acknowledged", acks and len(ok_acks) == len(acks) and max(ok_acks) < 1.0,
                   f"{len(ok_acks)}/{len(acks)} accepted" + (f", max {1000*max(ok_acks):.0f} ms" if ok_acks else ""))
        self.check("Lost / rejected messages", loss < 1.0 and rej == 0,
                   f"{lost} lost of {lost + rx} ({loss:.2f} %), {rej} rejected commands")
        self.check("Pilot radio (RC)", rc_gap < 0.5, f"longest RC gap {rc_gap:.2f} s")
        self.check("Video", video_gap < 1.0, f"longest frame gap {video_gap:.2f} s" if video_gap < 99 else "no frames")

    def _m3(self, mission):
        r = self.run
        p = r["p"]
        yield from self._start_flight(mission, TAKEOFF_M, fold=True)
        self._at(3)
        yield from self._up(p["alt1"], f"Straight up to {p['alt1']:g} m")
        self._at(4)
        yield from self._hold(p["hold1"], f"Hold at {p['alt1']:g} m")
        self._at(5)
        yield from self._up(p["alt2"], f"Straight {'up' if p['alt2'] >= p['alt1'] else 'down'} to {p['alt2']:g} m")
        self._at(6)
        yield from self._hold(p["hold2"], f"Hold at {p['alt2']:g} m")
        self._at(7)
        yield from self._rtl()

    def _m4(self, mission):
        r = self.run
        p = r["p"]
        yield from self._start_flight(mission, TAKEOFF_M, fold=True)
        self._at(3)
        yield from self._up(p["alt"], f"Straight up to {p['alt']:g} m")
        self._at(4)
        home = r["home"] or r["tel"]["position"]
        length = line_geometry("M4", p)[0]
        link = []

        def slant(t):
            return None if t["position"] is None or t["rel_alt"] is None else \
                math.hypot(_dist(home, t["position"]), t["rel_alt"])

        def done(t):
            if t["position"] is not None:
                link.append((_dist(home, t["position"]), t["hb_age"], t["rc_age"],
                             None if self.c.frame_t is None else self.now - self.c.frame_t))
            return (slant(t) or 0) >= p["slant"] - 0.5
        here = r["tel"]["position"] or r["pos_last"]
        yield from self._vline(f"Line {p['heading']:g}°, {p['angle']:g}° up", here, r["tel"]["rel_alt"] or p["alt"],
                               p["heading"], p["angle"], length + 2.0, p["speed"], p["climb"], done=done,
                               yaw=p["heading"] if p["angle"] < 80 else None)
        s = slant(r["tel"])
        self.check(f"Stopped at {p['slant']:g} m slant", s is not None and abs(s - p["slant"]) <= 5,
                   f"{s:.0f} m from home" if s else "no position")
        if link:
            hb = max(x[1] or 0 for x in link)
            rc = max(99 if x[2] is None else x[2] for x in link)
            vid = [x[3] for x in link]
            r["metrics"]["link_far_m"] = round(max(x[0] for x in link))
            self.check("FC link over the route", hb < 1.5, f"longest heartbeat gap {hb:.1f} s")
            self.check("Pilot radio over the route", rc < 1.0, f"longest RC gap {rc:.1f} s")
            self.check("Video over the route", all(v is not None and v < 1.0 for v in vid),
                       "frames all the way" if all(v is not None and v < 1.0 for v in vid) else
                       "video gaps (see the CSV video_age_s against home_m)")
        self._at(5)
        yield from self._rtl()

    def _m5(self, mission):
        r = self.run
        p = r["p"]
        yield from self._start_flight(mission, TAKEOFF_M, fold=True)
        self._at(3)
        tgt = min(p["target"], r["lim"]["alt_limit"] - 2.0)
        here, h0 = r["tel"]["position"] or r["pos_last"], r["tel"]["rel_alt"] or 0.0
        yield from self._vline(f"Climb towards {tgt:g} m", here, h0, 0.0, 90.0, tgt - h0, p["climb"], p["climb"],
                               done=lambda t: (t["rel_alt"] or 0) >= p["abort_at"])
        self._at(4)
        self._event(f"AUTOMATIC ABORT at {r['tel']['rel_alt']:.1f} m: stop ({HOVER_MODE})")
        r["aborted"] = True
        yield from self._abort_react("HOVER", "automatic ABORT")
        info = r["abort_info"]
        rise = (info["end_alt"] or 0) - (info["alt"] or 0)
        self.check("ABORT: FC reaction", info["mode_s"] is not None and info["mode_s"] < 2.0,
                   f"FC in {info['mode']} {info['mode_s']} s after the ABORT")
        self.check("ABORT: climb stopped", info["stop_s"] is not None and info["stop_s"] < 6.0,
                   f"vertical speed under 0.5 m/s after {info['stop_s']} s, {rise:+.1f} m after the ABORT "
                   f"(at {info['climb']:.1f} m/s)")
        self._at(5)
        yield from self._wait(p["hover_s"])
        self._at(6)
        yield from self._rtl()

    def _m6(self, mission):
        r = self.run
        p = r["p"]
        yield from self._start_flight(mission, TAKEOFF_M, fold=True)
        self._at(3)
        yield from self._up(p["alt"], f"Straight up to {p['alt']:g} m")
        self._at(4)
        here = r["tel"]["position"] or r["pos_last"]
        length = line_geometry("M6", p, r["lim"]["alt_limit"])[0]
        yield from self._vline(f"Line {p['heading']:g}°, {p['angle']:g}° up, to the geofence", here,
                               r["tel"]["rel_alt"] or p["alt"], p["heading"], p["angle"], length, p["speed"], p["climb"],
                               yaw=p["heading"])
        raise MissionFail("reached the give-up distance without a geofence stop")

    def _m7(self, mission):
        r, b = self.run, self.b
        p = r["p"]
        yield from self._start_flight(mission, TAKEOFF_M, fold=True)
        self._at(3)
        yield from self._up(p["alt"], f"Straight up to {p['alt']:g} m", M7_CLIMB)
        self._at(4)
        if p["dist"] > 0:
            here = r["tel"]["position"] or r["pos_last"]
            yield from self._vline(f"Line {p['angle']:g}° up", here, r["tel"]["rel_alt"] or p["alt"], r["heading"] or 0.0,
                                   p["angle"], p["dist"], M7_SPEED, M7_CLIMB,
                                   yaw=r["heading"] if p["angle"] < 80 else None)
        self._at(5)
        old = b.params.get("FS_GCS_ENABLE")
        self.restore = {"FS_GCS_ENABLE": old}
        self._save()
        hold = self._hold(3.0, "Hold before the link loss")
        sets = self._set_param("FS_GCS_ENABLE", 1.0, 6.0)
        for _ in hold:
            next(sets, None)
            yield
        yield from sets
        if b.params.get("FS_GCS_ENABLE") != 1.0:
            raise MissionFail("the FC did not take FS_GCS_ENABLE")
        self.check("GCS failsafe on", True, f"FS_GCS_ENABLE {old:g} -> 1 (RTL), restored after the test")
        self._at(6)
        # Hold here until the operator restarts the app (ABORT still works).
        point, alt = r["tel"]["position"] or r["pos_last"], r["target_alt"]
        r["wait_restart"], sent = True, -1e9
        while not r["restart"]:
            if self.now - sent >= self.RESEND_S:
                sent = self.now
                b.guided_goto(point[0], point[1], alt)
            yield
        r["wait_restart"] = False
        self._at(7)
        r["expect_fs"] = True
        r["state"], r["why"] = "RESTARTING", "the app restarts: the FC must fly home by itself"
        self._event("RESTART APP pressed: the app goes down now; nothing more from it until it is back")
        if not self._save_resume():
            raise MissionFail("could not save the run before the restart")
        # From here this app sends nothing (no heartbeat either): the FC sees the link go.
        b.gcs_muted = True
        self.dead = True
        self.restart_app()
        while True:
            yield

    def _set_param(self, name, value, timeout):
        b, t0, sent = self.b, self.now, -1e9
        while b.params.get(name) != value and self.now - t0 <= timeout:
            if self.now - sent >= 1.5:
                sent = self.now
                b.set_param(name, value)
            yield


    def _offer(self, kind, tgt, text):
        r = self.run
        self._offer_n += 1
        r["offer"] = dict(n=self._offer_n, kind=kind, key=tgt.get("key") if tgt else None,
                          label=(tgt or {}).get("label"), text=text, t=self.now)
        r["answer"] = None
        self._event(f"ASK: {text}")

    def _seek(self, mission):
        """Up and round to the cue, follow it in GUIDED until the camera has the
        target, then ALT_HOLD and the AI.  A new track on the feed asks to switch
        (or to follow, while it waits with no cue)."""
        r, b = self.run, self.b
        p = r["p"]
        t = r["tel"]
        first = self.cue.target(self.now, t.get("position"), self._home_amsl(t))
        if first is not None:
            self.cue.select(first["key"])     # pinned: 'nearest' must not jump mid-flight
        r["known"] = {x["key"] for x in self.cue.tracks(self.now)}
        r["rejected"], r["offer"], r["answer"] = set(), None, None
        yield from self._start_flight(mission, 5.0)
        self._at(4)
        b.change_speed(b.SPEED_GROUND, p["speed"])
        cap = self._climb_cap()
        if cap:
            b.change_speed(b.SPEED_CLIMB, cap)
        r["cue"], r["seek"] = None, "FOLLOW"
        hold, sent, det_since, lost_note = None, -1e9, None, False
        cue_lost_since = None
        while True:
            t = r["tel"]
            pos = t["position"] or r["pos_last"]
            home_amsl = self._home_amsl(t)
            # The operator's answer to the open question.
            o, ans = r["offer"], r["answer"]
            if o is not None and ans is not None and ans[0] == o["n"]:
                r["offer"], r["answer"] = None, None
                self._event(f"ANSWER {'YES' if ans[1] else 'NO'}: {o['text']}")
                if o["kind"] in ("switch", "follow") and ans[1]:
                    was_hold = r["seek"] == "HOLD"
                    self.cue.select(o["key"])
                    r["seek"], hold, det_since, lost_note = "FOLLOW", None, None, False
                    cue_lost_since = None
                    if was_hold:
                        yield from self._mode("GUIDED")
                        b.change_speed(b.SPEED_GROUND, p["speed"])
                elif o["kind"] == "engage":
                    if ans[1]:
                        break
                    r["rejected"].add(o["key"])
                    self.cue.select("")
                    r["seek"], r["cue"], det_since = "WAIT", None, None
                    self._event("NO: target rejected, waiting here for another ASTERIX target")
            # A track new on the feed: ask.
            if r["offer"] is None:
                for x in self.cue.tracks(self.now, pos, home_amsl):
                    if x["fresh"] and x["key"] not in r["known"] and x["key"] not in r["rejected"]:
                        r["known"].add(x["key"])
                        cur = (r["cue"] or {}).get("label") or self.cue.selected
                        if r["seek"] == "FOLLOW" and cur:
                            self._offer("switch", x, f"New target {x['label']} ({x['dist_m']} m, {x['bearing']}°): "
                                                     f"switch from {cur}?")
                        else:
                            self._offer("follow", x, f"New target {x['label']} ({x['dist_m']} m, {x['bearing']}°): follow it?")
                        break
            tgt = self.cue.target(self.now, pos, home_amsl) if r["seek"] not in ("WAIT", "HOLD") else None
            if tgt is not None and tgt["key"] in r["rejected"]:
                tgt = None
            if tgt is not None:
                cue_lost_since = None
                if r["seek"] == "HOLD":
                    self._event(f"Target {tgt['label']} reacquired: resuming SEEK in GUIDED")
                    self.check("Resume", True, f"target {tgt['label']}")
                    self.cue.select(tgt["key"])
                    yield from self._mode("GUIDED")
                    b.change_speed(b.SPEED_GROUND, p["speed"])
                    r["seek"], lost_note = "FOLLOW", False
                    hold, sent, det_since = None, -1e9, None
            elif r["seek"] == "FOLLOW" and cue_lost_since is None:
                cue_lost_since = self.now
            # The camera has it: frame it as the AI wants it before YES, nose on
            # it and the target ON the aim mark (below the dashed line), so that
            # PUSH's lean brings it to the centre.
            cam = self._camera_track(mission, t, p) if r["seek"] == "FOLLOW" else None
            if tgt is not None and pos is not None:
                r["cue"], lost_note = tgt, False
                alt = tgt["alt_rel"] - p["below"] if tgt.get("alt_rel") is not None else p["cruise_alt"]
                alt = max(p["min_alt"], min(alt, r["lim"]["alt_limit"] - 2.0))
                d, brg = _dist(pos, (tgt["lat"], tgt["lon"])), _bearing(pos, (tgt["lat"], tgt["lon"]))
                # A moving target: keep the standoff, also while asking.
                hold = None
                point = _toward(pos, brg, d - p["standoff"]) if d > p["standoff"] else pos
                if cam is not None:
                    alt, brg = cam["alt"], cam["yaw"]
                r["target_alt"] = alt
                if self.now - sent >= 0.25:
                    sent = self.now
                    b.guided_goto(point[0], point[1], alt, yaw_deg=brg)
            elif pos is not None and cam is not None:
                # Seen by the camera, no fresh cue: hold here and keep it framed.
                hold = hold or pos
                r["target_alt"] = cam["alt"]
                if self.now - sent >= 0.25:
                    sent = self.now
                    b.guided_goto(hold[0], hold[1], cam["alt"], yaw_deg=cam["yaw"])
            elif pos is not None and r["seek"] != "HOLD":
                # No cue (lost, rejected or waiting): hold here in GUIDED briefly.
                if hold is None:
                    hold = pos
                if r["seek"] == "FOLLOW" and not lost_note:
                    lost_note = True
                    self._event("ASTERIX cue lost: ALT_HOLD in 10 s if it doesn't return")
                if self.now - sent >= self.RESEND_S:
                    sent = self.now
                    b.guided_goto(hold[0], hold[1], r["target_alt"] or p["cruise_alt"])
                # Target lost > 10 s: switch to ALT_HOLD and hold altitude in place
                if (cue_lost_since is not None and r["seek"] == "FOLLOW"
                        and self.now - cue_lost_since >= 10.0):
                    self._event("ASTERIX target lost > 10 s: ALT_HOLD, holding altitude")
                    yield from self._mode("ALT_HOLD")
                    r["state"], r["why"] = "HOLD", "target lost — holding altitude"
                    r["seek"] = "HOLD"
                    self.check("Hold", None, "ALT_HOLD, waiting for pilot or new target")
            det_t, conf = self.ctx.get("det_t"), self.ctx.get("det_conf")
            seen = det_t is not None and self.now - det_t < 0.3 and conf is not None and conf >= mission.min_conf
            det_since = (det_since or self.now) if seen and r["seek"] == "FOLLOW" else None
            if det_since is not None and self.now - det_since >= p["det_s"]:
                if r["offer"] is None or r["offer"]["kind"] != "engage":
                    if r["offer"] is not None:
                        r["offer"], r["answer"] = None, None
                    cue_label = (r["cue"] or {}).get("label", "target")
                    self._offer("engage", r["cue"],
                                f"Camera locked on {cue_label}: ENGAGE the AI?")
                    det_since = None
            yield
        self._at(5)
        cue = r["cue"] or {}
        self._event(f"TARGET IN THE CAMERA (conf {conf:.2f}), {cue.get('label')} {cue.get('dist_m')} m from the cue: "
                    "ALT_HOLD, the AI takes it")
        r["expect_disarm"] = True
        yield from self._mode("ALT_HOLD")
        self.check("Handover to the AI", True, f"ALT_HOLD; target {cue.get('label', '?')} "
                   f"{cue.get('dist_m')} m away by ASTERIX")
        r["handover"] = True
        self.handover = self.now
        r["state"], r["why"] = "AI", "the AI has the target"
        while self.b.armed:
            yield
        self._landed("Landed", at_home=False)

    @property
    def blocking(self):
        """A run that flies the drone (the AI's own flight after a SEEK handover is not)."""
        r = self.run
        return r is not None and r.get("state") not in ("AI", None)

    CAM_GAIN = 0.12          # m/s of climb per degree the target sits above the aim mark
    CAM_CLIMB_MAX = 2.5
    CAM_DEADBAND_DEG = 1.0
    CAM_HOLD_S = 1.0         # a short gap in the detections keeps the last framing

    def _aim_el(self, mission):
        """Where the AI wants the target before PUSH: the aim mark under the
        dashed line, -min(PUSH lean, 20) degrees (the phone draws the same)."""
        aim = -min(float(mission.push_deg()), 20.0)
        law = getattr(self.c, "law", None)
        vfov = math.degrees(law.v_fov) if law is not None and getattr(law, "v_fov", None) else 48.8
        return max(aim, -(vfov/2 - 4.4))

    def _camera_track(self, mission, t, p):
        """Yaw and height that keep the camera's target centred left-right and on
        the aim mark; None without a fresh detection."""
        r = self.run
        cmd, det_t, conf = self.ctx.get("ai_cmd"), self.ctx.get("det_t"), self.ctx.get("det_conf")
        fresh = (cmd is not None and det_t is not None and self.now - det_t < 0.5
                 and conf is not None and conf >= mission.min_conf)
        prev = r.get("cam")
        if not fresh:
            if prev is not None and self.now - prev["t"] < self.CAM_HOLD_S:
                return prev
            r["cam"] = None
            return None
        hdg, h = t.get("heading"), t.get("rel_alt")
        if hdg is None or h is None:
            return None
        az, el = float(cmd.get("az_deg", 0.0)), float(cmd.get("el_cam_deg", 0.0))
        aim = self._aim_el(mission)
        err = el - aim
        dt = 0.05 if prev is None else max(0.0, min(0.5, self.now - prev["t"]))
        base = prev["alt"] if prev is not None else (r["target_alt"] or h)
        rate = 0.0 if abs(err) < self.CAM_DEADBAND_DEG else max(-self.CAM_CLIMB_MAX, min(self.CAM_CLIMB_MAX, self.CAM_GAIN*err))
        alt = max(p["min_alt"], min(base + rate*dt, r["lim"]["alt_limit"] - 2.0))
        r["cam"] = dict(t=self.now, az=round(az, 1), el=round(el, 1), aim=round(aim, 1), err=round(err, 1),
                        yaw=(hdg + az) % 360.0, alt=alt)
        return r["cam"]

    def take_handover(self):
        """The time of a SEEK handover the Mission has not started the AI for yet."""
        h, self.handover = self.handover, None
        return h

    # ---- results ---------------------------------------------------------------

    def check(self, name, ok, detail=""):
        r = self.run
        r["checks"].append(dict(name=name, ok=ok, detail=detail))
        self.log(f"[SEEKER] {'PASS' if ok else 'N/A' if ok is None else 'FAIL'} {name}: {detail}")
        self._row("check", text=f"{name}: {detail}", result="PASS" if ok else "N/A" if ok is None else "FAIL")

    def _event(self, text):
        r = self.run
        if r is None:
            return
        r["events"].append((round(self.now - r["t0"], 2), text))
        del r["events"][:-40]
        self.log(f"[SEEKER] {r['id']} {text}")
        self._row("event", text=text)

    def _row(self, kind, **kw):
        r = self.run
        b, t = self.b, r["tel"]
        pos, home = t.get("position"), r["home"]
        hm = None if pos is None or home is None else _dist(home, pos)
        st = b.link_stats() if b is not None else {}
        inside, edge = r["fence"]
        vf = getattr(self.c, "frame_t", None)
        r["log"].row(t_s=self.now - r["t0"], kind=kind, mission=r["id"],
                     step=r["steps"][r["step_i"]] if r["step_i"] < len(r["steps"]) else "",
                     mode=b.mode_name if b else None, armed=int(bool(b and b.armed)),
                     rel_alt_m=t.get("rel_alt"), target_alt_m=r["target_alt"], gs_mps=t.get("gs"),
                     vz_mps=None if t.get("vd") is None else -t["vd"],
                     lat=None if pos is None else f"{pos[0]:.7f}", lon=None if pos is None else f"{pos[1]:.7f}",
                     home_m=hm, slant_m=None if hm is None or t.get("rel_alt") is None else math.hypot(hm, t["rel_alt"]),
                     battery_v=t.get("battery_v"), battery_pct=t.get("battery_pct"), fc_hb_age_s=t.get("hb_age"),
                     rc_age_s=t.get("rc_age"), video_age_s=None if vf is None else self.now - vf,
                     mav_lost=st.get("lost"), mav_rx=st.get("received"), rejected=st.get("rejected"),
                     fence_edge_m=edge, fence_inside=inside, tilt_deg=t.get("tilt_deg"), **kw)

    def _fc_text(self):
        b = self.b
        recent = [x for t, x in getattr(b, "fc_text", [])[-6:] if self.run and t >= self.run["t0"] - 1]
        return ("FC: " + " | ".join(recent[-3:])) if recent else ""

    def _finish(self):
        r = self.run
        b = self.b
        if b is not None and b.gcs_muted:
            b.gcs_muted = False
        if r["id"] == "M6":
            br = r["breach"]
            if br is None or br["kind"] != "fence":
                if not any(c["ok"] is False for c in r["checks"]):
                    self.check("Geofence stop", False, "no geofence stop")
            else:
                worst = br["worst"]
                self.check("Geofence: stopped at the edge", worst >= -FENCE_BREACH_M - 0.5,
                           f"{'stayed ' + format(worst, '.1f') + ' m inside' if worst >= 0 else format(-worst, '.1f') + ' m past the edge'}")
                self.check("Geofence: turned back", bool(br.get("turned_back")), "flew back towards home"
                           if br.get("turned_back") else "did not turn back")
                sp = br.get("stop_pos")
                self.check("Geofence: final position", sp is not None,
                           f"{sp[0]:.6f}, {sp[1]:.6f}" if sp else "no GPS position")
                a = self.alert or {}
                self.check("Geofence: alert shown", a.get("shown_t") is not None,
                           f"on the phone {a['shown_t'] - a['t']:.1f} s after the stop" if a.get("shown_t") is not None
                           else "the phone did not show it")
        info = r["abort_info"]
        if info is not None and r["id"] != "M5":
            self.check(f"ABORT reaction ({info['why']})", info["mode_s"] is not None,
                       f"{info['action']}: FC in {info['mode']} after {info['mode_s']} s; "
                       f"vertical stop after {info['stop_s']} s; height {info['alt']} -> {info['end_alt']} m")
        if r["state"] == "PILOT":
            self.check("Finished by", None, "the pilot took over")
        fails = [c for c in r["checks"] if c["ok"] is False]
        verdict = "PASS" if not fails and not (r["aborted"] and r["id"] not in ("M5", "M6")) else "FAIL"
        if r["state"] == "PILOT" and not fails:
            verdict = "INCOMPLETE"
        dur = self.now - r["t0"]
        self._row("result", text=f"{verdict}: {len(r['checks']) - len(fails)}/{len(r['checks'])} checks",
                  result=verdict)
        r["log"].row(kind="metrics", mission=r["id"], text=json.dumps(r["metrics"]))
        r["log"].close()
        self.log(f"[SEEKER] {r['id']} {verdict} ({dur:.0f} s) {r['log'].path}")
        self.result = dict(id=r["id"], name=r["name"], verdict=verdict, why=r["why"],
                           checks=r["checks"], metrics=r["metrics"], csv=r["log"].path,
                           duration_s=round(dur, 1), events=r["events"][-12:])
        if r["id"] == "SEEK" and r.get("handover"):
            # The AI had it: no test report for the operator.
            self.result = None
        self.run = None

    # ---- snapshot for the phone -----------------------------------------------

    def snapshot(self, now):
        with self._lock:
            pre = list(self.pre)
            viewing = self.viewing
        r = self.run
        run = None
        if r is not None:
            t, b = r["tel"], self.b
            pos, home = t.get("position"), r["home"]
            vf = getattr(self.c, "frame_t", None)
            run = dict(id=r["id"], name=r["name"], state=r["state"], why=r["why"], steps=r["steps"],
                       step_i=r["step_i"], step_s=round(now - r["step_t"], 1), elapsed_s=round(now - r["t0"], 1),
                       wait_next=bool(r["wait_next"]), abort=r["abort"], wait_restart=bool(r["wait_restart"]),
                       can_after=bool(r["state"] == "HOVER" and b is not None and b.armed and not r["hands_off"]),
                       mode=None if b is None else b.mode_name, armed=bool(b and b.armed),
                       alt=None if t.get("rel_alt") is None else round(t["rel_alt"], 1),
                       target_alt=None if r["target_alt"] is None else round(r["target_alt"], 1),
                       gs=None if t.get("gs") is None else round(t["gs"], 1),
                       vz=None if t.get("vd") is None else round(-t["vd"], 1),
                       home_m=None if pos is None or home is None else round(_dist(home, pos)),
                       battery_v=t.get("battery_v"), battery_pct=t.get("battery_pct"),
                       fc_age=None if t.get("hb_age") is None else round(t["hb_age"], 1),
                       rc_ok=t.get("rc_age") is not None and t["rc_age"] < 1.0,
                       video_age=None if vf is None else round(now - vf, 1),
                       muted=bool(b and b.gcs_muted), cue=r.get("cue"), offer=r.get("offer"), seek=r.get("seek"),
                       cam=None if not r.get("cam") else {k: r["cam"][k] for k in ("az", "el", "aim", "err")},
                       # In the air with the throttle stick still down: a takeover would drop it.
                       throttle_low=bool(b and b.armed and (t.get("rel_alt") or 0) > 2.0
                                         and (self._throttle() if self.b is not None else 1) is not None
                                         and self._throttle() < 0.35),
                       checks=r["checks"][-6:], events=r["events"][-6:], csv=r["log"].path,
                       home=None if home is None else list(home), route=r["route"], trail=r["trail"][-300:],
                       line=r["line"], fence_alert=bool(self.alert and self.alert["kind"] == "FENCE"),
                       side=r.get("side", [])[-300:], profile=profile(r["id"], r["p"], r["lim"]),
                       summary=summary(r["id"], r["p"], r["lim"]))
        missions = [dict(id=m["id"], name=m["name"], short=m["short"], ground=m["id"] in GROUND,
                         fence=m["id"] == "M6", summary=summary(m["id"], self.params[m["id"]], self.lim),
                         profile=profile(m["id"], self.params[m["id"]], self.lim),
                         steps=describe(m["id"], self.params[m["id"]], self.lim),
                         params=[dict(x, v=self.params[m["id"]][x["k"]]) for x in m["params"]])
                    for m in CATALOG]
        if viewing in BY_ID and self.b is not None and r is None:
            # The planned track from where the drone is, for the mission screen's map.
            for x in missions:
                if x["id"] == viewing:
                    x["route"] = self.route(viewing, self.b.position())
        blocking = [c for c in pre if c["blocking"] and not c["ok"]]
        with self._lock:
            seek_pre = list(self.seek_pre)
        seek_bad = [c for c in seek_pre if c["blocking"] and not c["ok"]]
        t = self.b.telemetry() if self.b is not None else {}
        pos = t.get("position")
        tracks = self.cue.tracks(now, pos, self._home_amsl(t)) if self.b is not None else self.cue.tracks(now)
        tgt = self.cue.target(now, pos, self._home_amsl(t))
        seek = dict(params=[dict(x, v=self.params["SEEK"][x["k"]]) for x in SEEK["params"]],
                    steps=describe("SEEK", self.params["SEEK"], self.lim), checks=seek_pre,
                    ready=bool(seek_pre) and not seek_bad,
                    why="; ".join(f"{c['name']}: {c['detail']}" for c in seek_bad))
        return dict(missions=missions, limits=[dict(x, v=self.lim[x["k"]]) for x in LIMITS],
                    cue=dict(self.cue.status(now), tracks=tracks[:12], target=tgt), seek=seek,
                    viewing=viewing, checks=pre, ready=bool(pre) and not blocking,
                    ready_why="; ".join(f"{c['name']}: {c['detail']}" for c in blocking),
                    sim=bool(self.sim), run=run, result=self.result,
                    alert=None if self.alert is None else dict(self.alert, age_s=round(now - self.alert["t"], 1)),
                    restore_pending=bool(self.restore))
