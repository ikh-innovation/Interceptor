from __future__ import annotations

import json
import math
import os
import threading
from wasp import Wasp, WaspGuidance
from idle import Idle

SPEEDS = {
    "climb":   {"SLOW": 1.0, "MEDIUM": 3.0, "FAST": 8.0},
    "descent": {"SLOW": 0.5, "MEDIUM": 1.5, "FAST": 2.5},
    "yaw":     {"SLOW": 10.0, "MEDIUM": 20.0, "FAST": 45.0},
    "step":    {"SLOW": 2.0, "MEDIUM": 5.0, "FAST": 10.0},
    "push":    {"SLOW": 20.0, "MEDIUM": 30.0, "FAST": 40.0},
}
GROUPS = ("vert", "yaw", "push", "ai")
LEVELS = ("SLOW", "MEDIUM", "FAST")
AI_LEVELS = LEVELS + ("MAX",)

HOLD_KINDS = {"up": ("throttle", +1), "down": ("throttle", -1),
              "yawl": ("yaw", -1), "yawr": ("yaw", +1)}

PHASES = ("PILOT", "HOVER", "CLIMB", "DESCEND", "SCOUT",
          "AUTO_CLIMB", "AUTO_HOLD", "AUTO_ALIGN", "PUSH", "AI", "STOP",
          "WASP_READY", "WASP_FLIGHT", "WASP_DONE", "IDLE", "IDLE_FAILED")
AUTO_PHASES = ("AUTO_CLIMB", "AUTO_HOLD", "AUTO_ALIGN")


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def smoothstep(a):
    a = 0.0 if a <= 0.0 else 1.0 if a >= 1.0 else a
    return a * a * (3.0 - 2.0 * a)


class Ramp:

    def __init__(self, duration):
        self.duration = float(duration)
        self.v0 = 0.0
        self.target = 0.0
        self.t0 = None

    def reset(self, value=0.0):
        self.v0 = self.target = float(value)
        self.t0 = None

    def value(self, now):
        if self.t0 is None or self.duration <= 0.0:
            return self.target
        w = smoothstep((now - self.t0) / self.duration)
        return self.v0 + (self.target - self.v0) * w

    def set(self, target, now, duration=None):
        target = float(target)
        if target == self.target:
            return
        self.v0 = self.value(now)
        self.target = target
        self.t0 = now
        if duration is not None:
            self.duration = float(duration)

    def done(self, now):
        return self.t0 is None or (now - self.t0) >= self.duration

    def idle(self, now):
        return self.target == 0.0 and self.done(now)


def stop_distance(v, accel, jerk, ramp_s):
    v = abs(v)
    if v <= 0.0:
        return 0.0
    a, j = max(0.1, accel), max(0.1, jerk)
    if v <= a * a / j:
        t = 2.0 * math.sqrt(v / j)
    else:
        t = v / a + a / j
    return 0.5 * v * ramp_s + 0.5 * v * t


CONF_DIR = os.path.expanduser("~/.config/wasp")
AUTO_FILE = os.path.join(CONF_DIR, "auto.json")
LEVELS_FILE = os.path.join(CONF_DIR, "levels.json")
DETECT_FILE = os.path.join(CONF_DIR, "detection.json")
IDLE_FILE = os.path.join(CONF_DIR, "idle.json")
DETECT_SAVED = {"follow": bool, "cam": int}


def _read_json(path):
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def _write_json(path, d):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def save_auto_params(m, path=AUTO_FILE):
    if not _write_json(path, m.auto_params()):
        return False
    m.auto_defaults = m.auto_params()
    return True


def load_auto_params(m, path=AUTO_FILE):
    done = {}
    for k, v in _read_json(path).items():
        if m.set_auto_param(k, v) == "":
            done[k] = v
    now = m.auto_params()
    m.auto_defaults = {k: getattr(type(m), attrs[0]) for k, attrs, _c in m.AUTO_SETTINGS}
    m.auto_defaults.update({k: now[k] for k in done})
    return done


def load_levels(m, path=LEVELS_FILE):
    done = {}
    for g, lv in _read_json(path).items():
        if m.set_level(str(g), str(lv).strip().upper()) == "":
            done[g] = lv
    m.level_defaults = {g: "MEDIUM" for g in GROUPS}
    m.level_defaults.update({g: m.level[g] for g in done})
    return done


def load_detection_defaults(path=DETECT_FILE):
    out = {}
    for k, v in _read_json(path).items():
        if k in DETECT_SAVED:
            try:
                out[k] = DETECT_SAVED[k](v)
            except (TypeError, ValueError):
                pass
    return out


def load_idle_settings(m):
    value = _read_json(m.idle_path).get("seconds", 10.0) if m.idle_path else 10.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "invalid saved Idle seconds"
    if not math.isfinite(value) or not 1 <= value <= 600:
        return "invalid saved Idle seconds"
    # Startup reads configuration without commanding or engaging the aircraft.
    m.idle_s = value
    return ""


def save_detection_default(key, on, path=DETECT_FILE):
    if key not in DETECT_SAVED:
        return "the SKY GATE always starts ON: it is not saved"
    d = _read_json(path)
    d[key] = DETECT_SAVED[key](on)
    return "" if _write_json(path, d) else "could not save on the Jetson"


def save_default(m, what, auto_path=AUTO_FILE, levels_path=LEVELS_FILE):
    kind, _, key = str(what).partition(":")
    key = key.strip().lower()
    if kind == "auto":
        cur = m.auto_params()
        keys = list(cur) if key == "*" else [key]
        if any(k not in cur for k in keys):
            return "unknown AUTO setting"
        d = _read_json(auto_path)
        d.update({k: cur[k] for k in keys})
        if not _write_json(auto_path, d):
            return "could not save on the Jetson"
        m.auto_defaults.update({k: cur[k] for k in keys})
        return ""
    if kind == "lvl":
        if key not in GROUPS:
            return "unknown setting"
        d = _read_json(levels_path)
        d[key] = m.level[key]
        if not _write_json(levels_path, d):
            return "could not save on the Jetson"
        m.level_defaults[key] = m.level[key]
        return ""
    return "unknown setting"


def phone_command(m, key, value, now):
    value = str(value)
    if key == "hold":
        kind, _, session = value.partition(":")
        return m.request_hold(kind.strip().lower(), session.strip() or "?", now)
    if key == "release":
        m.request_release(value.strip())
        return ""
    if key == "step":
        return m.request_step(-1 if value.strip().lower().startswith("l") else 1)
    if key == "engage":
        return m.request_engage()
    if key == "lock":
        v = value.strip().lower()
        if v in ("on", "off", "1", "0", "true", "false"):
            return m.request_lock(v in ("1", "on", "true"))
        try:
            return m.request_lock(True, float(v))
        except ValueError:
            return "bad LOCK value"
    if key == "auto":
        return m.request_auto(value.strip().lower() in ("1", "on", "true"))
    if key == "autop":
        k, _, v = value.partition(":")
        return m.set_auto_param(k.strip().lower(), v.strip())
    if key == "autoreset":
        return m.reset_auto_params()
    if key == "autoen":
        v = value.strip().lower()
        if v not in ("on", "off", "1", "0", "true", "false"):
            return "bad AUTO value"
        return m.set_auto_enabled(v in ("1", "on", "true"))
    if key == "waspen":
        v = value.strip().lower()
        if v not in ("on", "off", "1", "0", "true", "false"):
            return "bad WASP value"
        return m.set_wasp_enabled(v in ("1", "on", "true"))
    if key == "flightmode":
        v = value.strip().lower()
        if v not in ("normal", "wasp"):
            return "flight mode: normal or wasp"
        return m.set_wasp_enabled(v == "wasp", selection=True)
    if key == "mount":
        return m.confirm_mount(value)
    if key == "waspp":
        k, _, v = value.partition(":")
        return m.set_wasp_param(k.strip(), v.strip())
    if key == "idles":
        return m.set_idle_seconds(value)
    if key == "abort":
        return m.request_abort()
    if key == "lvl":
        group, _, level = value.partition(":")
        return m.set_level(group.strip().lower(), level.strip().upper())
    raise ValueError("unknown command")


class Mission:
    DEADMAN_S = 0.30
    THR_RAMP_S = 0.50
    YAW_RAMP_S = 0.30
    PUSH_RAMP_S = 1.20
    PUSH_MIN_S = 1.50
    PUSH_MAX_S = 6.00
    BLEND_S = 0.40
    STOP_S = 1.00
    STEP_TIMEOUT_S = 3.00
    STEP_KP = 2.5
    STEP_ACCEL = 60.0
    LOCK_KP = 2.5
    LOCK_DB_ON = 1.0
    LOCK_DB_OFF = 0.4
    LOCK_TAU = 0.20
    LOCK_SETTLED_DPS = 4.0
    LOCK_SETTLE_S = 1.0
    LOCK_MAX_DPS = 15.0
    LOCK_ACCEL = 30.0
    TARGET_FRESH_S = 0.50
    YAW_FADE_S = 0.15
    REACQ_S = 2.50
    TERMINAL_GRACE_S = 1.20
    DET_FRESH_S = 0.25
    AUTO_HEIGHT_M = 15.0
    AUTO_CLIMB_MPS = 3.0
    AUTO_HEIGHT_KP = 1.0
    AUTO_THR_ACCEL = 3.0
    AUTO_YAW_KP = 2.0
    AUTO_YAW_MAX = 30.0
    AUTO_YAW_ACCEL = 60.0
    AUTO_EL_KP = 0.6
    AUTO_ALIGN_CLIMB_MPS = 5.0
    AUTO_DESCENT_MPS = 1.5
    AUTO_FLOOR_M = 3.0
    AUTO_AZ_OK = 5.0
    AUTO_EL_OK = 5.0
    AUTO_EL_LOW = 12.0
    AUTO_READY_S = 0.25
    AUTO_GO_TTG_S = 3.0
    AUTO_GO_HOLD_S = 0.5
    AUTO_ALIGN_MAX_S = 2.0
    AUTO_GO_AZ = 15.0
    AUTO_STEER_S = 0.6
    AUTO_LINE_DEG = 30.0
    AUTO_AI = "MAX"

    AUTO_SETTINGS = (
        ("height",   ("AUTO_HEIGHT_M",),               (5.0, 10.0, 15.0, 20.0)),
        ("climb",    ("AUTO_CLIMB_MPS",),              (2.0, 3.0, 5.0)),
        ("turn",     ("AUTO_YAW_MAX",),                (20.0, 30.0, 45.0)),
        ("adjust",   ("AUTO_ALIGN_CLIMB_MPS",),        (3.0, 5.0, 8.0)),
        ("window",   ("AUTO_AZ_OK", "AUTO_EL_OK"),     (3.0, 5.0, 8.0)),
        ("ready",    ("AUTO_READY_S",),                (0.25, 0.5, 1.0)),
        ("giveup",   ("AUTO_ALIGN_MAX_S",),            (1.0, 2.0, 3.0, 5.0)),
        ("closing",  ("AUTO_GO_TTG_S",),               (0.0, 2.0, 3.0, 4.0)),
        ("floor",    ("AUTO_FLOOR_M",),                (2.0, 3.0, 5.0)),
        ("line",     ("AUTO_LINE_DEG",),               (0.0, 20.0, 30.0, 40.0)),
        ("ai",       ("AUTO_AI",),                     ("SAME", "MEDIUM", "FAST", "MAX")),
        ("steer",    ("AUTO_STEER_S",),                (0.0, 0.3, 0.6)),
    )

    def __init__(self, mapper, min_conf=0.20, ceiling_m=30.0,
                 accel_z=5.0, jerk_z=5.0, log=print, idle_path=None):
        self.mapper = mapper
        self.min_conf = float(min_conf)
        self.ceiling_m = float(ceiling_m)
        self.accel_z = float(accel_z)
        self.jerk_z = float(jerk_z)
        self.log = log

        self._lock = threading.Lock()
        self._hold = None
        self._dead_sessions = set()
        self._released = set()
        self._req_step = None
        self._req_lock = None
        self.lock_hdg = None
        self._lock_away = False
        self._lock_err = None
        self._lock_act = False
        self._lock_prev = None
        self._lock_rate = 0.0
        self._lock_still = 0.0
        self._req_engage = False
        self._req_abort = False
        self._req_auto = None
        self.auto_enabled = False
        self.wasp = Wasp()
        self.idle_s = 10.0
        self.idle_path = idle_path
        self.idle = Idle()
        self.idle_supported = callable(getattr(mapper, "idle_pwm", None))
        self._run_t = None
        self.camera_mount_deg = float(getattr(mapper, "camera_mount_deg", 30.0))
        for _k, attrs, _c in self.AUTO_SETTINGS:
            for a in attrs:
                setattr(self, a, getattr(type(self), a))
        self.ai_override = None
        self._auto_ready_t = None
        self._auto_align_t = 0.0
        self._auto_close_t = None
        self._auto_turning = False
        self.auto_run = False
        self._steer_after = 0.0
        self.level = {g: "MEDIUM" for g in GROUPS}
        self.level_defaults = dict(self.level)
        self.auto_defaults = self.auto_params()

        self.phase = "PILOT"
        self.phase_t = 0.0
        self.event = ""
        self.event_t = 0.0
        self.refusal = ""
        self._thr = Ramp(self.THR_RAMP_S)
        self._yaw = Ramp(self.YAW_RAMP_S)
        self._thr_on = False
        self._yaw_on = False
        self._step = None
        self._push = None
        self._push_t = None
        self._stop_from = None
        self._stop_t = None
        self._last_out = {}
        self._terminal = None
        self._src = {}
        self._mix = {}
        self._push_steer = False
        self._steer_t = None
        self._t_last = None
        self._height = self._climb = None
        self.last_handover = None
        self.engaged = False

    def request_hold(self, kind, session, now):
        if kind not in HOLD_KINDS:
            return "unknown hold"
        with self._lock:
            if self.wasp.enabled:
                return "phone moves are disabled in WASP: aim the platform manually"
            if session in self._dead_sessions:
                return "hold ended by ABORT: lift the finger and press again"
            if session in self._released:
                return "hold already released"
            if self.phase in ("AUTO_CLIMB", "AUTO_HOLD") and HOLD_KINDS[kind][0] == "yaw":
                pass
            elif self.phase in ("PUSH", "AI", "STOP", "IDLE", "IDLE_FAILED") + AUTO_PHASES:
                return f"no manual moves during {self.phase}"
            if self.phase == "PILOT":
                return "the pilot has it (not armed / not ALT_HOLD)"
            self._hold = (kind, session, now)
        return ""

    def request_release(self, session):
        with self._lock:
            self._released.add(session)
            if len(self._released) > 64:
                self._released = set(list(self._released)[-32:])
            if self._hold and self._hold[1] == session:
                self._hold = None

    def request_step(self, direction):
        with self._lock:
            if self.phase not in ("HOVER", "SCOUT", "AUTO_CLIMB", "AUTO_HOLD"):
                return f"no yaw step during {self.phase}"
            self._req_step = -1 if direction < 0 else +1
        return ""

    def request_lock(self, on, hdg=None):
        if hdg is not None:
            try:
                hdg = float(hdg) % 360.0
            except (TypeError, ValueError):
                return "bad heading"
            if hdg != hdg:
                return "bad heading"
        with self._lock:
            if on and self.phase not in ("HOVER", "CLIMB", "DESCEND", "SCOUT"):
                return f"LOCK only while hovering, not during {self.phase}"
            self._req_lock = (bool(on), hdg if on else None)
        return ""

    def request_engage(self):
        with self._lock:
            if self.wasp.enabled:
                if self.phase != "WASP_READY" or self.wasp.used:
                    return "WASP launch unavailable: check readiness; disarm after each attempt"
                self._req_engage = True
                return ""
            if self.engaged:
                return "already engaged: pilot STAB -> ALT_HOLD for a new try"
            if self.phase not in ("HOVER", "SCOUT") + AUTO_PHASES:
                return f"ENGAGE only from HOVER, SCOUT or AUTO, not {self.phase}"
            self._req_engage = True
        return ""

    def set_auto_enabled(self, on):
        with self._lock:
            if on and self.wasp.enabled:
                return "turn WASP off before enabling AUTO"
            self.auto_enabled = bool(on)
            if not on and self.phase in AUTO_PHASES:
                self._req_auto = "off"
        return ""

    def _configuration_error(self):
        if getattr(self.mapper, "armed", False):
            return "disarm before changing flight parameters or the camera mount"
        if self.phase not in ("PILOT", "HOVER", "WASP_READY", "WASP_DONE"):
            return "finish the run before changing flight parameters or the camera mount"
        return ""

    def set_wasp_enabled(self, on, selection=False):
        with self._lock:
            # Reopening the UI in its current mode must never reset a flight.
            if selection and bool(on) == self.wasp.enabled:
                return ""
            error = self._configuration_error()
            if error:
                return error
            if on and self.auto_enabled and not selection:
                return "turn AUTO off before enabling WASP"
            if on and not callable(getattr(self.mapper, "wasp_pwm", None)):
                return "WASP requires the MAVLink controller (not the hover-only desk simulator)"
            if selection and on:
                self.auto_enabled = False
            self.wasp.enabled = bool(on)
            self.wasp.camera_confirmed = False
            self.wasp.started = None
            self.wasp.used = False
            self._req_engage = False
            self._req_auto = None
            self._hold = None
            self._reset_motion()
            self.phase = "PILOT"
        return ""

    def confirm_mount(self, value):
        try:
            value = float(value)
        except (ValueError, TypeError):
            return "camera mount: 30 or 90 degrees"
        if value not in (30.0, 90.0):
            return "camera mount: 30 or 90 degrees"
        with self._lock:
            error = self._configuration_error()
            if error:
                return error
            if self.wasp.enabled and value != 90.0:
                return "WASP requires the fixed 90 degree camera"
            self.camera_mount_deg = value
            self.wasp.camera_confirmed = self.wasp.enabled and value == 90.0
        return ""

    def set_wasp_param(self, key, value):
        try:
            v = float(value)
        except (ValueError, TypeError):
            return "bad WASP value"
        if key not in Wasp.SETTINGS or v not in Wasp.SETTINGS[key]:
            return "unknown WASP setting or value"
        with self._lock:
            error = self._configuration_error()
            if error:
                return error
            self.wasp.params[key] = v
            if key == "camera_roll":
                self.wasp.camera_confirmed = False
        return ""

    def set_idle_seconds(self, value, save=True):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "Idle: enter 1 to 600 seconds"
        if not math.isfinite(value) or not 1 <= value <= 600:
            return "Idle: enter 1 to 600 seconds"
        with self._lock:
            error = self._configuration_error()
            if error:
                return error
            if save and self.idle_path and not _write_json(self.idle_path, {"seconds": value}):
                return "could not save Idle on the Jetson"
            self.idle_s = value
        return ""

    def auto_params(self):
        return {k: getattr(self, attrs[0]) for k, attrs, _c in self.AUTO_SETTINGS}

    def set_auto_param(self, key, value):
        spec = next((x for x in self.AUTO_SETTINGS if x[0] == key), None)
        if spec is None:
            return "unknown AUTO setting"
        _k, attrs, choices = spec
        if isinstance(choices[0], str):
            v = str(value).strip().upper()
            ok = v in choices
        else:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return "bad AUTO value"
            ok = any(abs(v - c) < 1e-6 for c in choices)
        if not ok:
            return f"AUTO {key}: one of {', '.join(str(c) for c in choices)}"
        with self._lock:
            if self.phase in AUTO_PHASES or (self.auto_run and self.phase in ("PUSH", "AI", "STOP")):
                return "AUTO is flying: change its settings after the run"
            for a in attrs:
                setattr(self, a, v)
        return ""

    def reset_auto_params(self):
        with self._lock:
            if self.phase in AUTO_PHASES or (self.auto_run and self.phase in ("PUSH", "AI", "STOP")):
                return "AUTO is flying: change its settings after the run"
            for _k, attrs, _c in self.AUTO_SETTINGS:
                for a in attrs:
                    setattr(self, a, getattr(type(self), a))
        return ""

    def ai_level_now(self):
        return self.ai_override or self.level["ai"]

    def line_deg(self):
        return self.AUTO_LINE_DEG if (self.auto_run and self.AUTO_LINE_DEG) else self.push_deg()

    def request_auto(self, on):
        with self._lock:
            if on and self.wasp.enabled:
                return "AUTO and WASP are mutually exclusive"
            if not on:
                if self.phase in AUTO_PHASES:
                    self._req_auto = "off"
                return ""
            if self.phase in AUTO_PHASES:
                return ""
            if not self.auto_enabled:
                return "AUTO is OFF (☰ -> PARAMETERS -> AUTO)"
            if self.engaged:
                return "already engaged: pilot STAB -> ALT_HOLD for a new try"
            if self.phase not in ("HOVER", "CLIMB", "DESCEND", "SCOUT"):
                return f"AUTO only while hovering, not during {self.phase}"
            self._req_auto = "on"
        return ""

    def request_abort(self):
        with self._lock:
            self._req_abort = True
            if self._hold:
                self._dead_sessions.add(self._hold[1])
                self._hold = None
            if len(self._dead_sessions) > 64:
                self._dead_sessions = set(list(self._dead_sessions)[-32:])
        return ""

    def set_level(self, group, level):
        if group not in GROUPS or level not in (AI_LEVELS if group == "ai" else LEVELS):
            return "unknown speed"
        with self._lock:
            self.level[group] = level
        return ""

    def climb_rate(self):
        return SPEEDS["climb"][self.level["vert"]]

    def descent_rate(self):
        return SPEEDS["descent"][self.level["vert"]]

    def yaw_rate(self):
        return SPEEDS["yaw"][self.level["yaw"]]

    def step_deg(self):
        return SPEEDS["step"][self.level["yaw"]]

    def push_deg(self):
        return SPEEDS["push"][self.level["push"]]

    def _set_phase(self, phase, now, why=""):
        if phase == self.phase:
            return
        self.log(f"[MISSION] {self.phase} -> {phase}" + (f"  ({why})" if why else ""))
        self.phase = phase
        self.phase_t = now

    def _note(self, text, now):
        self.event, self.event_t = text, now
        self.log(f"[MISSION] {text}")

    def _reset_motion(self):
        self._run_t = None
        self._unlock()
        self._thr.reset(0.0)
        self._yaw.reset(0.0)
        self._thr_on = self._yaw_on = False
        self._step = None
        self._push = None
        self._push_t = None
        self._stop_from = self._stop_t = None
        self._terminal = None
        self._src = {}
        self._mix = {}
        self._push_steer = False
        self._steer_t = None
        self._steer_after = 0.0
        self._auto_ready_t = None
        self._auto_turning = False
        self.auto_run = False
        self.ai_override = None
        self._line_deg = None
        self._last_out = {}

    def _unlock(self, why=None, now=None):
        if getattr(self, "lock_hdg", None) is not None and why:
            self._note(f"LOCK off ({why})", now if now is not None else 0.0)
        self.lock_hdg = None
        self._lock_away = False
        self._lock_err = None
        self._lock_act = False
        self._lock_prev = None
        self._lock_rate = 0.0
        self._lock_still = 0.0

    def _pilot_value(self, axis):
        pwm = self.mapper.pilot_pwm(axis)
        return 0.0 if pwm is None else self.mapper.pwm_to_phys(axis, pwm)

    def cycle(self, now, ctx):
        self._dt = 0.02 if self._t_last is None else max(0.0, now - self._t_last)
        self._t_last = now
        self._height, self._climb = ctx.get("height"), ctx.get("climb")
        with self._lock:
            hold = self._hold
            if hold and (now - hold[2]) > self.DEADMAN_S:
                self._hold = hold = None
            step, self._req_step = self._req_step, None
            engage, self._req_engage = self._req_engage, False
            abort, self._req_abort = self._req_abort, False
            lock_req, self._req_lock = self._req_lock, None
            auto_req, self._req_auto = self._req_auto, None

        if self.idle.active:
            return self._idle_cycle(now, ctx)
        if self.phase == "IDLE_FAILED":
            if ctx.get("gate_ok") and getattr(self.mapper, "armed", False):
                return {}
            self._set_phase("PILOT", now)
        # ABORT can recover an active flight or an airborne aircraft; it must
        # not turn an armed, unlaunched Wasp into an automatic takeoff.
        if (abort and ctx.get("gate_ok") and self.idle_supported
                and (self._run_t is not None or ctx.get("airborne"))):
            return self._start_idle(now, ctx, "ABORT")
        if (self.idle_supported and self._run_t is not None
                and now-self._run_t >= self.idle_s and ctx.get("gate_ok")):
            return self._start_idle(now, ctx, "Idle time limit")

        if self.wasp.enabled:
            return self._wasp_cycle(now, ctx, engage, abort)

        if not ctx.get("gate_ok"):
            if self.phase != "PILOT":
                self._note(f"PILOT has it: {ctx.get('gate_why') or 'gate'}", now)
            self._unlock("the pilot has it", now)
            if self.engaged:
                self.engaged = False
                self._note("reset: ENGAGE ready again once back in ALT_HOLD", now)
            self._set_phase("PILOT", now, ctx.get("gate_why", ""))
            self._reset_motion()
            with self._lock:
                self._hold = None
            return {}
        if self.phase == "PILOT":
            self._set_phase("HOVER", now, "armed in ALT_HOLD")
            self._reset_motion()

        if abort:
            self._note(f"ABORT from {self.phase}: every channel to the pilot", now)
            self._unlock("ABORT", now)
            self._set_phase("HOVER", now, "ABORT")
            self._reset_motion()
            return {}

        if self.phase in ("HOVER", "CLIMB", "DESCEND", "SCOUT") + AUTO_PHASES:
            if engage:
                self._unlock("ENGAGE: the AI takes the yaw", now)
                return self._start_push(now, ctx)
        if self.phase in AUTO_PHASES:
            if auto_req == "off":
                if self.idle_supported:
                    return self._start_idle(now, ctx, "AUTO off")
                self._unlock()
                self._note("AUTO off: soft stop, then the pilot's sticks", now)
                return self._start_stop(now)
            return self._auto_cycle(now, ctx, hold, step)
        if self.phase in ("HOVER", "CLIMB", "DESCEND", "SCOUT"):
            if auto_req == "on":
                out = self._start_auto(now, ctx)
                if out is not None:
                    return out
            if lock_req is not None:
                self._apply_lock(lock_req, ctx, now)
            return self._manual(now, ctx, hold, step)
        if self.phase == "PUSH":
            return self._push_cycle(now, ctx)
        if self.phase == "AI":
            return self._ai_cycle(now, ctx)
        if self.phase == "STOP":
            return self._stop_cycle(now)
        return {}

    def _wasp_cycle(self, now, ctx, engage, abort):
        w = self.wasp
        if not getattr(self.mapper, "armed", False):
            w.used = False
        if abort or not ctx.get("gate_ok"):
            if w.started is not None:
                w.used = True
            w.started = None
            # The pilot took it back: the Idle clock starts again from zero.
            self._run_t = None
            self._set_phase("WASP_DONE" if w.used else "PILOT", now)
            if abort:
                self._note("WASP ABORT: every channel returned to the pilot", now)
            return {}
        if w.used and w.started is None:
            self._set_phase("WASP_DONE", now)
            return {}
        if w.started is None:
            self._set_phase("WASP_READY", now)
            if not engage:
                return {}
            why = ""
            if ctx.get("att") is None:
                why = "no fresh attitude"
            if not w.camera_confirmed or self.camera_mount_deg != 90.0:
                why = "confirm the fixed 90 degree camera before launch"
            if why:
                self._refuse(why, now)
                return {}
            w.begin(now, target_acquired=False)
            self._run_t = now
            self._note(f"WASP launched: Idle in {self.idle_s:g}s from throttle start", now)
            self._set_phase("WASP_FLIGHT", now)
        why = "no fresh attitude" if ctx.get("att") is None else ""
        if now - w.started >= self.idle_s:
            why = "Idle time limit"
        height = ctx.get("height")
        if height is not None and height >= self.ceiling_m:
            why = "ceiling reached"
        if not why:
            tracking = not w.fault(now, ctx, self.min_conf)
            if tracking:
                try:
                    # Validate every bearing, including after a detector gap.
                    if w.acquired_t is None:
                        w.acquire(now, ctx)
                        self._note("WASP target acquired: blended steering", now)
                    else:
                        WaspGuidance.rates(ctx["ai_cmd"], ctx["att"], now,
                            gain=w.params["gain"], lead_s=w.params["lead"], max_rate_dps=w.params["rate"])
                except (ValueError, KeyError, TypeError, OverflowError):
                    tracking = False
            if not tracking:
                w.lose_target()
        if not why:
            try:
                return w.output(self.mapper, now, ctx)
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                why = str(exc)
        w.started = None
        if self.idle_supported:
            return self._start_idle(now, ctx, why)
        self._note(f"WASP ended: {why}; every channel returned to the pilot", now)
        self._set_phase("WASP_DONE", now)
        return {}

    def _start_idle(self, now, ctx, reason):
        self.wasp.started = None
        if self.wasp.enabled:
            self.wasp.used = True
        self._reset_motion()
        self._unlock()
        self.engaged = True
        with self._lock:
            self._hold = None
        try:
            self.idle.begin(now, ctx, self.ceiling_m)
            if not self.mapper.idle_request_alt_hold():
                raise ValueError("could not request ALT_HOLD")
            self._note(f"IDLE: {reason}; level, yaw hold, height {self.idle.height:.1f} m; XY drift possible", now)
            self._set_phase("IDLE", now)
            return self._idle_cycle(now, ctx)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            return self._idle_failed(now, str(exc))

    def _idle_failed(self, now, why):
        self.idle.stop()
        self._note(f"IDLE unavailable: {why}; every channel returned to the pilot", now)
        self._set_phase("IDLE_FAILED", now)
        return {}

    def _idle_cycle(self, now, ctx):
        if not ctx.get("gate_ok"):
            self.idle.stop()
            self._reset_motion()
            self.engaged = False
            self._note(f"IDLE released to pilot: {ctx.get('gate_why') or 'gate'}", now)
            self._set_phase("PILOT", now)
            return {}
        try:
            return self.idle.output(self.mapper, now, ctx)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            return self._idle_failed(now, str(exc))

    def _manual(self, now, ctx, hold, step):
        kind = hold[0] if hold else None
        axis, sign = HOLD_KINDS.get(kind, (None, 0))
        height = ctx.get("height")
        climb = ctx.get("climb")

        want_thr = 0.0
        if axis == "throttle":
            if height is None:
                self._refuse("height unknown (no VFR_HUD): no climb / descent", now)
            else:
                v = self.climb_rate() if sign > 0 else -self.descent_rate()
                cur_v = climb if climb is not None else self._thr.value(now)
                margin = stop_distance(max(abs(cur_v), abs(v)), self.accel_z,
                                       self.jerk_z, self.THR_RAMP_S) + 0.3
                if sign > 0 and height + margin >= self.ceiling_m:
                    self._refuse(f"ceiling {self.ceiling_m:.0f} m", now)
                else:
                    want_thr = v
                    self.refusal = ""
        if want_thr != 0.0 and not self._thr_on:
            self._thr.reset(self._pilot_value("throttle"))
            self._thr_on = True
        self._thr.set(want_thr, now)

        att = ctx.get("att")
        yaw_deg = None if att is None else math.degrees(att.get("yaw", 0.0))
        want_yaw = 0.0
        locked = self.lock_hdg is not None
        if axis == "yaw":
            self._step = None
            want_yaw = sign * self.yaw_rate()
            if locked:
                self._lock_away = True
        elif step and yaw_deg is not None:
            self._step = (yaw_deg + step * self.step_deg(), now, step)
            if locked:
                self._lock_away = True
        elif step:
            self._refuse("no attitude: no yaw step", now)
        lock_loop = False
        if locked and axis != "yaw" and self._step is None:
            if self._lock_away:
                if self._yaw.idle(now) and yaw_deg is not None:
                    self._lock_away = False
                    self._lock_err, self._lock_act = None, False
                    self._lock_prev, self._lock_rate = None, 0.0
                    self._lock_still = self.LOCK_SETTLE_S
                    self._note(f"LOCK: back to {self.lock_hdg:.0f} deg", now)
            elif yaw_deg is None:
                self._lock_err, self._lock_act = None, False
                self._lock_prev = None
            else:
                want_yaw = self._lock_yaw(yaw_deg)
                lock_loop = True
        if self._step is not None and axis != "yaw":
            tgt, t0, _d = self._step
            err = tgt - yaw_deg if yaw_deg is not None else 0.0
            if (yaw_deg is None or abs(err) < 0.5
                    or (now - t0) > self.STEP_TIMEOUT_S):
                self._step = None
            else:
                lim = self.yaw_rate()
                want_yaw = max(-lim, min(lim, self.STEP_KP * err))
        if want_yaw != 0.0 and not self._yaw_on:
            self._yaw.reset(self._pilot_value("yaw"))
            self._yaw_on = True
        if (self._step is not None or lock_loop) and axis != "yaw":
            v = self._yaw.value(now)
            d = (self.LOCK_ACCEL if lock_loop else self.STEP_ACCEL) * self._dt
            self._yaw.reset(v + max(-d, min(d, want_yaw - v)))
        else:
            self._yaw.set(want_yaw, now)

        if self._thr_on and self._thr.idle(now) and want_thr == 0.0:
            self._thr_on = False
        if self._yaw_on and self._yaw.idle(now) and want_yaw == 0.0 \
                and self._step is None and not locked:
            self._yaw_on = False
        if locked and not self._yaw_on:
            self._yaw.reset(self._pilot_value("yaw"))
            self._yaw_on = True

        out = {}
        if self._thr_on:
            out["throttle"] = self.mapper.phys_to_pwm("throttle", self._thr.value(now))
        if self._yaw_on:
            out["yaw"] = self.mapper.phys_to_pwm("yaw", self._yaw.value(now))

        if self._thr_on:
            self._set_phase("CLIMB" if self._thr.target > 0 or
                            (self._thr.target == 0 and self._thr.v0 > 0)
                            else "DESCEND", now)
        elif self._yaw_on and (not locked or axis == "yaw"
                               or self._step is not None or self._lock_away):
            self._set_phase("SCOUT", now)
        else:
            self._set_phase("HOVER", now)
        self._last_out = out
        return out

    def _lock_yaw(self, yaw_deg):
        if self._lock_prev is not None and self._dt > 0:
            d = ((yaw_deg - self._lock_prev + 180.0) % 360.0 - 180.0) / self._dt
            self._lock_rate += (d - self._lock_rate) * (self._dt / (0.3 + self._dt))
        self._lock_prev = yaw_deg
        self._lock_still = (self._lock_still + self._dt
                            if abs(self._lock_rate) < self.LOCK_SETTLED_DPS else 0.0)
        err = (self.lock_hdg - yaw_deg + 180.0) % 360.0 - 180.0
        f = self._lock_err
        if f is None or abs(err - f) > 90.0:
            f = err
        else:
            f += (err - f) * (self._dt / (self.LOCK_TAU + self._dt))
        self._lock_err = f
        if abs(f) > self.LOCK_DB_ON and (
                self._lock_still >= self.LOCK_SETTLE_S or self._lock_act):
            self._lock_act = True
        if self._lock_act and (abs(err) < self.LOCK_DB_OFF or abs(f) < self.LOCK_DB_OFF):
            self._lock_act = False
        lim = self.LOCK_MAX_DPS
        return max(-lim, min(lim, self.LOCK_KP * err)) if self._lock_act else 0.0

    def _apply_lock(self, req, ctx, now):
        on, hdg = req
        att = ctx.get("att")
        if not on:
            self._unlock("LOCK pressed again", now)
            return
        if self.lock_hdg is not None and hdg is None:
            return
        if att is None:
            self._refuse("no attitude: cannot LOCK", now)
            return
        self._lock_err, self._lock_act = None, False
        self._lock_away = False
        self._step = None
        if hdg is None:
            self.lock_hdg = math.degrees(att.get("yaw", 0.0)) % 360.0
            self._note(f"LOCK on: heading held at {self.lock_hdg:.0f} deg", now)
        else:
            self.lock_hdg = hdg
            self._lock_prev, self._lock_rate = None, 0.0
            self._lock_still = self.LOCK_SETTLE_S
            self._note(f"LOCK on: turning to {self.lock_hdg:.0f} deg and holding it", now)

    def _refuse(self, why, now):
        if self.refusal != why or (now - self.event_t) > 2.0:
            self._note(f"refused: {why}", now)
        self.refusal = why

    AUTO_AIM_EDGE_DEG = 4.4

    def auto_aim_el(self):
        vf = getattr(self.mapper, "V_FOV", None)
        low = round((math.degrees(vf) / 2.0 if vf else 24.4) - self.AUTO_AIM_EDGE_DEG, 3)
        return max(-min(self.line_deg(), 20.0), -low)

    def _start_auto(self, now, ctx):
        att, height = ctx.get("att"), ctx.get("height")
        if att is None:
            self._refuse("no attitude: no AUTO", now)
            return None
        if height is None:
            self._refuse("height unknown (no VFR_HUD): no AUTO", now)
            return None
        if self._run_t is None:
            self._run_t = now
        hdg = (self.lock_hdg if self.lock_hdg is not None
               else math.degrees(att.get("yaw", 0.0)) % 360.0)
        self._unlock()
        self.lock_hdg = hdg
        self._step = None
        if not self._thr_on:
            self._thr.reset(self._pilot_value("throttle"))
            self._thr_on = True
        if not self._yaw_on:
            self._yaw.reset(self._pilot_value("yaw"))
            self._yaw_on = True
        self._auto_ready_t = None
        self.auto_run = True
        self.ai_override = None if self.AUTO_AI == "SAME" else self.AUTO_AI
        up = height < self.AUTO_HEIGHT_M - 0.5
        self._note(f"AUTO: {'up to %.0f m' % self.AUTO_HEIGHT_M if up else 'holding here'}, "
                   f"heading locked at {hdg:.0f} deg, watching", now)
        self._set_phase("AUTO_CLIMB" if up else "AUTO_HOLD", now, "AUTO")
        return self._auto_cycle(now, ctx)

    def _auto_watch(self, now, ctx, why):
        att, h = ctx.get("att"), ctx.get("height")
        self._unlock()
        if att is not None:
            self.lock_hdg = math.degrees(att.get("yaw", 0.0)) % 360.0
        self._auto_ready_t = None
        up = h is not None and h < self.AUTO_HEIGHT_M - 0.5
        self._note(f"AUTO: {why}; {'climbing on' if up else 'holding'}, watching", now)
        self._set_phase("AUTO_CLIMB" if up else "AUTO_HOLD", now, why)

    def _auto_cycle(self, now, ctx, hold=None, step=None):
        height, climb = ctx.get("height"), ctx.get("climb")
        att = ctx.get("att")
        yaw_deg = None if att is None else math.degrees(att.get("yaw", 0.0))
        cmd, ai_t, det_t = ctx.get("ai_cmd"), ctx.get("ai_t"), ctx.get("det_t")
        fresh = bool(cmd) and ai_t is not None and (now - ai_t) <= self.TARGET_FRESH_S
        if self.phase in ("AUTO_CLIMB", "AUTO_HOLD") and self._confirmed(now, ctx):
            self._unlock()
            self._auto_ready_t = self._auto_close_t = None
            self._auto_align_t = now
            self._note(f"AUTO: target (conf {ctx.get('det_conf') or 0:.2f}): "
                       f"stopping, putting it on the aim mark", now)
            self._set_phase("AUTO_ALIGN", now, "first detection")
        elif self.phase == "AUTO_ALIGN" and not fresh:
            if self.idle_supported:
                return self._start_idle(now, ctx, "target lost while aligning")
            self._auto_watch(now, ctx, "target lost while aligning")

        thr_v, yaw_v = self._thr.value(now), self._yaw.value(now)
        if self.phase == "AUTO_ALIGN":
            live = det_t is not None and (now - det_t) <= self.YAW_FADE_S
            az = float(cmd.get("az_deg", 0.0))
            err_el = float(cmd.get("el_cam_deg", 0.0)) - self.auto_aim_el()
            ff = float(cmd.get("los_az_dps", 0.0) or 0.0)
            want_yaw = max(-self.AUTO_YAW_MAX, min(self.AUTO_YAW_MAX,
                                                   ff + self.AUTO_YAW_KP * az)) if live else 0.0
            want_thr = (max(-self.AUTO_DESCENT_MPS, min(self.AUTO_ALIGN_CLIMB_MPS,
                                                        self.AUTO_EL_KP * err_el))
                        if live else 0.0)
            if want_thr < 0.0 and (height is None or height - stop_distance(
                    max(abs(climb or 0.0), -want_thr), self.accel_z, self.jerk_z,
                    self.THR_RAMP_S) <= self.AUTO_FLOOR_M):
                want_thr = 0.0
            if want_thr > 0.0 and height is not None and height + stop_distance(
                    max(abs(climb or 0.0), want_thr), self.accel_z, self.jerk_z,
                    self.THR_RAMP_S) + 0.3 >= self.ceiling_m:
                want_thr = 0.0
                self._refuse(f"ceiling {self.ceiling_m:.0f} m", now)
            d = self.AUTO_THR_ACCEL * self._dt
            self._thr.reset(thr_v + max(-d, min(d, want_thr - thr_v)))
            aligned = (live and abs(az) <= self.AUTO_AZ_OK
                       and -self.AUTO_EL_LOW <= err_el <= self.AUTO_EL_OK)
            if not aligned:
                self._auto_ready_t = None
            elif self._auto_ready_t is None:
                self._auto_ready_t = now
            rng, closing = cmd.get("range_m"), float(cmd.get("closing_mps") or 0.0)
            arriving = bool(live and rng and closing > 1.0
                            and rng / closing < self.AUTO_GO_TTG_S)
            if not arriving:
                self._auto_close_t = None
            elif self._auto_close_t is None:
                self._auto_close_t = now
            go = None
            if aligned and now - self._auto_ready_t >= self.AUTO_READY_S:
                go = f"on the mark {self.AUTO_READY_S:.1f} s"
            elif arriving and now - self._auto_close_t >= self.AUTO_GO_HOLD_S:
                go = f"closing {closing:.0f} m/s at {rng:.0f} m: no time to line up"
            elif (live and now - self._auto_align_t >= self.AUTO_ALIGN_MAX_S
                  and abs(az) <= self.AUTO_GO_AZ):
                go = f"{self.AUTO_ALIGN_MAX_S:.0f} s of aligning, near the nose"
            if go:
                self._note(f"AUTO: {go}: the second line, lean "
                           f"{self.line_deg():.0f} deg, then AI {self.ai_level_now()}", now)
                out = self._start_push(now, ctx)
                self._steer_after = now + self.AUTO_STEER_S
                return out
        else:
            want_thr = 0.0
            if self.phase == "AUTO_CLIMB":
                if height is None:
                    self._refuse("height unknown: AUTO holds", now)
                else:
                    top = min(self.AUTO_HEIGHT_M, self.ceiling_m - 1.0)
                    want_thr = max(0.0, min(self.AUTO_CLIMB_MPS,
                                            self.AUTO_HEIGHT_KP * (top - height)))
                    if top - height < 0.3:
                        want_thr = 0.0
                        self._note(f"AUTO: at {height:.1f} m, holding, watching", now)
                        self._set_phase("AUTO_HOLD", now, "height reached")
            if self.phase == "AUTO_CLIMB":
                d = self.AUTO_THR_ACCEL * self._dt
                self._thr.reset(thr_v + max(-d, min(d, want_thr - thr_v)))
            else:
                self._thr.set(want_thr, now)
            axis, sign = HOLD_KINDS.get(hold[0] if hold else None, (None, 0))
            if axis == "yaw":
                self._step = None
                want_yaw = sign * self.yaw_rate()
            elif step and yaw_deg is not None:
                self._step = (yaw_deg + step * self.step_deg(), now, step)
            if axis != "yaw" and self._step is not None and yaw_deg is not None:
                tgt, t0, _d = self._step
                err = wrap180(tgt - yaw_deg)
                if abs(err) < 0.5 or (now - t0) > self.STEP_TIMEOUT_S:
                    self._step = None
                else:
                    want_yaw = max(-self.yaw_rate(), min(self.yaw_rate(), self.STEP_KP * err))
            if axis == "yaw" or self._step is not None:
                self._auto_turning = True
            elif self._auto_turning and abs(yaw_v) < 0.5:
                self._auto_turning = False
            if self._auto_turning or yaw_deg is None:
                if yaw_deg is not None:
                    self.lock_hdg = yaw_deg % 360.0
                self._lock_err, self._lock_act = None, False
                self._lock_prev, self._lock_still = None, 0.0
                if axis != "yaw" and self._step is None:
                    want_yaw = 0.0
            else:
                want_yaw = self._lock_yaw(yaw_deg)
        d = (self.AUTO_YAW_ACCEL if self.phase == "AUTO_ALIGN" or self._auto_turning
             else self.LOCK_ACCEL) * self._dt
        self._yaw.reset(yaw_v + max(-d, min(d, want_yaw - yaw_v)))
        out = {"throttle": self.mapper.phys_to_pwm("throttle", self._thr.value(now)),
               "yaw": self.mapper.phys_to_pwm("yaw", self._yaw.value(now))}
        self._last_out = out
        return out

    def _start_push(self, now, ctx):
        if self._run_t is None:
            self._run_t = now
        start = {
            "roll": self._pilot_value("roll"),
            "pitch": self._pilot_value("pitch"),
            "yaw": self._yaw.value(now) if self._yaw_on else self._pilot_value("yaw"),
            "throttle": (self._thr.value(now) if self._thr_on
                         else self._pilot_value("throttle")),
        }
        self._line_deg = self.line_deg()
        lean = -self._line_deg
        self._push = {}
        for axis, dur, target in (("pitch", self.PUSH_RAMP_S, lean),
                                  ("roll", self.PUSH_RAMP_S, 0.0),
                                  ("yaw", self.YAW_RAMP_S, 0.0),
                                  ("throttle", self.THR_RAMP_S, 0.0)):
            r = Ramp(dur)
            r.reset(start[axis])
            r.set(target, now)
            self._push[axis] = r
        self._push_t = now
        self._steer_after = now
        self._push_deadline = self._run_t + self.idle_s
        self._src = {a: "push" for a in ("roll", "pitch", "yaw", "throttle")}
        self._mix = {}
        self._push_steer = False
        self._thr_on = self._yaw_on = False
        self._step = None
        self._note(f"ENGAGE: PUSH {self._line_deg:.0f} deg; the AI steers "
                   f"yaw / roll / throttle from the first detection >= "
                   f"{self.min_conf:.2f}, and the lean after "
                   f">= {self.PUSH_MIN_S:.1f} s", now)
        self._set_phase("PUSH", now, "ENGAGE")
        return self._push_output(now)

    STEER_AXES = ("yaw", "roll", "throttle")

    def _switch(self, axis, src, now):
        if self._src.get(axis) == src:
            return
        self._mix[axis] = (self._last_out.get(axis), now)
        self._src[axis] = src

    def _mixed_output(self, now, ai):
        out = {}
        for axis in ("roll", "pitch", "yaw", "throttle"):
            if self._src.get(axis) == "ai" and ai is not None and axis in ai:
                target = ai[axis]
            else:
                r = self._push.get(axis) if self._push else None
                target = self.mapper.phys_to_pwm(axis, r.value(now) if r else 0.0)
            frm = self._mix.get(axis)
            if frm is not None and frm[0] is not None:
                w = smoothstep((now - frm[1]) / self.BLEND_S)
                if w >= 1.0:
                    self._mix.pop(axis, None)
                else:
                    target = int(round((1.0 - w) * frm[0] + w * target))
            out[axis] = int(target)
        self._last_out = out
        return out

    def blending(self, now):
        return any(v is not None for v in self._mix.values())

    def _confirmed(self, now, ctx):
        det_t, conf = ctx.get("det_t"), ctx.get("det_conf")
        ai_t = ctx.get("ai_t")
        return (det_t is not None and (now - det_t) <= self.DET_FRESH_S
                and conf is not None and conf >= self.min_conf
                and ai_t is not None and (now - ai_t) <= self.TARGET_FRESH_S
                and bool(ctx.get("ai_cmd")))

    def _push_output(self, now):
        return self._mixed_output(now, None)

    def _push_cycle(self, now, ctx):
        t_in = now - self._push_t
        conf = ctx.get("det_conf")
        fresh_ai = (ctx.get("ai_cmd") and ctx.get("ai_t") is not None
                    and (now - ctx["ai_t"]) <= self.TARGET_FRESH_S)
        ai = self._faded(now, ctx, ctx["ai_cmd"]) if fresh_ai else None
        if not self._push_steer and now >= self._steer_after \
                and self._confirmed(now, ctx):
            self._push_steer = True
            self._steer_t = now
            self.engaged = True
            for a in self.STEER_AXES:
                self._switch(a, "ai", now)
            self._note(f"target confirmed (conf {conf:.2f}) {t_in:.1f} s into "
                       f"PUSH: AI steers yaw / roll / throttle, lean stays "
                       f"{self._line_deg:.0f} deg", now)
        elif self._push_steer and not fresh_ai:
            if self.idle_supported:
                return self._start_idle(now, ctx, "target lost during PUSH")
            self._push_steer = False
            for a in self.STEER_AXES:
                self._switch(a, "push", now)
            self._note("target lost during PUSH: back to heading / height "
                       "hold, still pushing", now)
            ai = None
        if self._push_steer and t_in >= self.PUSH_MIN_S and fresh_ai:
            self._switch("pitch", "ai", now)
            self.last_handover = (conf, t_in)
            self._note(f"{t_in:.1f} s of PUSH: AI takes the lean too "
                       f"(full AI), blended over {self.BLEND_S:.1f} s", now)
            self._set_phase("AI", now, "handover")
            return self._ai_cycle(now, ctx)
        if now >= self._push_deadline:
            if self.idle_supported:
                return self._start_idle(now, ctx, "no confirmed target in time")
            self._note("no confirmed target in time: soft stop", now)
            return self._start_stop(now)
        return self._mixed_output(now, ai)

    def _ai_cycle(self, now, ctx):
        cmd, ai_t = ctx.get("ai_cmd"), ctx.get("ai_t")
        stale = not cmd or ai_t is None or (now - ai_t) > self.TARGET_FRESH_S
        if stale and self.idle_supported:
            return self._start_idle(now, ctx, "target lost")
        if cmd and cmd.get("terminal") and not stale:
            self._terminal = (dict(cmd), now)
        if stale and self._terminal is not None \
                and (now - self._terminal[1]) <= self.TERMINAL_GRACE_S:
            cmd = self._terminal[0]
        elif stale and self._terminal is not None:
            self._note("terminal pass done: soft stop back to level", now)
            return self._start_stop(now)
        elif stale:
            for a in ("roll", "pitch", "yaw", "throttle"):
                self._switch(a, "push", now)
            self._push_steer = False
            self._push_deadline = now + self.REACQ_S
            self._terminal = None
            self._note(f"target lost: back to the straight push, looking again "
                       f"for {self.REACQ_S:.1f} s", now)
            self._set_phase("PUSH", now, "re-acquire")
            return self._mixed_output(now, None)
        return self._mixed_output(now, self._faded(now, ctx, cmd))

    def _faded(self, now, ctx, cmd):
        out = self.mapper.ai_pwm(cmd)
        det_t = ctx.get("det_t")
        gap = 1e9 if det_t is None else now - det_t
        if gap > self.YAW_FADE_S and "yaw" in out and not cmd.get("terminal"):
            k = max(0.0, 1.0 - (gap - self.YAW_FADE_S)
                    / max(1e-3, self.TARGET_FRESH_S - self.YAW_FADE_S))
            n = self.mapper.phys_to_pwm("yaw", 0.0)
            out["yaw"] = int(round(n + (out["yaw"] - n) * k))
        return out

    def _start_stop(self, now):
        self._stop_from = dict(self._last_out)
        self._stop_t = now
        self._push = None
        self._mix = {}
        self._set_phase("STOP", now)
        return self._stop_cycle(now)

    def _stop_cycle(self, now):
        w = smoothstep((now - self._stop_t) / self.STOP_S)
        if w >= 1.0:
            self._set_phase("HOVER", now, "soft stop done")
            self._reset_motion()
            return {}
        out = {}
        for axis, p in self._stop_from.items():
            n = self.mapper.phys_to_pwm(axis, 0.0)
            out[axis] = int(round((1.0 - w) * p + w * n))
        self._last_out = out
        return out

    def snapshot(self, now):
        with self._lock:
            hold = self._hold[0] if self._hold else None
            lv = dict(self.level)
        return dict(
            phase=self.phase, phase_s=round(now - self.phase_t, 1),
            hold=hold, event=self.event,
            event_age=round(now - self.event_t, 1) if self.event_t else None,
            levels=lv,
            speeds=dict(climb=SPEEDS["climb"][lv["vert"]],
                        descent=SPEEDS["descent"][lv["vert"]],
                        yaw=SPEEDS["yaw"][lv["yaw"]],
                        step=SPEEDS["step"][lv["yaw"]],
                        push=SPEEDS["push"][lv["push"]]),
            ceiling_m=self.ceiling_m,
            min_conf=self.min_conf,
            push_s=(round(now - self._push_t, 1) if self._push_t is not None
                    and self.phase == "PUSH" else None),
            blending=self.blending(now),
            ai_steering=bool(self._push_steer) or self.phase == "AI",
            engaged=bool(self.engaged),
            locked=self.lock_hdg is not None,
            auto=self.phase in AUTO_PHASES,
            auto_enabled=bool(self.auto_enabled),
            wasp_enabled=self.wasp.enabled,
            wasp_camera_confirmed=self.wasp.camera_confirmed,
            wasp_params=dict(self.wasp.params),
            idle_s=self.idle_s,
            idle=self.idle.snapshot(),
            idle_remaining_s=None if self._run_t is None else round(max(0.0, self.idle_s-(now-self._run_t)), 2),
            wasp_progress=self.wasp.progress(now),
            camera_mount_deg=self.camera_mount_deg,
            auto_run=bool(self.auto_run),
            auto_params=self.auto_params(),
            auto_defaults=dict(self.auto_defaults),
            level_defaults=dict(self.level_defaults),
            ai_now=self.ai_level_now(),
            line_deg=self.line_deg(),
            auto_height_m=self.AUTO_HEIGHT_M,
            lock_hdg=None if self.lock_hdg is None else round(self.lock_hdg, 1),
            lock_away=bool(self._lock_away),
            handover=self.last_handover)
