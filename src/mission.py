"""
mission.py  -  Hover -> Scout -> Engage, the manual-assist interceptor
======================================================================
The pilot arms and takes off in ALT_HOLD (STABILIZE is not used at all) and
hovers at 1-2 m with the throttle stick CENTRED.  Nothing is accepted until the
FC reports the drone IN THE AIR.  From there the phone drives it:

    HOVER     the Jetson sends nothing; the pilot's sticks are live.
    CLIMB     while ▲ is HELD: throttle channel only, a slow steady climb.
    DESCEND   while ▼ is HELD: throttle channel only.
    SCOUT     while ⟲ / ⟳ is HELD (or one tap = a small step): yaw only.
    AUTO      one button, two straight lines.  AUTO_CLIMB: straight up to
              AUTO_HEIGHT_M, heading locked, until the first detection;
              AUTO_HOLD: there, heading locked, watching.  AUTO_ALIGN: the
              first detection stops the climb and yaw / height put the target
              on the aim mark; held there AUTO_READY_S, it flies the second
              line: a PUSH whose steering the AI takes only after
              AUTO_STEER_S, then the lean, one thing at a time.  AUTO again
              = soft stop; ENGAGE = go now.
    LOCK      (a flag, not a phase) the heading is held where LOCK was
              pressed, or at a heading picked on the phone's degree scroll:
              a closed loop on the FC's yaw, no timeout, against any gust.
              ▲ / ▼ / ⟲ / ⟳ still work while locked; once let go the drone
              turns back to the LOCKED heading (the lock itself never moves
              with them).  LOCK again, ENGAGE, ABORT or the pilot's switch
              clear it.
    PUSH      ENGAGE: all four channels, a fixed nose-down lean, heading and
              altitude held.  The detector cannot steer yet.
    AI        the existing InterceptLaw (proportional navigation), blended in.
    STOP      a SOFT stop: everything eased back to level, then released.
    PILOT     not armed / not ALT_HOLD / on the ground / no link: nothing is
              ever driven.

This module is pure logic.  Time is passed in, the flight controller is
reached through a small `mapper` (physical units <-> PWM), and the only
threads that touch it are the commander's TX thread (`cycle`) and the HTTP
thread (the `request_*` calls, which only POST requests under a lock).  So
every transition can be tested without hardware or wall clock.

WHY EVERYTHING IS RAMPED (the operator's "no drift at takeover" rule)
---------------------------------------------------------------------
An attitude controller answers a STEP in its input with the fastest motor
response it has.  That kick is what reads as a lurch or a drift the moment
the Jetson takes over.  So:

  * BUMPLESS TAKEOVER.  When the Jetson takes a channel, its first value is
    the pilot's OWN stick on that channel (read from RC_CHANNELS while the
    channel was still the pilot's), and it glides from there.  A pilot holding
    2 deg of roll against the wind is not snapped to zero.
  * Every change of command is a SMOOTHSTEP: zero rate of change at both ends,
    so there is no kick when it starts and none when it lands.
  * PUSH -> AI is a 1 s blend in PWM space, from the PUSH output to the AI's.
  * Losing the target, or the push timing out, is a SOFT stop (1 s back to
    level, then release).  Only ABORT and the pilot's mode switch are instant:
    those are the pilot saying "mine", and must never wait.

THE DEAD-MAN
------------
A hold on the phone is a stream of keepalives, not a latch.  The axis moves
only while they keep arriving; 0.3 s without one and it ramps back to zero.
A lost "stop" message over field WiFi therefore stops the drone instead of
flying it away.
"""
from __future__ import annotations

import json
import math
import os
import threading

# ── operator speed choices: SLOW / MEDIUM / FAST for every group ────────────
# Units are physical; the mapper turns them into PWM.  The FC's own ceilings
# (PILOT_SPEED_UP 25 m/s, PILOT_SPEED_DN 2.5 m/s, ANGLE_MAX 40 deg, read at
# connect) cap everything below them -- the mapper clamps at full stick.
SPEEDS = {
    # vertical hold-to-move, m/s.  Descent is capped by PILOT_SPEED_DN (2.5 on
    # this FC): asking for more than the FC allows would just be full stick.
    "climb":   {"SLOW": 1.0, "MEDIUM": 3.0, "FAST": 8.0},
    "descent": {"SLOW": 0.5, "MEDIUM": 1.5, "FAST": 2.5},
    # scout yaw rate while held, deg/s, and the size of one tap, deg.
    "yaw":     {"SLOW": 10.0, "MEDIUM": 20.0, "FAST": 45.0},
    "step":    {"SLOW": 2.0, "MEDIUM": 5.0, "FAST": 10.0},
    # PUSH lean, deg nose-down.  ANGLE_MAX is 40 on this FC.
    "push":    {"SLOW": 20.0, "MEDIUM": 30.0, "FAST": 40.0},
    # after ENGAGE: how long the PUSH looks for a target before the soft
    # stop, s.  Longer = farther: at PUSH 20 deg the drone covers ~19 / 42 /
    # 69 m in 4 / 6 / 8 s (30 deg: 28 / 61 / 97 m; the sim's drag model).
    "search":  {"SLOW": 4.0, "MEDIUM": 6.0, "FAST": 8.0},
}
GROUPS = ("vert", "yaw", "push", "ai", "search")   # what the UI sets independently
LEVELS = ("SLOW", "MEDIUM", "FAST")
AI_LEVELS = LEVELS + ("MAX",)   # the AI group alone has a 4th: tracker MODES["MAX"]

HOLD_KINDS = {"up": ("throttle", +1), "down": ("throttle", -1),
              "yawl": ("yaw", -1), "yawr": ("yaw", +1)}

PHASES = ("PILOT", "HOVER", "CLIMB", "DESCEND", "SCOUT",
          "AUTO_CLIMB", "AUTO_HOLD", "AUTO_ALIGN", "PUSH", "AI", "STOP")
AUTO_PHASES = ("AUTO_CLIMB", "AUTO_HOLD", "AUTO_ALIGN")


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def smoothstep(a):
    a = 0.0 if a <= 0.0 else 1.0 if a >= 1.0 else a
    return a * a * (3.0 - 2.0 * a)


class Ramp:
    """One value that glides to each new target with a smoothstep of fixed
    duration.  Re-targeting mid-glide starts the new glide from wherever the
    value is right now, so it never jumps."""

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
        """At rest on zero: the axis can be handed back to the pilot."""
        return self.target == 0.0 and self.done(now)


def stop_distance(v, accel, jerk, ramp_s):
    """Metres travelled bringing a vertical speed `v` to zero: our own
    smoothstep ramp-out (average v/2 over ramp_s) plus ALT_HOLD's jerk- and
    acceleration-limited stop (PILOT_ACCEL_Z, PSC_JERK_Z).  Conservative."""
    v = abs(v)
    if v <= 0.0:
        return 0.0
    a, j = max(0.1, accel), max(0.1, jerk)
    if v <= a * a / j:
        t = 2.0 * math.sqrt(v / j)
    else:
        t = v / a + a / j
    return 0.5 * v * ramp_s + 0.5 * v * t


# What the Jetson powers up with: a change on the phone applies at once, and
# the phone then asks whether to make it the DEFAULT (operator, 2026-10-06).
# Only a YES writes here, one small JSON per kind.  Two switches are never
# saved, on purpose: AUTO ON / OFF starts OFF (a mode that climbs and attacks
# on its own is a choice made that session) and the SKY GATE starts ON (off,
# detections over the ground count as targets).
CONF_DIR = os.path.expanduser("~/.config/wasp")
AUTO_FILE = os.path.join(CONF_DIR, "auto.json")
LEVELS_FILE = os.path.join(CONF_DIR, "levels.json")
DETECT_FILE = os.path.join(CONF_DIR, "detection.json")
# what each saved detection setting is (the sky gate is not in it, see above);
# "cam" is the CAMERA switch, 30 or 60 fps (the tracker checks the value)
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
    """Every AUTO setting as it is now, as the power-up default."""
    if not _write_json(path, m.auto_params()):
        return False
    m.auto_defaults = m.auto_params()
    return True


def load_auto_params(m, path=AUTO_FILE):
    """Apply what was saved; a bad or unknown entry is skipped, never fatal.
    Returns {setting: value} actually applied.  The factory values with the
    saved ones on top are what the phone marks as DEFAULT."""
    done = {}
    for k, v in _read_json(path).items():
        if m.set_auto_param(k, v) == "":
            done[k] = v
    now = m.auto_params()
    m.auto_defaults = {k: getattr(type(m), attrs[0]) for k, attrs, _c in m.AUTO_SETTINGS}
    m.auto_defaults.update({k: now[k] for k in done})
    return done


def load_levels(m, path=LEVELS_FILE):
    """The speed levels saved as defaults (FLIGHT and ENGAGE on the phone).
    Without a saved file every level starts MEDIUM, as before."""
    done = {}
    for g, lv in _read_json(path).items():
        if m.set_level(str(g), str(lv).strip().upper()) == "":
            done[g] = lv
    m.level_defaults = {g: "MEDIUM" for g in GROUPS}
    m.level_defaults.update({g: m.level[g] for g in done})
    return done


def load_detection_defaults(path=DETECT_FILE):
    """{"follow": bool, "cam": int}, whichever were saved, else {}."""
    out = {}
    for k, v in _read_json(path).items():
        if k in DETECT_SAVED:
            try:
                out[k] = DETECT_SAVED[k](v)
            except (TypeError, ValueError):
                pass                    # a bad entry is skipped, never fatal
    return out


def save_detection_default(key, on, path=DETECT_FILE):
    if key not in DETECT_SAVED:
        return "the SKY GATE always starts ON: it is not saved"
    d = _read_json(path)
    d[key] = DETECT_SAVED[key](on)
    return "" if _write_json(path, d) else "could not save on the Jetson"


def save_default(m, what, auto_path=AUTO_FILE, levels_path=LEVELS_FILE):
    """The phone's YES to 'make it the default': the CURRENT value of one
    setting becomes what the Jetson powers up with.  what = 'auto:<key>',
    'auto:*' (every AUTO setting, after RESET) or 'lvl:<group>'.  Detection
    ('det:<key>') is the tracker's.  Returns '' or a refusal."""
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
    """One phone command applied to a Mission (the Jetson's and the desk
    sim's alike).  Returns '' or a refusal reason.  The AI speed group's
    side effect (the tracker's own mode) is the caller's."""
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
    if key == "lock":                   # lock=on | lock=off | lock=<deg> (idempotent)
        v = value.strip().lower()
        if v in ("on", "off", "1", "0", "true", "false"):
            return m.request_lock(v in ("1", "on", "true"))
        try:
            return m.request_lock(True, float(v))       # the degree scroll
        except ValueError:
            return "bad LOCK value"
    if key == "auto":                   # auto=on | auto=off (idempotent)
        return m.request_auto(value.strip().lower() in ("1", "on", "true"))
    if key == "autop":                  # autop=<setting>:<value>: PARAMETERS -> AUTO
        k, _, v = value.partition(":")
        return m.set_auto_param(k.strip().lower(), v.strip())
    if key == "autoreset":              # autoreset=1: AUTO settings to defaults
        return m.reset_auto_params()
    if key == "autoen":                 # autoen=on | autoen=off: PARAMETERS
        v = value.strip().lower()
        if v not in ("on", "off", "1", "0", "true", "false"):
            return "bad AUTO value"
        return m.set_auto_enabled(v in ("1", "on", "true"))
    if key == "abort":
        return m.request_abort()
    if key == "lvl":                    # lvl=<group>:<LEVEL>
        group, _, level = value.partition(":")
        return m.set_level(group.strip().lower(), level.strip().upper())
    raise ValueError("unknown command")


class Mission:
    # Timings, seconds.  Module-level knobs so the tests and the env can see them.
    DEADMAN_S = 0.30          # hold keepalive timeout
    THR_RAMP_S = 0.50         # climb in / out
    YAW_RAMP_S = 0.30         # scout yaw in / out
    PUSH_RAMP_S = 1.20        # lean eased in at PUSH start
    PUSH_MIN_S = 1.50         # the detector cannot hand over before this
    PUSH_MAX_S = 6.00         # default search: no confirmed target by then -> soft stop
                              # (the phone's SEARCH setting, SPEEDS["search"])
    # PUSH -> AI blend.  Sim, 96 engagements each (6 directions, 2 speeds,
    # 4 winds): blend/lock-ramp 1.0/0.5 s -> 65 touches, mean bank in the
    # first second 11.1 deg, sideways-speed jump p95 4.5 m/s; 1.5/1.0 s -> 67,
    # 5.7 deg, 3.2 m/s; 2.0/1.5 s -> 63, 2.8 deg, 2.0 m/s.  1.5 halves the
    # takeover kick at no cost in accuracy.  MISSION_BLEND_S overrides.
    #
    # 2026-09-25, operator's scenarios (enemy 0-10 m/s toward us, 2-3 Bf wind,
    # the push then the AI), 100 each: a 1.5 s blend lets the AI reach full
    # authority only after most of a fast encounter is gone.  0.4 s (with a
    # 0.3 s lock ramp): 5 m/s 23 -> 44% touches, 10 m/s 6 -> 17% (with the
    # other settings below); the price is ~15 deg of bank in the first second
    # instead of ~8.  Still no spins: yaw is capped at 40 deg/s.
    BLEND_S = 0.40
    STOP_S = 1.00             # soft stop back to level
    STEP_TIMEOUT_S = 3.00
    STEP_KP = 2.5             # deg/s of yaw rate per deg of step error
    STEP_ACCEL = 60.0         # deg/s^2: how fast a step's yaw rate may change
    # LOCK: the same loop as a step, without the timeout.  Inside the dead band the
    # command is zero and the FC's own heading hold does the work; outside it
    # the Jetson closes the loop on ATTITUDE.yaw.  The rate is capped at
    # LOCK_MAX_DPS whatever the yaw speed level: a gust is corrected
    # firmly, never with a spin.  See tests/sim_lock_wind.py.
    LOCK_KP = 2.5             # deg/s of yaw rate per deg of heading error
    # Dead band WITH hysteresis on a smoothed error: start correcting past
    # LOCK_DB_ON, stop inside LOCK_DB_OFF.  A single band chattered the stick
    # ~5 times a second on a 2-deg-noisy heading (tests/test_lock.py 1).
    LOCK_DB_ON = 1.0          # deg
    LOCK_DB_OFF = 0.4         # deg
    LOCK_TAU = 0.20           # s, the error's smoothing
    # Start a correction only once the heading has SETTLED (yaw rate under
    # this).  While a gust is still turning it, ArduPilot's own heading hold
    # is already pulling back; a correction on top is integrated into the
    # FC's target and overshoots (sim: worst excursion 21 -> 33 deg).  What
    # the FC cannot fix is settling at a WRONG heading after a gust dragged
    # its target -- that is what LOCK walks back.
    LOCK_SETTLED_DPS = 4.0
    # ...and has STAYED settled this long.  The apex of a gust swing also has
    # zero yaw rate for an instant; starting there fought the FC's own return
    # and overshot to the other side (-32 deg in the trace that found it).
    LOCK_SETTLE_S = 1.0
    # GENTLE on purpose (operator, 2026-09-27: "as stable and still as
    # possible, never a quick steer").  LOCK turns at most 15 deg/s and
    # changes its turn rate by at most 30 deg/s^2.  Sim (tests/sim_lock_wind,
    # 20 x 60 s per cell, Bf 1-3): the drone's own yaw motion is the WIND's;
    # the RMS yaw rate is the same at 30 or 15 deg/s, and the hold is as good
    # (Bf 3 medium: p95 error 2.2 deg, none ends > 3 deg off).  It never
    # corrects while a gust is still swinging it (LOCK_SETTLE_S above).
    LOCK_MAX_DPS = 15.0
    LOCK_ACCEL = 30.0          # deg/s^2, the rate limit of LOCK's own command
    TARGET_FRESH_S = 0.50     # no REAL detection for this long = target lost
    # A gap in real detections: from YAW_FADE_S the AI's yaw eases to zero by
    # TARGET_FRESH_S, so a target lost off the right edge no longer keeps the
    # drone turning right; roll / pitch / throttle hold the last real command.
    YAW_FADE_S = 0.15
    # Target lost during AI: go back to the PUSH hold (straight, the push
    # lean, heading and height held) and look for it again for this long;
    # a new confirmed detection hands straight back to the AI.
    REACQ_S = 2.50
    # NO GROUND CHECK (operator, 2026-09-30): neither ▼ nor PUSH / AI is
    # stopped near the ground any more.  The field flights were low (the
    # height read 0-2 m above the arming point) and the old 3 m AI floor
    # blocked the AI from descending toward the target 20 times in one
    # session; the ▼ floor (0.5 m) refused 8 times.  The pilot's mode switch
    # is the only guard against the ground.
    # In the last metre the target fills the frame and the detector drops out;
    # the law FREEZES the sticks there and flies through ("terminal commit",
    # held 1 s).  Soft-stopping on that loss would abort the touch itself.
    TERMINAL_GRACE_S = 1.20
    DET_FRESH_S = 0.25        # a handover detection must be this recent
    # AUTO (operator, 2026-10-06).  The first line: straight up at MEDIUM to
    # AUTO_HEIGHT_M with LOCK's heading loop, watching.  No GPS, so "holds
    # position" is heading and height; the wind still drifts it, as in LOCK.
    # Defaults: the operator's choice, 2026-10-06 -- 15 m, a 30 deg line, AI
    # MAX, the AI steering 0.6 s into the line.  Every one is a phone setting.
    AUTO_HEIGHT_M = 15.0
    AUTO_CLIMB_MPS = 3.0      # vert MEDIUM
    AUTO_HEIGHT_KP = 1.0      # m/s per m short: it slows into the height,
                              # rather than braking early on a worst case
    AUTO_THR_ACCEL = 3.0      # m/s^2: how fast ALIGN's climb command may change
    # ALIGN: yaw puts the azimuth on 0, height puts the elevation on the aim
    # mark (the phone's: -min(PUSH, 20) deg in the camera, so the target is
    # centred once the push lean is on).  Gentle on purpose: rate-limited like
    # LOCK, and nothing moves on a detection older than YAW_FADE_S.  The yaw
    # also turns at the target's own bearing rate (los_az_dps): a P loop alone
    # trails a crossing target by rate / KP and never counts as aligned.
    #
    # Sim (sim_sky_wind, 960 engagements per row, AI MEDIUM, PUSH 30), AUTO vs
    # ENGAGE pressed at once: hovering 4-8 m up, target in frame 76 vs 70%
    # touches; from 1.5 m, target 25-50 m out and 6-25 m up 79 vs 61%;
    # head-on targets 71 vs 49 / 29%.  It costs time: touch at ~7-8 s, not 5-6.
    # These values beat yaw 20 / climb 3 / KP 0.4 (72 / 76%) and ready 0.4 s /
    # 3 s cap (68 / 67%).
    AUTO_YAW_KP = 2.0         # deg/s per deg of azimuth
    AUTO_YAW_MAX = 30.0       # deg/s
    AUTO_YAW_ACCEL = 60.0     # deg/s^2
    AUTO_EL_KP = 0.6          # m/s of climb per deg of elevation off the mark
    AUTO_ALIGN_CLIMB_MPS = 5.0  # ALIGN's own climb cap
    AUTO_DESCENT_MPS = 1.5    # vert MEDIUM
    # ALIGN never descends below this.  AUTO flies itself: unlike the pilot's
    # ▼, nobody is watching the ground for it (a target low on the horizon
    # pulled the 1.5 m starts into the ground in 1% of the sim runs).
    AUTO_FLOOR_M = 3.0
    # Ready = inside these this long.  Elevation is one-sided on purpose: a
    # target BELOW the mark only rises in the frame once leaning; one above
    # it leaves the top.
    AUTO_AZ_OK = 5.0          # deg
    AUTO_EL_OK = 5.0          # deg above the mark...
    AUTO_EL_LOW = 12.0        # ...or below it
    AUTO_READY_S = 0.25
    # ...or go anyway: a target closing this fast is arriving, and waiting for
    # a perfect line would let it fly past (range_m needs target_width_m)...
    AUTO_GO_TTG_S = 3.0       # s to go at the measured closing speed,
    AUTO_GO_HOLD_S = 0.5      # held this long (it is a bbox derivative: noisy)
    # ...or once ALIGN has had this long and the target is near the nose.
    AUTO_ALIGN_MAX_S = 2.0
    AUTO_GO_AZ = 15.0         # deg
    # The second line.  The lean ramps in over PUSH_RAMP_S; the AI's yaw /
    # roll / throttle blend in over BLEND_S from AUTO_STEER_S, the lean from
    # PUSH_MIN_S.  Already lined up, the AI's first commands are small.  Sim:
    # holding the AI off longer buys NO smoothness (bank change in 0.2 s,
    # p95 31-32 deg at 0 / 0.3 / 0.6 / 1.2 s) and costs touches (76 / 72 /
    # 68 / 62% from a hover), so it started at once until the operator
    # chose 0.6 s (2026-10-06).
    AUTO_STEER_S = 0.6
    # How fast the run is.  The straight line's lean (0 = the PUSH setting)
    # and the AI level the run hands over to ("SAME" = the AI setting).  MAX
    # is the 35 deg / ~21 m/s profile (tracker MODES), the fastest there is.
    # 20 deg: sim_sky_wind SIM_START=horizon (864 runs each, 2026-10-06), AI
    # MEDIUM: line 20 38.7%, 30 37.8%; AI MAX: 20 39.7%, 30 39.1%, 40 34.0% --
    # a steeper line tips the camera off the target.  Same on the ground set.
    # The operator's choice is 30 deg and MAX (2026-10-06).
    AUTO_LINE_DEG = 30.0
    AUTO_AI = "MAX"

    # Every AUTO number above is a phone setting (☰ -> PARAMETERS -> AUTO),
    # picked from these choices and saved on the Jetson (AUTO_FILE).  The
    # class values are the defaults.
    AUTO_SETTINGS = (
        # key        Mission attribute(s)              choices
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
                 accel_z=5.0, jerk_z=5.0, log=print):
        self.mapper = mapper
        self.min_conf = float(min_conf)
        self.ceiling_m = float(ceiling_m)
        self.accel_z = float(accel_z)       # PILOT_ACCEL_Z, m/s^2
        self.jerk_z = float(jerk_z)         # PSC_JERK_Z, m/s^3
        self.log = log

        self._lock = threading.Lock()
        # Requests posted by the HTTP thread, consumed by cycle().
        self._hold = None                   # (kind, session, t_last)
        self._dead_sessions = set()
        self._released = set()              # sessions whose finger is up
        self._req_step = None               # -1 / +1
        self._req_lock = None               # (on, heading|None), applied by cycle()
        self.lock_hdg = None                # deg, the heading LOCK holds
        self._lock_away = False             # ⟲/⟳ turned it away: goes back once let go
        self._lock_err = None               # smoothed heading error, deg
        self._lock_act = False              # correcting (hysteresis state)
        self._lock_prev = None              # last yaw reading, deg
        self._lock_rate = 0.0               # smoothed yaw rate, deg/s
        self._lock_still = 0.0              # s the yaw has been still
        self._req_engage = False
        self._req_abort = False
        self._req_auto = None               # "on" / "off", applied by cycle()
        # ☰ -> PARAMETERS -> AUTO.  OFF at power-up: a mode that climbs and
        # attacks on its own should be a choice made that session, like FAST.
        self.auto_enabled = False
        # this instance's AUTO settings, starting from the class defaults
        for _k, attrs, _c in self.AUTO_SETTINGS:
            for a in attrs:
                setattr(self, a, getattr(type(self), a))
        self.ai_override = None             # an AUTO run's own AI level
        self._auto_ready_t = None           # aligned since
        self._auto_align_t = 0.0            # ALIGN began
        self._auto_close_t = None           # target arriving since
        self._auto_turning = False          # ⟲ / ⟳ during AUTO: the lock follows
        self.auto_run = False               # this attempt began with AUTO (UI)
        self._steer_after = 0.0             # PUSH: no AI steering before this
        # Every setting MEDIUM by default: height 3 / 1.5 m/s, turn 20 deg/s,
        # PUSH 30 deg, AI MEDIUM, search 6 s -- the best in the field
        # (operator, 2026-09-30).  The sims use the same defaults.
        self.level = {g: "MEDIUM" for g in GROUPS}
        # What it powers up with (the phone marks it DEFAULT): these until
        # load_levels / load_auto_params read what the operator saved.
        self.level_defaults = dict(self.level)
        self.auto_defaults = self.auto_params()

        self.phase = "PILOT"
        self.phase_t = 0.0
        self.event = ""                     # last notable thing, for the UI
        self.event_t = 0.0
        self.refusal = ""
        self._thr = Ramp(self.THR_RAMP_S)   # m/s of climb
        self._yaw = Ramp(self.YAW_RAMP_S)   # deg/s of yaw
        self._thr_on = False                # throttle channel currently ours
        self._yaw_on = False
        self._step = None                   # (target_yaw_deg, t0, dir)
        self._push = None                   # dict of Ramps per axis
        self._push_t = None
        self._stop_from = None              # {axis: pwm} at soft-stop start
        self._stop_t = None
        self._last_out = {}
        self._terminal = None
        self._src = {}
        self._mix = {}
        self._push_steer = False
        self._steer_t = None
        self._t_last = None
        self._height = self._climb = None
        self.last_handover = None           # (conf, t_in_push)
        # ONE attempt per pilot mode cycle.  Latched the moment the AI takes
        # any channel; cleared only when the pilot leaves ALT_HOLD (STAB) and
        # comes back.  The phone shows ENGAGE green while it is set.
        self.engaged = False

    # ── HTTP thread: requests only ─────────────────────────────────────────
    def request_hold(self, kind, session, now):
        """A keepalive for one held button.  Returns '' or a refusal reason."""
        if kind not in HOLD_KINDS:
            return "unknown hold"
        with self._lock:
            if session in self._dead_sessions:
                return "hold ended by ABORT: lift the finger and press again"
            if session in self._released:
                # a keepalive the WiFi delayed past its own release: the
                # finger is already up, so it must not restart the axis
                return "hold already released"
            if self.phase in ("AUTO_CLIMB", "AUTO_HOLD") and HOLD_KINDS[kind][0] == "yaw":
                pass                    # turning to look around: the lock follows
            elif self.phase in ("PUSH", "AI", "STOP") + AUTO_PHASES:
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
        """LOCK on / off from the phone.  Idempotent (on twice = on).  With
        `hdg` (deg, the degree scroll): lock at THAT heading and turn to it,
        also when already locked (the lock moves there)."""
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
            if self.engaged:
                return "already engaged: pilot STAB -> ALT_HOLD for a new try"
            if self.phase not in ("HOVER", "SCOUT") + AUTO_PHASES:
                return f"ENGAGE only from HOVER, SCOUT or AUTO, not {self.phase}"
            self._req_engage = True
        return ""

    def set_auto_enabled(self, on):
        """The PARAMETERS switch.  Turned OFF while AUTO flies = AUTO off."""
        with self._lock:
            self.auto_enabled = bool(on)
            if not on and self.phase in AUTO_PHASES:
                self._req_auto = "off"
        return ""

    def auto_params(self):
        return {k: getattr(self, attrs[0]) for k, attrs, _c in self.AUTO_SETTINGS}

    def set_auto_param(self, key, value):
        """One AUTO setting from the phone.  Only the listed choices, and not
        while an AUTO run is flying."""
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
        """The AI level the intercept law should fly right now: an AUTO run's
        own, else the AI setting.  The tracker applies it every frame."""
        return self.ai_override or self.level["ai"]

    def line_deg(self):
        """The straight line's lean: AUTO's own during an AUTO run, else PUSH."""
        return self.AUTO_LINE_DEG if (self.auto_run and self.AUTO_LINE_DEG) else self.push_deg()

    def request_auto(self, on):
        """AUTO on / off from the phone.  Idempotent: on twice = on."""
        with self._lock:
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

    # ── derived settings ───────────────────────────────────────────────────
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

    def search_s(self):
        return SPEEDS["search"][self.level["search"]]

    # ── TX thread ──────────────────────────────────────────────────────────
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
        # every path back to "the pilot / nothing driving" clears LOCK:
        # gate lost, ABORT, soft stop done, and ENGAGE (_start_push) itself
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
        """The pilot's own stick on `axis` in physical units, or 0."""
        pwm = self.mapper.pilot_pwm(axis)
        return 0.0 if pwm is None else self.mapper.pwm_to_phys(axis, pwm)

    def cycle(self, now, ctx):
        """One TX cycle.  ctx: gate_ok, gate_why, att (dict|None), height
        (m|None), climb (m/s|None), ai_cmd (dict|None), ai_t (time of the last
        AI command from a real target), det_t / det_conf (last gated detection).

        Returns {axis: pwm} to override (empty = release everything)."""
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

        # ── the pilot's gate: always first, always instant ────────────────
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

    # ── HOVER / CLIMB / DESCEND / SCOUT ────────────────────────────────────
    def _manual(self, now, ctx, hold, step):
        kind = hold[0] if hold else None
        axis, sign = HOLD_KINDS.get(kind, (None, 0))
        height = ctx.get("height")
        climb = ctx.get("climb")

        # vertical
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
            # Bumpless: start from the pilot's own throttle, as a climb rate.
            self._thr.reset(self._pilot_value("throttle"))
            self._thr_on = True
        self._thr.set(want_thr, now)

        # yaw: a held button, or a closed-loop step
        att = ctx.get("att")
        yaw_deg = None if att is None else math.degrees(att.get("yaw", 0.0))
        want_yaw = 0.0
        locked = self.lock_hdg is not None
        if axis == "yaw":
            self._step = None
            want_yaw = sign * self.yaw_rate()
            if locked:
                self._lock_away = True          # turn back once let go
        elif step and yaw_deg is not None:
            # while locked a tap is a turn away too: the lock does not move
            self._step = (yaw_deg + step * self.step_deg(), now, step)
            if locked:
                self._lock_away = True
        elif step:
            self._refuse("no attitude: no yaw step", now)
        lock_loop = False
        if locked and axis != "yaw" and self._step is None:
            if self._lock_away:
                # finger up: let the yaw ramp land, then turn straight back to
                # the LOCKED heading.  No settle wait: the FC's target moved
                # with the stick, there is no gust swing of its own to fight.
                if self._yaw.idle(now) and yaw_deg is not None:
                    self._lock_away = False
                    self._lock_err, self._lock_act = None, False
                    self._lock_prev, self._lock_rate = None, 0.0
                    self._lock_still = self.LOCK_SETTLE_S
                    self._note(f"LOCK: back to {self.lock_hdg:.0f} deg", now)
            elif yaw_deg is None:
                self._lock_err, self._lock_act = None, False   # no attitude: zero rate
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
            # A step's target moves every cycle, and a smoothstep restarted
            # every cycle never leaves its (zero-slope) start.  Rate-limit it.
            # LOCK's correction is the same kind of command.
            v = self._yaw.value(now)
            d = (self.LOCK_ACCEL if lock_loop else self.STEP_ACCEL) * self._dt
            self._yaw.reset(v + max(-d, min(d, want_yaw - v)))
        else:
            self._yaw.set(want_yaw, now)

        # hand an axis back only once its ramp has landed on zero
        if self._thr_on and self._thr.idle(now) and want_thr == 0.0:
            self._thr_on = False
        if self._yaw_on and self._yaw.idle(now) and want_yaw == 0.0 \
                and self._step is None and not locked:
            self._yaw_on = False
        if locked and not self._yaw_on:
            # LOCK owns the yaw channel for as long as it is on: centred
            # (zero rate) inside the dead band, a correction outside it
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
        """LOCK's heading loop, one cycle: the yaw rate (deg/s) that walks the
        heading back to lock_hdg once it has settled, else 0."""
        if self._lock_prev is not None and self._dt > 0:
            d = ((yaw_deg - self._lock_prev + 180.0) % 360.0 - 180.0) / self._dt
            self._lock_rate += (d - self._lock_rate) * (self._dt / (0.3 + self._dt))
        self._lock_prev = yaw_deg
        self._lock_still = (self._lock_still + self._dt
                            if abs(self._lock_rate) < self.LOCK_SETTLED_DPS else 0.0)
        err = (self.lock_hdg - yaw_deg + 180.0) % 360.0 - 180.0
        f = self._lock_err
        if f is None or abs(err - f) > 90.0:   # first sample, or a jump
            f = err
        else:
            f += (err - f) * (self._dt / (self.LOCK_TAU + self._dt))
        self._lock_err = f
        # smoothed error decides WHETHER to correct (no chatter on a
        # noisy heading); the actual error sets HOW MUCH (no lag, so
        # no overshoot as it arrives)
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
            return                  # already locked: lock=on again changes nothing
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
            # a picked heading: turn there at once (a deliberate turn, not a
            # gust, so no settle wait) and hold it
            self.lock_hdg = hdg
            self._lock_prev, self._lock_rate = None, 0.0
            self._lock_still = self.LOCK_SETTLE_S
            self._note(f"LOCK on: turning to {self.lock_hdg:.0f} deg and holding it", now)

    def _refuse(self, why, now):
        if self.refusal != why or (now - self.event_t) > 2.0:
            self._note(f"refused: {why}", now)
        self.refusal = why

    # ── AUTO ───────────────────────────────────────────────────────────────
    # The aim mark stays at least this far inside the bottom of the picture.
    # At 30 fps (+/-24.4 deg) the -20 deg mark is 4.4 deg in, so nothing moves;
    # the 60 fps camera sees only +/-14.85 deg, where -20 is below the picture
    # and no target could ever be lined up: the mark comes up to -10.45 deg.
    AUTO_AIM_EDGE_DEG = 4.4

    def auto_aim_el(self):
        """Camera elevation, deg, where ALIGN puts the target (the phone's aim
        mark): centred in the frame once the line's lean is on, as far as the
        camera's picture (the mapper's V_FOV, the camera mode's) reaches."""
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
        hdg = (self.lock_hdg if self.lock_hdg is not None
               else math.degrees(att.get("yaw", 0.0)) % 360.0)
        self._unlock()
        self.lock_hdg = hdg
        self._step = None
        # bumpless: both channels start from what is flown right now
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
        """Back to the first line: lock the heading where it is, keep climbing
        if not at AUTO_HEIGHT_M yet, else hold."""
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
            self._auto_watch(now, ctx, "target lost while aligning")

        thr_v, yaw_v = self._thr.value(now), self._yaw.value(now)
        if self.phase == "AUTO_ALIGN":
            # nothing moves on a stale detection: a target that left the frame
            # must not keep the drone turning after it
            live = det_t is not None and (now - det_t) <= self.YAW_FADE_S
            az = float(cmd.get("az_deg", 0.0))
            err_el = float(cmd.get("el_cam_deg", 0.0)) - self.auto_aim_el()
            ff = float(cmd.get("los_az_dps", 0.0) or 0.0)
            want_yaw = max(-self.AUTO_YAW_MAX, min(self.AUTO_YAW_MAX,
                                                   ff + self.AUTO_YAW_KP * az)) if live else 0.0
            # above the mark = climb (the target sinks in the frame)
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
            # ⟲ / ⟳ while watching: the operator turns it to look around (held,
            # or a tap = a step).  The lock moves with it and holds wherever
            # it stops, rather than turning back as plain LOCK does.
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
                self._auto_turning = False          # stopped: lock right here
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

    # ── PUSH ───────────────────────────────────────────────────────────────
    def _start_push(self, now, ctx):
        # Every axis starts from what is being flown RIGHT NOW: the pilot's
        # stick, or our own ramp if we already hold that channel.
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
        self._push_deadline = now + self.search_s()
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

    # Per-axis source switching.  Every axis is driven by exactly one source,
    # "push" (the held push command) or "ai" (the InterceptLaw), and a switch
    # glides from whatever that axis is outputting RIGHT NOW to the new source
    # over BLEND_S.  So PUSH -> AI steering, AI -> back to PUSH hold when the
    # target is lost mid-push, and the final pitch handover are all smooth.
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
        """PUSH: the lean is the push angle throughout.  From the FIRST
        confirmed detection the AI steers yaw, roll and throttle (it keeps the
        target centred and sets the height); after PUSH_MIN_S it takes pitch
        too, and that is the AI phase."""
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
            self._note("no confirmed target in time: soft stop", now)
            return self._start_stop(now)
        return self._mixed_output(now, ai)

    # ── AI ─────────────────────────────────────────────────────────────────
    def _ai_cycle(self, now, ctx):
        cmd, ai_t = ctx.get("ai_cmd"), ctx.get("ai_t")
        stale = not cmd or ai_t is None or (now - ai_t) > self.TARGET_FRESH_S
        # Only a FRESH command can (re)start the terminal hold.  A stale one
        # that still says "terminal" would otherwise renew the grace every
        # cycle and fly the frozen sticks for ever after a pass (the
        # commander clears ai_cmd when stale; this does not rely on it).
        if cmd and cmd.get("terminal") and not stale:
            self._terminal = (dict(cmd), now)
        if stale and self._terminal is not None \
                and (now - self._terminal[1]) <= self.TERMINAL_GRACE_S:
            cmd = self._terminal[0]           # keep flying the frozen command
        elif stale and self._terminal is not None:
            # The terminal hold ran out: it touched or flew past.  The attempt
            # is over -- do NOT push on blind past the target.
            self._note("terminal pass done: soft stop back to level", now)
            return self._start_stop(now)
        elif stale:
            # Lost: not a stop.  Back to the straight push, glide there, and
            # look again; the operator's rule is "go straight until it finds it".
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
        """The AI's PWM, with yaw eased to zero through a detection gap (not in
        the terminal hold, which is a deliberate freeze)."""
        out = self.mapper.ai_pwm(cmd)
        det_t = ctx.get("det_t")
        gap = 1e9 if det_t is None else now - det_t
        if gap > self.YAW_FADE_S and "yaw" in out and not cmd.get("terminal"):
            k = max(0.0, 1.0 - (gap - self.YAW_FADE_S)
                    / max(1e-3, self.TARGET_FRESH_S - self.YAW_FADE_S))
            n = self.mapper.phys_to_pwm("yaw", 0.0)
            out["yaw"] = int(round(n + (out["yaw"] - n) * k))
        return out

    # ── STOP (soft) ────────────────────────────────────────────────────────
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

    # ── read-back for the UI ───────────────────────────────────────────────
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
                        push=SPEEDS["push"][lv["push"]],
                        search=SPEEDS["search"][lv["search"]]),
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
