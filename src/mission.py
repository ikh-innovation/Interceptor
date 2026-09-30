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

import math
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

HOLD_KINDS = {"up": ("throttle", +1), "down": ("throttle", -1),
              "yawl": ("yaw", -1), "yawr": ("yaw", +1)}

PHASES = ("PILOT", "HOVER", "CLIMB", "DESCEND", "SCOUT", "PUSH", "AI", "STOP")


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
        # Every setting MEDIUM by default: height 3 / 1.5 m/s, turn 20 deg/s,
        # PUSH 30 deg, AI MEDIUM, search 6 s -- the best in the field
        # (operator, 2026-09-30).  The sims use the same defaults.
        self.level = {g: "MEDIUM" for g in GROUPS}

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
            if self.phase in ("PUSH", "AI", "STOP"):
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
            if self.phase not in ("HOVER", "SCOUT"):
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
            if self.phase not in ("HOVER", "SCOUT"):
                return f"ENGAGE only from HOVER or SCOUT, not {self.phase}"
            self._req_engage = True
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
        if group not in GROUPS or level not in LEVELS:
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

        if self.phase in ("HOVER", "CLIMB", "DESCEND", "SCOUT"):
            if engage:
                self._unlock("ENGAGE: the AI takes the yaw", now)
                return self._start_push(now, ctx)
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
                want_yaw = max(-lim, min(lim, self.LOCK_KP * err)) if self._lock_act else 0.0
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
        lean = -self.push_deg()
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
        self._push_deadline = now + self.search_s()
        self._src = {a: "push" for a in ("roll", "pitch", "yaw", "throttle")}
        self._mix = {}
        self._push_steer = False
        self._thr_on = self._yaw_on = False
        self._step = None
        self._note(f"ENGAGE: PUSH {self.push_deg():.0f} deg; the AI steers "
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
        if not self._push_steer and self._confirmed(now, ctx):
            self._push_steer = True
            self._steer_t = now
            self.engaged = True
            for a in self.STEER_AXES:
                self._switch(a, "ai", now)
            self._note(f"target confirmed (conf {conf:.2f}) {t_in:.1f} s into "
                       f"PUSH: AI steers yaw / roll / throttle, lean stays "
                       f"{self.push_deg():.0f} deg", now)
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
        if cmd and cmd.get("terminal"):
            self._terminal = (dict(cmd), now)
        stale = not cmd or ai_t is None or (now - ai_t) > self.TARGET_FRESH_S
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
            lock_hdg=None if self.lock_hdg is None else round(self.lock_hdg, 1),
            lock_away=bool(self._lock_away),
            handover=self.last_handover)
