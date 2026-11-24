"""SEEK's attack after the camera lock, in ALT_HOLD (the camera fixed at 90 degrees, looking up).

SEEK follows the ASTERIX cue 10 m under the target; when the camera locks on,
the drone climbs into it in ALT_HOLD instead of handing over to WASP in ACRO:

  * throttle stick  = climb rate (full stick = PILOT_SPEED_UP, accelerating at
    PILOT_ACCEL_Z); full while the target is near overhead, less when it is
    far off the vertical so the lean can bring it back first;
  * roll/pitch sticks = lean toward the target: its direction in the world
    (the camera ray turned by the attitude, the same as WASP's observation),
    proportional to how far it is off the vertical plus its drift (damping);
  * yaw stick centred: with the camera looking up the heading does not matter.

No fresh detection: sticks centred, so ALT_HOLD holds the height and levels.
The attack ends when the target is no longer above (passed or hit), when it
is lost for LOST_S, or after MAX_S; SEEK then hovers and flies RTL.  If the
overrides stop for any reason the FC is still in ALT_HOLD on the pilot's
sticks (throttle at the middle: it holds the height) - it never falls.
"""
from __future__ import annotations

import math

from wasp import rotation


class SeekAttack:
    # Gains from the ALT_HOLD simulation (tests/test_seek_attack.py): the middle of a range
    # (KP 1.25-1.75, KD 1.5-2.5) that reaches every tested target, standing to 12 m/s.
    KP = 1.5             # degrees of lean per degree the target sits off the vertical
    KD = 2.0             # seconds: lean per (degree/s) of its drift across the sky (leads a mover)
    LEAN_MAX = 35.0      # below ANGLE_MAX (40) so the FC never saturates
    FULL_CLIMB_DEG = 20.0   # full climb while the target is this close to overhead
    NO_CLIMB_DEG = 35.0     # no climb beyond: lean under it first, or it leaves the camera
    PASSED_DEG = 75.0    # this far off the vertical after closing in: no longer above
    FRESH_S = 0.3
    LOST_S = 1.5
    MAX_S = 15.0

    def __init__(self):
        self.active = False
        self.reset()

    def reset(self):
        self.started = self.last_seen = None
        self.done = None             # why the attack is over (it then holds a hover)
        self.closest_deg = None
        self.last = {}

    def begin(self, now):
        self.reset()
        self.active, self.started, self.last_seen = True, now, now

    def stop(self):
        self.active = False

    def _finish(self, why):
        if self.done is None:
            self.done = why

    def output(self, mapper, now, ctx):
        """The four sticks (PWM) for ALT_HOLD, or {} when not attacking."""
        if not self.active:
            return {}
        roll = pitch = 0.0
        climb_frac = 0.0
        obs, att = ctx.get("ai_cmd"), ctx.get("att")
        bearing = obs.get("bearing") if isinstance(obs, dict) else None
        fresh = (bearing is not None and att is not None and obs.get("capture_t") is not None
                 and 0.0 <= now - obs["capture_t"] <= self.FRESH_S)
        if self.done is None and fresh:
            n, e, d = (float(v) for v in bearing)
            up = -d
            off = math.degrees(math.acos(max(-1.0, min(1.0, up))))
            if off >= self.PASSED_DEG:
                self._finish(f"target {off:.0f}° off the vertical: passed")
            else:
                self.last_seen = now
                self.closest_deg = off if self.closest_deg is None else min(self.closest_deg, off)
                # Lean toward the target: angle off the vertical, north and east parts.
                an, ae = math.degrees(math.atan2(n, up)), math.degrees(math.atan2(e, up))
                rate = obs.get("bearing_rate") or (0.0, 0.0, 0.0)
                dn, de, dd = (float(v) for v in rate)
                # d/dt atan2(n, up) = (dn*up - n*dup) / (n^2 + up^2), with dup = -dd
                rn = math.degrees((dn*up + n*dd) / max(n*n + up*up, 1e-6))
                re = math.degrees((de*up + e*dd) / max(e*e + up*up, 1e-6))
                ln, le = self.KP*an + self.KD*rn, self.KP*ae + self.KD*re
                mag = math.hypot(ln, le)
                if mag > self.LEAN_MAX:
                    ln, le = ln*self.LEAN_MAX/mag, le*self.LEAN_MAX/mag
                r = rotation(att)
                yaw = math.atan2(r[1][0], r[0][0])
                fwd = ln*math.cos(yaw) + le*math.sin(yaw)
                right = -ln*math.sin(yaw) + le*math.cos(yaw)
                pitch, roll = -fwd, right          # nose down to go forward; roll right to go right
                climb_frac = max(0.0, min(1.0, (self.NO_CLIMB_DEG - off)/(self.NO_CLIMB_DEG - self.FULL_CLIMB_DEG)))
                self.last = dict(off_deg=round(off, 1), lean_n=round(ln, 1), lean_e=round(le, 1),
                                 roll=round(roll, 1), pitch=round(pitch, 1), climb=round(climb_frac, 2))
        if self.done is None:
            if now - self.last_seen > self.LOST_S:
                self._finish(f"target not seen for {self.LOST_S:g} s")
            elif now - self.started > self.MAX_S:
                self._finish(f"attack longer than {self.MAX_S:g} s")
        if self.done is not None:
            roll = pitch = climb_frac = 0.0
        limit = getattr(getattr(mapper, "backend", None), "max_climb_mps", None) or 2.5
        out = dict(mapper.idle_pwm(0.0, climb_frac * limit))
        out.update(mapper.lean_pwm(roll, pitch))
        return out
