
import math


class Idle:
    MODE_TIMEOUT_S = 2.0
    RISE_M = 1.5
    MAX_CLIMB_MPS = 1.0
    MAX_YAW_DPS = 15.0
    # WASP recovery (ABORT, or the pilot switching ACRO -> STABILIZE): ALT_HOLD
    # stops the sink at once, levels the drone so the launch speed bleeds off,
    # then climbs a little higher and holds there until the pilot changes mode.
    RECOVERY_RISE_M = 3.0
    RECOVERY_CLIMB_MPS = 2.0

    def __init__(self):
        self.started = None
        self.confirmed = False
        self.stage = "OFF"
        self.height = self.heading = None
        self.source_mode = None
        self.recovery = False
        self.accepted = False
        self.hold_throttle = None
        self.rise_m, self.max_climb = self.RISE_M, self.MAX_CLIMB_MPS
        self._t = None
        self._climb = self._yaw = 0.0

    @property
    def active(self):
        return self.started is not None

    @staticmethod
    def telemetry_error(ctx):
        if not ctx.get("idle_ok"):
            return ctx.get("idle_why") or "Idle controller unavailable"
        att = ctx.get("att")
        if att is None or ctx.get("height") is None or ctx.get("climb") is None:
            return "Idle needs fresh attitude, height and climb rate"
        values = [ctx["height"], ctx["climb"]]+[att.get(k, float("nan")) for k in ("roll", "pitch", "yaw")]
        if not all(math.isfinite(v) for v in values):
            return "invalid Idle telemetry"
        return ""

    def begin(self, now, ctx, ceiling, recovery=False, hold_throttle=None):
        why = self.telemetry_error(ctx)
        if why and not recovery:
            raise ValueError(why)
        self.source_mode = ctx["fc_mode"]
        sources = ("ACRO", "STABILIZE", "ALT_HOLD") if recovery else ("ACRO", "ALT_HOLD")
        if self.source_mode not in sources:
            raise ValueError("pilot mode: Idle must not take control")
        self.recovery = recovery
        self.hold_throttle = hold_throttle
        self.switch_band = ctx.get("mode_switch")
        self.rise_m, self.max_climb = ((self.RECOVERY_RISE_M, self.RECOVERY_CLIMB_MPS) if recovery
                                       else (self.RISE_M, self.MAX_CLIMB_MPS))
        self.ceiling = ceiling
        # A recovery must start even on a stale sample; the height target and
        # heading are then taken from the first fresh one.
        self.height = None if why else max(ctx["height"], min(ctx["height"]+self.rise_m, ceiling))
        self.heading = None if why else ctx["att"]["yaw"]
        self.started = self._t = now
        self.confirmed = self.source_mode == "ALT_HOLD"
        self.accepted = self.confirmed
        self._climb = self._yaw = 0.0
        self.stage = "LEVEL" if self.confirmed else "SWITCHING"

    def stop(self):
        self.started = None
        self.recovery = False
        self.accepted = False
        self.hold_throttle = None
        self.stage = "OFF"

    def output(self, mapper, now, ctx):
        why = self.telemetry_error(ctx)
        if why and not self.recovery:
            raise ValueError(why)
        mode = ctx["fc_mode"]
        if ctx.get("idle_mode_denied"):
            raise ValueError("FC rejected ALT_HOLD for Idle")
        if self.recovery and ctx.get("idle_mode_accepted"):
            # The FC's ACK arrives long before its 1 Hz heartbeat shows the new
            # mode; from the ACK on the sticks already mean ALT_HOLD.
            self.accepted = True
        if mode != "ALT_HOLD":
            # Until the heartbeat shows ALT_HOLD it may still show the launch
            # ACRO or the pilot's STABILIZE: neither is a new pilot choice.
            sources = ("ACRO", "STABILIZE") if self.recovery else (self.source_mode,)
            if self.confirmed or mode not in sources:
                raise ValueError("pilot changed mode")
            if now-self.started >= self.MODE_TIMEOUT_S:
                raise ValueError("ALT_HOLD was not confirmed")
            if not self.accepted:
                self.stage = "SWITCHING"
                self._t = now
                if self.hold_throttle is not None:
                    # Mid-stick is not hover in ACRO / STABILIZE: keep the launch
                    # throttle, sticks level, until ALT_HOLD holds the height.
                    return mapper.wasp_pwm((0.0, 0.0, 0.0), self.hold_throttle)
                return mapper.idle_pwm(0.0, 0.0)
        else:
            self.confirmed = True
        dt = max(0.0, min(.1, now-self._t))
        self._t = now
        if why:
            # Recovery in ALT_HOLD with a stale sample: centred sticks level the
            # drone and hold its height without any telemetry.
            self.stage = "HOLD"
            self._climb = self._yaw = 0.0
            return mapper.idle_pwm(0.0, 0.0)
        if self.height is None:
            self.height = max(ctx["height"], min(ctx["height"]+self.rise_m, self.ceiling))
        if self.heading is None:
            self.heading = ctx["att"]["yaw"]
        att = ctx["att"]
        if self.recovery and ctx["height"] > self.height:
            # Launch momentum can carry it above the target: hold the top
            # rather than descend back towards the ground.
            self.height = ctx["height"]
        level = max(abs(att["roll"]), abs(att["pitch"])) <= math.radians(10)
        error = self.height-ctx["height"]
        want_climb = max(-self.max_climb, min(self.max_climb, .8*error)) if level else 0.0
        yaw_error = math.degrees((self.heading-att["yaw"]+math.pi) % (2*math.pi)-math.pi)
        want_yaw = max(-self.MAX_YAW_DPS, min(self.MAX_YAW_DPS, 2*yaw_error))
        self._climb += max(-dt, min(dt, want_climb-self._climb))
        self._yaw += max(-30*dt, min(30*dt, want_yaw-self._yaw))
        self.stage = ("LEVEL" if not level else "HOLD"
                      if abs(error) <= .2 and abs(ctx["climb"]) <= .2 else "CLIMB")
        return mapper.idle_pwm(self._yaw, self._climb)

    def switch_moved(self, band):
        """True once the pilot moves the mode switch after a recovery began."""
        if not self.recovery or band is None:
            return False
        if self.switch_band is None:
            self.switch_band = band
            return False
        return band != self.switch_band

    def snapshot(self):
        return dict(active=self.active, stage=self.stage, horizontal_hold=False, recovery=self.recovery,
                    height_m=self.height,
                    heading_deg=None if self.heading is None else round(math.degrees(self.heading) % 360, 1))
