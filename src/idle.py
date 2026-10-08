"""Barometer/IMU recovery: level, hold yaw and altitude; no XY braking/hold."""
import math


class Idle:
    MODE_TIMEOUT_S = 2.0
    RISE_M = 1.5
    MAX_CLIMB_MPS = 1.0
    MAX_YAW_DPS = 15.0

    def __init__(self):
        self.started = None
        self.confirmed = False
        self.stage = "OFF"
        self.height = self.heading = None
        self.source_mode = None
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

    def begin(self, now, ctx, ceiling):
        why = self.telemetry_error(ctx)
        if why:
            raise ValueError(why)
        self.source_mode = ctx["fc_mode"]
        if self.source_mode not in ("ACRO", "ALT_HOLD"):
            raise ValueError("pilot mode: Idle must not take control")
        self.height = max(ctx["height"], min(ctx["height"]+self.RISE_M, ceiling))
        self.heading = ctx["att"]["yaw"]
        self.started = self._t = now
        self.confirmed = self.source_mode == "ALT_HOLD"
        self._climb = self._yaw = 0.0
        self.stage = "LEVEL" if self.confirmed else "SWITCHING"

    def stop(self):
        self.started = None
        self.stage = "OFF"

    def output(self, mapper, now, ctx):
        why = self.telemetry_error(ctx)
        if why:
            raise ValueError(why)
        mode = ctx["fc_mode"]
        if ctx.get("idle_mode_denied"):
            raise ValueError("FC rejected ALT_HOLD for Idle")
        if mode != "ALT_HOLD":
            if self.confirmed or mode != self.source_mode:
                raise ValueError("pilot changed mode")
            if now-self.started >= self.MODE_TIMEOUT_S:
                raise ValueError("ALT_HOLD was not confirmed")
            # Use neutral throttle, never the previous full-throttle launch.
            # While mode confirmation is pending, request no body rotation.
            self.stage = "SWITCHING"
            self._t = now
            return mapper.idle_pwm(0.0, 0.0)
        self.confirmed = True
        dt = max(0.0, min(.1, now-self._t))
        self._t = now
        att = ctx["att"]
        level = max(abs(att["roll"]), abs(att["pitch"])) <= math.radians(10)
        error = self.height-ctx["height"]
        want_climb = max(-self.MAX_CLIMB_MPS, min(self.MAX_CLIMB_MPS, .8*error)) if level else 0.0
        yaw_error = math.degrees((self.heading-att["yaw"]+math.pi) % (2*math.pi)-math.pi)
        want_yaw = max(-self.MAX_YAW_DPS, min(self.MAX_YAW_DPS, 2*yaw_error))
        self._climb += max(-dt, min(dt, want_climb-self._climb))
        self._yaw += max(-30*dt, min(30*dt, want_yaw-self._yaw))
        self.stage = ("LEVEL" if not level else "HOLD"
                      if abs(error) <= .2 and abs(ctx["climb"]) <= .2 else "CLIMB")
        return mapper.idle_pwm(self._yaw, self._climb)

    def snapshot(self):
        return dict(active=self.active, stage=self.stage, horizontal_hold=False,
                    height_m=self.height,
                    heading_deg=None if self.heading is None else round(math.degrees(self.heading) % 360, 1))
