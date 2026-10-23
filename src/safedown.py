import math

G = 9.80665


class Handover(Exception):
    """SafeDown is over and every channel goes back to the pilot: landed, or
    the pilot or the FC chose a mode of its own."""


class SafeDown:
    """SafeDown: the phone's SAFEDOWN button or the pilot's switch to STABILIZE.

    The FC does the flying wherever it can, so a Jetson that stops sending
    still leaves a safe mode behind it:

    ARREST     Only when climbing or diving fast: ALT_HOLD would stop it at
               PILOT_ACCEL_Z (5 m/s^2), a dive at 20 m/s only after 40 m.
               STABILIZE levels instead while the app's throttle uses the
               whole thrust range: near zero against a climb (gravity
               brakes it), near full against a dive.
    SWITCHING  ALT_HOLD is requested.  Until the FC takes it, ACRO gets body
               rates that level the drone at the launch throttle.
    LEVEL      ALT_HOLD with centred sticks: the FC levels, stops the vertical
               motion where it is and holds the heading.
    STOP       Level and steady; the launch speed bleeds off in the drag.  The
               accelerometer's horizontal force is that drag, so it is small
               once the drone has (nearly) stopped relative to the air.
    DESCEND    LAND: the FC descends at LAND_SPEED, keeps level and the
               heading, and disarms on touchdown, all without the Jetson.

    Without GPS the FC's idea of level drifts after a hard launch or a
    tumble, and a drone that believes it is level slides off sideways.  The
    accelerometer's horizontal force is the air's drag, whatever the EKF
    thinks: the app leans against it (an air brake) until the drone is still
    relative to the air.  Wind still carries it; only GPS could stop that.

    The FC's height hold trusts the EKF.  A tumble can wreck its vertical
    velocity while the barometer still reads true; when the two disagree the
    app flies STABILIZE on the barometer's climb rate until they agree again.

    Fallbacks keep it autonomous when the FC refuses a mode.  LAND refused:
    the descent is flown with the ALT_HOLD throttle stick.  ALT_HOLD refused
    (no altitude estimate after a hard launch, say): STABILIZE levels and the
    throttle holds the climb rate; ALT_HOLD and LAND are retried throughout.
    """

    MODE_RETRY_S = 0.3
    ALT_HOLD_TIMEOUT_S = 1.5
    LAND_TIMEOUT_S = 2.0
    RETRY_FALLBACK_S = 1.0
    LEVEL_DEG = 5.0
    STEADY_DPS = 20.0
    STEADY_CLIMB_MPS = 0.5
    STOPPED_ACCEL = 1.0
    SETTLE_S = 1.0
    MIN_STOP_S = 2.0
    MAX_STOP_S = 8.0
    DESCENT_MPS = 0.5
    HIGH_DESCENT_MPS = 1.5
    HIGH_M = 10.0
    TOUCHDOWN_S = 1.0
    GROUND_HEIGHT_M = 2.0
    DISARM_RETRY_S = 1.0
    FORCE_DISARM_S = 3.0
    REST_BARO_MPS = 0.4
    REST_TAU_S = 1.0
    ACRO_LEVEL_GAIN = 4.0
    ACRO_LEVEL_MAX_DPS = 300.0
    ACCEL_TAU_S = 0.4
    ARREST_ENTER_MPS = 6.0
    ARREST_EXIT_MPS = 2.0
    ARREST_LEVEL_DEG = 20.0
    ARREST_TIMEOUT_S = 6.0
    EKF_BAD_MPS = 3.0
    BRAKE_KP = 4.0         # lean, degrees per m/s^2 of drag
    BRAKE_KI = 2.0         # degrees per m/s^2 per second: undoes a level bias
    BRAKE_MAX_DEG = 12.0
    BRAKE_DEADBAND = 0.15  # m/s^2: accelerometer bias and vibration
    BRAKE_TAU_S = 0.3
    EKF_BAD_S = 1.0
    BARO_WINDOW_S = 0.6  # the backend's baro_climb slope window
    EKF_GOOD_MPS = 1.5
    EKF_GOOD_S = 3.0
    HOLD_THROTTLE_S = 0.5
    HOVER_TAU_S = 0.5
    CLIMB_KP = 2.0
    CLIMB_KI = 2.0
    ON_GROUND = 1

    def __init__(self):
        self.started = None
        self.stop()

    @property
    def active(self):
        return self.started is not None

    def begin(self, now, ctx, reason, hold_throttle):
        self.started = self._t = now
        self.reason = reason
        self.hold_throttle = hold_throttle
        self.source_mode = ctx.get("fc_mode")
        self.switch_band = ctx.get("mode_switch")
        self.stage = "SWITCHING"
        self.fallback = None
        self._left_acro = False
        self._last_mode = None
        self.landed = False
        self._want_t = now
        self._req_t = -1e9
        self._req_mode = None
        self._acked = self._hb_at_ack = self._ack_t = None
        self._settle_t = self._stop_t = None
        self._accel = None
        self._ground_t = self._disarm_t = None
        self._descending = False
        self._resting_t = None
        self._rest_baro = self._rest_t = None
        self._climb_cmd = 0.0
        self._thrust = None
        self._thr_i = 0.0
        self._hover = None
        self._arrest_t = None
        self.ekf_bad = False
        self._vz_bad_t = self._vz_good_t = None
        self._ekf_hist = []
        self._drag = None
        self._brake_i = [0.0, 0.0]
        self.brake = (0.0, 0.0)
        self.notes = []

    def stop(self):
        self.started = None
        self.stage = "OFF"
        self.fallback = None
        self.reason = ""
        self.landed = False
        self.ekf_bad = False
        self.brake = (0.0, 0.0)
        self.notes = []

    def _note(self, text):
        self.notes.append(text)

    def take_notes(self):
        notes, self.notes = self.notes, []
        return notes

    # -- modes -------------------------------------------------------------

    def _request(self, mapper, now, mode):
        if self._req_mode != mode:
            self._want_t = now
        self._req_t, self._req_mode = now, mode
        if not mapper.request_mode(mode):
            raise ValueError(f"could not request {mode}")

    def _effective_mode(self, ctx):
        # The ACK comes long before the 1 Hz heartbeat shows the new mode.
        # Until the heartbeat moves off what it showed at the ACK, the sticks
        # already mean the ACKed mode; a heartbeat sent after the ACK is the
        # truth either way.
        hb, hb_t = ctx.get("fc_mode"), ctx.get("fc_mode_t")
        ack = ctx.get("mode_ack")
        if ack and ack[1] and ack[0] == self._req_mode and ack[2] != self._ack_t:
            self._acked, self._hb_at_ack, self._ack_t = ack[0], hb, ack[2]
        if (self._acked and hb == self._hb_at_ack and hb != self._acked
                and (hb_t is None or hb_t <= self._ack_t)):
            return self._acked
        self._acked = None
        return hb

    def _denied(self, ctx, mode):
        ack = ctx.get("mode_ack")
        return bool(ack and ack[0] == mode and self._req_mode == mode and not ack[1])

    def _pursue(self, mapper, now, ctx, mode, timeout):
        """Request `mode` until the FC is in it: True once it is, False once
        refused or out of time, None while waiting."""
        if self._effective_mode(ctx) == mode:
            return True
        if self._req_mode != mode:
            self._request(mapper, now, mode)
            return None
        if self._denied(ctx, mode) or now - self._want_t >= timeout:
            return False
        if now - self._req_t >= self.MODE_RETRY_S:
            self._req_t = now
            if not mapper.request_mode(mode):
                raise ValueError(f"could not request {mode}")
        return None

    # -- control -------------------------------------------------------------

    @staticmethod
    def _fresh(ctx):
        att, h, climb = ctx.get("att"), ctx.get("height"), ctx.get("climb")
        if att is None or h is None or climb is None:
            return False
        values = [h, climb] + [att.get(k, float("nan")) for k in ("roll", "pitch", "yaw")]
        return all(math.isfinite(v) for v in values)

    @staticmethod
    def _tilt(att):
        return math.acos(max(-1.0, min(1.0, math.cos(att["roll"])*math.cos(att["pitch"]))))

    def _acro_level_rates(self, att):
        # Body rates that turn the thrust axis back up: about the axis of
        # body z x gravity, at a rate proportional to the tilt.
        if att is None:
            return 0.0, 0.0, 0.0
        r, p = att["roll"], att["pitch"]
        gx, gy, gz = -math.sin(p), math.sin(r)*math.cos(p), math.cos(r)*math.cos(p)
        ax, ay = -gy, gx
        n = math.hypot(ax, ay)
        tilt = math.atan2(n, gz)
        if tilt < 1e-4:
            return 0.0, 0.0, 0.0
        if n < 1e-6:
            ax, ay, n = 1.0, 0.0, 1.0
        rate = min(math.radians(self.ACRO_LEVEL_MAX_DPS), self.ACRO_LEVEL_GAIN*tilt)
        return rate*ax/n, rate*ay/n, 0.0

    def _filter_accel(self, ctx, dt):
        a = ctx.get("accel")
        if a is None or not all(math.isfinite(v) for v in a[:2]):
            self._accel = self._drag = None
            return
        k = 1.0 if self._drag is None else min(1.0, dt/self.BRAKE_TAU_S)
        self._drag = ((a[0], a[1]) if self._drag is None else
                      tuple(d + k*(v-d) for d, v in zip(self._drag, a[:2])))
        h = math.hypot(a[0], a[1])
        k = 1.0 if self._accel is None else min(1.0, dt/self.ACCEL_TAU_S)
        self._accel = h if self._accel is None else self._accel + k*(h-self._accel)

    def _level(self, att, limit_deg):
        rates = [abs(att.get(k, 0.0)) for k in ("rollspeed", "pitchspeed")]
        return (math.degrees(self._tilt(att)) <= limit_deg
                and max(rates) <= math.radians(self.STEADY_DPS))

    def _stopped(self, ctx):
        return (abs(ctx["climb"]) <= self.STEADY_CLIMB_MPS
                and (self._accel is None or self._accel <= self.STOPPED_ACCEL))

    def _descent_rate(self, ctx):
        h = ctx.get("height")
        return self.HIGH_DESCENT_MPS if h is not None and h > self.HIGH_M else self.DESCENT_MPS

    @staticmethod
    def stabilize_stick(thrust, hover):
        # Inverse of ArduCopter's manual throttle curve, whose mid stick is
        # the hover throttle: out = in*(1-e) + e*in^3.
        e = max(-0.5, min(1.0, -(hover-0.5)/0.375))
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo+hi)/2
            if mid*(1-e) + e*mid**3 < thrust:
                lo = mid
            else:
                hi = mid
        return (lo+hi)/2

    def _manual_throttle(self, mapper, ctx, climb_target, dt):
        """Throttle stick fraction holding `climb_target` in ACRO / STABILIZE.

        The FC's hover throttle may be wrong for this payload, so the hover
        thrust is learnt from the IMU: the specific force along the thrust
        axis is thrust / mass."""
        curve = getattr(mapper, "hover_thrust", None) or 0.5
        if self._hover is None:
            self._hover = curve
        a = ctx.get("accel")
        if (self._thrust is not None and self._thrust > .05 and a is not None
                and math.isfinite(a[2]) and -a[2] > 2.0):
            seen = max(.05, min(.9, self._thrust*G/-a[2]))
            self._hover += min(1.0, dt/self.HOVER_TAU_S)*(seen-self._hover)
        if self._fresh(ctx):
            error = climb_target - ctx["climb"]
            self._thr_i = max(-G, min(G, self._thr_i + self.CLIMB_KI*error*dt))
            lo, hi = (-G, 3*G) if self.stage == "ARREST" else (-4.0, 4.0)
            accel = max(lo, min(hi, self.CLIMB_KP*error + self._thr_i))
            tilt = self._tilt(ctx["att"])
            # Past 60 degrees thrust pushes it sideways or down: keep just
            # enough for the FC's attitude control while it rights itself.
            lift = math.cos(min(tilt, math.radians(60)))
            self._thrust = self._hover*(1 + accel/G)/lift if tilt < math.radians(75) else .1
        elif self._thrust is None:
            self._thrust = self._hover
        self._thrust = max(.02 if self.stage == "ARREST" else .05, min(.95, self._thrust))
        return self.stabilize_stick(self._thrust, curve)

    def _climb(self, target, dt):
        self._climb_cmd += max(-dt, min(dt, target-self._climb_cmd))
        return self._climb_cmd

    # -- cycle -------------------------------------------------------------

    def _target(self):
        if self.stage == "ARREST":
            return "STABILIZE"
        if self.fallback:
            return self.fallback
        return "LAND" if self.stage == "DESCEND" else "ALT_HOLD"

    def _preferred(self):
        return "LAND" if self.stage == "DESCEND" else "ALT_HOLD"

    def _check_vertical(self, now, ctx):
        """ctx with the climb rate SafeDown can trust: the EKF's, or the
        barometer's once the two disagree."""
        ekf, baro = ctx.get("climb"), ctx.get("baro_climb")
        if ekf is not None and math.isfinite(ekf):
            # The barometer's climb is a slope over BARO_WINDOW_S: compare it
            # with the EKF's average over the same window, or a hard
            # deceleration alone looks like a broken EKF.
            self._ekf_hist.append((now, ekf))
            while self._ekf_hist[0][0] < now - self.BARO_WINDOW_S:
                self._ekf_hist.pop(0)
            ekf = sum(v for _, v in self._ekf_hist)/len(self._ekf_hist)
        if ekf is None or baro is None or not math.isfinite(ekf) or not math.isfinite(baro):
            self._vz_bad_t = self._vz_good_t = None
        elif not self.ekf_bad:
            if abs(ekf-baro) > self.EKF_BAD_MPS:
                self._vz_bad_t = self._vz_bad_t or now
                if now - self._vz_bad_t >= self.EKF_BAD_S:
                    self._note(f"EKF climb {ekf:+.1f} m/s but barometer {baro:+.1f} m/s:"
                               " STABILIZE, the app flies the height on the barometer")
                    self.ekf_bad, self._vz_good_t = True, None
                    if self.stage != "ARREST":
                        self.fallback, self._req_mode = "STABILIZE", None
            else:
                self._vz_bad_t = None
        else:
            if abs(ekf-baro) < self.EKF_GOOD_MPS:
                self._vz_good_t = self._vz_good_t or now
                if now - self._vz_good_t >= self.EKF_GOOD_S:
                    self._note("EKF climb agrees with the barometer again")
                    self.ekf_bad, self._req_t = False, -1e9
            else:
                self._vz_good_t = None
        if self.ekf_bad:
            ctx = dict(ctx, climb=baro)
        return ctx

    def output(self, mapper, now, ctx):
        dt = max(0.0, min(.1, now-self._t))
        self._t = now
        ctx = self._check_vertical(now, ctx)
        fresh = self._fresh(ctx)
        self._filter_accel(ctx, dt)
        mode = self._effective_mode(ctx)
        if mode == "LAND" and self.stage != "DESCEND":
            # A failsafe LAND (radio, battery, EKF) is where SafeDown is
            # heading anyway.
            self._note("FC switched to LAND itself: descending")
            self.stage, self.fallback = "DESCEND", None
        if mode not in ("ACRO", "STABILIZE", "ALT_HOLD", "LAND"):
            raise Handover(f"FC in {mode}")
        if mode == "ACRO" and self._left_acro:
            raise Handover("FC back in ACRO")
        # Out of ACRO for good only once a heartbeat says so.
        self._left_acro = self._left_acro or ctx.get("fc_mode") not in ("ACRO", None)
        self._arrest(now, ctx, fresh)
        if self.stage == "SWITCHING" and mode in ("ALT_HOLD", "LAND"):
            self.stage = "LEVEL"
        if self.stage in ("LEVEL", "STOP") and mode != "ACRO":
            self._level_and_stop(mapper, now, ctx, fresh)
        self._pursue_target(mapper, now, ctx, mode)
        mode = self._effective_mode(ctx)
        if self.stage == "DESCEND":
            self._watch_touchdown(mapper, now, ctx)
        self._last_mode = mode
        return self._sticks(mapper, ctx, mode, dt)

    def _arrest(self, now, ctx, fresh):
        if (self.stage == "SWITCHING" and self._arrest_t is None and fresh
                and abs(ctx["climb"]) > self.ARREST_ENTER_MPS):
            self._note(f"{'climbing' if ctx['climb'] > 0 else 'diving'} at {abs(ctx['climb']):.0f} m/s:"
                       " STABILIZE and full-range throttle to stop it first")
            self.stage, self._arrest_t, self._req_mode = "ARREST", now, None
        elif self.stage == "ARREST":
            stopped = (fresh and abs(ctx["climb"]) < self.ARREST_EXIT_MPS
                       and math.degrees(self._tilt(ctx["att"])) < self.ARREST_LEVEL_DEG)
            if stopped or not fresh or now - self._arrest_t > self.ARREST_TIMEOUT_S:
                self.stage, self._req_mode = "SWITCHING", None
                if self.ekf_bad:
                    self.fallback, self.stage = "STABILIZE", "LEVEL"

    def _pursue_target(self, mapper, now, ctx, mode):
        target = self._target()
        if self._last_mode == target and mode != target:
            # Lost the mode (the pilot's switch onto STABILIZE): ask afresh.
            self._req_mode = None
        if mode == target:
            preferred = self._preferred()
            if self.fallback and not self.ekf_bad and now - self._req_t >= self.RETRY_FALLBACK_S:
                # Keep offering the FC's own mode; a refusal changes nothing.
                self._req_t = now
                self._req_mode = preferred
                mapper.request_mode(preferred)
            return
        if self.fallback and mode == self._preferred() and not self.ekf_bad:
            self._note(f"{mode} accepted: the FC flies it again")
            self.fallback = None
            return
        got = self._pursue(mapper, now, ctx, target, self.LAND_TIMEOUT_S if target == "LAND"
                           else self.ALT_HOLD_TIMEOUT_S)
        if got is not False:
            return
        why = "refused" if self._denied(ctx, target) else "not confirmed"
        if self.stage == "ARREST":
            self._note(f"STABILIZE {why}: straight to ALT_HOLD")
            self.stage, self._req_mode = "SWITCHING", None
            self._arrest_t = -1e9
            return
        if target == "STABILIZE":
            raise ValueError(f"FC {why} ALT_HOLD and STABILIZE")
        if target == "LAND" and mode == "ALT_HOLD":
            self.fallback = "ALT_HOLD"
            self._note(f"LAND {why}: the app descends at {self.DESCENT_MPS:g} m/s in ALT_HOLD")
        else:
            self.fallback = "STABILIZE"
            self._note(f"{target} {why}: STABILIZE levels, the app flies the throttle")
        self._req_mode = None
        if self.stage == "SWITCHING":
            self.stage = "LEVEL"
        if mode != self.fallback:
            self._pursue(mapper, now, ctx, self.fallback, self.ALT_HOLD_TIMEOUT_S)

    def _level_and_stop(self, mapper, now, ctx, fresh):
        if self._stop_t is None:
            self._stop_t = now
        if fresh:
            # STOP once level and steady; back to LEVEL only well past it.
            if self._level(ctx["att"], self.LEVEL_DEG):
                self.stage = "STOP"
            elif not self._level(ctx["att"], 2*self.LEVEL_DEG):
                self.stage = "LEVEL"
        if fresh and self.stage == "STOP" and self._level(ctx["att"], self.LEVEL_DEG) and self._stopped(ctx):
            self._settle_t = self._settle_t or now
        else:
            self._settle_t = None
        held = now - self._stop_t
        settled = self._settle_t is not None and now - self._settle_t >= self.SETTLE_S
        if (settled and held >= self.MIN_STOP_S) or held >= self.MAX_STOP_S:
            self._note("level and stopped: LAND" if settled
                       else f"not settled after {self.MAX_STOP_S:g} s: LAND anyway")
            self.stage = "DESCEND"
            if self.fallback == "STABILIZE":
                self._req_t = -1e9
            else:
                self.fallback = None
                self._req_mode = None

    def _watch_touchdown(self, mapper, now, ctx):
        state = ctx.get("landed_state")
        on_ground = state == self.ON_GROUND
        if self.fallback in ("STABILIZE", "ALT_HOLD") and self._fresh(ctx):
            # The app flies the descent, and the FC's land detector may never
            # fire in STABILIZE: low, commanding a descent, and it was
            # descending but has stopped means the legs are down.
            if ctx["climb"] < -.25:
                self._descending = True
            on_ground = on_ground or (self._descending and self._climb_cmd < -.3 and ctx["climb"] > -.1
                                      and ctx["height"] < self.GROUND_HEIGHT_M)
        if not on_ground:
            self._ground_t = None
            return
        self._ground_t = self._ground_t or now
        if now - self._ground_t < self.TOUCHDOWN_S:
            return
        if not self.landed:
            self._note("touchdown")
            self._resting_t = None
        self.landed = True
        if self._disarm_t is None or now - self._disarm_t >= self.DISARM_RETRY_S:
            # LAND disarms by itself; this covers the fallbacks.  The FC
            # refuses a GCS disarm unless it too thinks it has landed, which
            # a broken EKF can keep it from ever doing: then, on the app's
            # own proof that the ground holds it, the disarm is forced.
            self._disarm_t = now
            force = self._resting_t is not None and now - self._resting_t >= self.FORCE_DISARM_S
            if force:
                self._note("on the ground but the FC will not disarm: forcing it")
            mapper.request_disarm(force=force)
        if self._resting(ctx, now):
            self._resting_t = self._resting_t or now
        else:
            self._resting_t = None

    def _resting(self, ctx, now):
        # Throttle at the bottom yet the IMU feels a full 1 g, and the
        # barometer is still (smoothed: its slope alone jitters by
        # +-0.5 m/s at rest): only the ground can be holding it up.
        a, baro = ctx.get("accel"), ctx.get("baro_climb")
        if baro is None:
            self._rest_baro = None
            return False
        k = 1.0 if self._rest_baro is None else min(1.0, (now - self._rest_t)/self.REST_TAU_S)
        self._rest_baro = baro if self._rest_baro is None else self._rest_baro + k*(baro - self._rest_baro)
        self._rest_t = now
        return (self.fallback is not None and a is not None
                and abs(math.sqrt(a[0]**2 + a[1]**2 + a[2]**2) - G) < 1.5
                and abs(self._rest_baro) < self.REST_BARO_MPS and ctx.get("height") is not None
                and ctx["height"] < self.GROUND_HEIGHT_M)

    def _air_brake(self, mapper, ctx, mode, dt):
        """(roll, pitch) degrees that lean against the drag, or None."""
        att = ctx.get("att")
        # With a GPS position the FC holds it itself, and LAND would take our
        # sticks as a request to move.
        if (ctx.get("position_ok") or mode not in ("STABILIZE", "ALT_HOLD", "LAND")
                or self.stage not in ("LEVEL", "STOP", "DESCEND")
                or self.landed or self._drag is None or att is None
                or math.degrees(self._tilt(att)) > 2*self.ARREST_LEVEL_DEG
                or not callable(getattr(mapper, "lean_pwm", None))):
            return None
        out = []
        # Drag forward (+x) means it moves back through the air: nose down
        # to brake, so pitch (nose up positive) follows -x; roll follows +y.
        for i, err in enumerate((self._drag[1], -self._drag[0])):
            err = math.copysign(max(0.0, abs(err)-self.BRAKE_DEADBAND), err)
            self._brake_i[i] = max(-self.BRAKE_MAX_DEG, min(self.BRAKE_MAX_DEG,
                                   self._brake_i[i] + self.BRAKE_KI*err*dt))
            out.append(max(-self.BRAKE_MAX_DEG, min(self.BRAKE_MAX_DEG,
                                                    self.BRAKE_KP*err + self._brake_i[i])))
        return tuple(out)

    def _sticks(self, mapper, ctx, mode, dt):
        out = self._sticks_core(mapper, ctx, mode, dt)
        brake = self._air_brake(mapper, ctx, mode, dt) if out else None
        self.brake = brake or (0.0, 0.0)
        if brake:
            out = dict(out, **mapper.lean_pwm(*brake))
        return out

    def _sticks_core(self, mapper, ctx, mode, dt):
        if mode in ("ACRO", "STABILIZE"):
            att = ctx.get("att")
            rates = self._acro_level_rates(att) if mode == "ACRO" else (0.0, 0.0, 0.0)
            if self.landed:
                return mapper.wasp_pwm(rates, 0.0)
            if (self.stage == "SWITCHING" and self.hold_throttle is not None
                    and self._t - self.started < self.HOLD_THROTTLE_S
                    and (att is None or self._tilt(att) <= math.radians(60))):
                # Mid stick is not hover in ACRO / STABILIZE: bridge the moment
                # until ALT_HOLD with the launch throttle and level sticks.
                self._thrust = None
                return mapper.wasp_pwm(rates, self.hold_throttle)
            target = -self._descent_rate(ctx) if self.stage == "DESCEND" else 0.0
            self._climb_cmd = target
            return mapper.wasp_pwm(rates, self._manual_throttle(mapper, ctx, target, dt))
        self._thrust = None
        if mode == "ALT_HOLD" and self.stage == "DESCEND":
            if self.landed:
                return mapper.idle_pwm(0.0, -1e3)
            return mapper.idle_pwm(0.0, self._climb(-self._descent_rate(ctx), dt))
        # ALT_HOLD levelling / stopping and LAND: centred sticks.  The FC
        # levels, holds the heading and its own height or descent.
        self._climb_cmd = 0.0
        return mapper.idle_pwm(0.0, 0.0)

    def switch_moved(self, band):
        """The pilot's mode switch moved since the last call (None if unknown)."""
        if band is None:
            return False
        if self.switch_band is None or band == self.switch_band:
            self.switch_band = band
            return False
        self.switch_band = band
        return True

    def reassert(self):
        # The pilot moved the switch onto STABILIZE: the FC is (or will be)
        # in STABILIZE, so ask for the current goal again from there.
        self._acked = None
        self._req_mode = None

    def snapshot(self):
        return dict(active=self.active, stage=self.stage, ekf_bad=self.ekf_bad,
                    brake_deg=[round(v, 1) for v in self.brake],
                    fallback=self.fallback, landed=self.landed, reason=self.reason)
