import math

G = 9.80665


class Handover(Exception):
    """SafeDown is over and every channel goes back to the pilot: landed, or
    the pilot or the FC chose a mode of its own."""


class SafeDown:
    """SafeDown: the phone's SAFEDOWN button or the pilot's switch up to ALT_HOLD.

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
    RETURN     With a GPS position and a home: RTL.  The FC brakes, flies back
               to where it was armed on the stand (RTL_ALT at least), lands
               there and disarms, all without the Jetson.
    DESCEND    No GPS, or RTL refused or lost: LAND where it is.  The FC
               descends at LAND_SPEED, keeps level and the heading, and
               disarms on touchdown, all without the Jetson.

    Without GPS the FC's idea of level drifts after a hard launch or a
    tumble, and a drone that believes it is level slides off sideways.  The
    accelerometer's horizontal force is the air's drag, whatever the EKF
    thinks: the app leans against it (an air brake) until the drone is still
    relative to the air.  Wind still carries it; only GPS could stop that.

    The FC's height hold trusts the EKF.  A tumble can wreck its vertical
    velocity while the barometer still reads true; when the two disagree the
    app flies STABILIZE on the barometer's climb rate until they agree again.

    CATCH (the pilot's switch to STABILIZE) is the first part on its own: in
    STABILIZE, with no mode changes, the app levels it from any attitude,
    holds the climb on the barometer and brakes the ground speed until the
    drone is hovering. Every channel then goes to the pilot, still in
    STABILIZE. Nothing gives the sticks back on a timer while it is moving.

    Fallbacks keep it autonomous when the FC refuses a mode.  LAND refused:
    the descent is flown with the ALT_HOLD throttle stick.  ALT_HOLD refused
    (no altitude estimate after a hard launch, say): STABILIZE levels and the
    throttle holds the climb rate; ALT_HOLD and LAND are retried throughout.
    """

    MODE_RETRY_S = 0.3
    ALT_HOLD_TIMEOUT_S = 1.5
    LAND_TIMEOUT_S = 2.0
    RETRY_FALLBACK_S = 1.0
    # Relaxed for windy conditions: the earlier tight limits made the drone
    # stall for minutes in a steady breeze waiting for "settled" before RTL.
    # RTL and LAND both brake horizontally on their own, so we hand off as
    # soon as the attitude is roughly level.
    LEVEL_DEG = 10.0
    STEADY_DPS = 45.0
    STEADY_CLIMB_MPS = 2.0
    STOPPED_ACCEL = 2.5
    SETTLE_S = 0.5
    MIN_STOP_S = 1.0
    MAX_STOP_S = 5.0
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
    SWITCH_MODE = "ALT_HOLD"   # the pilot's switch position that means SafeDown
    CATCH_MODE = "STABILIZE"   # the one that means: steady it, then it is mine
    CATCH_MIN_S = 1.0
    CATCH_MAX_S = 5.0          # hard limit: the pilot gets the sticks after this
    CATCH_HOVER_S = 1.0        # inside the hover band this long, then the pilot
    CATCH_LEVEL_DEG = 8.0
    CATCH_CLIMB_MPS = 1.5      # EKF climb band when the barometer is absent
    CATCH_BARO_MPS = 0.20      # smoothed barometer climb band
    CATCH_EKF_MPS = 0.25       # EKF climb band once the barometer is present
    CATCH_HOLD_M = 0.8         # baro height must sit on the captured hover
    CATCH_STOP_MPS = 0.6       # ground speed counted as stopped
    CATCH_BRAKE_MAX_DEG = 30.0 # under ANGLE_MAX (40 deg): stops a fast run
    CATCH_BRAKE_TAU_S = 1.2
    CATCH_BRAKE_FADE_MPS = 3.0 # lean fades out across this band above the stop
    CATCH_BRAKE_RATE_DPS = 60.0
    CATCH_HEARTBEAT_LAG_S = 1.5
    RETURN_CLIMB_MPS = 2.0     # vertical speed RTL may start from
    POSITION_LOST_S = 2.0
    RETURN_MIN_H = 5.0        # a position found on the way down still flies it home above this

    def __init__(self):
        self.started = None
        self.stop()

    @property
    def active(self):
        return self.started is not None

    def begin(self, now, ctx, reason, hold_throttle, catch=False,
              direct_return=False, land_only=False):
        self.started = self._t = now
        self.catch = catch
        self._steady_t = None
        self.reason = reason
        self.hold_throttle = hold_throttle
        self.source_mode = ctx.get("fc_mode")
        self.switch_band = ctx.get("mode_switch")
        # land_only still runs SWITCHING -> LEVEL -> STOP (so the drone is
        # stabilised before descent), then DESCEND.  _no_return (set below)
        # blocks the usual STOP -> RETURN transition, so no GPS is required
        # and no RTL ever attempts.
        if land_only:
            self.stage = "SWITCHING"
        elif direct_return:
            self.stage = "RETURN"
        else:
            self.stage = "SWITCHING"
        self._need_set_home = direct_return and not land_only
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
        self._catch_baro = self._catch_baro_t = None
        self._catch_lean = [0.0, 0.0]
        self._catch_src = None
        self._catch_hold_h = None
        self._hover_set_t = None
        self._hover_anchor = None
        self.home_m = None
        self._no_return = bool(land_only)
        self._lost_t = None
        self.notes = []

    def stop(self):
        self.started = None
        self.catch = False
        self.stage = "OFF"
        self.fallback = None
        self.reason = ""
        self.landed = False
        self.ekf_bad = False
        self.brake = (0.0, 0.0)
        self.home_m = None
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

    def _manual_throttle(self, mapper, ctx, climb_target, dt, full=False,
                         accel_limit=None, kp=None, ki=None, i_limit=None,
                         freeze_i=False, thrust_slew=None):
        """Throttle stick fraction holding `climb_target` in ACRO / STABILIZE.

        The FC's hover throttle may be wrong for this payload, so the hover
        thrust is learnt from the IMU: the specific force along the thrust
        axis is thrust / mass. The extra arguments are the catch hover; left
        unset, SafeDown's own loop is unchanged."""
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
            ki_v = self.CLIMB_KI if ki is None else ki
            lim = G if i_limit is None else i_limit
            if not freeze_i:
                self._thr_i = max(-lim, min(lim, self._thr_i + ki_v*error*dt))
            kp_v = self.CLIMB_KP if kp is None else kp
            if accel_limit is None:
                full = full or self.stage == "ARREST"
                lo, hi = (-G, 3*G) if full else (-4.0, 4.0)
            else:
                lo, hi = -float(accel_limit), float(accel_limit)
                full = True
            accel = max(lo, min(hi, kp_v*error + self._thr_i))
            tilt = self._tilt(ctx["att"])
            # Past 60 degrees thrust pushes it sideways or down: keep just
            # enough for the FC's attitude control while it rights itself.
            lift = math.cos(min(tilt, math.radians(60)))
            desired = self._hover*(1 + accel/G)/lift if tilt < math.radians(75) else .1
            if thrust_slew is not None and self._thrust is not None:
                step = max(0.0, thrust_slew)*dt
                desired = max(self._thrust-step, min(self._thrust+step, desired))
            self._thrust = desired
        elif self._thrust is None:
            self._thrust = self._hover
        self._thrust = max(.02 if full else .05, min(.95, self._thrust))
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
        return self._preferred()

    def _preferred(self):
        return {"DESCEND": "LAND", "RETURN": "RTL"}.get(self.stage, "ALT_HOLD")

    def _can_return(self, ctx):
        return bool(ctx.get("position_ok") and ctx.get("home_ok")) and not self._no_return

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
                    # A catch already flies the barometer. Do not leave a
                    # STABILIZE fallback behind for a later return home.
                    if self.stage != "ARREST" and not self.catch:
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
        if self.catch:
            return self._catch(mapper, now, ctx, fresh, dt)
        mode = self._effective_mode(ctx)
        self.home_m = ctx.get("home_m")
        if mode == "LAND" and self.stage != "DESCEND":
            # A failsafe LAND (radio, battery, EKF) is where SafeDown is
            # heading anyway.
            self._note("FC switched to LAND itself: descending")
            # The FC's own failsafe: never fight it with an RTL.
            self.stage, self.fallback, self._no_return = "DESCEND", None, True
        if mode not in ("ACRO", "STABILIZE", "ALT_HOLD", "LAND", "RTL"):
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
        if self._need_set_home and self.stage == "RETURN":
            self._need_set_home = False
            set_home = getattr(mapper, "set_home_to_launch", None)
            if callable(set_home):
                set_home()
            where = f" ({ctx['home_m']:.0f} m)" if ctx.get("home_m") is not None else ""
            self._note(f"RTL, flying back to the launch point{where}")
        if self.stage == "RETURN":
            self._watch_position(now, ctx)
        elif (self.stage == "DESCEND" and self.fallback is None and not self.landed and fresh
              and self._can_return(ctx) and ctx["height"] > self.RETURN_MIN_H):
            # No compass: the EKF found its heading only on the way down.
            self._start_return(mapper, ctx, "GPS position found on the way down: RTL")
        self._pursue_target(mapper, now, ctx, mode)
        mode = self._effective_mode(ctx)
        if self.stage in ("DESCEND", "RETURN"):
            self._watch_touchdown(mapper, now, ctx)
        self._last_mode = mode
        return self._sticks(mapper, ctx, mode, dt)

    def _catch_vertical(self, ctx, now):
        """(climb m/s, barometer is the source). The slope jitters at rest,
        so a present barometer is smoothed and is the authority even when the
        EKF climb says something else."""
        baro = ctx.get("baro_climb")
        if baro is None or not math.isfinite(baro):
            self._catch_baro = self._catch_baro_t = None
            climb = ctx.get("climb")
            return (0.0 if climb is None or not math.isfinite(climb) else climb), False
        k = 1.0 if self._catch_baro is None else min(1.0, (now - (self._catch_baro_t or now))/self.REST_TAU_S)
        self._catch_baro = baro if self._catch_baro is None else self._catch_baro + k*(baro - self._catch_baro)
        self._catch_baro_t = now
        return self._catch_baro, True

    def _catch_rate(self, ctx):
        """Climb the throttle follows. The EKF rate has little lag, so it
        stops a climb without overshooting into a dive. The barometer takes
        over when the two disagree, and it is what the hover gate trusts."""
        baro = ctx.get("baro_climb")
        ekf = ctx.get("climb")
        baro_ok = baro is not None and math.isfinite(baro)
        ekf_ok = ekf is not None and math.isfinite(ekf)
        if baro_ok and ekf_ok and abs(baro - ekf) > 3.0:
            return baro
        if ekf_ok:
            return ekf
        if baro_ok:
            return baro
        return 0.0

    def _catch_climb_target(self, ctx, rate, baro_rate=None):
        """Zero until the climb has actually stopped, then hold the height.

        The barometer altitude trails a fast climb. Locking it at the first
        moment the EKF is under 0.8 m/s stores a point below the crest, and
        the hold then descends back to that point. The lock waits until the
        smoothed barometer and the EKF are both slow."""
        ekf = ctx.get("climb")
        ekf_v = ekf if ekf is not None and math.isfinite(ekf) else rate
        baro_slow = baro_rate is None or abs(baro_rate) <= self.CATCH_EKF_MPS
        if (self._catch_hold_h is None and baro_slow
                and abs(rate) <= self.CATCH_EKF_MPS and abs(ekf_v) <= self.CATCH_EKF_MPS):
            h = ctx.get("height")
            if h is not None and math.isfinite(h):
                self._catch_hold_h = h
        if self._catch_hold_h is None:
            return 0.0
        h = ctx.get("height")
        if h is None or not math.isfinite(h):
            return 0.0
        return max(-0.8, min(0.8, 0.35*(self._catch_hold_h - h)))

    def _catch_hovering(self, ctx, fresh, vz, have_baro):
        if not fresh or self._catch_hold_h is None or ctx.get("att") is None:
            return False
        if not self._level(ctx["att"], self.CATCH_LEVEL_DEG):
            return False
        h = ctx.get("height")
        if h is None or abs(h - self._catch_hold_h) > self.CATCH_HOLD_M:
            return False
        if abs(vz) > (self.CATCH_BARO_MPS if have_baro else self.CATCH_CLIMB_MPS):
            return False
        ekf = ctx.get("climb")
        ekf_lim = self.CATCH_EKF_MPS if have_baro else self.CATCH_CLIMB_MPS
        if ekf is not None and math.isfinite(ekf) and abs(ekf) > ekf_lim:
            return False
        # Level and the height is held: the pilot can fly it. Ground speed is
        # not checked — wind at altitude keeps it moving in STABILIZE, and
        # waiting for it to stop left the pilot without sticks for minutes.
        return True

    def _catch_speed_lean(self, ctx):
        """(roll, pitch) degrees that brake the GPS ground speed, or None.

        Positive pitch is nose up and brakes a forward run. Positive roll
        lowers the right side and brakes a leftward run."""
        vel, att = ctx.get("vel_ned"), ctx.get("att")
        if not ctx.get("position_ok") or vel is None or att is None or len(vel) < 2:
            return None
        if not all(math.isfinite(v) for v in vel[:2]):
            return None
        yaw = att.get("yaw", 0.0)
        if yaw is None or not math.isfinite(yaw):
            return None
        vn, ve = vel[0], vel[1]
        vf = vn*math.cos(yaw) + ve*math.sin(yaw)
        vr = -vn*math.sin(yaw) + ve*math.cos(yaw)
        speed = math.hypot(vf, vr)
        fade = 0.0 if speed <= self.CATCH_STOP_MPS else min(
            1.0, (speed - self.CATCH_STOP_MPS)/self.CATCH_BRAKE_FADE_MPS)
        af = fade*(-vf/self.CATCH_BRAKE_TAU_S)
        ar = fade*(-vr/self.CATCH_BRAKE_TAU_S)
        return (math.degrees(math.atan2(ar, G)), math.degrees(math.atan2(-af, G)))

    def _slew_pair(self, cur, target, dt, cap, rate):
        out = []
        for i, want in enumerate(target):
            want = max(-cap, min(cap, want))
            step = max(-rate*dt, min(rate*dt, want - cur[i]))
            cur[i] = cur[i] + step
            out.append(cur[i])
        return tuple(out)

    def _catch_brake(self, mapper, ctx, mode, dt):
        """Lean that replaces the level sticks, or None to stay centred.

        Angle sticks in ACRO would be read as a rate, so the brake waits
        until the FC is in STABILIZE."""
        if mode != "STABILIZE" or not callable(getattr(mapper, "lean_pwm", None)):
            self._catch_lean = [0.0, 0.0]
            return None
        target = self._catch_speed_lean(ctx)
        if target is None:
            dragged = self._air_brake(mapper, ctx, mode, dt, max_deg=self.CATCH_BRAKE_MAX_DEG)
            target = dragged if dragged is not None else (0.0, 0.0)
        lean = self._slew_pair(self._catch_lean, target, dt, self.CATCH_BRAKE_MAX_DEG,
                               self.CATCH_BRAKE_RATE_DPS)
        return None if max(abs(v) for v in lean) < 0.2 else lean

    # ArduCopter refuses a hover throttle outside this band.
    HOVER_MIN, HOVER_MAX = 0.125, 0.6875

    def _hover_stick_ready(self, mapper, now):
        """True once mid stick is the thrust that is holding this weight."""
        raw = self._hover
        if raw is None or not math.isfinite(raw):
            self._hover_anchor = None
            return False
        learned = min(self.HOVER_MAX, max(self.HOVER_MIN, raw))
        # The IMU average is still moving just after the stop. Wait until it
        # holds still, or the stick is set to a thrust this weight has already
        # left behind.
        anchor = self._hover_anchor
        if anchor is None or abs(learned - anchor[1]) > 0.005:
            self._hover_anchor = (now, learned)
            return False
        if now - anchor[0] < 0.5:
            return False
        current = getattr(mapper, "hover_thrust", None)
        if current is not None and abs(float(current) - learned) <= 0.01:
            return True
        setter = getattr(mapper, "set_hover_thrust", None)
        if not callable(setter):
            return True
        if self._hover_set_t is None or now - self._hover_set_t >= 0.5:
            if setter(learned):
                first = self._hover_set_t is None
                self._hover_set_t = now
                if first:
                    self._note(f"this weight hovers at {learned:.1%} thrust: middle of the stick set there")
        return False

    def _catch(self, mapper, now, ctx, fresh, dt):
        # The pilot's STABILIZE, never a mode request: level it, hold height,
        # then hand over.  The levelling and braking are the same as before; what
        # changed is that hovering no longer waits for the ground speed to drop
        # (wind at altitude keeps it moving in STABILIZE) or for the FC to echo
        # the hover-thrust parameter.
        mode = ctx.get("fc_mode")
        if mode not in ("STABILIZE", "ACRO"):
            # The switch moved it to STABILIZE; the 1 Hz heartbeat may still
            # show the mode before for a moment.
            if mode != self.source_mode or now - self.started > self.CATCH_HEARTBEAT_LAG_S:
                raise Handover(f"FC in {mode}")
            mode = "STABILIZE"
        self.stage = "CATCH"
        vz, have_baro = self._catch_vertical(ctx, now)
        src = "speed" if self._catch_speed_lean(ctx) is not None else "drag"
        if src != self._catch_src:
            self._catch_src = src
            self._note("CATCH: braking to a hover on the "
                       + ("GPS speed" if src == "speed" else "air drag")
                       + " and the " + ("barometer" if have_baro else "EKF climb"))
        held = now - self.started
        hovering = self._catch_hovering(ctx, fresh, vz, have_baro)
        self._steady_t = (self._steady_t or now) if hovering else None
        # Best effort: write the learned hover thrust so mid stick hovers, but
        # do not block the handover waiting for the FC to echo it back.
        if hovering:
            self._hover_stick_ready(mapper, now)
        if held >= self.CATCH_MAX_S:
            self._note(f"CATCH: {self.CATCH_MAX_S:g} s limit reached, handing over")
            raise Handover("time limit: the pilot flies it in STABILIZE")
        if (held >= self.CATCH_MIN_S and self._steady_t is not None
                and now - self._steady_t >= self.CATCH_HOVER_S):
            raise Handover(
                "hovering: level, barometer steady; the pilot flies it in STABILIZE"
                if have_baro else
                "hovering: level and steady; the pilot flies it in STABILIZE")
        rates = self._acro_level_rates(ctx.get("att")) if mode == "ACRO" else (0.0, 0.0, 0.0)
        rate = self._catch_rate(ctx)
        hard = abs(rate) > 6.0
        target = 0.0 if hard else self._catch_climb_target(
            ctx, rate, vz if have_baro else None)
        self._climb_cmd = target
        # Full brake only while it is still fast. The integrator stays frozen
        # until the height is captured, so it cannot wind up during the stop
        # and then push the drone through the hover.
        out = mapper.wasp_pwm(rates, self._manual_throttle(
            mapper, dict(ctx, climb=rate), target, dt,
            accel_limit=25.0 if hard else 2.5, kp=2.0 if hard else 0.8, ki=0.4,
            i_limit=1.2, freeze_i=hard or self._catch_hold_h is None,
            thrust_slew=0.8))
        brake = self._catch_brake(mapper, ctx, mode, dt)
        self.brake = brake or (0.0, 0.0)
        return dict(out, **mapper.lean_pwm(*brake)) if brake else out

    def to_safedown(self, now):
        """The pilot's switch went on up from a catch: RTL directly."""
        self.catch, self.stage, self._req_mode = False, "RETURN", None
        self._need_set_home = True
        self._want_t = now

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
        if target == "RTL":
            # No way home: stop and land where it is, as without GPS.
            self._note(f"RTL {why}: stopping, then LAND here")
            self._no_return, self.stage, self._req_mode = True, "STOP", None
            return
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
        if (self.stage == "STOP" and fresh and self._can_return(ctx)
                and abs(ctx["climb"]) <= self.RETURN_CLIMB_MPS and self.fallback is None):
            # Level and no longer shooting up or down: RTL brakes the
            # horizontal speed itself and flies home.
            self._start_return(mapper, ctx, "level: RTL")
            return
        held = now - self._stop_t
        settled = self._settle_t is not None and now - self._settle_t >= self.SETTLE_S
        if (settled and held >= self.MIN_STOP_S) or held >= self.MAX_STOP_S:
            # Prefer RTL when GPS + home are available even if not fully
            # settled: RTL's own brake handles residual horizontal speed, so
            # we should fly home rather than land in place (likely still
            # outside a breached fence) just because wind kept the attitude
            # jittery.  Fall back to LAND only when RTL is impossible.
            if self._can_return(ctx) and self.fallback is None:
                self._start_return(mapper, ctx,
                    "level: RTL" if settled
                    else f"not settled after {self.MAX_STOP_S:g} s: RTL anyway")
                return
            self._note("level and stopped: LAND" if settled
                       else f"not settled after {self.MAX_STOP_S:g} s: LAND anyway")
            self.stage = "DESCEND"
            if self.fallback == "STABILIZE":
                self._req_t = -1e9
            else:
                self.fallback = None
                self._req_mode = None

    def _start_return(self, mapper, ctx, why):
        # Home becomes the GPS spot kept at GO, so RTL lands exactly there.
        set_home = getattr(mapper, "set_home_to_launch", None)
        if callable(set_home):
            set_home()
        self._note(f"{why}, flying back to the launch point"
                   + (f" ({ctx['home_m']:.0f} m)" if ctx.get("home_m") is not None else ""))
        self.stage, self._req_mode = "RETURN", None

    def _watch_position(self, now, ctx):
        # The FC's EKF failsafe lands on its own when the position goes; this
        # does the same if the GPS fades without tripping it.
        if ctx.get("position_ok"):
            self._lost_t = None
            return
        self._lost_t = self._lost_t or now
        if now - self._lost_t >= self.POSITION_LOST_S:
            self._note("GPS position lost: LAND here")
            self._no_return, self.stage, self._req_mode = True, "DESCEND", None

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

    def _air_brake(self, mapper, ctx, mode, dt, max_deg=None):
        """(roll, pitch) degrees that lean against the drag, or None."""
        att = ctx.get("att")
        cap = self.BRAKE_MAX_DEG if max_deg is None else max_deg
        # With a GPS position the FC holds it itself, and LAND would take our
        # sticks as a request to move.
        if ((ctx.get("position_ok") and mode != "STABILIZE") or mode not in ("STABILIZE", "ALT_HOLD", "LAND")
                or self.stage not in ("LEVEL", "STOP", "DESCEND", "CATCH")
                or self.landed or self._drag is None or att is None
                or math.degrees(self._tilt(att)) > 2*self.ARREST_LEVEL_DEG
                or not callable(getattr(mapper, "lean_pwm", None))):
            return None
        out = []
        # Drag forward (+x) means it moves back through the air: nose down
        # to brake, so pitch (nose up positive) follows -x; roll follows +y.
        for i, err in enumerate((self._drag[1], -self._drag[0])):
            err = math.copysign(max(0.0, abs(err)-self.BRAKE_DEADBAND), err)
            self._brake_i[i] = max(-cap, min(cap, self._brake_i[i] + self.BRAKE_KI*err*dt))
            out.append(max(-cap, min(cap, self.BRAKE_KP*err + self._brake_i[i])))
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
        # The pilot moved the switch onto SafeDown's position: the FC is (or
        # will be) in that mode, so ask for the current goal again from there.
        self._acked = None
        self._req_mode = None

    def snapshot(self):
        return dict(active=self.active, stage=self.stage, ekf_bad=self.ekf_bad, catch=self.catch,
                    brake_deg=[round(v, 1) for v in self.brake],
                    fallback=self.fallback, landed=self.landed, reason=self.reason,
                    home_m=None if self.home_m is None else round(self.home_m))
