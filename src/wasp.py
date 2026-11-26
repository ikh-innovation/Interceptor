




import math


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    if not math.isfinite(n) or n < 1e-9:
        raise ValueError("invalid bearing")
    return tuple(x / n for x in v)


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def rotation(att):
    if "rotation" in att:
        return att["rotation"]
    r, p, y = (float(att[k]) for k in ("roll", "pitch", "yaw"))
    if not all(math.isfinite(v) for v in (r, p, y)):
        raise ValueError("invalid attitude")
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return ((cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy),
            (cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy),
            (-sp, sr*cp, cr*cp))


def interpolate_rotation(a, b, fraction):

    def quaternion(angles):
        r, p, y = (v/2 for v in angles)
        cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
        return (cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy)
    qa, qb = quaternion(a), quaternion(b)
    dot = sum(x*y for x, y in zip(qa, qb))
    if dot < 0:
        qb, dot = tuple(-v for v in qb), -dot
    if dot > 0.9995:
        q = unit(tuple(x+fraction*(y-x) for x, y in zip(qa, qb)))
    else:
        angle = math.acos(max(-1.0, min(1.0, dot)))
        u, v = math.sin((1-fraction)*angle)/math.sin(angle), math.sin(fraction*angle)/math.sin(angle)
        q = tuple(u*x+v*y for x, y in zip(qa, qb))
    w, x, y, z = q
    return ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


def rotate(mat, v, inverse=False):
    if inverse:
        mat = tuple(zip(*mat))
    return tuple(sum(a*b for a, b in zip(row, v)) for row in mat)


def camera_ray(ex, ey, fw, fh, tan_h, tan_v, camera_roll=0):
    x, y = 2 * ex / fw * tan_h, 2 * ey / fh * tan_v
    a = math.radians(camera_roll)
    x, y = x*math.cos(a)-y*math.sin(a), x*math.sin(a)+y*math.cos(a)
    return unit((y, x, -1.0))


class WaspGuidance:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.velocity = (0.0, 0.0, 0.0)

    def observe(self, ex, ey, fw, fh, tan_h, tan_v, att, capture_t, camera_roll=0):

        ray = camera_ray(ex, ey, fw, fh, tan_h, tan_v, camera_roll)
        world = rotate(rotation(att), ray)
        if self.previous is not None:
            t, old = self.previous
            dt = capture_t - t
            if 0.001 < dt <= 0.25:
                alpha = dt / (0.08 + dt)
                raw = tuple((a-b) / dt for a, b in zip(world, old))
                self.velocity = tuple(v + alpha*(r-v) for v, r in zip(self.velocity, raw))
            else:
                self.velocity = (0.0, 0.0, 0.0)
        self.previous = (capture_t, world)
        return dict(wasp=True, bearing=world, bearing_rate=self.velocity,
                    capture_t=capture_t,
                    az_deg=math.degrees(math.atan(2*ex/fw*tan_h)),
                    el_cam_deg=-math.degrees(math.atan(2*ey/fh*tan_v)), have_att=True)

    @staticmethod
    def rates(observation, att, now, gain=2.0, lead_s=0.08, max_rate_dps=90.0):
        error, ff = WaspGuidance.aim(observation, att, now, lead_s)
        rates = (gain*error[0]+ff[0], gain*error[1]+ff[1], 0.0)
        norm = math.hypot(*rates[:2])
        limit = math.radians(max_rate_dps)
        scale = min(1.0, limit / max(norm, 1e-9))
        return tuple(v*scale for v in rates)

    @staticmethod
    def aim(observation, att, now, lead_s=0.08):
        """Body error and line-of-sight rate, the same pair `rates` commands from."""
        age = now - observation["capture_t"]
        if not 0 <= age <= 0.25:
            raise ValueError("camera observation stale")
        world = observation["bearing"]
        velocity = observation["bearing_rate"]
        predicted = unit(tuple(v + (age+lead_s)*d for v, d in zip(world, velocity)))
        mat = rotation(att)
        body = rotate(mat, predicted, inverse=True)
        if body[2] >= -0.15:
            raise ValueError("target outside thrust-axis guidance cone")
        error = cross((0.0, 0.0, -1.0), body)
        ff = rotate(mat, cross(world, velocity), inverse=True)
        return error, ff


class Wasp:




    SETTINGS = {"throttle": (0.75, 0.85, 1.0), "ramp": (0.0, 0.1, 0.25, 0.5),
                "straight": (0.0, 0.5, 1.0, 2.0), "blend": (0.25, 0.5, 1.0),
                "rate": (30.0, 45.0, 60.0, 90.0, 120.0), "gain": (2.0, 4.0, 6.0),
                "lead": (0.0, 0.08, 0.15), "camera_roll": (0.0, 90.0, 180.0, 270.0),
                "steady": (0.0, 1.0)}
    LAUNCH_SECONDS_MIN = 0.0
    LAUNCH_SECONDS_MAX = 60.0
    SHARED_KEYS = ("throttle", "launch_s", "ramp", "steady")
    # Steady steering: a large miss is eased in, and a reversal waits until the
    # picture has stayed on the new side.  Off leaves the command untouched.
    STEADY_KNEE = math.radians(8.0)
    STEADY_DEADBAND = math.radians(0.4)
    STEADY_HOLD_S = 0.12
    STEADY_BRAKE_S = 0.45
    STEADY_ACCEL = math.radians(80.0)
    # Dynamic centers the box on the first sample, wherever it sits: low,
    # off to one side, or both.  Roll and pitch are what move the box.
    # Yaw only turns the nose onto the target's bearing, and only when that
    # bearing is off the nose; it is not what centers the picture.  Steady
    # on or off does not enter this law: the rate below already refuses to hunt.
    CLASSIC_PROFILES = ("smooth", "standard", "responsive", "extra")
    LOCKED_STEERING = ("straight", "blend", "rate", "gain", "lead")
    # One design lag for the rate loop (seconds).  A proportional chase
    # k = 1/(4 τ) is critically damped on q' = (u-q)/τ, e' = -q, before the
    # camera delay.  DYN_LAM is a little quicker; the prediction below gives
    # the delay back so the stick still does not cross the crosshair.
    DYN_TAU = 0.12
    DYN_LAM = 3.2
    DYN_KP = DYN_LAM
    DYN_KD = 0.35
    # Residual rate past the wanted rate.  The stick may reverse only when
    # that extra will carry the lock through the crosshair, or a hard
    # acceleration (the climb) is still adding to it.  Below this, a look-up
    # may only ease off to zero.
    DYN_KICK = math.radians(50.0)
    DYN_KICK_ACCEL = math.radians(450.0)
    DYN_BRAKE_KD = 1.2
    DYN_PRED = 0.08
    DYN_RATE = 140.0
    DYN_YAW_LAM = 2.0
    DYN_YAW_RATE = 90.0
    DYN_DEAD = math.radians(0.35)
    DYN_NOSE_DEAD = math.radians(1.5)
    DYN_GYRO_TAU = 0.04
    # Line-of-sight rate of the target itself, after a cap and a short filter.
    # Gain 1 holds a steady crossing on the crosshair.  The cap stops a bad
    # velocity sample from throwing the stick.
    DYN_FF_GAIN = 1.0
    DYN_FF_CAP = math.radians(45.0)
    DYN_FF_TAU = 0.08
    STEERING_PROFILES = {
        "smooth": dict(blend=1.0, rate=45.0, gain=2.0),
        "standard": dict(blend=0.5, rate=60.0, gain=2.0),
        "responsive": dict(blend=0.5, rate=90.0, gain=4.0),
        "extra": dict(blend=0.25, rate=120.0, gain=6.0, lead=0.08, camera_roll=0.0),
        "dynamic": dict(straight=0.0, blend=0.15, rate=DYN_RATE, gain=DYN_KP,
                        lead=0.0, camera_roll=0.0, dynamic=1.0),
    }

    def __init__(self):
        self.enabled = False
        self.camera_confirmed = False
        self.params = dict(throttle=1.0, ramp=0.1, straight=0.5, blend=0.5,
                           rate=60.0, gain=2.0, lead=0.08, camera_roll=0.0, launch_s=0.0,
                           centre_timer=3.0, steady=0.0)
        profile_defaults = {key: value for key, value in self.params.items() if key not in self.SHARED_KEYS}
        self.profiles = {name: dict(profile_defaults, **settings)
                         for name, settings in self.STEERING_PROFILES.items()}
        self.active_profile = "standard"
        self.started = None
        self.launch_t = None
        self.acquired_t = None
        self.used = False
        self._output_t = None
        self._rates = (0.0, 0.0, 0.0)
        self._reset_steady()

    def steering_profile(self):
        return self.active_profile

    def dynamic_active(self):
        return bool(self.params.get("dynamic"))

    @classmethod
    def valid_param(cls, key, value):
        return (math.isfinite(value) and cls.LAUNCH_SECONDS_MIN <= value <= cls.LAUNCH_SECONDS_MAX
                if key in ("launch_s", "centre_timer") else key in cls.SETTINGS and value in cls.SETTINGS[key])

    def load_defaults(self, saved):
        selected = saved.get("selected")
        if isinstance(selected, str) and selected in self.profiles:
            self.active_profile = selected
        profiles = saved.get("profiles", {})
        if isinstance(profiles, dict):
            for name, settings in profiles.items():
                if name not in self.profiles or not isinstance(settings, dict):
                    continue
                for key, value in settings.items():
                    if key in self.SHARED_KEYS or key == "dynamic":
                        continue
                    if name == "dynamic" and key in self.LOCKED_STEERING:
                        continue
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        continue
                    if self.valid_param(key, value):
                        self.profiles[name][key] = value
        self.params.pop("dynamic", None)
        self.params.update(self.profiles[self.active_profile])


        legacy = profiles.get(self.active_profile, {}) if isinstance(profiles, dict) else {}
        legacy = legacy if isinstance(legacy, dict) else {}
        shared = saved.get("shared", {})
        shared = shared if isinstance(shared, dict) else {}
        for key in self.SHARED_KEYS:
            if key == "launch_s":
                continue  # no launch timer any more: ARM, then LAUNCH WASP goes at once
            try:
                value = float(shared.get(key, legacy.get(key, self.params[key])))
            except (TypeError, ValueError):
                continue
            if key == "throttle":
                value = {0.25: 0.75, 0.5: 0.85}.get(value, value)
            if self.valid_param(key, value):
                self.params[key] = value

    def begin(self, now, target_acquired=True):
        self.started = now
        self.launch_t = None
        self.acquired_t = now if target_acquired else None
        self.used = True
        self._output_t = now
        self._rates = (0.0, 0.0, 0.0)
        self._reset_steady()

    def progress(self, now):
        if self.launch_t is not None:
            countdown = max(0.0, self.launch_t - now)
            return dict(stage="COUNTDOWN", straight_remaining_s=round(countdown, 2),
                        steering_blend=0.0, target_acquired=False,
                        launch_countdown_s=round(countdown, 2))
        if self.started is None:
            return dict(stage="IDLE", straight_remaining_s=0.0, steering_blend=0.0,
                        target_acquired=False)
        elapsed = max(0.0, now - self.started)
        remaining = max(0.0, self.params["straight"] - elapsed)
        if self.acquired_t is None:
            return dict(stage="SEARCH", straight_remaining_s=round(remaining, 2),
                        steering_blend=0.0,
                        target_acquired=False)


        blend_t = max(self.started + self.params["straight"], self.acquired_t)
        fraction = max(0.0, min(1.0, (now-blend_t)/self.params["blend"]))
        blend = fraction*fraction*(3.0-2.0*fraction)
        return dict(stage="STRAIGHT" if remaining > 0 else "BLEND" if fraction < 1 else "GUIDE",
                    straight_remaining_s=round(remaining, 2), steering_blend=blend,
                    target_acquired=True)

    def _reset_steady(self):
        self._steady_allow = [0, 0]
        self._steady_hold = [0.0, 0.0]
        self._steady_prev = None
        self._steady_rate = [0.0, 0.0]
        self._gyro_f = None
        self._gyro_dot = (0.0, 0.0, 0.0)
        self._ff_f = None
        self._axis_hist = [[], [], []]

    def _steady_desired(self, desired, error, dt):
        # A correction stays until the picture has kept the new side.  Braking
        # only shrinks a command; it cannot flip it.  The small axis waits
        # while one axis is still far from the target.
        knee, dead = self.STEADY_KNEE, self.STEADY_DEADBAND
        cmd = [0.0, 0.0, 0.0]
        prev = self._steady_prev
        alpha = 0.0 if dt <= 0 else dt / (0.08 + dt)
        for i in (0, 1):
            e = error[i]
            de = 0.0 if prev is None or dt <= 0 else (e - prev[i]) / dt
            self._steady_rate[i] += alpha * (de - self._steady_rate[i])
            if abs(e) <= dead:
                shaped, sign = 0.0, 0
            else:
                sign = 1 if e > 0 else -1
                scale = 1.0 if abs(e) <= knee else (knee + 0.25 * (abs(e) - knee)) / abs(e)
                shaped = desired[i] * scale
                braked = shaped + self.STEADY_BRAKE_S * self._steady_rate[i]
                if shaped == 0 or braked * shaped < 0:
                    braked = 0.0
                elif abs(braked) > abs(shaped):
                    braked = shaped
                shaped = braked
            if sign == 0 or self._steady_allow[i] in (0, sign):
                if sign:
                    self._steady_allow[i] = sign
                self._steady_hold[i] = 0.0
                cmd[i] = shaped
            else:
                self._steady_hold[i] += dt
                if self._steady_hold[i] >= self.STEADY_HOLD_S:
                    self._steady_allow[i] = sign
                    self._steady_hold[i] = 0.0
                    cmd[i] = shaped
                else:
                    cmd[i] = 0.0
        major = 0 if abs(error[0]) >= abs(error[1]) else 1
        minor = 1 - major
        if abs(error[major]) > knee and abs(error[major]) > 2 * abs(error[minor]):
            cmd[minor] = 0.0
        self._steady_prev = (error[0], error[1])
        return tuple(cmd)

    def _gyro_sample(self, att):
        sample = []
        for key in ("rollspeed", "pitchspeed", "yawspeed"):
            try:
                value = float((att or {}).get(key) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            sample.append(value if math.isfinite(value) else 0.0)
        return sample

    def _los_feedforward(self, ff, dt):
        if self._ff_f is None:
            self._ff_f = [0.0, 0.0]
        cap = self.DYN_FF_CAP
        alpha = 0.0 if dt <= 0.0 else dt / (self.DYN_FF_TAU + dt)
        out = []
        for i in (0, 1):
            try:
                sample = float(ff[i] if ff is not None else 0.0)
            except (TypeError, ValueError, IndexError):
                sample = 0.0
            if not math.isfinite(sample):
                sample = 0.0
            sample = max(-cap, min(cap, sample))
            self._ff_f[i] += alpha * (sample - self._ff_f[i])
            out.append(self.DYN_FF_GAIN * self._ff_f[i])
        return out

    def _filtered_gyro(self, att, dt):
        measured = self._gyro_sample(att)
        prev = None if self._gyro_f is None else tuple(self._gyro_f)
        if self._gyro_f is None or len(self._gyro_f) != 3:
            self._gyro_f = measured
        elif dt > 0.0:
            alpha = dt / (self.DYN_GYRO_TAU + dt)
            for i in range(3):
                self._gyro_f[i] += alpha * (measured[i] - self._gyro_f[i])
        if prev is not None and dt > 1e-4:
            self._gyro_dot = tuple((a - b) / dt for a, b in zip(self._gyro_f, prev))
        else:
            self._gyro_dot = (0.0, 0.0, 0.0)
        return self._gyro_f

    def _damped_axis(self, err, gyro, feed, cap, lam, gyro_dot=0.0, hist=None):
        # Positive gyro closes a positive error (e' = -q on that axis).
        # `feed` is the target's own rate and is matched, not braked.
        # The residual rate is what would carry the box through the crosshair.
        if abs(err) <= self.DYN_DEAD:
            err = 0.0
        rel = gyro - feed
        e_hat = err - rel * self.DYN_PRED
        if err * e_hat < 0.0:
            # The residual rate reaches the crosshair before this command lands.
            e_hat = 0.0
        # Coast distance of a rate q with the stick released is about q*τ.
        # Stay under that, and under the critical gain, and under the cap.
        coast = abs(e_hat) / self.DYN_TAU
        g_des = max(-coast, min(coast, lam * e_hat))
        # The command we send now arrives one prediction later, and a kick
        # keeps accelerating until then.  Judge the brake on that future rate.
        rel_soon = rel + gyro_dot * self.DYN_PRED
        excess = rel_soon - g_des
        # Releasing the stick stops a rate q in about q*τ.  A saturated chase
        # is still inside that distance for a long time: ease off, do not slam
        # the other way.  Full opposite stick is for a kick that will cross
        # anyway, or for an acceleration still speeding the gyro up.
        will_cross = abs(rel_soon) * self.DYN_TAU > abs(e_hat) and err * rel_soon > 0.0
        # The stick from ~80 ms ago is what the rate loop is reacting to now.
        # Acceleration beyond that is a kick (the climb), not the command
        # still arriving.
        delayed = 0.0 if not hist or len(hist) < 4 else hist[-4]
        a_extra = gyro_dot - (delayed - gyro) / self.DYN_TAU
        launched = err * a_extra > 0.0 and abs(a_extra) > self.DYN_KICK_ACCEL
        if err * excess > 0.0 and (
            (will_cross and abs(excess) > self.DYN_KICK)
            or (launched and abs(excess) > math.radians(35.0))
        ):
            u = feed - math.copysign(cap, excess)
        elif err * excess > 0.0:
            # A little fast: ease the command down.  Do not cross zero.  Crossing
            # zero here is what made the stick hunt up and down on a still target.
            u = feed + g_des - self.DYN_BRAKE_KD * excess
            if err != 0.0 and u * err < 0.0:
                u = 0.0
        else:
            u = feed + g_des + self.DYN_KD * (g_des - rel)
            if err != 0.0 and u * err < 0.0 and abs(excess) <= self.DYN_KICK:
                u = 0.0
        u = max(-cap, min(cap, u))
        if hist is not None:
            hist.append(u)
            if len(hist) > 8:
                del hist[:-8]
        return u

    def _nose_heading_error(self, error, att):
        # Radians the nose must yaw to face the target's bearing.  Positive
        # means the target is to the right of the nose.  Roll can center the
        # box and leave this bearing untouched, so the yaw loop uses the
        # bearing and not the picture angle.  Nearly overhead, the bearing
        # is meaningless and the nose stays where it is.
        alpha, beta = -error[1], error[0]
        level = alpha * alpha + beta * beta
        if level >= 1.0:
            return 0.0
        body = (alpha, beta, -math.sqrt(1.0 - level))
        try:
            mat = rotation(att)
            world = rotate(mat, body)
            nose = rotate(mat, (1.0, 0.0, 0.0))
        except (TypeError, ValueError, KeyError):
            return 0.0
        if math.hypot(world[0], world[1]) < 0.08 or math.hypot(nose[0], nose[1]) < 0.20:
            return 0.0
        err = math.atan2(world[1], world[0]) - math.atan2(nose[1], nose[0])
        err = math.atan2(math.sin(err), math.cos(err))
        if abs(err) <= self.DYN_NOSE_DEAD:
            return 0.0
        return err

    def _dynamic_rates(self, error, att, dt, blend, ff=None):
        # `blend` is unused on purpose.  Dynamic centers on the first sample,
        # wherever the box sits.  Steady is not applied either: this rate
        # already dies at the crosshair instead of reversing the stick.
        del blend
        cap = math.radians(self.DYN_RATE)
        ycap = math.radians(self.DYN_YAW_RATE)
        gyro = self._filtered_gyro(att, dt)
        los = self._los_feedforward(ff, dt)
        # Roll moves the box sideways, pitch moves it along the nose.  Either
        # or both, on the first sample.  The stop-limited rate makes that one
        # move onto the crosshair.
        dot = self._gyro_dot or (0.0, 0.0, 0.0)
        # Heading.  φ' = -ψ for the bearing, so the same damped axis turns the
        # nose once and stops on it.  A target already on the nose line gives
        # φ = 0 and the yaw stick stays centered.
        alpha, beta = -error[1], error[0]
        spin = gyro[2]
        phi = self._nose_heading_error(error, att)
        yaw = self._damped_axis(phi, spin, 0.0, ycap, self.DYN_YAW_LAM, dot[2], self._axis_hist[2])
        # Body rates p, q and a yaw rate ψ move the picture as
        # α' = q + ψ β, β' = -p - ψ α.  The closing rates are p+ψα and q+ψβ.
        # Brake those.  Then take the yaw part back off the stick, using the
        # yaw the gyro is actually doing, so the box stays on a straight path
        # while the nose turns underneath it.
        close_roll = gyro[0] + spin * alpha
        close_pitch = gyro[1] + spin * beta
        # Kick from the body gyro's own acceleration.  The picture-rate
        # derivative also contains the yaw spooling up, and that is not a kick.
        roll = self._damped_axis(error[0], close_roll, los[0], cap, self.DYN_LAM, dot[0], self._axis_hist[0])
        pitch = self._damped_axis(error[1], close_pitch, los[1], cap, self.DYN_LAM, dot[1], self._axis_hist[1])
        # Yaw trim.  It may shrink a centering stick.  It may not flip one:
        # a flip is the stick hunting left and right against the box.  A brake
        # the axis already chose (the climb) keeps its sign.
        centered = (roll, pitch)
        roll -= spin * alpha
        pitch -= spin * beta
        kept = []
        for axis, cmd_axis, err in ((0, roll, error[0]), (1, pitch, error[1])):
            if err != 0.0 and cmd_axis * err < 0.0 and centered[axis] * err >= 0.0:
                cmd_axis = 0.0
            kept.append(cmd_axis)
        roll, pitch = kept
        cmd = (max(-cap, min(cap, roll)), max(-cap, min(cap, pitch)), yaw)
        self._rates = cmd
        return cmd

    def lose_target(self):


        self.acquired_t = None
        self._rates = (0.0, 0.0, 0.0)
        self._reset_steady()

    def acquire(self, now, ctx):

        p = self.params
        WaspGuidance.rates(ctx["ai_cmd"], ctx["att"], now,
                           gain=p["gain"], lead_s=p["lead"], max_rate_dps=p["rate"])
        self.acquired_t = now

    def fault(self, now, ctx, min_conf):
        cmd = ctx.get("ai_cmd") or {}
        if ctx.get("att") is None:
            return "no fresh attitude"
        if (not cmd.get("wasp") or not cmd.get("have_att")
                or cmd.get("camera_roll") != self.params["camera_roll"]
                or not 0 <= now - cmd.get("capture_t", -1e9) <= 0.25
                or ctx.get("det_t") is None or not 0 <= now - ctx["det_t"] <= 0.25
                or not math.isfinite(ctx.get("det_conf") or 0)
                or (ctx.get("det_conf") or 0) < min_conf):
            return "no fresh confirmed target"
        return ""

    def output(self, mapper, now, ctx):
        p = self.params
        if self.started is None:
            raise ValueError("WASP launch has not started")



        dt = max(0.0, min(0.1, now-self._output_t))
        blend = self.progress(now)["steering_blend"]
        if self.dynamic_active() and self.acquired_t is not None:
            # Keep the target's own motion.  A wild velocity sample is clamped
            # so it cannot push the aim point out of the guidance cone or
            # silence the gyro brake.  Lead stays 0: attitude lag must not
            # be added again on top of the gyro.
            held = dict(ctx["ai_cmd"])
            raw_rate = held.get("bearing_rate") or (0.0, 0.0, 0.0)
            clamped = []
            for value in list(raw_rate)[:3]:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = 0.0
                if not math.isfinite(value):
                    value = 0.0
                clamped.append(max(-8.0, min(8.0, value)))
            held["bearing_rate"] = tuple(clamped)
            error, ff = WaspGuidance.aim(held, ctx["att"], now, lead_s=0.0)
            commanded = self._dynamic_rates(error, ctx.get("att"), dt, blend, ff)
            self._output_t = now
            ramp = p.get("ramp", 0.0)
            elapsed = now - self.started
            fraction = 1.0 if ramp <= 0 or elapsed >= ramp - 1e-6 else max(0.0, elapsed/ramp)
            return mapper.wasp_pwm(commanded, p["throttle"]*fraction)
        rates = ((0.0, 0.0, 0.0) if self.acquired_t is None else
                 WaspGuidance.rates(ctx["ai_cmd"], ctx["att"], now,
                                    gain=p["gain"], lead_s=p["lead"], max_rate_dps=p["rate"]))
        desired = tuple(v*blend for v in rates)
        if p.get("steady") and self.acquired_t is not None:
            error, _ff = WaspGuidance.aim(ctx["ai_cmd"], ctx["att"], now, lead_s=p["lead"])
            desired = self._steady_desired(desired, error, dt)
        delta = tuple(a-b for a, b in zip(desired, self._rates))
        norm = math.hypot(*delta[:2])
        # Off: the stick acceleration is the one this profile has always used.
        cap = 2*math.radians(p["rate"])/p["blend"]
        if p.get("steady"):
            cap = min(cap, self.STEADY_ACCEL)
        scale = min(1.0, cap*dt/max(norm, 1e-9))
        self._rates = tuple(v+d*scale for v, d in zip(self._rates, delta))
        self._output_t = now
        # Optional ramp from the pilot's zero stick.  Off by default: the FC
        # already spools up from ground idle over MOT_SPOOL_TIME, and a fast
        # launch clears a tripod or slope before the tilt costs height.
        ramp = p.get("ramp", 0.0)
        elapsed = now - self.started
        fraction = 1.0 if ramp <= 0 or elapsed >= ramp - 1e-6 else max(0.0, elapsed/ramp)
        return mapper.wasp_pwm(self._rates, p["throttle"]*fraction)
