"""Fixed upward camera / thrust-axis guidance. Body axes are forward, right, down.

At camera roll 0, image right is body right and image bottom is body forward.
World bearings use NED. No range is inferred from the detector's box size.
"""
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
    """Quaternion SLERP avoids Euler wrap and the vertical-pitch singularity."""
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
        """Remove ego rotation before estimating target angular motion."""
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
        # Shortest rotation from body up to target, plus inertial LOS feedforward.
        error = cross((0.0, 0.0, -1.0), body)
        ff = rotate(mat, cross(world, velocity), inverse=True)
        rates = (gain*error[0]+ff[0], gain*error[1]+ff[1], 0.0)
        norm = math.hypot(*rates[:2])
        limit = math.radians(max_rate_dps)
        scale = min(1.0, limit / max(norm, 1e-9))
        return tuple(v*scale for v in rates)


class Wasp:
    """Launch lifecycle; output is released, never a blind full-throttle search."""
    SETTINGS = {"throttle": (0.25, 0.5, 1.0), "duration": (1.0, 2.0, 3.0, 5.0, 10.0),
                "straight": (0.0, 0.5, 1.0, 2.0), "blend": (0.25, 0.5, 1.0),
                "rate": (30.0, 45.0, 60.0, 90.0), "gain": (2.0, 4.0, 6.0),
                "lead": (0.0, 0.08, 0.15), "camera_roll": (0.0, 90.0, 180.0, 270.0)}

    def __init__(self):
        self.enabled = False
        self.camera_confirmed = False
        self.params = dict(throttle=1.0, duration=10.0, straight=1.0, blend=0.5,
                           rate=60.0, gain=2.0, lead=0.08, camera_roll=0.0)
        self.started = None
        self.used = False
        self._output_t = None
        self._rates = (0.0, 0.0, 0.0)

    @staticmethod
    def timing_error(params):
        if params["straight"] + params["blend"] >= params["duration"]:
            return "maximum flight time must exceed straight launch + steering blend"
        return ""

    def begin(self, now):
        self.started = now
        self.used = True
        self._output_t = now
        self._rates = (0.0, 0.0, 0.0)

    def progress(self, now):
        if self.started is None:
            return dict(stage="IDLE", straight_remaining_s=0.0, steering_blend=0.0)
        elapsed = max(0.0, now - self.started)
        remaining = max(0.0, self.params["straight"] - elapsed)
        fraction = max(0.0, min(1.0, (elapsed-self.params["straight"])/self.params["blend"]))
        blend = fraction*fraction*(3.0-2.0*fraction)
        return dict(stage="STRAIGHT" if remaining > 0 else "BLEND" if fraction < 1 else "GUIDE",
                    straight_remaining_s=round(remaining, 2), steering_blend=blend)

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
        # Continue validating the live bearing during straight flight, without
        # applying target steering. Zero ACRO rates request no body rotation;
        # they do not enforce a straight ground trajectory or compensate gravity.
        rates = WaspGuidance.rates(ctx["ai_cmd"], ctx["att"], now,
                                  gain=p["gain"], lead_s=p["lead"], max_rate_dps=p["rate"])
        blend = self.progress(now)["steering_blend"]
        desired = tuple(v*blend for v in rates)
        # Bound changes even if a detection jumps during/after the handover.
        # Cap dt so a stalled sender cannot resume with a large steering step.
        dt = max(0.0, min(0.1, now-self._output_t))
        delta = tuple(a-b for a, b in zip(desired, self._rates))
        norm = math.hypot(*delta[:2])
        scale = min(1.0, 2*math.radians(p["rate"])/p["blend"]*dt/max(norm, 1e-9))
        self._rates = tuple(v+d*scale for v, d in zip(self._rates, delta))
        self._output_t = now
        return mapper.wasp_pwm(self._rates, p["throttle"])
