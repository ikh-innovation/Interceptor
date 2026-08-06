#!/usr/bin/env python3
"""
fc_control.py  —  flight-control layer for the WASP head tracker
==============================================================================
Turns "the head is N pixels off-centre" into stick commands the flight
controller actually honours.

TWO CONTROL LAWS, FOR TWO DIFFERENT PROBLEMS
--------------------------------------------
  * `ControlLaw`    — visual servo. yaw+pitch BODY RATES null the pixel error, in
                      ACRO. Correct ON A GIMBAL, where the airframe cannot
                      translate and pointing the camera IS the whole job.
  * `InterceptLaw`  — proportional navigation on ALL FOUR channels, in ALT_HOLD.
                      Correct IN FREE FLIGHT. yaw aims the seeker, ROLL banks to
                      turn the velocity vector, pitch sets closing speed, THROTTLE
                      closes the vertical.

They are not variants of one another. In free flight, pointing the camera and
moving the airframe are the same two actuators, so aiming and intercepting
compete for them — which is why the servo law tracked perfectly on the gimbal and
then flew BACKWARDS in the air. Pick with CONTROL_LAW=servo|intercept.

WHY THIS EXISTS  (i.e. what was broken before)
----------------------------------------------
The previous control path streamed GUIDED velocity setpoints
(SET_POSITION_TARGET_LOCAL_NED).  That can never move this airframe, for two
independent reasons:

  1. GUIDED is a *flight mode*.  The drone is flown in ACRO, so the tracker's
     own GUIDED gate was False on every single frame and NOT ONE BYTE was ever
     transmitted to the FC.  The HUD said "TX" because it reported "a setpoint
     was computed", not "a setpoint left the Jetson".
  2. Even in GUIDED, a velocity setpoint needs an EKF position estimate
     (GPS / optical flow).  On a stationary gimbal there is none, so ArduPilot
     would have refused the setpoints anyway.

ACRO is the RIGHT mode for this job.  In ACRO a stick deflection commands a
BODY ANGULAR RATE, and "rotate the airframe until the target is centred" is
precisely a body-rate command.  The transport that injects stick values is
RC_CHANNELS_OVERRIDE (ArduPilot) or MSP_SET_RAW_RC (Betaflight / INAV).

THE SILENT KILLER  (ArduPilot)
------------------------------
ArduPilot accepts RC_CHANNELS_OVERRIDE **only** from the MAVLink system id
held in SYSID_MYGCS (renamed MAV_GCS_SYSID on 4.6+), whose default is 255:

    void GCS_MAVLINK::handle_rc_channels_override(const mavlink_message_t &msg)
    {
        if (msg.sysid != sysid_my_gcs()) {
            return;          // silently dropped: no ACK, no STATUSTEXT, nothing
        }

The old code sent from source_system=1, so every override was discarded
without a single diagnostic anywhere.  This module reads SYSID_MYGCS off the
FC and adopts it, and then CONFIRMS the override actually landed by watching
the FC's own RC_CHANNELS echo — ArduPilot reports the overridden value back in
RC_CHANNELS (RC_Channel::update() copies override_value into radio_in), so a
matching echo is proof the FC accepted our sticks.  If the echo never matches,
this module says so loudly instead of pretending to fly.

CONTROL LAW
-----------
    pixel offset  ->  normalized offset  ->  angular error  ->  body rate
                  ->  stick deflection  ->  PWM / MSP channel value

  * Horizontal error is nulled with YAW rate.
  * Vertical error is nulled with PITCH rate (nose down when the target sits
    below the frame centre).  Positive body pitch rate is nose UP — right-hand
    rule about the body Y axis, which points right — hence the sign flip.
  * Roll and throttle are never touched: they stay with the pilot.

The angular error uses the pinhole (tangent) model rather than a linear
pixels-to-degrees ratio, which matters on the wide CSI lenses.  If the
configured FOV is wrong the loop still converges — a wrong FOV only rescales
the effective gain — so tune KP_YAW / KP_PITCH rather than chasing the lens
spec.  Note the tracker resizes 1280x720 to 640x640 before detection: that is
a linear (if anisotropic) resize, so the NORMALIZED offset is preserved and
feeding it the true camera FOV is correct.

WHAT WILL STILL STOP THE DRONE MOVING  (not software)
-----------------------------------------------------
  * DISARMED.  Overrides are accepted but the motors are off.
  * ACRO at idle throttle.  In ACRO the attitude controller's authority scales
    with throttle: at minimum stick the mixer has no headroom to differentiate
    the motors, so yaw/pitch commands produce almost no torque.  The pilot must
    hold mid throttle for the airframe to respond.  PROPS OFF on the bench.
  * RC_OPTIONS bit 1 (value 2, "IGNORE_OVERRIDES") set on the FC.
  * ACRO_TRAINER = 1/2 auto-levels and limits lean angle, which fights a
    sustained pitch command.  Set ACRO_TRAINER = 0 for the bench.
All four are checked and reported at connect time.

AND ONE MORE, WHICH IS OPTICS RATHER THAN SOFTWARE
--------------------------------------------------
At nose-down lean `theta` a target at the airframe's OWN altitude appears at
elevation +theta in a body-fixed camera.  The lens is 48.8 deg vertically
(+/-24.4), and sensor mode 4 is a CROPPED IMX219 mode so the truth is narrower.
So at 12 deg of lean a level target is already halfway to the top edge, and past
roughly 24 deg it leaves the frame entirely.  A level-mounted camera therefore
cannot both fly fast and keep the target in view, at any gain.  Tilt the mount UP
by about the cruise lean and tell the law with CAM_TILT_DEG.
"""

import math
import os
import struct
import threading
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep

try:
    from pymavlink import mavutil
except ImportError:                      # dev box without pymavlink
    mavutil = None


# ══════════════════════════════════════════════════════════════════════════════
# small helpers
# ══════════════════════════════════════════════════════════════════════════════

def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _deadband(n, db):
    """Re-scaled deadband: |n| <= db -> 0, and the remaining range is stretched
    back out to +/-1 so there is no step at the edge of the band."""
    if db <= 0.0:
        return n
    a = abs(n)
    if a <= db:
        return 0.0
    return math.copysign((a - db) / (1.0 - db), n)


# ArduPilot Copter custom_mode -> name.  ACRO is what this module targets.
COPTER_MODES = {
    0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED",
    5: "LOITER", 6: "RTL", 7: "CIRCLE", 9: "LAND", 11: "DRIFT", 13: "SPORT",
    14: "FLIP", 15: "AUTOTUNE", 16: "POSHOLD", 17: "BRAKE", 18: "THROW",
    19: "AVOID_ADSB", 20: "GUIDED_NOGPS", 21: "SMART_RTL", 22: "FLOWHOLD",
    23: "FOLLOW", 24: "ZIGZAG", 25: "SYSTEMID", 26: "AUTOROTATE", 27: "AUTO_RTL",
}
ACRO_MODE = 1

# Modes in which a stick deflection commands a RATE (what this control law
# assumes).  STABILIZE/SPORT command an ANGLE instead — the loop still centres
# the target there, just with different dynamics, so they are allowed with a
# warning rather than refused.
RATE_MODES = {"ACRO"}
ANGLE_MODES = {"STABILIZE", "SPORT", "ALT_HOLD", "LOITER", "POSHOLD"}


class WrongFirmware(RuntimeError):
    """Raised when a backend finds a different autopilot than it drives, so the
    factory can try the next transport instead of dying."""


# ══════════════════════════════════════════════════════════════════════════════
# CONTROL LAW   pixel offset -> body rate -> stick
# ══════════════════════════════════════════════════════════════════════════════

class ControlLaw:
    """Proportional (optionally PD) visual servo on the bounding-box centre.

    A stick command in ACRO *is* a body-rate command, so a plain proportional
    law on the angular error gives a first-order response: the error decays
    with time constant 1/KP and cannot overshoot in the absence of latency.
    KD is available to damp the overshoot that detection latency introduces,
    and defaults to 0 because the position fed in here is already Kalman
    smoothed.
    """

    def __init__(self,
                 h_fov_deg=62.2,          # TRUE horizontal FOV of the lens
                 v_fov_deg=48.8,          # TRUE vertical FOV of the lens
                 kp_yaw=2.0,              # rad/s of yaw   per rad of az error
                 kp_pitch=2.0,            # rad/s of pitch per rad of el error
                 kd_yaw=0.0,
                 kd_pitch=0.0,
                 deadband_frac=0.03,      # of the half-frame; kills centre jitter
                 stick_limit=0.55,        # max |deflection| we will ever command
                 slew_per_s=3.0,          # max change in deflection per second
                 max_yaw_rate=None,       # rad/s at full stick (FC ACRO_Y_RATE)
                 max_pitch_rate=None,     # rad/s at full stick (FC ACRO_RP_RATE)
                 invert_yaw=False,
                 invert_pitch=False,
                 shape="linear",          # "linear" | "adaptive"
                 shape_gamma=1.6,         # curve exponent, >1 = soft near centre
                 shape_sat=0.30,          # |normalized offset| commanding full stick
                 # Per-axis ceilings.  None = fall back to stick_limit.  These
                 # exist because the two axes are NOT equivalent: ArduPilot's
                 # ACRO_RP_RATE is 360 deg/s against ACRO_Y_RATE's 202, so the
                 # same deflection is ~1.8x more angular rate in pitch.  Pitch is
                 # therefore the axis that goes unstable first against detection
                 # latency, and a single shared ceiling has to be set low enough
                 # for pitch — throwing away yaw authority for no reason.
                 stick_limit_yaw=None,
                 stick_limit_pitch=None,
                 # ── pitch role ───────────────────────────────────────────────
                 # "aim"      : pitch nulls the target's VERTICAL pixel error.
                 #              Correct on a gimbal, where the airframe cannot
                 #              translate and pitch only points the camera.
                 # "approach" : pitch holds a nose-DOWN lean so the airframe flies
                 #              FORWARD, and ignores vertical pixel error
                 #              entirely.  Correct in free flight.
                 #
                 # Off a gimbal these two are not variations of the same thing,
                 # they are in direct conflict: a multirotor's pitch attitude IS
                 # its direction of travel, so "aim the camera up at a target
                 # above me" and "fly towards it" are opposite commands on one
                 # axis.  Aiming up is what made the airframe retreat.
                 pitch_mode="aim",
                 approach_deg=12.0,       # max nose-down lean, at full throttle
                 approach_kp=2.5,         # rad/s of pitch rate per rad of lean err
                 approach_thr_deadband=0.05):
        # Guard the lean hard: a multirotor at 30 deg of lean is accelerating at
        # ~0.58 g and still climbing in commanded angle if the loop misbehaves.
        self.approach_deg = float(_clamp(approach_deg, 0.0, 30.0))
        self.approach_kp = max(0.0, float(approach_kp))
        self.approach_thr_deadband = float(_clamp(approach_thr_deadband, 0.0, 0.9))
        self.pitch_mode = ("approach" if str(pitch_mode).lower() == "approach"
                           else "aim")
        self.h_fov = math.radians(float(h_fov_deg))
        self.v_fov = math.radians(float(v_fov_deg))
        self.kp_yaw = float(kp_yaw)
        self.kp_pitch = float(kp_pitch)
        self.kd_yaw = float(kd_yaw)
        self.kd_pitch = float(kd_pitch)
        self.deadband_frac = float(deadband_frac)
        self.stick_limit = float(_clamp(stick_limit, 0.0, 1.0))
        self.slew_per_s = float(slew_per_s)
        # ArduPilot Copter defaults when the FC cannot be read.
        self.max_yaw_rate = float(max_yaw_rate or math.radians(202.5))
        self.max_pitch_rate = float(max_pitch_rate or math.radians(360.0))
        self.invert_yaw = bool(invert_yaw)
        self.invert_pitch = bool(invert_pitch)
        self.shape = "adaptive" if str(shape).lower() == "adaptive" else "linear"
        self.shape_gamma = max(1e-3, float(shape_gamma))
        self.shape_sat = float(_clamp(shape_sat, 1e-3, 1.0))
        self.stick_limit_yaw = float(_clamp(
            self.stick_limit if stick_limit_yaw is None else stick_limit_yaw,
            0.0, 1.0))
        self.stick_limit_pitch = float(_clamp(
            self.stick_limit if stick_limit_pitch is None else stick_limit_pitch,
            0.0, 1.0))

        self._tan_h = math.tan(self.h_fov / 2.0)
        self._tan_v = math.tan(self.v_fov / 2.0)

        self._ax_prev = 0.0
        self._ay_prev = 0.0
        self._t_prev = None
        self._yaw_stick = 0.0
        self._pitch_stick = 0.0

    # ── the approach term ─────────────────────────────────────────────────────
    def approach_pitch_stick(self, measured_pitch_rad, throttle_frac):
        """Pitch deflection that flies the airframe FORWARD, whatever the target's
        vertical position in frame.

        In ACRO a stick is a RATE, so a fixed forward lean cannot be commanded
        open-loop: hold a nose-down rate and the airframe keeps rotating past
        vertical.  So this closes an outer loop on the FC's own measured pitch:

            target_lean = -approach_deg * throttle_demand      (negative = nose down)
            pitch_rate  = approach_kp * (target_lean - measured_pitch)

        Once the lean is reached the error is zero, the rate command is zero, and
        ACRO holds the attitude — which is a steady forward acceleration.  Because
        the loop is closed on a measured angle it cannot run away.

        Two hard guarantees, both of which are the reported bug:
          * `target_lean` is never positive, so this NEVER commands nose-up and can
            therefore never fly the airframe backwards.
          * a missing or stale attitude reading returns 0.0 rather than a guess,
            because commanding a rate to correct a frozen angle would never stop.
        """
        if measured_pitch_rad is None:
            return 0.0
        db = self.approach_thr_deadband
        demand = (throttle_frac - db) / (1.0 - db) if throttle_frac > db else 0.0
        demand = _clamp(demand, 0.0, 1.0)
        if demand <= 0.0:
            return 0.0
        target_lean = -math.radians(self.approach_deg) * demand
        # Never nose-up: clamping the TARGET (not the output) means an airframe
        # already leaning further down than asked is allowed to come back up to
        # the target, but the target itself can never point backwards.
        target_lean = min(0.0, target_lean)
        rate = self.approach_kp * (target_lean - measured_pitch_rad)
        return rate / self.max_pitch_rate

    # ── the law ───────────────────────────────────────────────────────────────
    def compute(self, err_x, err_y, fw, fh, dt=None,
                measured_pitch_rad=None, throttle_frac=0.0):
        """err_x/err_y: bbox-centre minus frame-centre, in DISPLAY pixels.
        err_x > 0 -> target right of centre.  err_y > 0 -> target below centre.

        dt: seconds since the previous call, used by the derivative term and the
        slew limiter.  Left None in production, where the wall-clock gap between
        detection frames IS the correct timestep; passed explicitly by the
        closed-loop simulation so its timestep is not the host's loop speed.
        """
        now = perf_counter()
        if dt is None:
            dt = (now - self._t_prev) if self._t_prev is not None else 0.0
        self._t_prev = now

        # Normalized offset: +/-1 at the frame edge.  Preserved by the tracker's
        # anisotropic resize, so it pairs correctly with the true lens FOV.
        nx = _deadband(_clamp(2.0 * err_x / max(1, fw), -1.0, 1.0),
                       self.deadband_frac)
        ny = _deadband(_clamp(2.0 * err_y / max(1, fh), -1.0, 1.0),
                       self.deadband_frac)

        # Pinhole model: angle off boresight.
        ax = math.atan(nx * self._tan_h)          # azimuth   (right positive)
        ay = math.atan(ny * self._tan_v)          # elevation (down  positive)

        if dt > 1e-6:
            dax = (ax - self._ax_prev) / dt
            day = (ay - self._ay_prev) / dt
        else:
            dax = day = 0.0
        self._ax_prev, self._ay_prev = ax, ay

        if self.shape == "adaptive":
            # Deflection is a SHAPED function of how far off-centre the box is,
            # not a fixed multiple of it.  Beyond `shape_sat` of the half-frame
            # the stick is pinned at the limit, so the airframe swings at its
            # full ACRO rate the instant the target is properly off-boresight —
            # that is the "turn immediately" half.  Inside `shape_sat` the
            # exponent (>1) collapses the command far faster than a straight line
            # would, so the last stretch into the centre is deliberately gentle —
            # that is the "slow down as it approaches" half.  A linear law cannot
            # do both: one KP has to serve the far field and the near field, and
            # whichever you tune for, the other is wrong.
            #
            # Signs are taken from the SAME rule as the linear path: target right
            # of centre -> positive yaw; target below centre -> nose down, i.e.
            # negative pitch rate.  Only the magnitude curve differs.
            yaw_stick = self._shape(nx) * self.stick_limit_yaw
            pitch_stick = -self._shape(ny) * self.stick_limit_pitch
            # KD stays live and is what stops the fast far-field swing from
            # overshooting through the centre.  Converted into stick units,
            # because in this mode the command is a deflection, not a rate.
            yaw_stick += (self.kd_yaw * dax) / self.max_yaw_rate
            pitch_stick += -(self.kd_pitch * day) / self.max_pitch_rate
        else:
            # Target right of centre -> yaw RIGHT  (positive body yaw rate).
            yaw_rate = self.kp_yaw * ax + self.kd_yaw * dax
            # Target below centre -> nose DOWN.  Positive body pitch rate is nose
            # UP (right-hand rule about body Y, which points right), hence the
            # negation.
            pitch_rate = -(self.kp_pitch * ay + self.kd_pitch * day)

            yaw_stick = yaw_rate / self.max_yaw_rate
            pitch_stick = pitch_rate / self.max_pitch_rate

        if self.pitch_mode == "approach":
            # Replace, never blend.  Adding the aim term back in would reintroduce
            # exactly the nose-up-when-the-target-is-high behaviour this mode
            # exists to remove; vertical framing is the pilot's altitude to manage.
            pitch_stick = self.approach_pitch_stick(measured_pitch_rad,
                                                    throttle_frac)

        if self.invert_yaw:
            yaw_stick = -yaw_stick
        if self.invert_pitch:
            pitch_stick = -pitch_stick

        yaw_stick = _clamp(yaw_stick, -self.stick_limit_yaw, self.stick_limit_yaw)
        pitch_stick = _clamp(pitch_stick, -self.stick_limit_pitch,
                             self.stick_limit_pitch)

        yaw_stick = self._slew(self._yaw_stick, yaw_stick, dt)
        pitch_stick = self._slew(self._pitch_stick, pitch_stick, dt)
        self._yaw_stick, self._pitch_stick = yaw_stick, pitch_stick

        # Report the rate we will ACTUALLY get after clamping + slew, not the
        # raw law output, so the HUD and the CSV never overstate the command.
        return dict(
            yaw_stick=yaw_stick,
            pitch_stick=pitch_stick,
            yaw_rate=(-yaw_stick if self.invert_yaw else yaw_stick) * self.max_yaw_rate,
            pitch_rate=(-pitch_stick if self.invert_pitch else pitch_stick) * self.max_pitch_rate,
            az_deg=math.degrees(ax),
            el_deg=math.degrees(-ay),          # HUD convention: up is positive
        )

    # Timestep assumed for the slew limiter on the FIRST sample after a reset,
    # where there is no measured dt.  Without it the limiter would be bypassed
    # and the stick would jump straight to the commanded value on every target
    # re-acquisition — a visible jerk on the gimbal.
    _NOMINAL_DT = 0.05

    def _shape(self, n):
        """Map a normalized offset to a signed magnitude in [-1, 1].

        Monotonic, sign-preserving, zero at zero, and saturating at
        |n| >= shape_sat.  gamma > 1 flattens the response near the centre;
        gamma == 1 with shape_sat == 1 degenerates to the identity, so the
        adaptive mode is a strict generalisation of a proportional stick map.
        """
        a = abs(n)
        if a == 0.0:
            return 0.0
        m = min(1.0, a / self.shape_sat) ** self.shape_gamma
        return math.copysign(m, n)

    def _slew(self, prev, target, dt):
        if self.slew_per_s <= 0.0:
            return target
        step = self.slew_per_s * (dt if dt > 1e-6 else self._NOMINAL_DT)
        return _clamp(target, prev - step, prev + step)

    def neutral(self):
        """Centred sticks, and forget the derivative/slew history so the next
        acquisition does not inherit a stale ramp."""
        self._ax_prev = self._ay_prev = 0.0
        self._t_prev = None
        self._yaw_stick = self._pitch_stick = 0.0
        return dict(yaw_stick=0.0, pitch_stick=0.0, yaw_rate=0.0, pitch_rate=0.0,
                    az_deg=0.0, el_deg=0.0)

    def describe(self):
        if self.shape == "adaptive":
            law = (f"ADAPTIVE gamma {self.shape_gamma:.2f}  "
                   f"full stick at {self.shape_sat * 100:.0f}% off-centre  "
                   f"(KP unused in this mode)  "
                   f"KD {self.kd_yaw:.2f}/{self.kd_pitch:.2f}  ")
        else:
            law = (f"KP yaw/pitch {self.kp_yaw:.2f}/{self.kp_pitch:.2f}  "
                   f"KD {self.kd_yaw:.2f}/{self.kd_pitch:.2f}  ")
        if self.pitch_mode == "approach":
            law += (f"pitch=APPROACH (lean up to {self.approach_deg:.0f} deg "
                    f"nose-down on pilot throttle, KP {self.approach_kp:.1f}; "
                    f"vertical pixel error ignored)  ")
        else:
            law += "pitch=AIM (nulls vertical pixel error)  "
        return (f"FOV {math.degrees(self.h_fov):.1f}x{math.degrees(self.v_fov):.1f} deg  "
                + law
                + f"deadband {self.deadband_frac * 100:.1f}%  "
                + (f"stick<={self.stick_limit_yaw * 100:.0f}%  "
                   if self.stick_limit_yaw == self.stick_limit_pitch else
                   f"stick<= yaw {self.stick_limit_yaw * 100:.0f}%"
                   f"/pitch {self.stick_limit_pitch * 100:.0f}%  ")
                + f"slew {self.slew_per_s:.1f}/s  "
                + f"full-stick rate yaw/pitch "
                + f"{math.degrees(self.max_yaw_rate):.0f}/"
                + f"{math.degrees(self.max_pitch_rate):.0f} deg/s")


# ══════════════════════════════════════════════════════════════════════════════
# INTERCEPT LAW   proportional navigation on four channels, for FREE FLIGHT
# ══════════════════════════════════════════════════════════════════════════════

class _Rate:
    """Low-pass differentiator.

    A raw (x[n]-x[n-1])/dt on a detection position is unusable at intercept
    speeds: the bbox centre jitters by several pixels frame to frame and dt
    varies with inference load, so the quotient is dominated by noise. This is a
    single-pole filter on the derivative with time constant `tau` — enough to
    make an LOS rate that proportional navigation can multiply by a large gain
    without screaming.
    """

    def __init__(self, tau=0.12):
        self.tau = max(1e-3, float(tau))
        self.value = 0.0
        self._prev = None

    def reset(self):
        self.value = 0.0
        self._prev = None

    def update(self, x, dt):
        if self._prev is None or dt <= 1e-6:
            self._prev = x
            return self.value
        raw = (x - self._prev) / dt
        self._prev = x
        a = dt / (self.tau + dt)
        self.value += a * (raw - self.value)
        return self.value


class InterceptLaw:
    """Proportional-navigation intercept guidance. FREE FLIGHT ONLY.

    WHY NOT `ControlLaw`
    --------------------
    `ControlLaw` is a visual servo: it points the camera at the target. On the
    gimbal that IS the whole job and it works. In free flight it does not
    intercept, for reasons that are structural rather than tuning:

      * Pure pursuit (turn rate proportional to the CURRENT bearing error) lags a
        crossing target. Against someone running across the field it settles into
        a trailing tail-chase that never closes.
      * Yaw rotates the camera but does not move the airframe. A multirotor turns
        its VELOCITY vector by banking, and roll was never driven.
      * Nothing controlled the vertical. `pitch_mode="aim"` used pitch and flew
        backwards; `"approach"` correctly stopped using pitch and put nothing in
        its place. The only channel that changes altitude is throttle.

    THE FOUR CHANNELS, AND WHY EACH ONE
    -----------------------------------
    ==========  =========================  ====================================
    channel     job                        driven by
    ==========  =========================  ====================================
    yaw         keep the seeker on target  pursuit on azimuth error
    roll        lateral acceleration       PROPORTIONAL NAVIGATION on LOS rate
    pitch       closing speed              commanded nose-down lean x commit
    throttle    vertical closure           elevation error + LOS-rate lead
    ==========  =========================  ====================================

    PROPORTIONAL NAVIGATION
    -----------------------
    PN commands lateral acceleration proportional to the rate of rotation of the
    line of sight, not to the LOS angle::

        a_lat = GAIN * lambda_dot_azimuth

    Driving the LOS *rate* to zero holds the target at a CONSTANT BEARING, which
    is a collision course — that is the whole point, and it is why every missile
    does this instead of pure pursuit.

    The LOS rate is NOT the rate of change of the pixel error. With yaw slaving
    the nose to the target the pixel error stays near zero by construction, and
    the LOS rotation hides in the airframe's own heading instead. So the law
    differentiates the WORLD-referenced LOS bearing, as a single signal::

        lambda     = yaw_measured + azimuth_in_body_frame
        lambda_dot = d(lambda)/dt          (unwrapped, then low-pass filtered)

    Writing it as one signal is not cosmetic, it is what makes the loop stable.
    The obvious formulation, `d(az_body)/dt + yawspeed`, is algebraically the same
    thing but mixes a HEAVILY FILTERED term with an INSTANTANEOUS one: when the
    seeker yaws to null a pixel error, `yawspeed` jumps at once while the
    filtered `d(az_body)/dt` has not yet caught up, so the sum transiently reads
    a large LOS rate that is not there. PN then banks hard on the airframe's own
    yaw, which turns the airframe, which moves the pixel error, which yaws
    again — a limit cycle at full bank. It was measured doing exactly that in
    section 9 of tests/test_intercept.py: 30 deg of bank, sign-flipping, and the
    heading swinging through 180 deg while the range never closed.

    With the single-signal form the seeker's yaw CANNOT drive PN at all, and for
    a real reason rather than a tuned one: lambda is the bearing from the airframe
    to the target measured in the world, so rotating the airframe does not change
    it. Only relative TRANSLATION does. That is precisely the quantity
    proportional navigation is defined on.

    `yaw` and the body rates come from ATTITUDE, which the FC already streams.
    Without it PN cannot work at all, so a missing attitude commands zero roll
    rather than a guess.

    Sanity check on the gain: a COORDINATED turn — one where the velocity vector
    rotates exactly as fast as the nose — needs `a_lat = V_own * yaw_rate`. So
    `roll_pn_gain` is dimensionally an own-speed in m/s, and setting it to the
    real airspeed gives coordinated flight while setting it to ~2-3x that gives
    PN's lead. It is expressed as ONE gain because `N * V` is what the law
    actually multiplies by, and pretending to know N and V separately when there
    is no airspeed sensor and no GPS would be false precision.

    Bank angle follows from the acceleration: `phi = atan(a_lat / g)`.

    WHY THE CLOSING SPEED IS NOT IN THE GAIN
    ----------------------------------------
    Textbook PN uses `N * V_closing * lambda_dot`. Range here comes from bbox
    width against an assumed head size, which is noisy and biased, and V_closing
    collapses toward zero when the target runs away at a similar speed — which
    would command NO turn exactly when the chase needs one. So range and closing
    speed are computed for the terminal trigger and the log, and deliberately
    kept out of the loop gain.

    ANGLE MODE, NOT ACRO
    --------------------
    This law commands ANGLES (lean, bank) and a CLIMB RATE, so it wants
    ALT_HOLD, not ACRO:

      * ACRO's stick is a rate, which is why `approach_pitch_stick` had to close
        an outer loop on measured ATTITUDE just to hold a lean. In ALT_HOLD the
        stick IS the lean and that entire loop disappears.
      * ALT_HOLD's throttle stick is a CLIMB RATE, and centre means HOLD
        ALTITUDE. That is what makes it safe for this code to hold the throttle
        at all: a frozen neutral throttle keeps flying level, where in ACRO it
        would be an arbitrary thrust setting.
      * ALT_HOLD needs only the barometer. There is no GPS on this airframe.

    THE CAMERA-TILT COUPLING
    ------------------------
    At nose-down lean `theta`, a target at the airframe's OWN altitude appears at
    elevation `+theta` in a body-fixed camera. The lens is 48.8 deg vertical
    (+/-24.4), and sensor mode 4 is a cropped IMX219 mode so the truth is
    narrower. At 12 deg of lean a level target is already halfway to the top edge;
    past ~24 deg it leaves the frame entirely. This is the geometry that made
    `aim` and `approach` both unwinnable — `aim` kept the target in frame by
    pitching up and flying backwards, `approach` flew forwards and lost it out of
    the top.

    Two fixes, both applied:
      * `cam_tilt_deg` — the camera is physically tilted UP by this many degrees,
        so at cruise lean the boresight is level. Hardware.
      * elevation is resolved into the WORLD frame using the FC's measured pitch:
        `el_world = pitch + cam_tilt + el_camera`. Software, and free, because
        ATTITUDE is already streaming.

    Pixel error is also DE-ROTATED by measured roll before anything else. Once
    roll is driven the horizon tilts in the image, and without de-rotation a bank
    injects error straight into the other axis.
    """

    # Same role as ControlLaw._NOMINAL_DT: the slew limiter needs a timestep on
    # the first sample after a reset, or it would be bypassed and the stick would
    # jump on every re-acquisition.
    _NOMINAL_DT = 0.05

    def __init__(self,
                 h_fov_deg=62.2,
                 v_fov_deg=48.8,
                 # Physical up-tilt of the camera mount, degrees. See the
                 # camera-tilt section above: without it, flying fast points the
                 # lens at the ground and the target leaves the top of the frame.
                 cam_tilt_deg=0.0,
                 # ── yaw: the seeker ──────────────────────────────────────────
                 kp_yaw=4.5,              # rad/s of yaw per rad of azimuth error
                 # ── roll: proportional navigation ────────────────────────────
                 # m/s. Dimensionally an own-speed: a_lat = gain * lambda_dot.
                 # Equal to the real airspeed = coordinated turn; 2-3x = PN lead.
                 roll_pn_gain=45.0,
                 max_bank_deg=30.0,
                 # ── pitch: closing speed ─────────────────────────────────────
                 max_lean_deg=20.0,       # nose-down lean at full commit
                 commit_default=1.0,      # commit when no commit channel is wired
                 # ── throttle: vertical closure ───────────────────────────────
                 climb_gain=8.0,          # m/s of climb per rad of elevation error
                 climb_lead=3.0,          # m/s of climb per rad/s of LOS el rate
                 # ── shaping / limits ─────────────────────────────────────────
                 deadband_frac=0.02,      # of the half-FOV, angular
                 stick_limit=0.90,
                 stick_limit_yaw=None,
                 stick_limit_roll=None,
                 stick_limit_pitch=None,
                 stick_limit_throttle=None,
                 slew_per_s=11.0,
                 rate_tau=0.12,           # LOS-rate filter time constant
                 # ── latency compensation ─────────────────────────────────────
                 # The grid search found every configuration unstable beyond
                 # ~0.2 s of detection latency, and at intercept speed 0.2 s is
                 # metres of stale geometry. The fix is to steer at where the
                 # target WILL be, not where it was seen.
                 latency_s=0.0,           # 0 = measure it live
                 latency_max_s=0.35,      # never extrapolate further than this
                 # ── terminal phase ───────────────────────────────────────────
                 target_width_m=0.20,     # a human head, for the range estimate
                 terminal_bbox_frac=0.35,
                 terminal_range_m=2.5,
                 terminal_hold_s=1.0,
                 # ── axis polarity ────────────────────────────────────────────
                 invert_yaw=False,
                 invert_pitch=False,
                 invert_roll=False,
                 invert_throttle=False,
                 # ── filled in from the FC at connect ─────────────────────────
                 max_yaw_rate=None,       # rad/s at full yaw stick
                 angle_max=None,          # rad of lean at full roll/pitch stick
                 max_climb_mps=2.5,       # m/s at full throttle stick
                 max_pitch_rate=None):    # ACRO compatibility only
        self.h_fov = math.radians(float(h_fov_deg))
        self.v_fov = math.radians(float(v_fov_deg))
        self.cam_tilt = math.radians(float(cam_tilt_deg))
        self.kp_yaw = float(kp_yaw)
        self.roll_pn_gain = max(0.0, float(roll_pn_gain))
        self.max_bank = math.radians(_clamp(float(max_bank_deg), 0.0, 45.0))
        # Hard ceiling on the lean, independent of any gain: a multirotor at
        # 30 deg is accelerating at ~0.58 g.
        self.max_lean = math.radians(_clamp(float(max_lean_deg), 0.0, 35.0))
        self.commit_default = _clamp(float(commit_default), 0.0, 1.0)
        self.climb_gain = float(climb_gain)
        self.climb_lead = float(climb_lead)
        self.deadband_frac = float(deadband_frac)
        self.stick_limit = float(_clamp(stick_limit, 0.0, 1.0))
        self.slew_per_s = float(slew_per_s)
        self.rate_tau = float(rate_tau)
        self.latency_s = max(0.0, float(latency_s))
        self.latency_max_s = max(0.0, float(latency_max_s))
        self.target_width_m = max(1e-3, float(target_width_m))
        self.terminal_bbox_frac = float(terminal_bbox_frac)
        self.terminal_range_m = float(terminal_range_m)
        self.terminal_hold_s = float(terminal_hold_s)
        self.invert_yaw = bool(invert_yaw)
        self.invert_pitch = bool(invert_pitch)
        self.invert_roll = bool(invert_roll)
        self.invert_throttle = bool(invert_throttle)

        lim = self.stick_limit
        self.stick_limit_yaw = float(_clamp(lim if stick_limit_yaw is None
                                            else stick_limit_yaw, 0.0, 1.0))
        self.stick_limit_roll = float(_clamp(lim if stick_limit_roll is None
                                             else stick_limit_roll, 0.0, 1.0))
        self.stick_limit_pitch = float(_clamp(lim if stick_limit_pitch is None
                                              else stick_limit_pitch, 0.0, 1.0))
        self.stick_limit_throttle = float(_clamp(
            lim if stick_limit_throttle is None else stick_limit_throttle,
            0.0, 1.0))

        # ArduPilot Copter defaults, replaced by the real values at connect.
        self.max_yaw_rate = float(max_yaw_rate or math.radians(202.5))
        self.angle_max = float(angle_max or math.radians(30.0))
        self.max_climb_mps = max(0.1, float(max_climb_mps))
        self.max_pitch_rate = float(max_pitch_rate or math.radians(360.0))

        self._tan_h = math.tan(self.h_fov / 2.0)
        self._tan_v = math.tan(self.v_fov / 2.0)

        # Marks this law for the commander and the preflight report. The
        # `approach` machinery in ControlLaw does not apply here.
        self.pitch_mode = "intercept"

        self._az_rate = _Rate(self.rate_tau)        # world LOS bearing rate
        self._el_rate = _Rate(self.rate_tau)        # world LOS elevation rate
        self._az_body_rate = _Rate(self.rate_tau)   # across-frame rate, for lead
        self._range_rate = _Rate(self.rate_tau)
        self._lam_prev = None                       # unwrapping accumulator
        self._lat_est = None
        self._t_prev = None
        self._sticks = dict(yaw=0.0, roll=0.0, pitch=0.0, throttle=0.0)
        self._terminal_since = None
        self._terminal_cmd = None
        self._range_m = None

    # ── geometry ──────────────────────────────────────────────────────────────
    def _de_rotate(self, ex, ey, roll):
        """Pixel offsets in the BODY camera frame -> offsets in a LEVEL frame.

        The camera is bolted to the body, so a bank of `roll` rotates the whole
        image. Forward map, for a target at level-frame offset (H right,
        V down)::

            ex =  H*cos(roll) + V*sin(roll)
            ey = -H*sin(roll) + V*cos(roll)

        which inverts to the expression below. Check it at roll = +90 deg (right
        wing fully down): a target straight up in the world must appear on the
        image LEFT, and it does.
        """
        if roll is None:
            return ex, ey
        c, s = math.cos(roll), math.sin(roll)
        return ex * c - ey * s, ex * s + ey * c

    def _angles(self, ex, ey, fw, fh):
        """Level-frame pixel offsets -> azimuth (right +) and camera-frame
        elevation (UP +), via the pinhole/tangent model."""
        nx = _clamp(2.0 * ex / max(1, fw), -1.0, 1.0)
        ny = _clamp(2.0 * ey / max(1, fh), -1.0, 1.0)
        az = math.atan(nx * self._tan_h)
        el = -math.atan(ny * self._tan_v)       # image y is DOWN, elevation is UP
        return az, el

    def _unwrap(self, lam):
        """Keep the world LOS bearing continuous across the +/-pi seam.

        Differentiating a wrapped heading would inject a 2*pi spike every time the
        airframe passes through north — which, banked at the gain PN runs at,
        would be a full-deflection roll command out of nowhere.
        """
        if self._lam_prev is not None:
            while lam - self._lam_prev > math.pi:
                lam -= 2.0 * math.pi
            while lam - self._lam_prev < -math.pi:
                lam += 2.0 * math.pi
        self._lam_prev = lam
        return lam

    def range_m(self, bbox_w, fw):
        """Range from apparent target width. Rough — an assumed head size against
        a jittering bbox — so it is used for the terminal trigger and the log,
        never in the loop gain."""
        if not bbox_w or bbox_w <= 0 or not fw:
            return None
        f_px = (fw / 2.0) / self._tan_h
        return f_px * self.target_width_m / float(bbox_w)

    # ── the law ───────────────────────────────────────────────────────────────
    def compute(self, err_x, err_y, fw, fh, dt=None, bbox_w=None, bbox_h=None,
                att=None, commit=None, capture_age=None):
        """Pixel offset -> four stick deflections.

        err_x/err_y  bbox centre minus frame centre, DISPLAY pixels.
                     err_x > 0 target right, err_y > 0 target below.
        att          dict(roll, pitch, yawspeed, pitchspeed) in rad and rad/s
                     from the FC's ATTITUDE, or None. Positive pitch is NOSE UP.
        commit       0..1 "how hard to go", from the pilot's commit channel.
                     None falls back to `commit_default`.
        capture_age  seconds between the frame being captured and this call, if
                     the caller can measure it. Feeds the latency estimate.
        """
        now = perf_counter()
        if dt is None:
            dt = (now - self._t_prev) if self._t_prev is not None else 0.0
        self._t_prev = now

        roll = pitch = yaw = None
        if att:
            roll = att.get("roll")
            pitch = att.get("pitch")
            yaw = att.get("yaw")
        have_att = pitch is not None

        # ── 1. measurement: de-rotate, then resolve into the world frame ──────
        ex, ey = self._de_rotate(err_x, err_y, roll)
        az, el_cam = self._angles(ex, ey, fw, fh)
        # The camera's boresight elevation in the world is the airframe's pitch
        # plus the mount's up-tilt; the target's world elevation is that plus its
        # elevation off the boresight.
        el_world = (el_cam + pitch + self.cam_tilt) if have_att else None

        rng = self.range_m(bbox_w, fw)
        if rng is not None:
            self._range_m = rng
            closing = -self._range_rate.update(rng, dt)
        else:
            closing = 0.0

        # ── 2. LOS rates, each from ONE world-referenced signal ──────────────
        # See the class docstring: splitting these into "filtered pixel rate plus
        # instantaneous body rate" is algebraically identical and dynamically
        # unstable, because the two halves have different lag.
        los_az = 0.0
        if yaw is not None:
            los_az = self._az_rate.update(self._unwrap(yaw + az), dt)
        # el_world already contains the airframe's pitch, so it is likewise a
        # single world-referenced signal and needs no rate term added.
        los_el = self._el_rate.update(el_world, dt) if el_world is not None else 0.0
        # For the LEAD term only: how fast the target is moving across the frame,
        # which is what a fixed pipeline delay actually smears.
        d_az_body = self._az_body_rate.update(az, dt)

        # ── 3. latency, and where the target will be ─────────────────────────
        lat = self._latency(capture_age, dt)
        az_lead = az + d_az_body * lat
        el_lead = (el_world + los_el * lat) if el_world is not None else None

        # ── 4. angular deadbands, applied to the LED angles ──────────────────
        # On the angle, not the pixel: with tilt compensation a zero WORLD
        # elevation is not a zero pixel offset, so a pixel deadband would sit in
        # the wrong place.
        half_h, half_v = self.h_fov / 2.0, self.v_fov / 2.0
        az_eff = _deadband(_clamp(az_lead / half_h, -1.0, 1.0),
                           self.deadband_frac) * half_h
        if el_lead is None:
            el_eff = 0.0
        else:
            el_eff = _deadband(_clamp(el_lead / half_v, -1.0, 1.0),
                               self.deadband_frac) * half_v

        # ── 5. YAW — keep the seeker on the target ───────────────────────────
        yaw_stick = (self.kp_yaw * az_eff) / self.max_yaw_rate

        # ── 6. ROLL — proportional navigation ────────────────────────────────
        # Zero without a measured heading: the world LOS bearing cannot be formed
        # from pixels alone, and the body-frame rate is not a substitute — the
        # seeker holds it near zero by construction.
        a_lat = 0.0 if yaw is None else self.roll_pn_gain * los_az
        bank = _clamp(math.atan2(a_lat, 9.80665), -self.max_bank, self.max_bank)
        roll_stick = bank / self.angle_max

        # ── 7. PITCH — closing speed, nose DOWN ──────────────────────────────
        cmt = self.commit_default if commit is None else _clamp(commit, 0.0, 1.0)
        # Do not drive hard while the nose is pointed away from the target: turn
        # first, accelerate as the bearing closes. cos() falls to zero at 90 deg
        # off-boresight and is clamped so it can never command reverse.
        aim_gate = max(0.0, math.cos(az))
        lean = -self.max_lean * cmt * aim_gate          # negative = nose down
        pitch_stick = lean / self.angle_max

        # ── 8. THROTTLE — vertical closure ───────────────────────────────────
        # A missing attitude commands ZERO climb, never a guess: without measured
        # pitch we cannot tell a target above us from a nose-down attitude, and
        # the two want opposite corrections.
        if el_lead is None:
            climb = 0.0
        else:
            climb = self.climb_gain * el_eff + self.climb_lead * los_el
        thr_stick = climb / self.max_climb_mps

        # ── 9. polarity, ceilings, slew ──────────────────────────────────────
        if self.invert_yaw:
            yaw_stick = -yaw_stick
        if self.invert_roll:
            roll_stick = -roll_stick
        if self.invert_pitch:
            pitch_stick = -pitch_stick
        if self.invert_throttle:
            thr_stick = -thr_stick

        yaw_stick = _clamp(yaw_stick, -self.stick_limit_yaw, self.stick_limit_yaw)
        roll_stick = _clamp(roll_stick, -self.stick_limit_roll, self.stick_limit_roll)
        pitch_stick = _clamp(pitch_stick, -self.stick_limit_pitch,
                             self.stick_limit_pitch)
        thr_stick = _clamp(thr_stick, -self.stick_limit_throttle,
                           self.stick_limit_throttle)

        yaw_stick = self._slew("yaw", yaw_stick, dt)
        roll_stick = self._slew("roll", roll_stick, dt)
        pitch_stick = self._slew("pitch", pitch_stick, dt)
        thr_stick = self._slew("throttle", thr_stick, dt)

        # Report the PHYSICAL quantity that will actually be flown — i.e. derived
        # back out of the stick AFTER the ceiling and the slew limiter, and with
        # any inversion flag undone so the number is the physical intent rather
        # than the wire value.  The raw law output would overstate every command
        # the moment anything saturated, and these numbers drive the HUD gauges,
        # the CSV and the closed-loop test's plant.
        out_yaw = -yaw_stick if self.invert_yaw else yaw_stick
        out_roll = -roll_stick if self.invert_roll else roll_stick
        out_pitch = -pitch_stick if self.invert_pitch else pitch_stick
        out_thr = -thr_stick if self.invert_throttle else thr_stick
        cmd = dict(
            yaw_stick=yaw_stick,
            roll_stick=roll_stick,
            pitch_stick=pitch_stick,
            throttle_stick=thr_stick,
            yaw_rate=out_yaw * self.max_yaw_rate,
            pitch_rate=0.0,
            az_deg=math.degrees(az),
            el_deg=math.degrees(el_world) if el_world is not None else 0.0,
            el_cam_deg=math.degrees(el_cam),
            bank_deg=math.degrees(out_roll * self.angle_max),
            lean_deg=math.degrees(out_pitch * self.angle_max),
            climb_mps=out_thr * self.max_climb_mps,
            los_az_dps=math.degrees(los_az),
            los_el_dps=math.degrees(los_el),
            range_m=rng,
            closing_mps=closing,
            latency_s=lat,
            commit=cmt,
            have_att=bool(have_att),
            terminal=False,
        )

        # ── 10. terminal commit ──────────────────────────────────────────────
        return self._terminal(cmd, bbox_h, fh, rng, dt)

    def _latency(self, capture_age, dt):
        """Seconds to extrapolate the target forward by.

        An explicit `latency_s` wins. Otherwise prefer a real measured
        capture-to-here age; failing that, fall back to the detection interval,
        which is a lower bound on the true pipeline delay (one frame must elapse
        before a detection can exist). Clamped, because extrapolating a running
        human on a stale, jittering track diverges fast.
        """
        if self.latency_s > 0.0:
            return min(self.latency_s, self.latency_max_s)
        obs = capture_age if (capture_age is not None and capture_age > 0) else dt
        if obs and obs > 0:
            self._lat_est = obs if self._lat_est is None else \
                self._lat_est + 0.2 * (obs - self._lat_est)
        return min(self._lat_est or 0.0, self.latency_max_s)

    def _terminal(self, cmd, bbox_h, fh, rng, dt):
        """Freeze guidance for the last stretch.

        Inside a metre or two the head fills the frame, the detector (trained on
        distant heads) starts dropping out and the pixel error goes large and
        meaningless — so the last half-second is exactly where a live loop veers
        off. Past the trigger the sticks are held at their last value and the
        airframe flies the trajectory it already had.

        Latched, so one lost detection does not un-commit, and time-limited, so a
        FALSE trigger cannot fly away holding a lean for ever.

        The hold is measured in ACCUMULATED dt rather than wall clock, so it means
        the same thing to the closed-loop sim as it does in flight (where dt IS
        wall clock) — and so it is testable at all.
        """
        big = (bbox_h is not None and fh and
               (bbox_h / float(fh)) >= self.terminal_bbox_frac)
        close = (rng is not None and rng <= self.terminal_range_m)
        if self._terminal_since is None:
            if not (big or close):
                return cmd
            self._terminal_since = 0.0
            self._terminal_cmd = dict(cmd)
        else:
            self._terminal_since += max(0.0, dt)
            if self._terminal_since > self.terminal_hold_s:
                # Hold expired without an impact. Drop the latch and let live
                # guidance resume from a clean filter state on the next frame.
                return self.neutral()
        held = dict(self._terminal_cmd)
        held["terminal"] = True
        held["range_m"] = rng
        # Keep the slew state honest, or the first frame after a release would
        # ramp from a stale value.
        for k, ax in (("yaw_stick", "yaw"), ("roll_stick", "roll"),
                      ("pitch_stick", "pitch"), ("throttle_stick", "throttle")):
            self._sticks[ax] = held[k]
        return held

    def _slew(self, axis, target, dt):
        if self.slew_per_s <= 0.0:
            self._sticks[axis] = target
            return target
        prev = self._sticks[axis]
        step = self.slew_per_s * (dt if dt > 1e-6 else self._NOMINAL_DT)
        out = _clamp(target, prev - step, prev + step)
        self._sticks[axis] = out
        return out

    def neutral(self):
        """Centred sticks, history forgotten so the next acquisition does not
        inherit a stale ramp, a stale LOS rate or a latched terminal commit.

        Centred throttle in ALT_HOLD means HOLD ALTITUDE, not idle — which is
        what makes releasing safe here.
        """
        self._az_rate.reset()
        self._el_rate.reset()
        self._az_body_rate.reset()
        self._range_rate.reset()
        self._lam_prev = None
        self._t_prev = None
        self._terminal_since = None
        self._terminal_cmd = None
        for k in self._sticks:
            self._sticks[k] = 0.0
        return dict(yaw_stick=0.0, roll_stick=0.0, pitch_stick=0.0,
                    throttle_stick=0.0, yaw_rate=0.0, pitch_rate=0.0,
                    az_deg=0.0, el_deg=0.0, el_cam_deg=0.0, bank_deg=0.0,
                    lean_deg=0.0, climb_mps=0.0, los_az_dps=0.0, los_el_dps=0.0,
                    range_m=None, closing_mps=0.0, latency_s=0.0, commit=0.0,
                    have_att=False, terminal=False)

    def describe(self):
        return (
            f"INTERCEPT (proportional navigation)  "
            f"FOV {math.degrees(self.h_fov):.1f}x{math.degrees(self.v_fov):.1f} deg  "
            f"cam tilt +{math.degrees(self.cam_tilt):.0f} deg up\n"
            f"                yaw=SEEKER KP {self.kp_yaw:.2f}  "
            f"roll=PN gain {self.roll_pn_gain:.0f} m/s, bank<="
            f"{math.degrees(self.max_bank):.0f} deg\n"
            f"                pitch=LEAN {math.degrees(self.max_lean):.0f} deg "
            f"nose-down x commit  "
            f"throttle=CLIMB {self.climb_gain:.1f} m/s/rad "
            f"+{self.climb_lead:.1f} lead, <={self.max_climb_mps:.1f} m/s\n"
            f"                deadband {self.deadband_frac * 100:.1f}%  "
            f"stick<={self.stick_limit * 100:.0f}%  slew {self.slew_per_s:.1f}/s  "
            f"LOS tau {self.rate_tau:.2f}s  "
            + (f"latency FIXED {self.latency_s * 1000:.0f} ms"
               if self.latency_s > 0 else "latency MEASURED")
            + f" (cap {self.latency_max_s * 1000:.0f} ms)\n"
            f"                terminal: freeze at bbox>="
            f"{self.terminal_bbox_frac * 100:.0f}% of frame or range<="
            f"{self.terminal_range_m:.1f} m, hold {self.terminal_hold_s:.1f}s  "
            f"full stick: yaw {math.degrees(self.max_yaw_rate):.0f} deg/s, "
            f"lean {math.degrees(self.angle_max):.0f} deg")


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND BASE
# ══════════════════════════════════════════════════════════════════════════════

class FcBackend:
    """A transport that can put stick values into the flight controller."""

    name = "none"

    def __init__(self):
        self.linked = False
        self.armed = False
        self.mode_name = "?"
        self.fw_string = "?"
        # None = no evidence yet, True = the FC echoed our sticks back,
        # False = we have been steering long enough that it should have.
        self.override_confirmed = None
        self.blockers = []          # human-readable reasons nothing will move

    def connect(self):
        raise NotImplementedError

    def steer(self, cmd):
        """Push one stick command.  Returns True if it was written to the link."""
        raise NotImplementedError

    def release(self):
        """Hand every channel back to the pilot.  Returns True if written."""
        raise NotImplementedError

    def close(self):
        pass

    def pitch_rad(self, max_age=0.5):
        """Measured pitch angle, positive NOSE-UP, or None if unavailable.

        None is the honest answer for a transport that cannot measure attitude
        (MSP, offline), and the approach controller treats None as "do not command
        a lean" rather than assuming level.
        """
        return None

    def attitude(self, max_age=0.5):
        """dict(roll, pitch, yawspeed, pitchspeed, rollspeed) in rad and rad/s,
        or None if there is no fresh ATTITUDE.

        The intercept law needs all of it: roll to de-rotate the image, pitch to
        resolve elevation into the world frame, and yawspeed because that is
        where the azimuth LOS rate actually lives once yaw is tracking the target.
        """
        return None

    def throttle_frac(self):
        """The pilot's throttle above mid, 0..1.  0 when unknown, which means the
        approach controller asks for nothing.

        VALID ONLY WHILE WE DO NOT OVERRIDE THROTTLE.  It is read out of
        RC_CHANNELS, which reports `radio_in` — and ArduPilot copies an active
        override straight into `radio_in` (that is the mechanism the whole
        confirmation check relies on).  Override throttle and this reads our own
        command back, i.e. it becomes a feedback loop.  See `commit()`.
        """
        return 0.0

    def commit(self):
        """The pilot's "how hard to go" input, 0..1, or None if no commit channel
        is configured.

        This exists because `throttle_frac()` cannot survive the AI taking the
        throttle channel, and the intercept law needs the throttle for vertical
        closure.  So the commit command moves to a channel we never write, which
        keeps it genuinely the pilot's — and gives them a fast abort that does not
        wait on the 1 Hz HEARTBEAT the mode gate depends on.
        """
        return None

    def in_rate_mode(self):
        """True when a stick deflection means what this control law assumes, on
        this FC, right now.

        Named for the original ACRO-only law and kept for call-compatibility;
        `steer_modes` is what actually decides, so the intercept law can gate on
        ALT_HOLD instead.  A backend answers for itself rather than the commander
        string-matching `mode_name`, because only ArduPilot reports a live flight
        mode.  MSP and the offline backend have no equivalent, so they must not be
        gated off by a mode they never report.
        """
        return True

    def status(self):
        return f"{self.name} {self.mode_name} {'ARMED' if self.armed else 'DISARMED'}"


# ══════════════════════════════════════════════════════════════════════════════
# ARDUPILOT  —  RC_CHANNELS_OVERRIDE in ACRO
# ══════════════════════════════════════════════════════════════════════════════

class ArduPilotAcro(FcBackend):
    """Drives ArduPilot Copter in ACRO by overriding the pilot's yaw and pitch
    channels.  Everything about the transmitter (which channel each axis is on,
    its calibrated endpoints, its reversal) is read from the FC — nothing is
    hardcoded.  Channels we do not drive are sent as 0, which ArduPilot treats
    as "no override", so the pilot keeps roll, throttle and arming.
    """

    name = "ardupilot"

    # ArduPilot's own default GCS system id.  We start here so overrides work
    # even if the SYSID_MYGCS read fails, then adopt whatever the FC reports.
    DEFAULT_GCS_SYSID = 255

    # RC_OPTIONS bit 1 -> ignore all GCS overrides.
    RC_OPT_IGNORE_OVERRIDES = 2

    def __init__(self, conn, baud=115200, axes=("yaw", "pitch"),
                 source_component=190, force_sysid=None, verbose=True,
                 steer_modes=None, commit_chan=0):
        super().__init__()
        self.conn = conn
        self.baud = int(baud)
        self.axes = tuple(axes)
        self.source_component = int(source_component)
        self.force_sysid = force_sysid
        self.verbose = verbose
        # Which flight modes this control law is valid in.  ACRO for the rate
        # law, ALT_HOLD for the intercept law (whose commands are angles and a
        # climb rate, not rates).
        self.steer_modes = frozenset(steer_modes or RATE_MODES)
        # RC channel carrying the pilot's commit/aggression input, or 0 for none.
        # NEVER overridden — that is the entire point of putting it on its own
        # channel rather than reusing throttle.
        self.commit_chan = int(commit_chan or 0)

        self.master = None
        self.sysid_mygcs = None
        self.rc = {}                  # axis -> {chan,min,max,trim,reversed}
        self.params = {}              # raw param snapshot, for the report
        self.acro_yaw_rate = None     # rad/s at full stick
        self.acro_rp_rate = None
        # Angle-mode (ALT_HOLD) equivalents, read from the FC at connect.
        self.angle_max = None         # rad of lean at full roll/pitch stick
        self.pilot_yaw_rate = None    # rad/s at full yaw stick in ALT_HOLD
        self.max_climb_mps = None     # m/s at full throttle stick
        self.thr_deadzone = 100       # PWM either side of mid that holds altitude

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._rx = None
        self._hb = None
        self._sent_pwm = {}           # chan -> last PWM we asked for
        self._steer_since = None      # when we started steering (for the check)
        self._echo_warned = False
        self._t0 = perf_counter()
        # Measured attitude, for the approach controller's outer loop.  None until
        # the first ATTITUDE arrives; `_attitude_t` lets callers reject stale data
        # rather than lean on a number that stopped updating.
        self._pitch_rad = None
        self._roll_rad = None
        # Heading.  Not a nicety: the world LOS bearing that proportional
        # navigation differentiates is `yaw + azimuth_in_frame`, and it cannot be
        # formed from pixels alone.
        self._yaw_rad = None
        self._attitude_t = None
        self._rollspeed = None
        self._pitchspeed = None
        self._yawspeed = None
        # The pilot's commit input, off a channel we never write.
        self._commit = None
        # The pilot's own throttle stick, 0..1 of the travel above mid.  This is
        # the "come at me" command: the pilot asks for approach with the stick they
        # are already holding, and backing it off is an instant, natural abort.
        self._throttle_frac = 0.0
        self._throttle_chan = None

    # ── link bring-up ─────────────────────────────────────────────────────────
    def connect(self):
        if mavutil is None:
            raise RuntimeError("pymavlink not installed:  pip install pymavlink")

        self._say(f"[FC] opening {self.conn}"
                  + (f" @ {self.baud}" if not str(self.conn).startswith(("udp", "tcp"))
                     else ""))
        self.master = mavutil.mavlink_connection(
            self.conn, baud=self.baud,
            source_system=self.DEFAULT_GCS_SYSID,
            source_component=self.source_component)

        hb = self._wait_vehicle_heartbeat(timeout=15)
        if hb is None:
            raise RuntimeError(
                f"[FC] no autopilot heartbeat on {self.conn} within 15 s.\n"
                "     If this goes through the MAVProxy router, check that the "
                "router is up and owns the FC serial port.")

        ap_id = hb.autopilot
        if ap_id != mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
            raise WrongFirmware(
                f"heartbeat reports autopilot id {ap_id}, not ArduPilot "
                f"({mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA})")

        self.linked = True
        self._apply_heartbeat(hb)
        self._say(f"[FC] ArduPilot  sys={self.master.target_system} "
                  f"comp={self.master.target_component}  mode={self.mode_name}  "
                  f"{'ARMED' if self.armed else 'DISARMED'}")

        # Params MUST be read before the rx thread starts, or it eats the
        # PARAM_VALUE replies.
        self._read_config()
        self._adopt_gcs_sysid()
        self._request_rc_channels_stream()
        self._request_attitude_stream()

        self._stop.clear()
        self._rx = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx.start()
        self._hb = threading.Thread(target=self._hb_loop, daemon=True)
        self._hb.start()

    def _say(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _wait_vehicle_heartbeat(self, timeout):
        """Latch onto the AUTOPILOT, never a GCS.  The router bus also carries
        Mission Planner's and MAVProxy's heartbeats; targeting one of those
        would send our overrides to a ground station."""
        deadline = perf_counter() + timeout
        while perf_counter() < deadline:
            msg = self.master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if msg is None:
                continue
            if msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                continue
            if msg.type == mavutil.mavlink.MAV_TYPE_GCS:
                continue
            self.master.target_system = msg.get_srcSystem()
            self.master.target_component = msg.get_srcComponent()
            return msg
        return None

    def _apply_heartbeat(self, hb):
        self.mode_name = COPTER_MODES.get(hb.custom_mode, f"MODE{hb.custom_mode}")
        self.armed = bool(hb.base_mode
                          & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def in_rate_mode(self):
        """ArduPilot reports its flight mode in every HEARTBEAT, and the rx
        thread keeps `mode_name` current, so this tracks the pilot's mode switch
        live.  HEARTBEAT arrives at 1 Hz, which bounds how fast we can notice a
        switch — see AcroCommander's mode gate for why that is acceptable, and
        why a commit channel is worth wiring when the AI holds all four sticks.
        """
        return self.mode_name in self.steer_modes

    # ── FC configuration read-back ────────────────────────────────────────────
    def _param(self, name, timeout=2.5):
        """One synchronous PARAM_REQUEST_READ.  Must run before the rx thread."""
        try:
            self.master.mav.param_request_read_send(
                self.master.target_system, self.master.target_component,
                name.encode(), -1)
        except Exception:
            return None
        deadline = perf_counter() + timeout
        while perf_counter() < deadline:
            msg = self.master.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
            if msg is None:
                continue
            pid = msg.param_id
            if isinstance(pid, bytes):
                pid = pid.decode(errors="ignore")
            if pid.strip("\x00") == name:
                self.params[name] = float(msg.param_value)
                return float(msg.param_value)
        return None

    def _read_config(self):
        """Learn the GCS sysid, the override policy, the ACRO full-stick rates
        and the transmitter map/calibration straight from the FC."""
        # SYSID_MYGCS was renamed MAV_GCS_SYSID in ArduPilot 4.6.
        sysid = self._param("SYSID_MYGCS")
        if sysid is None:
            sysid = self._param("MAV_GCS_SYSID")
        self.sysid_mygcs = int(sysid) if sysid is not None else None

        rc_options = self._param("RC_OPTIONS")
        if rc_options is not None and int(rc_options) & self.RC_OPT_IGNORE_OVERRIDES:
            self.blockers.append(
                f"RC_OPTIONS={int(rc_options)} has bit 1 (IGNORE_OVERRIDES) set — "
                "the FC discards every GCS override.  Clear it: RC_OPTIONS="
                f"{int(rc_options) & ~self.RC_OPT_IGNORE_OVERRIDES}")

        trainer = self._param("ACRO_TRAINER")
        # ACRO_TRAINER only applies in ACRO, so it is only a blocker when that is
        # the mode we steer in.  In ALT_HOLD the FC levels and limits lean by
        # design and the intercept law is built around that.
        if (trainer is not None and int(trainer) != 0
                and "ACRO" in self.steer_modes):
            self.blockers.append(
                f"ACRO_TRAINER={int(trainer)} auto-levels and limits lean angle, "
                "which fights a sustained pitch command.  Set ACRO_TRAINER=0.")

        y_rate = self._param("ACRO_Y_RATE")          # deg/s at full yaw stick
        rp_rate = self._param("ACRO_RP_RATE")        # deg/s at full pitch stick
        self.acro_yaw_rate = math.radians(y_rate) if y_rate else math.radians(202.5)
        self.acro_rp_rate = math.radians(rp_rate) if rp_rate else math.radians(360.0)

        # ── angle-mode (ALT_HOLD) stick scaling ───────────────────────────────
        # In an angle mode the stick means something completely different: full
        # roll/pitch is ANGLE_MAX of lean, full throttle is PILOT_SPEED_UP of
        # climb rate, and full yaw is PILOT_Y_RATE.  Reading them means the
        # intercept law's stick mapping matches THIS airframe rather than a guess,
        # exactly as the ACRO rates already did.
        angle_max = self._param("ANGLE_MAX")             # centidegrees
        self.angle_max = (math.radians(angle_max / 100.0) if angle_max
                          else math.radians(30.0))
        py_rate = self._param("PILOT_Y_RATE")            # deg/s, 4.3+
        if py_rate is None:
            py_rate = y_rate                             # older firmware
        self.pilot_yaw_rate = (math.radians(py_rate) if py_rate
                               else math.radians(202.5))
        up = self._param("PILOT_SPEED_UP")               # cm/s
        self.max_climb_mps = (up / 100.0) if up else 2.5
        dz = self._param("THR_DZ")
        self.thr_deadzone = int(dz) if dz is not None else 100

        for axis in self.axes:
            self.rc[axis] = self._read_axis(axis)

        # Throttle calibration is needed even when throttle is not one of our
        # axes, because the approach controller reads the pilot's stick off it.
        # Loaded quietly: a failure here degrades the approach to zero lean, it
        # must not block steering.
        if "throttle" not in self.rc:
            try:
                self.rc["throttle"] = self._read_axis("throttle", announce=False)
            except Exception:
                pass

        # The commit channel is an input only.  Its calibration comes from the
        # same RC{n}_MIN/MAX so a non-standard endpoint still reads 0..1.
        if self.commit_chan:
            ch = self.commit_chan
            mn = self._param(f"RC{ch}_MIN")
            mx = self._param(f"RC{ch}_MAX")
            self.rc["commit"] = dict(chan=ch,
                                     min=int(mn) if mn else 1000,
                                     max=int(mx) if mx else 2000,
                                     trim=1500, reversed=False,
                                     from_fc=None not in (mn, mx))
            self._say(f"[FC][RC] commit   ch{ch}  "
                      f"min={self.rc['commit']['min']} "
                      f"max={self.rc['commit']['max']}  (input only, never "
                      f"overridden)")
            if ch in {cfg["chan"] for a, cfg in self.rc.items()
                      if a in self.axes}:
                self.blockers.append(
                    f"COMMIT_CHAN={ch} is also a driven axis.  The commit input "
                    "must be on a channel we never write, or it reads our own "
                    "override back.  Move it to a spare channel.")

    def _read_axis(self, axis, announce=True):
        fallback_chan = {"yaw": 4, "pitch": 2, "roll": 1, "throttle": 3}
        ch = self._param(f"RCMAP_{axis.upper()}")
        ch = int(ch) if ch else fallback_chan.get(axis, 0)
        mn = self._param(f"RC{ch}_MIN")
        mx = self._param(f"RC{ch}_MAX")
        tr = self._param(f"RC{ch}_TRIM")
        rv = self._param(f"RC{ch}_REVERSED")
        from_fc = None not in (mn, mx, tr)
        cfg = dict(chan=ch,
                   min=int(mn) if mn else 1000,
                   max=int(mx) if mx else 2000,
                   trim=int(tr) if tr else 1500,
                   reversed=bool(int(rv)) if rv is not None else False,
                   from_fc=from_fc)
        src = "from FC" if from_fc else "DEFAULTS (param read failed!)"
        warn = "" if 1 <= cfg["chan"] <= 8 else "  [chan>8 cannot be overridden!]"
        if announce:
            self._say(f"[FC][RC] {axis:8s} ch{cfg['chan']}  "
                      f"min={cfg['min']} trim={cfg['trim']} max={cfg['max']} "
                      f"rev={int(cfg['reversed'])}  ({src}){warn}")
        if not announce:
            return cfg
        if not 1 <= cfg["chan"] <= 8:
            self.blockers.append(
                f"{axis} is on RC channel {cfg['chan']}; RC_CHANNELS_OVERRIDE only "
                "reaches channels 1-8.  Remap RCMAP_%s." % axis.upper())
        return cfg

    def _adopt_gcs_sysid(self):
        """THE fix.  ArduPilot drops RC_CHANNELS_OVERRIDE from any system id
        other than SYSID_MYGCS, silently.  Become that system id."""
        want = self.force_sysid if self.force_sysid is not None else self.sysid_mygcs
        if want is None:
            want = self.DEFAULT_GCS_SYSID
            self._say(f"[FC] could not read SYSID_MYGCS — assuming the ArduPilot "
                      f"default ({want}).  If overrides are refused, set "
                      f"MAV_SOURCE_SYSTEM to the FC's real SYSID_MYGCS.")
        want = int(want)
        self.master.mav.srcSystem = want
        try:
            self.master.source_system = want
        except Exception:
            pass
        note = "" if self.sysid_mygcs is None else f" (FC SYSID_MYGCS={self.sysid_mygcs})"
        self._say(f"[FC] sending overrides as system id {want}{note} — ArduPilot "
                  f"only accepts RC_CHANNELS_OVERRIDE from SYSID_MYGCS")

    def _request_rc_channels_stream(self, hz=10.0):
        """Ask for RC_CHANNELS so we can verify the override actually landed.
        ArduPilot echoes the overridden value back in RC_CHANNELS.

        10 Hz rather than 5: this stream now also carries the pilot's commit
        channel, which is the fastest abort path available, so its latency is a
        safety number and not just a diagnostic one.
        """
        try:
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS,
                int(1e6 / max(1.0, hz)), 0, 0, 0, 0, 0)
        except Exception:
            pass
        try:                       # legacy fallback for older firmware
            self.master.mav.request_data_stream_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, int(hz), 1)
        except Exception:
            pass

    def _request_attitude_stream(self, hz=20.0):
        """Ask for ATTITUDE.  Required by the approach controller: holding a fixed
        forward lean is an ATTITUDE problem, and ACRO gives us only a rate input,
        so the outer loop has to be closed on the FC's own measured pitch.

        ArduPilot sends nothing but HEARTBEAT until a GCS asks, so this is not
        optional — measured on the real board: no request, no ATTITUDE at all;
        with the request, a clean 20 Hz.
        """
        try:
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
                int(1e6 / max(1.0, hz)), 0, 0, 0, 0, 0)
        except Exception:
            pass
        try:                       # legacy fallback for older firmware
            self.master.mav.request_data_stream_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, int(hz), 1)
        except Exception:
            pass

    # ── background rx / heartbeat ─────────────────────────────────────────────
    def _rx_loop(self):
        while not self._stop.is_set():
            try:
                msg = self.master.recv_match(blocking=True, timeout=1)
            except Exception:
                continue
            if msg is None:
                continue
            t = msg.get_type()
            if (t == "HEARTBEAT"
                    and msg.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID
                    and msg.type != mavutil.mavlink.MAV_TYPE_GCS):
                with self._lock:
                    self._apply_heartbeat(msg)
            elif t == "RC_CHANNELS":
                self._check_echo(msg)
                self._apply_pilot_throttle(msg)
                self._apply_commit(msg)
            elif t == "ATTITUDE":
                with self._lock:
                    # MAVLink ATTITUDE.pitch is positive NOSE-UP.  The *speeds
                    # are body angular rates in rad/s.
                    self._pitch_rad = float(msg.pitch)
                    self._roll_rad = float(msg.roll)
                    self._yaw_rad = float(msg.yaw)
                    self._rollspeed = float(msg.rollspeed)
                    self._pitchspeed = float(msg.pitchspeed)
                    self._yawspeed = float(msg.yawspeed)
                    self._attitude_t = perf_counter()

    def _hb_loop(self):
        """Announce ourselves as a GCS at 1 Hz.  Standard behaviour for anything
        holding SYSID_MYGCS, and it makes the link visible in Mission Planner."""
        while not self._stop.wait(1.0):
            try:
                self.master.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            except Exception:
                pass

    def _apply_pilot_throttle(self, rc_msg):
        """Track how far the pilot has the throttle above mid, as 0..1.

        Read from RC_CHANNELS, which is the pilot's real stick — we never override
        throttle, so this is genuinely their input.  Calibration comes from the
        FC's own RC{n}_MIN/TRIM/MAX, so a transmitter with an offset throttle trim
        still reads correctly.  Mid is taken as the midpoint of MIN..MAX rather
        than TRIM, because on a throttle channel TRIM sits at the bottom of the
        travel (measured on this airframe: RC1 min 988, trim 990, max 2010) and
        using it would report near-full throttle at hover.
        """
        cfg = self.rc.get("throttle")
        if cfg is None:
            return
        chan = cfg["chan"]
        if not (1 <= chan <= 18):
            return
        raw = getattr(rc_msg, f"chan{chan}_raw", 0)
        if not raw or raw == 65535:
            return
        lo, hi = float(cfg["min"]), float(cfg["max"])
        if hi - lo < 1.0:
            return
        mid = 0.5 * (lo + hi)
        if cfg.get("reversed"):
            frac = (mid - raw) / (mid - lo)
        else:
            frac = (raw - mid) / (hi - mid)
        with self._lock:
            self._throttle_chan = chan
            self._throttle_frac = _clamp(frac, 0.0, 1.0)

    def pitch_rad(self, max_age=0.5):
        """Measured pitch (positive NOSE-UP), or None if it is missing or stale.

        Returning None rather than a last-known value is deliberate: the approach
        controller commands a rate to correct an attitude error, and a frozen
        attitude reading would make it command that rate forever.
        """
        with self._lock:
            t, p = self._attitude_t, self._pitch_rad
        if t is None or p is None:
            return None
        return None if (perf_counter() - t) > max_age else p

    def throttle_frac(self):
        with self._lock:
            return self._throttle_frac

    def attitude(self, max_age=0.5):
        """Full attitude + body rates, or None if missing or stale.

        One accessor rather than five, so a caller can never end up mixing a
        fresh roll with a stale yaw rate.  Staleness is judged once, on the
        message timestamp, and returns None for the lot.
        """
        with self._lock:
            t = self._attitude_t
            if t is None or self._pitch_rad is None:
                return None
            if (perf_counter() - t) > max_age:
                return None
            return dict(roll=self._roll_rad, pitch=self._pitch_rad,
                        yaw=self._yaw_rad,
                        rollspeed=self._rollspeed,
                        pitchspeed=self._pitchspeed,
                        yawspeed=self._yawspeed, age=perf_counter() - t)

    def commit(self):
        with self._lock:
            return self._commit

    def _apply_commit(self, rc_msg):
        """Read the pilot's commit input off its own channel.

        Genuinely the pilot's, because this channel is never in `axes` and is
        therefore never overridden — unlike throttle once the intercept law owns
        it.  Also the fastest abort available: RC_CHANNELS streams far quicker
        than the 1 Hz HEARTBEAT the mode gate has to wait for.
        """
        cfg = self.rc.get("commit")
        if cfg is None:
            return
        raw = getattr(rc_msg, f"chan{cfg['chan']}_raw", 0)
        if not raw or raw == 65535:
            return
        lo, hi = float(cfg["min"]), float(cfg["max"])
        if hi - lo < 1.0:
            return
        with self._lock:
            self._commit = _clamp((raw - lo) / (hi - lo), 0.0, 1.0)

    def _check_echo(self, rc_msg):
        """Confirm the FC took our sticks: ArduPilot copies an active override
        into radio_in, which is what RC_CHANNELS reports."""
        with self._lock:
            steer_since = self._steer_since
            sent = dict(self._sent_pwm)
        if steer_since is None or not sent:
            return
        held = perf_counter() - steer_since
        if held < 0.4:                      # give the FC a couple of frames
            return

        # Only judge on a channel whose commanded value is clearly away from its
        # trim.  A stick sitting AT trim reads the same whether the override was
        # applied or the pilot's centred radio is being reported, so matching on
        # it would manufacture a false "confirmed" and hide a real refusal.
        distinctive = {c: p for c, p in sent.items()
                       if any(abs(p - cfg["trim"]) > 40
                              for cfg in self.rc.values() if cfg["chan"] == c)}
        if not distinctive:
            return

        matched = True
        for chan, pwm in distinctive.items():
            got = getattr(rc_msg, f"chan{chan}_raw", None)
            if got is None or abs(int(got) - int(pwm)) > 25:
                matched = False
                break

        if matched:
            if self.override_confirmed is not True:
                self._say("[FC] override CONFIRMED — the FC is echoing our "
                          "stick values back in RC_CHANNELS")
            self.override_confirmed = True
        elif held > 2.0:
            self.override_confirmed = False
            if not self._echo_warned:
                self._echo_warned = True
                detail = ", ".join(
                    f"ch{c}: sent {p}, FC reports "
                    f"{getattr(rc_msg, f'chan{c}_raw', '?')}"
                    for c, p in sorted(sent.items()))
                print("\n[FC][REFUSED] The FC is NOT applying our RC override.\n"
                      f"              {detail}\n"
                      "              Most likely causes, in order:\n"
                      f"                1. SYSID_MYGCS on the FC != {self.master.mav.srcSystem} "
                      "(we send from that id; ArduPilot drops any other).\n"
                      "                2. RC_OPTIONS bit 1 (value 2) set = ignore overrides.\n"
                      "                3. The override is not reaching the FC "
                      "(router not forwarding).\n"
                      "              Run fc_check.py --verify for a full report.",
                      flush=True)

    # ── stick output ──────────────────────────────────────────────────────────
    def _axis_pwm(self, axis, norm):
        """Normalized deflection (-1..1) -> PWM, using the FC's own calibration
        so an asymmetric trim is honoured."""
        if axis == "throttle":
            return self._throttle_pwm(norm)
        cfg = self.rc[axis]
        norm = _clamp(norm, -1.0, 1.0)
        if cfg["reversed"]:
            norm = -norm
        if norm >= 0.0:
            pwm = cfg["trim"] + norm * (cfg["max"] - cfg["trim"])
        else:
            pwm = cfg["trim"] + norm * (cfg["trim"] - cfg["min"])
        return int(round(_clamp(pwm, cfg["min"], cfg["max"])))

    def _throttle_pwm(self, norm):
        """Climb-rate command (-1..1) -> throttle PWM, for an ANGLE MODE.

        Throttle cannot go through `_axis_pwm`, for two independent reasons:

        1. **TRIM is not the centre.**  On a throttle channel the transmitter's
           trim sits at the BOTTOM of the travel (measured on this airframe:
           min 988, trim 990, max 2010), so pivoting on trim would read and write
           near-idle as "centre".  Mid is the midpoint of MIN..MAX.

        2. **THR_DZ.**  ArduPilot holds altitude while the throttle sits within
           `THR_DZ` PWM of mid, so any command inside that band is silently zero.
           A proportional map would therefore have a dead zone with a step at its
           edge.  This escapes the band first and scales the remainder, the same
           trick `_deadband` uses on the input side.

        A command of exactly 0 returns exactly mid — HOLD ALTITUDE.  That matters
        beyond neatness: it is what makes a frozen or released throttle safe here,
        where in ACRO the same PWM would be an arbitrary thrust setting.
        """
        cfg = self.rc["throttle"]
        norm = _clamp(norm, -1.0, 1.0)
        if cfg["reversed"]:
            norm = -norm
        lo, hi = float(cfg["min"]), float(cfg["max"])
        mid = 0.5 * (lo + hi)
        if norm == 0.0:
            return int(round(mid))
        half = 0.5 * (hi - lo)
        dz = float(_clamp(self.thr_deadzone, 0.0, max(0.0, half - 1.0)))
        pwm = mid + math.copysign(dz + abs(norm) * (half - dz), norm)
        return int(round(_clamp(pwm, lo, hi)))

    _AXIS_FIELD = {"yaw": "yaw_stick", "pitch": "pitch_stick",
                   "roll": "roll_stick", "throttle": "throttle_stick"}

    def steer(self, cmd):
        chans = [0] * 8               # 0 = "no override", pilot keeps the channel
        sent = {}
        for axis in self.axes:
            cfg = self.rc.get(axis)
            if not cfg or not 1 <= cfg["chan"] <= 8:
                continue
            pwm = self._axis_pwm(axis, cmd.get(self._AXIS_FIELD[axis], 0.0))
            chans[cfg["chan"] - 1] = pwm
            sent[cfg["chan"]] = pwm
        with self._lock:
            self._sent_pwm = sent
            if self._steer_since is None:
                self._steer_since = perf_counter()
        return self._send_override(chans)

    def release(self):
        with self._lock:
            self._sent_pwm = {}
            self._steer_since = None
        return self._send_override([0] * 8)

    def _send_override(self, chans):
        if self.master is None:
            return False
        try:
            self.master.mav.rc_channels_override_send(
                self.master.target_system, self.master.target_component, *chans)
            return True
        except Exception:
            # A connected UDP socket can raise a transient ECONNREFUSED when a
            # peer blips; that must never kill the control stream.
            return False

    def statustext(self, text):
        if self.master is None:
            return
        try:
            self.master.mav.statustext_send(
                mavutil.mavlink.MAV_SEVERITY_INFO, text[:50].encode())
        except Exception:
            pass

    def close(self):
        self._stop.set()
        for th in (self._rx, self._hb):
            if th is not None:
                th.join(timeout=2)
        if self.master is not None:
            try:                       # several times: a datagram can be dropped
                for _ in range(5):
                    self._send_override([0] * 8)
            except Exception:
                pass
            try:
                self.master.close()
            except Exception:
                pass
            self.master = None

    # ── report ────────────────────────────────────────────────────────────────
    def describe(self):
        want = "/".join(sorted(self.steer_modes))
        lines = [f"firmware      : ArduPilot Copter",
                 f"mode          : {self.mode_name}"
                 + ("" if self.mode_name in self.steer_modes
                    else f"   <-- {want} is what this expects"),
                 f"armed         : {'YES' if self.armed else 'NO'}",
                 f"our sysid     : {self.master.mav.srcSystem if self.master else '?'}"
                 f"   (FC SYSID_MYGCS = {self.sysid_mygcs})",
                 f"ACRO rates    : yaw {math.degrees(self.acro_yaw_rate):.0f} deg/s, "
                 f"roll/pitch {math.degrees(self.acro_rp_rate):.0f} deg/s at full stick"]
        if self.angle_max is not None:
            lines.append(
                f"angle mode    : lean <= {math.degrees(self.angle_max):.0f} deg "
                f"(ANGLE_MAX), yaw {math.degrees(self.pilot_yaw_rate):.0f} deg/s "
                f"(PILOT_Y_RATE), climb <= {self.max_climb_mps:.1f} m/s "
                f"(PILOT_SPEED_UP), THR_DZ {self.thr_deadzone}")
        for axis, cfg in self.rc.items():
            lines.append(f"{axis:14s}: ch{cfg['chan']} "
                         f"[{cfg['min']}..{cfg['trim']}..{cfg['max']}] "
                         f"rev={int(cfg['reversed'])}"
                         + ("   (driven)" if axis in self.axes else "   (read only)"))
        for k in ("RC_OPTIONS", "ACRO_TRAINER", "ACRO_Y_EXPO", "ACRO_RP_EXPO",
                  "ANGLE_MAX", "PILOT_SPEED_UP", "THR_DZ"):
            if k in self.params:
                lines.append(f"{k:14s}: {self.params[k]:g}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# BETAFLIGHT / INAV  —  MSP_SET_RAW_RC
# ══════════════════════════════════════════════════════════════════════════════

MSP_API_VERSION = 1
MSP_FC_VARIANT = 2
MSP_FC_VERSION = 3
MSP_RX_MAP = 64
MSP_STATUS = 101
MSP_RC = 105
MSP_SET_RAW_RC = 200


def _msp_frame(cmd, payload=b""):
    """MSP v1 request frame:  '$M<' len cmd payload xor-crc"""
    n = len(payload)
    crc = n ^ cmd
    for b in payload:
        crc ^= b
    return b"$M<" + bytes((n, cmd)) + payload + bytes((crc & 0xFF,))


class BetaflightAcro(FcBackend):
    """Drives Betaflight / INAV over MSP.  Betaflight's ACRO ("rate") mode is
    the default whenever ANGLE/HORIZON are off, so a stick value here is again
    a body-rate command.

    REQUIRED BETAFLIGHT CONFIGURATION — without it the FC ignores MSP sticks:

        feature MSP_OVERRIDE
        set msp_override_channels_mask = 10     # ch2 (pitch) + ch4 (yaw)
        save

    and assign the "MSP RC OVERRIDE" mode to a spare aux switch in the Modes
    tab, so the pilot can enable and kill the tracker's authority in flight.
    With the mask set, Betaflight takes ONLY the masked channels from MSP and
    everything else (throttle, roll, arming) from the real receiver.

    Each cycle the current live stick frame is read back with MSP_RC and used
    as the seed, so the channels we are not driving are echoed rather than
    invented.
    """

    name = "betaflight"

    # Betaflight's rcmap letters, in MSP channel order.
    _MAP_LETTERS = "AERT"

    def __init__(self, conn, baud=115200, axes=("yaw", "pitch"), verbose=True):
        super().__init__()
        self.conn = conn
        self.baud = int(baud)
        self.axes = tuple(axes)
        self.verbose = verbose
        self.ser = None
        self.variant = "?"
        self.chan_index = {}          # axis -> 0-based MSP channel index
        self.n_channels = 8
        self._live = [1500] * 16
        self._sent = {}
        self._steer_since = None
        self._echo_warned = False
        self._io = threading.Lock()

    def _say(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def connect(self):
        try:
            import serial
        except ImportError:
            raise RuntimeError("pyserial not installed:  pip install pyserial")
        self._say(f"[FC] opening {self.conn} @ {self.baud} for MSP")
        self.ser = serial.Serial(self.conn, self.baud, timeout=0.3)

        variant = self._request(MSP_FC_VARIANT, timeout=2.0)
        if not variant:
            raise WrongFirmware("no MSP response — not Betaflight/INAV either")
        self.variant = variant[:4].decode(errors="ignore")
        ver = self._request(MSP_FC_VERSION) or b"\0\0\0"
        self.fw_string = (f"{self.variant} "
                          f"{ver[0]}.{ver[1]}.{ver[2]}" if len(ver) >= 3 else self.variant)
        self.linked = True
        self._say(f"[FC] MSP firmware: {self.fw_string}")

        if self.variant not in ("BTFL", "INAV", "EMUF", "CLFL"):
            self._say(f"[FC] warning: unrecognised MSP variant '{self.variant}'")

        self._read_rx_map()
        self._poll_status()
        self._poll_rc()

        self.blockers.append(
            "Betaflight ignores MSP_SET_RAW_RC unless MSP override is configured: "
            "`feature MSP_OVERRIDE`, `set msp_override_channels_mask="
            f"{self._mask_hint()}`, and the 'MSP RC OVERRIDE' mode assigned to an "
            "aux switch that is currently ON.")

    def _mask_hint(self):
        mask = 0
        for axis in self.axes:
            idx = self.chan_index.get(axis)
            if idx is not None:
                mask |= 1 << idx
        return mask

    # ── MSP plumbing ──────────────────────────────────────────────────────────
    def _request(self, cmd, payload=b"", timeout=1.0):
        """Send an MSP command and return the payload of the matching reply."""
        if self.ser is None:
            return None
        with self._io:
            try:
                self.ser.reset_input_buffer()
                self.ser.write(_msp_frame(cmd, payload))
            except Exception:
                return None
            return self._read_reply(cmd, timeout)

    def _send(self, cmd, payload=b""):
        if self.ser is None:
            return False
        with self._io:
            try:
                self.ser.write(_msp_frame(cmd, payload))
                return True
            except Exception:
                return False

    def _read_reply(self, want_cmd, timeout):
        deadline = perf_counter() + timeout
        state = 0
        size = cmd = 0
        buf = bytearray()
        while perf_counter() < deadline:
            b = self.ser.read(1)
            if not b:
                continue
            c = b[0]
            if state == 0:
                state = 1 if c == ord("$") else 0
            elif state == 1:
                state = 2 if c == ord("M") else 0
            elif state == 2:
                # '>' reply, '!' error
                state = 3 if c in (ord(">"), ord("!")) else 0
            elif state == 3:
                size, state = c, 4
                buf.clear()
            elif state == 4:
                cmd, state = c, 5
                if size == 0:
                    state = 6
            elif state == 5:
                buf.append(c)
                if len(buf) >= size:
                    state = 6
            if state == 6:
                _crc = self.ser.read(1)          # trailing checksum
                if cmd == want_cmd:
                    return bytes(buf)
                state = 0
        return None

    def _read_rx_map(self):
        """MSP channel order is raw; Betaflight's `map` setting (default AETR)
        says which raw index is which axis."""
        raw = self._request(MSP_RX_MAP)
        order = "AETR"
        if raw:
            letters = []
            for idx in raw[:4]:
                letters.append(self._MAP_LETTERS[idx] if idx < 4 else "?")
            # raw[i] = which AERT slot raw channel i carries
            order = "".join(letters)
        axis_letter = {"roll": "A", "pitch": "E", "yaw": "R", "throttle": "T"}
        for axis in ("roll", "pitch", "yaw", "throttle"):
            letter = axis_letter[axis]
            self.chan_index[axis] = order.find(letter) if letter in order else None
        self._say(f"[FC] MSP rx map = {order}  -> "
                  + ", ".join(f"{a}:ch{(self.chan_index[a] or 0) + 1}"
                              for a in self.axes))

    def _poll_status(self):
        raw = self._request(MSP_STATUS)
        if raw and len(raw) >= 11:
            flags = struct.unpack_from("<I", raw, 6)[0]
            self.armed = bool(flags & 1)
            self.mode_name = "ARMED" if self.armed else "DISARMED"
            # Betaflight is in ACRO whenever ANGLE and HORIZON are both off, but
            # their bit positions are box-order dependent, so do not guess here.
            self.mode_name = "ACRO/rate (BF default)" if self.armed else "DISARMED"

    def _poll_rc(self):
        raw = self._request(MSP_RC)
        if raw and len(raw) >= 2:
            n = len(raw) // 2
            self.n_channels = n
            self._live = list(struct.unpack_from("<" + "H" * n, raw, 0)) + \
                [1500] * max(0, 16 - n)
            self._verify_echo()

    def _verify_echo(self):
        if self._steer_since is None or not self._sent:
            return
        if perf_counter() - self._steer_since < 0.4:
            return
        # Same caveat as the MAVLink path: a channel commanded to centre reads the
        # same whether or not the override was applied, so judge only on channels
        # pushed clearly away from centre.
        distinctive = {i: v for i, v in self._sent.items()
                       if abs(v - 1500) > 40 and i < len(self._live)}
        if not distinctive:
            return
        ok = all(abs(self._live[i] - v) <= 25 for i, v in distinctive.items())
        if ok:
            if self.override_confirmed is not True:
                self._say("[FC] MSP override CONFIRMED — the FC is reporting our "
                          "stick values back in MSP_RC")
            self.override_confirmed = True
        elif perf_counter() - self._steer_since > 2.0:
            self.override_confirmed = False
            if not self._echo_warned:
                self._echo_warned = True
                print("\n[FC][REFUSED] Betaflight is NOT applying our MSP sticks.\n"
                      "              Enable MSP override on the FC:\n"
                      "                feature MSP_OVERRIDE\n"
                      f"                set msp_override_channels_mask = {self._mask_hint()}\n"
                      "                save\n"
                      "              then assign 'MSP RC OVERRIDE' to an aux "
                      "switch and turn it ON.", flush=True)

    # ── stick output ──────────────────────────────────────────────────────────
    _AXIS_FIELD = ArduPilotAcro._AXIS_FIELD

    def steer(self, cmd):
        self._poll_rc()                       # seed from the pilot's live frame
        self._poll_status()
        frame = list(self._live[:max(8, self.n_channels)])
        sent = {}
        for axis in self.axes:
            idx = self.chan_index.get(axis)
            if idx is None:
                continue
            norm = _clamp(cmd.get(self._AXIS_FIELD[axis], 0.0), -1.0, 1.0)
            pwm = int(round(_clamp(1500 + norm * 500, 1000, 2000)))
            frame[idx] = pwm
            sent[idx] = pwm
        self._sent = sent
        if self._steer_since is None:
            self._steer_since = perf_counter()
        payload = struct.pack("<" + "H" * len(frame), *frame)
        return self._send(MSP_SET_RAW_RC, payload)

    def release(self):
        """Stop overriding.  With msp_override_channels_mask configured the
        cleanest release is simply to stop sending: Betaflight falls back to the
        receiver after msp_override_failsafe.  Echo the live frame once so there
        is no step if the mask is not set."""
        self._sent = {}
        self._steer_since = None
        self._poll_rc()
        return True

    def statustext(self, text):
        pass

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def describe(self):
        lines = [f"firmware      : {self.fw_string}",
                 f"armed         : {'YES' if self.armed else 'NO'}",
                 f"MSP channels  : {self.n_channels}"]
        for axis in self.axes:
            idx = self.chan_index.get(axis)
            lines.append(f"{axis:14s}: MSP index {idx} "
                         f"(ch{(idx + 1) if idx is not None else '?'})")
        lines.append(f"override mask : {self._mask_hint()} "
                     "(set msp_override_channels_mask to this)")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# OFFLINE  —  no FC attached, commands are computed and printed only
# ══════════════════════════════════════════════════════════════════════════════

class OfflineBackend(FcBackend):
    name = "offline"

    def connect(self):
        self.fw_string = "none (offline)"
        self.mode_name = "SIM"
        print("[FC] no flight controller configured — commands are print-only")

    def steer(self, cmd):
        return False

    def release(self):
        return False

    def statustext(self, text):
        pass

    def describe(self):
        return "firmware      : none (offline / print-only)"


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND FACTORY  —  identify the firmware, then pick the transport
# ══════════════════════════════════════════════════════════════════════════════

def build_backend(conn, baud=115200, axes=("yaw", "pitch"), transport="auto",
                  force_sysid=None, verbose=True, steer_modes=None,
                  commit_chan=0):
    """Return a connected backend for `conn`.

    transport: "auto"       try MAVLink/ArduPilot, fall back to MSP/Betaflight
               "mavlink"    ArduPilot only
               "msp"        Betaflight/INAV only
               "none"       offline

    steer_modes / commit_chan are ArduPilot-only: no other transport reports a
    flight mode or a full RC frame.
    """
    if conn is None or str(conn).lower() in ("none", ""):
        be = OfflineBackend()
        be.connect()
        return be

    order = {"auto": ("mavlink", "msp"),
             "mavlink": ("mavlink",),
             "msp": ("msp",),
             "none": ()}.get(transport, ("mavlink", "msp"))

    errors = []
    for kind in order:
        # MSP needs a real serial port; it cannot run over the UDP router.
        if kind == "msp" and str(conn).startswith(("udp", "tcp")):
            errors.append("msp: needs a serial port, not a UDP/TCP link")
            continue
        cls = ArduPilotAcro if kind == "mavlink" else BetaflightAcro
        kwargs = dict(conn=conn, baud=baud, axes=axes, verbose=verbose)
        if kind == "mavlink":
            kwargs["force_sysid"] = force_sysid
            kwargs["steer_modes"] = steer_modes
            kwargs["commit_chan"] = commit_chan
        be = cls(**kwargs)
        try:
            be.connect()
            return be
        except WrongFirmware as e:
            errors.append(f"{kind}: {e}")
            if verbose:
                print(f"[FC] {kind} transport rejected: {e}", flush=True)
            be.close()
        except Exception as e:
            errors.append(f"{kind}: {e}")
            be.close()
            if transport != "auto":
                raise

    raise RuntimeError("could not talk to the flight controller on "
                       f"{conn}:\n  " + "\n  ".join(errors))


# ══════════════════════════════════════════════════════════════════════════════
# COMMANDER  —  what the tracker talks to
# ══════════════════════════════════════════════════════════════════════════════

class AcroCommander:
    """Owns the control law, the backend, the fixed-rate transmit thread, the
    safety gating and the CSV log.

    Safety model (unchanged in spirit from the RC path it replaces):
      * sticks are driven ONLY while the FC is ARMED and a target is locked;
      * every other instant, all channels are released back to the pilot;
      * on exit the release is sent repeatedly, then the link is closed.
    """

    def __init__(self,
                 conn="udpin:127.0.0.1:14551",
                 baud=115200,
                 transport="auto",
                 axes=("yaw", "pitch"),
                 law=None,
                 send_hz=20.0,
                 statustext_hz=2.0,
                 no_target="release",       # "release" | "hold"
                 target_timeout=0.5,        # s of no detection before releasing
                 force_sysid=None,
                 log_dir=None,
                 require_armed=True,
                 require_rate_mode=True,
                 verbose=True,
                 steer_modes=None,
                 commit_chan=0,
                 commit_abort_below=0.0,
                 engage_max_s=0.0):
        self.conn = conn
        self.baud = baud
        self.transport = transport
        self.axes = tuple(axes)
        self.law = law or ControlLaw()
        # The flight mode in which this law's sticks mean what it thinks they
        # mean.  ACRO for the rate law, ALT_HOLD for the intercept law.
        self.steer_modes = frozenset(steer_modes or RATE_MODES)
        self.commit_chan = int(commit_chan or 0)
        # Below this commit value, release EVERYTHING back to the pilot — not just
        # reduce the lean.  A stick-based abort that does not have to wait for the
        # next 1 Hz HEARTBEAT the way the mode gate does.  0 disables it.
        self.commit_abort_below = float(commit_abort_below)
        # Hard ceiling on one continuous engagement, seconds.  0 = none.  This
        # bounds the worst case when the AI holds all four channels: mode-switch
        # handback is only noticed at HEARTBEAT rate, so without an independent
        # limit a hung detection could hold a lean for up to a second longer than
        # the pilot expects.
        self.engage_max_s = float(engage_max_s)
        self._engaged_since = None
        self._engage_timed_out = False
        self._commit_aborted = False
        self.send_hz = float(send_hz)
        self.statustext_hz = float(statustext_hz)
        self.no_target = no_target
        self.target_timeout = float(target_timeout)
        self.force_sysid = force_sysid
        self.log_dir = log_dir
        self.require_armed = bool(require_armed)
        self.require_rate_mode = bool(require_rate_mode)
        self.verbose = verbose

        self.backend = None
        self._lock = threading.Lock()
        self._cmd = self.law.neutral()
        self._target_off = None
        self._last_target_t = 0.0
        self._stop = threading.Event()
        self._tx = None
        self._log_fh = None
        self.log_path = None
        self._last_stt = 0.0
        self._t0 = perf_counter()
        self._warned_disarmed = False
        self._steering = False
        # None = not yet observed.  Tracks the mode gate so each flip of the
        # pilot's mode switch is announced exactly once, in both directions,
        # however many times they toggle it.
        self._mode_gate = None
        self._rate_mode_ok = True

    # ── HUD/legacy compatibility surface ─────────────────────────────────────
    @property
    def linked(self):
        return bool(self.backend is not None and self.backend.linked)

    @property
    def H_FOV(self):
        return self.law.h_fov

    @property
    def V_FOV(self):
        return self.law.v_fov

    @property
    def max_yaw_rate(self):
        return self.law.max_yaw_rate

    @property
    def max_pitch_rate(self):
        return self.law.max_pitch_rate

    @property
    def armed(self):
        return bool(self.backend is not None and self.backend.armed)

    @property
    def mode_name(self):
        return self.backend.mode_name if self.backend is not None else "?"

    @property
    def is_intercept(self):
        """True when the law commands ANGLES and a CLIMB RATE (free-flight
        intercept) rather than body rates (gimbal visual servo)."""
        return isinstance(self.law, InterceptLaw)

    @property
    def rate_mode_ok(self):
        """False only when the mode gate is active AND the FC is not in a rate
        mode — i.e. the pilot currently holds every channel.  The HUD reads this.
        """
        return bool(self._rate_mode_ok)

    @property
    def override_confirmed(self):
        return None if self.backend is None else self.backend.override_confirmed

    @property
    def steering(self):
        """True right now the sticks are being driven (link up + target fresh +
        FC armed).  Always current, unlike a per-frame return value, so a skip
        frame does not make the HUD claim the stream stopped."""
        return self._steering

    # ── bring-up ──────────────────────────────────────────────────────────────
    def connect(self):
        self.backend = build_backend(self.conn, baud=self.baud, axes=self.axes,
                                     transport=self.transport,
                                     force_sysid=self.force_sysid,
                                     verbose=self.verbose,
                                     steer_modes=self.steer_modes,
                                     commit_chan=self.commit_chan)

        # Adopt the FC's real full-stick scaling so the stick mapping is correct
        # for THIS airframe's tuning rather than a guess.  Which numbers matter
        # depends on the law: a rate law needs the ACRO rates, an angle-mode law
        # needs ANGLE_MAX, PILOT_Y_RATE and PILOT_SPEED_UP.
        if isinstance(self.backend, ArduPilotAcro):
            if self.is_intercept:
                self.law.max_yaw_rate = self.backend.pilot_yaw_rate
                self.law.angle_max = self.backend.angle_max
                self.law.max_climb_mps = self.backend.max_climb_mps
                self.law.max_pitch_rate = self.backend.acro_rp_rate
            else:
                self.law.max_yaw_rate = self.backend.acro_yaw_rate
                self.law.max_pitch_rate = self.backend.acro_rp_rate

        print("\n" + "=" * 74)
        print("FLIGHT CONTROL")
        print("=" * 74)
        print(self.backend.describe())
        print(f"control law   : {self.law.describe()}")
        untouched = [a for a in ("roll", "pitch", "yaw", "throttle")
                     if a not in self.axes]
        print(f"driving axes  : {', '.join(self.axes)}"
              + (f"  ({', '.join(untouched)} + arming stay with the pilot)"
                 if untouched else "  (ALL FOUR STICKS — only arming stays with "
                                   "the pilot)"))
        print(f"send rate     : {self.send_hz:.0f} Hz   "
              f"no-target: {self.no_target}")
        want = "/".join(sorted(self.steer_modes))
        print("mode gate     : "
              + (f"{want} only — the pilot's mode switch decides who steers; any "
                 "other mode returns every channel to the pilot"
                 if self.require_rate_mode else
                 f"OFF (REQUIRE_RATE_MODE=0) — steering is NOT gated on {want}"))
        if self.commit_chan:
            print(f"commit        : ch{self.commit_chan} scales the forward lean "
                  f"0..1"
                  + (f"; below {self.commit_abort_below:.2f} releases EVERY channel"
                     if self.commit_abort_below > 0 else ""))
        elif self.is_intercept:
            print(f"commit        : NO commit channel (COMMIT_CHAN=0) — lean is "
                  f"fixed at {self.law.commit_default * 100:.0f}% whenever a "
                  f"target is locked.\n"
                  "                The pilot's mode switch is then the ONLY "
                  "abort, and it is noticed at\n"
                  "                HEARTBEAT rate (~1 s).  Wiring any spare "
                  "channel to COMMIT_CHAN gives a\n"
                  "                live speed dial AND a ~100 ms abort.")
        if self.engage_max_s > 0:
            print(f"engage limit  : {self.engage_max_s:.1f} s per continuous "
                  f"engagement, then release until the target is re-acquired")

        blockers = list(self.backend.blockers)
        if self.is_intercept:
            # The intercept law needs ATTITUDE for three separate things — roll to
            # de-rotate the image, pitch to resolve elevation into the world frame,
            # and yawspeed for the PN term.  Without it yaw and the forward lean
            # still work but roll and throttle command zero, i.e. it aims and
            # flies straight and does not manoeuvre.  Say so here rather than let
            # it be diagnosed in the air.
            deadline = perf_counter() + 3.0
            while self.backend.attitude() is None and perf_counter() < deadline:
                sleep(0.1)
                if perf_counter() > deadline:
                    break
            if self.backend.attitude() is None:
                blockers.append(
                    "INTERCEPT needs ATTITUDE from the FC and none has arrived. "
                    "Roll (proportional navigation) and throttle (vertical "
                    "closure) will command ZERO until it does — the airframe will "
                    "aim and fly forward but will not turn its velocity vector or "
                    "change altitude.  If this persists the stream request was "
                    "refused.")
            if "throttle" in self.axes:
                blockers.append(
                    "THROTTLE IS DRIVEN BY THE AI.  In ALT_HOLD a centred "
                    "throttle holds altitude, so a release or a freeze is safe — "
                    "but the pilot no longer has direct thrust while engaged, and "
                    "throttle_frac() is meaningless from here on (it would read "
                    "our own override back).")
            if self.law.cam_tilt <= 0.0:
                blockers.append(
                    "CAM_TILT_DEG=0.  At nose-down lean theta a target at the "
                    "airframe's own altitude sits at +theta in frame, and the lens "
                    f"is only +/-{math.degrees(self.law.v_fov) / 2:.0f} deg "
                    "vertically — so at speed the target climbs out of the TOP of "
                    "the frame.  Tilt the camera UP by about the cruise lean and "
                    "set CAM_TILT_DEG to match.")
        if self.law.pitch_mode == "approach":
            # The approach loop is closed on the FC's measured attitude, so if
            # ATTITUDE never arrives the pitch command is silently 0 and the
            # airframe simply will not go forward.  That is safe but baffling in
            # the field, so say it here rather than let it be diagnosed in the air.
            #
            # Wait for the first sample before judging: preflight runs within a
            # few hundred ms of the stream request, so testing immediately reports
            # a failure on every healthy start.
            deadline = perf_counter() + 3.0
            while self.backend.pitch_rad() is None and perf_counter() < deadline:
                sleep(0.1)
            if self.backend.pitch_rad() is None:
                blockers.append(
                    "PITCH_MODE=approach needs ATTITUDE from the FC and none has "
                    "arrived yet.  Pitch will command 0 (no forward lean) until it "
                    "does.  If this persists the stream request was refused.")
        if self.require_armed and not self.backend.armed:
            blockers.append(f"The FC is DISARMED — overrides are accepted but the "
                            f"motors are off.  Arm from the transmitter in {want}.")
        if self.require_rate_mode and not self.backend.in_rate_mode():
            blockers.append(
                f"The FC is in {self.backend.mode_name}, not {want}.  The pilot has "
                f"FULL control and the tracker will not touch the sticks.  It hands "
                f"over by itself the moment the pilot switches to {want}, and takes "
                f"them back the moment the pilot switches away.")
        elif not self.is_intercept and self.backend.mode_name in ANGLE_MODES:
            blockers.append(
                f"The FC is in {self.backend.mode_name}, where a stick commands an "
                "ANGLE, not a rate.  The loop still centres the target but the "
                "gains are wrong for it — switch to ACRO, or use the intercept law "
                "(CONTROL_LAW=intercept), which is built for an angle mode.")
        if blockers:
            print("-" * 74)
            print("BLOCKERS — the drone will not move until these are fixed:")
            for i, b in enumerate(blockers, 1):
                print(f"  {i}. {b}")
        print("-" * 74)
        if self.is_intercept:
            print("REMINDER: ALT_HOLD holds altitude on the BAROMETER — there is no")
            print("GPS on this airframe, so it will drift with the wind and it has no")
            print("idea where it is.  Roll and throttle command ZERO without a fresh")
            print("ATTITUDE.  The terminal phase freezes the sticks and flies the")
            print("trajectory it already had, so whatever is in front of it gets hit.")
            print("PROPS OFF on the bench.")
        else:
            print("REMINDER: in ACRO the attitude controller's authority scales with")
            print("throttle.  At idle throttle the mixer has no headroom and yaw/pitch")
            print("commands produce almost no torque — the pilot must hold mid throttle")
            print("for the airframe to actually rotate.  PROPS OFF on the bench.")
        print("=" * 74 + "\n", flush=True)

        self._open_log()
        self._stop.clear()
        self._tx = threading.Thread(target=self._tx_loop, daemon=True)
        self._tx.start()

    # ── per-detection-frame input ────────────────────────────────────────────
    def compute(self, err_x, err_y, state, fw, fh, bbox=None, capture_t=None):
        """Pixel offset -> stick command.  `state` is the tracker's LOCK /
        PREDICT / SEARCH; only a real fix should steer, so SEARCH is neutral.

        Everything the law needs from the FC — attitude, body rates, the pilot's
        throttle, the commit input — is pulled from the backend here rather than
        plumbed through the tracker, so the tracker's call site does not need to
        know which law is running or what it needs.

        bbox       (x, y, w, h) in DISPLAY pixels, for the range estimate and the
                   terminal trigger.  Optional: without it the intercept law
                   simply never triggers terminal commit.
        capture_t  perf_counter() at frame capture, for the latency estimate.
                   Optional: the law falls back to the detection interval, which
                   is a lower bound on the true pipeline delay.
        """
        if state == "SEARCH":
            return self.law.neutral()

        if self.is_intercept:
            att = self.backend.attitude() if self.backend is not None else None
            commit = self.backend.commit() if self.backend is not None else None
            bw = bh = None
            if bbox is not None and len(bbox) >= 4:
                bw, bh = bbox[2], bbox[3]
            age = None if capture_t is None else max(0.0, perf_counter() - capture_t)
            return self.law.compute(err_x, err_y, fw, fh, bbox_w=bw, bbox_h=bh,
                                    att=att, commit=commit, capture_age=age)

        pitch_rad = throttle = None
        if self.backend is not None:
            pitch_rad = self.backend.pitch_rad()
            throttle = self.backend.throttle_frac()
        return self.law.compute(err_x, err_y, fw, fh,
                                measured_pitch_rad=pitch_rad,
                                throttle_frac=(throttle or 0.0))

    def neutral_command(self):
        return self.law.neutral()

    # Kept for tracker call-compatibility with the old commander.
    def search_command(self):
        return self.law.neutral()

    def update(self, cmd, target_off, have_face):
        """Publish the latest command for the tx thread.  Returns True when the
        sticks are actually being driven right now."""
        now = perf_counter()
        with self._lock:
            self._cmd = dict(cmd)
            self._target_off = target_off
            if have_face:
                self._last_target_t = now
        return self._steering

    # ── fixed-rate transmit ──────────────────────────────────────────────────
    def _tx_loop(self):
        period = 1.0 / self.send_hz
        next_t = perf_counter()
        while not self._stop.is_set():
            now = perf_counter()
            with self._lock:
                cmd = dict(self._cmd)
                off = self._target_off
                fresh = (now - self._last_target_t) <= self.target_timeout

            armed = self.backend.armed if self.backend else False
            have_target = off is not None and fresh
            # The pilot's mode switch is the master arbiter: the AI may steer
            # only while the FC is in a rate mode (ACRO).  Flipping back to
            # STABILIZE returns every channel to the pilot on the very next
            # transmit cycle, and flipping to ACRO again re-engages — the gate is
            # re-evaluated from scratch each cycle, so it toggles indefinitely.
            rate_ok = (not self.require_rate_mode
                       or self.backend is None
                       or self.backend.in_rate_mode())
            self._rate_mode_ok = rate_ok
            self._announce_mode_gate(rate_ok)
            steer = bool(self.linked and have_target and rate_ok
                         and (armed or not self.require_armed))

            # ── independent aborts ───────────────────────────────────────────
            # Both of these exist because the mode gate can only be as fast as
            # the 1 Hz HEARTBEAT it reads, which is up to a second of travel at
            # intercept speed.  Neither depends on the AI noticing anything.
            steer = self._commit_gate(steer)
            steer = self._engage_gate(steer, now)

            if steer:
                written = self.backend.steer(cmd)
            else:
                cmd = self.law.neutral() if self.no_target == "hold" else cmd
                written = self.backend.release() if self.linked else False
            self._steering = steer

            if (self.linked and have_target and not armed
                    and self.require_armed and not self._warned_disarmed):
                self._warned_disarmed = True
                print("[FC][WARN] target locked and the control loop is running, "
                      "but the FC is DISARMED — nothing will move until the pilot "
                      "arms in ACRO.", flush=True)

            if self.linked and (now - self._last_stt) >= (1.0 / self.statustext_hz):
                self._last_stt = now
                self.backend.statustext(self._status_text(cmd, off, steer))

            self._log_row(now, cmd, off, steer, written)

            next_t += period
            dt = next_t - perf_counter()
            if dt > 0:
                self._stop.wait(dt)
            else:
                next_t = perf_counter()          # fell behind, resync

    def _commit_gate(self, steer):
        """Pull the commit stick below the threshold and every channel goes back
        to the pilot, immediately.

        This is the abort worth having when the AI holds all four sticks: the
        commit channel is never overridden, so it arrives on the RC_CHANNELS
        stream at 10 Hz instead of waiting on the 1 Hz HEARTBEAT that the mode
        gate depends on.
        """
        if not steer or self.commit_abort_below <= 0.0 or self.backend is None:
            return steer
        c = self.backend.commit()
        if c is None:                      # no commit channel wired / no frame yet
            return steer
        if c >= self.commit_abort_below:
            if self._commit_aborted:
                self._commit_aborted = False
                print(f"[FC][COMMIT] commit stick back up ({c:.2f}) — AI may steer "
                      f"again", flush=True)
            return steer
        if not self._commit_aborted:
            self._commit_aborted = True
            print(f"[FC][COMMIT] ABORT — commit stick at {c:.2f}, below "
                  f"{self.commit_abort_below:.2f}.  Every channel released to the "
                  f"pilot.", flush=True)
        return False

    def _engage_gate(self, steer, now):
        """Cap one continuous engagement.

        A bound on the worst case rather than a feature: if detection wedges or
        the pilot's handback is slow to be noticed, this releases anyway.

        `steer` here is the UPSTREAM decision (target + mode + armed + commit),
        which is what the reset has to watch.  Resetting on the gate's own output
        would clear the latch on the very next cycle and re-engage for one cycle
        in every two, for ever.
        """
        if self.engage_max_s <= 0.0:
            return steer
        if not steer:
            # Upstream released it — target lost, mode switched, commit dropped.
            # That is a genuine disengagement, so the allowance starts over.
            self._engaged_since = None
            self._engage_timed_out = False
            return steer
        if self._engage_timed_out:
            return False               # latched until upstream releases
        if self._engaged_since is None:
            self._engaged_since = now
        if (now - self._engaged_since) <= self.engage_max_s:
            return True
        self._engage_timed_out = True
        print(f"[FC][LIMIT] engagement hit ENGAGE_MAX_S "
              f"({self.engage_max_s:.1f}s) — releasing every channel.  Break the "
              f"target lock to re-arm the AI.", flush=True)
        return False

    def _announce_mode_gate(self, rate_ok):
        """Print one line each time the pilot's mode switch changes who is in
        charge.  Deliberately not a one-shot warning: the pilot is expected to
        hand control back and forth repeatedly, and every handover should be
        visible in the console and in the log.
        """
        if rate_ok == self._mode_gate:
            return
        first = self._mode_gate is None
        self._mode_gate = rate_ok
        if not self.linked:
            return
        mode = self.mode_name
        if rate_ok and not self.require_rate_mode:
            print(f"[FC][MODE] {mode} — mode gate is OFF "
                  f"(REQUIRE_RATE_MODE=0), so a target lock drives yaw/pitch "
                  f"regardless of the pilot's mode switch.", flush=True)
        elif rate_ok:
            print(f"[FC][MODE] {mode} — rate mode: AI STEERING ENGAGED "
                  f"(a target lock will now drive yaw/pitch)", flush=True)
        elif first:
            print(f"[FC][MODE] {mode} — not a rate mode: the pilot has full "
                  f"control.  Switch to ACRO to hand yaw/pitch to the tracker.",
                  flush=True)
        else:
            print(f"[FC][MODE] {mode} — pilot left ACRO: FULL CONTROL RETURNED "
                  f"TO THE PILOT, all channels released.", flush=True)

    def _status_text(self, cmd, off, steer):
        if off is None:
            return f"WASP idle {self.mode_name[:10]}"
        tag = ("TERM" if steer and cmd.get("terminal")
               else "DRV" if steer
               else "ABORT" if self._commit_aborted or self._engage_timed_out
               else "PILOT" if not self._rate_mode_ok
               else "ARM?" if not self.armed else "off")
        if self.is_intercept:
            return (f"WASP {tag} b{cmd.get('bank_deg', 0.0):+.0f} "
                    f"l{cmd.get('lean_deg', 0.0):+.0f} "
                    f"c{cmd.get('climb_mps', 0.0):+.1f}")
        return (f"WASP {tag} e{off[0]:+d},{off[1]:+d} "
                f"y{cmd['yaw_stick']:+.2f} p{cmd['pitch_stick']:+.2f}")

    # ── shutdown ─────────────────────────────────────────────────────────────
    def close(self):
        self._stop.set()
        if self._tx is not None:
            self._tx.join(timeout=2)
        if self.backend is not None:
            try:
                self.backend.release()
            except Exception:
                pass
            self.backend.close()
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    # ── logging ──────────────────────────────────────────────────────────────
    def _open_log(self):
        if not self.log_dir:
            return
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = str(Path(self.log_dir) / f"control_{stamp}.csv")
        self._log_fh = open(self.log_path, "w", buffering=1)
        cols = ("t_s,mode,armed,steering,written,confirmed,"
                "ex,ey,az_deg,el_deg,yaw_stick,pitch_stick,"
                "yaw_rate_dps,pitch_rate_dps")
        if self.is_intercept:
            # The extra columns are the whole diagnostic story for an intercept:
            # los_az_dps is what PN is actually reacting to, latency_s is the
            # constraint that decides whether any of it is stable, and terminal
            # marks where the loop stopped looking.
            cols += (",roll_stick,throttle_stick,el_cam_deg,bank_deg,lean_deg,"
                     "climb_mps,los_az_dps,los_el_dps,range_m,closing_mps,"
                     "latency_s,commit,have_att,terminal")
        self._log_fh.write(cols + "\n")
        print(f"[FC] control log -> {self.log_path}", flush=True)

    def _log_row(self, now, cmd, off, steer, written):
        if self._log_fh is None:
            return
        ex = "" if off is None else off[0]
        ey = "" if off is None else off[1]
        conf = self.override_confirmed
        conf = "" if conf is None else int(conf)
        try:
            row = (f"{now - self._t0:.3f},{self.mode_name},{int(self.armed)},"
                   f"{int(steer)},{int(bool(written))},{conf},{ex},{ey},"
                   f"{cmd.get('az_deg', 0.0):.2f},{cmd.get('el_deg', 0.0):.2f},"
                   f"{cmd['yaw_stick']:+.4f},{cmd['pitch_stick']:+.4f},"
                   f"{math.degrees(cmd.get('yaw_rate', 0.0)):+.2f},"
                   f"{math.degrees(cmd.get('pitch_rate', 0.0)):+.2f}")
            if self.is_intercept:
                rng = cmd.get("range_m")
                row += (f",{cmd.get('roll_stick', 0.0):+.4f}"
                        f",{cmd.get('throttle_stick', 0.0):+.4f}"
                        f",{cmd.get('el_cam_deg', 0.0):.2f}"
                        f",{cmd.get('bank_deg', 0.0):+.2f}"
                        f",{cmd.get('lean_deg', 0.0):+.2f}"
                        f",{cmd.get('climb_mps', 0.0):+.2f}"
                        f",{cmd.get('los_az_dps', 0.0):+.2f}"
                        f",{cmd.get('los_el_dps', 0.0):+.2f}"
                        f",{'' if rng is None else f'{rng:.2f}'}"
                        f",{cmd.get('closing_mps', 0.0):+.2f}"
                        f",{cmd.get('latency_s', 0.0):.4f}"
                        f",{cmd.get('commit', 0.0):.2f}"
                        f",{int(bool(cmd.get('have_att')))}"
                        f",{int(bool(cmd.get('terminal')))}")
            self._log_fh.write(row + "\n")
        except Exception:
            pass

    # ── console line (one per frame) ─────────────────────────────────────────
    def format_command(self, state, cmd, target_off, sent, live, is_yolo):
        """One honest line describing what is being commanded and where it went.

        `sent` is the commander's own "the sticks are being driven" flag, so a
        line reading TX->FC means bytes really left the Jetson AND the FC was
        armed — never merely that a number was computed.
        """
        if self.is_intercept:
            rng = cmd.get("range_m")
            sticks = (f"yaw {cmd['yaw_stick']:+.2f}"
                      f"({math.degrees(cmd['yaw_rate']):+5.0f}d/s) "
                      f"bank {cmd.get('bank_deg', 0.0):+5.1f} "
                      f"lean {cmd.get('lean_deg', 0.0):+5.1f} "
                      f"climb {cmd.get('climb_mps', 0.0):+4.1f}m/s | "
                      f"LOS {cmd.get('los_az_dps', 0.0):+5.1f},"
                      f"{cmd.get('los_el_dps', 0.0):+5.1f}d/s "
                      f"R {'--' if rng is None else f'{rng:.1f}m'} "
                      f"lat {cmd.get('latency_s', 0.0) * 1000:3.0f}ms")
            if cmd.get("terminal"):
                sticks += "  [TERMINAL]"
            elif not cmd.get("have_att"):
                sticks += "  [NO ATT: roll+throttle 0]"
        else:
            sticks = (f"yaw {cmd['yaw_stick']:+.2f} ({math.degrees(cmd['yaw_rate']):+6.1f} d/s)  "
                      f"pitch {cmd['pitch_stick']:+.2f} "
                      f"({math.degrees(cmd['pitch_rate']):+6.1f} d/s)")
        if not live:
            dest = "SIM (no FC)"
        elif sent:
            conf = self.override_confirmed
            dest = ("TX->FC (confirmed)" if conf is True
                    else "TX->FC (REFUSED!)" if conf is False
                    else "TX->FC")
        elif not is_yolo:
            dest = "held (skip frame)"
        elif self._commit_aborted:
            dest = "released (COMMIT ABORT)"
        elif self._engage_timed_out:
            dest = "released (ENGAGE_MAX_S)"
        elif not self._rate_mode_ok:
            dest = (f"PILOT has control ({self.mode_name}, not "
                    f"{'/'.join(sorted(self.steer_modes))})")
        elif not self.armed and self.require_armed:
            dest = "released (FC disarmed)"
        else:
            dest = f"released (state={state})"

        if target_off is None:
            body = "no target — sticks released to pilot"
        else:
            body = f"target {target_off[0]:+05d},{target_off[1]:+05d}px -> centre"
        return f"[CTRL] {state:<7} {body} | {sticks} | {dest}"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG FROM ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

def _env_f(name, default):
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_i(name, default):
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_b(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def commander_from_env(log_dir=None, verbose=True):
    """Build an AcroCommander from environment variables so the deployed
    service and a manual run behave identically.

        CONTROL_LAW     servo (default) | intercept.
                        servo     = ControlLaw. Visual servo: yaw+pitch body
                                    rates null the pixel error. Correct ON A
                                    GIMBAL, in ACRO.
                        intercept = InterceptLaw. Proportional navigation on four
                                    channels for FREE FLIGHT, in ALT_HOLD. See
                                    _intercept_from_env for its own env block.
        MAV_CONN        link to the FC.  Default udpin:127.0.0.1:14551 (router).
                        A serial path like /dev/ttyACM0 also works.
                        "none" runs offline / print-only.
        MAV_BAUD        serial baud (USB CDC-ACM ignores it).  Default 115200.
        FC_TRANSPORT    auto | mavlink | msp | none.  Default auto.
        CONTROL_AXES    comma list of axes to drive.  Default yaw,pitch.
        MAV_SOURCE_SYSTEM
                        override the system id we send overrides from.  Leave
                        unset: it is read from the FC's SYSID_MYGCS.
        CAM_HFOV_DEG / CAM_VFOV_DEG      true lens FOV.  Default 62.2 / 48.8.
        KP_YAW / KP_PITCH / KD_YAW / KD_PITCH
        STICK_LIMIT     max deflection, 0..1.  Default 0.55.
        DEADBAND_FRAC   centre deadband as a fraction of the half-frame.
        SLEW_PER_S      max stick change per second.
        PITCH_MODE      aim (default) | approach.
                        aim      = pitch nulls the target's vertical pixel error.
                                   Correct ON A GIMBAL, where pitch only points
                                   the camera and the airframe cannot translate.
                        approach = pitch holds a nose-DOWN lean so the airframe
                                   flies FORWARD, and ignores vertical pixel error.
                                   Correct IN FREE FLIGHT.  Requires ATTITUDE from
                                   the FC; without it the pitch command is 0.
        APPROACH_DEG    max nose-down lean in approach mode, at full throttle.
                        Default 12.  Hard-clamped to 30.
        APPROACH_KP     rad/s of pitch rate per rad of lean error.  Default 2.5.
        APPROACH_THR_DEADBAND
                        throttle fraction above mid before any lean is commanded.
                        Default 0.05.
        CONTROL_SHAPE   linear (default) | adaptive.  "adaptive" replaces the
                        proportional stick map with a saturating power curve:
                        full deflection beyond SHAPE_SAT off-centre so the turn
                        starts hard, and a gamma-shaped collapse inside it so the
                        approach to centre is soft.  KP_* are unused in this mode;
                        KD_* still damp.
        SHAPE_GAMMA     adaptive curve exponent.  Default 1.6.  Higher = softer
                        near the centre, later commitment.
        SHAPE_SAT       |normalized offset| at which adaptive commands full stick.
                        Default 0.30 (30% of the half-frame).  Lower = snappier.
        YAW_REVERSED / PITCH_REVERSED    set to 1 if an axis drives the wrong way.
        SEND_HZ         override stream rate.  Default 20.
        REQUIRE_ARMED   0 to drive the sticks even while disarmed (bench only).
        REQUIRE_RATE_MODE
                        1 (default) steers ONLY while the FC is in ACRO, so the
                        pilot's mode switch is the handover control: STABILIZE (or
                        any non-rate mode) = pilot has everything, ACRO = the
                        tracker drives yaw/pitch.  Toggles both ways, any number of
                        times.  0 disables the gate (the old behaviour).
    """
    if os.environ.get("CONTROL_LAW", "servo").strip().lower() == "intercept":
        return _intercept_from_env(log_dir=log_dir, verbose=verbose)

    law = ControlLaw(
        h_fov_deg=_env_f("CAM_HFOV_DEG", 62.2),
        v_fov_deg=_env_f("CAM_VFOV_DEG", 48.8),
        kp_yaw=_env_f("KP_YAW", 2.0),
        kp_pitch=_env_f("KP_PITCH", 2.0),
        kd_yaw=_env_f("KD_YAW", 0.0),
        kd_pitch=_env_f("KD_PITCH", 0.0),
        deadband_frac=_env_f("DEADBAND_FRAC", 0.03),
        stick_limit=_env_f("STICK_LIMIT", 0.55),
        slew_per_s=_env_f("SLEW_PER_S", 3.0),
        invert_yaw=_env_b("YAW_REVERSED", False),
        invert_pitch=_env_b("PITCH_REVERSED", False),
        shape=os.environ.get("CONTROL_SHAPE", "linear"),
        shape_gamma=_env_f("SHAPE_GAMMA", 1.6),
        shape_sat=_env_f("SHAPE_SAT", 0.30),
        pitch_mode=os.environ.get("PITCH_MODE", "aim"),
        approach_deg=_env_f("APPROACH_DEG", 12.0),
        approach_kp=_env_f("APPROACH_KP", 2.5),
        approach_thr_deadband=_env_f("APPROACH_THR_DEADBAND", 0.05),
        stick_limit_yaw=(_env_f("STICK_LIMIT_YAW", None)),
        stick_limit_pitch=(_env_f("STICK_LIMIT_PITCH", None)),
    )
    conn = os.environ.get("MAV_CONN", "udpin:127.0.0.1:14551")
    if conn.strip().lower() == "none":
        conn = None
    sysid = os.environ.get("MAV_SOURCE_SYSTEM")
    return AcroCommander(
        conn=conn,
        baud=_env_i("MAV_BAUD", 115200),
        transport=os.environ.get("FC_TRANSPORT", "auto"),
        axes=tuple(a.strip() for a in
                   os.environ.get("CONTROL_AXES", "yaw,pitch").split(",") if a.strip()),
        law=law,
        send_hz=_env_f("SEND_HZ", 20.0),
        no_target=os.environ.get("NO_TARGET_BEHAVIOR", "release"),
        force_sysid=int(sysid) if sysid else None,
        log_dir=log_dir,
        require_armed=_env_b("REQUIRE_ARMED", True),
        require_rate_mode=_env_b("REQUIRE_RATE_MODE", True),
        verbose=verbose,
    )


def _intercept_from_env(log_dir=None, verbose=True):
    """Build the FREE-FLIGHT intercept commander (CONTROL_LAW=intercept).

    Proportional navigation on four channels in ALT_HOLD. See `InterceptLaw` for
    why each channel does what it does; this is only the env surface.

        CAM_TILT_DEG    how far the camera is physically tilted UP, degrees.
                        NOT optional in practice. At nose-down lean theta a target
                        at the airframe's own altitude sits at +theta in frame, and
                        the lens is only +/-24 deg vertically, so a level-mounted
                        camera loses the target out of the TOP as soon as it flies
                        fast. Tilt the mount up by roughly the cruise lean and put
                        that number here. Default 0, which raises a blocker.

        KP_YAW          seeker gain: rad/s of yaw per rad of azimuth error.
                        Default 4.5. This only points the camera; it does not move
                        the airframe.
        ROLL_PN_GAIN    THE turn gain, m/s. a_lat = gain * LOS_rate, so it is
                        dimensionally an own-speed: gain = N * airspeed, where N is
                        proportional navigation's constant. N=1 is exactly a
                        COORDINATED turn (the velocity vector rotates as fast as
                        the nose); classical PN uses N=3..5.
                        Default 45 = N 3 x the ~15 m/s this airframe settles at
                        with MAX_LEAN_DEG=20. **RAISE IT IF YOU RAISE THE LEAN**,
                        or the turn gain no longer matches the speed it flies.
                        Measured in tests/test_intercept.py section 9: 25 loses a
                        hard zigzag outright (no hit, target in frame only 37% of
                        the time), 45 intercepts it with the target in frame 100%
                        of the time, and 80 is worse again — it is not monotonic.
                        This is THE knob for "it does not follow me when I cut
                        left and right".
        MAX_BANK_DEG    ceiling on commanded bank. Default 30. The FC's ANGLE_MAX
                        clamps it again, so raise both to go harder.
        MAX_LEAN_DEG    nose-down lean at full commit = closing speed.
                        Default 20. Hard-clamped to 35.
        CLIMB_GAIN      m/s of climb per rad of elevation error. Default 8.
        CLIMB_LEAD      m/s of climb per rad/s of vertical LOS rate. Default 3.
                        Together these are the fix for "it does not follow me when
                        I am higher or lower than it".

        COMMIT_CHAN     RC channel carrying the pilot's commit/aggression input,
                        0..1 scaling the forward lean. 0 = none, and then the lean
                        is fixed at COMMIT_DEFAULT whenever a target is locked.
                        WIRE THIS IF YOU CAN: the channel is never overridden, so
                        it is both a live speed dial and a ~100 ms abort, where the
                        mode-switch gate can only be as fast as the 1 Hz HEARTBEAT.
                        It cannot be a driven axis (it would read our own override).
        COMMIT_DEFAULT  commit to assume with no commit channel. Default 1.0.
        COMMIT_ABORT_BELOW
                        commit below this releases EVERY channel, not just the
                        lean. Default 0.05 when a commit channel is set, else off.
        ENGAGE_MAX_S    hard cap on one continuous engagement, seconds. 0 = none.
                        Default 20. Bounds the worst case while the AI holds all
                        four sticks.

        TARGET_WIDTH_M  assumed target width for the range estimate. Default 0.20
                        (a human head). Range feeds the terminal trigger and the
                        log; it is deliberately NOT in the loop gain.
        TERMINAL_BBOX_FRAC / TERMINAL_RANGE_M / TERMINAL_HOLD_S
                        when the bbox height reaches this fraction of the frame, or
                        the range drops below this, guidance FREEZES and the
                        airframe flies the trajectory it already had, for at most
                        HOLD_S. Defaults 0.35 / 2.5 / 1.0. This exists because in
                        the last metre the head fills the frame, the detector drops
                        out and the pixel error goes meaningless.

        CONTROL_LATENCY_S
                        fixed end-to-end latency to extrapolate the target by.
                        0 (default) measures it live from the frame capture stamp.
        LATENCY_MAX_S   never extrapolate further than this. Default 0.35.
        LOS_TAU         LOS-rate filter time constant. Default 0.12.

        ROLL_REVERSED / THROTTLE_REVERSED / YAW_REVERSED / PITCH_REVERSED
                        set to 1 if an axis drives the wrong way. Check roll on the
                        bench: standing to the drone's LEFT must command LEFT bank.
        STICK_LIMIT and STICK_LIMIT_{YAW,ROLL,PITCH,THROTTLE}
                        per-channel ceilings. Default 0.90.
        SLEW_PER_S      max stick change per second. Default 11.
        DEADBAND_FRAC   angular deadband as a fraction of the half-FOV. Default
                        0.02. Applied to the ANGLE, not the pixel offset, because
                        with tilt compensation a zero world elevation is not a zero
                        pixel offset.
        SEND_HZ         override stream rate. Default 50 here rather than 20: at
                        20 Hz the stick is quantised to 50 ms, which is 10 deg of
                        attitude at 200 deg/s.
        CONTROL_AXES    default roll,pitch,yaw,throttle — all four.
        FC_STEER_MODES  flight modes in which the AI may steer. Default ALT_HOLD.
    """
    law = InterceptLaw(
        h_fov_deg=_env_f("CAM_HFOV_DEG", 62.2),
        v_fov_deg=_env_f("CAM_VFOV_DEG", 48.8),
        cam_tilt_deg=_env_f("CAM_TILT_DEG", 0.0),
        kp_yaw=_env_f("KP_YAW", 4.5),
        roll_pn_gain=_env_f("ROLL_PN_GAIN", 45.0),
        max_bank_deg=_env_f("MAX_BANK_DEG", 30.0),
        max_lean_deg=_env_f("MAX_LEAN_DEG", 20.0),
        commit_default=_env_f("COMMIT_DEFAULT", 1.0),
        climb_gain=_env_f("CLIMB_GAIN", 8.0),
        climb_lead=_env_f("CLIMB_LEAD", 3.0),
        deadband_frac=_env_f("DEADBAND_FRAC", 0.02),
        stick_limit=_env_f("STICK_LIMIT", 0.90),
        stick_limit_yaw=_env_f("STICK_LIMIT_YAW", None),
        stick_limit_roll=_env_f("STICK_LIMIT_ROLL", None),
        stick_limit_pitch=_env_f("STICK_LIMIT_PITCH", None),
        stick_limit_throttle=_env_f("STICK_LIMIT_THROTTLE", None),
        slew_per_s=_env_f("SLEW_PER_S", 11.0),
        rate_tau=_env_f("LOS_TAU", 0.12),
        latency_s=_env_f("CONTROL_LATENCY_S", 0.0),
        latency_max_s=_env_f("LATENCY_MAX_S", 0.35),
        target_width_m=_env_f("TARGET_WIDTH_M", 0.20),
        terminal_bbox_frac=_env_f("TERMINAL_BBOX_FRAC", 0.35),
        terminal_range_m=_env_f("TERMINAL_RANGE_M", 2.5),
        terminal_hold_s=_env_f("TERMINAL_HOLD_S", 1.0),
        invert_yaw=_env_b("YAW_REVERSED", False),
        invert_pitch=_env_b("PITCH_REVERSED", False),
        invert_roll=_env_b("ROLL_REVERSED", False),
        invert_throttle=_env_b("THROTTLE_REVERSED", False),
    )
    conn = os.environ.get("MAV_CONN", "udpin:127.0.0.1:14551")
    if conn.strip().lower() == "none":
        conn = None
    sysid = os.environ.get("MAV_SOURCE_SYSTEM")
    commit_chan = _env_i("COMMIT_CHAN", 0)
    # Only default the abort threshold on when there is a channel to read it from.
    # A non-zero threshold with no commit channel would be dead code that looks
    # like a safety feature.
    abort_below = _env_f("COMMIT_ABORT_BELOW", 0.05 if commit_chan else 0.0)
    return AcroCommander(
        conn=conn,
        baud=_env_i("MAV_BAUD", 115200),
        transport=os.environ.get("FC_TRANSPORT", "auto"),
        axes=tuple(a.strip() for a in
                   os.environ.get("CONTROL_AXES",
                                  "roll,pitch,yaw,throttle").split(",")
                   if a.strip()),
        law=law,
        send_hz=_env_f("SEND_HZ", 50.0),
        no_target=os.environ.get("NO_TARGET_BEHAVIOR", "release"),
        force_sysid=int(sysid) if sysid else None,
        log_dir=log_dir,
        require_armed=_env_b("REQUIRE_ARMED", True),
        require_rate_mode=_env_b("REQUIRE_RATE_MODE", True),
        steer_modes=frozenset(
            m.strip().upper()
            for m in os.environ.get("FC_STEER_MODES", "ALT_HOLD").split(",")
            if m.strip()),
        commit_chan=commit_chan,
        commit_abort_below=abort_below,
        engage_max_s=_env_f("ENGAGE_MAX_S", 20.0),
        verbose=verbose,
    )
