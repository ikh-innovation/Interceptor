#!/usr/bin/env python3
"""
fc_control.py  —  ACRO visual-servo control layer for the WASP head tracker
==============================================================================
Turns "the head is N pixels off-centre" into stick commands the flight
controller actually honours while the pilot flies in ACRO.

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
"""

import math
import os
import struct
import threading
from datetime import datetime
from pathlib import Path
from time import perf_counter

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
                 invert_pitch=False):
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

        self._tan_h = math.tan(self.h_fov / 2.0)
        self._tan_v = math.tan(self.v_fov / 2.0)

        self._ax_prev = 0.0
        self._ay_prev = 0.0
        self._t_prev = None
        self._yaw_stick = 0.0
        self._pitch_stick = 0.0

    # ── the law ───────────────────────────────────────────────────────────────
    def compute(self, err_x, err_y, fw, fh, dt=None):
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

        # Target right of centre -> yaw RIGHT  (positive body yaw rate).
        yaw_rate = self.kp_yaw * ax + self.kd_yaw * dax
        # Target below centre -> nose DOWN.  Positive body pitch rate is nose UP
        # (right-hand rule about body Y, which points right), hence the negation.
        pitch_rate = -(self.kp_pitch * ay + self.kd_pitch * day)

        yaw_stick = yaw_rate / self.max_yaw_rate
        pitch_stick = pitch_rate / self.max_pitch_rate

        if self.invert_yaw:
            yaw_stick = -yaw_stick
        if self.invert_pitch:
            pitch_stick = -pitch_stick

        lim = self.stick_limit
        yaw_stick = _clamp(yaw_stick, -lim, lim)
        pitch_stick = _clamp(pitch_stick, -lim, lim)

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
        return (f"FOV {math.degrees(self.h_fov):.1f}x{math.degrees(self.v_fov):.1f} deg  "
                f"KP yaw/pitch {self.kp_yaw:.2f}/{self.kp_pitch:.2f}  "
                f"KD {self.kd_yaw:.2f}/{self.kd_pitch:.2f}  "
                f"deadband {self.deadband_frac * 100:.1f}%  "
                f"stick<={self.stick_limit * 100:.0f}%  "
                f"slew {self.slew_per_s:.1f}/s  "
                f"full-stick rate yaw/pitch "
                f"{math.degrees(self.max_yaw_rate):.0f}/"
                f"{math.degrees(self.max_pitch_rate):.0f} deg/s")


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

    def in_rate_mode(self):
        """True when a stick deflection commands a body RATE on this FC right
        now — the only condition under which this control law is valid.

        A backend answers for itself rather than the commander string-matching
        `mode_name`, because only ArduPilot reports a live flight mode.  MSP and
        the offline backend have no equivalent, so they must not be gated off by
        a mode they never report.
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
                 source_component=190, force_sysid=None, verbose=True):
        super().__init__()
        self.conn = conn
        self.baud = int(baud)
        self.axes = tuple(axes)
        self.source_component = int(source_component)
        self.force_sysid = force_sysid
        self.verbose = verbose

        self.master = None
        self.sysid_mygcs = None
        self.rc = {}                  # axis -> {chan,min,max,trim,reversed}
        self.params = {}              # raw param snapshot, for the report
        self.acro_yaw_rate = None     # rad/s at full stick
        self.acro_rp_rate = None

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._rx = None
        self._hb = None
        self._sent_pwm = {}           # chan -> last PWM we asked for
        self._steer_since = None      # when we started steering (for the check)
        self._echo_warned = False
        self._t0 = perf_counter()

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
        switch — see AcroCommander's mode gate for why that is acceptable.
        """
        return self.mode_name in RATE_MODES

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
        if trainer is not None and int(trainer) != 0:
            self.blockers.append(
                f"ACRO_TRAINER={int(trainer)} auto-levels and limits lean angle, "
                "which fights a sustained pitch command.  Set ACRO_TRAINER=0.")

        y_rate = self._param("ACRO_Y_RATE")          # deg/s at full yaw stick
        rp_rate = self._param("ACRO_RP_RATE")        # deg/s at full pitch stick
        self.acro_yaw_rate = math.radians(y_rate) if y_rate else math.radians(202.5)
        self.acro_rp_rate = math.radians(rp_rate) if rp_rate else math.radians(360.0)

        for axis in self.axes:
            self.rc[axis] = self._read_axis(axis)

    def _read_axis(self, axis):
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
        self._say(f"[FC][RC] {axis:8s} ch{cfg['chan']}  "
                  f"min={cfg['min']} trim={cfg['trim']} max={cfg['max']} "
                  f"rev={int(cfg['reversed'])}  ({src}){warn}")
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

    def _request_rc_channels_stream(self):
        """Ask for RC_CHANNELS so we can verify the override actually landed.
        ArduPilot echoes the overridden value back in RC_CHANNELS."""
        try:
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS, 200000,  # 5 Hz
                0, 0, 0, 0, 0)
        except Exception:
            pass
        try:                       # legacy fallback for older firmware
            self.master.mav.request_data_stream_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, 5, 1)
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
        cfg = self.rc[axis]
        norm = _clamp(norm, -1.0, 1.0)
        if cfg["reversed"]:
            norm = -norm
        if norm >= 0.0:
            pwm = cfg["trim"] + norm * (cfg["max"] - cfg["trim"])
        else:
            pwm = cfg["trim"] + norm * (cfg["trim"] - cfg["min"])
        return int(round(_clamp(pwm, cfg["min"], cfg["max"])))

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
        lines = [f"firmware      : ArduPilot Copter",
                 f"mode          : {self.mode_name}"
                 + ("" if self.mode_name in RATE_MODES
                    else "   <-- not a rate mode; ACRO is what this expects"),
                 f"armed         : {'YES' if self.armed else 'NO'}",
                 f"our sysid     : {self.master.mav.srcSystem if self.master else '?'}"
                 f"   (FC SYSID_MYGCS = {self.sysid_mygcs})",
                 f"ACRO rates    : yaw {math.degrees(self.acro_yaw_rate):.0f} deg/s, "
                 f"roll/pitch {math.degrees(self.acro_rp_rate):.0f} deg/s at full stick"]
        for axis, cfg in self.rc.items():
            lines.append(f"{axis:14s}: ch{cfg['chan']} "
                         f"[{cfg['min']}..{cfg['trim']}..{cfg['max']}] "
                         f"rev={int(cfg['reversed'])}")
        for k in ("RC_OPTIONS", "ACRO_TRAINER", "ACRO_Y_EXPO", "ACRO_RP_EXPO"):
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
                  force_sysid=None, verbose=True):
    """Return a connected backend for `conn`.

    transport: "auto"       try MAVLink/ArduPilot, fall back to MSP/Betaflight
               "mavlink"    ArduPilot only
               "msp"        Betaflight/INAV only
               "none"       offline
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
                 verbose=True):
        self.conn = conn
        self.baud = baud
        self.transport = transport
        self.axes = tuple(axes)
        self.law = law or ControlLaw()
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
                                     verbose=self.verbose)

        # Adopt the FC's real full-stick ACRO rates so the stick mapping is
        # correct for THIS airframe's tuning rather than a guess.
        if isinstance(self.backend, ArduPilotAcro):
            self.law.max_yaw_rate = self.backend.acro_yaw_rate
            self.law.max_pitch_rate = self.backend.acro_rp_rate

        print("\n" + "=" * 74)
        print("FLIGHT CONTROL")
        print("=" * 74)
        print(self.backend.describe())
        print(f"control law   : {self.law.describe()}")
        print(f"driving axes  : {', '.join(self.axes)}  "
              f"(roll + throttle + arming stay with the pilot)")
        print(f"send rate     : {self.send_hz:.0f} Hz   "
              f"no-target: {self.no_target}")
        print("mode gate     : "
              + ("ACRO only — the pilot's mode switch decides who steers; any "
                 "other mode returns every channel to the pilot"
                 if self.require_rate_mode else
                 "OFF (REQUIRE_RATE_MODE=0) — steering is NOT gated on ACRO"))

        blockers = list(self.backend.blockers)
        if self.require_armed and not self.backend.armed:
            blockers.append("The FC is DISARMED — overrides are accepted but the "
                            "motors are off.  Arm from the transmitter in ACRO.")
        if self.require_rate_mode and not self.backend.in_rate_mode():
            blockers.append(
                f"The FC is in {self.backend.mode_name}, not ACRO.  The pilot has "
                "FULL control and the tracker will not touch the sticks.  It hands "
                "yaw/pitch over by itself the moment the pilot switches to ACRO, "
                "and takes them back the moment the pilot switches away.")
        elif self.backend.mode_name in ANGLE_MODES:
            blockers.append(
                f"The FC is in {self.backend.mode_name}, where a stick commands an "
                "ANGLE, not a rate.  The loop still centres the target but the "
                "gains are wrong for it — switch to ACRO.")
        if blockers:
            print("-" * 74)
            print("BLOCKERS — the drone will not move until these are fixed:")
            for i, b in enumerate(blockers, 1):
                print(f"  {i}. {b}")
        print("-" * 74)
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
    def compute(self, err_x, err_y, state, fw, fh):
        """Pixel offset -> stick command.  `state` is the tracker's LOCK /
        PREDICT / SEARCH; only a real fix should steer, so SEARCH is neutral."""
        if state == "SEARCH":
            return self.law.neutral()
        return self.law.compute(err_x, err_y, fw, fh)

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
        tag = ("DRV" if steer
               else "PILOT" if not self._rate_mode_ok
               else "ARM?" if not self.armed else "off")
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
        self._log_fh.write("t_s,mode,armed,steering,written,confirmed,"
                           "ex,ey,az_deg,el_deg,yaw_stick,pitch_stick,"
                           "yaw_rate_dps,pitch_rate_dps\n")
        print(f"[FC] control log -> {self.log_path}", flush=True)

    def _log_row(self, now, cmd, off, steer, written):
        if self._log_fh is None:
            return
        ex = "" if off is None else off[0]
        ey = "" if off is None else off[1]
        conf = self.override_confirmed
        conf = "" if conf is None else int(conf)
        try:
            self._log_fh.write(
                f"{now - self._t0:.3f},{self.mode_name},{int(self.armed)},"
                f"{int(steer)},{int(bool(written))},{conf},{ex},{ey},"
                f"{cmd.get('az_deg', 0.0):.2f},{cmd.get('el_deg', 0.0):.2f},"
                f"{cmd['yaw_stick']:+.4f},{cmd['pitch_stick']:+.4f},"
                f"{math.degrees(cmd.get('yaw_rate', 0.0)):+.2f},"
                f"{math.degrees(cmd.get('pitch_rate', 0.0)):+.2f}\n")
        except Exception:
            pass

    # ── console line (one per frame) ─────────────────────────────────────────
    def format_command(self, state, cmd, target_off, sent, live, is_yolo):
        """One honest line describing what is being commanded and where it went.

        `sent` is the commander's own "the sticks are being driven" flag, so a
        line reading TX->FC means bytes really left the Jetson AND the FC was
        armed — never merely that a number was computed.
        """
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
        elif not self._rate_mode_ok:
            dest = f"PILOT has control ({self.mode_name}, not ACRO)"
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
