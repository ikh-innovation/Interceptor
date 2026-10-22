#!/usr/bin/env python3

import math
import os
import struct
import threading
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
from wasp import WaspGuidance, interpolate_rotation
from flight_logging import FlightLogs

try:
    from pymavlink import mavutil
except ImportError:
    mavutil = None



def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _deadband(n, db):
    if db <= 0.0:
        return n
    a = abs(n)
    if a <= db:
        return 0.0
    return math.copysign((a - db) / (1.0 - db), n)


def _set_crop(law, cx, cy):
    law._tan_h = law.ref_tan_h * cx
    law._tan_v = law.ref_tan_v * cy
    law.h_fov = law.ref_h_fov if cx == 1.0 else 2.0 * math.atan(law._tan_h)
    law.v_fov = law.ref_v_fov if cy == 1.0 else 2.0 * math.atan(law._tan_v)


COPTER_MODES = {
    0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED",
    5: "LOITER", 6: "RTL", 7: "CIRCLE", 9: "LAND", 11: "DRIFT", 13: "SPORT",
    14: "FLIP", 15: "AUTOTUNE", 16: "POSHOLD", 17: "BRAKE", 18: "THROW",
    19: "AVOID_ADSB", 20: "GUIDED_NOGPS", 21: "SMART_RTL", 22: "FLOWHOLD",
    23: "FOLLOW", 24: "ZIGZAG", 25: "SYSTEMID", 26: "AUTOROTATE", 27: "AUTO_RTL",
}
ACRO_MODE = 1

RATE_MODES = {"ACRO"}
ANGLE_MODES = {"STABILIZE", "SPORT", "ALT_HOLD", "LOITER", "POSHOLD"}


class WrongFirmware(RuntimeError):
    pass



class ControlLaw:

    def __init__(self,
                 h_fov_deg=62.2,
                 v_fov_deg=48.8,
                 kp_yaw=2.0,
                 kp_pitch=2.0,
                 kd_yaw=0.0,
                 kd_pitch=0.0,
                 deadband_frac=0.03,
                 stick_limit=0.55,
                 slew_per_s=3.0,
                 max_yaw_rate=None,
                 max_pitch_rate=None,
                 invert_yaw=False,
                 invert_pitch=False,
                 shape="linear",
                 shape_gamma=1.6,
                 shape_sat=0.30,
                 stick_limit_yaw=None,
                 stick_limit_pitch=None,
                 pitch_mode="aim",
                 approach_deg=12.0,
                 approach_kp=2.5,
                 approach_thr_deadband=0.05):
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
        self.ref_h_fov, self.ref_v_fov = self.h_fov, self.v_fov
        self.ref_tan_h, self.ref_tan_v = self._tan_h, self._tan_v

        self._ax_prev = 0.0
        self._ay_prev = 0.0
        self._t_prev = None
        self._yaw_stick = 0.0
        self._pitch_stick = 0.0

    def set_crop(self, cx=1.0, cy=1.0):
        _set_crop(self, cx, cy)

    def approach_pitch_stick(self, measured_pitch_rad, throttle_frac):
        if measured_pitch_rad is None:
            return 0.0
        db = self.approach_thr_deadband
        demand = (throttle_frac - db) / (1.0 - db) if throttle_frac > db else 0.0
        demand = _clamp(demand, 0.0, 1.0)
        if demand <= 0.0:
            return 0.0
        target_lean = -math.radians(self.approach_deg) * demand
        target_lean = min(0.0, target_lean)
        rate = self.approach_kp * (target_lean - measured_pitch_rad)
        return rate / self.max_pitch_rate

    def compute(self, err_x, err_y, fw, fh, dt=None,
                measured_pitch_rad=None, throttle_frac=0.0):
        now = perf_counter()
        if dt is None:
            dt = (now - self._t_prev) if self._t_prev is not None else 0.0
        self._t_prev = now

        nx = _deadband(_clamp(2.0 * err_x / max(1, fw), -1.0, 1.0)
                       * (self._tan_h / self.ref_tan_h), self.deadband_frac)
        ny = _deadband(_clamp(2.0 * err_y / max(1, fh), -1.0, 1.0)
                       * (self._tan_v / self.ref_tan_v), self.deadband_frac)

        ax = math.atan(nx * self.ref_tan_h)
        ay = math.atan(ny * self.ref_tan_v)

        if dt > 1e-6:
            dax = (ax - self._ax_prev) / dt
            day = (ay - self._ay_prev) / dt
        else:
            dax = day = 0.0
        self._ax_prev, self._ay_prev = ax, ay

        if self.shape == "adaptive":
            yaw_stick = self._shape(nx) * self.stick_limit_yaw
            pitch_stick = -self._shape(ny) * self.stick_limit_pitch
            yaw_stick += (self.kd_yaw * dax) / self.max_yaw_rate
            pitch_stick += -(self.kd_pitch * day) / self.max_pitch_rate
        else:
            yaw_rate = self.kp_yaw * ax + self.kd_yaw * dax
            pitch_rate = -(self.kp_pitch * ay + self.kd_pitch * day)

            yaw_stick = yaw_rate / self.max_yaw_rate
            pitch_stick = pitch_rate / self.max_pitch_rate

        if self.pitch_mode == "approach":
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

        return dict(
            yaw_stick=yaw_stick,
            pitch_stick=pitch_stick,
            yaw_rate=(-yaw_stick if self.invert_yaw else yaw_stick) * self.max_yaw_rate,
            pitch_rate=(-pitch_stick if self.invert_pitch else pitch_stick) * self.max_pitch_rate,
            az_deg=math.degrees(ax),
            el_deg=math.degrees(-ay),
        )

    _NOMINAL_DT = 0.05

    def _shape(self, n):
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



class _Rate:

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

    _NOMINAL_DT = 0.05

    def __init__(self,
                 h_fov_deg=62.2,
                 v_fov_deg=48.8,
                 cam_tilt_deg=0.0,
                 kp_yaw=4.5,
                 roll_pn_gain=45.0,
                 max_bank_deg=30.0,
                 bank_far_deg=None,
                 sched_lo_frac=0.02,
                 sched_hi_frac=0.35,
                 pn_near_mult=1.0,
                 sens=0.5,
                 sens_max_mult=2.0,
                 lock_ramp_s=0.0,
                 fill_tau=0.30,
                 max_lean_deg=20.0,
                 lean_near=1.0,
                 lean_far_frac=0.007,
                 lean_near_frac=0.03,
                 lean_el_guard=0.0,
                 climb_los=False,
                 edge_guard=0.0,
                 cruise_mps=15.0,
                 cruise_lean_deg=20.0,
                 climb_trim=6.0,
                 commit_default=1.0,
                 climb_gain=8.0,
                 climb_lead=3.0,
                 deadband_frac=0.02,
                 stick_limit=0.90,
                 stick_limit_yaw=None,
                 stick_limit_roll=None,
                 stick_limit_pitch=None,
                 stick_limit_throttle=None,
                 slew_per_s=11.0,
                 rate_tau=0.12,
                 latency_s=0.0,
                 latency_max_s=0.35,
                 target_width_m=0.30,
                 terminal_bbox_frac=0.35,
                 terminal_range_m=2.5,
                 terminal_hold_s=1.0,
                 invert_yaw=False,
                 invert_pitch=False,
                 invert_roll=False,
                 invert_throttle=False,
                 max_yaw_rate=None,
                 angle_max=None,
                 max_climb_mps=2.5,
                 acro_rp_rate=None,
                 acro_yaw_rate=None,
                 att_kp=6.0,
                 max_descent_mps=None,
                 climb_limit_mps=None,
                 max_pitch_rate=None):
        self.h_fov = math.radians(float(h_fov_deg))
        self.v_fov = math.radians(float(v_fov_deg))
        self.cam_tilt = math.radians(float(cam_tilt_deg))
        self.kp_yaw = float(kp_yaw)
        self.roll_pn_gain = max(0.0, float(roll_pn_gain))
        self.max_bank = math.radians(_clamp(float(max_bank_deg), 0.0, 45.0))
        self.bank_far = self.max_bank if bank_far_deg is None else min(
            math.radians(_clamp(float(bank_far_deg), 0.0, 45.0)), self.max_bank)
        self.sched_lo_frac = max(0.0, float(sched_lo_frac))
        self.sched_hi_frac = max(self.sched_lo_frac + 1e-3, float(sched_hi_frac))
        self.pn_near_mult = max(1.0, float(pn_near_mult))
        self.sens = _clamp(float(sens), 0.0, 1.0)
        self.sens_max_mult = max(1.0, float(sens_max_mult))
        self.lock_ramp_s = max(0.0, float(lock_ramp_s))
        self.fill_tau = max(1e-3, float(fill_tau))
        self.max_lean = math.radians(_clamp(float(max_lean_deg), 0.0, 35.0))
        self.lean_near = _clamp(float(lean_near), 0.0, 1.0)
        self.lean_far_frac = max(0.0, float(lean_far_frac))
        self.lean_near_frac = max(self.lean_far_frac + 1e-3, float(lean_near_frac))
        self.lean_el_guard = _clamp(float(lean_el_guard), 0.0, 1.0)
        self.climb_los = bool(climb_los)
        self.edge_guard = _clamp(float(edge_guard), 0.0, 1.0)
        self.climb_trim = max(0.0, float(climb_trim))
        _cl = math.radians(_clamp(float(cruise_lean_deg), 1.0, 45.0))
        self.drag_k = (9.80665 * math.tan(_cl)
                       / max(1.0, float(cruise_mps) ** 2))
        self._v_fwd = 0.0
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

        self.max_yaw_rate = float(max_yaw_rate or math.radians(202.5))
        self.angle_max = float(angle_max or math.radians(30.0))
        self.max_climb_mps = max(0.1, float(max_climb_mps))
        self.acro_rp_rate = float(acro_rp_rate) if acro_rp_rate else None
        self.acro_yaw_rate = float(acro_yaw_rate) if acro_yaw_rate else None
        self.att_kp = max(0.0, float(att_kp))
        self.max_descent_mps = (self.max_climb_mps if max_descent_mps is None
                                else max(0.1, float(max_descent_mps)))
        self.climb_limit_mps = (None if climb_limit_mps is None
                                else max(0.1, float(climb_limit_mps)))
        self.max_pitch_rate = float(max_pitch_rate or math.radians(360.0))

        self._tan_h = math.tan(self.h_fov / 2.0)
        self._tan_v = math.tan(self.v_fov / 2.0)
        self.ref_h_fov, self.ref_v_fov = self.h_fov, self.v_fov
        self.ref_tan_h, self.ref_tan_v = self._tan_h, self._tan_v

        self.pitch_mode = "intercept"

        self._az_rate = _Rate(self.rate_tau)
        self._el_rate = _Rate(self.rate_tau)
        self._az_body_rate = _Rate(self.rate_tau)
        self._range_rate = _Rate(self.rate_tau)
        self._lam_prev = None
        self._lat_est = None
        self._t_prev = None
        self._sticks = dict(yaw=0.0, roll=0.0, pitch=0.0, throttle=0.0)
        self._terminal_since = None
        self._terminal_cmd = None
        self._range_m = None
        self._fill = 0.0
        self._lock_t = None
        self._ramp = 0.25
        self._gain_eff = self.roll_pn_gain
        self._bank_cap = self.bank_far

    def _de_rotate(self, ex, ey, roll):
        if roll is None:
            return ex, ey
        c, s = math.cos(roll), math.sin(roll)
        return ex * c - ey * s, ex * s + ey * c

    def set_crop(self, cx=1.0, cy=1.0):
        _set_crop(self, cx, cy)

    def _angles(self, ex, ey, fw, fh):
        nx = _clamp(2.0 * ex / max(1, fw), -1.0, 1.0)
        ny = _clamp(2.0 * ey / max(1, fh), -1.0, 1.0)
        az = math.atan(nx * self._tan_h)
        el = -math.atan(ny * self._tan_v)
        return az, el

    def _unwrap(self, lam):
        if self._lam_prev is not None:
            while lam - self._lam_prev > math.pi:
                lam -= 2.0 * math.pi
            while lam - self._lam_prev < -math.pi:
                lam += 2.0 * math.pi
        self._lam_prev = lam
        return lam

    def range_m(self, bbox_w, fw):
        if not bbox_w or bbox_w <= 0 or not fw:
            return None
        f_px = (fw / 2.0) / self._tan_h
        return f_px * self.target_width_m / float(bbox_w)

    def set_sens(self, v):
        try:
            v = _clamp(float(v), 0.0, 1.0)
        except (TypeError, ValueError):
            return self.sens
        self.sens = v
        return v

    def sens_state(self):
        gain, cap, ramp = self._schedule_values(self._fill, self._lock_t,
                                                self.sens)
        return dict(sens=self.sens,
                    gain_floor=self.roll_pn_gain,
                    gain_eff=gain,
                    bank_cap_deg=math.degrees(cap),
                    bank_far_deg=math.degrees(self.bank_far),
                    max_bank_deg=math.degrees(self.max_bank),
                    fill_frac=self._fill,
                    lock_ramp=ramp,
                    locked=self._lock_t is not None)

    def set_speed(self, climb_limit_mps=None, climb_gain=None,
                  max_lean_deg=None, lean_near=None):
        if climb_limit_mps is not None:
            self.climb_limit_mps = max(0.1, float(climb_limit_mps))
        if climb_gain is not None:
            self.climb_gain = max(0.0, float(climb_gain))
        if max_lean_deg is not None:
            self.max_lean = math.radians(_clamp(float(max_lean_deg), 0.0, 35.0))
        if lean_near is not None:
            self.lean_near = _clamp(float(lean_near), 0.0, 1.0)
        return self.speed_state()

    def set_stability(self, slew_per_s=None, rate_tau=None,
                      deadband_frac=None):
        if slew_per_s is not None:
            self.slew_per_s = max(0.0, float(slew_per_s))
        if deadband_frac is not None:
            self.deadband_frac = _clamp(float(deadband_frac), 0.0, 0.5)
        if rate_tau is not None:
            self.rate_tau = max(1e-3, float(rate_tau))
            for _f in (self._az_rate, self._el_rate, self._az_body_rate,
                       self._range_rate):
                _f.tau = self.rate_tau
        return self.stability_state()

    def stability_state(self):
        return dict(slew_per_s=self.slew_per_s,
                    rate_tau=self.rate_tau,
                    deadband_frac=self.deadband_frac)

    def speed_state(self):
        return dict(
            climb_limit=(self.max_climb_mps if self.climb_limit_mps is None
                         else min(self.max_climb_mps, self.climb_limit_mps)),
            climb_ceiling=self.max_climb_mps,
            descent_limit=self.max_descent_mps,
            climb_gain=self.climb_gain,
            max_lean_deg=math.degrees(self.max_lean),
            lean_near=self.lean_near)

    def _frame_edge_gate(self, el_cam, az):
        if self.edge_guard <= 0.0:
            return 1.0
        worst = 0.0
        for ang, half in ((el_cam, self.v_fov / 2.0), (az, self.h_fov / 2.0)):
            if ang is None or half <= 1e-6:
                continue
            worst = max(worst, abs(_clamp(ang / half, -1.0, 1.0)))
        lo = self.edge_guard
        if worst <= lo:
            return 1.0
        return _clamp(1.0 - (worst - lo) / max(1e-6, 1.0 - lo), 0.0, 1.0)

    def _climb_command(self, el_world, el_eff, los_el, lean, dt):
        if dt and dt > 1e-6:
            a = 9.80665 * math.tan(min(abs(lean), math.radians(45.0)))
            self._v_fwd = max(0.0, self._v_fwd
                              + (a - self.drag_k * self._v_fwd ** 2) * dt)

        if el_world is None:
            return 0.0
        if not self.climb_los:
            return self.climb_gain * el_eff + self.climb_lead * los_el

        el_c = _clamp(el_world, math.radians(-50.0), math.radians(50.0))
        return (self._v_fwd * math.tan(el_c)
                + self.climb_trim * el_eff
                + self.climb_lead * los_el)

    def _angle_to_stick(self, target_rad, measured_rad, rate_mode):
        if not rate_mode:
            return target_rad / self.angle_max
        if measured_rad is None or not self.acro_rp_rate:
            return 0.0
        rate_cmd = _clamp(self.att_kp * (target_rad - measured_rad),
                          -self.acro_rp_rate, self.acro_rp_rate)
        return rate_cmd / self.acro_rp_rate

    def _lean_el_gate(self, el_cam):
        if self.lean_el_guard <= 0.0 or el_cam is None:
            return 1.0
        half_v = self.v_fov / 2.0
        if half_v <= 1e-6:
            return 1.0
        frac = _clamp(el_cam / half_v, -1.0, 1.0)
        lo = self.lean_el_guard
        if frac <= lo:
            return 1.0
        return _clamp(1.0 - (frac - lo) / max(1e-6, 1.0 - lo), 0.0, 1.0)

    def _schedule(self):
        self._gain_eff, self._bank_cap, self._ramp = self._schedule_values(
            self._fill, self._lock_t, self.sens)

    def _schedule_values(self, fill, lock_t, sens):
        lo, hi = self.sched_lo_frac, self.sched_hi_frac
        s = _clamp((fill - lo) / max(1e-6, hi - lo), 0.0, 1.0)

        sens_gain = 1.0 + max(0.0, sens - 0.5) * 2.0 * (self.sens_max_mult - 1.0)
        sens_bank = min(1.0, 0.5 + sens)

        if self.lock_ramp_s > 0.0:
            t = 0.0 if lock_t is None else lock_t
            ramp = 0.25 + 0.75 * _clamp(t / self.lock_ramp_s, 0.0, 1.0)
        else:
            ramp = 1.0

        gain = self.roll_pn_gain * (1.0 + s * (self.pn_near_mult - 1.0)) * sens_gain
        bank = self.bank_far + s * (self.max_bank - self.bank_far)
        return gain, _clamp(bank * sens_bank * ramp, 0.0, self.max_bank), ramp

    def compute(self, err_x, err_y, fw, fh, dt=None, bbox_w=None, bbox_h=None,
                att=None, commit=None, capture_age=None, rate_mode=False):
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

        ex, ey = self._de_rotate(err_x, err_y, roll)
        az, el_cam = self._angles(ex, ey, fw, fh)
        el_world = (el_cam + pitch + self.cam_tilt) if have_att else None

        rng = self.range_m(bbox_w, fw)
        if rng is not None:
            self._range_m = rng
            closing = -self._range_rate.update(rng, dt)
        else:
            closing = 0.0

        fill = (float(bbox_w) / float(fw) * (self._tan_h / self.ref_tan_h)
                if (bbox_w and fw) else 0.0)
        a = dt / (self.fill_tau + dt) if dt > 1e-6 else 1.0
        self._fill += a * (fill - self._fill)
        self._lock_t = dt if self._lock_t is None else self._lock_t + max(0.0, dt)
        self._schedule()

        los_az = 0.0
        if yaw is not None:
            los_az = self._az_rate.update(self._unwrap(yaw + az), dt)
        los_el = self._el_rate.update(el_world, dt) if el_world is not None else 0.0
        d_az_body = self._az_body_rate.update(az, dt)

        lat = self._latency(capture_age, dt)
        az_lead = az + d_az_body * lat
        el_lead = (el_world + los_el * lat) if el_world is not None else None

        half_h, half_v = self.ref_h_fov / 2.0, self.ref_v_fov / 2.0
        az_eff = _deadband(_clamp(az_lead / half_h, -1.0, 1.0),
                           self.deadband_frac) * half_h
        if el_lead is None:
            el_eff = 0.0
        else:
            el_eff = _deadband(_clamp(el_lead / half_v, -1.0, 1.0),
                               self.deadband_frac) * half_v

        _yaw_scale = ((self.acro_yaw_rate or self.max_yaw_rate) if rate_mode
                      else self.max_yaw_rate)
        yaw_stick = (self.kp_yaw * az_eff) / _yaw_scale

        a_lat = 0.0 if yaw is None else self._gain_eff * los_az
        bank = _clamp(math.atan2(a_lat, 9.80665),
                      -self._bank_cap, self._bank_cap)
        roll_stick = self._angle_to_stick(bank, roll, rate_mode)

        cmt = self.commit_default if commit is None else _clamp(commit, 0.0, 1.0)
        aim_gate = max(0.0, math.cos(az))
        el_gate = self._lean_el_gate(el_cam) * self._frame_edge_gate(el_cam, az)
        near = _clamp((self._fill - self.lean_far_frac)
                      / (self.lean_near_frac - self.lean_far_frac), 0.0, 1.0)
        lean_sched = 1.0 - near * (1.0 - self.lean_near)
        lean = (-self.max_lean * lean_sched * cmt * aim_gate
                * el_gate)
        pitch_stick = self._angle_to_stick(lean, pitch, rate_mode)

        climb = self._climb_command(el_lead, el_eff, los_el, lean, dt)
        _lim_up = (self.max_climb_mps if self.climb_limit_mps is None
                   else min(self.max_climb_mps, self.climb_limit_mps))
        climb = _clamp(climb, -self.max_descent_mps, _lim_up)
        _thr_scale = (self.max_climb_mps if climb >= 0.0
                      else self.max_descent_mps)
        thr_stick = climb / _thr_scale
        if rate_mode:
            thr_stick = 0.0
            climb = 0.0

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
            bank_deg=math.degrees(out_roll * self.angle_max if not rate_mode
                                  else bank),
            lean_deg=math.degrees(out_pitch * self.angle_max if not rate_mode
                                  else lean),
            climb_mps=out_thr * (self.max_climb_mps if out_thr >= 0.0
                                 else self.max_descent_mps),
            los_az_dps=math.degrees(los_az),
            los_el_dps=math.degrees(los_el),
            range_m=rng,
            closing_mps=closing,
            latency_s=lat,
            commit=cmt,
            have_att=bool(have_att),
            terminal=False,
            rate_mode=bool(rate_mode),
            sens=self.sens,
            lean_el_gate=el_gate,
            v_fwd_mps=self._v_fwd,
            pn_gain_eff=self._gain_eff,
            bank_cap_deg=math.degrees(self._bank_cap),
            fill_frac=self._fill,
        )

        return self._terminal(cmd, bbox_h, fh, rng, dt)

    def _latency(self, capture_age, dt):
        if self.latency_s > 0.0:
            return min(self.latency_s, self.latency_max_s)
        obs = capture_age if (capture_age is not None and capture_age > 0) else dt
        if obs and obs > 0:
            self._lat_est = obs if self._lat_est is None else \
                self._lat_est + 0.2 * (obs - self._lat_est)
        return min(self._lat_est or 0.0, self.latency_max_s)

    def _terminal(self, cmd, bbox_h, fh, rng, dt):
        big = (bbox_h is not None and fh and
               (bbox_h / float(fh)) * (self._tan_v / self.ref_tan_v)
               >= self.terminal_bbox_frac)
        close = (rng is not None and rng <= self.terminal_range_m)
        if self._terminal_since is None:
            if not (big or close):
                return cmd
            self._terminal_since = 0.0
            self._terminal_cmd = dict(cmd)
        else:
            self._terminal_since += max(0.0, dt)
            if self._terminal_since > self.terminal_hold_s:
                return self.neutral()
        held = dict(self._terminal_cmd)
        held["terminal"] = True
        held["range_m"] = rng
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
        self._az_rate.reset()
        self._el_rate.reset()
        self._az_body_rate.reset()
        self._range_rate.reset()
        self._lam_prev = None
        self._t_prev = None
        self._terminal_since = None
        self._terminal_cmd = None
        self._fill = 0.0
        self._lock_t = None
        self._ramp = 0.25
        self._gain_eff = self.roll_pn_gain
        self._bank_cap = self.bank_far
        self._v_fwd = 0.0
        for k in self._sticks:
            self._sticks[k] = 0.0
        _n_gain, _n_cap, _ = self._schedule_values(0.0, None, self.sens)
        return dict(yaw_stick=0.0, roll_stick=0.0, pitch_stick=0.0,
                    throttle_stick=0.0, yaw_rate=0.0, pitch_rate=0.0,
                    az_deg=0.0, el_deg=0.0, el_cam_deg=0.0, bank_deg=0.0,
                    lean_deg=0.0, climb_mps=0.0, los_az_dps=0.0, los_el_dps=0.0,
                    range_m=None, closing_mps=0.0, latency_s=0.0, commit=0.0,
                    have_att=False, terminal=False, rate_mode=False,
                    sens=self.sens, pn_gain_eff=_n_gain,
                    bank_cap_deg=math.degrees(_n_cap), fill_frac=0.0)

    def describe(self):
        return (
            f"INTERCEPT (proportional navigation)  "
            f"FOV {math.degrees(self.h_fov):.1f}x{math.degrees(self.v_fov):.1f} deg  "
            f"cam tilt +{math.degrees(self.cam_tilt):.0f} deg up\n"
            f"                yaw=SEEKER KP {self.kp_yaw:.2f}  "
            f"roll=PN gain {self.roll_pn_gain:.0f} m/s, bank<="
            f"{math.degrees(self.max_bank):.0f} deg\n"
            f"                schedule: bank ceiling "
            f"{math.degrees(self.bank_far):.0f}->"
            f"{math.degrees(self.max_bank):.0f} deg and gain x1.0->"
            f"x{self.pn_near_mult:.1f} as the bbox fills "
            f"{self.sched_lo_frac * 100:.0f}->{self.sched_hi_frac * 100:.0f}% "
            f"of the frame\n"
            f"                SENS {self.sens:.2f} (0.5 = x1; below trims BANK "
            f"only, gain floor {self.roll_pn_gain:.0f} m/s is never crossed)  "
            f"lock ramp {self.lock_ramp_s:.1f}s\n"
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
            f"lean {math.degrees(self.angle_max):.0f} deg\n"
            f"                ACRO: "
            + (f"rate law armed (ATT_KP {self.att_kp:.1f}, ACRO_RP_RATE "
               f"{math.degrees(self.acro_rp_rate):.0f} deg/s) — same bank/lean "
               f"target, reached by an outer rate loop instead of the FC's "
               f"onboard angle hold; throttle stays with the pilot"
               if self.acro_rp_rate else
               "not connected yet — scales arrive from the FC at connect()"))



class FcBackend:

    name = "none"

    def __init__(self):
        self.linked = False
        self.armed = False
        self.mode_name = "?"
        self.fw_string = "?"
        self.override_confirmed = None
        self.blockers = []

    def connect(self):
        raise NotImplementedError

    def steer(self, cmd):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError

    def close(self):
        pass

    def pitch_rad(self, max_age=0.5):
        return None

    def attitude(self, max_age=0.5):
        return None

    def throttle_frac(self):
        return 0.0

    def commit(self):
        return None

    def in_rate_mode(self):
        return True

    def status(self):
        return f"{self.name} {self.mode_name} {'ARMED' if self.armed else 'DISARMED'}"



class ArduPilotAcro(FcBackend):

    name = "ardupilot"

    DEFAULT_GCS_SYSID = 255

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
        self.steer_modes = frozenset(steer_modes or RATE_MODES)
        self.commit_chan = int(commit_chan or 0)

        self.master = None
        self.sysid_mygcs = None
        self.rc = {}
        self.params = {}
        self.acro_yaw_rate = None
        self.acro_rp_rate = None
        self.angle_max = None
        self.pilot_yaw_rate = None
        self.max_climb_mps = None
        self.thr_deadzone = 100

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._rx = None
        self._hb = None
        self._sent_pwm = {}
        self._sent_hist = {}
        self._steer_since = None
        self._echo_warned = False
        self._t0 = perf_counter()
        self._pitch_rad = None
        self._roll_rad = None
        self._yaw_rad = None
        self._attitude_t = None
        self._att_hist = []
        self._rollspeed = None
        self._pitchspeed = None
        self._yawspeed = None
        self._commit = None
        self._throttle_frac = 0.0
        self._throttle_chan = None
        self._vfr = None
        self._baro = None
        self._baro_ref = None
        self._vibe = None
        self._landed = None
        self._alt_ref = None
        self._was_armed = None
        self._pilot_pwm = {}
        self._override_t = {}
        self._idle_mode_ack = None
        self.fltmode_ch = 5
        self.flight_modes = None
        self._mode_switch = None

    def connect(self):
        if mavutil is None:
            raise RuntimeError("pymavlink not installed:  pip install pymavlink")




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




        self._read_config()
        self._read_mode_switch()
        self._adopt_gcs_sysid()
        self._request_rc_channels_stream(hz=20.0)
        self._request_attitude_stream()
        self._request_message(mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD, 10.0)
        self._request_message(mavutil.mavlink.MAVLINK_MSG_ID_SCALED_PRESSURE, 10.0)
        self._request_message(mavutil.mavlink.MAVLINK_MSG_ID_VIBRATION, 10.0)
        self._request_message(mavutil.mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 5.0)

        self._stop.clear()
        self._rx = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx.start()
        self._hb = threading.Thread(target=self._hb_loop, daemon=True)
        self._hb.start()

    def _say(self, msg):
        if self.verbose:
            pass

    def _wait_vehicle_heartbeat(self, timeout):
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
        self._heartbeat_t = perf_counter()
        self.mode_name = COPTER_MODES.get(hb.custom_mode, f"MODE{hb.custom_mode}")
        self.armed = bool(hb.base_mode
                          & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        if self.armed and self._was_armed is False and self._vfr is not None:
            self._alt_ref = self._vfr[1]
        if self.armed and self._was_armed is False and self._baro is not None:
            self._baro_ref = self._baro[1]
        self._was_armed = self.armed

    def in_rate_mode(self):
        return self.mode_name in self.steer_modes

    def _param(self, name, timeout=2.5):
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

    def _params(self, names, timeout=1.0):
        # One burst of requests, then collect: missing parameters cost one
        # timeout in total instead of one each.
        want = set(names)
        for name in names:
            try:
                self.master.mav.param_request_read_send(
                    self.master.target_system, self.master.target_component, name.encode(), -1)
            except Exception:
                return {}
        got = {}
        deadline = perf_counter() + timeout
        while want - set(got) and perf_counter() < deadline:
            msg = self.master.recv_match(type="PARAM_VALUE", blocking=True, timeout=.2)
            if msg is None:
                continue
            pid = msg.param_id
            if isinstance(pid, bytes):
                pid = pid.decode(errors="ignore")
            pid = pid.strip("\x00")
            if pid in want:
                got[pid] = float(msg.param_value)
                self.params[pid] = got[pid]
        return got

    def _read_mode_switch(self):
        # The pilot's mode switch is never overridden, so its channel shows a
        # flip at RC_CHANNELS rate, long before the 1 Hz heartbeat reports it.
        names = ["FLTMODE_CH"] + [f"FLTMODE{i}" for i in range(1, 7)]
        got = self._params(names)
        ch = got.get("FLTMODE_CH")
        self.fltmode_ch = int(ch) if ch and 1 <= ch <= 18 else 5
        modes = [got.get(f"FLTMODE{i}") for i in range(1, 7)]
        self.flight_modes = ([COPTER_MODES.get(int(m), f"MODE{int(m)}") for m in modes]
                             if None not in modes else None)

    MODE_SWITCH_BANDS = (1230, 1360, 1490, 1620, 1749)

    def pilot_mode_switch(self, max_age=0.5):
        """(position 0-5, mode it selects or None) of the pilot's mode switch."""
        with self._lock:
            v = self._mode_switch
        if v is None or not 0 <= perf_counter() - v[0] <= max_age:
            return None
        band = sum(v[1] > b for b in self.MODE_SWITCH_BANDS)
        return band, (self.flight_modes[band] if self.flight_modes else None)

    def _read_config(self):
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
        if (trainer is not None and int(trainer) != 0
                and "ACRO" in self.steer_modes):
            self.blockers.append(
                f"ACRO_TRAINER={int(trainer)} auto-levels and limits lean angle, "
                "which fights a sustained pitch command.  Set ACRO_TRAINER=0.")

        y_rate = self._param("ACRO_Y_RATE")
        rp_rate = self._param("ACRO_RP_RATE")
        self._param("ACRO_RP_EXPO")
        self._param("ACRO_Y_EXPO")
        self.acro_yaw_rate = math.radians(y_rate) if y_rate else math.radians(202.5)
        self.acro_rp_rate = math.radians(rp_rate) if rp_rate else math.radians(360.0)

        angle_max = self._param("ANGLE_MAX")
        self.angle_max = (math.radians(angle_max / 100.0) if angle_max
                          else math.radians(30.0))
        py_rate = self._param("PILOT_Y_RATE")
        if py_rate is None:
            py_rate = y_rate
        self.pilot_yaw_rate = (math.radians(py_rate) if py_rate
                               else math.radians(202.5))
        up = self._param("PILOT_SPEED_UP")
        self.max_climb_mps = (up / 100.0) if up else 2.5
        dz = self._param("THR_DZ")
        self.thr_deadzone = int(dz) if dz is not None else 100
        dn = self._param("PILOT_SPEED_DN")
        self.max_descent_mps = (dn / 100.0) if dn else self.max_climb_mps
        acc = self._param("PILOT_ACCEL_Z")
        self.accel_z = (acc / 100.0) if acc else 2.5
        jz = self._param("PSC_JERK_Z")
        self.jerk_z = jz if jz else 5.0
        for _p in ("RC_OVERRIDE_TIME", "ATC_INPUT_TC", "ATC_ACCEL_R_MAX",
                   "ATC_ACCEL_P_MAX", "LOG_BACKEND_TYPE", "LOG_BITMASK",
                   "INS_LOG_BAT_MASK", "FS_GCS_ENABLE"):
            self._param(_p)

        for axis in self.axes:
            self.rc[axis] = self._read_axis(axis)

        if "throttle" not in self.rc:
            try:
                self.rc["throttle"] = self._read_axis("throttle", announce=False)
            except Exception:
                pass

        if self.commit_chan:
            ch = self.commit_chan
            mn = self._param(f"RC{ch}_MIN")
            mx = self._param(f"RC{ch}_MAX")
            self.rc["commit"] = dict(chan=ch,
                                     min=int(mn) if mn else 1000,
                                     max=int(mx) if mx else 2000,
                                     trim=1500, reversed=False,
                                     from_fc=None not in (mn, mx))




            if ch in {cfg["chan"] for a, cfg in self.rc.items()
                      if a in self.axes}:
                self.blockers.append(
                    f"COMMIT_CHAN={ch} is also a driven axis.  The commit input "
                    "must be on a channel we never write, or it reads our own "
                    "override back.  Move it to a spare channel.")

    def _read_axis(self, axis, announce=True):
        fallback_chan = {"yaw": 4, "pitch": 2, "roll": 1, "throttle": 3}
        ch = self._param(f"RCMAP_{axis.upper()}")
        mapping_read = ch is not None
        ch = int(ch) if ch else fallback_chan.get(axis, 0)
        mn = self._param(f"RC{ch}_MIN")
        mx = self._param(f"RC{ch}_MAX")
        tr = self._param(f"RC{ch}_TRIM")
        rv = self._param(f"RC{ch}_REVERSED")
        dzp = self._param(f"RC{ch}_DZ")
        from_fc = None not in (mn, mx, tr)
        cfg = dict(chan=ch,
                   min=int(mn) if mn else 1000,
                   max=int(mx) if mx else 2000,
                   trim=int(tr) if tr else 1500,
                   reversed=bool(int(rv)) if rv is not None else False,
                   dz=int(dzp) if dzp else 0,
                   from_fc=from_fc)
        cfg["wasp_calibrated"] = mapping_read and None not in (mn, mx, tr, rv, dzp)
        src = "from FC" if from_fc else "DEFAULTS (param read failed!)"
        warn = "" if 1 <= cfg["chan"] <= 8 else "  [chan>8 cannot be overridden!]"
        if announce:
            pass


        if not announce:
            return cfg
        if not 1 <= cfg["chan"] <= 8:
            self.blockers.append(
                f"{axis} is on RC channel {cfg['chan']}; RC_CHANNELS_OVERRIDE only "
                "reaches channels 1-8.  Remap RCMAP_%s." % axis.upper())
        return cfg

    def _adopt_gcs_sysid(self):
        want = self.force_sysid if self.force_sysid is not None else self.sysid_mygcs
        if want is None:
            want = self.DEFAULT_GCS_SYSID



        want = int(want)
        self.master.mav.srcSystem = want
        try:
            self.master.source_system = want
        except Exception:
            pass
        note = "" if self.sysid_mygcs is None else f" (FC SYSID_MYGCS={self.sysid_mygcs})"



    def _request_rc_channels_stream(self, hz=10.0):
        try:
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS,
                int(1e6 / max(1.0, hz)), 0, 0, 0, 0, 0)
        except Exception:
            pass
        try:
            self.master.mav.request_data_stream_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, int(hz), 1)
        except Exception:
            pass

    def _request_message(self, msg_id, hz):
        try:
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                msg_id, int(1e6 / max(1.0, hz)), 0, 0, 0, 0, 0)
        except Exception:
            pass

    def _request_attitude_stream(self, hz=50.0):
        try:
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
                int(1e6 / max(1.0, hz)), 0, 0, 0, 0, 0)
        except Exception:
            pass
        try:
            self.master.mav.request_data_stream_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, int(hz), 1)
        except Exception:
            pass

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
            elif t == "VFR_HUD":
                with self._lock:
                    self._vfr = (perf_counter(), float(msg.alt),
                                 float(msg.climb), float(msg.heading))
            elif t == "SCALED_PRESSURE":
                with self._lock:
                    self._baro = (perf_counter(), float(msg.press_abs))
            elif t == "EXTENDED_SYS_STATE":
                with self._lock:
                    self._landed = (perf_counter(), int(msg.landed_state))
            elif t == "COMMAND_ACK" and msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
                with self._lock:
                    self._idle_mode_ack = (perf_counter(), int(msg.result))
            elif t == "VIBRATION":
                with self._lock:
                    self._vibe = (perf_counter(), float(msg.vibration_x),
                                  float(msg.vibration_y), float(msg.vibration_z),
                                  int(msg.clipping_0), int(msg.clipping_1),
                                  int(msg.clipping_2))
            elif t == "RC_CHANNELS":
                self._apply_pilot_sticks(msg)
                self._check_echo(msg)
                self._apply_pilot_throttle(msg)
                self._apply_commit(msg)
            elif t == "ATTITUDE":
                now = perf_counter()
                with self._lock:
                    self._pitch_rad = float(msg.pitch)
                    self._roll_rad = float(msg.roll)
                    self._rollspeed = float(msg.rollspeed)
                    self._pitchspeed = float(msg.pitchspeed)
                    self._yawspeed = float(msg.yawspeed)
                    y = float(msg.yaw)
                    if self._yaw_rad is not None:
                        while y - self._yaw_rad > math.pi:
                            y -= 2.0 * math.pi
                        while y - self._yaw_rad < -math.pi:
                            y += 2.0 * math.pi
                    self._yaw_rad = y
                    self._attitude_t = now
                    self._att_hist.append((now, self._roll_rad, self._pitch_rad,
                                           y, self._rollspeed, self._pitchspeed,
                                           self._yawspeed))
                    cut = now - self.ATT_HISTORY_S
                    while self._att_hist and self._att_hist[0][0] < cut:
                        self._att_hist.pop(0)

    def _hb_loop(self):
        while not self._stop.wait(1.0):
            try:
                self.master.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            except Exception:
                pass

    def _apply_pilot_throttle(self, rc_msg):
        cfg = self.rc.get("throttle")
        if cfg is None:
            return
        if "throttle" in self.axes:
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
        with self._lock:
            t, p = self._attitude_t, self._pitch_rad
        if t is None or p is None:
            return None
        return None if (perf_counter() - t) > max_age else p

    def throttle_frac(self):
        with self._lock:
            return self._throttle_frac

    ATT_HISTORY_S = 1.5

    def attitude(self, max_age=0.5, at_t=None):
        with self._lock:
            hist = list(self._att_hist)
            t_last = self._attitude_t
        if not hist or t_last is None:
            return None
        if (perf_counter() - t_last) > max_age:
            return None

        if at_t is None or len(hist) == 1 or at_t >= hist[-1][0]:
            t, r, p, y, rs, ps, ys = hist[-1]
        elif at_t <= hist[0][0]:
            t, r, p, y, rs, ps, ys = hist[0]
        else:
            lo = hi = hist[-1]
            for i in range(len(hist) - 1):
                if hist[i][0] <= at_t <= hist[i + 1][0]:
                    lo, hi = hist[i], hist[i + 1]
                    break
            span = hi[0] - lo[0]
            f = 0.0 if span <= 0 else (at_t - lo[0]) / span
            t = at_t
            r, p, y, rs, ps, ys = (a + f * (b - a)
                                   for a, b in zip(lo[1:], hi[1:]))
        return dict(roll=r, pitch=p, yaw=y, rollspeed=rs, pitchspeed=ps,
                    yawspeed=ys, age=perf_counter() - t)

    def wasp_attitude(self, at_t=None):


        now = perf_counter()
        with self._lock:
            hist = list(self._att_hist)
        if not hist or not 0 <= now - hist[-1][0] <= 0.15:
            return None
        if not all(math.isfinite(v) for v in hist[-1]):
            return None
        if at_t is None:
            at_t = hist[-1][0]
        if not math.isfinite(at_t) or not 0 <= now - at_t <= 0.25:
            return None
        if at_t < hist[0][0] or at_t > hist[-1][0] + 0.05:
            return None
        sample = hist[-1]
        mat = None
        for lo, hi in zip(hist, hist[1:]):
            if lo[0] <= at_t <= hi[0]:
                if hi[0] - lo[0] > 0.10 or not all(math.isfinite(v) for v in (*lo, *hi)):
                    return None
                f = (at_t-lo[0]) / max(1e-9, hi[0]-lo[0])
                angles = tuple(a + f*((b-a+math.pi) % (2*math.pi)-math.pi)
                               for a, b in zip(lo[1:4], hi[1:4]))
                sample = (at_t,) + angles + hi[4:]
                mat = interpolate_rotation(lo[1:4], hi[1:4], f)
                break
        if not all(math.isfinite(v) for v in sample):
            return None
        att = dict(zip(("roll", "pitch", "yaw", "rollspeed", "pitchspeed", "yawspeed"), sample[1:]))
        if mat is not None:
            att["rotation"] = mat
        return att

    def wasp_ready(self):
        if not self.armed or self.mode_name != "ACRO":
            return "WASP needs ARMED + ACRO"
        if perf_counter() - getattr(self, "_heartbeat_t", -1e9) > 1.5:
            return "no fresh FC heartbeat"
        why = self.wasp_setup_error()
        if why:
            return why
        if self.wasp_attitude() is None:
            return "no fresh attitude for WASP"
        return ""

    def wasp_setup_error(self):

        if self.params.get("ACRO_TRAINER") != 0:
            return "WASP needs ACRO_TRAINER=0 (no auto-level)"
        for k in ("ACRO_RP_RATE", "ACRO_Y_RATE"):
            v = self.params.get(k)
            if v is None or not math.isfinite(v) or v <= 0:
                return f"WASP needs a valid {k} read from the FC"
        for k in ("ACRO_RP_EXPO", "ACRO_Y_EXPO"):
            v = self.params.get(k)
            if v is None or not 0 <= v <= 1:
                return f"WASP needs a valid {k} read from the FC"
        timeout = self.params.get("RC_OVERRIDE_TIME", 0)
        if not 0 < timeout <= 1.0:
            return "WASP needs 0 < RC_OVERRIDE_TIME <= 1 second"
        if self.blockers:
            return self.blockers[0]
        channels = []
        for axis in ("roll", "pitch", "yaw", "throttle"):
            cfg = self.rc.get(axis, {})
            if axis not in self.axes or not cfg.get("wasp_calibrated"):
                return f"WASP needs calibrated {axis} control"
            dz_limit = (cfg["max"]-cfg["min"] if axis == "throttle" else
                        min(cfg["trim"]-cfg["min"], cfg["max"]-cfg["trim"]))
            if not (1 <= cfg["chan"] <= 8 and cfg["min"] <= cfg["trim"] <= cfg["max"]
                    and 0 <= cfg["dz"] < dz_limit):
                return f"invalid {axis} RC calibration"
            channels.append(cfg["chan"])
        if len(set(channels)) != 4:
            return "WASP RC axes must use distinct channels"
        return ""

    def wasp_pwm(self, rates, throttle):
        if not all(math.isfinite(v) for v in (*rates, throttle)) or not 0 <= throttle <= 1:
            raise ValueError("invalid WASP command")
        sticks = []
        for axis, rate in zip(("roll", "pitch", "yaw"), rates):
            prefix = "ACRO_Y" if axis == "yaw" else "ACRO_RP"
            target = min(1.0, abs(rate) / math.radians(self.params[prefix+"_RATE"]))
            expo = self.params[prefix+"_EXPO"]
            lo, hi = 0.0, 1.0
            for _ in range(24):
                mid = (lo+hi)/2
                if (1-expo)*mid + expo*mid**3 < target:
                    lo = mid
                else:
                    hi = mid
            sticks.append(math.copysign((lo+hi)/2, rate) if rate else 0.0)
        scale = max(1.0, math.hypot(*sticks[:2]))
        sticks[:2] = [v/scale for v in sticks[:2]]
        out = {a: self._exact_pwm(a, v) for a, v in zip(("roll", "pitch", "yaw"), sticks)}
        cfg = self.rc["throttle"]

        fraction = 1-throttle if cfg["reversed"] else throttle
        out["throttle"] = round(cfg["min"] + fraction*(cfg["max"]-cfg["min"]))
        return out

    def wasp_flight_ready(self):
        # Once WASP has launched only the pilot ends it: disarm, or a mode
        # switch out of ACRO.  A late heartbeat or attitude sample must not
        # drop the throttle back to a stick that is probably at the bottom.
        if not self.armed:
            return "disarmed"
        if self.mode_name != "ACRO":
            return f"pilot switched to {self.mode_name}"
        return ""

    def steer_wasp_pwm(self, pwm_by_axis, in_flight=False):
        if self.wasp_flight_ready() if in_flight else self.wasp_ready():
            return self.release()
        return self._steer_calibrated_pwm(pwm_by_axis)

    def idle_ready(self, allow_acro=False):
        if not self.linked or not self.armed:
            return "Idle needs FC link + armed"
        if self.mode_name not in (("ACRO", "STABILIZE", "ALT_HOLD") if allow_acro else ("ALT_HOLD",)):
            return "pilot changed mode"
        if perf_counter()-getattr(self, "_heartbeat_t", -1e9) > 1.5:
            return "no fresh FC heartbeat"
        if self.wasp_attitude() is None or self.height(max_age=.5) is None:
            return "Idle needs fresh attitude and height"
        if self.blockers:
            return self.blockers[0]
        if not 0 < self.params.get("RC_OVERRIDE_TIME", 0) <= 1:
            return "Idle needs RC_OVERRIDE_TIME <= 1 s"
        channels = []
        for axis in ("roll", "pitch", "yaw", "throttle"):
            cfg = self.rc.get(axis, {})
            if axis not in self.axes or not cfg.get("wasp_calibrated"):
                return f"Idle needs calibrated {axis}"
            values = [cfg.get(k, float("nan")) for k in ("min", "max", "trim", "dz", "chan")]
            if not all(math.isfinite(v) for v in values):
                return f"invalid {axis} RC calibration"
            dz_limit = (cfg["max"]-cfg["min"] if axis == "throttle" else
                        min(cfg["trim"]-cfg["min"], cfg["max"]-cfg["trim"]))
            if not (1 <= cfg["chan"] <= 8 and cfg["min"] <= cfg["trim"] <= cfg["max"]
                    and 0 <= cfg["dz"] < dz_limit):
                return f"invalid {axis} RC calibration"
            channels.append(cfg["chan"])
        if len(set(channels)) != 4:
            return "Idle needs distinct RC channels"
        return ""

    def recovery_flight_ready(self, switching=False):
        # WASP recovery holds the sticks until the pilot changes mode again or
        # disarms; a late sample must not drop it back to an unknown stick.
        if not self.linked or not self.armed:
            return "disarmed"
        if self.mode_name not in (("ACRO", "STABILIZE", "ALT_HOLD") if switching else ("ALT_HOLD",)):
            return "pilot changed mode"
        return ""

    def idle_request_alt_hold(self):
        if not self.linked or not self.armed:
            return False
        with self._lock:
            self._idle_mode_ack = None
        if self.mode_name == "ALT_HOLD":
            return True
        try:
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 2, 0, 0, 0, 0, 0)
            return True
        except Exception:
            return False

    def idle_pwm(self, yaw_dps, climb_mps):
        if not all(math.isfinite(v) for v in (yaw_dps, climb_mps)):
            raise ValueError("invalid Idle control")
        yaw = math.radians(yaw_dps)/(self.pilot_yaw_rate or math.radians(202.5))
        limit = self.max_climb_mps if climb_mps >= 0 else self.max_descent_mps
        throttle = climb_mps/max(.1, limit or 2.5)
        return {axis: self._exact_pwm(axis, value) for axis, value in
                (("roll", 0.0), ("pitch", 0.0), ("yaw", yaw), ("throttle", throttle))}

    def steer_idle_pwm(self, pwm_by_axis, allow_acro=False, recovery=False):
        if self.recovery_flight_ready(allow_acro) if recovery else self.idle_ready(allow_acro):
            return self.release()
        return self._steer_calibrated_pwm(pwm_by_axis)

    def _steer_calibrated_pwm(self, pwm_by_axis):
        chans = [0] * 8
        sent = {}
        for axis in ("roll", "pitch", "yaw", "throttle"):
            cfg = self.rc[axis]
            pwm = pwm_by_axis.get(axis)
            if pwm is None or not math.isfinite(pwm) or not cfg["min"] <= pwm <= cfg["max"]:
                return self.release()
            chans[cfg["chan"]-1] = int(pwm)
            sent[cfg["chan"]] = int(pwm)
        return self._commit_override(chans, sent)

    def commit(self):
        with self._lock:
            return self._commit

    def _apply_commit(self, rc_msg):
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

    ECHO_HISTORY_S = 0.7
    ECHO_TOL_US = 40

    def _check_echo(self, rc_msg):
        with self._lock:
            steer_since = self._steer_since
            sent = dict(self._sent_pwm)
            hist = {c: [pwm for _t, pwm in v]
                    for c, v in self._sent_hist.items()}
        if steer_since is None or not sent:
            return
        held = perf_counter() - steer_since
        if held < 0.4:
            return

        distinctive = {c: p for c, p in sent.items()
                       if any(abs(p - cfg["trim"]) > 40
                              for cfg in self.rc.values() if cfg["chan"] == c)}
        if not distinctive:
            return

        matched = True
        for chan in distinctive:
            got = getattr(rc_msg, f"chan{chan}_raw", None)
            if got is None:
                matched = False
                break
            candidates = hist.get(chan) or [distinctive[chan]]
            if not any(abs(int(got) - int(p)) <= self.ECHO_TOL_US
                       for p in candidates):
                matched = False
                break

        if matched:
            if self.override_confirmed is not True:
                pass

            self.override_confirmed = True
        elif held > 2.0:
            self.override_confirmed = False
            if not self._echo_warned:
                self._echo_warned = True
                detail = ", ".join(
                    f"ch{c}: sent {p}, FC reports "
                    f"{getattr(rc_msg, f'chan{c}_raw', '?')}"
                    for c, p in sorted(sent.items()))











    def height(self, max_age=1.0):



        with self._lock:
            v, ref = self._vfr, self._alt_ref
            p, p_ref = self._baro, self._baro_ref
        now = perf_counter()
        if v is None or (now - v[0]) > max_age:
            return None
        if p is not None and p_ref and p[1] > 0 and (now - p[0]) <= max_age:
            return (44330.0 * (1.0 - (p[1] / p_ref) ** 0.190284), v[2], v[3])
        return (v[1] - (ref or 0.0), v[2], v[3])

    AIRBORNE_MIN_M = 0.7

    def airborne(self, max_age=2.0):
        with self._lock:
            ls = self._landed
        if ls is not None and (perf_counter() - ls[0]) <= max_age and ls[1]:
            return ls[1] in (2, 3, 4), "FC landed state"
        h = self.height()
        if h is None:
            return False, "no height"
        return h[0] >= self.AIRBORNE_MIN_M, "height"

    def vibration(self, max_age=1.0):
        with self._lock:
            v = self._vibe
        if v is None or (perf_counter() - v[0]) > max_age:
            return None
        return dict(x=v[1], y=v[2], z=v[3], clip=v[4] + v[5] + v[6])

    PILOT_SETTLE_S = 0.35

    def _apply_pilot_sticks(self, rc_msg):
        now = perf_counter()
        with self._lock:
            raw = getattr(rc_msg, f"chan{self.fltmode_ch}_raw", 0)
            if raw and raw != 65535:
                self._mode_switch = (now, int(raw))
            for axis, cfg in self.rc.items():
                if axis not in ("roll", "pitch", "yaw", "throttle"):
                    continue
                ch = cfg["chan"]
                if not 1 <= ch <= 18:
                    continue
                if now - self.__dict__.get("_override_t", {}).get(ch, -1e9) \
                        < self.PILOT_SETTLE_S:
                    continue
                raw = getattr(rc_msg, f"chan{ch}_raw", 0)
                if raw and raw != 65535:
                    self._pilot_pwm[axis] = (now, int(raw))

    def pilot_pwm(self, axis, max_age=0.6):
        with self._lock:
            v = self._pilot_pwm.get(axis)
        if v is None or (perf_counter() - v[0]) > max_age:
            return None
        return v[1]

    def _exact_pwm(self, axis, norm):
        cfg = self.rc[axis]
        norm = _clamp(norm, -1.0, 1.0)
        lo, hi = float(cfg["min"]), float(cfg["max"])
        dz = float(cfg.get("dz", 0))
        if axis == "throttle":
            span = hi - lo - dz
            mid_pwm = (int(lo) + int(hi)) // 2
            mid = int(1000.0 * (mid_pwm - lo - dz) / span)
            if norm == 0.0:
                pwm = float(mid_pwm)
            elif norm > 0.0:
                top = mid + self.thr_deadzone
                pwm = lo + dz + (top + norm * (1000.0 - top)) * span / 1000.0
            else:
                bot = mid - self.thr_deadzone
                pwm = lo + dz + (bot + norm * bot) * span / 1000.0
            if cfg["reversed"]:
                pwm = hi - (pwm - lo)
                away_up = pwm <= mid_pwm
            else:
                away_up = pwm >= mid_pwm
            pwm = (math.ceil(pwm - 1e-9) if away_up else math.floor(pwm + 1e-9))
            return int(_clamp(pwm, lo, hi))
        if cfg["reversed"]:
            norm = -norm
        tr = float(cfg["trim"])
        if norm > 0.0:
            pwm = math.ceil(tr + dz + norm * (hi - tr - dz) - 1e-9)
        elif norm < 0.0:
            pwm = math.floor(tr - dz + norm * (tr - dz - lo) + 1e-9)
        else:
            pwm = tr
        return int(_clamp(pwm, lo, hi))

    def _exact_norm(self, axis, pwm):
        cfg = self.rc[axis]
        lo, hi = float(cfg["min"]), float(cfg["max"])
        dz = float(cfg.get("dz", 0))
        pwm = _clamp(float(pwm), lo, hi)
        if axis == "throttle":
            if cfg["reversed"]:
                pwm = hi - (pwm - lo)
            span = hi - lo - dz
            c = 1000.0 * (pwm - lo - dz) / span if pwm > lo + dz else 0.0
            mid = int(1000.0 * ((int(lo) + int(hi)) // 2 - lo - dz) / span)
            top, bot = mid + self.thr_deadzone, mid - self.thr_deadzone
            if c > top:
                return _clamp((c - top) / (1000.0 - top), 0.0, 1.0)
            if c < bot and bot > 0:
                return _clamp((c - bot) / bot, -1.0, 0.0)
            return 0.0
        tr = float(cfg["trim"])
        if pwm < tr - dz and tr - dz > lo:
            n = (pwm - (tr - dz)) / ((tr - dz) - lo)
        elif pwm > tr + dz and hi > tr + dz:
            n = (pwm - (tr + dz)) / (hi - (tr + dz))
        else:
            n = 0.0
        n = _clamp(n, -1.0, 1.0)
        return -n if cfg["reversed"] else n

    def _axis_pwm(self, axis, norm):
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

    def ai_pwm(self, cmd):
        out = {}
        for axis in self.axes:
            if axis in self.rc:
                out[axis] = self._axis_pwm(axis, cmd.get(self._AXIS_FIELD[axis], 0.0))
        return out

    def steer_pwm(self, pwm_by_axis):
        chans = [0] * 8
        sent = {}
        for axis, pwm in pwm_by_axis.items():
            if axis == "throttle" and self.mode_name not in ANGLE_MODES:
                continue
            cfg = self.rc.get(axis)
            if not cfg or not 1 <= cfg["chan"] <= 8 or not pwm:
                continue
            chans[cfg["chan"] - 1] = int(pwm)
            sent[cfg["chan"]] = int(pwm)
        return self._commit_override(chans, sent)

    def steer(self, cmd):
        chans = [0] * 8
        sent = {}
        for axis in self.axes:
            if axis == "throttle" and self.mode_name not in ANGLE_MODES:
                continue
            cfg = self.rc.get(axis)
            if not cfg or not 1 <= cfg["chan"] <= 8:
                continue
            pwm = self._axis_pwm(axis, cmd.get(self._AXIS_FIELD[axis], 0.0))
            chans[cfg["chan"] - 1] = pwm
            sent[cfg["chan"]] = pwm
        return self._commit_override(chans, sent)

    def _commit_override(self, chans, sent):
        now = perf_counter()
        with self._lock:
            ot = self.__dict__.setdefault("_override_t", {})
            for ch in sent:
                ot[ch] = now
            if not sent:
                self._steer_since = None
            self._sent_pwm = sent
            for ch, pwm in sent.items():
                h = self._sent_hist.setdefault(ch, [])
                h.append((now, pwm))
                cut = now - self.ECHO_HISTORY_S
                while h and h[0][0] < cut:
                    h.pop(0)
            for ch in list(self._sent_hist):
                if ch not in sent:
                    del self._sent_hist[ch]
            if self._steer_since is None and sent:
                self._steer_since = now
        return self._send_override(chans)

    def release(self):
        with self._lock:
            self._sent_pwm = {}
            self._sent_hist = {}
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
            try:
                for _ in range(5):
                    self._send_override([0] * 8)
            except Exception:
                pass
            try:
                self.master.close()
            except Exception:
                pass
            self.master = None

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



MSP_API_VERSION = 1
MSP_FC_VARIANT = 2
MSP_FC_VERSION = 3
MSP_RX_MAP = 64
MSP_STATUS = 101
MSP_RC = 105
MSP_SET_RAW_RC = 200


def _msp_frame(cmd, payload=b""):
    n = len(payload)
    crc = n ^ cmd
    for b in payload:
        crc ^= b
    return b"$M<" + bytes((n, cmd)) + payload + bytes((crc & 0xFF,))


class BetaflightAcro(FcBackend):

    name = "betaflight"

    _MAP_LETTERS = "AERT"

    def __init__(self, conn, baud=115200, axes=("yaw", "pitch"), verbose=True):
        super().__init__()
        self.conn = conn
        self.baud = int(baud)
        self.axes = tuple(axes)
        self.verbose = verbose
        self.ser = None
        self.variant = "?"
        self.chan_index = {}
        self.n_channels = 8
        self._live = [1500] * 16
        self._sent = {}
        self._steer_since = None
        self._echo_warned = False
        self._io = threading.Lock()

    def _say(self, msg):
        if self.verbose:
            pass

    def connect(self):
        try:
            import serial
        except ImportError:
            raise RuntimeError("pyserial not installed:  pip install pyserial")

        self.ser = serial.Serial(self.conn, self.baud, timeout=0.3)

        variant = self._request(MSP_FC_VARIANT, timeout=2.0)
        if not variant:
            raise WrongFirmware("no MSP response — not Betaflight/INAV either")
        self.variant = variant[:4].decode(errors="ignore")
        ver = self._request(MSP_FC_VERSION) or b"\0\0\0"
        self.fw_string = (f"{self.variant} "
                          f"{ver[0]}.{ver[1]}.{ver[2]}" if len(ver) >= 3 else self.variant)
        self.linked = True


        if self.variant not in ("BTFL", "INAV", "EMUF", "CLFL"):
            pass

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

    def _request(self, cmd, payload=b"", timeout=1.0):
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
                _crc = self.ser.read(1)
                if cmd == want_cmd:
                    return bytes(buf)
                state = 0
        return None

    def _read_rx_map(self):
        raw = self._request(MSP_RX_MAP)
        order = "AETR"
        if raw:
            letters = []
            for idx in raw[:4]:
                letters.append(self._MAP_LETTERS[idx] if idx < 4 else "?")
            order = "".join(letters)
        axis_letter = {"roll": "A", "pitch": "E", "yaw": "R", "throttle": "T"}
        for axis in ("roll", "pitch", "yaw", "throttle"):
            letter = axis_letter[axis]
            self.chan_index[axis] = order.find(letter) if letter in order else None




    def _poll_status(self):
        raw = self._request(MSP_STATUS)
        if raw and len(raw) >= 11:
            flags = struct.unpack_from("<I", raw, 6)[0]
            self.armed = bool(flags & 1)
            self.mode_name = "ARMED" if self.armed else "DISARMED"
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
        distinctive = {i: v for i, v in self._sent.items()
                       if abs(v - 1500) > 40 and i < len(self._live)}
        if not distinctive:
            return
        ok = all(abs(self._live[i] - v) <= 25 for i, v in distinctive.items())
        if ok:
            if self.override_confirmed is not True:
                pass

            self.override_confirmed = True
        elif perf_counter() - self._steer_since > 2.0:
            self.override_confirmed = False
            if not self._echo_warned:
                self._echo_warned = True








    _AXIS_FIELD = ArduPilotAcro._AXIS_FIELD

    def steer(self, cmd):
        self._poll_rc()
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



class OfflineBackend(FcBackend):
    name = "offline"

    def connect(self):
        self.fw_string = "none (offline)"
        self.mode_name = "SIM"


    def steer(self, cmd):
        return False

    def release(self):
        return False

    def statustext(self, text):
        pass

    def describe(self):
        return "firmware      : none (offline / print-only)"



def build_backend(conn, baud=115200, axes=("yaw", "pitch"), transport="auto",
                  force_sysid=None, verbose=True, steer_modes=None,
                  commit_chan=0):
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
                pass
            be.close()
        except Exception as e:
            errors.append(f"{kind}: {e}")
            be.close()
            if transport != "auto":
                raise

    raise RuntimeError("could not talk to the flight controller on "
                       f"{conn}:\n  " + "\n  ".join(errors))



class AcroCommander:

    def __init__(self,
                 conn="udpin:127.0.0.1:14551",
                 baud=115200,
                 transport="auto",
                 axes=("yaw", "pitch"),
                 law=None,
                 send_hz=20.0,
                 statustext_hz=2.0,
                 no_target="release",
                 target_timeout=0.5,
                 force_sysid=None,
                 log_dir=None,
                 require_armed=True,
                 require_rate_mode=True,
                 verbose=True,
                 steer_modes=None,
                 commit_chan=0,
                 commit_abort_below=0.0,
                 engage_max_s=0.0,
                 mission=False,
                 mission_min_conf=0.20,
                 mission_ceiling_m=30.0,
                 mission_require_airborne=True):
        self.conn = conn
        self.mission_enabled = bool(mission)
        self.mission = None
        self._mission_cfg = dict(min_conf=float(mission_min_conf),
                                 ceiling_m=float(mission_ceiling_m))
        self.mission_require_airborne = bool(mission_require_airborne)
        self._mission_out = {}
        self._mission_why = ""
        self._det_t = None
        self._det_conf = None
        self._det_bbox = None
        self._cap_age = None
        self.baud = baud
        self.transport = transport
        self.axes = tuple(axes)
        self.law = law or ControlLaw()
        self.wasp_guidance = WaspGuidance()
        self.camera_mount_deg = float(os.environ.get("CAMERA_MOUNT_DEG",
                                      math.degrees(getattr(self.law, "cam_tilt", math.pi/6))))
        self.steer_modes = frozenset(steer_modes or RATE_MODES)
        self.commit_chan = int(commit_chan or 0)
        self.commit_abort_below = float(commit_abort_below)
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
        self._flight_logs = None
        self.log_context = {}
        self._log_error = None
        self._last_stt = 0.0
        self._t0 = perf_counter()
        self._warned_disarmed = False
        self._steering = False
        self._mode_gate = None
        self._rate_mode_ok = True

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
        return isinstance(self.law, InterceptLaw)

    @property
    def rate_mode_ok(self):
        return bool(self._rate_mode_ok)

    @property
    def override_confirmed(self):
        return None if self.backend is None else self.backend.override_confirmed

    @property
    def steering(self):
        return self._steering

    def connect(self):
        self.backend = build_backend(self.conn, baud=self.baud, axes=self.axes,
                                     transport=self.transport,
                                     force_sysid=self.force_sysid,
                                     verbose=self.verbose,
                                     steer_modes=self.steer_modes,
                                     commit_chan=self.commit_chan)

        if isinstance(self.backend, ArduPilotAcro):
            if self.is_intercept:
                self.law.max_yaw_rate = self.backend.pilot_yaw_rate
                self.law.angle_max = self.backend.angle_max
                self.law.max_climb_mps = self.backend.max_climb_mps
                _dn = getattr(self.backend, "max_descent_mps", None)
                if _dn:
                    self.law.max_descent_mps = _dn
                self.law.acro_rp_rate = self.backend.acro_rp_rate
                self.law.acro_yaw_rate = self.backend.acro_yaw_rate
            else:
                self.law.max_yaw_rate = self.backend.acro_yaw_rate
                self.law.max_pitch_rate = self.backend.acro_rp_rate






        untouched = [a for a in ("roll", "pitch", "yaw", "throttle")
                     if a not in self.axes]






        want = "/".join(sorted(self.steer_modes))





        if self.commit_chan:
            pass



        elif self.is_intercept:
            pass







        if self.engage_max_s > 0:
            pass


        blockers = list(self.backend.blockers)
        if self.is_intercept:
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


            for i, b in enumerate(blockers, 1):
                pass

        if self.is_intercept:






            if "ACRO" in self.steer_modes:
                pass







        else:
            pass





        if self.mission_enabled:
            self._build_mission()
        self._open_log()
        self._stop.clear()
        self._tx = threading.Thread(target=self._tx_loop, daemon=True)
        self._tx.start()

    def compute(self, err_x, err_y, state, fw, fh, bbox=None, capture_t=None):
        if self.mission is not None and self.mission.wasp.enabled:
            cmd = self.law.neutral()
            if state == "SEARCH":
                self.wasp_guidance.reset()
                return cmd
            b = self.backend
            att = b.wasp_attitude(at_t=capture_t) if isinstance(b, ArduPilotAcro) else None
            if att is None or capture_t is None:
                self.wasp_guidance.reset()
                return cmd
            try:
                cmd.update(self.wasp_guidance.observe(
                    err_x, err_y, fw, fh, self.law._tan_h, self.law._tan_v,
                    att, capture_t, self.mission.wasp.params["camera_roll"]))
                cmd["camera_roll"] = self.mission.wasp.params["camera_roll"]
            except (ValueError, TypeError, KeyError, ZeroDivisionError):
                self.wasp_guidance.reset()
            return cmd
        if state == "SEARCH":
            return self.law.neutral()

        if self.is_intercept:
            att = (self.backend.attitude(at_t=capture_t)
                   if self.backend is not None else None)
            commit = self.backend.commit() if self.backend is not None else None
            if self.commit_chan and commit is None:
                commit = 0.0
            bw = bh = None
            if bbox is not None and len(bbox) >= 4:
                bw, bh = bbox[2], bbox[3]
            age = None if capture_t is None else max(0.0, perf_counter() - capture_t)
            rate_mode = bool(self.backend is not None
                             and self.backend.mode_name in RATE_MODES)
            self._cap_age = age
            return self.law.compute(err_x, err_y, fw, fh, bbox_w=bw, bbox_h=bh,
                                    att=att, commit=commit, capture_age=age,
                                    rate_mode=rate_mode)

        pitch_rad = throttle = None
        if self.backend is not None:
            pitch_rad = self.backend.pitch_rad()
            throttle = self.backend.throttle_frac()
        return self.law.compute(err_x, err_y, fw, fh,
                                measured_pitch_rad=pitch_rad,
                                throttle_frac=(throttle or 0.0))

    def neutral_command(self):
        return self.law.neutral()

    def search_command(self):
        return self.law.neutral()

    def update(self, cmd, target_off, have_target, detected=None, conf=None,
               bbox=None):
        now = perf_counter()
        with self._lock:
            self._cmd = dict(cmd)
            self._target_off = target_off
            if have_target:
                self._last_target_t = now
            if detected:
                self._det_t = now
                self._det_conf = None if conf is None else float(conf)
                self._det_bbox = bbox
        return self._steering

    def _tx_loop(self):
        period = 1.0 / self.send_hz
        next_t = perf_counter()
        while not self._stop.is_set():
            now = perf_counter()
            if self.mission is not None:
                try:
                    self._mission_tx(now)
                except Exception as exc:
                    # A dead TX thread lets the overrides lapse mid-launch, so
                    # keep repeating the last WASP output until the pilot ends it.
                    print(f"[CTRL] mission cycle failed: {exc!r}", flush=True)
                    if (self.mission.wasp.started is not None and self._mission_out
                            and self.linked):
                        try:
                            self.backend.steer_wasp_pwm(self._mission_out, in_flight=True)
                        except Exception:
                            pass
                next_t += period
                dt = next_t - perf_counter()
                if dt > 0:
                    self._stop.wait(dt)
                else:
                    next_t = perf_counter()
                continue
            with self._lock:
                cmd = dict(self._cmd)
                off = self._target_off
                fresh = (now - self._last_target_t) <= self.target_timeout

            armed = self.backend.armed if self.backend else False
            have_target = off is not None and fresh
            rate_ok = (not self.require_rate_mode
                       or self.backend is None
                       or self.backend.in_rate_mode())
            self._rate_mode_ok = rate_ok
            self._announce_mode_gate(rate_ok)
            steer = bool(self.linked and have_target and rate_ok
                         and (armed or not self.require_armed))

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




            if self.linked and (now - self._last_stt) >= (1.0 / self.statustext_hz):
                self._last_stt = now
                self.backend.statustext(self._status_text(cmd, off, steer))

            self._log_row(now, cmd, off, steer, written)

            next_t += period
            dt = next_t - perf_counter()
            if dt > 0:
                self._stop.wait(dt)
            else:
                next_t = perf_counter()

    def _build_mission(self):
        import mission as _mission_mod
        b = self.backend
        _bs = os.environ.get("MISSION_BLEND_S", "").strip()
        if _bs:
            _mission_mod.Mission.BLEND_S = max(0.0, float(_bs))
        self.mission = _mission_mod.Mission(
            self, min_conf=self._mission_cfg["min_conf"],
            ceiling_m=self._mission_cfg["ceiling_m"],
            accel_z=getattr(b, "accel_z", 2.5) or 2.5,
            jerk_z=getattr(b, "jerk_z", 5.0) or 5.0,
            log=lambda m: None, idle_path=_mission_mod.IDLE_FILE, wasp_path=_mission_mod.WASP_FILE,
            auto_path=_mission_mod.AUTO_FILE, levels_path=_mission_mod.LEVELS_FILE,
            flight_path=_mission_mod.FLIGHT_FILE)
        _saved = _mission_mod.load_auto_params(self.mission)
        _lv = _mission_mod.load_levels(self.mission)
        _mission_mod.load_idle_settings(self.mission)
        _mission_mod.load_flight_settings(self.mission)











        for line in self.mission_checklist():
            pass


    _INVERT = {"roll": "invert_roll", "pitch": "invert_pitch",
               "yaw": "invert_yaw", "throttle": "invert_throttle"}

    def _scale(self, axis, v):
        b = self.backend
        if axis == "throttle":
            up = b.max_climb_mps or 2.5
            dn = getattr(b, "max_descent_mps", None) or up
            return v / up if v >= 0.0 else v / dn
        if axis in ("roll", "pitch"):
            return math.radians(v) / (b.angle_max or math.radians(30.0))
        return math.radians(v) / (b.pilot_yaw_rate or math.radians(202.5))

    def phys_to_pwm(self, axis, v):
        n = self._scale(axis, float(v))
        if getattr(self.law, self._INVERT[axis], False):
            n = -n
        return self.backend._exact_pwm(axis, n)

    def pwm_to_phys(self, axis, pwm):
        n = self.backend._exact_norm(axis, pwm)
        if getattr(self.law, self._INVERT[axis], False):
            n = -n
        b = self.backend
        if axis == "throttle":
            up = b.max_climb_mps or 2.5
            dn = getattr(b, "max_descent_mps", None) or up
            return n * (up if n >= 0.0 else dn)
        if axis in ("roll", "pitch"):
            return math.degrees(n * (b.angle_max or math.radians(30.0)))
        return math.degrees(n * (b.pilot_yaw_rate or math.radians(202.5)))

    def pilot_pwm(self, axis):
        f = getattr(self.backend, "pilot_pwm", None)
        return f(axis) if f else None

    def ai_pwm(self, cmd):
        return self.backend.ai_pwm(cmd)

    def wasp_pwm(self, rates, throttle):
        rates = tuple(-v if getattr(self.law, self._INVERT[a], False) else v
                      for a, v in zip(("roll", "pitch", "yaw"), rates))
        return self.backend.wasp_pwm(rates, throttle)

    def idle_pwm(self, yaw_dps, climb_mps):
        if not isinstance(self.backend, ArduPilotAcro):
            raise ValueError("Idle requires ArduPilot MAVLink")
        return self.backend.idle_pwm(yaw_dps, climb_mps)

    def idle_request_alt_hold(self):
        return isinstance(self.backend, ArduPilotAcro) and self.backend.idle_request_alt_hold()

    def _mission_gate(self):
        b = self.backend
        if not self.linked:
            return False, "no FC link"
        if self.require_armed and not b.armed:
            return False, "disarmed"
        wasp = self.mission is not None and self.mission.wasp.enabled
        idle = self.mission is not None and self.mission.idle.active
        if idle:
            if not isinstance(b, ArduPilotAcro):
                return False, "Idle requires ArduPilot MAVLink"
            why = (b.recovery_flight_ready(switching=not self.mission.idle.confirmed)
                   if self.mission.idle.recovery else
                   b.idle_ready(allow_acro=not self.mission.idle.confirmed))
            if why:
                return False, why
        elif wasp:
            if not isinstance(b, ArduPilotAcro):
                return False, "WASP requires ArduPilot MAVLink"
            if self.mission.wasp.started is not None:
                why = b.wasp_flight_ready()
                if why:
                    return False, why
                switch = b.pilot_mode_switch()
                if switch and switch[1] and switch[1] != "ACRO":
                    return False, f"pilot switched to {switch[1]}"
            else:
                if not self.mission.wasp.camera_confirmed or self.mission.camera_mount_deg != 90:
                    return False, "confirm camera fixed at 90 degrees in PARAMETERS > WASP"
                why = b.wasp_ready()
                if why:
                    return False, why
        elif self.mission is not None and (abs(self.mission.camera_mount_deg - 30) > 0.1
                or abs(math.degrees(getattr(self.law, "cam_tilt", math.pi/6)) - 30) > 0.1):
            return False, "restore and confirm the 30 degree camera for AUTO / SEMI-AUTO"
        elif b.mode_name != "ALT_HOLD":
            return False, f"FC in {b.mode_name}, not ALT_HOLD"
        if not wasp and not idle and self.mission_require_airborne and hasattr(b, "airborne"):
            air, how = b.airborne()
            if not air:
                return False, f"on the ground ({how})"
        if self.commit_chan:
            c = b.commit()
            if c is None:
                return False, "commit channel not read yet"
            if c < max(self.commit_abort_below, 1e-6):
                return False, f"commit switch down ({c:.2f})"
        return True, ""

    def _mission_tx(self, now):
        ok, why = self._mission_gate()
        self._mission_why = why
        b = self.backend
        h = b.height(max_age=.5) if isinstance(b, ArduPilotAcro) else b.height() if hasattr(b, "height") else None
        with self._lock:
            cmd = dict(self._cmd)
            off = self._target_off
            det_t, det_conf = self._det_t, self._det_conf
        ai_t = det_t if det_t is not None else -1e9
        fresh = (now - ai_t) <= self.target_timeout and off is not None
        idle_why = b.idle_ready(allow_acro=True) if isinstance(b, ArduPilotAcro) else "Idle requires MAVLink"
        idle_ack = b._idle_mode_ack if isinstance(b, ArduPilotAcro) else None
        # The pilot flipping ACRO -> STABILIZE mid-launch hands WASP to its
        # recovery instead of straight to a stick that may be at the bottom.
        switch = b.pilot_mode_switch() if isinstance(b, ArduPilotAcro) else None
        takeover = bool(not ok and self.mission.wasp.enabled and self.mission.wasp.started is not None
                        and isinstance(b, ArduPilotAcro) and b.linked and b.armed
                        and (b.mode_name == "STABILIZE" or (switch and switch[1] == "STABILIZE")))
        ctx = dict(gate_ok=ok, gate_why=why, pilot_takeover=takeover,
                   mode_switch=None if switch is None else switch[0],
                   att=((b.wasp_attitude() if isinstance(b, ArduPilotAcro) else b.attitude())
                        if ok or takeover else None),
                   height=None if h is None else h[0],
                   climb=None if h is None else h[1],
                   ai_cmd=cmd if fresh else None,
                   ai_t=ai_t if fresh else None,
                   det_t=det_t, det_conf=det_conf,
                   fc_mode=b.mode_name,
                   airborne=b.airborne()[0] if isinstance(b, ArduPilotAcro) else False,
                   idle_ok=not idle_why, idle_why=idle_why,
                   idle_mode_denied=bool(self.mission.idle.active and not self.mission.idle.confirmed
                       and idle_ack and idle_ack[0] >= self.mission.idle.started
                       and idle_ack[1] not in (mavutil.mavlink.MAV_RESULT_ACCEPTED, mavutil.mavlink.MAV_RESULT_IN_PROGRESS)),
                   idle_mode_accepted=bool(self.mission.idle.active and idle_ack
                       and idle_ack[0] >= self.mission.idle.started
                       and idle_ack[1] == mavutil.mavlink.MAV_RESULT_ACCEPTED))
        out = self.mission.cycle(now, ctx)
        if out:
            if self.mission.idle.active:
                written = b.steer_idle_pwm(out, allow_acro=not self.mission.idle.confirmed,
                                           recovery=self.mission.idle.recovery)
            else:
                written = (b.steer_wasp_pwm(out, in_flight=self.mission.wasp.started is not None)
                           if self.mission.wasp.enabled else b.steer_pwm(out))
        else:
            written = b.release() if self.linked else False
        self._mission_out = dict(out)
        self._steering = bool(out)
        self._rate_mode_ok = ok
        if self.linked and (now - self._last_stt) >= (1.0 / self.statustext_hz):
            self._last_stt = now
            b.statustext(f"WASP {self.mission.phase} "
                         + (f"h{h[0]:.1f}" if h else ""))
        self._log_row(now, cmd, off, bool(out), written)

    def mission_checklist(self):
        p = getattr(self.backend, "params", {}) or {}

        def v(name):
            return p.get(name)

        rows = []

        def row(name, want, ok, why):
            cur = v(name)
            cur_s = "?" if cur is None else (f"{cur:g}")
            rows.append(f"{'ok ' if ok else '!! '}{name:<17s} {cur_s:>8s}  "
                        f"-> {want:<8s} {why}")

        up, dn = v("PILOT_SPEED_UP"), v("PILOT_SPEED_DN")
        row("PILOT_SPEED_UP", "2500", up is not None and up >= 2500,
            "25 m/s: the AI's climb authority")
        row("PILOT_SPEED_DN", "250", dn is not None and 0 < dn <= 300,
            "slow descents (0 would copy UP = 25 m/s down)")
        row("PILOT_ACCEL_Z", "500", v("PILOT_ACCEL_Z") is not None,
            "sets how fast climbs start/stop (stop distance uses it)")
        row("ANGLE_MAX", "4000", (v("ANGLE_MAX") or 0) >= 4000,
            "40 deg: PUSH FAST and the AI's lean")
        row("RC_OVERRIDE_TIME", "1", (v("RC_OVERRIDE_TIME") or 3.0) <= 1.0,
            "a hung Jetson's last command dies in 1 s, not 3")
        row("ATC_INPUT_TC", "0.2-0.25", (v("ATC_INPUT_TC") or 0.15) >= 0.2,
            "softer attitude response to a new command: less kick")
        lb = v("LOG_BACKEND_TYPE")
        row("LOG_BACKEND_TYPE", "1 + SD", bool(lb), "fit an SD card: no FC logs exist")
        row("INS_LOG_BAT_MASK", "1", bool(v("INS_LOG_BAT_MASK")),
            "IMU batch log for vibration FFT")
        return rows

    def mission_state(self):
        now = perf_counter()
        b = self.backend
        st = dict(app="mission" if self.mission is not None else "auto",
                  gate_why=self._mission_why)
        if self.mission is not None:
            st.update(self.mission.snapshot(now))
            st["wasp_setup_why"] = (b.wasp_setup_error() if isinstance(b, ArduPilotAcro)
                                    else "WASP requires ArduPilot MAVLink")
            st["wasp_attitude_fresh"] = bool(isinstance(b, ArduPilotAcro)
                                            and b.wasp_attitude() is not None)
        h = b.height() if b is not None and hasattr(b, "height") else None
        st["height_m"] = None if h is None else round(h[0], 2)
        st["climb_mps"] = None if h is None else round(h[1], 2)
        st["heading_deg"] = None if h is None else round(h[2])
        vib = b.vibration() if b is not None and hasattr(b, "vibration") else None
        st["vibe"] = None if vib is None else dict(
            x=round(vib["x"], 1), y=round(vib["y"], 1), z=round(vib["z"], 1),
            clip=vib["clip"])
        with self._lock:
            det_t, conf = self._det_t, self._det_conf
            cmd = dict(self._cmd)
            fresh = (now - self._last_target_t) <= self.target_timeout
        if self.mission is not None and self.mission.wasp.enabled:
            fresh = (bool(cmd.get("wasp")) and bool(cmd.get("have_att"))
                     and 0 <= now - cmd.get("capture_t", -1e9) <= 0.25)
        st["det_age_s"] = None if det_t is None else round(now - det_t, 2)
        st["det_conf"] = None if conf is None else round(conf, 2)
        st["target_fresh"] = bool(fresh)
        st["target_az_deg"] = round(cmd.get("az_deg", 0.0), 1) if fresh else None
        st["target_el_deg"] = round(cmd.get("el_cam_deg", 0.0), 1) if fresh else None
        st["pwm"] = dict(self._mission_out)
        return st

    def _commit_gate(self, steer):
        if not steer or self.backend is None:
            return steer
        c = self.backend.commit()
        if c is None:
            return steer if not self.commit_chan else False
        if self.commit_abort_below <= 0.0:
            return steer
        if c >= self.commit_abort_below:
            if self._commit_aborted:
                self._commit_aborted = False


            return steer
        if not self._commit_aborted:
            self._commit_aborted = True



        return False

    def _engage_gate(self, steer, now):
        if self.engage_max_s <= 0.0:
            return steer
        if not steer:
            self._engaged_since = None
            self._engage_timed_out = False
            return steer
        if self._engage_timed_out:
            return False
        if self._engaged_since is None:
            self._engaged_since = now
        if (now - self._engaged_since) <= self.engage_max_s:
            return True
        self._engage_timed_out = True



        return False

    def _announce_mode_gate(self, rate_ok):
        if rate_ok == self._mode_gate:
            return
        first = self._mode_gate is None
        self._mode_gate = rate_ok
        if not self.linked:
            return
        mode = self.mode_name
        want = "/".join(sorted(self.steer_modes))
        if rate_ok and not self.require_rate_mode:
            pass


        elif rate_ok:
            pass

        elif first:
            pass


        else:
            pass


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
        if self._flight_logs is not None:
            self._flight_logs.close(perf_counter())
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def _open_log(self):
        if not self.log_dir:
            return
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        stamp = (os.environ.get("WASP_RUN_STAMP")
                 or datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.log_path = str(Path(self.log_dir) / f"control_{stamp}.csv")
        self._log_fh = open(self.log_path, "w", buffering=1)
        cols = ("t_s,mode,armed,steering,written,confirmed,"
                "ex,ey,az_deg,el_deg,yaw_stick,pitch_stick,"
                "yaw_rate_dps,pitch_rate_dps")
        if self.is_intercept:
            cols += (",roll_stick,throttle_stick,el_cam_deg,bank_deg,lean_deg,"
                     "cmd_climb_mps,los_az_dps,los_el_dps,range_m,closing_mps,"
                     "latency_s,commit,have_att,terminal,"
                     "sens,pn_gain_eff,bank_cap_deg,fill_frac,lean_el_gate")
        if self.mission_enabled:
            cols += (",phase,pwm_roll,pwm_pitch,pwm_yaw,pwm_thr,"
                     "meas_roll_deg,meas_pitch_deg,meas_yaw_deg,"
                     "rollspeed_dps,pitchspeed_dps,yawspeed_dps,"
                     "height_m,climb_mps,vibe_x,vibe_y,vibe_z,clip,"
                     "det_age_s,det_conf,bbox_w,bbox_h,cap_age_s")
            self._flight_logs = FlightLogs(self.log_dir, stamp, cols)
            cols = self._flight_logs.header
        self._log_fh.write(cols + "\n")


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
                        f",{int(bool(cmd.get('terminal')))}"
                        f",{cmd.get('sens', 0.0):.2f}"
                        f",{cmd.get('pn_gain_eff', 0.0):.1f}"
                        f",{cmd.get('bank_cap_deg', 0.0):.2f}"
                        f",{cmd.get('fill_frac', 0.0):.5f}"
                        f",{cmd.get('lean_el_gate', 1.0):.3f}")
            if self.mission_enabled:
                row += self._mission_csv(now)
                if self._flight_logs is not None and self.mission is not None:
                    row = self._flight_logs.record(now, row, self._flight_log_state(now),
                                                   lambda: self._flight_metadata(now))
            self._log_fh.write(row + "\n")
            self._log_error = None
        except Exception as exc:
            if str(exc) != self._log_error:
                self._log_error = str(exc)


    def _flight_log_state(self, now):
        m = self.mission
        w = m.wasp.progress(now)
        idle = m.idle.snapshot()
        with self._lock:
            det_t, confidence = self._det_t, self._det_conf
            context = dict(self.log_context)
            cmd = dict(self._cmd)
        timeout = .25 if m.wasp.enabled else m.TARGET_FRESH_S
        fresh = (det_t is not None and 0 <= now-det_t <= timeout
                 and confidence is not None and confidence >= m.min_conf)
        rates = m.wasp._rates if m.phase == "WASP_FLIGHT" else (None, None, None)
        return dict(t_s=round(now-self._t0, 3), phase=m.phase,
                    armed=self.armed, linked=self.linked,
                    wasp_enabled=m.wasp.enabled, auto_run=m.auto_run,
                    idle_limit_s=m.idle_s, idle_stage=idle["stage"],
                    idle_height_target_m=idle["height_m"],
                    idle_heading_target_deg=idle["heading_deg"],
                    detection_fresh=int(fresh), wasp_stage=w["stage"] if m.wasp.enabled else "",
                    wasp_steering_blend=w["steering_blend"] if m.wasp.enabled else "",
                    wasp_target_acquired=int(w["target_acquired"]) if m.wasp.enabled else "",
                    **{f"cmd_{axis}_rate_dps": None if rate is None else round(math.degrees(rate), 3)
                       for axis, rate in zip(("roll", "pitch", "yaw"), rates)},
                    cam_fps=context.get("cam_fps"), processing_fps=context.get("processing_fps"),
                    mission_event=m.event, mission_event_t=m.event_t,
                    gate_reason=self._mission_why)

    def _flight_metadata(self, now):
        with self._lock:
            runtime = dict(self.log_context)
        return dict(runtime=runtime, mission=self.mission.snapshot(now),
                    fc_mode=self.mode_name,
                    fc_params=dict(getattr(self.backend, "params", {})),
                    rc_calibration=dict(getattr(self.backend, "rc", {})),
                    control_law=type(self.law).__name__, send_hz=self.send_hz)

    def update_log_context(self, **values):
        with self._lock:
            self.log_context.update(values)

    def logging_state(self):
        return dict(control_csv=self.log_path, control_error=self._log_error,
                    flight=None if self._flight_logs is None else self._flight_logs.snapshot())

    def _mission_csv(self, now):
        b = self.backend
        o = self._mission_out
        ph = self.mission.phase if self.mission is not None else ""
        att = b.attitude() if b is not None else None
        h = b.height() if hasattr(b, "height") else None
        vib = b.vibration() if hasattr(b, "vibration") else None

        def f(x, fmt):
            return "" if x is None else fmt % x
        a = (["", "", "", "", "", ""] if att is None else
             [f(math.degrees(att[k]), "%+.2f") for k in
              ("roll", "pitch", "yaw", "rollspeed", "pitchspeed", "yawspeed")])
        with self._lock:
            det_t, conf, bb, age = (self._det_t, self._det_conf,
                                    self._det_bbox, self._cap_age)
        bw = bh = None
        if bb is not None and len(bb) >= 4:
            bw, bh = bb[2], bb[3]
        return ("," + ph
                + "".join(f",{o.get(ax, 0)}" for ax in
                          ("roll", "pitch", "yaw", "throttle"))
                + "," + ",".join(a)
                + f",{'' if h is None else '%.2f' % h[0]}"
                + f",{'' if h is None else '%+.2f' % h[1]}"
                + (",,,," if vib is None else
                   f",{vib['x']:.2f},{vib['y']:.2f},{vib['z']:.2f},{vib['clip']}")
                + f",{'' if det_t is None else '%.3f' % (now - det_t)}"
                + f",{f(conf, '%.3f')},{f(bw, '%d')},{f(bh, '%d')}"
                + f",{f(age, '%.4f')}")

    def format_command(self, state, cmd, target_off, sent, live, is_yolo):
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
        elif not self._rate_mode_ok and self.mission is not None:
            dest = f"PILOT has control ({self._mission_why or 'gate'})"
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
    law = InterceptLaw(
        h_fov_deg=_env_f("CAM_HFOV_DEG", 62.2),
        v_fov_deg=_env_f("CAM_VFOV_DEG", 48.8),
        cam_tilt_deg=_env_f("CAM_TILT_DEG", 0.0),
        kp_yaw=_env_f("KP_YAW", 4.5),
        roll_pn_gain=_env_f("ROLL_PN_GAIN", 45.0),
        max_bank_deg=_env_f("MAX_BANK_DEG", 30.0),
        bank_far_deg=_env_f("BANK_FAR_DEG", None),
        sched_lo_frac=_env_f("SCHED_LO_FRAC", 0.02),
        sched_hi_frac=_env_f("SCHED_HI_FRAC", 0.35),
        pn_near_mult=_env_f("PN_NEAR_MULT", 1.0),
        sens=_env_f("SENS_DEFAULT", 0.5),
        sens_max_mult=_env_f("SENS_MAX_MULT", 2.0),
        lock_ramp_s=_env_f("LOCK_RAMP_S", 0.0),
        fill_tau=_env_f("FILL_TAU", 0.30),
        max_lean_deg=_env_f("MAX_LEAN_DEG", 20.0),
        lean_el_guard=_env_f("LEAN_EL_GUARD", 0.0),
        climb_los=_env_b("CLIMB_LOS", False),
        edge_guard=_env_f("EDGE_GUARD", 0.0),
        cruise_mps=_env_f("CRUISE_MPS", 15.0),
        cruise_lean_deg=_env_f("CRUISE_LEAN_DEG", 20.0),
        climb_trim=_env_f("CLIMB_TRIM", 6.0),
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
        target_width_m=_env_f("TARGET_WIDTH_M", 0.30),
        terminal_bbox_frac=_env_f("TERMINAL_BBOX_FRAC", 0.35),
        terminal_range_m=_env_f("TERMINAL_RANGE_M", 2.5),
        terminal_hold_s=_env_f("TERMINAL_HOLD_S", 1.0),
        invert_yaw=_env_b("YAW_REVERSED", False),
        invert_pitch=_env_b("PITCH_REVERSED", False),
        invert_roll=_env_b("ROLL_REVERSED", False),
        invert_throttle=_env_b("THROTTLE_REVERSED", False),
        att_kp=_env_f("ATT_KP", 6.0),
    )
    conn = os.environ.get("MAV_CONN", "udpin:127.0.0.1:14551")
    if conn.strip().lower() == "none":
        conn = None
    sysid = os.environ.get("MAV_SOURCE_SYSTEM")
    commit_chan = _env_i("COMMIT_CHAN", 0)
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
        engage_max_s=_env_f("ENGAGE_MAX_S", 0.0),
        mission=os.environ.get("WASP_APP", "mission").strip().lower() == "mission",
        mission_min_conf=_env_f("ENGAGE_MIN_CONF", 0.20),
        mission_ceiling_m=_env_f("MISSION_CEILING_M", 30.0),
        mission_require_airborne=_env_b("MISSION_REQUIRE_AIRBORNE", True),
        verbose=verbose,
    )
