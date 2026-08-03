#!/usr/bin/env python3
"""Simulated ArduPilot Copter, reproducing the real sysid-gated RC override
behaviour.  Importable without running any tests."""
import threading
from time import perf_counter, sleep
from pymavlink import mavutil


class FakeArduPilot(threading.Thread):
    """Minimal ArduPilot Copter in ACRO: heartbeats, params, and the real
    sysid-gated RC override behaviour."""

    def __init__(self, port, sysid_mygcs=255, rc_options=0, armed=True,
                 mode=1, sysid=1):
        super().__init__(daemon=True)
        self.port = port
        self.sysid_mygcs = sysid_mygcs
        self.rc_options = rc_options
        self.armed = armed
        self.mode = mode
        self.sysid = sysid
        self.stop = threading.Event()
        self.overrides = {}            # chan(1-based) -> pwm, when accepted
        self.rejected_from = set()     # sysids whose overrides we dropped
        self.radio = {i: 1500 for i in range(1, 19)}
        self.params = {
            "SYSID_MYGCS": float(sysid_mygcs),
            "RC_OPTIONS": float(rc_options),
            "ACRO_TRAINER": 0.0,
            "ACRO_Y_RATE": 202.5,
            "ACRO_RP_RATE": 360.0,
            "RCMAP_YAW": 4.0, "RCMAP_PITCH": 2.0,
            "RC4_MIN": 1100.0, "RC4_MAX": 1900.0, "RC4_TRIM": 1500.0,
            "RC4_REVERSED": 0.0,
            "RC2_MIN": 1100.0, "RC2_MAX": 1900.0, "RC2_TRIM": 1500.0,
            "RC2_REVERSED": 0.0,
        }

    def run(self):
        # udpout -> the commander's udpin, mirroring the real MAVProxy router.
        self.m = mavutil.mavlink_connection(f"udpout:127.0.0.1:{self.port}",
                                            source_system=self.sysid,
                                            source_component=1)
        t_hb = t_rc = 0.0
        while not self.stop.is_set():
            now = perf_counter()
            if now - t_hb > 0.15:
                t_hb = now
                base = 0x80 if self.armed else 0
                self.m.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_QUADROTOR,
                    mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
                    base, self.mode, mavutil.mavlink.MAV_STATE_ACTIVE)
            if now - t_rc > 0.1:
                t_rc = now
                vals = [self.overrides.get(i, self.radio[i]) for i in range(1, 19)]
                self.m.mav.rc_channels_send(int(now * 1000), 8, *vals, 100)
            try:
                msg = self.m.recv_match(blocking=True, timeout=0.02)
            except Exception:
                msg = None
            if msg is None:
                continue
            t = msg.get_type()
            if t == "PARAM_REQUEST_READ":
                pid = msg.param_id
                if isinstance(pid, bytes):
                    pid = pid.decode(errors="ignore")
                pid = pid.strip("\x00")
                if pid in self.params:
                    self.m.mav.param_value_send(
                        pid.encode(), self.params[pid],
                        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
                        len(self.params), 0)
            elif t == "RC_CHANNELS_OVERRIDE":
                # THE rule under test.
                if msg.get_srcSystem() != self.sysid_mygcs:
                    self.rejected_from.add(msg.get_srcSystem())
                    continue
                if int(self.rc_options) & 2:          # IGNORE_OVERRIDES
                    continue
                for i in range(1, 9):
                    v = getattr(msg, f"chan{i}_raw", 0)
                    if v == 0:
                        self.overrides.pop(i, None)   # 0 = release to the pilot
                    elif v != 65535:
                        self.overrides[i] = v

    def shutdown(self):
        self.stop.set()
        self.join(timeout=2)


