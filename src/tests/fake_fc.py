#!/usr/bin/env python3
import threading
from time import perf_counter, sleep
from pymavlink import mavutil


class FakeArduPilot(threading.Thread):

    def __init__(self, port, sysid_mygcs=255, rc_options=0, armed=True,
                 mode=1, sysid=1, attitude=True):
        super().__init__(daemon=True)
        self.port = port
        self.sysid_mygcs = sysid_mygcs
        self.rc_options = rc_options
        self.armed = armed
        self.mode = mode
        self.sysid = sysid
        self.attitude = attitude
        self.mode_requests = []
        self.mode_ack = mavutil.mavlink.MAV_RESULT_ACCEPTED
        self.apply_mode = True
        # Test knobs: heartbeat period / on-off, VFR_HUD on-off, and a delay
        # before the DO_SET_MODE ACK (the mode is applied with the ACK).
        self.hb_period = 0.15
        self.heartbeat = True
        self.vfr_on = True
        self.ack_delay = 0.0
        self._pending_acks = []
        # SafeDown knobs: modes this FC refuses, the landed state it reports
        # (None: no EXTENDED_SYS_STATE), the IMU it reports (None: none), and
        # LAND touching down and disarming after land_s seconds.
        self.deny_modes = set()
        self.landed_state = None
        self.imu = None
        self.land_s = 1.0
        self.disarm_requests = []
        self._land_t = None
        # GPS: an EKF with a real position and a home on the stand; RTL flies
        # home and lands there, disarming after rtl_s seconds.
        self.gps = True
        self.home = (37.9838, 23.7275)
        self.position = (37.9840, 23.7278)
        self.rtl_s = 1.5
        # No compass: a GPS fix on the ground, but the EKF has a position only
        # once it flies (ekf_position False until then).
        self.ekf_position = True
        self.set_home_requests = []
        self.att = dict(roll=0.0, pitch=0.0, rollspeed=0.0, pitchspeed=0.0,
                        yawspeed=0.0, yaw=0.0)
        self.stop = threading.Event()
        self.overrides = {}
        self.rejected_from = set()
        self.radio = {i: 1500 for i in range(1, 19)}
        self.vfr = dict(alt=2.0, climb=0.0, heading=0)
        self.vibe = dict(x=8.0, y=9.0, z=12.0, clip=0)
        self.params = {
            "SYSID_MYGCS": float(sysid_mygcs),
            "RC_OPTIONS": float(rc_options),
            "ACRO_TRAINER": 0.0,
            "ACRO_Y_RATE": 202.5,
            "ACRO_RP_RATE": 360.0,
            "ANGLE_MAX": 3000.0,
            "PILOT_Y_RATE": 202.0,
            "PILOT_SPEED_UP": 250.0,
            "THR_DZ": 100.0,
            "RCMAP_YAW": 4.0, "RCMAP_PITCH": 2.0,
            "RCMAP_ROLL": 1.0, "RCMAP_THROTTLE": 3.0,
            "RC4_MIN": 1100.0, "RC4_MAX": 1900.0, "RC4_TRIM": 1500.0,
            "RC4_REVERSED": 0.0,
            "RC2_MIN": 1100.0, "RC2_MAX": 1900.0, "RC2_TRIM": 1500.0,
            "RC2_REVERSED": 0.0,
            "RC1_MIN": 1100.0, "RC1_MAX": 1900.0, "RC1_TRIM": 1500.0,
            "RC1_REVERSED": 0.0,
            "RC3_MIN": 1000.0, "RC3_MAX": 2000.0, "RC3_TRIM": 1002.0,
            "RC3_REVERSED": 0.0,
            "RC6_MIN": 1000.0, "RC6_MAX": 2000.0, "RC6_TRIM": 1500.0,
            "RC6_REVERSED": 0.0,
        }

    def run(self):
        self.m = mavutil.mavlink_connection(f"udpout:127.0.0.1:{self.port}",
                                            source_system=self.sysid,
                                            source_component=1)
        t_hb = t_rc = t_att = 0.0
        while not self.stop.is_set():
            now = perf_counter()
            while self._pending_acks and self._pending_acks[0][0] <= now:
                _due, command, mode = self._pending_acks.pop(0)
                self._ack_mode(command, mode)
            if self.heartbeat and now - t_hb > self.hb_period:
                t_hb = now
                self._send_heartbeat()
            if self.mode in (9, 6) and self.armed:
                self._land_t = self._land_t or now
                if now - self._land_t >= (self.land_s if self.mode == 9 else self.rtl_s):
                    # LAND: touchdown, then the FC disarms itself.
                    self.armed = False
                    if self.landed_state is not None:
                        self.landed_state = 1
            else:
                self._land_t = None
            if now - t_rc > 0.1:
                t_rc = now
                vals = [self.overrides.get(i, self.radio[i]) for i in range(1, 19)]
                self.m.mav.rc_channels_send(int(now * 1000), 8, *vals, 100)
            if self.vfr_on and now - t_att > 0.05:
                self.m.mav.vfr_hud_send(0.0, 0.0, int(self.vfr["heading"]), 0,
                                        float(self.vfr["alt"]),
                                        float(self.vfr["climb"]))
                self.m.mav.vibration_send(int(now * 1e6), self.vibe["x"],
                                          self.vibe["y"], self.vibe["z"],
                                          int(self.vibe["clip"]), 0, 0)
                if self.landed_state is not None:
                    self.m.mav.extended_sys_state_send(0, int(self.landed_state))
                self.m.mav.ekf_status_report_send(0x37 if self.gps and self.ekf_position else 0xa7, 0, 0, 0, 0, 0)
                if self.gps:
                    self.m.mav.gps_raw_int_send(int(now*1e6), 3, int(self.position[0]*1e7), int(self.position[1]*1e7),
                                                int(self.vfr["alt"]*1000), 100, 100, 0, 0, 10)
                if self.gps:
                    lat, lon = self.home if self.mode == 6 and self._land_t else self.position
                    self.m.mav.global_position_int_send(int(now*1000), int(lat*1e7), int(lon*1e7),
                                                        int(self.vfr["alt"]*1000), int(self.vfr["alt"]*1000),
                                                        0, 0, 0, 0)
                    self.m.mav.home_position_send(int(self.home[0]*1e7), int(self.home[1]*1e7), 0,
                                                  0, 0, 0, [1, 0, 0, 0], 0, 0, 0)
                if self.imu is not None:
                    ax, ay, az = (int(v*1000/9.80665) for v in self.imu)
                    self.m.mav.scaled_imu_send(int(now * 1000), ax, ay, az, 0, 0, 0, 0, 0, 0)
            if self.attitude and now - t_att > 0.05:
                t_att = now
                a = self.att
                self.m.mav.attitude_send(
                    int(now * 1000), a["roll"], a["pitch"], a["yaw"],
                    a["rollspeed"], a["pitchspeed"], a["yawspeed"])
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
            elif t == "COMMAND_INT" and msg.command == mavutil.mavlink.MAV_CMD_DO_SET_HOME:
                self.set_home_requests.append((msg.x*1e-7, msg.y*1e-7, msg.z))
                self.home = (msg.x*1e-7, msg.y*1e-7)
                self.m.mav.command_ack_send(msg.command, mavutil.mavlink.MAV_RESULT_ACCEPTED)
            elif t == "COMMAND_LONG" and msg.command == mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE:
                if int(msg.param1) == mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT and self.heartbeat:
                    self._send_heartbeat()
            elif t == "COMMAND_LONG" and msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                if int(msg.param1) == 0:
                    # Like ArduCopter: a GCS disarm only once it has landed.
                    landed = self.landed_state == 1
                    self.disarm_requests.append((perf_counter(), landed))
                    if landed:
                        self.armed = False
                    self.m.mav.command_ack_send(msg.command, mavutil.mavlink.MAV_RESULT_ACCEPTED if landed
                                                else mavutil.mavlink.MAV_RESULT_FAILED)
            elif t == "COMMAND_LONG" and msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
                self.mode_requests.append((perf_counter(), int(msg.param2)))
                if self.ack_delay:
                    self._pending_acks.append((perf_counter()+self.ack_delay, msg.command, int(msg.param2)))
                else:
                    self._ack_mode(msg.command, int(msg.param2))
            elif t == "RC_CHANNELS_OVERRIDE":
                if msg.get_srcSystem() != self.sysid_mygcs:
                    self.rejected_from.add(msg.get_srcSystem())
                    continue
                if int(self.rc_options) & 2:
                    continue
                for i in range(1, 9):
                    v = getattr(msg, f"chan{i}_raw", 0)
                    if v == 0:
                        self.overrides.pop(i, None)
                    elif v != 65535:
                        self.overrides[i] = v

    def _send_heartbeat(self):
        base = 0x80 if self.armed else 0
        self.m.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_QUADROTOR,
            mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
            base, self.mode, mavutil.mavlink.MAV_STATE_ACTIVE)

    def _ack_mode(self, command, mode):
        result = mavutil.mavlink.MAV_RESULT_DENIED if mode in self.deny_modes else self.mode_ack
        self.m.mav.command_ack_send(command, result)
        if self.apply_mode and result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            self.mode = mode

    def shutdown(self):
        self.stop.set()
        self.join(timeout=2)
