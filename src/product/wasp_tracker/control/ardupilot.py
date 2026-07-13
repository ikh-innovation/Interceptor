from time import perf_counter

import numpy as np

try:
    from pymavlink import mavutil
except ImportError:
    mavutil = None


class ArduPilotCommander:
    GUIDED_MODE = 4
    FRAME_BODY_OFFSET_NED = 9
    TYPE_MASK = 0b010111000111

    def __init__(self, config):
        self.device = config.mav_device
        self.baud = config.mav_baud
        self.H_FOV = np.radians(config.h_fov_deg)
        self.V_FOV = np.radians(config.v_fov_deg)
        self.K_YAW = config.k_yaw
        self.K_VZ = config.k_vz
        self.FWD = {"LOCK": config.fwd_lock,
                    "PREDICT": config.fwd_predict,
                    "SEARCH": config.fwd_search}
        self.MAX_YAW = config.max_yaw
        self.MAX_VZ = config.max_vz
        self.MAX_FWD = config.max_fwd
        self.master = None
        self._t0 = perf_counter()

    def connect(self):
        if self.device is None:
            print("[MAV] no device configured — offline, commands are print-only")
            return
        if mavutil is None:
            raise RuntimeError("pymavlink not installed:  pip install pymavlink")

        print(f"[MAV] connecting {self.device} @ {self.baud} ...")
        self.master = mavutil.mavlink_connection(self.device, baud=self.baud)
        hb = self.master.wait_heartbeat(timeout=10)
        if hb is None:
            raise RuntimeError(f"[MAV] no heartbeat on {self.device} within 10 s")
        if hb.autopilot != mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
            raise RuntimeError(
                f"[MAV] FC is NOT running ArduPilot (autopilot id {hb.autopilot}). "
                "The Lucid H7 is likely flashed with Betaflight/INAV — MAVLink "
                "setpoints are the wrong transport there. Flash ArduPilot Copter "
                "or revisit the integration brief.")
        print(f"[MAV] ArduPilot heartbeat  sys={self.master.target_system} "
              f"comp={self.master.target_component}  mode={hb.custom_mode}")
        if hb.custom_mode != self.GUIDED_MODE:
            print("[MAV] WARNING: vehicle is not in GUIDED — velocity setpoints "
                  "are ignored until GUIDED is engaged (RC/GCS or set_guided()).")

    def set_guided(self):
        if self.master is None:
            return
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            self.GUIDED_MODE, 0, 0, 0, 0, 0)

    def close(self):
        if self.master is not None:
            self.send(dict(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0))
            self.master.close()
            self.master = None

    def compute(self, err_x, err_y, state, fw, fh):
        ang_x = err_x * (self.H_FOV / fw)
        ang_y = err_y * (self.V_FOV / fh)

        yaw_rate = float(np.clip(self.K_YAW * ang_x, -self.MAX_YAW, self.MAX_YAW))
        vz = float(np.clip(self.K_VZ * ang_y, -self.MAX_VZ, self.MAX_VZ))
        vx = float(np.clip(self.FWD.get(state, 0.0), 0.0, self.MAX_FWD))
        vy = 0.0

        return dict(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)

    def send(self, cmd):
        if self.master is None:
            return
        self.master.mav.set_position_target_local_ned_send(
            int((perf_counter() - self._t0) * 1000),
            self.master.target_system,
            self.master.target_component,
            self.FRAME_BODY_OFFSET_NED,
            self.TYPE_MASK,
            0.0, 0.0, 0.0,
            cmd["vx"], cmd["vy"], cmd["vz"],
            0.0, 0.0, 0.0,
            0.0, cmd["yaw_rate"])

    @staticmethod
    def zero_command():
        return dict(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)

    @staticmethod
    def format(cmd, state):
        return (
            f"[ARDUPILOT GUIDED BODY_OFFSET_NED]  "
            f"fwd:{cmd['vx']:+.2f} m/s  "
            f"lat:{cmd['vy']:+.2f} m/s  "
            f"vz:{cmd['vz']:+.2f} m/s  "
            f"yaw:{cmd['yaw_rate']:+.3f} rad/s"
            f"  | state:{state}"
        )
