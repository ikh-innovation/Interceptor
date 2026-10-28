#!/usr/bin/env python3
"""Compare a flight controller's parameters with the reference dump, one by one.

    fc_param_check.py                      # read-only report
    fc_param_check.py --apply              # then offer each difference, y/N each
    fc_param_check.py --conn /dev/ttyACM0  # FC directly (stop wasp-tracker first)

By default it listens on udpin:127.0.0.1:14550, the copy of the FC link that
run.py's MAVProxy forwards for Mission Planner, so the app can keep running.

Board-specific parameters (sensor calibrations and IDs, board settings) are
listed but never written: a new FC needs its own accelerometer, compass, radio
and ESC calibration.  Nothing is written while the FC is armed, and every write
is read back.
"""
import argparse
import math
import sys
import time
from pathlib import Path

from pymavlink import mavutil

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT/"deploy"/"fc"/"TBS_LUCID_H7_2026-10-28.param"

# Required by the app or by SafeDown: a difference here blocks WASP or SafeDown.
CRITICAL = {
    "ACRO_TRAINER", "RC_OVERRIDE_TIME", "RC_OPTIONS", "SYSID_MYGCS", "FLTMODE_CH",
    "FLTMODE1", "FLTMODE2", "FLTMODE3", "FLTMODE4", "FLTMODE5", "FLTMODE6",
    "FLTMODE_GCSBLOCK", "RCMAP_ROLL", "RCMAP_PITCH", "RCMAP_THROTTLE", "RCMAP_YAW",
    "RC5_OPTION", "FRAME_CLASS", "FRAME_TYPE", "SERVO1_FUNCTION", "SERVO2_FUNCTION",
    "SERVO3_FUNCTION", "SERVO4_FUNCTION", "ACRO_RP_RATE", "ACRO_Y_RATE",
    "ACRO_RP_EXPO", "ACRO_Y_EXPO", "LAND_SPEED", "LAND_SPEED_HIGH", "LAND_ALT_LOW",
    "LAND_REPOSITION", "PILOT_ACCEL_Z", "PILOT_SPEED_UP", "PILOT_SPEED_DN",
    "PILOT_THR_BHV", "THR_DZ", "DISARM_DELAY", "ANGLE_MAX", "FS_THR_ENABLE",
    "FS_THR_VALUE", "FS_GCS_ENABLE", "ARMING_RUDDER", "MOT_THST_HOVER", "MOT_PWM_TYPE",
    # GPS and compass: SafeDown home, the phone's ARM and the geofence need them. Written
    # although COMPASS_ / GPS / SERIAL look board-specific: same FC model, same wiring.
    "COMPASS_ENABLE", "COMPASS_USE", "COMPASS_AUTODEC", "EK3_SRC1_POSXY", "EK3_SRC1_VELXY",
    "EK3_SRC1_POSZ", "EK3_SRC1_YAW", "GPS1_TYPE", "GPS_AUTO_CONFIG", "SERIAL2_PROTOCOL",
    "RTL_ALT", "RTL_SPEED", "RTL_LOIT_TIME", "RTL_ALT_FINAL", "FS_EKF_ACTION", "FENCE_ENABLE",
}
# Flight behaviour copied from the reference: tuning, filters, failsafes, EKF.
TUNING_PREFIXES = ("ATC_", "ACRO_", "PILOT_", "PSC_", "LAND_", "MOT_", "FS_", "BATT_FS",
                   "EK3_", "AHRS_", "INS_GYRO_FILTER", "INS_ACCEL_FILTER", "INS_HNTCH",
                   "INS_HNTC2", "INS_LOG", "WPNAV_", "SCHED_", "ARMING_", "RC_", "RCMAP_",
                   "FLTMODE", "SERVO", "FRAME_", "ANGLE_MAX", "DISARM_", "THR_", "LOG_")
# Specific to one board or its sensors: report, never write.
BOARD_PREFIXES = ("BRD_", "STAT_", "SERIAL", "CAN_", "GPS_", "GPS1_", "GPS2_", "BARO",
                  "COMPASS_", "OSD", "NTF_", "VTX_", "RELAY", "CAM", "MNT", "SR")
BOARD_PATTERNS = ("_ID", "_DEVID", "OFFS", "SCAL", "_CALTEMP", "TCAL", "_OFS", "_DIA",
                  "_ODI", "_MOT", "_ORIENT", "TRIM_X", "TRIM_Y", "_PRIO")
# The transmitter's calibration: the same radio should match, but recalibrate.
RADIO_PATTERNS = ("_MIN", "_MAX", "_TRIM", "_DZ", "_REVERSED")


def category(name):
    if name in CRITICAL:
        return "CRITICAL"
    if name.startswith("RC") and name[2:3].isdigit() and name.endswith(RADIO_PATTERNS):
        return "RADIO"
    if name.startswith("AHRS_TRIM") or name.startswith(BOARD_PREFIXES) or any(p in name for p in BOARD_PATTERNS):
        return "BOARD"
    if name.startswith(TUNING_PREFIXES):
        return "TUNING"
    return "OTHER"


def load_reference(path):
    ref = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.replace("\t", ",").partition(",")
        try:
            ref[name.strip()] = float(value)
        except ValueError:
            pass
    return ref


def same(a, b):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)


def fmt(v):
    if v is None:
        return "-"
    return str(int(v)) if float(v).is_integer() else f"{v:.6g}"


class Fc:
    def __init__(self, conn, baud):
        self.m = mavutil.mavlink_connection(conn, baud=baud, source_system=254)
        hb = self.m.wait_heartbeat(timeout=15)
        if hb is None:
            sys.exit(f"no heartbeat on {conn}")
        self.hb = hb

    def armed(self):
        hb = self.m.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
        hb = hb or self.hb
        return bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def read_all(self, timeout=60):
        self.m.mav.param_request_list_send(self.m.target_system, self.m.target_component)
        params, seen, total = {}, set(), None
        start = last = time.time()
        while time.time() - last < 5 and time.time() - start < timeout:
            msg = self.m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
            if msg is None:
                if total and len(params) < total:
                    # Lost on the link: ask again for the missing indices.
                    for i in [i for i in range(total) if i not in seen][:20]:
                        self.m.mav.param_request_read_send(self.m.target_system,
                                                           self.m.target_component, b"", i)
                continue
            last = time.time()
            name = msg.param_id if isinstance(msg.param_id, str) else msg.param_id.decode()
            params[name.strip("\x00")] = float(msg.param_value)
            seen.add(msg.param_index)
            total = msg.param_count
            if len(params) >= total:
                break
        return params, total

    def set(self, name, value):
        self.m.mav.param_set_send(self.m.target_system, self.m.target_component, name.encode(),
                                  float(value), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        end = time.time() + 3
        while time.time() < end:
            msg = self.m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
            if msg is not None:
                got = msg.param_id if isinstance(msg.param_id, str) else msg.param_id.decode()
                if got.strip("\x00") == name:
                    return float(msg.param_value)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="udpin:127.0.0.1:14550")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--reference", default=str(REFERENCE))
    ap.add_argument("--apply", action="store_true", help="offer each difference, one by one")
    ap.add_argument("--all", action="store_true", help="also list parameters that match")
    a = ap.parse_args()

    ref = load_reference(a.reference)
    fc = Fc(a.conn, a.baud)
    cur, total = fc.read_all()
    print(f"FC: system {fc.m.target_system}, {len(cur)}/{total} parameters read; "
          f"reference: {len(ref)} from {Path(a.reference).name}\n")
    if total and len(cur) < total:
        print("WARNING: not every parameter was read; run again before trusting this.\n")

    rows = []
    for name in sorted(set(ref) | set(cur)):
        r, c = ref.get(name), cur.get(name)
        status = ("MATCH" if same(r, c) else "MISSING ON FC" if c is None
                  else "NOT IN REFERENCE" if r is None else "DIFFERENT")
        rows.append((category(name), name, r, c, status))

    order = ("CRITICAL", "TUNING", "RADIO", "OTHER", "BOARD")
    for cat in order:
        shown = [x for x in rows if x[0] == cat and (a.all or x[4] != "MATCH")]
        n = sum(1 for x in rows if x[0] == cat)
        bad = sum(1 for x in rows if x[0] == cat and x[4] != "MATCH")
        print(f"== {cat}: {n - bad}/{n} match" + ("" if cat != "BOARD" else
              "  (calibrations and sensor IDs: differences are expected, never copied)"))
        for _cat, name, r, c, status in shown:
            print(f"   {name:<22} reference {fmt(r):>12}   FC {fmt(c):>12}   {status}")
        print()

    # The checks the app itself makes, on the FC's actual values.
    v = lambda k: cur.get(k)
    checks = [
        ("ACRO_TRAINER = 0", v("ACRO_TRAINER") == 0),
        ("0 < RC_OVERRIDE_TIME <= 1", v("RC_OVERRIDE_TIME") is not None and 0 < v("RC_OVERRIDE_TIME") <= 1),
        ("RC_OPTIONS bit 1 clear", v("RC_OPTIONS") is not None and not int(v("RC_OPTIONS")) & 2),
        ("SYSID_MYGCS = 255", v("SYSID_MYGCS") == 255),
        ("FLTMODE_GCSBLOCK allows STABILIZE and ALT_HOLD", not int(v("FLTMODE_GCSBLOCK") or 0) & 5),
        ("FLTMODE_CH = 6, switch ACRO / STABILIZE / ALT_HOLD",
         v("FLTMODE_CH") == 6 and v("FLTMODE1") == 1 and v("FLTMODE6") == 2
         and all(v(f"FLTMODE{i}") == 0 for i in range(2, 6))),
        ("RC5_OPTION = 154 (arm switch)", v("RC5_OPTION") == 154),
    ]
    checks += [
        # SafeDown flies home, the phone arms and the geofence works only with these.
        ("EK3_SRC1_POSXY = 3 and EK3_SRC1_VELXY = 3 (GPS in the EKF)", v("EK3_SRC1_POSXY") == 3 and v("EK3_SRC1_VELXY") == 3),
        ("COMPASS_ENABLE = 1 and COMPASS_USE = 1 (no heading, no GPS)", v("COMPASS_ENABLE") == 1 and v("COMPASS_USE") == 1),
        ("COMPASS_AUTODEC = 1 (declination from the GPS)", v("COMPASS_AUTODEC") == 1),
        ("SERIAL2_PROTOCOL = 5 (GPS port)", v("SERIAL2_PROTOCOL") == 5),
        ("a compass detected (COMPASS_DEV_ID set)", bool(v("COMPASS_DEV_ID"))),
        ("FS_EKF_ACTION = 1 (LAND on a bad EKF)", v("FS_EKF_ACTION") == 1),
    ]
    print("== App, SafeDown, GPS and ARM checks on the FC's values")
    for label, okay in checks:
        print(f"   {'OK  ' if okay else 'FAIL'} {label}")
    print()

    todo = [x for x in rows if x[0] in ("CRITICAL", "TUNING", "RADIO", "OTHER")
            and x[4] == "DIFFERENT"]
    if not a.apply:
        print(f"{len(todo)} differences outside BOARD. Run with --apply to set them one by one.")
        return 0
    if fc.armed():
        print("The FC is ARMED: nothing will be written. Disarm and run again.")
        return 1
    print(f"{len(todo)} differences outside BOARD. For each: y = set the reference value, "
          "anything else = keep.\n")
    changed = 0
    for cat, name, r, c, _status in todo:
        answer = input(f"[{cat}] {name}: FC {fmt(c)} -> reference {fmt(r)} ? [y/N] ").strip().lower()
        if answer != "y":
            continue
        if fc.armed():
            print("The FC was ARMED: stopping.")
            break
        got = fc.set(name, r)
        if same(got, r):
            changed += 1
            print(f"   {name} = {fmt(got)}  (read back)")
        else:
            print(f"   {name}: NOT CONFIRMED (read back {fmt(got)})")
    print(f"\n{changed} parameters written.")
    if changed:
        print("Next: reboot the FC, then calibrate accelerometer, compass, radio and ESCs, "
              "check motor order and direction (props off), and restart wasp-tracker.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
