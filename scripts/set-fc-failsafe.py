#!/usr/bin/env python3
"""Set the FC's GCS failsafe params so a Jetson death triggers RTL.

Stop wasp-tracker first — it owns the MAVLink serial link while running.

    sudo systemctl stop wasp-tracker
    python3 scripts/set-fc-failsafe.py                 # default: /dev/ttyACM0 @ 115200
    python3 scripts/set-fc-failsafe.py --port /dev/ttyACM0 --baud 115200
    sudo systemctl start wasp-tracker

Sets:
  FS_GCS_ENABLE  = 1    (RTL on Jetson loss)
  FS_GCS_TIMEOUT = 3    (seconds without a heartbeat before RTL fires)
"""
import argparse
import sys
import time

try:
    from pymavlink import mavutil
except ImportError:
    sys.exit("pymavlink missing.  Install:  pip3 install pymavlink")


TARGETS = {
    "FS_GCS_ENABLE": 1.0,
    "FS_GCS_TIMEOUT": 3.0,
}


def fetch(mav, name, timeout=3.0):
    mav.mav.param_request_read_send(mav.target_system, mav.target_component,
                                    name.encode(), -1)
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg is None:
            continue
        pid = msg.param_id
        if isinstance(pid, bytes):
            pid = pid.decode("ascii", errors="ignore")
        if pid.strip("\x00") == name:
            return float(msg.param_value)
    return None


def set_param(mav, name, value):
    mav.mav.param_set_send(mav.target_system, mav.target_component,
                           name.encode(), float(value),
                           mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    t0 = time.time()
    while time.time() - t0 < 3.0:
        msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg is None:
            continue
        pid = msg.param_id
        if isinstance(pid, bytes):
            pid = pid.decode("ascii", errors="ignore")
        if pid.strip("\x00") == name:
            return float(msg.param_value)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    print(f"[set-failsafe] connecting {args.port} @ {args.baud}")
    mav = mavutil.mavlink_connection(args.port, baud=args.baud,
                                     source_system=255, source_component=190)
    hb = mav.wait_heartbeat(timeout=15)
    if hb is None:
        sys.exit("no heartbeat — is the FC on, and is wasp-tracker stopped?")
    print(f"[set-failsafe] FC up: sysid={mav.target_system} comp={mav.target_component}")

    failed = []
    for name, target in TARGETS.items():
        before = fetch(mav, name)
        print(f"  {name}: {before}  ->  {target}", end="", flush=True)
        after = set_param(mav, name, target)
        if after is None:
            print("   NO ACK")
            failed.append(name)
        elif abs(after - target) > 1e-3:
            print(f"   got {after} (refused)")
            failed.append(name)
        else:
            print("   OK")

    if failed:
        sys.exit(f"failed: {', '.join(failed)}")
    print("\n[set-failsafe] done.  FC will now RTL if the Jetson stops heartbeating for 3 s.")


if __name__ == "__main__":
    main()
