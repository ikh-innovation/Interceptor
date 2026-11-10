#!/usr/bin/env python3
"""Verify the FC has every parameter WASP requires to launch.

Stop wasp-tracker first; it owns the MAVLink serial link while running.

    sudo systemctl stop wasp-tracker
    python3 scripts/verify-fc-params.py
    sudo systemctl start wasp-tracker

Prints PASS/FAIL per requirement and an overall verdict.  Exit code 0 means
every mandatory check passed.
"""
import argparse
import sys
import time

try:
    from pymavlink import mavutil
except ImportError:
    sys.exit("pymavlink missing.  Install:  pip3 install pymavlink")


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


MANDATORY = [
    ("ACRO_TRAINER",     lambda v: v == 0,                   "must be 0 (no auto-level in ACRO)"),
    ("ACRO_RP_RATE",     lambda v: v and v > 0,              "must be > 0 (app uses it for stick mapping)"),
    ("ACRO_Y_RATE",      lambda v: v and v > 0,              "must be > 0"),
    ("ACRO_RP_EXPO",     lambda v: v is not None and 0 <= v <= 1, "must be 0..1"),
    ("ACRO_Y_EXPO",      lambda v: v is not None and 0 <= v <= 1, "must be 0..1"),
    ("RC_OVERRIDE_TIME", lambda v: v and 0 < v <= 1,         "must be > 0 and <= 1 second"),
    ("FLTMODE_GCSBLOCK", lambda v: v is not None and not (int(v) & 0b101), "bits 0 and 2 must be clear (STABILIZE and ALT_HOLD allowed)"),
    ("FS_GCS_ENABLE",    lambda v: int(v or 0) in (1, 2, 3, 4, 5), "must be 1 (RTL), 2 (LAND), 3-5 for Jetson-dead failsafe"),
    ("FS_GCS_TIMEOUT",   lambda v: v is None or v <= 5,      "must be <= 5 s"),
    ("RC_OPTIONS",       lambda v: v is None or not (int(v) & 2), "bit 1 (IGNORE_OVERRIDES) must be CLEAR"),
    ("SYSID_MYGCS",      lambda v: v is not None,            "must be set (matches Jetson sysid)"),
    ("FLTMODE_CH",       lambda v: v is not None and 1 <= v <= 18, "RC channel for mode switch"),
    ("FLTMODE1",         lambda v: int(v or -1) == 1,        "position 1 of mode switch must be ACRO (1) for WASP launch"),
    ("RC5_OPTION",       lambda v: int(v or 0) == 154,       "channel 5 should be arm/disarm switch (154)"),
]

OPTIONAL_GPS = [
    ("COMPASS_ENABLE",   lambda v: int(v or 0) == 1,         "compass on (only needed for geofence RTL or FS_GCS_ENABLE=1)"),
    ("EK3_SRC1_POSXY",   lambda v: int(v or 0) == 3,         "GPS as position source"),
    ("EK3_SRC1_VELXY",   lambda v: int(v or 0) == 3,         "GPS as velocity source"),
    ("EK3_SRC1_YAW",     lambda v: int(v or 0) == 1,         "compass as heading"),
    ("FS_EKF_ACTION",    lambda v: int(v or 0) == 1,         "LAND on EKF failure"),
    ("FENCE_ENABLE",     lambda v: v is None or int(v) == 0, "0 (we use app-level fence)"),
]


def check(mav, specs):
    results = []
    for name, ok, why in specs:
        v = fetch(mav, name)
        passed = ok(v) if v is not None else ok(None)
        results.append((name, v, passed, why))
    return results


def report(title, results):
    print(f"\n=== {title} ===")
    for name, v, passed, why in results:
        tag = "PASS" if passed else "FAIL"
        vs = "missing" if v is None else f"{v:g}"
        print(f"  [{tag}]  {name:20s} = {vs:10s}   {why}")
    fails = sum(1 for _, _, p, _ in results if not p)
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    print(f"[verify] connecting {args.port} @ {args.baud}")
    mav = mavutil.mavlink_connection(args.port, baud=args.baud,
                                     source_system=255, source_component=190)
    hb = mav.wait_heartbeat(timeout=15)
    if hb is None:
        sys.exit("no heartbeat — FC on?  wasp-tracker stopped?")
    print(f"[verify] FC up: sysid={mav.target_system} comp={mav.target_component}")

    m_fails = report("MANDATORY (app refuses to ARM if any FAIL)", check(mav, MANDATORY))
    g_fails = report("GPS / COMPASS (only needed for geofence RTL or Jetson-dead RTL)",
                     check(mav, OPTIONAL_GPS))

    print()
    if m_fails == 0:
        print(f"OK  Mandatory: all PASS.  You can fly WASP (throttle takeover + UI SAFEDOWN).")
        if g_fails == 0:
            print(f"OK  GPS/Compass: all PASS.  Geofence RTL and Jetson-dead RTL are also live.")
        else:
            print(f"NOTE  GPS/Compass has {g_fails} FAIL.  Without them: no geofence, and")
            print(f"      Jetson-dead failsafe lands in place instead of RTL (if FS_GCS_ENABLE=2).")
        sys.exit(0)
    else:
        print(f"NOT READY: {m_fails} mandatory FAIL.  Fix in Mission Planner before flying.")
        sys.exit(1)


if __name__ == "__main__":
    main()
