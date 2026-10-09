#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import time

from pymavlink import mavutil

COPTER_MODES = {
    0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED",
    5: "LOITER", 6: "RTL", 7: "CIRCLE", 9: "LAND", 11: "DRIFT",
    13: "SPORT", 14: "FLIP", 15: "AUTOTUNE", 16: "POSHOLD", 17: "BRAKE",
    18: "THROW", 19: "AVOID_ADSB", 20: "GUIDED_NOGPS", 21: "SMART_RTL",
    22: "FLOWHOLD", 23: "FOLLOW", 24: "ZIGZAG", 25: "SYSTEMID",
    26: "AUTOROTATE", 27: "AUTO_RTL", 28: "TURTLE",
}

SWITCH_BANDS = ["<= 1230", "1231-1360", "1361-1490",
                "1491-1620", "1621-1749", ">= 1750"]


def find_port():
    for pat in ("/dev/serial/by-id/*ArduPilot*if00",
                "/dev/serial/by-id/*if00", "/dev/ttyACM0"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def read_params(m, names, timeout=25.0):
    got, deadline = {}, time.time() + timeout
    pending = list(names)
    while pending and time.time() < deadline:
        for n in pending:
            m.mav.param_request_read_send(
                m.target_system, m.target_component, n.encode(), -1)
        slice_end = time.time() + 3.0
        while time.time() < slice_end:
            msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
            if msg is None:
                continue
            name = msg.param_id.strip("\x00")
            if name in names:
                got[name] = msg.param_value
        pending = [n for n in pending if n not in got]
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default=None,
                    help="link to probe; default: auto-detect the FC serial")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--steer-mode",
                    default=os.getenv("FC_STEER_MODES", "ALT_HOLD").split(",")[0].strip(),
                    help="the mode the tracker steers in (default: FC_STEER_MODES "
                         "or ALT_HOLD)")
    ap.add_argument("--commit-chan", type=int,
                    default=int(os.getenv("COMMIT_CHAN", "0") or 0),
                    help="commit/abort channel to validate (0 = none configured)")
    ap.add_argument("--watch", type=float, default=6.0,
                    help="seconds to watch live RC in, so you can flick the "
                         "switch and see which position it lands on")
    a = ap.parse_args()

    port = a.connect or find_port()
    if not port:

        return 2







    m = mavutil.mavlink_connection(port, baud=a.baud)
    if not m.wait_heartbeat(timeout=20):

        return 2


    names = [f"FLTMODE{i}" for i in range(1, 7)] + ["FLTMODE_CH"]
    names += ["RCMAP_ROLL", "RCMAP_PITCH", "RCMAP_THROTTLE", "RCMAP_YAW"]
    p = read_params(m, names)

    missing = [n for n in names if n not in p]
    if missing:
        pass

    fltmode_ch = int(p["FLTMODE_CH"]) if "FLTMODE_CH" in p else None



    want = a.steer_mode.strip().upper()
    found_at = []
    for i in range(1, 7):
        key = f"FLTMODE{i}"
        if key not in p:

            continue
        num = int(p[key])
        name = COPTER_MODES.get(num, f"mode {num}")
        mark = ""
        if name == want:
            found_at.append(i)
            mark = "   <== the tracker steers HERE"


    problems = []

    if not found_at:
        problems.append(
            f"{want} is not assigned to ANY switch position.  The mode gate can "
            f"never open and the tracker will never steer — every log line will "
            f"say 'PILOT has control'.  Set one FLTMODEn to {want}.")
    else:
        pass


    have_stab = [i for i in range(1, 7)
                 if f"FLTMODE{i}" in p and int(p[f"FLTMODE{i}"]) == 0]
    if not have_stab:
        problems.append(
            "STABILIZE is not on the switch either.  That is the mode the pilot "
            "takes off in and the primary abort — the drone must be able to get "
            "back to it.")
    else:
        pass


    driven = {}
    for axis in ("roll", "pitch", "throttle", "yaw"):
        k = f"RCMAP_{axis.upper()}"
        if k in p:
            driven[int(p[k])] = axis



    if fltmode_ch:
        pass

    cc = a.commit_chan
    if not cc:




        free = [c for c in range(5, 9)
                if c not in driven and c != fltmode_ch]
        if free:
            pass

    else:

        if cc in driven:
            problems.append(
                f"COMMIT_CHAN={cc} is also the {driven[cc]} stick.  It would read "
                f"our own override back.  Move it to a spare channel.")
        elif fltmode_ch and cc == fltmode_ch:
            problems.append(
                f"COMMIT_CHAN={cc} is FLTMODE_CH — the flight-mode channel.  "
                f"'Commit' would be the mode switch, not a speed dial, and the "
                f"fast abort does not exist.  Move it to a spare channel.")
        elif not 1 <= cc <= 8:
            problems.append(
                f"COMMIT_CHAN={cc} is outside channels 1-8.")
        else:
            pass

    if a.watch > 0:



        end, last, seen_modes = time.time() + a.watch, None, set()
        while time.time() < end:
            msg = m.recv_match(type=["RC_CHANNELS", "HEARTBEAT"],
                               blocking=True, timeout=1)
            if msg is None:
                continue
            if msg.get_type() == "HEARTBEAT":
                seen_modes.add(mavutil.mode_string_v10(msg))
                continue
            vals = [getattr(msg, f"chan{i}_raw", 0) for i in range(1, 9)]
            line = "   " + " ".join(f"{v:5d}" for v in vals)
            if fltmode_ch and 1 <= fltmode_ch <= 8:
                line += f"   | mode ch{fltmode_ch} = {vals[fltmode_ch-1]}"
            if line != last:

                last = line
        if seen_modes:
            pass


    if problems:

        for i, s in enumerate(problems, 1):
            pass

        return 1




    return 0


if __name__ == "__main__":
    sys.exit(main())
