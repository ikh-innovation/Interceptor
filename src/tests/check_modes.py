#!/usr/bin/env python3
"""check_modes.py — is the mode the tracker steers in actually REACHABLE?

The gap this closes.  run.py's preflight warns when the FC is not *currently* in
the steering mode, but it never asks the question that killed a demo: is that
mode assigned to ANY position of the pilot's mode switch?  On this airframe it
was not.  FLTMODE1..6 read 0,0,0,4,0,1 — STABILIZE/STABILIZE/STABILIZE/GUIDED/
STABILIZE/ACRO — with ALT_HOLD (mode 2) nowhere.  The mode gate can then never
open, the AI can never steer, and every log line says only "PILOT has control",
which reads like a tracking problem rather than a one-parameter FC problem.

It also checks the commit channel, which has its own trap: COMMIT_CHAN must not
be FLTMODE_CH.  fc_control only rejects a commit channel that collides with a
DRIVEN axis, so the flight-mode channel slips through — and then "commit" is
really the mode switch.  README and SCENARIO both say channel 6, which is
exactly FLTMODE_CH on this board.

Read-only: it sends no override and changes no parameter.  Safe armed or not.

    python3 tests/check_modes.py                       # auto-detect the FC
    python3 tests/check_modes.py --connect /dev/ttyACM0
    python3 tests/check_modes.py --steer-mode ACRO --commit-chan 7
"""
import argparse
import glob
import os
import sys
import time

from pymavlink import mavutil

# ArduPilot Copter flight-mode numbers, as they appear in FLTMODEn.
COPTER_MODES = {
    0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED",
    5: "LOITER", 6: "RTL", 7: "CIRCLE", 9: "LAND", 11: "DRIFT",
    13: "SPORT", 14: "FLIP", 15: "AUTOTUNE", 16: "POSHOLD", 17: "BRAKE",
    18: "THROW", 19: "AVOID_ADSB", 20: "GUIDED_NOGPS", 21: "SMART_RTL",
    22: "FLOWHOLD", 23: "FOLLOW", 24: "ZIGZAG", 25: "SYSTEMID",
    26: "AUTOROTATE", 27: "AUTO_RTL", 28: "TURTLE",
}

# The six PWM bands ArduPilot maps FLTMODE1..6 onto, on FLTMODE_CH.
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
    """Fetch a set of parameters by name, retrying the ones that do not answer."""
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
        print("  [FAIL] no flight controller found.  Power the FC and re-run.")
        return 2

    print("=" * 74)
    print("mode-switch reachability check")
    print(f"  link       : {port}")
    print(f"  steer mode : {a.steer_mode}")
    print("=" * 74)

    m = mavutil.mavlink_connection(port, baud=a.baud)
    if not m.wait_heartbeat(timeout=20):
        print("  [FAIL] no heartbeat.")
        return 2
    print(f"  [ok] heartbeat: sys={m.target_system} comp={m.target_component}")

    names = [f"FLTMODE{i}" for i in range(1, 7)] + ["FLTMODE_CH"]
    names += ["RCMAP_ROLL", "RCMAP_PITCH", "RCMAP_THROTTLE", "RCMAP_YAW"]
    p = read_params(m, names)

    missing = [n for n in names if n not in p]
    if missing:
        print(f"  [warn] these parameters did not answer: {', '.join(missing)}")

    fltmode_ch = int(p["FLTMODE_CH"]) if "FLTMODE_CH" in p else None
    print(f"\n== the pilot's mode switch  (FLTMODE_CH = "
          f"{fltmode_ch if fltmode_ch else '?'}) ==")

    want = a.steer_mode.strip().upper()
    found_at = []
    for i in range(1, 7):
        key = f"FLTMODE{i}"
        if key not in p:
            print(f"  pos {i}  {SWITCH_BANDS[i-1]:>10} us   ?")
            continue
        num = int(p[key])
        name = COPTER_MODES.get(num, f"mode {num}")
        mark = ""
        if name == want:
            found_at.append(i)
            mark = "   <== the tracker steers HERE"
        print(f"  pos {i}  {SWITCH_BANDS[i-1]:>10} us   {name}{mark}")

    problems = []

    if not found_at:
        problems.append(
            f"{want} is not assigned to ANY switch position.  The mode gate can "
            f"never open and the tracker will never steer — every log line will "
            f"say 'PILOT has control'.  Set one FLTMODEn to {want}.")
    else:
        print(f"\n  [ok] {want} is reachable at switch position(s): "
              f"{', '.join(map(str, found_at))}")

    have_stab = [i for i in range(1, 7)
                 if f"FLTMODE{i}" in p and int(p[f"FLTMODE{i}"]) == 0]
    if not have_stab:
        problems.append(
            "STABILIZE is not on the switch either.  That is the mode the pilot "
            "takes off in and the primary abort — the drone must be able to get "
            "back to it.")
    else:
        print(f"  [ok] STABILIZE (take-off + abort) reachable at: "
              f"{', '.join(map(str, have_stab))}")

    # ── the commit channel ────────────────────────────────────────────────────
    driven = {}
    for axis in ("roll", "pitch", "throttle", "yaw"):
        k = f"RCMAP_{axis.upper()}"
        if k in p:
            driven[int(p[k])] = axis
    print(f"\n== channels ==")
    print("  driven by the AI : " + ", ".join(
        f"ch{c} ({ax})" for c, ax in sorted(driven.items())) or "  (unknown)")
    if fltmode_ch:
        print(f"  flight-mode      : ch{fltmode_ch}")

    cc = a.commit_chan
    if not cc:
        print("  commit/abort     : NONE configured (COMMIT_CHAN=0)")
        print("                     -> the mode switch is the only abort, noticed at")
        print("                        HEARTBEAT rate (~1 s).  A spare channel makes")
        print("                        it ~100 ms and adds a live speed dial.")
        free = [c for c in range(5, 9)
                if c not in driven and c != fltmode_ch]
        if free:
            print(f"                     -> free candidates on this FC: "
                  f"{', '.join('ch%d' % c for c in free)}")
    else:
        print(f"  commit/abort     : ch{cc}")
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
            print(f"  [ok] ch{cc} is neither a driven stick nor the mode channel")

    # ── live RC, so the pilot can prove the switch reaches the position ───────
    if a.watch > 0:
        print(f"\n== live RC in, {a.watch:.0f}s — TURN THE TRANSMITTER ON and flick "
              f"the mode switch ==")
        print("   (all zeros just means the transmitter is off — not a fault)")
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
                print(line)
                last = line
        if seen_modes:
            print(f"   modes reported while watching: {', '.join(sorted(seen_modes))}")

    print("\n" + "=" * 74)
    if problems:
        print("NOT READY — fix these on the flight controller:")
        for i, s in enumerate(problems, 1):
            print(f"  {i}. {s}")
        print("=" * 74)
        return 1
    print(f"READY — {want} is on the switch, STABILIZE is on the switch, and the")
    print("commit channel does not collide.  The remaining untested thing is")
    print("roll/throttle POLARITY: do the props-off bench check before flying.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
