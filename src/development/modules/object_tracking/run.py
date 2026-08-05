#!/usr/bin/env python3
"""One-command launcher for the WASP head tracker.

The single entry point.  Everything the tracker needs — interpreter, gains, FC
discovery, display — is decided here so that what flies is never a function of
what somebody remembered to type.

What it does, in order:

  1. Re-execs itself under the yolo-env interpreter if it was started with a
     python that has no ultralytics (the system python3 does not).
  2. Applies the tuned gains as DEFAULTS — anything already set in the
     environment wins, so `KP_YAW=6 ./run.py` still works.
  3. WAITS for the flight controller to appear and answer a heartbeat, so this
     can be started BEFORE the drone is powered up.  Ctrl-C while waiting exits
     cleanly without ever touching the FC.
  4. Hands off to run_pipeline.py --direct.

The pilot's mode switch remains the handover control (REQUIRE_RATE_MODE=1, the
default): STABILIZE = the pilot has every channel, ACRO = the tracker drives
yaw and pitch.  Starting this script never takes control on its own.

    ./run.py                 the tuned preset.  There is only one.
    ./run.py --tuned         same thing; accepted so the systemd unit stays valid
    ./run.py --no-wait       fail immediately if the FC is not already up
    ./run.py -- --no-verify  everything after `--` goes to run_pipeline.py

Individual values still override, so a one-off experiment needs no new flag:

    KP_YAW=6 STICK_LIMIT_YAW=0.8 ./run.py

The pilot must hold MID THROTTLE throughout.  In ACRO the attitude controller's
authority scales with throttle: at idle the mixer has no headroom to differentiate
the motors, so even a full-stick yaw command produces almost no torque.  No gain
setting in here can compensate for a closed throttle.
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
YOLO_PY = Path.home() / "yolo-env" / "bin" / "python"

# ── 1. make sure we are on the interpreter that has ultralytics ───────────────
# The system python3 has pymavlink but NOT ultralytics, so a plain `python3
# run.py` would die deep inside the tracker instead of here.
if os.environ.get("_WASP_REEXEC") != "1" and YOLO_PY.exists():
    try:
        import ultralytics                                   # noqa: F401
    except ImportError:
        os.environ["_WASP_REEXEC"] = "1"
        os.execv(str(YOLO_PY), [str(YOLO_PY), str(Path(__file__).resolve())]
                 + sys.argv[1:])

# ── 2. the preset ─────────────────────────────────────────────────────────────
# TUNED is the only preset.  ADAPTIVE / MAX / FAST / GENTLE were removed after the
# field flights: extra knobs meant the thing that actually flew was whichever flag
# somebody last typed, which is the opposite of what a flight-control default is
# for.  One preset, tuned as hard as the hardware allows, is the whole point.
#
# Raised 2026-08-05 after the field test reported imprecise, sluggish movement.
# The numbers come from a grid search over KP x per-axis ceiling x KD x slew,
# scored on settling time across EIGHT starting geometries and then re-scored for
# robustness at worse latency and against 8 px of detection jitter on five noise
# seeds.  Against the previous values:
#
#                        settle @0.1s lat   @0.15s lat   stable w/ jitter
#     previous tuned          1.10 s          1.05 s          3 of 5 seeds
#     THIS                    0.50 s          0.90 s          5 of 5 seeds
#
# So it is both ~2x quicker onto the target and strictly more robust, not a trade.
#
# What changed, and why it is not symmetric:
#
#   YAW got the aggression.  Ceiling 0.35 -> 0.65 and slew 3 -> 11/s.  The slew
#   limiter was a hidden brake: at 3/s the stick needed a third of a second just
#   to reach its ceiling, so every turn began late.  That, more than KP, is what
#   "not precise" felt like.  KP 3.5 -> 4.5.
#
#   PITCH got its ceiling LOWERED, 0.35 -> 0.22, and KP left at 6.0.  This is
#   deliberate and it is the one place the field request cannot be honoured as
#   asked.  ArduPilot is set to ACRO_RP_RATE=360 deg/s against ACRO_Y_RATE=202,
#   so an identical deflection is 1.8x the angular rate in pitch.  Pitch is
#   therefore the axis that goes unstable first against detection latency, and
#   every configuration in the search that raised pitch authority started
#   limit-cycling: it overshoots the target, corrects, overshoots back.  A pitch
#   axis that hunts IS the imprecision, so more pitch gain would make the
#   reported symptom worse, not better.
#
#   KD went DOWN, 0.15 -> 0.10.  The derivative term is computed from the
#   detection position, so it amplifies detection jitter; less of it is what took
#   the jitter robustness from 3/5 seeds to 5/5.
#
# If pitch needs to feel stronger, the lever is on the FC, not here: lowering
# ACRO_RP_RATE toward yaw's 202 deg/s would let pitch use much more stick travel
# for the same angular rate. That also changes how the pilot's own pitch stick
# feels, so it is a decision for the pilot, not a default to change quietly.
#
# ROLL is not driven and is not in this preset.  Roll, throttle and arming stay
# with the pilot by design; the tracker only ever touches yaw and pitch.
TUNED = {
    "KP_YAW": "4.5", "KP_PITCH": "6.0",
    "KD_YAW": "0.10", "KD_PITCH": "0.10",
    "STICK_LIMIT": "0.65",
    "STICK_LIMIT_YAW": "0.65",
    "STICK_LIMIT_PITCH": "0.22",
    "SLEW_PER_S": "11.0",
    "DEADBAND_FRAC": "0.03",
}


def resolve_fc_dev():
    """Same discovery rule as run_pipeline.py: the board's by-id string and its
    ACM number both change across boots, so never hardcode either."""
    env = os.environ.get("FC_DEV")
    if env and Path(env).exists():
        return env
    by_id = Path("/dev/serial/by-id")
    if by_id.is_dir():
        for pattern in ("usb-ArduPilot_*-if00", "*-if00", "*"):
            matches = sorted(by_id.glob(pattern))
            if matches:
                return str(matches[0])
    for dev in ("/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0"):
        if Path(dev).exists():
            return dev
    return None


def wait_for_fc_device(quiet=False):
    """Block until the FC's serial interface exists.  Returns its path."""
    dev = resolve_fc_dev()
    if dev:
        return dev
    if not quiet:
        print("[wait] no flight controller on USB yet.  Power up the drone "
              "whenever you like — waiting.  (Ctrl-C to give up.)", flush=True)
    dots = 0
    while True:
        time.sleep(1.0)
        dev = resolve_fc_dev()
        if dev:
            print(f"\n[wait] flight controller appeared: {dev}", flush=True)
            return dev
        dots += 1
        if dots % 5 == 0:
            print(f"[wait] still nothing after {dots}s ...", flush=True)


def wait_for_heartbeat(dev, tries=20):
    """The device node appears as soon as USB enumerates, but ArduPilot needs a
    moment more before it talks MAVLink.  Probe until it answers, then close the
    port — run_pipeline.py needs exclusive ownership of it.
    """
    from pymavlink import mavutil
    for attempt in range(1, tries + 1):
        m = None
        try:
            m = mavutil.mavlink_connection(dev, baud=115200)
            hb = m.wait_heartbeat(timeout=3)
            if hb is not None:
                print(f"[wait] MAVLink heartbeat OK (system {hb.get_srcSystem()})",
                      flush=True)
                return True
        except Exception as e:
            if attempt == 1:
                print(f"[wait] port not talking yet ({e.__class__.__name__}) — "
                      f"retrying", flush=True)
        finally:
            if m is not None:
                try:
                    m.close()
                except Exception:
                    pass
        if attempt % 5 == 0:
            print(f"[wait] no heartbeat after {attempt} tries — is the FC "
                  f"booted?  Still waiting.", flush=True)
        time.sleep(1.0)
    return False


def main():
    argv = sys.argv[1:]
    wait = True
    passthrough = []

    while argv:
        a = argv.pop(0)
        # --tuned is accepted and is a no-op: it is the only preset, and the
        # systemd unit passes it explicitly.  Keep it valid.
        if a == "--tuned":
            pass
        elif a == "--no-wait":
            wait = False
        elif a == "--":
            passthrough += argv
            argv = []
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            passthrough.append(a)

    preset = TUNED
    for k, v in preset.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("DISPLAY", ":0")

    print("=" * 74)
    print("WASP head tracker")
    print("=" * 74)
    env = os.environ
    print(f"  preset       : tuned  (raised 2026-08-05 after the field test)")
    print(f"  gains        : KP {env['KP_YAW']}/{env['KP_PITCH']}  "
          f"KD {env['KD_YAW']}/{env['KD_PITCH']}  slew {env['SLEW_PER_S']}/s")
    print(f"  ceilings     : yaw {env['STICK_LIMIT_YAW']}  "
          f"pitch {env['STICK_LIMIT_PITCH']}"
          f"   (pitch lower on purpose: ACRO_RP_RATE 360 vs ACRO_Y_RATE 202 deg/s)")
    print(f"  driven axes  : yaw, pitch.  Roll, throttle and arming stay with "
          f"the pilot.")
    print("  mode gate    : ON — the pilot's switch decides.  STABILIZE = the "
          "pilot has\n                 everything; ACRO = the tracker drives "
          "yaw and pitch.")
    print("  handover     : automatic, both ways, as many times as you like.")
    print("  stop         : Ctrl-C — releases every channel back to the pilot.")
    print("-" * 74)
    print("  THROTTLE: the pilot must hold MID throttle the whole time.  In ACRO")
    print("  rotation comes from differential motor thrust, so at low throttle")
    print("  even full stick barely turns the airframe.  No gain fixes that.")
    print("=" * 74, flush=True)

    if wait:
        dev = wait_for_fc_device()
        if not wait_for_heartbeat(dev):
            print("\n[FAIL] the FC serial port is there but never answered "
                  "MAVLink.\n       Check that nothing else owns it "
                  "(pkill -f mavproxy.py) and try again.", file=sys.stderr)
            return 1
        os.environ.setdefault("FC_DEV", dev)
        # Let the OS fully release the port we just probed before handing it on.
        time.sleep(0.5)

    cmd = [sys.executable, str(HERE / "run_pipeline.py"), "--direct"] + passthrough
    print(f"\n[run] {' '.join(cmd)}\n", flush=True)
    os.chdir(HERE)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("\n[abort] never connected to the FC — nothing was commanded.")
        sys.exit(130)
