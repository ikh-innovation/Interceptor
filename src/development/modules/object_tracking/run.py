#!/usr/bin/env python3
"""One-command launcher for the WASP head tracker.

Replaces the long incantation:

    cd .../object_tracking && KP_YAW=3.5 KP_PITCH=6.0 KD_YAW=0.15 KD_PITCH=0.15 \
      STICK_LIMIT=0.35 DISPLAY=:0 ~/yolo-env/bin/python run_pipeline.py --direct

with:

    ./run.py

What it does, in order:

  1. Re-execs itself under the yolo-env interpreter if it was started with a
     python that has no ultralytics (the system python3 does not).
  2. Applies the bench-tuned gains as DEFAULTS — anything already set in the
     environment wins, so `KP_YAW=6 ./run.py` still works.
  3. WAITS for the flight controller to appear and answer a heartbeat, so this
     can be started BEFORE the drone is powered up.  Ctrl-C while waiting exits
     cleanly without ever touching the FC.
  4. Hands off to run_pipeline.py --direct.

The pilot's mode switch remains the handover control (REQUIRE_RATE_MODE=1, the
default): STABILIZE = the pilot has every channel, ACRO = the tracker drives
yaw and pitch.  Starting this script never takes control on its own.

    ./run.py                 ADAPTIVE (default): snap onto the target, ease in.
                             Fastest and most robust option in simulation.
    ./run.py --tuned         linear, hardware-tested (KP 3.5/6.0, stick <= 0.35)
    ./run.py --fast          linear, strong but bounded (KP 6.0/15.0)
    ./run.py --max           full stick authority.  DIVERGES in the kinematic
                             simulator — kept only for deliberate experiments.
    ./run.py --gentle        original conservative gains, for backing right out
    ./run.py --no-wait       fail immediately if the FC is not already up
    ./run.py -- --no-verify  everything after `--` goes to run_pipeline.py

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

# ── 2. presets ────────────────────────────────────────────────────────────────
# ADAPTIVE — snap onto the target, then ease into the centre.  The default.
#
# The complaint every linear preset shares: one KP has to serve both the far
# field and the near field.  Tune it to swing hard onto a target at the frame
# edge and it twitches on a nearly-centred one; tune it to be calm at the centre
# and it crawls toward anything far away.  KP is a straight line through the
# origin, and no value of it is both steep far out and shallow close in.
#
# So this preset stops using KP and shapes the deflection directly:
#
#     |offset| >= SHAPE_SAT (42%)  ->  the axis ceiling, immediately
#     |offset| inside that         ->  (offset/0.42) ** 1.6, collapsing fast
#
# The two axes get DIFFERENT ceilings, which is the whole reason this preset can
# be fast at all.  ACRO_RP_RATE is 360 deg/s against ACRO_Y_RATE's 202, so the
# same deflection is 1.8x the angular rate in pitch, and pitch is what goes
# unstable first against detection latency.  One shared ceiling has to be set low
# enough for pitch, which needlessly throttles yaw — and yaw is the axis you
# actually watch swing onto the target.  So: yaw 60%, pitch 18%.
#
# These numbers are not guesses.  They are the best of a grid search over
# SHAPE_SAT x yaw ceiling x pitch ceiling x gamma x KD, scored on settling time
# across six starting geometries and then re-scored for robustness against worse
# latency and against 8 px of detection jitter.  Measured against the kinematic
# simulator in tests/test_control_law.py, at 0.1 s detection latency:
#
#     --adaptive      settles in 0.60 s worst case
#     --tuned         settles in 1.10 s worst case
#     --max           NEVER settles: it diverges on every geometry tried
#
# and with 8 px of detection noise at 0.15 s latency, --adaptive still settles
# (0.90 s) while --tuned goes unstable.  Faster AND more robust, not a trade.
ADAPTIVE = {
    "CONTROL_SHAPE": "adaptive",
    "SHAPE_GAMMA": "1.6",
    "SHAPE_SAT": "0.42",
    "KD_YAW": "0.18", "KD_PITCH": "0.18",
    "STICK_LIMIT": "0.60",
    "STICK_LIMIT_YAW": "0.60",
    "STICK_LIMIT_PITCH": "0.18",
    "SLEW_PER_S": "12.0",
    "DEADBAND_FRAC": "0.03",
}
# MAX — the default: the fastest yaw/pitch this airframe can be commanded to do.
#
# The ceiling is the FC's own ACRO full-stick rates, ACRO_Y_RATE=202.5 deg/s and
# ACRO_RP_RATE=360 deg/s.  STICK_LIMIT=1.0 reaches them; nothing here can exceed
# them without changing those FC params, which would also change how the pilot's
# own sticks feel, so they are left alone.
#
# The gains are set so full stick is reached at roughly a QUARTER-frame error
# (~160 px yaw, ~150 px pitch) rather than at a few pixels.  Saturating on tiny
# errors is what turns high gain into a limit-cycle oscillation: the loop slams
# full deflection, overshoots, slams the other way.  Reaching the ceiling on a
# large error keeps the response linear where the target actually sits.
#
#   yaw:   stick 1.0 at atan(1.0*3.5343/KP) -> ~161 px  with KP_YAW=12
#   pitch: stick 1.0 at atan(1.0*6.2832/KP) -> ~150 px  with KP_PITCH=30
#
# KD is raised hard because damping is what buys back stability at this gain, and
# SLEW_PER_S is raised so the limiter is no longer the thing capping the response
# (at 3.0/s, reaching full stick took a third of a second — that WAS the lag).
# DEADBAND is widened slightly: at this gain, detection jitter on a centred face
# would otherwise be amplified into visible twitching.
MAXSPD = {
    "KP_YAW": "12.0", "KP_PITCH": "30.0",
    "KD_YAW": "0.4", "KD_PITCH": "0.4",
    "STICK_LIMIT": "1.0",
    "SLEW_PER_S": "10.0",
    "DEADBAND_FRAC": "0.04",
}
# Strong but with deliberate headroom left in the stick, if MAX proves twitchy.
FAST = {
    "KP_YAW": "6.0", "KP_PITCH": "15.0",
    "KD_YAW": "0.25", "KD_PITCH": "0.25",
    "STICK_LIMIT": "0.55",
    "SLEW_PER_S": "6.0",
}
# Tested on the real FC on 2026-08-04, props off, armed in ACRO:
#   656/656 frames locked, 619 confirmed, 0 refused, 0 wrong directions,
#   3 stick reversals in 78 s (no oscillation), peak stick 0.16 yaw / 0.10 pitch.
TUNED = {
    "KP_YAW": "3.5", "KP_PITCH": "6.0",
    "KD_YAW": "0.15", "KD_PITCH": "0.15",
    "STICK_LIMIT": "0.35",
}
# The original defaults, for backing out if anything looks twitchy.
GENTLE = {
    "KP_YAW": "2.0", "KP_PITCH": "2.0",
    "KD_YAW": "0.0", "KD_PITCH": "0.0",
    "STICK_LIMIT": "0.35",
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
    # ADAPTIVE is the default because it is both the fastest and the most robust
    # option in simulation.  MAXSPD is kept reachable, but it diverges in the
    # kinematic sim and must not be the thing that runs by accident.
    preset, wait = ADAPTIVE, True
    preset_name = "ADAPTIVE (snap on, ease in)"
    passthrough = []

    while argv:
        a = argv.pop(0)
        if a == "--adaptive":
            preset, preset_name = ADAPTIVE, "ADAPTIVE (snap on, ease in)"
        elif a in ("--max", "--maxspeed"):
            preset, preset_name = MAXSPD, "MAX SPEED"
        elif a == "--fast":
            preset, preset_name = FAST, "fast"
        elif a == "--tuned":
            preset, preset_name = TUNED, "tuned (hardware-tested 2026-08-04)"
        elif a == "--gentle":
            preset, preset_name = GENTLE, "gentle"
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

    for k, v in preset.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("DISPLAY", ":0")

    print("=" * 74)
    print("WASP head tracker")
    print("=" * 74)
    env = os.environ
    print(f"  preset       : {preset_name}")
    if env.get("CONTROL_SHAPE", "linear").lower() == "adaptive":
        # KP is deliberately absent in this mode, so never read it here.
        print(f"  law          : adaptive  ceiling at "
              f"{float(env['SHAPE_SAT']) * 100:.0f}% off-centre, "
              f"gamma {env['SHAPE_GAMMA']}")
        print(f"  ceilings     : yaw {env.get('STICK_LIMIT_YAW', env['STICK_LIMIT'])}"
              f"  pitch {env.get('STICK_LIMIT_PITCH', env['STICK_LIMIT'])}"
              f"   (pitch is lower on purpose: ACRO_RP_RATE 360 vs yaw 202 deg/s)")
        print(f"  gains        : KD {env['KD_YAW']}/{env['KD_PITCH']}  "
              f"slew {env['SLEW_PER_S']}/s")
    else:
        print(f"  gains        : KP {env.get('KP_YAW', '2.0')}/"
              f"{env.get('KP_PITCH', '2.0')}  "
              f"KD {env.get('KD_YAW', '0.0')}/{env.get('KD_PITCH', '0.0')}  "
              f"stick<={env.get('STICK_LIMIT', '0.55')}  "
              f"slew {env.get('SLEW_PER_S', '3.0')}/s")
    print("  mode gate    : ON — the pilot's switch decides.  STABILIZE = the "
          "pilot has\n                 everything; ACRO = the tracker drives "
          "yaw and pitch.")
    print("  handover     : automatic, both ways, as many times as you like.")
    print("  stop         : Ctrl-C — releases every channel back to the pilot.")
    if preset is MAXSPD:
        print("-" * 74)
        print("  MAX SPEED: full stick authority.  At the ceiling this commands")
        print("  202 deg/s yaw and 360 deg/s pitch — the FC's own ACRO_Y_RATE /")
        print("  ACRO_RP_RATE.  That is ~6x the deflection the last hardware run")
        print("  actually used, so OSCILLATION is now the failure mode to watch:")
        print("  if it shakes or hunts around your face, Ctrl-C and use --fast.")
        print("  PROPS OFF for the first run at this preset.")
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
