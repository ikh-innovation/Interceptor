#!/usr/bin/env python3
"""One-command launcher for the WASP head tracker.

The single entry point.  Everything the tracker needs — interpreter, gains, FC
discovery, firmware identification, transport, display — is decided here so that
what flies is never a function of what somebody remembered to type.

What it does, in order:

  1. Re-execs itself under the yolo-env interpreter if it was started with a
     python that has no ultralytics (the system python3 does not).
  2. Applies the tuned gains as DEFAULTS — anything already set in the
     environment wins, so `KP_YAW=6 ./run.py` still works.
  3. WAITS for the flight controller to appear and answer a heartbeat, so this
     can be started BEFORE the drone is powered up.  Ctrl-C while waiting exits
     cleanly without ever touching the FC.
  4. Identifies the flight controller's FIRMWARE, on its own serial port, BEFORE
     anything claims the port.  ArduPilot and Betaflight need completely
     different plumbing, so this decides everything below.
       ArduPilot  ->  MAVLink.  Direct on the FC serial by default; --router puts
                      MAVProxy in the path (UDP 14550 for Mission Planner, 14551
                      for the tracker).  Then verify a real RC override is
                      accepted by the FC.
       Betaflight ->  NO router (it cannot speak MSP) and no Mission Planner;
                      the tracker owns the serial port and talks MSP directly.
  5. Runs tracker.py in the foreground.

Preflight stops on the first failure and names it, instead of letting a run look
healthy while nothing reaches the FC — which is exactly how the previous version
could stream setpoints into a void for a whole demo.

TWO PRESETS, because there are two different control problems:

  --intercept  (DEFAULT)  free flight.  Proportional navigation on ALL FOUR
               channels in ALT_HOLD: yaw aims the seeker, ROLL banks to turn the
               velocity vector, pitch sets closing speed, THROTTLE closes the
               vertical.  This is the one that chases a person running across a
               field.
  --gimbal     the bring-up rig.  yaw+pitch body rates in ACRO, nulling pixel
               error on an airframe bolted to a gimbal that cannot translate.

They are not retunes of each other.  On a gimbal, pointing the camera IS the job.
In free flight, pointing the camera and moving the airframe are the same two
actuators, so aiming and intercepting compete — which is why the gimbal preset
tracked beautifully and then flew backwards in the air.

The pilot's mode switch remains the handover control (REQUIRE_RATE_MODE=1, the
default): STABILIZE = the pilot has every channel, ALT_HOLD (or ACRO on the
gimbal preset) = the tracker drives.  Starting this script never takes control on
its own.

    ./run.py                 the intercept preset
    ./run.py --gimbal        the gimbal preset
    ./run.py --tuned         no-op; accepted so the systemd unit stays valid
    ./run.py --no-wait       fail immediately if the FC is not already up
    ./run.py --router        put MAVProxy in the path instead of going direct
    ./run.py --no-mp         skip Mission Planner (implied by the default direct)
    ./run.py --no-verify     skip the RC-override acceptance check
    ./run.py --stream-port N move the remote seeker view off port 8080
    ./run.py --no-stream     do not serve the seeker view over WiFi

THE SEEKER VIEW.  The Jetson has no screen in the field, so the annotated
detection frame — boxes, HUD, stick command, the lot — is served as MJPEG over
HTTP and watched from a phone or laptop on the same WiFi:

    http://<jetson-ip>:8080/          the live view (any browser, no app)
    http://<jetson-ip>:8080/snapshot.jpg   one frame, for a bug report

It is ON by default; the banner prints the exact address before the FC wait, so
the page can already be open when frames start.  It is a HUD and nothing else —
there is no control on that page, and the pilot's transmitter remains the only
abort.  Nothing is JPEG-encoded while nobody is watching, so leaving it on costs
the flight computer nothing.

Individual values still override, so a one-off experiment needs no new flag:

    ROLL_PN_GAIN=40 CAM_TILT_DEG=20 ./run.py
    COMMIT_CHAN=6 ./run.py          # live speed dial AND a ~100 ms abort

Environment overrides: FC_DEV, FC_BAUD, CAM_DEV, plus every control knob in
fc_control.commander_from_env (KP_YAW, STICK_LIMIT, YAW_REVERSED, ...).

The pilot must hold MID THROTTLE throughout.  In ACRO the attitude controller's
authority scales with throttle: at idle the mixer has no headroom to differentiate
the motors, so even a full-stick yaw command produces almost no torque.  No gain
setting in here can compensate for a closed throttle.
"""
import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# fc_check.py lives with the tests because that is what it is — a bench
# diagnostic.  Preflight reuses its probes rather than owning a second copy of
# the firmware-identification logic.
sys.path.insert(0, str(HERE / "tests"))

HOME       = Path.home()
YOLO_PY    = HOME / "yolo-env/bin/python"
TRACKER    = HERE / "tracker.py"
MP_EXE     = HOME / "MissionPlanner/MissionPlanner.exe"
MAVPROXY   = HOME / ".local/bin/mavproxy.py"
FT_DIR     = HOME / "face_tracker"                     # logs + mavproxy state
LOG_DIR    = FT_DIR / "logs"
STATE_DIR  = FT_DIR / "mavproxy-state"

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

# ── 2. the presets ────────────────────────────────────────────────────────────
# There are exactly TWO, and they are not variants of each other — they are two
# different control problems (see the module docstring).  ADAPTIVE / MAX / FAST /
# GENTLE were removed after the field flights: extra knobs meant the thing that
# actually flew was whichever flag somebody last typed, which is the opposite of
# what a flight-control default is for.  Nothing below reintroduces a dial that
# changes what flies without changing what problem is being solved.
#
# ── the gimbal preset ─────────────────────────────────────────────────────────
# For the bring-up rig: an airframe bolted to a gimbal, which can rotate but not
# translate.  Pointing the camera IS the whole job there, and this does it well.
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
#
# PITCH_MODE=approach — the fix for "it flies backwards when I am above it".
#
# Off the gimbal, a multirotor's pitch attitude IS its direction of travel.  So
# using pitch to aim the camera vertically and expecting forward flight are
# opposite commands on one axis: with the target above the nose, aiming up meant
# leaning back, and leaning back means flying away.  That is not a gain problem
# and no tuning fixes it.
#
# In approach mode pitch stops looking at vertical pixel error entirely and instead
# holds a nose-DOWN lean, scaled by how far the pilot has the throttle above mid.
# Full throttle = maximum lean = come at me; throttle back to mid = level = stop.
# The lean is held by closing a loop on the FC's own measured ATTITUDE (requested
# at 20 Hz at connect — ArduPilot sends nothing unasked), because in ACRO a stick
# is a rate and an open-loop nose-down command would just keep rotating.
#
# The target's vertical position in frame no longer affects pitch at all, so:
# above, below, left or right, the airframe yaws onto the target and flies FORWARD.
# Vertical framing becomes altitude, which is the pilot's throttle — and the lens
# has 48.8 deg of vertical view (+/-24 deg) of tolerance for it.
GIMBAL = {
    "CONTROL_LAW": "servo",
    "CONTROL_AXES": "yaw,pitch",
    "FC_STEER_MODES": "ACRO",
    "KP_YAW": "4.5", "KP_PITCH": "6.0",
    "KD_YAW": "0.10", "KD_PITCH": "0.10",
    "STICK_LIMIT": "0.65",
    "STICK_LIMIT_YAW": "0.65",
    "STICK_LIMIT_PITCH": "0.22",
    "SLEW_PER_S": "11.0",
    "DEADBAND_FRAC": "0.03",
    "PITCH_MODE": "approach",
    "APPROACH_DEG": "12.0",
    "APPROACH_KP": "2.5",
    "APPROACH_THR_DEADBAND": "0.05",
}

# ── the free-flight preset ────────────────────────────────────────────────────
# INTERCEPT.  This is the one that chases a person running across a field, and it
# is a different control problem from the gimbal, not a retune of it.
#
# The gimbal preset above points the camera. That IS the whole job when the
# airframe cannot translate, and it worked. In free flight it does not intercept,
# for reasons no gain value reaches:
#
#   * Pure pursuit — turn rate proportional to the CURRENT bearing error — lags a
#     crossing target. Chasing someone cutting left and right it settles into a
#     trailing tail-chase and never closes. The fix is PROPORTIONAL NAVIGATION:
#     lateral acceleration proportional to the RATE the line of sight rotates,
#     which holds the target at a constant bearing = a collision course.
#   * Yaw rotates the camera; it does not move the airframe. A multirotor turns
#     its VELOCITY vector by BANKING, and roll was never driven at all.
#   * Nothing controlled the vertical. `aim` used pitch and flew backwards;
#     `approach` correctly stopped using pitch and put nothing in its place. The
#     only channel that changes altitude is THROTTLE.
#
# So all four channels are driven, each with one job:
#
#     yaw       keep the seeker on the target   (pursuit on azimuth)
#     roll      lateral acceleration            (PN on the azimuth LOS rate)
#     pitch     closing speed                   (nose-down lean x commit)
#     throttle  vertical closure                (elevation error + LOS-rate lead)
#
# ALT_HOLD, not ACRO. In ALT_HOLD a roll/pitch stick IS a lean angle and the
# throttle stick IS a climb rate, so the outer loop that ACRO needed just to hold
# a lean disappears — and a centred throttle means HOLD ALTITUDE, which is what
# makes it safe for this code to hold the throttle at all. It needs only the
# barometer; there is no GPS on this airframe.
#
# ROLL_PN_GAIN is the knob for "it does not follow me when I cut left and right".
# It is dimensionally an own-speed in m/s: set it to the true airspeed for a
# coordinated turn, 2-3x for PN's lead. Up if it trails, down if it weaves.
#
# CLIMB_GAIN / CLIMB_LEAD are the knob for "it does not follow me when I am
# higher or lower than it".
#
# SEND_HZ is 50, not 20: at 20 Hz the stick is quantised to 50 ms, which is 10 deg
# of attitude at 200 deg/s.
#
# CAM_TILT_DEG, MAX_LEAN_DEG and ROLL_PN_GAIN are DERIVED, see below — they are not
# three independent knobs and setting them independently is how you get a run that
# looks tuned and cannot see the target.
INTERCEPT = {
    "CONTROL_LAW": "intercept",
    "CONTROL_AXES": "roll,pitch,yaw,throttle",
    "FC_STEER_MODES": "ALT_HOLD",
    "KP_YAW": "4.5",
    # The pair below move TOGETHER: gain = 3 x cruise speed, and cruise speed
    # follows the lean.  20 deg / 45 is what the closed-loop sim was scored at and
    # the best of everything tried (100% of frames on target, and it closes).
    # `pn_gain_for()` prints a warning if you change one and not the other.
    "MAX_LEAN_DEG": "20.0",
    "ROLL_PN_GAIN": "45.0",
    "MAX_BANK_DEG": "30.0",
    # Vertical gain left at the scored value on purpose: raising CLIMB_GAIN or
    # PILOT_SPEED_UP changed NOTHING measurable in any geometry tried (the height
    # error is absorbed long before the endgame), so there is nothing to buy here.
    # Raise them only if the target genuinely outruns you vertically.
    "CLIMB_GAIN": "8.0",
    "CLIMB_LEAD": "3.0",
    "STICK_LIMIT": "0.90",
    "SLEW_PER_S": "11.0",
    "DEADBAND_FRAC": "0.02",
    "LOS_TAU": "0.12",
    "TARGET_WIDTH_M": "0.20",
    "TERMINAL_BBOX_FRAC": "0.35",
    "TERMINAL_RANGE_M": "2.5",
    "TERMINAL_HOLD_S": "1.0",
    "SEND_HZ": "50.0",
}

# The remote seeker view, ON by default and applied to BOTH presets — it is a
# property of the airframe (a Jetson with no screen), not of the control law.
#
# Declared here rather than left to tracker.py's own fallbacks so the default
# lives in the same file as every other decision, and so `--no-stream` /
# STREAM=0 has exactly one thing to switch off.  tracker.py keeps matching
# fallbacks for when it is run directly, without this launcher.
#
# It is safe to leave on with no network at all: the server binds on loopback,
# the address probe fails closed, and nothing is JPEG-encoded until a viewer
# actually connects.  See mjpeg_stream.py.
SEEKER_VIEW = {
    "STREAM": "1",
    "STREAM_PORT": "8080",
    "STREAM_QUALITY": "70",     # drop to ~50 on a weak link
    "STREAM_FPS": "15",         # encode cap, independent of capture FPS
}

# Lens half-height, and how much of it to keep in reserve for pitch overshoot,
# detection jitter and a target that is not exactly level.
_LENS_HALF_V_DEG = 24.4
_FRAME_MARGIN_DEG = 8.0
# Ceiling on the lean regardless of what the optics would allow: 20 deg is what the
# closed-loop sim was scored at, and past ~26 deg the seeker starts losing the
# target for other reasons.
_LEAN_CEILING_DEG = 20.0
# Drag coefficient used ONLY to estimate terminal speed from the lean, so the turn
# gain can be scaled to it. It is the sim's value for a ~15 m/s airframe, i.e. an
# assumption, not a measurement. Measure the real cruise speed and set
# ROLL_PN_GAIN = 3 x it explicitly when you know it.
_DRAG_EST = 0.0159
_PN_N = 3.0                    # proportional navigation constant, classical 3..5


def geometry_check(tilt_deg, lean_deg):
    """Can the seeker actually SEE a level target at this lean and this tilt?

    Returns (ok, band_lo, band_hi, advice).

    These are not independent knobs, and treating them as such is the most
    expensive mistake available here:

      * At nose-down lean `theta` a target at the airframe's OWN altitude appears
        at `+theta` in a body-fixed camera. The lens is only +/-24.4 deg vertically,
        so a LEVEL-mounted camera loses the target out of the TOP of the frame as
        soon as it flies fast. Measured in the closed-loop sim: with no tilt, the
        target is in frame at most 48% of the time and it NEVER closes, at any lean
        from 8 to 20 deg. A level camera is a bench configuration, not a flying one.
      * Tilting the camera UP buys that back, but spends DOWNWARD vision. At tilt 15
        with the airframe launched 6 m above the target it is in frame 13% of the
        time — effectively blind. Tilt 10 was the best of everything tried.
      * The turn gain is an own-speed (`gain = N x airspeed`) and airspeed follows
        the lean, so the two move together. Raise the lean alone and it trails.
    """
    lo = -_LENS_HALF_V_DEG + tilt_deg - lean_deg
    hi = _LENS_HALF_V_DEG + tilt_deg - lean_deg
    # A level target sits at 0 deg about the horizon. Keep margin above it for
    # pitch overshoot, detection jitter and a target that is not exactly level.
    ok = hi >= _FRAME_MARGIN_DEG
    advice = ""
    if not ok:
        need = math.ceil(lean_deg + _FRAME_MARGIN_DEG - _LENS_HALF_V_DEG)
        advice = (f"tilt the camera up to at least {max(0, need)} deg, "
                  f"or drop MAX_LEAN_DEG to "
                  f"{max(8.0, _LENS_HALF_V_DEG - _FRAME_MARGIN_DEG + tilt_deg):.0f}")
    return ok, lo, hi, advice


def pn_gain_for(lean_deg):
    """The turn gain that matches a given lean, i.e. `N x estimated airspeed`.

    Only an estimate: the drag figure is the sim's, for a ~15 m/s airframe. Measure
    the real cruise speed and set ROLL_PN_GAIN = 3 x it once you know it. Printed
    as advice rather than applied, because a derived number that has not been flown
    should not quietly become the default.
    """
    v = math.sqrt(9.80665 * math.tan(math.radians(lean_deg)) / _DRAG_EST)
    return _PN_N * v


def resolve_fc_dev():
    """Locate the flight controller's serial interface.

    The board's by-id serial string is NOT stable across boots (observed
    changing), and its raw ACM number re-enumerates too — so a hardcoded path
    breaks after a reboot.  Prefer the ArduPilot vendor prefix on the MAVLink
    interface (``-if00``), but fall back through any by-id entry, because on
    Betaflight the descriptor is a different string entirely and this launcher
    must still find the board in order to identify it.  An explicit FC_DEV that
    actually exists always wins."""
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


FC_DEV   = resolve_fc_dev() or "/dev/ttyACM0"
FC_BAUD  = os.environ.get("FC_BAUD", "115200")        # nominal; USB CDC-ACM ignores it
CAM_DEV  = os.environ.get("CAM_DEV", "/dev/video0")

UDP_MP      = "127.0.0.1:14550"                        # Mission Planner listens here
UDP_TRACKER = "127.0.0.1:14551"                        # tracker.py binds udpin here


def info(m): print(f"  {m}", flush=True)
def ok(m):   print(f"  [ok] {m}", flush=True)
def warn(m): print(f"  [warn] {m}", flush=True)


def die(msg):
    print(f"\n[PREFLIGHT FAIL] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# WAITING FOR THE BOARD  (so this can start before the drone is powered)
# ══════════════════════════════════════════════════════════════════════════════

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
    port — preflight needs exclusive ownership of it.
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


def router_running() -> bool:
    """True if a MAVProxy router is already up (reuse it, don't manage it)."""
    return subprocess.run(["pgrep", "-f", "mavproxy.py"],
                          stdout=subprocess.DEVNULL).returncode == 0


def stop_router_or_die():
    """The firmware probe and the MSP transport both need exclusive access to
    the FC serial port, which a running router holds."""
    if router_running():
        die("a MAVProxy router is already running and owns the FC serial port.\n"
            "                 Stop it first:  pkill -f mavproxy.py")


# ══════════════════════════════════════════════════════════════════════════════
# FIRMWARE IDENTIFICATION  (before anything claims the port)
# ══════════════════════════════════════════════════════════════════════════════

def identify_firmware(dev, baud):
    """Returns ("ardupilot"|"betaflight"|None, detail).

    Probes MAVLink first, then MSP.  This is the question everything else hangs
    on: MAVLink RC overrides cannot drive Betaflight, and MSP cannot drive
    ArduPilot.
    """
    from fc_check import probe_mavlink, probe_msp

    master, hb = probe_mavlink(dev, int(baud), timeout=8.0)
    if master is not None:
        mode = hb.custom_mode
        armed = bool(hb.base_mode & 0x80)
        master.close()                       # release the port for the router
        time.sleep(0.4)
        return "ardupilot", dict(custom_mode=mode, armed=armed)

    msp = probe_msp(dev, int(baud))
    if msp is not None:
        return "betaflight", msp
    return None, None


def verify_overrides(dev, baud, conn):
    """Send a real RC override and prove the FC applies it.  Runs against
    whatever link the tracker will use, so it tests the ACTUAL path including
    the router.  Skipped when the FC is armed, because it would command real
    rotation."""
    from fc_check import (probe_mavlink, read_param, request_rc_stream,
                          verify_override)

    master, hb = probe_mavlink(conn, int(baud), timeout=10.0)
    if master is None:
        return None
    armed = bool(hb.base_mode & 0x80)
    if armed:
        master.close()
        warn("FC is ARMED — skipping the override check (it would command real "
             "rotation).  Disarm and re-run to verify the transport.")
        return None

    # Minimal config for the check: the yaw channel and the GCS sysid.
    sysid = read_param(master, "SYSID_MYGCS")
    if sysid is None:
        sysid = read_param(master, "MAV_GCS_SYSID")
    ch = read_param(master, "RCMAP_YAW")
    ch = int(ch) if ch else 4
    axes = {"yaw": dict(chan=ch,
                        min=int(read_param(master, f"RC{ch}_MIN") or 1000),
                        max=int(read_param(master, f"RC{ch}_MAX") or 2000),
                        trim=int(read_param(master, f"RC{ch}_TRIM") or 1500),
                        reversed=False)}
    request_rc_stream(master)
    result = verify_override(master, dict(axes=axes, sysid_mygcs=sysid))
    master.close()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER / MISSION PLANNER
# ══════════════════════════════════════════════════════════════════════════════

def wait_router_heartbeat(timeout=10) -> bool:
    """Confirm the router is passing FC traffic by briefly binding tracker's port
    (udpin 14551) and waiting for an AUTOPILOT heartbeat, then releasing it.

    Filters out GCS heartbeats: MAVProxy and Mission Planner also emit them, so
    accepting any heartbeat would declare success while the router was not
    actually reaching the board."""
    from pymavlink import mavutil
    m = mavutil.mavlink_connection(f"udpin:{UDP_TRACKER}")
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if hb is None:
                continue
            if (hb.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID
                    and hb.type != mavutil.mavlink.MAV_TYPE_GCS):
                return True
        return False
    finally:
        m.close()


def start_router() -> subprocess.Popen:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_DIR / "router.log", "w", buffering=1)
    argv = [
        str(MAVPROXY),
        f"--master={FC_DEV},{FC_BAUD}",
        f"--out=udpout:{UDP_MP}",
        f"--out=udpout:{UDP_TRACKER}",
        "--daemon",
        f"--state-basedir={STATE_DIR}",
    ]
    info(f"launching router: {' '.join(argv)}")
    info(f"router log: {LOG_DIR / 'router.log'}")
    return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)


def start_mission_planner():
    if not MP_EXE.exists():
        warn(f"Mission Planner not found at {MP_EXE} — skipping")
        return
    if subprocess.run(["pgrep", "-f", "MissionPlanner.exe"],
                      stdout=subprocess.DEVNULL).returncode == 0:
        ok("Mission Planner already running")
        return
    info("launching Mission Planner (mono) — connect it to UDP 14550")
    env = dict(os.environ, DISPLAY=os.environ.get("DISPLAY", ":0"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "missionplanner.log", "w") as log:
        subprocess.Popen(["mono", str(MP_EXE)], stdout=log, stderr=subprocess.STDOUT,
                         env=env)
    time.sleep(2)


# ══════════════════════════════════════════════════════════════════════════════

def banner(intercept):
    env = os.environ
    print("=" * 74)
    print("WASP head tracker — " + ("INTERCEPT (free flight)" if intercept
                                    else "GIMBAL (visual servo)"))
    print("=" * 74)
    if intercept:
        print("  law          : proportional navigation, four channels, in ALT_HOLD")
        print(f"  yaw          : seeker, KP {env['KP_YAW']}  "
              f"(points the camera; does NOT move the airframe)")
        print(f"  roll         : PN gain {env['ROLL_PN_GAIN']} m/s, bank <= "
              f"{env['MAX_BANK_DEG']} deg   <-- the left/right chase")
        print(f"  pitch        : lean <= {env['MAX_LEAN_DEG']} deg nose-down "
              f"= closing speed")
        print(f"  throttle     : climb {env['CLIMB_GAIN']} m/s/rad "
              f"+{env['CLIMB_LEAD']} lead   <-- the higher/lower fix")
        tilt = float(env["CAM_TILT_DEG"])
        lean = float(env["MAX_LEAN_DEG"])
        ok, lo, hi, advice = geometry_check(tilt, lean)
        print(f"  camera tilt  : {tilt:.1f} deg UP  <-- must match the REAL mount")
        print(f"  visible band : {lo:+.0f} .. {hi:+.0f} deg about the horizon at "
              f"{lean:.0f} deg of lean;  "
              f"{-_LENS_HALF_V_DEG + tilt:+.0f} .. {_LENS_HALF_V_DEG + tilt:+.0f} "
              f"when level")
        print(f"                 a level target sits at 0 deg — it must be INSIDE "
              f"the band to be seen")
        want = pn_gain_for(lean)
        have = float(env["ROLL_PN_GAIN"])
        if abs(have - want) > 0.20 * want:
            print(f"  ! PN gain {have:.0f} does not match a {lean:.0f} deg lean "
                  f"(~{want:.0f} = 3 x the estimated {want / _PN_N:.0f} m/s).")
            print(f"  ! Too low trails a crossing target, too high weaves. "
                  f"gain = 3 x cruise speed.")
        if not ok:
            print(f"  !! THE TARGET WILL LEAVE THE TOP OF THE FRAME at this lean: "
                  f"only {hi:+.0f} deg")
            print(f"  !! of headroom above a level target. {advice}.")
            print("  !! With no tilt at all it was measured in frame <=48% of "
                  "the time, never closing.")
        print(f"  stream       : {env['SEND_HZ']} Hz   latency: measured live, "
              f"target extrapolated forward")
        print("  mode gate    : ON — STABILIZE = the pilot has everything; "
              "ALT_HOLD = AI drives\n                 all four sticks.")
        commit = env.get("COMMIT_CHAN", "0")
        if commit and commit != "0":
            print(f"  commit       : ch{commit} scales the lean and is the fast "
                  f"abort (~100 ms)")
        else:
            print("  commit       : NONE.  Lean is fixed while a target is "
                  "locked, and the mode\n                 switch is the only "
                  "abort (~1 s, HEARTBEAT rate).  Set COMMIT_CHAN.")
        # No engagement timer by default: the pilot is the abort.  A timer that
        # drops the sticks mid-engagement is its own surprise, and an experienced
        # pilot watching the airframe stops it sooner and better than a constant.
        lim = env.get("ENGAGE_MAX_S", "0")
        print(f"  engage limit : {lim} s per engagement" if float(lim or 0) > 0
              else "  engage limit : none — the pilot stops it")
        print("  stop         : Ctrl-C — releases every channel back to the pilot.")
        print("-" * 74)
        print("  NO GPS.  ALT_HOLD holds altitude on the BAROMETER only: it will")
        print("  drift with the wind and has no idea where it is.  The terminal")
        print("  phase FREEZES guidance and flies the trajectory it already had.")
    else:
        print(f"  preset       : gimbal  (raised 2026-08-05 after the field test)")
        print(f"  gains        : KP {env['KP_YAW']}/{env['KP_PITCH']}  "
              f"KD {env['KD_YAW']}/{env['KD_PITCH']}  slew {env['SLEW_PER_S']}/s")
        print(f"  ceilings     : yaw {env['STICK_LIMIT_YAW']}  "
              f"pitch {env['STICK_LIMIT_PITCH']}"
              f"   (pitch lower on purpose: ACRO_RP_RATE 360 vs ACRO_Y_RATE 202)")
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
    _banner_seeker_view(env)
    print("=" * 74, flush=True)


def _banner_seeker_view(env):
    """Print where to point a phone to watch what the seeker sees.

    Printed in the banner, before the FC wait, so the operator can already have
    the page open on the other device by the time frames start.
    """
    # Same truthiness test as tracker.env_bool, so the banner can never claim a
    # stream the tracker is not going to serve.
    if env.get("STREAM", "1").strip().lower() not in ("1", "true", "yes", "on"):
        print("-" * 74)
        print("  seeker view  : OFF (STREAM=%s).  Nothing to watch remotely."
              % env.get("STREAM", ""))
        return
    port = env.get("STREAM_PORT", "8080")
    try:
        # Deferred: this pulls in cv2, which only exists under the yolo-env
        # interpreter we re-exec into.  The launcher must not need it to
        # print a banner.
        from mjpeg_stream import local_ips
        addrs = local_ips()
    except Exception:
        addrs = []
    print("-" * 74)
    if addrs:
        for ip in addrs:
            print(f"  seeker view  : http://{ip}:{port}/   <-- open on the "
                  f"phone / laptop")
    else:
        print(f"  seeker view  : http://<jetson-ip>:{port}/   "
              f"(`hostname -I` for the address)")
    print("                 same WiFi as the Jetson.  It is the live HUD, not a "
          "control\n                 surface — the pilot's transmitter is still "
          "the only abort.")


def main():
    global FC_DEV

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --tuned is accepted and is a no-op: it is the only preset, and the systemd
    # unit passes it explicitly.  Keep it valid.
    ap.add_argument("--tuned", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--gimbal", action="store_true",
                    help="the gimbal preset: yaw+pitch body rates in ACRO, for a "
                         "rig that cannot translate.  The default is --intercept.")
    ap.add_argument("--intercept", action="store_true",
                    help="the free-flight preset (DEFAULT): proportional "
                         "navigation on all four channels in ALT_HOLD.")
    ap.add_argument("--no-wait", action="store_true",
                    help="fail immediately if the FC is not already up")
    ap.add_argument("--router", action="store_true",
                    help="put a MAVProxy router in the path.  The default is "
                         "direct: the tracker owns the FC serial port, which "
                         "removes the router as a failure mode, but then Mission "
                         "Planner cannot share the port.")
    ap.add_argument("--no-mp", action="store_true",
                    help="skip launching Mission Planner (only relevant with "
                         "--router)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the RC-override acceptance check")
    ap.add_argument("--stream-port", type=int, metavar="N",
                    help="port for the remote seeker view (default 8080).  The "
                         "annotated detection frame is served as MJPEG over "
                         "HTTP; open http://<jetson-ip>:N/ in a browser on any "
                         "device on the same WiFi.")
    ap.add_argument("--no-stream", action="store_true",
                    help="do not serve the seeker view over WiFi at all")
    # `./run.py -- --no-verify` used to be the way to pass flags through to the
    # old second stage.  There is no second stage now, so just drop the marker.
    args = ap.parse_args([a for a in sys.argv[1:] if a != "--"])

    direct = not args.router

    # Intercept is the default: the gimbal rig was the bring-up case, free flight
    # is the job.  An explicit CONTROL_LAW in the environment still wins, so a
    # one-off experiment needs no flag.
    intercept = not args.gimbal
    if os.environ.get("CONTROL_LAW", "").strip().lower() == "servo":
        intercept = False
    elif os.environ.get("CONTROL_LAW", "").strip().lower() == "intercept":
        intercept = True

    for k, v in (INTERCEPT if intercept else GIMBAL).items():
        os.environ.setdefault(k, v)

    # Not preset-specific: the seeker view is wanted on the gimbal rig too.
    for k, v in SEEKER_VIEW.items():
        os.environ.setdefault(k, v)

    # HARDCODED at the operator's instruction during the 2026-08-26 bring-up: the
    # mount is built at 10 deg and this is the ONE place that number lives, so the
    # launcher, the systemd unit and a bare `./run.py` can no longer disagree about
    # it.  IF THE MOUNT IS EVER CHANGED, CHANGE IT HERE.
    #
    # This block used to refuse to start, and the reasoning behind that refusal has
    # NOT gone away: the tilt is a HARDWARE FACT, the law adds it to every elevation
    # it computes, and a value that disagrees with the real mount makes the airframe
    # fly high or low by exactly that error, consistently, with nothing in the log
    # to say why.  So the number is still never applied SILENTLY — it is announced
    # loudly below, every run, as an assumption rather than a measurement.
    HARDCODED_CAM_TILT_DEG = "10"

    if intercept and "CAM_TILT_DEG" not in os.environ:
        os.environ["CAM_TILT_DEG"] = HARDCODED_CAM_TILT_DEG
        print(f"""
[SETUP] CAM_TILT_DEG was not set — using the HARDCODED {HARDCODED_CAM_TILT_DEG} deg.

  This is an ASSUMPTION about the hardware, not a measurement.  Confirm it:
  put the drone on the floor, stand 10 m in front of it, and check that your
  HEAD falls in the CENTRE of the image.  If it does not, measure the real
  angle and either export CAM_TILT_DEG=<measured> or edit
  HARDCODED_CAM_TILT_DEG in run.py.

  Overriding still works and still wins:  CAM_TILT_DEG=<measured> ./run.py

  What was measured, for the record:

    tilt  0   the target leaves the TOP of the frame as soon as it flies fast.
              In the closed-loop sim it is in frame at most 48% of the time and
              NEVER closes, at any lean.  Bench configuration only.
    tilt 10   best of everything tried: in frame 100% of the time on the flat
              chase, and it closes.   <-- the hardcoded value
    tilt 15   as good on the easy cases, but launched 6 m above the target it is
              in frame 13% of the time — effectively blind.  Tilting further
              spends downward vision.

  For the bench polarity check (props off), 0 is the honest value:

      CAM_TILT_DEG=0 REQUIRE_ARMED=0 ./run.py --no-wait
""", file=sys.stderr)

    # The seeker view over WiFi.  The flags are a friendlier face on the same
    # STREAM / STREAM_PORT the tracker reads; unlike the gain defaults these
    # are set, not setdefault, because a flag the operator typed on the field
    # must beat whatever the systemd unit put in the environment.
    if args.no_stream:
        os.environ["STREAM"] = "0"
    if args.stream_port:
        os.environ["STREAM_PORT"] = str(args.stream_port)

    os.environ.setdefault("DISPLAY", ":0")
    banner(intercept)

    if not args.no_wait:
        dev = wait_for_fc_device()
        if not wait_for_heartbeat(dev):
            print("\n[FAIL] the FC serial port is there but never answered "
                  "MAVLink.\n       Check that nothing else owns it "
                  "(pkill -f mavproxy.py) and try again.", file=sys.stderr)
            return 1
        os.environ.setdefault("FC_DEV", dev)
        FC_DEV = dev
        # Let the OS fully release the port we just probed before handing it on.
        time.sleep(0.5)

    print("\n== Head-tracker preflight ==", flush=True)
    if not Path(FC_DEV).exists():
        die(f"FC serial '{FC_DEV}' not found — is the flight controller plugged in?")
    ok(f"FC serial: {FC_DEV}")
    if not Path(CAM_DEV).exists():
        die(f"camera '{CAM_DEV}' not found — check CSI ribbon / USB cam")
    ok(f"camera:    {CAM_DEV}")
    if not YOLO_PY.exists():
        die(f"yolo-env python missing: {YOLO_PY}")
    if not TRACKER.exists():
        die(f"tracker.py missing: {TRACKER}")
    ok(f"tracker:   {TRACKER}")

    # ── 1) which firmware? ────────────────────────────────────────────────────
    print("\n== 1) flight controller identification ==", flush=True)
    stop_router_or_die()
    fw, detail = identify_firmware(FC_DEV, FC_BAUD)

    if fw is None:
        die(f"nothing answered on {FC_DEV} — neither MAVLink nor MSP.\n"
            "                 Check power/cable, that nothing else holds the "
            "port, and group\n"
            "                 membership (sudo usermod -aG dialout $USER, then "
            "re-login).\n"
            "                 Full report:  python3 tests/fc_check.py "
            f"--connect {FC_DEV}")

    env = dict(os.environ)
    router = None                      # the router process WE started (else None)

    try:
        if fw == "betaflight":
            # Betaflight cannot be driven over MAVLink at all, and MSP needs the
            # serial port exclusively — so no router and no Mission Planner.
            ok(f"firmware: {detail['variant']} {detail['version']} (MSP)")
            print("\n== 2) transport: MSP direct (no router, no Mission Planner) ==",
                  flush=True)
            warn("This board is NOT running ArduPilot.  The tracker will drive it "
                 "over MSP,")
            warn("but Betaflight ignores MSP sticks until you configure the "
                 "override:")
            print("""
        feature MSP_OVERRIDE
        set msp_override_channels_mask = 10       # ch2 pitch + ch4 yaw
        save

    then assign "MSP RC OVERRIDE" to a spare aux switch in the Modes tab and
    turn it ON.  Run  python3 tests/fc_check.py  for the full report.

    Strongly recommended instead: flash ArduPilot Copter (target
    TBS_LUCID_H7).  That gives parameter access, telemetry, Mission Planner
    and the RC_CHANNELS_OVERRIDE path this pipeline is built around.
""", flush=True)
            env["MAV_CONN"] = FC_DEV
            env["FC_TRANSPORT"] = "msp"
            env["MAV_BAUD"] = str(FC_BAUD)

        else:
            from fc_control import COPTER_MODES
            mode = COPTER_MODES.get(detail["custom_mode"],
                                    f"MODE{detail['custom_mode']}")
            ok(f"firmware: ArduPilot Copter   mode={mode}   "
               f"{'ARMED' if detail['armed'] else 'DISARMED'}")
            want = os.environ.get("FC_STEER_MODES",
                                  "ALT_HOLD" if intercept else "ACRO")
            if mode not in {m.strip().upper() for m in want.split(",")}:
                warn(f"the FC is in {mode}, not {want}.  Switch the pilot's mode "
                     f"switch to {want} —")
                warn("that is the mode this control law's sticks are defined in "
                     + ("(lean angles + climb rate)." if intercept
                        else "(body rates)."))

            if direct:
                print("\n== 2) transport: MAVLink direct on the FC serial ==",
                      flush=True)
                info("no router: Mission Planner cannot share the port")
                conn = FC_DEV
            else:
                print("\n== 2) MAVLink router (background subprocess) ==",
                      flush=True)
                if not MAVPROXY.exists():
                    die(f"mavproxy.py missing: {MAVPROXY}  "
                        "(or drop --router to go direct)")
                router = start_router()
                time.sleep(4)
                if router.poll() is not None:
                    die(f"router exited immediately — see {LOG_DIR / 'router.log'}")
                ok(f"router started (pid {router.pid})")
                info("verifying the router is passing FC traffic "
                     "(autopilot heartbeat on 14551) ...")
                if not wait_router_heartbeat(timeout=12):
                    die("no AUTOPILOT heartbeat via the router on 14551 — see "
                        f"{LOG_DIR / 'router.log'}\n"
                        "                 Drop --router to cut the router out of "
                        "the path.")
                ok("router is passing FC heartbeats")
                conn = f"udpin:{UDP_TRACKER}"

            env["MAV_CONN"] = conn
            env["FC_TRANSPORT"] = "mavlink"
            env["MAV_BAUD"] = str(FC_BAUD)

            # ── 3) does an override actually land? ────────────────────────────
            if not args.no_verify:
                print("\n== 3) RC override acceptance check ==", flush=True)
                result = verify_overrides(FC_DEV, FC_BAUD, conn)
                if result is False:
                    die("the FC is NOT accepting RC overrides, so the tracker "
                        "cannot steer it.\n"
                        "                 The report above names the cause. "
                        "Re-run after fixing it,\n"
                        "                 or pass --no-verify to start anyway.")
                elif result is True:
                    ok("the FC accepts our RC overrides — the control transport "
                       "works")

            if not args.no_mp and not direct:
                print("\n== 4) Mission Planner ==", flush=True)
                start_mission_planner()

        # ── run the tracker ──────────────────────────────────────────────────
        print("\n== tracker.py ==", flush=True)
        info(f"MAV_CONN={env['MAV_CONN']}  FC_TRANSPORT={env['FC_TRANSPORT']}")
        info("Ctrl-C stops the tracker; it releases every overridden channel "
             "back to the")
        info("pilot, then this launcher stops the router it started.")
        # Foreground: Ctrl-C reaches tracker (same process group); it exits
        # cleanly, then the finally block stops the router we started.
        subprocess.run([str(YOLO_PY), str(TRACKER)], cwd=str(HERE), env=env)

    except KeyboardInterrupt:
        pass
    finally:
        if router is not None and router.poll() is None:
            print(f"\n  stopping router (pid {router.pid})", flush=True)
            router.terminate()
            try:
                router.wait(timeout=5)
            except subprocess.TimeoutExpired:
                router.kill()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("\n[abort] never connected to the FC — nothing was commanded.")
        sys.exit(130)
