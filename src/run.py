#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

HOME       = Path.home()


def _find_yolo_py() -> Path:
    override = os.environ.get("WASP_YOLO_PY")
    if override:
        return Path(override).expanduser()
    project_venv = HERE.parent / ".venv/bin/python"
    if project_venv.exists():
        return project_venv
    for name in ("yolo-env", "yolo"):
        cand = HOME / name / "bin/python"
        if cand.exists():
            return cand
    if importlib.util.find_spec("ultralytics") is not None:
        return Path(sys.executable)
    return HOME / "yolo-env/bin/python"


YOLO_PY    = _find_yolo_py()
TRACKER    = HERE / "tracker.py"
MP_EXE     = HOME / "MissionPlanner/MissionPlanner.exe"
MAVPROXY   = HOME / ".local/bin/mavproxy.py"
FT_DIR     = HOME / "wasp_tracker"
LOG_DIR    = FT_DIR / "logs"
STATE_DIR  = FT_DIR / "mavproxy-state"

if os.environ.get("_WASP_REEXEC") != "1" and YOLO_PY.exists():
    try:
        import ultralytics
    except ImportError:
        os.environ["_WASP_REEXEC"] = "1"
        os.execv(str(YOLO_PY), [str(YOLO_PY), str(Path(__file__).resolve())]
                 + sys.argv[1:])

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

INTERCEPT = {
    "CONTROL_LAW": "intercept",
    "CONTROL_AXES": "roll,pitch,yaw,throttle",
    "FC_STEER_MODES": "ALT_HOLD",
    "KP_YAW": "4.5",
    "MAX_LEAN_DEG": "20.0",
    "ROLL_PN_GAIN": "45.0",
    "MAX_BANK_DEG": "30.0",
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

SEEKER_VIEW = {
    "STREAM": "1",
    "STREAM_PORT": "8080",
    "STREAM_QUALITY": "70",
    "STREAM_FPS": "15",
}

_LENS_HALF_V_DEG = 24.4
_FRAME_MARGIN_DEG = 8.0
_LEAN_CEILING_DEG = 20.0
_DRAG_EST = 0.0159
_PN_N = 3.0


def geometry_check(tilt_deg, lean_deg):
    lo = -_LENS_HALF_V_DEG + tilt_deg - lean_deg
    hi = _LENS_HALF_V_DEG + tilt_deg - lean_deg
    ok = hi >= _FRAME_MARGIN_DEG
    advice = ""
    if not ok:
        need = math.ceil(lean_deg + _FRAME_MARGIN_DEG - _LENS_HALF_V_DEG)
        advice = (f"tilt the camera up to at least {max(0, need)} deg, "
                  f"or drop MAX_LEAN_DEG to "
                  f"{max(8.0, _LENS_HALF_V_DEG - _FRAME_MARGIN_DEG + tilt_deg):.0f}")
    return ok, lo, hi, advice


def pn_gain_for(lean_deg):
    v = math.sqrt(9.80665 * math.tan(math.radians(lean_deg)) / _DRAG_EST)
    return _PN_N * v


def resolve_fc_dev():
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
FC_BAUD  = os.environ.get("FC_BAUD", "115200")
CAM_DEV  = os.environ.get("CAM_DEV", "/dev/video0")

UDP_MP      = "127.0.0.1:14550"
UDP_TRACKER = "127.0.0.1:14551"


def info(m): print(f"  {m}", flush=True)
def ok(m):   print(f"  [ok] {m}", flush=True)
def warn(m): print(f"  [warn] {m}", flush=True)


def die(msg):
    print(f"\n[PREFLIGHT FAIL] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def check_detector_device():
    probe = (
        "import json,sys\n"
        "try:\n"
        "    import torch\n"
        "    print(json.dumps({'ok':True,'cuda':torch.cuda.is_available(),\n"
        "                      'ver':torch.__version__,\n"
        "                      'dev':torch.cuda.get_device_name(0)\n"
        "                             if torch.cuda.is_available() else 'cpu'}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok':False,'err':str(e)}))\n"
    )
    try:
        out = subprocess.run([str(YOLO_PY), "-c", probe],
                             capture_output=True, text=True, timeout=120)
        report = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:
        warn(f"could not probe the detector device ({exc}) — continuing.")
        return

    if not report.get("ok"):
        warn(f"torch did not import in {YOLO_PY}: {report.get('err')}")
        return

    if report["cuda"]:
        ok(f"detector:  CUDA on {report['dev']}  (torch {report['ver']})")
        return

    warn(f"DETECTOR IS ON THE CPU — torch {report['ver']} has no CUDA.")
    warn("  yolov8x measures ~3.3 s per frame this way (0.3 fps) against a")
    warn("  50 Hz control loop, so the law flies a FROZEN detection and the")
    warn("  symptom looks like bad tracking, not a bad install.")
    warn(f"  Interpreter: {YOLO_PY}")
    warn("  Fix: install the JetPack CUDA wheel, e.g.")
    warn("    <venv>/bin/pip install --no-deps --force-reinstall \\")
    warn("      --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \\")
    warn("      torch==2.8.0 torchvision==0.23.0")
    if os.environ.get("ALLOW_CPU_DETECTOR") == "1":
        warn("  ALLOW_CPU_DETECTOR=1 — continuing anyway.")
        return
    die("refusing to start a blind seeker.  Set ALLOW_CPU_DETECTOR=1 to "
        "override\n                 (bench//HUD work only — NOT for flight).")



def wait_for_fc_device(quiet=False):
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
    return subprocess.run(["pgrep", "-f", "mavproxy.py"],
                          stdout=subprocess.DEVNULL).returncode == 0


def stop_router_or_die():
    if router_running():
        die("a MAVProxy router is already running and owns the FC serial port.\n"
            "                 Stop it first:  pkill -f mavproxy.py")



def identify_firmware(dev, baud):
    from fc_check import probe_mavlink, probe_msp

    master, hb = probe_mavlink(dev, int(baud), timeout=8.0)
    if master is not None:
        mode = hb.custom_mode
        armed = bool(hb.base_mode & 0x80)
        master.close()
        time.sleep(0.4)
        return "ardupilot", dict(custom_mode=mode, armed=armed)

    msp = probe_msp(dev, int(baud))
    if msp is not None:
        return "betaflight", msp
    return None, None


def verify_overrides(dev, baud, conn):
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



def wait_router_heartbeat(timeout=10) -> bool:
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



def banner(intercept):
    env = os.environ
    print("=" * 74)
    print("WASP drone tracker — " + ("INTERCEPT (free flight)" if intercept
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
    if env.get("STREAM", "1").strip().lower() not in ("1", "true", "yes", "on"):
        print("-" * 74)
        print("  seeker view  : OFF (STREAM=%s).  Nothing to watch remotely."
              % env.get("STREAM", ""))
        return
    port = env.get("STREAM_PORT", "8080")
    try:
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
    args = ap.parse_args([a for a in sys.argv[1:] if a != "--"])

    direct = not args.router

    intercept = not args.gimbal
    if os.environ.get("CONTROL_LAW", "").strip().lower() == "servo":
        intercept = False
    elif os.environ.get("CONTROL_LAW", "").strip().lower() == "intercept":
        intercept = True

    for k, v in (INTERCEPT if intercept else GIMBAL).items():
        os.environ.setdefault(k, v)

    for k, v in SEEKER_VIEW.items():
        os.environ.setdefault(k, v)

    HARDCODED_CAM_TILT_DEG = "30"

    if intercept and "CAM_TILT_DEG" not in os.environ:
        os.environ["CAM_TILT_DEG"] = HARDCODED_CAM_TILT_DEG
        print(f"""
[SETUP] CAM_TILT_DEG was not set — using the sky-mode {HARDCODED_CAM_TILT_DEG} deg.

  This is an ASSUMPTION about the hardware, not a measurement.  The measured
  mount angle belongs in flight_config.py (MOUNT_TILT_DEG); start-wasp.sh
  exports it.  Overriding still works and still wins:

      CAM_TILT_DEG=<measured> ./run.py

  For the bench polarity check (props off), 0 is the honest value:

      CAM_TILT_DEG=0 REQUIRE_ARMED=0 ./run.py --no-wait
""", file=sys.stderr)

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
        time.sleep(0.5)

    print("\n== Tracker preflight ==", flush=True)
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
    check_detector_device()

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
    router = None
    tracker_rc = 0

    try:
        if fw == "betaflight":
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

        print("\n== tracker.py ==", flush=True)
        info(f"MAV_CONN={env['MAV_CONN']}  FC_TRANSPORT={env['FC_TRANSPORT']}")
        info("Ctrl-C stops the tracker; it releases every overridden channel "
             "back to the")
        info("pilot, then this launcher stops the router it started.")
        _gld = "/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0"
        if os.path.exists(_gld) and _gld not in env.get("LD_PRELOAD", ""):
            env["LD_PRELOAD"] = ":".join(x for x in (_gld, env.get("LD_PRELOAD", "")) if x)
        rc = subprocess.run([str(YOLO_PY), str(TRACKER)],
                            cwd=str(HERE), env=env).returncode
        if rc < 0:
            sig = -rc
            tracker_rc = 0 if sig in (2, 15) else 128 + sig
        else:
            tracker_rc = rc

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
    return tracker_rc


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("\n[abort] never connected to the FC — nothing was commanded.")
        sys.exit(130)
