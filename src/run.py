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


def info(m): pass
def ok(m):   pass
def warn(m): pass


def die(msg):

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

        return

    if not report.get("ok"):

        return

    if report["cuda"]:

        return










    if os.environ.get("ALLOW_CPU_DETECTOR") == "1":

        return
    die("refusing to start a blind seeker.  Set ALLOW_CPU_DETECTOR=1 to "
        "override\n                 (bench//HUD work only — NOT for flight).")



def wait_for_fc_device(quiet=False):
    dev = resolve_fc_dev()
    if dev:
        return dev
    if not quiet:
        pass

    dots = 0
    while True:
        time.sleep(1.0)
        dev = resolve_fc_dev()
        if dev:

            return dev
        dots += 1
        if dots % 5 == 0:
            pass


def wait_for_heartbeat(dev, tries=20):
    from pymavlink import mavutil
    for attempt in range(1, tries + 1):
        m = None
        try:
            m = mavutil.mavlink_connection(dev, baud=115200)
            hb = m.wait_heartbeat(timeout=3)
            if hb is not None:


                return True
        except Exception as e:
            if attempt == 1:
                pass

        finally:
            if m is not None:
                try:
                    m.close()
                except Exception:
                    pass
        if attempt % 5 == 0:
            pass

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


    return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)


def start_mission_planner():
    if not MP_EXE.exists():

        return
    if subprocess.run(["pgrep", "-f", "MissionPlanner.exe"],
                      stdout=subprocess.DEVNULL).returncode == 0:

        return

    env = dict(os.environ, DISPLAY=os.environ.get("DISPLAY", ":0"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "missionplanner.log", "w") as log:
        subprocess.Popen(["mono", str(MP_EXE)], stdout=log, stderr=subprocess.STDOUT,
                         env=env)
    time.sleep(2)



def banner(intercept):
    env = os.environ




    if intercept:









        tilt = float(env["CAM_TILT_DEG"])
        lean = float(env["MAX_LEAN_DEG"])
        ok, lo, hi, advice = geometry_check(tilt, lean)








        want = pn_gain_for(lean)
        have = float(env["ROLL_PN_GAIN"])
        if abs(have - want) > 0.20 * want:
            pass



        if not ok:
            pass








        commit = env.get("COMMIT_CHAN", "0")
        if commit and commit != "0":
            pass

        else:
            pass


        lim = env.get("ENGAGE_MAX_S", "0")







    else:
        pass
















    _banner_seeker_view(env)



def _banner_seeker_view(env):
    if env.get("STREAM", "1").strip().lower() not in ("1", "true", "yes", "on"):



        return
    port = env.get("STREAM_PORT", "8080")
    try:
        from mjpeg_stream import local_ips
        addrs = local_ips()
    except Exception:
        addrs = []

    if addrs:
        for ip in addrs:
            pass

    else:
        pass






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
    from flight_config import MOUNT_TILT_DEG
    os.environ.setdefault("CAMERA_MOUNT_DEG", str(MOUNT_TILT_DEG))

    if intercept and "CAM_TILT_DEG" not in os.environ:
        os.environ["CAM_TILT_DEG"] = HARDCODED_CAM_TILT_DEG















    if args.no_stream:
        os.environ["STREAM"] = "0"
    if args.stream_port:
        os.environ["STREAM_PORT"] = str(args.stream_port)

    os.environ.setdefault("DISPLAY", ":0")
    banner(intercept)

    if not args.no_wait:
        dev = wait_for_fc_device()
        if not wait_for_heartbeat(dev):



            return 1
        os.environ.setdefault("FC_DEV", dev)
        FC_DEV = dev
        time.sleep(0.5)


    if not Path(FC_DEV).exists():
        die(f"FC serial '{FC_DEV}' not found — is the flight controller plugged in?")

    if not Path(CAM_DEV).exists():
        die(f"camera '{CAM_DEV}' not found — check CSI ribbon / USB cam")

    if not YOLO_PY.exists():
        die(f"yolo-env python missing: {YOLO_PY}")
    if not TRACKER.exists():
        die(f"tracker.py missing: {TRACKER}")

    check_detector_device()


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



















            env["MAV_CONN"] = FC_DEV
            env["FC_TRANSPORT"] = "msp"
            env["MAV_BAUD"] = str(FC_BAUD)

        else:
            from fc_control import COPTER_MODES
            mode = COPTER_MODES.get(detail["custom_mode"],
                                    f"MODE{detail['custom_mode']}")


            want = os.environ.get("FC_STEER_MODES",
                                  "ALT_HOLD" if intercept else "ACRO")
            if mode not in {m.strip().upper() for m in want.split(",")}:
                pass





            if direct:



                conn = FC_DEV
            else:


                if not MAVPROXY.exists():
                    die(f"mavproxy.py missing: {MAVPROXY}  "
                        "(or drop --router to go direct)")
                router = start_router()
                time.sleep(4)
                if router.poll() is not None:
                    die(f"router exited immediately — see {LOG_DIR / 'router.log'}")



                if not wait_router_heartbeat(timeout=12):
                    die("no AUTOPILOT heartbeat via the router on 14551 — see "
                        f"{LOG_DIR / 'router.log'}\n"
                        "                 Drop --router to cut the router out of "
                        "the path.")

                conn = f"udpin:{UDP_TRACKER}"

            env["MAV_CONN"] = conn
            env["FC_TRANSPORT"] = "mavlink"
            env["MAV_BAUD"] = str(FC_BAUD)

            if not args.no_verify:

                result = verify_overrides(FC_DEV, FC_BAUD, conn)
                if result is False:
                    die("the FC is NOT accepting RC overrides, so the tracker "
                        "cannot steer it.\n"
                        "                 The report above names the cause. "
                        "Re-run after fixing it,\n"
                        "                 or pass --no-verify to start anyway.")
                elif result is True:
                    pass


            if not args.no_mp and not direct:

                start_mission_planner()






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

        sys.exit(130)
