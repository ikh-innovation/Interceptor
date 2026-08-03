#!/usr/bin/env python3
"""
Head-tracker bench pipeline launcher  (Jetson Orin)  — pure Python, no systemd.

Brings the pipeline up in the only order that can work, and refuses to start
YOLO if the drone could never be steered anyway:

    1. Identify the flight controller's FIRMWARE, on its own serial port,
       BEFORE anything claims the port.  ArduPilot and Betaflight need
       completely different plumbing, so this decides everything below.
    2. ArduPilot  ->  MAVLink router (MAVProxy: FC serial -> UDP 14550 for
                      Mission Planner + 14551 for the tracker), then verify a
                      real RC override is accepted by the FC.
       Betaflight ->  NO router (it cannot speak MSP) and no Mission Planner;
                      the tracker owns the serial port and talks MSP directly.
    3. Mission Planner (mono)                            [skip with --no-mp]
    4. tracker.py     (yolo-env)                         [foreground]

Preflight stops on the first failure and names it, instead of letting a run
look healthy while nothing reaches the FC — which is exactly how the previous
version could stream setpoints into a void for a whole demo.

Run it:
    /home/iknowhow/yolo-env/bin/python run_pipeline.py
    python run_pipeline.py --no-mp        # router + tracker only
    python run_pipeline.py --direct       # tracker owns the FC serial, no router
    python run_pipeline.py --no-verify    # skip the override check

Environment overrides: FC_DEV, FC_BAUD, CAM_DEV, plus every control knob in
fc_control.commander_from_env (KP_YAW, STICK_LIMIT, YAW_REVERSED, ...).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ── paths / config ────────────────────────────────────────────────────────────
HOME       = Path("/home/iknowhow")
YOLO_PY    = HOME / "yolo-env/bin/python"
TRACKER    = HERE / "tracker.py"
MP_EXE     = HOME / "MissionPlanner/MissionPlanner.exe"
MAVPROXY   = HOME / ".local/bin/mavproxy.py"
FT_DIR     = HOME / "face_tracker"                     # logs + mavproxy state
LOG_DIR    = FT_DIR / "logs"
STATE_DIR  = FT_DIR / "mavproxy-state"


def _resolve_fc_dev() -> str:
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
    return env or "/dev/ttyACM0"


FC_DEV   = _resolve_fc_dev()
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
# 1) FIRMWARE IDENTIFICATION  (before anything claims the port)
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
# 2) ROUTER / MISSION PLANNER
# ══════════════════════════════════════════════════════════════════════════════

def wait_heartbeat(timeout=10) -> bool:
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
    info("launching Mission Planner (mono) — connect it to UDP 14550 (see README)")
    env = dict(os.environ, DISPLAY=os.environ.get("DISPLAY", ":0"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "missionplanner.log", "w") as log:
        subprocess.Popen(["mono", str(MP_EXE)], stdout=log, stderr=subprocess.STDOUT,
                         env=env)
    time.sleep(2)


# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-mp", action="store_true", help="skip launching Mission Planner")
    ap.add_argument("--direct", action="store_true",
                    help="skip the router: the tracker owns the FC serial port. "
                         "Removes the router as a failure mode, but Mission "
                         "Planner cannot share the port.")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the RC-override acceptance check")
    args = ap.parse_args()

    print("== Head-tracker preflight ==", flush=True)
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
            "                 Full report:  python3 fc_check.py "
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
    turn it ON.  Run  python3 fc_check.py  for the full report.

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
            if mode != "ACRO":
                warn(f"the FC is in {mode}, not ACRO.  Switch the pilot's mode "
                     "switch to ACRO —")
                warn("that is the mode this tracker's rate control law is built "
                     "for.")

            if args.direct:
                print("\n== 2) transport: MAVLink direct on the FC serial "
                      "(--direct) ==", flush=True)
                info("no router: Mission Planner cannot share the port")
                conn = FC_DEV
            else:
                print("\n== 2) MAVLink router (background subprocess) ==",
                      flush=True)
                if not MAVPROXY.exists():
                    die(f"mavproxy.py missing: {MAVPROXY}  "
                        "(or re-run with --direct)")
                router = start_router()
                time.sleep(4)
                if router.poll() is not None:
                    die(f"router exited immediately — see {LOG_DIR / 'router.log'}")
                ok(f"router started (pid {router.pid})")
                info("verifying the router is passing FC traffic "
                     "(autopilot heartbeat on 14551) ...")
                if not wait_heartbeat(timeout=12):
                    die("no AUTOPILOT heartbeat via the router on 14551 — see "
                        f"{LOG_DIR / 'router.log'}\n"
                        "                 Try --direct to cut the router out of "
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

            if not args.no_mp and not args.direct:
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


if __name__ == "__main__":
    main()
