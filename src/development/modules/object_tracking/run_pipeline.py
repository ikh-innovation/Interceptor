#!/usr/bin/env python3
"""
Face-tracker bench pipeline launcher  (Jetson Orin Nano)  — pure Python, no systemd.

Brings the pipeline up in order, then blocks on the tracker:
    1. MAVLink router   (MAVProxy: FC serial  ->  UDP 14550 [Mission Planner] + 14551 [tracker])
    2. Mission Planner  (mono)                              [skip with --no-mp]
    3. tracker.py       (yolo-env)                          [foreground]

Preflight verifies the FC serial, the camera, and that the router is passing FC
heartbeats before launching anything, and stops on the first failure.  The router
this launcher starts is stopped again when tracker.py exits; a router that is
already running is reused and left alone.

Run it:
    /home/iknowhow/yolo-env/bin/python run_pipeline.py
    python run_pipeline.py --no-mp          # skip Mission Planner
Environment overrides: FC_DEV, FC_BAUD, CAM_DEV.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ── paths / config ────────────────────────────────────────────────────────────
HOME       = Path("/home/iknowhow")
YOLO_PY    = HOME / "yolo-env/bin/python"
HERE       = Path(__file__).resolve().parent          # the object_tracking folder
TRACKER    = HERE / "tracker.py"
MP_EXE     = HOME / "MissionPlanner/MissionPlanner.exe"
MAVPROXY   = HOME / ".local/bin/mavproxy.py"
FT_DIR     = HOME / "face_tracker"                     # logs + mavproxy state
LOG_DIR    = FT_DIR / "logs"
STATE_DIR  = FT_DIR / "mavproxy-state"

def _resolve_fc_dev() -> str:
    """Locate the flight controller's MAVLink serial interface.

    The board's by-id serial string is NOT stable across boots (observed
    changing), and its raw ACM number re-enumerates too — so a hardcoded path
    breaks after a reboot.  Match the stable ArduPilot vendor prefix on the
    MAVLink interface (``-if00``) instead.  An explicit FC_DEV that actually
    exists always wins; otherwise auto-detect, falling back to /dev/ttyACM0."""
    env = os.environ.get("FC_DEV")
    if env and Path(env).exists():
        return env
    matches = sorted(Path("/dev/serial/by-id").glob("usb-ArduPilot_*-if00"))
    if matches:
        return str(matches[0])
    return env or "/dev/ttyACM0"


FC_DEV   = _resolve_fc_dev()
FC_BAUD  = os.environ.get("FC_BAUD", "115200")        # nominal; USB CDC-ACM ignores it
CAM_DEV  = os.environ.get("CAM_DEV", "/dev/video0")

UDP_MP      = "127.0.0.1:14550"                        # Mission Planner listens here
UDP_TRACKER = "127.0.0.1:14551"                        # tracker.py binds udpin here


def info(m): print(f"  {m}")
def ok(m):   print(f"  [ok] {m}")


def die(msg):
    print(f"\n[PREFLIGHT FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def router_running() -> bool:
    """True if a MAVProxy router is already up (reuse it, don't manage it)."""
    return subprocess.run(["pgrep", "-f", "mavproxy.py"],
                          stdout=subprocess.DEVNULL).returncode == 0


def wait_heartbeat(timeout=10) -> bool:
    """Confirm the router is passing FC traffic by briefly binding tracker's port
    (udpin 14551) and waiting for a heartbeat, then releasing it so tracker can
    bind it a moment later."""
    from pymavlink import mavutil
    m = mavutil.mavlink_connection(f"udpin:{UDP_TRACKER}")
    hb = m.wait_heartbeat(timeout=timeout)
    m.close()
    return hb is not None


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
        info(f"[warn] Mission Planner not found at {MP_EXE} — skipping")
        return
    if subprocess.run(["pgrep", "-f", "MissionPlanner.exe"],
                      stdout=subprocess.DEVNULL).returncode == 0:
        ok("Mission Planner already running")
        return
    info("launching Mission Planner (mono) — connect it to UDP 14550 (see README)")
    env = dict(os.environ, DISPLAY=os.environ.get("DISPLAY", ":0"))
    with open(LOG_DIR / "missionplanner.log", "w") as log:
        subprocess.Popen(["mono", str(MP_EXE)], stdout=log, stderr=subprocess.STDOUT,
                         env=env)
    time.sleep(2)


def main():
    ap = argparse.ArgumentParser(description="Face-tracker bench pipeline launcher")
    ap.add_argument("--no-mp", action="store_true", help="skip launching Mission Planner")
    args = ap.parse_args()

    print("== Face-tracker preflight ==")
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
    if not MAVPROXY.exists():
        die(f"mavproxy.py missing: {MAVPROXY}")
    ok(f"tracker:   {TRACKER}")

    router = None                      # the router process WE started (else None)
    try:
        print("== 1) MAVLink router (background subprocess) ==")
        if router_running():
            ok("a MAVProxy router is already running — reusing it (not managing it)")
        else:
            router = start_router()
            time.sleep(4)
            if router.poll() is not None:
                die(f"router exited immediately — see {LOG_DIR / 'router.log'}")
            ok(f"router started (pid {router.pid})")

        info("verifying router is passing FC traffic (heartbeat on 14551) ...")
        if not wait_heartbeat(timeout=10):
            die(f"no heartbeat via router on 14551 — see {LOG_DIR / 'router.log'}")
        ok("router is passing FC heartbeats")

        if not args.no_mp:
            print("== 2) Mission Planner ==")
            start_mission_planner()
        else:
            print("== 2) Mission Planner (skipped: --no-mp) ==")

        print("== 3) tracker.py ==")
        info("starting tracker (Ctrl-C to stop; sends all-stop + closes link, "
             "then stops the router it started)")
        # Foreground: Ctrl-C reaches tracker (same process group); it exits
        # cleanly, then the finally block stops the router we started.
        subprocess.run([str(YOLO_PY), str(TRACKER)], cwd=str(HERE))

    except KeyboardInterrupt:
        pass
    finally:
        if router is not None and router.poll() is None:
            print(f"\n  stopping router (pid {router.pid})")
            router.terminate()
            try:
                router.wait(timeout=5)
            except subprocess.TimeoutExpired:
                router.kill()


if __name__ == "__main__":
    main()
