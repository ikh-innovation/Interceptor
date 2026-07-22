#!/usr/bin/env python3
"""
test_mavlink_nav_commands.py

Tests whether a flight controller (reached from a Jetson Orin Nano companion
computer) is ALIVE and ACCEPTING MAVLink navigation-related commands.

SAFETY MODEL
------------
By default this script only runs READ-ONLY / NON-MOTION checks:
    - Heartbeat (link alive, autopilot type/mode reported)
    - Autopilot version / capabilities request
    - Current mode + armed-state readback
    - "Dry" command test: sends MAV_CMD_DO_SET_MODE to whatever mode the FC
      is ALREADY in (harmless no-op) and confirms COMMAND_ACK is received.

Arming, mode changes to a different mode, and takeoff/waypoint commands are
GATED behind --enable-motion-tests and an interactive confirmation, because
COMMAND_ACK just means "the command was accepted" -- on a live vehicle with
props on, an accepted command can actually move it.

Install:
    pip install pymavlink --break-system-packages

Usage examples:
    # Serial link (typical Jetson <-> Pixhawk UART)
    python3 test_mavlink_nav_commands.py -connect /dev/ttyACM0 --baud 115200

    # Include gated motion-capable tests (arm/disarm, mode change) -- CAUTION
    python3 test_mavlink_nav_commands.py -connect /dev/ttyACM0 --baud 115200 --enable-motion-tests
"""

import argparse
import sys
import time

try:
    from pymavlink import mavutil
except ImportError:
    mavutil = None


def connect(link, baud, timeout=10):
    print(f"[MAVLINK] Connecting to {link} ...")
    conn = mavutil.mavlink_connection(link, baud=baud)
    print("[MAVLINK] Waiting for heartbeat...")
    hb = conn.wait_heartbeat(timeout=timeout)
    if hb is None:
        print("[MAVLINK] FAIL: no heartbeat received -- link is not responsive.")
        return None
    print(f"[MAVLINK] PASS: heartbeat from system {conn.target_system}, "
          f"component {conn.target_component} (autopilot={hb.autopilot}, type={hb.type})")
    return conn


def get_mode_and_armed(conn, timeout=5):
    hb = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=timeout)
    if hb is None:
        return None, None
    armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    try:
        mode = mavutil.mode_string_v10(hb)
    except Exception:
        mode = str(hb.custom_mode)
    return mode, armed


def request_autopilot_capabilities(conn, timeout=5):
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0,
        mavutil.mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION, 0, 0, 0, 0, 0, 0,
    )
    msg = conn.recv_match(type="AUTOPILOT_VERSION", blocking=True, timeout=timeout)
    if msg is None:
        return False, "no AUTOPILOT_VERSION response"
    return True, f"flight_sw_version={msg.flight_sw_version}, capabilities={msg.capabilities}"


def wait_for_ack(conn, command_id, timeout=5):
    start = time.time()
    while time.time() - start < timeout:
        msg = conn.recv_match(type="COMMAND_ACK", blocking=True, timeout=timeout)
        if msg is None:
            return None
        if msg.command == command_id:
            return msg.result
    return None


def test_dry_mode_command(conn, current_mode_num, timeout=5):
    """Re-send the FC's own current mode -- a no-op that still requires the FC
    to parse and ACK a real MAV_CMD_DO_SET_MODE, proving command channel works."""
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        current_mode_num, 0, 0, 0, 0, 0,
    )
    result = wait_for_ack(conn, mavutil.mavlink.MAV_CMD_DO_SET_MODE, timeout=timeout)
    return result


RESULT_NAMES = {
    0: "MAV_RESULT_ACCEPTED",
    1: "MAV_RESULT_TEMPORARILY_REJECTED",
    2: "MAV_RESULT_DENIED",
    3: "MAV_RESULT_UNSUPPORTED",
    4: "MAV_RESULT_FAILED",
    5: "MAV_RESULT_IN_PROGRESS",
}


def run_readonly_tests(conn):
    results = []

    ok, detail = request_autopilot_capabilities(conn)
    results.append(("Request autopilot capabilities", ok, detail))

    mode, armed = get_mode_and_armed(conn)
    results.append(("Read current mode/armed state", mode is not None,
                     f"mode={mode}, armed={armed}"))

    hb = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=5)
    if hb is not None:
        ack = test_dry_mode_command(conn, hb.custom_mode)
        ok = ack is not None
        detail = RESULT_NAMES.get(ack, str(ack)) if ok else "no ACK received"
        results.append(("Dry-run DO_SET_MODE (same mode, no-op)", ok, detail))
    else:
        results.append(("Dry-run DO_SET_MODE", False, "no heartbeat to read current mode"))

    return results


def get_current_position(conn, timeout=5):
    """Returns (lat, lon, alt_m) from GLOBAL_POSITION_INT, or None if unavailable
    (e.g. no GPS lock -- common on a bench/indoors)."""
    msg = conn.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=timeout)
    if msg is None:
        return None
    return msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000.0


def test_set_mode(conn, mode_name, timeout=5):
    """Set a specific named flight mode (e.g. GUIDED) and confirm ACK."""
    mode_mapping = conn.mode_mapping()
    if not mode_mapping or mode_name not in mode_mapping:
        return None, f"mode '{mode_name}' not found in this autopilot's mode mapping"
    mode_num = mode_mapping[mode_name]
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_num, 0, 0, 0, 0, 0,
    )
    result = wait_for_ack(conn, mavutil.mavlink.MAV_CMD_DO_SET_MODE, timeout=timeout)
    return result, None


def test_takeoff(conn, altitude_m, timeout=5):
    """Send MAV_CMD_NAV_TAKEOFF. Expected to be DENIED/REJECTED if disarmed or
    pre-arm checks fail -- that's correct FC behavior, not a channel failure."""
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
        0, 0, 0, 0, 0, 0, altitude_m,  # param7 = target altitude
    )
    result = wait_for_ack(conn, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, timeout=timeout)
    return result


def test_waypoint_upload(conn, lat, lon, alt_m, timeout=10):
    """Uploads a single-item mission (one waypoint) using the standard MAVLink
    mission protocol handshake and confirms the FC accepts it via MISSION_ACK.
    This exercises the real navigation/waypoint pathway, not just command ACKs."""
    target_system = conn.target_system
    target_component = conn.target_component

    conn.mav.mission_count_send(target_system, target_component, 1)

    req = conn.recv_match(type=["MISSION_REQUEST", "MISSION_REQUEST_INT"],
                           blocking=True, timeout=timeout)
    if req is None:
        return False, "FC did not request the mission item (no MISSION_REQUEST)"

    conn.mav.mission_item_int_send(
        target_system, target_component,
        0,  # seq
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        0, 1,  # current, autocontinue
        0, 0, 0, 0,  # params 1-4 (hold time, accept radius, pass radius, yaw)
        int(lat * 1e7), int(lon * 1e7), alt_m,
    )

    ack = conn.recv_match(type="MISSION_ACK", blocking=True, timeout=timeout)
    if ack is None:
        return False, "no MISSION_ACK received"
    accepted = ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED
    return accepted, f"MISSION_ACK type={ack.type}"


def run_motion_tests(conn, target_altitude, waypoint_offset_deg):
    """Gated behind --enable-motion-tests. These can cause real movement /
    arming if run against a live, armed-capable vehicle. Use on a bench /
    props-off setup only."""
    results = []

    print("\n!!! MOTION-CAPABLE TESTS ENABLED !!!")
    confirm = input("Type 'YES' to confirm props are OFF / vehicle is safe to test: ")
    if confirm.strip() != "YES":
        print("Aborting motion tests -- confirmation not given.")
        return results

    # Example: request arm (many FCs will reject this without pre-arm checks
    # passing, e.g. GPS lock, calibration, etc. -- that's expected/fine).
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1, 0, 0, 0, 0, 0, 0,  # param1=1 -> arm
    )
    ack = wait_for_ack(conn, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, timeout=5)
    ok = ack is not None
    detail = RESULT_NAMES.get(ack, str(ack)) if ok else "no ACK received"
    results.append(("ARM command accepted/ack'd", ok, detail))

    # --- Real navigation command tests ---
    # These test whether the FC's navigation pathway accepts commands. A
    # MAV_RESULT_DENIED/TEMPORARILY_REJECTED is often the *correct* outcome
    # (e.g. disarmed, no GPS lock) -- it still proves the command channel and
    # nav logic are alive and evaluating the command, which is what "can
    # navigation commands be accepted" is really asking.

    mode_result, mode_err = test_set_mode(conn, "GUIDED")
    if mode_err:
        results.append(("Set mode GUIDED", False, mode_err))
    else:
        ok = mode_result is not None
        detail = RESULT_NAMES.get(mode_result, str(mode_result)) if ok else "no ACK received"
        results.append(("Set mode GUIDED", ok, detail))

    takeoff_result = test_takeoff(conn, target_altitude)
    ok = takeoff_result is not None
    detail = RESULT_NAMES.get(takeoff_result, str(takeoff_result)) if ok else "no ACK received"
    results.append((f"MAV_CMD_NAV_TAKEOFF ({target_altitude}m)", ok, detail))

    pos = get_current_position(conn, timeout=5)
    if pos is not None:
        lat, lon, alt = pos
        wp_lat = lat + waypoint_offset_deg
        wp_lon = lon
        wp_alt = max(alt, target_altitude)
        note = ""
    else:
        # No GPS fix (e.g. bench-testing indoors). Fall back to placeholder
        # coordinates -- this still validates that the FC's mission protocol
        # accepts an uploaded waypoint, independent of whether it could
        # actually navigate there right now.
        wp_lat, wp_lon, wp_alt = 0.0001, 0.0001, target_altitude
        note = " (no GPS fix -- using placeholder coordinates, protocol-only test)"

    ok, detail = test_waypoint_upload(conn, wp_lat, wp_lon, wp_alt)
    results.append(("Waypoint upload (MISSION_ITEM)", ok, detail + note))

    # Immediately disarm again for safety, regardless of prior results.
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        0, 0, 0, 0, 0, 0, 0,  # param1=0 -> disarm
    )
    ack = wait_for_ack(conn, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, timeout=5)
    ok = ack is not None
    detail = RESULT_NAMES.get(ack, str(ack)) if ok else "no ACK received"
    results.append(("DISARM command accepted/ack'd", ok, detail))

    return results


def print_report(title, results):
    print(f"\n=== {title} ===")
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:40} -> {detail}")


def main():
    parser = argparse.ArgumentParser(description="Test MAVLink navigation command responsiveness")
    parser.add_argument("--connect", required=True,
                         help="MAVLink connection string, e.g. /dev/ttyTHS1 or udp:127.0.0.1:14550")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate for serial connections")
    parser.add_argument("--enable-motion-tests", action="store_true",
                         help="CAUTION: also test arm/disarm, mode change, takeoff, and "
                              "waypoint-upload command acceptance. Props off only.")
    parser.add_argument("--target-altitude", type=float, default=5.0,
                         help="Altitude (m) used in the takeoff/waypoint test commands")
    parser.add_argument("--waypoint-offset-deg", type=float, default=0.0001,
                         help="Latitude offset (deg) from current position for the test "
                              "waypoint (~0.0001 deg is about 11m)")
    args = parser.parse_args()

    if mavutil is None:
        print("pymavlink not installed. Run: pip install pymavlink --break-system-packages")
        sys.exit(1)

    conn = connect(args.connect, args.baud)
    if conn is None:
        sys.exit(1)

    results = run_readonly_tests(conn)
    print_report("Read-only navigation command checks", results)

    if args.enable_motion_tests:
        motion_results = run_motion_tests(conn, args.target_altitude, args.waypoint_offset_deg)
        if motion_results:
            print_report("Motion-capable command checks", motion_results)
            results += motion_results

    all_pass = all(r[1] for r in results)
    print("\nOverall:", "ALL PASS" if all_pass else "SOME FAILURES")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()