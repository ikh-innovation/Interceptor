#!/usr/bin/env python3
"""
fc_check.py  —  bench diagnostic for the WASP head-tracker's flight-control link
==============================================================================
Answers, without YOLO or a camera in the loop, the only question that matters:
"will this flight controller actually move when the Jetson tells it to?"

It does four things:

  1. IDENTIFY   what firmware is on the board (ArduPilot / PX4 / Betaflight /
                INAV), over MAVLink first and MSP second.
  2. REPORT     every setting that decides whether an override is honoured:
                SYSID_MYGCS, RC_OPTIONS, ACRO_TRAINER, the RCMAP channel
                assignment, each channel's calibration, and the ACRO full-stick
                rates.
  3. VERIFY     send a real RC override and prove the FC applied it, by reading
                the value back out of the FC's own RC_CHANNELS stream.  This is
                the test the tracker never had: previously an override could be
                dropped in silence and nothing anywhere would say so.  Safe
                while disarmed — it proves the transport, not the motors.
  4. WIGGLE     (opt-in) sweep the yaw and pitch sticks and report the motor
                output spread, proving the FC turns sticks into differential
                thrust.  PROPS OFF — the airframe will not rotate without props.

Typical session on the Jetson
-----------------------------
    # through the MAVProxy router that run_pipeline.py starts
    python3 fc_check.py --verify

    # straight at the FC's USB port (stop the router first, it owns the port)
    python3 fc_check.py --connect /dev/ttyACM0 --verify

    # then, with the pilot armed in ACRO holding mid throttle, PROPS OFF:
    python3 fc_check.py --wiggle 8 --yes
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fc_control import (COPTER_MODES, MSP_FC_VARIANT, MSP_FC_VERSION, MSP_RC,
                        MSP_STATUS, _msp_frame)

try:
    from pymavlink import mavutil
except ImportError:
    mavutil = None


DEFAULT_ROUTER = "udpin:127.0.0.1:14551"

AUTOPILOT_NAMES = {
    0: "GENERIC", 3: "ARDUPILOTMEGA (ArduPilot)", 8: "INVALID (a GCS, not an FC)",
    12: "PX4", 13: "SMACCMPILOT", 14: "AUTOQUAD", 16: "PPZ", 19: "FP",
}


# ══════════════════════════════════════════════════════════════════════════════
# device discovery
# ══════════════════════════════════════════════════════════════════════════════

def resolve_device(explicit=None):
    """Pick the link to probe.  An explicit value always wins.  Otherwise use a
    running router if there is one, else the FC's own USB serial port."""
    if explicit:
        return explicit
    if router_running():
        print(f"[i] a MAVProxy router is running — probing through it "
              f"({DEFAULT_ROUTER})")
        return DEFAULT_ROUTER
    for pat in ("usb-ArduPilot_*-if00", "*-if00", "*"):
        matches = sorted(Path("/dev/serial/by-id").glob(pat)) \
            if Path("/dev/serial/by-id").is_dir() else []
        if matches:
            print(f"[i] no router running — probing the FC serial port directly")
            return str(matches[0])
    for dev in ("/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0"):
        if Path(dev).exists():
            return dev
    return DEFAULT_ROUTER


def router_running():
    try:
        return subprocess.run(["pgrep", "-f", "mavproxy.py"],
                              stdout=subprocess.DEVNULL).returncode == 0
    except Exception:
        return False


def is_network(conn):
    return str(conn).startswith(("udp", "tcp"))


# ══════════════════════════════════════════════════════════════════════════════
# 1) IDENTIFY
# ══════════════════════════════════════════════════════════════════════════════

def probe_mavlink(conn, baud, timeout=12.0):
    """Wait for an AUTOPILOT heartbeat (never a GCS's).  Returns the connection
    plus the heartbeat, or (None, None)."""
    if mavutil is None:
        print("[x] pymavlink is not installed:  pip install pymavlink")
        return None, None
    print(f"\n[1] MAVLink probe on {conn} ...")
    try:
        master = mavutil.mavlink_connection(conn, baud=baud, source_system=255,
                                            source_component=190)
    except Exception as e:
        print(f"    [x] cannot open the link: {e}")
        return None, None

    deadline = perf_counter() + timeout
    saw_gcs = False
    while perf_counter() < deadline:
        msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if msg is None:
            continue
        if (msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_INVALID
                or msg.type == mavutil.mavlink.MAV_TYPE_GCS):
            saw_gcs = True
            continue
        master.target_system = msg.get_srcSystem()
        master.target_component = msg.get_srcComponent()
        name = AUTOPILOT_NAMES.get(msg.autopilot, f"id {msg.autopilot}")
        print(f"    [ok] autopilot heartbeat: {name}   "
              f"sys={master.target_system} comp={master.target_component}")
        return master, msg

    if saw_gcs:
        print("    [x] only GCS/router heartbeats on this link — the FC itself is "
              "NOT reaching it.\n"
              "        The router is up but not talking to the board: check its "
              "--master serial path and that the FC is powered.")
    else:
        print("    [x] no MAVLink heartbeat at all")
    master.close()
    return None, None


def probe_msp(dev, baud=115200):
    """No MAVLink?  Then the board is probably still on Betaflight/INAV, which
    speaks MSP.  Identify it so the report can say what to do about it."""
    if is_network(dev):
        print("\n[1b] MSP probe skipped — MSP needs a serial port, not a UDP link")
        return None
    print(f"\n[1b] MSP probe on {dev} (is this Betaflight/INAV?) ...")
    try:
        import serial
    except ImportError:
        print("    [x] pyserial not installed:  pip install pyserial")
        return None
    try:
        ser = serial.Serial(dev, baud, timeout=0.3)
    except Exception as e:
        print(f"    [x] cannot open {dev}: {e}")
        return None

    def request(cmd, timeout=1.5):
        try:
            ser.reset_input_buffer()
            ser.write(_msp_frame(cmd))
        except Exception:
            return None
        deadline = perf_counter() + timeout
        state, size, got, buf = 0, 0, 0, bytearray()
        while perf_counter() < deadline:
            b = ser.read(1)
            if not b:
                continue
            c = b[0]
            if state == 0:
                state = 1 if c == ord("$") else 0
            elif state == 1:
                state = 2 if c == ord("M") else 0
            elif state == 2:
                state = 3 if c in (ord(">"), ord("!")) else 0
            elif state == 3:
                size, state, buf = c, 4, bytearray()
            elif state == 4:
                got, state = c, (6 if size == 0 else 5)
            elif state == 5:
                buf.append(c)
                if len(buf) >= size:
                    state = 6
            if state == 6:
                ser.read(1)
                if got == cmd:
                    return bytes(buf)
                state = 0
        return None

    variant = request(MSP_FC_VARIANT)
    if not variant:
        print("    [x] no MSP response either")
        ser.close()
        return None
    name = variant[:4].decode(errors="ignore")
    ver = request(MSP_FC_VERSION) or b"\0\0\0"
    vstr = f"{ver[0]}.{ver[1]}.{ver[2]}" if len(ver) >= 3 else "?"
    print(f"    [ok] MSP firmware: {name} {vstr}")
    status = request(MSP_STATUS)
    armed = None
    if status and len(status) >= 11:
        import struct
        armed = bool(struct.unpack_from("<I", status, 6)[0] & 1)
        print(f"    armed: {'YES' if armed else 'NO'}")
    rc = request(MSP_RC)
    if rc:
        import struct
        n = len(rc) // 2
        vals = struct.unpack_from("<" + "H" * n, rc, 0)
        print(f"    live sticks ({n} ch): "
              + " ".join(f"{v}" for v in vals[:8]))
    ser.close()
    return dict(variant=name, version=vstr, armed=armed)


# ══════════════════════════════════════════════════════════════════════════════
# 2) REPORT  (ArduPilot)
# ══════════════════════════════════════════════════════════════════════════════

def read_param(master, name, timeout=2.5):
    try:
        master.mav.param_request_read_send(
            master.target_system, master.target_component, name.encode(), -1)
    except Exception:
        return None
    deadline = perf_counter() + timeout
    while perf_counter() < deadline:
        msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
        if msg is None:
            continue
        pid = msg.param_id
        if isinstance(pid, bytes):
            pid = pid.decode(errors="ignore")
        if pid.strip("\x00") == name:
            return float(msg.param_value)
    return None


def request_rc_stream(master, hz=5):
    try:
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS, int(1e6 / hz),
            0, 0, 0, 0, 0)
    except Exception:
        pass
    try:
        master.mav.request_data_stream_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, hz, 1)
    except Exception:
        pass


def report_ardupilot(master, hb):
    """Everything that decides whether an override is honoured."""
    print("\n[2] ArduPilot configuration")
    mode = COPTER_MODES.get(hb.custom_mode, f"MODE{hb.custom_mode}")
    armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    print(f"    mode                 : {mode}"
          + ("" if mode == "ACRO" else "        <-- expected ACRO"))
    print(f"    armed                : {'YES' if armed else 'NO'}")

    sysid = read_param(master, "SYSID_MYGCS")
    pname = "SYSID_MYGCS"
    if sysid is None:
        sysid = read_param(master, "MAV_GCS_SYSID")
        pname = "MAV_GCS_SYSID"
    if sysid is None:
        print("    SYSID_MYGCS          : ?  (param read failed)")
    else:
        print(f"    {pname:21s}: {int(sysid)}"
              + ("        <-- overrides must be SENT FROM this system id"
                 if int(sysid) != 255 else ""))

    rc_options = read_param(master, "RC_OPTIONS")
    if rc_options is not None:
        bad = int(rc_options) & 2
        print(f"    RC_OPTIONS           : {int(rc_options)}"
              + ("   <-- bit 1 set = FC IGNORES ALL OVERRIDES" if bad else "  (ok)"))

    for name, note in (("ACRO_TRAINER", "0 = pure rate mode (what we want)"),
                       ("ACRO_Y_RATE", "deg/s at full yaw stick"),
                       ("ACRO_RP_RATE", "deg/s at full pitch stick"),
                       ("ACRO_Y_EXPO", "stick curve; 0 = linear"),
                       ("ACRO_RP_EXPO", "stick curve; 0 = linear"),
                       ("RC_OVERRIDE_TIME", "s before an override goes stale"),
                       ("FS_GCS_ENABLE", "0 = no GCS-loss failsafe"),
                       ("MOT_SPIN_ARM", "idle motor output when armed")):
        v = read_param(master, name)
        if v is not None:
            print(f"    {name:21s}: {v:g}   ({note})")

    print("\n    transmitter map / calibration (read from the FC):")
    axes = {}
    for axis in ("roll", "pitch", "throttle", "yaw"):
        ch = read_param(master, f"RCMAP_{axis.upper()}")
        ch = int(ch) if ch else {"roll": 1, "pitch": 2, "throttle": 3, "yaw": 4}[axis]
        mn = read_param(master, f"RC{ch}_MIN")
        mx = read_param(master, f"RC{ch}_MAX")
        tr = read_param(master, f"RC{ch}_TRIM")
        rv = read_param(master, f"RC{ch}_REVERSED")
        axes[axis] = dict(chan=ch,
                          min=int(mn) if mn else 1000,
                          max=int(mx) if mx else 2000,
                          trim=int(tr) if tr else 1500,
                          reversed=bool(int(rv)) if rv is not None else False)
        flag = "" if 1 <= ch <= 8 else "   <-- >8 CANNOT be overridden"
        print(f"      {axis:9s} ch{ch}  "
              f"[{axes[axis]['min']}..{axes[axis]['trim']}..{axes[axis]['max']}]  "
              f"rev={int(axes[axis]['reversed'])}{flag}")

    request_rc_stream(master)
    rc = master.recv_match(type="RC_CHANNELS", blocking=True, timeout=4)
    if rc is None:
        print("\n    [!] no RC_CHANNELS stream — cannot verify overrides "
              "(raise SR*_RC_CHAN or check the link)")
    else:
        vals = [getattr(rc, f"chan{i}_raw", 0) for i in range(1, 9)]
        print(f"\n    live RC in (ch1-8)   : " + " ".join(f"{v:4d}" for v in vals))
        if all(v == 0 for v in vals):
            print("    [!] every channel reads 0 — NO RECEIVER SIGNAL.  The pilot "
                  "cannot arm, and\n        releasing a channel leaves it with no "
                  "source.  Bind/power the receiver.")
    return dict(mode=mode, armed=armed, sysid_mygcs=sysid, axes=axes,
                rc_options=rc_options)


# ══════════════════════════════════════════════════════════════════════════════
# 3) VERIFY  —  does an override actually land?
# ══════════════════════════════════════════════════════════════════════════════

def verify_override(master, cfg, hold=2.5):
    """Send a distinctly-offset yaw override and read it back out of the FC's own
    RC_CHANNELS.  ArduPilot copies an active override into radio_in, which is
    what RC_CHANNELS reports, so a matching echo proves acceptance.

    Sweeps candidate system ids, because sending from the wrong one is THE
    failure mode: ArduPilot drops overrides from any sysid other than
    SYSID_MYGCS, without a single diagnostic message.

    Safe while disarmed: an override changes the FC's stick input, not the
    motors.
    """
    print("\n[3] override verification")
    yaw = cfg["axes"]["yaw"]
    chan = yaw["chan"]
    if not 1 <= chan <= 8:
        print(f"    [x] yaw is on channel {chan}; RC_CHANNELS_OVERRIDE only "
              "reaches 1-8.  Remap RCMAP_YAW.")
        return False

    # A value clearly distinguishable from whatever the radio is sending.
    test_pwm = int(min(yaw["max"], yaw["trim"] + 200))
    candidates = []
    if cfg["sysid_mygcs"] is not None:
        candidates.append(int(cfg["sysid_mygcs"]))
    for s in (255, 1):
        if s not in candidates:
            candidates.append(s)

    for sysid in candidates:
        master.mav.srcSystem = sysid
        print(f"    trying source_system={sysid}: sending ch{chan}={test_pwm} "
              f"for {hold:.1f}s ...", end="", flush=True)
        chans = [0] * 8
        chans[chan - 1] = test_pwm

        deadline = perf_counter() + hold
        best = None
        matched = False
        next_tx = 0.0
        while perf_counter() < deadline:
            now = perf_counter()
            if now >= next_tx:
                next_tx = now + 0.05                 # 20 Hz
                try:
                    master.mav.rc_channels_override_send(
                        master.target_system, master.target_component, *chans)
                except Exception as e:
                    print(f" [x] send failed: {e}")
                    break
            msg = master.recv_match(type="RC_CHANNELS", blocking=True, timeout=0.2)
            if msg is None:
                continue
            got = getattr(msg, f"chan{chan}_raw", None)
            if got is None:
                continue
            best = got
            if abs(int(got) - test_pwm) <= 25 and now > (deadline - hold + 0.4):
                matched = True
                break

        # release before judging, so a failed attempt leaves nothing latched
        try:
            master.mav.rc_channels_override_send(
                master.target_system, master.target_component, *([0] * 8))
        except Exception:
            pass

        if matched:
            print(f" PASS  (FC echoed {best})")
            print(f"\n    ==> The FC ACCEPTS overrides sent from system id {sysid}.")
            if cfg["sysid_mygcs"] is not None and sysid != int(cfg["sysid_mygcs"]):
                print(f"        Note this differs from the SYSID_MYGCS read "
                      f"({int(cfg['sysid_mygcs'])}) — set MAV_SOURCE_SYSTEM={sysid}.")
            else:
                print("        The tracker reads SYSID_MYGCS at startup and adopts "
                      "it automatically, so no extra configuration is needed.")
            return True
        print(f" fail  (FC reports {best})")

    print("\n    [x] NO system id worked — the FC is not applying our overrides.")
    print("        Work through these in order:")
    print("          1. RC_OPTIONS bit 1 (value 2) = ignore overrides.  Clear it.")
    print("          2. If this went through the MAVProxy router, the router may not")
    print("             be forwarding.  Stop it (pkill -f mavproxy.py) and re-run")
    print("             with --connect /dev/ttyACM0 to test the FC directly.")
    print("          3. Confirm the board really is running ArduPilot Copter, and not")
    print("             a build with RC overrides compiled out.")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 4) WIGGLE  —  make it visibly move
# ══════════════════════════════════════════════════════════════════════════════

def wiggle(master, cfg, seconds=8.0, amplitude=0.35, period=3.0):
    """Sweep yaw and pitch with a slow sine and print the SERVO_OUTPUT_RAW spread.

    The spread is the proof: it shows the FC turned our sticks into differential
    MOTOR output, which is the last link in the chain before physics.

    Note what this does and does not show.  With the props OFF the airframe will
    NOT rotate — rotation comes from differential prop thrust, and bare motors
    produce negligible torque.  Props off you are checking the numbers and the
    sound; actual rotation only happens with props fitted.
    """
    print(f"\n[4] wiggle for {seconds:.0f}s  "
          f"(amplitude {amplitude * 100:.0f}% stick, {period:.0f}s period)")
    print("    The FC must be ARMED in ACRO with the pilot holding mid throttle,")
    print("    or the mixer has no headroom and the motor spread stays at 0.")
    print("    PROPS OFF: expect the motors to change pitch audibly and the spread")
    print("    below to swing — but NOT the airframe to turn.  Rotation needs props.")

    yaw, pitch = cfg["axes"]["yaw"], cfg["axes"]["pitch"]

    def pwm(axis_cfg, norm):
        if axis_cfg["reversed"]:
            norm = -norm
        if norm >= 0:
            v = axis_cfg["trim"] + norm * (axis_cfg["max"] - axis_cfg["trim"])
        else:
            v = axis_cfg["trim"] + norm * (axis_cfg["trim"] - axis_cfg["min"])
        return int(round(max(axis_cfg["min"], min(axis_cfg["max"], v))))

    try:
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW, 200000, 0, 0, 0, 0, 0)
    except Exception:
        pass

    t0 = perf_counter()
    next_tx = 0.0
    servo = None
    try:
        while (t := perf_counter() - t0) < seconds:
            if perf_counter() >= next_tx:
                next_tx = perf_counter() + 0.05
                phase = 2 * math.pi * t / period
                ny = amplitude * math.sin(phase)
                np_ = amplitude * math.sin(phase + math.pi / 2)
                chans = [0] * 8
                if 1 <= yaw["chan"] <= 8:
                    chans[yaw["chan"] - 1] = pwm(yaw, ny)
                if 1 <= pitch["chan"] <= 8:
                    chans[pitch["chan"] - 1] = pwm(pitch, np_)
                master.mav.rc_channels_override_send(
                    master.target_system, master.target_component, *chans)
                spread = ""
                if servo is not None:
                    outs = [getattr(servo, f"servo{i}_raw", 0) for i in range(1, 5)]
                    spread = f"  motors {min(outs)}-{max(outs)} (spread {max(outs) - min(outs)})"
                print(f"\r    t={t:4.1f}s  yaw {ny:+.2f} pitch {np_:+.2f}  "
                      f"ch{yaw['chan']}={chans[yaw['chan'] - 1]} "
                      f"ch{pitch['chan']}={chans[pitch['chan'] - 1]}{spread}   ",
                      end="", flush=True)
            m = master.recv_match(type=["SERVO_OUTPUT_RAW", "HEARTBEAT"],
                                  blocking=False)
            if m is not None and m.get_type() == "SERVO_OUTPUT_RAW":
                servo = m
            sleep(0.005)
    except KeyboardInterrupt:
        print("\n    interrupted")
    finally:
        for _ in range(5):
            try:
                master.mav.rc_channels_override_send(
                    master.target_system, master.target_component, *([0] * 8))
            except Exception:
                pass
            sleep(0.02)
        print("\n    sticks released back to the pilot")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--connect", default=None,
                    help="link to probe: /dev/ttyACM0, or udpin:127.0.0.1:14551 "
                         "for the router.  Default: auto-detect.")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--verify", action="store_true",
                    help="send a real override and prove the FC applied it "
                         "(safe while disarmed)")
    ap.add_argument("--wiggle", type=float, default=0.0, metavar="SECONDS",
                    help="sweep yaw/pitch and report the motor spread. PROPS OFF")
    ap.add_argument("--amplitude", type=float, default=0.35,
                    help="wiggle stick amplitude, 0..1 (default 0.35)")
    ap.add_argument("--yes", action="store_true",
                    help="skip the props-off confirmation for --wiggle")
    args = ap.parse_args()

    conn = resolve_device(args.connect)
    print("=" * 74)
    print("WASP flight-control link check")
    print(f"  link : {conn}")
    print("=" * 74)

    master, hb = probe_mavlink(conn, args.baud)

    if master is None:
        msp = probe_msp(conn if not is_network(conn) else resolve_device(None),
                        args.baud)
        print("\n" + "=" * 74)
        if msp:
            print(f"VERDICT: the board is running {msp['variant']} {msp['version']}, "
                  "NOT ArduPilot.")
            print("""
This is a Betaflight-family firmware.  It does not speak MAVLink, so the
tracker's MAVLink path can never drive it.  Two ways forward:

  A) RECOMMENDED — flash ArduPilot Copter (target TBS_LUCID_H7) and fly in
     ACRO.  You then get parameter access, telemetry, Mission Planner and the
     RC_CHANNELS_OVERRIDE path the tracker is built around.

  B) Stay on Betaflight and drive it over MSP.  The tracker supports this
     (FC_TRANSPORT=msp, pointed at the FC's serial port, not the router), but
     Betaflight ignores MSP sticks until you configure the override:
         feature MSP_OVERRIDE
         set msp_override_channels_mask = 10      # ch2 pitch + ch4 yaw
         save
     then assign "MSP RC OVERRIDE" to a spare aux switch in the Modes tab and
     turn it on.  Note MSP needs the serial port exclusively, so the MAVProxy
     router and Mission Planner cannot share it.""")
        else:
            print("VERDICT: nothing answered on this link — neither MAVLink nor MSP.")
            print("""
Check, in order:
  1. Is the FC powered and its USB cable seated?  ls /dev/serial/by-id/
  2. Is something else holding the port?  A running MAVProxy router owns the
     FC serial exclusively — either probe through it (--connect
     udpin:127.0.0.1:14551) or stop it and probe the port directly.
  3. Does the user have permission?  sudo usermod -aG dialout $USER, re-login.""")
        print("=" * 74)
        sys.exit(1)

    cfg = report_ardupilot(master, hb)

    ok = None
    if args.verify or args.wiggle > 0:
        # An override IS a stick input.  While disarmed that is inert, but on an
        # armed airframe it commands real rotation — so never do it silently.
        if cfg["armed"] and not args.yes:
            print("\n[3] override verification SKIPPED — the FC is ARMED, and the "
                  "test moves a stick\n    for 2.5 s per attempt.  Disarm and "
                  "re-run, or pass --yes with the props OFF.")
        else:
            ok = verify_override(master, cfg)

    if args.wiggle > 0:
        if ok is False:
            print("\n[4] wiggle skipped — the FC is not accepting overrides, so it "
                  "would do nothing.\n    Fix the verification failure above first.")
        elif not cfg["armed"]:
            print("\n[4] wiggle skipped — the FC is DISARMED, so the motors are off "
                  "and nothing will\n    move.  Arm from the transmitter in ACRO "
                  "(PROPS OFF) and run again.")
        else:
            if not args.yes:
                print("\n!!! The airframe is ARMED and this will command real "
                      "rotation !!!")
                if input("Type YES to confirm the props are OFF: ").strip() != "YES":
                    print("aborted")
                    master.close()
                    sys.exit(1)
            wiggle(master, cfg, seconds=args.wiggle, amplitude=args.amplitude)

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  firmware   : ArduPilot Copter")
    print(f"  mode       : {cfg['mode']}"
          + ("  (correct)" if cfg["mode"] == "ACRO" else "  <-- switch to ACRO"))
    print(f"  armed      : {'YES' if cfg['armed'] else 'NO   <-- motors are off'}")
    if ok is True:
        print("  overrides  : ACCEPTED — the control transport works")
        print("\n  The link is good.  If the drone still does not move, it is the")
        print("  throttle: in ACRO the mixer needs headroom, so the pilot must hold")
        print("  mid throttle.  Watch the motor spread with:  --wiggle 8")
    elif ok is False:
        print("  overrides  : REFUSED — see [3] above; the tracker cannot steer yet")
    else:
        print("  overrides  : not tested (re-run with --verify)")
    print("=" * 74)

    master.close()


if __name__ == "__main__":
    main()
