#!/usr/bin/env python3

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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



def resolve_device(explicit=None):
    if explicit:
        return explicit
    if router_running():


        return DEFAULT_ROUTER
    for pat in ("usb-ArduPilot_*-if00", "*-if00", "*"):
        matches = sorted(Path("/dev/serial/by-id").glob(pat)) \
            if Path("/dev/serial/by-id").is_dir() else []
        if matches:

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



def probe_mavlink(conn, baud, timeout=12.0):
    if mavutil is None:

        return None, None

    try:
        master = mavutil.mavlink_connection(conn, baud=baud, source_system=255,
                                            source_component=190)
    except Exception as e:

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


        return master, msg

    if saw_gcs:
        pass



    else:
        pass
    master.close()
    return None, None


def probe_msp(dev, baud=115200):
    if is_network(dev):

        return None

    try:
        import serial
    except ImportError:

        return None
    try:
        ser = serial.Serial(dev, baud, timeout=0.3)
    except Exception as e:

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

        ser.close()
        return None
    name = variant[:4].decode(errors="ignore")
    ver = request(MSP_FC_VERSION) or b"\0\0\0"
    vstr = f"{ver[0]}.{ver[1]}.{ver[2]}" if len(ver) >= 3 else "?"

    status = request(MSP_STATUS)
    armed = None
    if status and len(status) >= 11:
        import struct
        armed = bool(struct.unpack_from("<I", status, 6)[0] & 1)

    rc = request(MSP_RC)
    if rc:
        import struct
        n = len(rc) // 2
        vals = struct.unpack_from("<" + "H" * n, rc, 0)


    ser.close()
    return dict(variant=name, version=vstr, armed=armed)



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

    mode = COPTER_MODES.get(hb.custom_mode, f"MODE{hb.custom_mode}")
    armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)




    sysid = read_param(master, "SYSID_MYGCS")
    pname = "SYSID_MYGCS"
    if sysid is None:
        sysid = read_param(master, "MAV_GCS_SYSID")
        pname = "MAV_GCS_SYSID"
    if sysid is None:
        pass
    else:
        pass



    rc_options = read_param(master, "RC_OPTIONS")
    if rc_options is not None:
        bad = int(rc_options) & 2



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
            pass


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




    request_rc_stream(master)
    rc = master.recv_match(type="RC_CHANNELS", blocking=True, timeout=4)
    if rc is None:
        pass

    else:
        vals = [getattr(rc, f"chan{i}_raw", 0) for i in range(1, 9)]

        if all(v == 0 for v in vals):
            pass


    return dict(mode=mode, armed=armed, sysid_mygcs=sysid, axes=axes,
                rc_options=rc_options)



def verify_override(master, cfg, hold=2.5):

    yaw = cfg["axes"]["yaw"]
    chan = yaw["chan"]
    if not 1 <= chan <= 8:


        return False

    test_pwm = int(min(yaw["max"], yaw["trim"] + 200))
    candidates = []
    if cfg["sysid_mygcs"] is not None:
        candidates.append(int(cfg["sysid_mygcs"]))
    for s in (255, 1):
        if s not in candidates:
            candidates.append(s)

    for sysid in candidates:
        master.mav.srcSystem = sysid


        chans = [0] * 8
        chans[chan - 1] = test_pwm

        deadline = perf_counter() + hold
        best = None
        matched = False
        next_tx = 0.0
        while perf_counter() < deadline:
            now = perf_counter()
            if now >= next_tx:
                next_tx = now + 0.05
                try:
                    master.mav.rc_channels_override_send(
                        master.target_system, master.target_component, *chans)
                except Exception as e:

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

        try:
            master.mav.rc_channels_override_send(
                master.target_system, master.target_component, *([0] * 8))
        except Exception:
            pass

        if matched:


            if cfg["sysid_mygcs"] is not None and sysid != int(cfg["sysid_mygcs"]):
                pass

            else:
                pass

            return True










    return False



def wiggle(master, cfg, seconds=8.0, amplitude=0.35, period=3.0):







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




            m = master.recv_match(type=["SERVO_OUTPUT_RAW", "HEARTBEAT"],
                                  blocking=False)
            if m is not None and m.get_type() == "SERVO_OUTPUT_RAW":
                servo = m
            sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(5):
            try:
                master.mav.rc_channels_override_send(
                    master.target_system, master.target_component, *([0] * 8))
            except Exception:
                pass
            sleep(0.02)




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





    master, hb = probe_mavlink(conn, args.baud)

    if master is None:
        msp = probe_msp(conn if not is_network(conn) else resolve_device(None),
                        args.baud)

        if msp:
            pass


















        else:
            pass








        sys.exit(1)

    cfg = report_ardupilot(master, hb)

    ok = None
    if args.verify or args.wiggle > 0:
        if cfg["armed"] and not args.yes:
            pass


        else:
            ok = verify_override(master, cfg)

    if args.wiggle > 0:
        if ok is False:
            pass

        elif not cfg["armed"]:
            pass


        else:
            if not args.yes:


                if input("Type YES to confirm the props are OFF: ").strip() != "YES":

                    master.close()
                    sys.exit(1)
            wiggle(master, cfg, seconds=args.wiggle, amplitude=args.amplitude)








    if ok is True:
        pass



    elif ok is False:
        pass
    else:
        pass


    master.close()


if __name__ == "__main__":
    main()
