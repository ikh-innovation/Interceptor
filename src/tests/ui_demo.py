#!/usr/bin/env python3
"""The phone app on a desk: no drone, no FC, no camera.

    ../.env/yolo/bin/python tests/ui_demo.py [--port 8090]

Open the printed link, then ☰ → TEST → START TEST: the same test mode the
Jetson serves (desk_sim.py: the real mission code against a model of
ArduPilot's yaw hold and gusty wind).  The "real" drone here is a stand-in
that is never armed, so the test mode is always allowed.
"""
import argparse
import contextlib
import io
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

with contextlib.redirect_stdout(io.StringIO()):
    import mjpeg_stream                                            # noqa: E402

IDLE = dict(linked=False, armed=False, fc_mode=None, detector="TensorRT", fps=0.0, lat_ms=None,
            tilt_deg=30.0, target="DRONE", console_seq=0,
            mission=dict(app="mission", phase="PILOT", gate_why="desk: no FC", levels={}, speeds={},
                         locked=False, lock_hdg=None, engaged=False, ai_steering=False,
                         height_m=None, climb_mps=None, heading_deg=None, vibe=None,
                         det_age_s=None, det_conf=None, target_fresh=False))


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    srv = mjpeg_stream.MJPEGServer(port=a.port, state_get=lambda: dict(IDLE),
                                   command=lambda k, v: "desk: no FC -- use ☰ → TEST",
                                   console_get=lambda since: []).start()
    print(f"\n  phone app on the desk:\n    http://{lan_ip()}:{a.port}/\n"
          f"    http://127.0.0.1:{a.port}/\n  then ☰ → TEST.  Ctrl-C stops.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.stop()


if __name__ == "__main__":
    main()
