#!/usr/bin/env python3
"""Send a live GPS target to the Jetson over the RFD serial link.

    python rfd_send.py --dev /dev/ttyUSB0 --lat 37.98 --lon 23.72 --alt 5

The newest fix is repeated until the process stops. A line on stdin,
``lat,lon,alt`` or ``lat,lon,alt,REL|MSL``, replaces it.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

from rfd_gps import format_line


def main(argv=None):
    p = argparse.ArgumentParser(description="Send a live RFD GPS target")
    p.add_argument("--dev", required=True, help="serial device, for example /dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=57600)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--alt", type=float, required=True, help="metres, REL above home or MSL")
    p.add_argument("--frame", default="REL", choices=("REL", "MSL"))
    p.add_argument("--hz", type=float, default=5.0)
    args = p.parse_args(argv)

    try:
        import serial
    except ImportError:
        print("pyserial is not installed", file=sys.stderr)
        return 1
    try:
        port = serial.Serial(args.dev, args.baud, timeout=0.2)
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1

    state = {"lat": args.lat, "lon": args.lon, "alt": args.alt, "frame": args.frame}
    lock = threading.Lock()
    stop = threading.Event()

    def read_stdin():
        for line in sys.stdin:
            parts = [x.strip() for x in line.replace(" ", "").split(",") if x.strip()]
            if len(parts) < 3:
                continue
            try:
                lat, lon, alt = float(parts[0]), float(parts[1]), float(parts[2])
                frame = parts[3].upper() if len(parts) > 3 else state["frame"]
                format_line(lat, lon, alt, frame)
            except ValueError as exc:
                print(exc, file=sys.stderr)
                continue
            with lock:
                state.update(lat=lat, lon=lon, alt=alt, frame=frame)

    threading.Thread(target=read_stdin, daemon=True).start()
    period = 1.0 / max(0.5, args.hz)
    try:
        while not stop.is_set():
            with lock:
                line = format_line(state["lat"], state["lon"], state["alt"], state["frame"])
            port.write((line + "\r\n").encode("ascii"))
            port.flush()
            time.sleep(period)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        try:
            port.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
