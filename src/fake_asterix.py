#!/usr/bin/env python3
"""
fake_asterix.py — EKETA-format ASTERIX CAT062 emitter for field testing.

Sends EKETA CAT062 UDP packets (with SP CH v1) at 2 Hz to the Jetson,
mimicking what the real EKETA-EAB radar system would send.  The Jetson's
AsterixCue picks them up and the SEEK mission flies to the target.

Examples:

    # Static target at a GPS point, 50 m above sea level
    python fake_asterix.py --target 37.80067,23.79301,50 --dest 192.168.1.100:8600

    # Circle around a point (200 m radius, 60 s period)
    python fake_asterix.py --circle 37.80067,23.79301,50 --radius 200 --period 60 \
        --dest 192.168.1.100:8600

    # Line between two points (back and forth)
    python fake_asterix.py --line 37.800,23.793,50,37.802,23.795,50 \
        --dest 192.168.1.100:8600

    # Multiple static targets
    python fake_asterix.py --target 37.800,23.793,50 --target 37.801,23.794,60 \
        --dest 192.168.1.100:8600

    # Send to localhost for testing
    python fake_asterix.py --target 37.80067,23.79301,50

The reference point (lat0, lon0, H0) for the SP field's polar coordinates
defaults to the first target's position.  Override with --ref lat,lon,alt.
"""
import argparse
import math
import socket
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, __file__ and __import__("os").path.dirname(__file__) or ".")
from asterix_cue import encode_eketa_cat062


def _polar(lat0, lon0, h0, lat, lon, h_msl):
    """SP CH v1 polar coordinates from reference point to target."""
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0) * math.cos(math.radians(lat0))
    north = dlat * 6371000.0
    east = dlon * 6371000.0
    d = math.hypot(north, east)
    hrel = h_msl - h0
    R = math.hypot(d, hrel)
    theta = math.degrees(math.atan2(east, north)) % 360.0
    phi = math.degrees(math.atan2(hrel, d)) if R > 0.01 else 0.0
    return dict(azimuth_deg=theta, elevation_deg=phi,
                slant_range_m=R, hrel_m=hrel)


def static_gen(points):
    while True:
        yield list(points)


def circle_gen(center, radius_m, period_s, h_msl):
    lat0, lon0 = center
    start = time.time()
    while True:
        a = 2 * math.pi * ((time.time() - start) / period_s)
        dlat = radius_m * math.cos(a) / 111320.0
        dlon = radius_m * math.sin(a) / (111320.0 * math.cos(math.radians(lat0)))
        vx = radius_m * (2 * math.pi / period_s) * math.cos(a)
        vy = -radius_m * (2 * math.pi / period_s) * math.sin(a)
        yield [dict(lat=lat0 + dlat, lon=lon0 + dlon, h_msl_m=h_msl,
                    vx=vx, vy=vy, track_no=1)]


def line_gen(p1, p2, speed_mps):
    dist = math.hypot((p2[0] - p1[0]) * 111320,
                      (p2[1] - p1[1]) * 111320 * math.cos(math.radians(p1[0])))
    if dist < 0.1:
        dist = 1.0
    period = dist / speed_mps * 2
    start = time.time()
    while True:
        t = ((time.time() - start) % period) / period
        frac = 2 * t if t < 0.5 else 2 * (1 - t)
        lat = p1[0] + frac * (p2[0] - p1[0])
        lon = p1[1] + frac * (p2[1] - p1[1])
        alt = p1[2] + frac * (p2[2] - p1[2])
        bearing = math.atan2((p2[1] - p1[1]) * math.cos(math.radians(p1[0])),
                             p2[0] - p1[0])
        sign = 1.0 if t < 0.5 else -1.0
        vx = sign * speed_mps * math.sin(bearing)
        vy = sign * speed_mps * math.cos(bearing)
        yield [dict(lat=lat, lon=lon, h_msl_m=alt, vx=vx, vy=vy, track_no=1)]


def main():
    p = argparse.ArgumentParser(
        description="Fake EKETA ASTERIX CAT062 emitter for field testing")
    p.add_argument("--target", action="append", metavar="LAT,LON,ALT_MSL",
                   help="static target (repeat for multiple); ALT is metres MSL")
    p.add_argument("--circle", metavar="LAT,LON,ALT_MSL",
                   help="target circling around this point")
    p.add_argument("--radius", type=float, default=200,
                   help="circle radius in metres (default 200)")
    p.add_argument("--period", type=float, default=60,
                   help="circle period in seconds (default 60)")
    p.add_argument("--line", metavar="LAT1,LON1,ALT1,LAT2,LON2,ALT2",
                   help="target moving between two points")
    p.add_argument("--speed", type=float, default=10,
                   help="line speed in m/s (default 10)")
    p.add_argument("--dest", default="127.0.0.1:8600",
                   help="Jetson IP:port (default 127.0.0.1:8600)")
    p.add_argument("--hz", type=float, default=2.0,
                   help="send rate per track (default 2.0)")
    p.add_argument("--sac", type=int, default=0)
    p.add_argument("--sic", type=int, default=1)
    p.add_argument("--ref", metavar="LAT,LON,ALT",
                   help="SP reference point (default: first target)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if not args.target and not args.circle and not args.line:
        p.error("give --target, --circle or --line")

    host, _, port = args.dest.rpartition(":")
    port = int(port)
    host = host or "127.0.0.1"

    ref = None
    if args.ref:
        parts = args.ref.split(",")
        ref = (float(parts[0]), float(parts[1]), float(parts[2]))

    if args.target:
        points = []
        for i, t in enumerate(args.target, 1):
            parts = t.split(",")
            lat, lon, alt = float(parts[0]), float(parts[1]), float(parts[2])
            points.append(dict(lat=lat, lon=lon, h_msl_m=alt, vx=0.0, vy=0.0,
                               track_no=i))
        if ref is None:
            ref = (points[0]["lat"], points[0]["lon"], points[0]["h_msl_m"])
        gen = static_gen(points)
    elif args.circle:
        parts = args.circle.split(",")
        clat, clon, calt = float(parts[0]), float(parts[1]), float(parts[2])
        if ref is None:
            ref = (clat, clon, calt)
        gen = circle_gen((clat, clon), args.radius, args.period, calt)
    elif args.line:
        parts = args.line.split(",")
        p1 = (float(parts[0]), float(parts[1]), float(parts[2]))
        p2 = (float(parts[3]), float(parts[4]), float(parts[5]))
        if ref is None:
            ref = (p1[0], p1[1], p1[2])
        gen = line_gen(p1, p2, args.speed)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[EKETA-FAKE] Sending CAT062 to {host}:{port} at {args.hz} Hz")
    print(f"[EKETA-FAKE] Reference point: {ref[0]:.6f}, {ref[1]:.6f}, {ref[2]:.1f} m MSL")
    print(f"[EKETA-FAKE] Press Ctrl+C to stop")

    n = 0
    try:
        for track_list in gen:
            for t in track_list:
                polar = _polar(ref[0], ref[1], ref[2],
                               t["lat"], t["lon"], t["h_msl_m"])
                t.update(polar)
                t.setdefault("classif", "unknown")
            data = encode_eketa_cat062(track_list, sac=args.sac, sic=args.sic)
            sock.sendto(data, (host, port))
            n += 1
            if not args.quiet and n % max(1, int(args.hz)) == 0:
                t = track_list[0]
                print(f"  #{n}: TRK{t['track_no']} {t['lat']:.6f},{t['lon']:.6f} "
                      f"alt {t['h_msl_m']:.1f}m  "
                      f"v=({t.get('vx', 0):.1f},{t.get('vy', 0):.1f}) m/s")
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        print(f"\n[EKETA-FAKE] Stopped after {n} packets.")


if __name__ == "__main__":
    main()
