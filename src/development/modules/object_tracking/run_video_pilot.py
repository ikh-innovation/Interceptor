#!/usr/bin/env python3
"""
run_video_pilot.py  --  WASP tracker on a recorded clip, steering the pilot (FC)

Runs tracker.py's DroneTracker against a video file and, when --connect is
given, opens the SAME MAVLink link controller_testing.py validates (device +
baud) so the flight controller ("the pilot") receives the live GUIDED
velocity/yaw-rate setpoints computed from the video. Without --connect the
tracker stays offline: setpoints are computed and printed every frame, same
as controller_testing.py's read-only default, but nothing is sent.

Validate the link first with:
    python3 ../../utils/controller_testing.py --connect /dev/ttyACM0 --baud 115200

Usage:
    # Safe default: video only, commands computed + printed, nothing sent
    python3 run_video_pilot.py

    # Live: also steer the FC on /dev/ttyACM0 (asks for confirmation first)
    python3 run_video_pilot.py --connect /dev/ttyACM0 --baud 115200
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracker import DroneTracker

REPO_ROOT     = Path(__file__).resolve().parents[4]
DEFAULT_VIDEO = REPO_ROOT / "data" / "ClipExport.mp4"
DEFAULT_MODEL = REPO_ROOT / "best.engine"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=str(DEFAULT_VIDEO),
                    help="Input video (default: data/ClipExport.mp4)")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO weights path")
    p.add_argument("--connect", default=None,
                    help="MAVLink link to the pilot/FC, e.g. /dev/ttyACM0 (same "
                         "device/convention as controller_testing.py --connect). "
                         "Omit to stay offline: commands are computed and printed "
                         "but nothing is sent.")
    p.add_argument("--baud", type=int, default=115200, help="Baud rate for --connect")
    p.add_argument("--conf-thresh", type=float, default=0.05)
    p.add_argument("--skip-frames", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--headless", action="store_true", help="No display window")
    p.add_argument("--record-dir", default=None,
                    help="If set, save detection-triggered clips here")
    p.add_argument("--yes", action="store_true",
                    help="Skip the live-link confirmation prompt (non-interactive runs)")
    return p.parse_args()


def main():
    args = parse_args()
    live = args.connect is not None

    print("=" * 78)
    print("WASP TRACKER  --  video input -> pilot (MAVLink) output")
    print(f"  video : {args.video}")
    print(f"  model : {args.model}")
    print(f"  pilot : "
          + (f"LIVE {args.connect} @ {args.baud} baud" if live
             else "offline (print-only, nothing sent)"))
    print("=" * 78)

    if live and not args.yes:
        print("\n!!! LIVE MAVLINK LINK REQUESTED !!!")
        print("If the FC is armed/GUIDED-capable this WILL drive it (yaw/climb/forward).")
        confirm = input("Type 'YES' to confirm props are OFF / vehicle is safe to test: ")
        if confirm.strip() != "YES":
            print("Aborting -- confirmation not given.")
            sys.exit(1)

    DroneTracker(
        model_path   = args.model,
        video_path   = args.video,
        conf_thresh  = args.conf_thresh,
        skip_frames  = args.skip_frames,
        max_frames   = args.max_frames,
        headless     = args.headless,
        record_dir   = args.record_dir,
        mav_device   = args.connect,
        mav_baud     = args.baud,
    ).run()


if __name__ == "__main__":
    main()
