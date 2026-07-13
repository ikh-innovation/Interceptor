import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wasp_tracker import Config, TrackingPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        prog="wasp-tracker",
        description="WASP sky-gated adaptive-zoom drone interceptor tracker.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")),
                        help="Path to the YAML config file.")
    parser.add_argument("--source",
                        help="Video file, camera index (e.g. 0), /dev/videoN, "
                             "rtsp:// stream, or a GStreamer pipeline string.")
    parser.add_argument("--model", help="Path to the YOLO weights.")
    parser.add_argument("--conf", type=float, dest="conf_thresh",
                        help="Confidence acceptance threshold.")
    parser.add_argument("--skip-frames", type=int, dest="skip_frames",
                        help="Frames skipped between YOLO passes (detection rate = fps/(skip+1)).")
    parser.add_argument("--max-frames", type=int, dest="max_frames",
                        help="Stop after this many frames.")
    parser.add_argument("--mav-device", dest="mav_device",
                        help="MAVLink device, e.g. /dev/ttyACM0. Unset = offline print-only.")
    parser.add_argument("--mav-baud", type=int, dest="mav_baud", help="MAVLink baud rate.")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable all visualization (fastest; Jetson deployment).")
    parser.add_argument("--no-window", action="store_true",
                        help="Keep the HUD but do not open an imshow window.")
    parser.add_argument("--no-log", action="store_true", help="Disable all logging.")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable per-frame logging; keep the end-of-stream summary.")
    parser.add_argument("--log-level", dest="log_level", help="Logging level (INFO, DEBUG, ...).")
    return parser.parse_args()


def build_overrides(args):
    overrides = {
        "source": args.source,
        "model": args.model,
        "conf_thresh": args.conf_thresh,
        "skip_frames": args.skip_frames,
        "max_frames": args.max_frames,
        "mav_device": args.mav_device,
        "mav_baud": args.mav_baud,
        "log_level": args.log_level,
    }
    if args.no_viz:
        overrides["viz_enabled"] = False
    if args.no_window:
        overrides["show_window"] = False
    if args.no_log:
        overrides["log_enabled"] = False
    if args.quiet:
        overrides["log_per_frame"] = False
    return overrides


def main():
    args = parse_args()
    config = Config.load(args.config, build_overrides(args))
    TrackingPipeline(config).run()


if __name__ == "__main__":
    main()
