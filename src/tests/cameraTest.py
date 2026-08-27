"""Live camera check for the Jetson CSI camera (IMX219).

Standalone on purpose: imports nothing from the project, so it runs on the
system python as well as the yolo env, and needs neither the flight controller
nor the YOLO weights.  This is the "is the camera alive" test, nothing more.

    python3 tests/cameraTest.py              # live window, q to quit
    python3 tests/cameraTest.py --record     # also write data/cameratest_<stamp>.mp4
    python3 tests/cameraTest.py --headless   # no window, just frame stats

Environment overrides: CAM_SENSOR_ID, CAM_SENSOR_MODE, CAM_WIDTH, CAM_HEIGHT,
CAM_FPS, CAM_FLIP_METHOD  (same names tracker.py uses).
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

SENSOR_ID   = int(os.getenv("CAM_SENSOR_ID", "0"))
# IMX219 modes: 0:3280x2464@21  1:3280x1848@28  2:1920x1080@30
#               3:1640x1232@30  4:1280x720@60
# Mode 4 matches the 720p defaults below; -1 lets Argus pick from the caps.
SENSOR_MODE = int(os.getenv("CAM_SENSOR_MODE", "4"))
WIDTH       = int(os.getenv("CAM_WIDTH", "1280"))
HEIGHT      = int(os.getenv("CAM_HEIGHT", "720"))
FPS         = int(os.getenv("CAM_FPS", "30"))
FLIP        = int(os.getenv("CAM_FLIP_METHOD", "0"))


def build_pipeline() -> str:
    mode = f"sensor-mode={SENSOR_MODE} " if SENSOR_MODE >= 0 else ""
    return (
        f"nvarguscamerasrc sensor-id={SENSOR_ID} {mode}! "
        f"video/x-raw(memory:NVMM), width={WIDTH}, height={HEIGHT}, "
        f"framerate={FPS}/1 ! "
        f"nvvidconv flip-method={FLIP} ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Live CSI camera check")
    ap.add_argument("--record", action="store_true",
                    help="also write an mp4 next to the other run data")
    ap.add_argument("--headless", action="store_true",
                    help="no window; print frame stats only")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop after N seconds (0 = run until 'q'/Ctrl-C)")
    args = ap.parse_args()

    headless = args.headless or not os.environ.get("DISPLAY")
    if args.headless is False and headless:
        print("[WARN] no DISPLAY set — falling back to headless.", file=sys.stderr)

    pipeline = build_pipeline()
    print(f"[CAP] {pipeline}", flush=True)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(
            "\n[FAIL] could not open the camera.\n"
            "  * is the CSI ribbon seated (blue side toward the fan)?\n"
            "  * does /dev/video0 exist, and does\n"
            "      cat /sys/class/video4linux/video0/name\n"
            "    name your sensor?\n"
            "  * is the Argus daemon up:  systemctl status nvargus-daemon\n"
            f"  * does this mode exist:    CAM_SENSOR_MODE={SENSOR_MODE} "
            f"at {WIDTH}x{HEIGHT}@{FPS}\n",
            file=sys.stderr)
        return 1

    writer = None
    if args.record:
        data_dir = Path(__file__).resolve().parent.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = data_dir / f"cameratest_{stamp}.mp4"
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(FPS), (WIDTH, HEIGHT))
        if not writer.isOpened():
            print(f"[WARN] could not open {out_path} for writing — "
                  "continuing without recording.", file=sys.stderr)
            writer = None
        else:
            print(f"[REC] writing {out_path}", flush=True)

    print("Press 'q' in the window to stop (Ctrl-C works too).", flush=True)

    frames = 0
    failures = 0
    t0 = time.time()
    fps_live = 0.0
    last_report = t0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                # nvarguscamerasrc drops the odd buffer; only a run of failures
                # means the camera is actually gone.
                if failures >= 30:
                    print("\n[FAIL] 30 consecutive read failures — camera lost.",
                          file=sys.stderr)
                    break
                continue
            failures = 0
            frames += 1

            now = time.time()
            elapsed = now - t0
            if elapsed > 0:
                fps_live = frames / elapsed

            if writer is not None:
                writer.write(frame)

            if headless:
                if now - last_report >= 1.0:
                    print(f"  {frames:6d} frames  {fps_live:5.1f} fps  "
                          f"mean px {frame.mean():5.1f}", flush=True)
                    last_report = now
            else:
                hud = frame.copy()
                for text, org in (
                    (f"{WIDTH}x{HEIGHT} mode {SENSOR_MODE}", (12, 30)),
                    (f"{fps_live:.1f} fps   frame {frames}", (12, 60)),
                    ("q = quit", (12, 90)),
                ):
                    cv2.putText(hud, text, (org[0] + 1, org[1] + 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3,
                                cv2.LINE_AA)
                    cv2.putText(hud, text, org,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1,
                                cv2.LINE_AA)
                cv2.imshow("camera check", hud)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.seconds and elapsed >= args.seconds:
                break
    except KeyboardInterrupt:
        print("\n[stop] Ctrl-C", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    if frames == 0:
        print("\n[FAIL] the pipeline opened but delivered no frames.\n"
              "  Argus sometimes fails to map the first buffers; just retry.\n"
              "  If it never works:  sudo systemctl restart nvargus-daemon\n",
              file=sys.stderr)
        return 1

    print(f"\n[OK] {frames} frames in {elapsed:.1f}s "
          f"({frames / elapsed:.1f} fps average)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
