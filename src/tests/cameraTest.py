
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

SENSOR_ID   = int(os.getenv("CAM_SENSOR_ID", "0"))
SENSOR_MODE = int(os.getenv("CAM_SENSOR_MODE", "3"))
WIDTH       = int(os.getenv("CAM_WIDTH", "1280"))
HEIGHT      = int(os.getenv("CAM_HEIGHT", "960"))
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
        pass

    pipeline = build_pipeline()


    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():










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


            writer = None
        else:
            pass



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
                if failures >= 30:


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
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    if frames == 0:




        return 1



    return 0


if __name__ == "__main__":
    sys.exit(main())
