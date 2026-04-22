"""
tracker2.py  –  Lightweight Drone Tracker
==========================================
Pipeline per frame:
  1. Kalman predict  (~0.01 ms, always)
  2. YOLO on velocity-adaptive ROI (LOCK/PREDICT) or full frame (SEARCH)
  3. YOLO miss → Kalman holds position, ROI expands with |v| × missed_frames

States:
  SEARCH   – No target.  Full-frame YOLO scan, Kalman reset.
  LOCK     – YOLO hitting. Tight ROI centered on detection.
  PREDICT  – YOLO missing. Kalman predicts position + direction.
             ROI grows with velocity magnitude × time lost.
             PID gain reduced so we don't over-command on stale position.
"""

import cv2
import numpy as np
from time import perf_counter
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
# PID
# ══════════════════════════════════════════════════════════════════════════════

class PID:
    def __init__(self, kp=0.1, ki=0.002, kd=0.03):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.ix = self.iy = self.px = self.py = 0.0

    def reset(self):
        self.ix = self.iy = self.px = self.py = 0.0

    def compute(self, ex, ey):
        self.ix += ex;  self.iy += ey
        ox = self.kp * ex + self.ki * self.ix + self.kd * (ex - self.px)
        oy = self.kp * ey + self.ki * self.iy + self.kd * (ey - self.py)
        self.px, self.py = ex, ey
        return ox, oy


# ══════════════════════════════════════════════════════════════════════════════
# KALMAN  (4-state: x, y, vx, vy – constant velocity)
# ══════════════════════════════════════════════════════════════════════════════

class KalmanTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32) * 10.0
        self.initialized = False

    def init(self, x, y):
        s = np.array([[float(x)], [float(y)], [0.], [0.]], np.float32)
        self.kf.statePre = self.kf.statePost = s.copy()
        self.initialized = True

    def predict(self):
        p = self.kf.predict()
        return int(p[0]), int(p[1])

    def update(self, x, y):
        if not self.initialized:
            self.init(x, y)
            return
        self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))

    def velocity(self):
        if not self.initialized:
            return 0.0, 0.0
        return float(self.kf.statePost[2]), float(self.kf.statePost[3])

    def pos_sigma(self):
        """Position uncertainty (1-sigma) from error covariance diagonal."""
        if not self.initialized:
            return 80.0, 80.0
        P = self.kf.errorCovPost
        return float(np.sqrt(abs(float(P[0, 0])))), float(np.sqrt(abs(float(P[1, 1]))))


# ══════════════════════════════════════════════════════════════════════════════
# DRONE TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class DroneTracker:
    _WIN = "INTERCEPTOR DRONE TRACKER v2"
    _COLOR = {
        "SEARCH":  (0,   0,   255),   # red
        "LOCK":    (0,   255, 0),     # green
        "PREDICT": (0,   165, 255),   # orange
    }

    def __init__(
        self,
        model_path   = "./runs/detect/v0/weights/best.pt",
        video_path   = "./data/videos_drones/10.mp4",
        conf_thresh  = 0.05,
        display_size = (640, 640),
        skip_frames  = 0,
    ):
        """
        skip_frames:
          0 → every frame goes through YOLO (default)
          N → YOLO runs every (N+1)th frame; skipped frames use Kalman-only prediction
        Examples:
          skip_frames=1  →  YOLO on frames 0,2,4,...  (every other frame)
          skip_frames=2  →  YOLO on frames 0,3,6,...
        """
        self.model        = YOLO(model_path)
        self.video_path   = video_path
        self.conf_thresh  = conf_thresh
        self.display_size = display_size
        self.fw, self.fh  = display_size
        self.skip_frames  = max(0, int(skip_frames))

        self.pid    = PID()
        self.kalman = KalmanTracker()

        self.state     = "SEARCH"
        self.missed    = 0
        self.last_bbox = None   # (x1, y1, w, h) from last YOLO detection
        self.pos       = None
        self.layer     = "—"
        self._frame_idx = 0     # global frame counter for skip logic

        # ── Thresholds ───────────────────────────────────────────────────────
        self.MISS_TO_PREDICT = 2    # YOLO misses before entering PREDICT
        self.MISS_TO_SEARCH  = 20   # YOLO misses before full reset → SEARCH
        self.MIN_ROI_HALF    = 50   # smallest ROI half-size in px
        self.PREDICT_GAIN    = 0.4  # PID gain scale while predicting (avoid overreaction)

    # ──────────────────────────────────────────────────────────────────────────
    # YOLO
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_yolo(self, frame, roi=None):
        """Run YOLO on full frame or sub-ROI. Returns (cx,cy,x1,y1,w,h,conf) or None."""
        ox = oy = 0
        if roi is not None:
            x1r, y1r, x2r, y2r = roi
            crop = frame[y1r:y2r, x1r:x2r]
            if crop.size == 0:
                return None
            results = self.model(crop, verbose=False)
            ox, oy  = x1r, y1r
        else:
            results = self.model(frame, verbose=False)

        if not results or not results[0].boxes:
            return None

        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()
        best  = int(np.argmax(confs))
        if confs[best] < self.conf_thresh:
            return None

        x1, y1, x2, y2 = boxes.xyxy[best].int().cpu().tolist()
        x1 += ox;  y1 += oy;  x2 += ox;  y2 += oy
        return (x1 + x2) // 2, (y1 + y2) // 2, x1, y1, x2 - x1, y2 - y1, float(confs[best])

    # ──────────────────────────────────────────────────────────────────────────
    # VELOCITY-ADAPTIVE ROI
    # ──────────────────────────────────────────────────────────────────────────

    def _predict_roi(self):
        """
        Build the YOLO search window from Kalman state.
        Centre = Kalman predicted position (already velocity-extrapolated).
        Half-size grows with |velocity| × missed_frames so a fast-moving
        target never escapes the search window just because YOLO had a gap.
        """
        px, py = self.kalman.predict()
        vx, vy = self.kalman.velocity()
        bw, bh = (self.last_bbox[2], self.last_bbox[3]) if self.last_bbox else (80, 80)

        base = max(self.MIN_ROI_HALF, int(max(bw, bh) * 0.6))
        hw   = min(base + int(abs(vx) * self.missed * 1.5), self.fw // 2)
        hh   = min(base + int(abs(vy) * self.missed * 1.5), self.fh // 2)

        return (
            max(0, px - hw),   max(0, py - hh),
            min(self.fw, px + hw), min(self.fh, py + hh),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # FRAME PROCESSING
    # ──────────────────────────────────────────────────────────────────────────

    def _process(self, frame):
        # Kalman predict — nanoseconds
        t0 = perf_counter()
        kf_pred = self.kalman.predict() if self.kalman.initialized else None
        t_kf = (perf_counter() - t0) * 1000

        # YOLO — dominant cost
        t0      = perf_counter()
        roi     = None
        yolo_det = None

        if self.state in ("LOCK", "PREDICT") and kf_pred:
            roi      = self._predict_roi()
            yolo_det = self._detect_yolo(frame, roi=roi)
            # Full-frame fallback only when ROI hasn't already covered the frame
            if yolo_det is None:
                rw = roi[2] - roi[0];  rh = roi[3] - roi[1]
                if rw < self.fw * 0.85 or rh < self.fh * 0.85:
                    yolo_det = self._detect_yolo(frame)
                    roi = None  # full frame was used
        else:
            yolo_det = self._detect_yolo(frame)

        t_yolo = (perf_counter() - t0) * 1000

        # ── Fuse ────────────────────────────────────────────────────────────
        if yolo_det:
            cx, cy, x1, y1, w, h, conf = yolo_det
            self.kalman.update(cx, cy)
            self.last_bbox = (x1, y1, w, h)
            self.missed    = 0
            self.state     = "LOCK"
            fused          = (cx, cy)
            self.layer     = f"YOLO {conf:.2f}"

        elif kf_pred and self.state != "SEARCH":
            self.missed += 1
            if self.missed >= self.MISS_TO_SEARCH:
                self._reset()
                fused      = None
                self.layer = "LOST"
            else:
                self.state = "PREDICT" if self.missed >= self.MISS_TO_PREDICT else self.state
                fused      = kf_pred
                self.layer = f"KF  miss={self.missed}"

        else:
            self.missed += 1
            fused      = None
            self.layer = "LOST"

        self.pos = fused
        return fused, t_kf, t_yolo, roi

    def _reset(self):
        self.state     = "SEARCH"
        self.missed    = 0
        self.last_bbox = None
        self.kalman    = KalmanTracker()
        self.pid.reset()
        self.layer     = "—"

    # ──────────────────────────────────────────────────────────────────────────
    # VISUALISATION
    # ──────────────────────────────────────────────────────────────────────────

    def _draw(self, frame, pos, roi):
        color        = self._COLOR.get(self.state, (255, 255, 255))
        h, w         = frame.shape[:2]
        cx_f, cy_f   = w // 2, h // 2

        # Frame-centre reticle
        cv2.drawMarker(frame, (cx_f, cy_f), (80, 80, 80),
                       cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)

        # Search ROI box during PREDICT (shows how wide we're looking)
        if roi and self.state == "PREDICT":
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]),
                          (60, 60, 200), 1)

        if pos:
            px, py = int(pos[0]), int(pos[1])

            # Line from frame centre to target
            cv2.line(frame, (cx_f, cy_f), (px, py), color, 1, cv2.LINE_AA)

            # Last YOLO bounding box (dimmed during PREDICT = stale)
            if self.last_bbox:
                bx, by, bw, bh = self.last_bbox
                box_col = color if self.state == "LOCK" else (80, 80, 180)
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_col, 2)

            # Target crosshair
            cv2.drawMarker(frame, (px, py), color,
                           cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)

            # Velocity arrow (shows predicted direction)
            vx, vy = self.kalman.velocity()
            if np.hypot(vx, vy) > 0.5:
                tip = (int(px + vx * 8), int(py + vy * 8))
                cv2.arrowedLine(frame, (px, py), tip,
                                (0, 255, 255), 2, tipLength=0.4, line_type=cv2.LINE_AA)

            # PREDICT: draw 3-sigma uncertainty ellipse from Kalman covariance
            if self.state == "PREDICT":
                sx, sy = self.kalman.pos_sigma()
                ax = max(10, int(sx * 3))
                ay = max(10, int(sy * 3))
                cv2.ellipse(frame, (px, py), (ax, ay), 0, 0, 360,
                            (80, 80, 255), 1, cv2.LINE_AA)

        # HUD — state + active layer, compact
        cv2.putText(frame, self.state, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, self.layer, (12, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {self.video_path}")

        cv2.namedWindow(self._WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._WIN, *self.display_size)

        period = self.skip_frames + 1   # e.g. skip=2 → run YOLO every 3rd frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame      = cv2.resize(frame, self.display_size)
            frame      = np.ascontiguousarray(frame)
            h, w       = frame.shape[:2]
            cx_f, cy_f = w // 2, h // 2

            self._frame_idx += 1
            is_yolo_frame = (self._frame_idx % period == 0)

            if is_yolo_frame:
                # Full pipeline: YOLO + Kalman update + state machine
                fused, t_kf, t_yolo, roi = self._process(frame)
                skip_tag = ""
            else:
                # Skipped frame: Kalman-only prediction, no YOLO, state frozen
                t0    = perf_counter()
                fused = self.kalman.predict() if self.kalman.initialized else None
                t_kf  = (perf_counter() - t0) * 1000
                t_yolo = 0.0
                roi    = None
                if fused:
                    self.pos   = fused
                    self.layer = "KF-SKIP"
                skip_tag = " [SKIP]"

            # PID — reduce gain in PREDICT / skipped frames to avoid overcorrecting
            if fused:
                ex, ey = fused[0] - cx_f, fused[1] - cy_f
                gain   = self.PREDICT_GAIN if (self.state == "PREDICT" or not is_yolo_frame) else 1.0
                cmd_x, cmd_y = self.pid.compute(ex * gain, ey * gain)
                cmd_x = float(np.clip(cmd_x, -25, 25))
                cmd_y = float(np.clip(cmd_y, -25, 25))
            else:
                cmd_x = cmd_y = 0.0
                self.pid.reset()

            self._draw(frame, fused, roi)

            print(f"YOLO:{t_yolo:5.1f}ms  KF:{t_kf:.3f}ms"
                  f"  | total:{t_yolo + t_kf:5.1f}ms"
                  f"  | {self.state:7s} | {self.layer}{skip_tag}")

            cv2.imshow(self._WIN, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    DroneTracker(
        model_path   = "./runs/detect/v0/weights/best.pt",
        video_path   = "./data/videos_drones/10.mp4",
        conf_thresh  = 0.05,
        display_size = (640, 640),
        skip_frames  = 3,   # 0=every frame  1=every 2nd  2=every 3rd ...
    ).run()
