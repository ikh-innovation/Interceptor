"""
tracker_skip.py  –  Drone Intercept Tracker  (no PID)
=======================================================
Pipeline per frame:
  1. Kalman predict  (~0.01 ms, always)
  2. YOLO on velocity-adaptive ROI (LOCK/PREDICT) or full frame (SEARCH)
  3. YOLO miss → Kalman holds + predicts re-acquisition direction
  4. PX4 OFFBOARD body-frame commands derived from pixel error + Kalman state

States:
  SEARCH   – No target.  Full-frame YOLO, Kalman reset.
  LOCK     – YOLO active. Tight ROI.
  PREDICT  – YOLO missed. Kalman predicts where to look next.

Visuals:
  • Turn-direction arrow  — from frame centre toward target, tells drone 1 where to yaw
  • Target velocity arrow — at predicted position, shows where drone 2 is going
  • Uncertainty ellipse   — 3-sigma Kalman covariance (PREDICT only)
  • ROI box               — active YOLO search window (PREDICT only)

PX4 OFFBOARD output (MAVLink SET_POSITION_TARGET_LOCAL_NED, BODY_NED frame):
  fwd   →  vx body frame (m/s)      approach speed
  lat   →  vy body frame (m/s)      always 0 (yaw handles horizontal)
  vz    →  vz NED (m/s, +DOWN)      from image vertical error
  yaw_r →  yaw_rate (rad/s, +CW)   from image horizontal error
"""

import cv2
import numpy as np
from time import perf_counter
from ultralytics import YOLO


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
        if not self.initialized:
            return 80.0, 80.0
        P = self.kf.errorCovPost
        return float(np.sqrt(abs(float(P[0, 0])))), float(np.sqrt(abs(float(P[1, 1]))))


# ══════════════════════════════════════════════════════════════════════════════
# PX4 COMMANDER
# Convert image-space error + Kalman state → body-frame OFFBOARD commands
# ══════════════════════════════════════════════════════════════════════════════

class PX4Commander:
    """
    Maps pixel error to PX4 OFFBOARD velocity setpoints (body NED frame).

    MAVLink message: SET_POSITION_TARGET_LOCAL_NED (#84)
      coordinate_frame = MAV_FRAME_BODY_NED (8)
      type_mask        = 0b0000_0111_1100_0111  (velocity + yaw_rate only)
    """

    # Camera horizontal/vertical FOV (radians). Tune to actual lens.
    H_FOV = np.radians(90.0)
    V_FOV = np.radians(90.0)

    # Proportional gains  (error_rad → command)
    K_YAW = 1.5    # yaw_rate  = K_YAW  × angle_err_x   [rad/s per rad]
    K_VZ  = 2.0    # vz        = K_VZ   × angle_err_y   [m/s  per rad]

    # Forward approach speeds per state
    FWD = {"LOCK": 2.0, "PREDICT": 0.5, "SEARCH": 0.0}

    # Hard limits
    MAX_YAW  = 1.5   # rad/s
    MAX_VZ   = 2.0   # m/s
    MAX_FWD  = 3.0   # m/s

    def compute(self, err_x, err_y, state, fw, fh):
        """
        err_x, err_y : pixel error from frame centre (right/down = positive)
        Returns dict with vx, vy, vz, yaw_rate (all floats, SI units).
        """
        # Convert pixel error to angular error
        ang_x = err_x * (self.H_FOV / fw)   # rad, positive = target right of centre
        ang_y = err_y * (self.V_FOV / fh)   # rad, positive = target below centre

        yaw_rate = float(np.clip( self.K_YAW * ang_x,  -self.MAX_YAW, self.MAX_YAW))
        # image y-down → NED z-down: target below centre → drone descends → +vz
        vz       = float(np.clip( self.K_VZ  * ang_y,  -self.MAX_VZ,  self.MAX_VZ))
        vx       = float(np.clip( self.FWD.get(state, 0.0), 0.0, self.MAX_FWD))
        vy       = 0.0   # lateral strafe unused; yaw handles horizontal

        return dict(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)

    @staticmethod
    def format(cmd, state):
        return (
            f"[PX4 BODY_NED]  "
            f"fwd:{cmd['vx']:+.2f} m/s  "
            f"lat:{cmd['vy']:+.2f} m/s  "
            f"vz:{cmd['vz']:+.2f} m/s  "
            f"yaw:{cmd['yaw_rate']:+.3f} rad/s"
            f"  | state:{state}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# DRONE TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class DroneTracker:
    _WIN = "INTERCEPTOR DRONE TRACKER"
    _COLOR = {
        "SEARCH":  (0,   0,   255),
        "LOCK":    (0,   255, 0),
        "PREDICT": (0,   165, 255),
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
          0 → every frame runs YOLO  (default)
          N → YOLO runs every (N+1)th frame; skipped frames use Kalman-only
        """
        self.model        = YOLO(model_path)
        self.video_path   = video_path
        self.conf_thresh  = conf_thresh
        self.display_size = display_size
        self.fw, self.fh  = display_size
        self.skip_frames  = max(0, int(skip_frames))

        self.kalman  = KalmanTracker()
        self.px4     = PX4Commander()

        self.state     = "SEARCH"
        self.missed    = 0
        self.last_bbox = None
        self.pos       = None
        self.layer     = "—"
        self._frame_idx = 0

        # Thresholds
        self.MISS_TO_PREDICT = 2
        self.MISS_TO_SEARCH  = 20
        self.MIN_ROI_HALF    = 50

    # ──────────────────────────────────────────────────────────────────────────
    # YOLO
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_yolo(self, frame, roi=None):
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
        px, py = self.kalman.predict()
        vx, vy = self.kalman.velocity()
        bw, bh = (self.last_bbox[2], self.last_bbox[3]) if self.last_bbox else (80, 80)

        base = max(self.MIN_ROI_HALF, int(max(bw, bh) * 0.6))
        hw   = min(base + int(abs(vx) * self.missed * 1.5), self.fw // 2)
        hh   = min(base + int(abs(vy) * self.missed * 1.5), self.fh // 2)

        return (
            max(0, px - hw),        max(0, py - hh),
            min(self.fw, px + hw),  min(self.fh, py + hh),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # FRAME PROCESSING
    # ──────────────────────────────────────────────────────────────────────────

    def _process(self, frame):
        t0     = perf_counter()
        kf_pred = self.kalman.predict() if self.kalman.initialized else None
        t_kf   = (perf_counter() - t0) * 1000

        t0      = perf_counter()
        roi     = None
        yolo_det = None

        if self.state in ("LOCK", "PREDICT") and kf_pred:
            roi      = self._predict_roi()
            yolo_det = self._detect_yolo(frame, roi=roi)
            if yolo_det is None:
                rw = roi[2] - roi[0];  rh = roi[3] - roi[1]
                if rw < self.fw * 0.85 or rh < self.fh * 0.85:
                    yolo_det = self._detect_yolo(frame)
                    roi = None
        else:
            yolo_det = self._detect_yolo(frame)

        t_yolo = (perf_counter() - t0) * 1000

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
        self.layer     = "—"

    # ──────────────────────────────────────────────────────────────────────────
    # VISUALISATION
    # ──────────────────────────────────────────────────────────────────────────

    def _draw(self, frame, pos, roi):
        color      = self._COLOR.get(self.state, (255, 255, 255))
        h, w       = frame.shape[:2]
        cx_f, cy_f = w // 2, h // 2

        # Frame-centre reticle
        cv2.drawMarker(frame, (cx_f, cy_f), (80, 80, 80),
                       cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)

        # PREDICT: show active YOLO search ROI
        if roi and self.state == "PREDICT":
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]),
                          (60, 60, 200), 1)

        if pos:
            px, py = int(pos[0]), int(pos[1])
            err_x  = px - cx_f
            err_y  = py - cy_f
            mag    = float(np.hypot(err_x, err_y))

            # ── Turn-direction arrow (frame centre → target) ─────────────
            # Tells drone 1: "rotate in this direction to centre the target"
            if mag > 5:
                arrow_len = min(mag * 0.55, 110)
                nx = err_x / mag
                ny = err_y / mag
                tip = (int(cx_f + nx * arrow_len), int(cy_f + ny * arrow_len))

                if   mag < 30:  arrow_col = (0, 255, 0)     # green  – nearly centred
                elif mag < 100: arrow_col = (0, 200, 255)   # yellow – moderate
                else:           arrow_col = (0, 60,  255)   # red    – large error

                cv2.arrowedLine(frame, (cx_f, cy_f), tip,
                                arrow_col, 3, tipLength=0.25, line_type=cv2.LINE_AA)

            # Thin line centre → target position (full reach)
            cv2.line(frame, (cx_f, cy_f), (px, py), color, 1, cv2.LINE_AA)

            # Last YOLO bbox (dimmed when stale)
            if self.last_bbox:
                bx, by, bw, bh = self.last_bbox
                box_col = color if self.state == "LOCK" else (80, 80, 180)
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_col, 2)

            # Target crosshair
            cv2.drawMarker(frame, (px, py), color,
                           cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)

            # ── Target velocity arrow (where drone 2 is going) ──────────
            vx, vy = self.kalman.velocity()
            if np.hypot(vx, vy) > 0.5:
                tip_v = (int(px + vx * 8), int(py + vy * 8))
                cv2.arrowedLine(frame, (px, py), tip_v,
                                (0, 255, 255), 2, tipLength=0.4, line_type=cv2.LINE_AA)

            # ── PREDICT: 3-sigma uncertainty ellipse ────────────────────
            if self.state == "PREDICT":
                sx, sy = self.kalman.pos_sigma()
                cv2.ellipse(frame, (px, py),
                            (max(10, int(sx * 3)), max(10, int(sy * 3))),
                            0, 0, 360, (80, 80, 255), 1, cv2.LINE_AA)

        # HUD
        cv2.putText(frame, self.state, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, self.layer, (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {self.video_path}")

        cv2.namedWindow(self._WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._WIN, *self.display_size)

        period = self.skip_frames + 1

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
                fused, t_kf, t_yolo, roi = self._process(frame)
                skip_tag = ""
            else:
                t0    = perf_counter()
                fused = self.kalman.predict() if self.kalman.initialized else None
                t_kf  = (perf_counter() - t0) * 1000
                t_yolo = 0.0
                roi    = None
                if fused:
                    self.pos   = fused
                    self.layer = "KF-SKIP"
                skip_tag = " [SKIP]"

            # PX4 commands from pixel error
            if fused:
                ex  = fused[0] - cx_f
                ey  = fused[1] - cy_f
                cmd = self.px4.compute(ex, ey, self.state, self.fw, self.fh)
            else:
                cmd = dict(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)

            self._draw(frame, fused, roi)

            # Console output
            timing = f"YOLO:{t_yolo:5.1f}ms  KF:{t_kf:.3f}ms  | total:{t_yolo+t_kf:5.1f}ms{skip_tag}"
            px4_str = PX4Commander.format(cmd, self.state)
            print(f"{timing}\n{px4_str}\n")

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
        video_path   = "./data/videos_drones/15.mp4",
        conf_thresh  = 0.05,
        display_size = (640, 640),
        skip_frames  = 0,   # 0=every frame  1=every 2nd  2=every 3rd ...
    ).run()
