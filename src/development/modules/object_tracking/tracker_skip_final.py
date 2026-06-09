"""
tracker_skip_final.py  –  Drone Intercept Tracker  (SKY-GATED + ADAPTIVE ZOOM)
==============================================================================
Final tracker.  Detects SMALL / DISTANT drones in the sky that the plain
640x640 pipeline misses, while staying lightweight on big/close targets.

Key idea
--------
The source footage is 1440x1080.  Resizing it to 640x640 before inference
(as the original did) shrinks a far drone to ~1 px and YOLO loses it.  Instead:

  1. ZOOM into the sky  (only when the target is SMALL/far).
     We crop the NATIVE sky bounding box and run YOLO on just that crop at a
     LARGER imgsz (IMGSZ_SEARCH, e.g. 1280).  The crop is smaller than the full
     frame AND inferred at higher resolution, so a distant drone keeps far more
     pixels than in the original.  Ground is excluded from the crop, so it is
     also cheaper than a naive full-frame high-res pass.

  2. BOOST CONTRAST (CLAHE)  — optional, OFF by default.
     A black drone differs from bright sky mostly in luminance.  CLAHE on the
     crop's L channel stretches that gap.  Disabled by default because this
     model was trained on un-equalised footage and CLAHE was measured to
     SUPPRESS valid detections; flip USE_CLAHE on only after fine-tuning.

ADAPTIVE SIZE GATE  (the "only zoom small drones" rule)
-------------------------------------------------------
The zoom + CLAHE are only worth their cost on tiny targets.  Once the drone is
CLOSE (its box exceeds BIG_OBJ_FRAC of the frame height) it already has plenty
of pixels, so the tracker drops to a plain low-res ROI pass (IMGSZ_CLOSE) with
no zoom and no CLAHE — cheaper, and it can never distort an easy big target.
The on-screen tag reads ZOOM for small targets and CLOSE for big ones.

Zoom inset
----------
A bottom-right picture-in-picture shows a ~100x100 native-px window around the
detection, magnified, so even a sub-pixel-on-screen drone is actually visible.

Sky gate, Kalman and PX4 output are unchanged from tracker_skip_sky.py.
"""

import os
os.environ.pop("SESSION_MANAGER", None)  # prevents ICE crash with Qt on domain-user sessions

import cv2
import numpy as np
from time import perf_counter
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
# SKY DETECTOR  (unchanged from tracker_skip_sky.py)
# ══════════════════════════════════════════════════════════════════════════════

class SkyDetector:
    V_MIN  = 150
    S_MAX  = 60
    H_LO, H_HI = 90, 140

    RING_PAD_FRAC = 0.6
    RING_PAD_MIN  = 12
    SKY_FRAC      = 0.60

    TOP_BAND = 3

    def __init__(self):
        self._k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def mask(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        bright = V >= self.V_MIN
        blue   = (H >= self.H_LO) & (H <= self.H_HI)
        lowsat = S <= self.S_MAX
        sky = (bright & (blue | lowsat)).astype(np.uint8) * 255

        sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN,  self._k)
        sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, self._k)

        n, labels = cv2.connectedComponents(sky)
        if n <= 1:
            return np.zeros_like(sky)
        top_labels = np.unique(labels[: self.TOP_BAND, :])
        top_labels = top_labels[top_labels != 0]
        if top_labels.size == 0:
            return np.zeros_like(sky)
        return np.isin(labels, top_labels).astype(np.uint8) * 255

    def background_is_sky(self, mask, bbox):
        x, y, w, h = bbox
        pad = max(self.RING_PAD_MIN, int(max(w, h) * self.RING_PAD_FRAC))
        Hm, Wm = mask.shape

        ox1 = max(0, x - pad);     oy1 = max(0, y - pad)
        ox2 = min(Wm, x + w + pad); oy2 = min(Hm, y + h + pad)
        outer = mask[oy1:oy2, ox1:ox2]
        if outer.size == 0:
            return False

        ix1 = max(0, x - ox1);     iy1 = max(0, y - oy1)
        ix2 = min(outer.shape[1], (x + w) - ox1)
        iy2 = min(outer.shape[0], (y + h) - oy1)
        inner = outer[iy1:iy2, ix1:ix2]

        ring_total = outer.size - inner.size
        if ring_total <= 0:
            return False
        ring_sky = cv2.countNonZero(outer) - (cv2.countNonZero(inner) if inner.size else 0)
        return (ring_sky / ring_total) >= self.SKY_FRAC


# ══════════════════════════════════════════════════════════════════════════════
# KALMAN  (unchanged)
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
# PX4 COMMANDER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class PX4Commander:
    H_FOV = np.radians(90.0)
    V_FOV = np.radians(90.0)

    K_YAW = 1.5
    K_VZ  = 2.0

    FWD = {"LOCK": 2.0, "PREDICT": 0.5, "SEARCH": 0.0}

    MAX_YAW  = 1.5
    MAX_VZ   = 2.0
    MAX_FWD  = 3.0

    def compute(self, err_x, err_y, state, fw, fh):
        ang_x = err_x * (self.H_FOV / fw)
        ang_y = err_y * (self.V_FOV / fh)

        yaw_rate = float(np.clip( self.K_YAW * ang_x,  -self.MAX_YAW, self.MAX_YAW))
        vz       = float(np.clip( self.K_VZ  * ang_y,  -self.MAX_VZ,  self.MAX_VZ))
        vx       = float(np.clip( self.FWD.get(state, 0.0), 0.0, self.MAX_FWD))
        vy       = 0.0

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
# DRONE TRACKER  (sky-gated + sky-crop zoom + CLAHE)
# ══════════════════════════════════════════════════════════════════════════════

class DroneTracker:
    _WIN = "INTERCEPTOR DRONE TRACKER  [SKY + ADAPTIVE ZOOM]"
    _COLOR = {
        "SEARCH":  (0,   0,   255),
        "LOCK":    (0,   255, 0),
        "PREDICT": (0,   165, 255),
    }

    # ── zoom-inset (picture-in-picture) ─────────────────────────────────────────
    INSET_SRC  = 100   # native-px window cropped around the detection
    INSET_DISP = 180   # on-screen size of the inset (square, bottom-right)

    # ── zoom / enhancement parameters ──────────────────────────────────────────
    IMGSZ_SEARCH = 1280   # high-res inference on the sky crop (SMALL/far target)
    IMGSZ_TRACK  = 640    # ROI inference for a SMALL target once locked
    IMGSZ_CLOSE  = 416    # cheap ROI inference for a BIG/close target (no zoom)
    CROP_PAD     = 24     # native px of padding added around the sky bbox

    # Size gate: the zoom (high imgsz) + CLAHE are only worth their cost on
    # small/distant drones.  Once the target is CLOSE it already has plenty of
    # pixels, so we drop to a plain low-res pass — cheaper, and it also avoids
    # CLAHE ever distorting an easy big target.  A target counts as "big" when
    # the longer side of its last box exceeds BIG_OBJ_FRAC of the frame height.
    BIG_OBJ_FRAC = 0.06   # ~38 px in a 640-tall display frame
    # CLAHE is OFF by default: this model was trained on un-equalised footage,
    # so histogram-equalising the crop shifts the pixel distribution and was
    # measured to SUPPRESS valid small-drone detections (e.g. a conf-0.37 hit
    # at imgsz=1280 vanished with CLAHE on).  The ZOOM (crop + high imgsz) is
    # what recovers small drones here.  Flip this to True only if you retrain /
    # fine-tune the model on contrast-enhanced data.
    USE_CLAHE    = False  # contrast-boost the crop before inference

    def __init__(
        self,
        model_path   = "./runs/detect/v0/weights/best.pt",
        video_path   = "./data/videos_drones/10.mp4",
        conf_thresh  = 0.05,
        display_size = (640, 640),
        skip_frames  = 0,
        show_sky     = True,
    ):
        self.model        = YOLO(model_path)
        self.video_path   = video_path
        self.conf_thresh  = conf_thresh
        self.display_size = display_size
        self.fw, self.fh  = display_size
        self.skip_frames  = max(0, int(skip_frames))
        self.show_sky     = show_sky

        self.kalman  = KalmanTracker()
        self.px4     = PX4Commander()
        self.sky     = SkyDetector()
        self._clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.state     = "SEARCH"
        self.missed    = 0
        self.last_bbox = None
        self.pos       = None
        self.layer     = "—"
        self._frame_idx = 0

        self.sky_mask        = None
        self._ground_rejected = False

        self.full_w = self.full_h = None

        self.MISS_TO_PREDICT = 2
        self.MISS_TO_SEARCH  = 20
        self.MIN_ROI_HALF    = 50

    # ──────────────────────────────────────────────────────────────────────────
    # helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _enhance(self, crop):
        """Local-contrast boost on the L channel so dark drones stand out."""
        if not self.USE_CLAHE:
            return crop
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _accept_box(self, x1n, y1n, x2n, y2n, conf):
        """Native-space box → display space, with the sky gate applied."""
        sx = self.fw / self.full_w
        sy = self.fh / self.full_h
        dx1, dy1 = x1n * sx, y1n * sy
        dx2, dy2 = x2n * sx, y2n * sy
        bw, bh   = dx2 - dx1, dy2 - dy1
        bbox = (int(dx1), int(dy1), max(1, int(round(bw))), max(1, int(round(bh))))

        if self.sky_mask is not None and not self.sky.background_is_sky(self.sky_mask, bbox):
            self._ground_rejected = True
            return None

        cx = int((dx1 + dx2) / 2)
        cy = int((dy1 + dy2) / 2)
        return (cx, cy, int(dx1), int(dy1), bbox[2], bbox[3], float(conf))

    def _detect_region(self, frame_full, region_native, imgsz, enhance):
        """Run one YOLO pass on a native sub-region; return best sky-valid box."""
        x1r, y1r, x2r, y2r = region_native
        x1r = max(0, x1r); y1r = max(0, y1r)
        x2r = min(self.full_w, x2r); y2r = min(self.full_h, y2r)
        if x2r - x1r < 2 or y2r - y1r < 2:
            return None

        crop = frame_full[y1r:y2r, x1r:x2r]
        if enhance:
            crop = self._enhance(crop)

        res = self.model(crop, imgsz=imgsz, verbose=False)
        if not res or not res[0].boxes:
            return None
        boxes = res[0].boxes
        confs = boxes.conf.cpu().numpy()
        order = np.argsort(confs)[::-1]
        for idx in order:
            if confs[idx] < self.conf_thresh:
                break
            x1, y1, x2, y2 = boxes.xyxy[idx].int().cpu().tolist()
            det = self._accept_box(x1 + x1r, y1 + y1r, x2 + x1r, y2 + y1r, confs[idx])
            if det is not None:
                return det
        return None

    def _target_is_small(self):
        """True when the last known target is small/distant (→ zoom + CLAHE),
        False when it is big/close (→ plain cheap pass).  Unknown ⇒ small."""
        if self.last_bbox is None:
            return True
        bw, bh = self.last_bbox[2], self.last_bbox[3]
        return max(bw, bh) < self.fh * self.BIG_OBJ_FRAC

    def _sky_bbox_native(self):
        """Bounding box of the sky mask, in native coords, padded."""
        if self.sky_mask is None:
            return None
        ys, xs = np.where(self.sky_mask > 0)
        if xs.size == 0:
            return None
        sx = self.full_w / self.fw
        sy = self.full_h / self.fh
        x1 = int(xs.min() * sx) - self.CROP_PAD
        x2 = int(xs.max() * sx) + self.CROP_PAD
        y1 = int(ys.min() * sy) - self.CROP_PAD
        y2 = int(ys.max() * sy) + self.CROP_PAD
        return (x1, y1, x2, y2)

    # ──────────────────────────────────────────────────────────────────────────
    # VELOCITY-ADAPTIVE ROI  (display coords)
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

    def _roi_disp_to_native(self, roi):
        sx = self.full_w / self.fw
        sy = self.full_h / self.fh
        return (int(roi[0] * sx), int(roi[1] * sy),
                int(roi[2] * sx), int(roi[3] * sy))

    # ──────────────────────────────────────────────────────────────────────────
    # FRAME PROCESSING
    # ──────────────────────────────────────────────────────────────────────────

    def _process(self, frame_full, frame_disp):
        self.sky_mask = self.sky.mask(frame_disp)
        self._ground_rejected = False

        t0      = perf_counter()
        kf_pred = self.kalman.predict() if self.kalman.initialized else None
        t_kf    = (perf_counter() - t0) * 1000

        t0       = perf_counter()
        roi      = None
        yolo_det = None

        small = self._target_is_small()   # gates the zoom + CLAHE

        if self.state in ("LOCK", "PREDICT") and kf_pred:
            roi         = self._predict_roi()
            track_imgsz = self.IMGSZ_TRACK if small else self.IMGSZ_CLOSE
            yolo_det = self._detect_region(frame_full, self._roi_disp_to_native(roi),
                                           track_imgsz, enhance=(self.USE_CLAHE and small))
            if yolo_det is None:
                # lost in the ROI → re-search the whole sky crop.  Use the
                # high-res zoom only for a small target; a big one needs no zoom.
                sky_bbox = self._sky_bbox_native()
                if sky_bbox is not None:
                    srch_imgsz = self.IMGSZ_SEARCH if small else self.IMGSZ_TRACK
                    yolo_det = self._detect_region(frame_full, sky_bbox,
                                                   srch_imgsz, enhance=(self.USE_CLAHE and small))
                roi = None
        else:
            # cold search: target size unknown → assume small, use the zoom
            sky_bbox = self._sky_bbox_native()
            if sky_bbox is not None:
                yolo_det = self._detect_region(frame_full, sky_bbox,
                                               self.IMGSZ_SEARCH, enhance=self.USE_CLAHE)

        t_yolo = (perf_counter() - t0) * 1000

        if yolo_det:
            cx, cy, x1, y1, w, h, conf = yolo_det
            self.kalman.update(cx, cy)
            self.last_bbox = (x1, y1, w, h)
            self.missed    = 0
            self.state     = "LOCK"
            fused          = (cx, cy)
            self.layer     = (f"ZOOM {conf:.2f}" if small else f"CLOSE {conf:.2f}")
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
            self.layer = "GROUND-REJ" if self._ground_rejected else "LOST"

        self.pos = fused
        return fused, t_kf, t_yolo, roi

    def _reset(self):
        self.state     = "SEARCH"
        self.missed    = 0
        self.last_bbox = None
        self.kalman    = KalmanTracker()
        self.layer     = "—"

    # ──────────────────────────────────────────────────────────────────────────
    # VISUALISATION  (unchanged)
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_sky(self, frame):
        if self.sky_mask is None:
            return
        tint = frame.copy()
        tint[self.sky_mask > 0] = (255, 120, 0)
        cv2.addWeighted(tint, 0.18, frame, 0.82, 0, frame)
        cnts, _ = cv2.findContours(self.sky_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, cnts, -1, (255, 200, 0), 1, cv2.LINE_AA)

    def _draw_inset(self, frame, frame_full, pos):
        """Bottom-right picture-in-picture: a ~100x100 native-px window around
        the detection, magnified, so the (tiny) object is actually visible."""
        if pos is None or frame_full is None:
            return
        src = self.INSET_SRC
        # detection centre in NATIVE coords (pos is in display coords)
        ncx = int(pos[0] * self.full_w / self.fw)
        ncy = int(pos[1] * self.full_h / self.fh)
        # window, clamped to stay fully inside the native frame
        x1 = min(max(0, ncx - src // 2), max(0, self.full_w - src))
        y1 = min(max(0, ncy - src // 2), max(0, self.full_h - src))
        crop = frame_full[y1:y1 + src, x1:x1 + src]
        if crop.size == 0:
            return

        z = cv2.resize(crop, (self.INSET_DISP, self.INSET_DISP),
                       interpolation=cv2.INTER_LINEAR)
        color = self._COLOR.get(self.state, (255, 255, 255))
        # mark the detection centre inside the inset
        rel_x = int((ncx - x1) * self.INSET_DISP / src)
        rel_y = int((ncy - y1) * self.INSET_DISP / src)
        cv2.drawMarker(z, (rel_x, rel_y), color, cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
        cv2.rectangle(z, (0, 0), (self.INSET_DISP - 1, self.INSET_DISP - 1), color, 2)
        cv2.putText(z, "ZOOM", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

        h, w = frame.shape[:2]
        m = 10
        y0 = h - self.INSET_DISP - m
        x0 = w - self.INSET_DISP - m
        if y0 >= 0 and x0 >= 0:
            frame[y0:y0 + self.INSET_DISP, x0:x0 + self.INSET_DISP] = z

    def _draw(self, frame, frame_full, pos, roi):
        if self.show_sky:
            self._draw_sky(frame)

        color      = self._COLOR.get(self.state, (255, 255, 255))
        h, w       = frame.shape[:2]
        cx_f, cy_f = w // 2, h // 2

        cv2.drawMarker(frame, (cx_f, cy_f), (80, 80, 80),
                       cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)

        if roi and self.state == "PREDICT":
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]),
                          (60, 60, 200), 1)

        if pos:
            px, py = int(pos[0]), int(pos[1])
            err_x  = px - cx_f
            err_y  = py - cy_f
            mag    = float(np.hypot(err_x, err_y))

            if mag > 5:
                arrow_len = min(mag * 0.55, 110)
                nx = err_x / mag
                ny = err_y / mag
                tip = (int(cx_f + nx * arrow_len), int(cy_f + ny * arrow_len))

                if   mag < 30:  arrow_col = (0, 255, 0)
                elif mag < 100: arrow_col = (0, 200, 255)
                else:           arrow_col = (0, 60,  255)

                cv2.arrowedLine(frame, (cx_f, cy_f), tip,
                                arrow_col, 3, tipLength=0.25, line_type=cv2.LINE_AA)

            cv2.line(frame, (cx_f, cy_f), (px, py), color, 1, cv2.LINE_AA)

            if self.last_bbox:
                bx, by, bw, bh = self.last_bbox
                box_col = color if self.state == "LOCK" else (80, 80, 180)
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_col, 2)

            cv2.drawMarker(frame, (px, py), color,
                           cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)

            vx, vy = self.kalman.velocity()
            if np.hypot(vx, vy) > 0.5:
                tip_v = (int(px + vx * 8), int(py + vy * 8))
                cv2.arrowedLine(frame, (px, py), tip_v,
                                (0, 255, 255), 2, tipLength=0.4, line_type=cv2.LINE_AA)

            if self.state == "PREDICT":
                sx, sy = self.kalman.pos_sigma()
                cv2.ellipse(frame, (px, py),
                            (max(10, int(sx * 3)), max(10, int(sy * 3))),
                            0, 0, 360, (80, 80, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, self.state, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, self.layer, (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1, cv2.LINE_AA)

        self._draw_inset(frame, frame_full, pos)

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
            ret, frame_full = cap.read()
            if not ret or frame_full is None:
                break

            frame_full           = np.ascontiguousarray(frame_full)
            self.full_h, self.full_w = frame_full.shape[:2]

            frame_disp = cv2.resize(frame_full, self.display_size)
            frame_disp = np.ascontiguousarray(frame_disp)
            cx_f, cy_f = self.fw // 2, self.fh // 2

            self._frame_idx += 1
            is_yolo_frame = (self._frame_idx % period == 0)

            if is_yolo_frame:
                fused, t_kf, t_yolo, roi = self._process(frame_full, frame_disp)
                skip_tag = ""
            else:
                self.sky_mask = self.sky.mask(frame_disp)
                t0    = perf_counter()
                fused = self.kalman.predict() if self.kalman.initialized else None
                t_kf  = (perf_counter() - t0) * 1000
                t_yolo = 0.0
                roi    = None
                if fused:
                    self.pos   = fused
                    self.layer = "KF-SKIP"
                skip_tag = " [SKIP]"

            if fused:
                ex  = fused[0] - cx_f
                ey  = fused[1] - cy_f
                cmd = self.px4.compute(ex, ey, self.state, self.fw, self.fh)
            else:
                cmd = dict(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)

            self._draw(frame_disp, frame_full, fused, roi)

            timing  = f"YOLO:{t_yolo:5.1f}ms  KF:{t_kf:.3f}ms  | total:{t_yolo+t_kf:5.1f}ms{skip_tag}"
            px4_str = PX4Commander.format(cmd, self.state)
            print(f"{timing}\n{px4_str}\n")

            cv2.imshow(self._WIN, frame_disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    DroneTracker(
        model_path   = "./runs/detect/v0/weights/best.pt",
        video_path   = "./data/videos_drones/field.MP4",
        conf_thresh  = 0.01,
        display_size = (640, 640),
        skip_frames  = 0,
        show_sky     = True,
    ).run()
