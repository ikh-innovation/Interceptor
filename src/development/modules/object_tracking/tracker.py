"""
tracker.py  –  Drone Intercept Tracker  (SKY-GATED + ADAPTIVE ZOOM)
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

Sky gate
--------
The sky mask is colour-thresholded (bright + blue/low-sat) and then refined with
a ground-from-bottom pass: only a non-sky region that rises from the bottom edge
of the frame counts as ground.  Dark/blue-ish blobs floating inside the sky are
reclaimed as sky rather than rejected as ground, and sky-only footage yields no
ground at all.  Kalman and PX4 output are unchanged from tracker_skip_sky.py.
"""

import os
os.environ.pop("SESSION_MANAGER", None) 

import cv2
import numpy as np
from collections import deque
from pathlib import Path
from time import perf_counter
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
# SKY DETECTOR  (color sky + ground-from-bottom reclaim)
# ══════════════════════════════════════════════════════════════════════════════

class SkyDetector:
    V_MIN  = 150
    S_MAX  = 60
    H_LO, H_HI = 90, 140

    RING_PAD_FRAC = 0.6
    RING_PAD_MIN  = 12
    SKY_FRAC      = 0.60

    TOP_BAND = 3

    # ── ground-from-bottom reclaim ──────────────────────────────────────────────
    # A non-sky region is only believed to be GROUND if its connected component
    # rises from the BOTTOM of the frame.  Dark / blue-ish blobs that float
    # inside the sky (haze, dark clouds, distant terrain on the horizon, codec
    # noise) are NOT connected to that bottom mass, so they are reclaimed as SKY
    # instead of being mistaken for ground — which is what used to make the
    # sky-gate reject valid detections over them.
    BOTTOM_BAND     = 3      # rows at the bottom edge that seed the ground mass
    # Guard for sky-only footage: a non-sky blob that merely clips the bottom
    # edge must cover at least this fraction of the frame to be trusted as real
    # ground.  Below it the scene is treated as all-sky → no false ground.
    GROUND_MIN_FRAC = 0.02

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
            top_sky = np.zeros_like(sky)
        else:
            top_labels = np.unique(labels[: self.TOP_BAND, :])
            top_labels = top_labels[top_labels != 0]
            top_sky = (np.isin(labels, top_labels).astype(np.uint8) * 255
                       if top_labels.size else np.zeros_like(sky))

        return self._reclaim_embedded_blobs(top_sky)

    def _reclaim_embedded_blobs(self, sky):
        """Treat as SKY any non-sky region that is not part of the ground mass.

        Ground is modelled as the non-sky pixels connected to the BOTTOM edge of
        the frame and large enough to be real terrain.  Everything that is not
        that bottom-rising mass — dark blue-ish blobs nestled inside the sky
        included — is reclaimed as sky, so the sky-gate stops rejecting
        detections that sit over such blobs.  When no bottom-connected mass is
        large enough (sky-only footage) the whole frame becomes sky and nothing
        is ever flagged as ground.
        """
        Hm, Wm  = sky.shape
        non_sky = (sky == 0).astype(np.uint8)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(non_sky)
        if n <= 1:
            return np.full_like(sky, 255)        # frame is entirely sky-coloured

        band          = max(1, self.BOTTOM_BAND)
        bottom_labels = np.unique(labels[Hm - band:, :])
        bottom_labels = bottom_labels[bottom_labels != 0]

        min_area = self.GROUND_MIN_FRAC * Hm * Wm
        keep     = [lab for lab in bottom_labels
                    if stats[lab, cv2.CC_STAT_AREA] >= min_area]

        ground = np.isin(labels, keep) if keep else np.zeros(sky.shape, bool)
        out    = np.where(ground, 0, 255).astype(np.uint8)
        out[sky > 0] = 255                       # never drop genuine top sky
        return out

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
    _WIN = "WASP DRONE TRACKER  [SKY + ADAPTIVE ZOOM]"
    _COLOR = {
        "SEARCH":  (0,   0,   255),
        "LOCK":    (0,   255, 0),
        "PREDICT": (0,   165, 255),
    }

    # ── HUD palette (BGR) — clean targeting-system look ─────────────────────────
    C_LOCK    = ( 96, 255, 120)   # locked: green
    C_SEARCH  = ( 70, 180, 255)   # searching: amber
    C_PREDICT = (230, 200, 100)   # coasting on Kalman: cyan
    C_HUD     = (200, 230, 205)   # neutral chrome (lines, reticle, labels)
    C_DIM     = (110, 120, 110)   # de-emphasised
    C_WARN    = ( 70,  70, 240)   # red (lost / rejected)
    C_INK     = ( 22,  26,  22)   # translucent panel fill
    _ACCENT   = {"SEARCH": C_SEARCH, "LOCK": C_LOCK, "PREDICT": C_PREDICT}
    _HUD_FONT = cv2.FONT_HERSHEY_DUPLEX

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
        model_path     = "./runs/detect/v0/weights/best.pt",
        video_path     = "./data/videos_drones/10.mp4",
        conf_thresh    = 0.05,
        display_size   = (640, 640),
        skip_frames    = 0,
        show_sky       = True,
        record_dir     = None,    
        record_pad_sec = 2.0,    
        headless       = False,   
        max_frames     = None,  
    ):
        self.model        = YOLO(model_path)
        self.video_path   = video_path
        self.conf_thresh  = conf_thresh
        self.display_size = display_size
        self.fw, self.fh  = display_size
        self.skip_frames  = max(0, int(skip_frames))
        self.show_sky     = show_sky
        self.record_dir     = record_dir
        self.record_pad_sec = float(record_pad_sec)
        self.headless       = headless
        self.max_frames     = max_frames

        self.kalman  = KalmanTracker()
        self.px4     = PX4Commander()
        self.sky     = SkyDetector()
        self._clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.state     = "SEARCH"
        self.missed    = 0
        self.last_bbox = None
        self.pos       = None
        self.layer     = "STANDBY"
        self.conf      = None    
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

        small = self._target_is_small()  

        if self.state in ("LOCK", "PREDICT") and kf_pred:
            roi         = self._predict_roi()
            track_imgsz = self.IMGSZ_TRACK if small else self.IMGSZ_CLOSE
            yolo_det = self._detect_region(frame_full, self._roi_disp_to_native(roi),
                                           track_imgsz, enhance=(self.USE_CLAHE and small))
            if yolo_det is None:
                sky_bbox = self._sky_bbox_native()
                if sky_bbox is not None:
                    srch_imgsz = self.IMGSZ_SEARCH if small else self.IMGSZ_TRACK
                    yolo_det = self._detect_region(frame_full, sky_bbox,
                                                   srch_imgsz, enhance=(self.USE_CLAHE and small))
                roi = None
        else:
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
            self.conf      = float(conf)
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
        return fused, t_kf, t_yolo, roi, (yolo_det is not None)

    def _reset(self):
        self.state     = "SEARCH"
        self.missed    = 0
        self.last_bbox = None
        self.kalman    = KalmanTracker()
        self.layer     = "STANDBY"
        self.conf      = None

    # ──────────────────────────────────────────────────────────────────────────
    # VISUALISATION  (unchanged)
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_sky(self, frame, color):
        if self.sky_mask is None:
            return
        tint = frame.copy()
        tint[self.sky_mask > 0] = (120, 90, 40)            
        cv2.addWeighted(tint, 0.12, frame, 0.88, 0, frame)
        cnts, _ = cv2.findContours(self.sky_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            self._dashed_poly(frame, c, color, dash=8, gap=7, thick=1)

    def _draw_inset(self, frame, frame_full, pos):
        """Bottom-right picture-in-picture: a ~100x100 native-px window around
        the detection, magnified, so the (tiny) object is actually visible."""
        if pos is None or frame_full is None:
            return
        src = self.INSET_SRC
        ncx = int(pos[0] * self.full_w / self.fw)
        ncy = int(pos[1] * self.full_h / self.fh)
        x1 = min(max(0, ncx - src // 2), max(0, self.full_w - src))
        y1 = min(max(0, ncy - src // 2), max(0, self.full_h - src))
        crop = frame_full[y1:y1 + src, x1:x1 + src]
        if crop.size == 0:
            return

        z = cv2.resize(crop, (self.INSET_DISP, self.INSET_DISP),
                       interpolation=cv2.INTER_LINEAR)
        color = self._COLOR.get(self.state, (255, 255, 255))
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

    # ── HUD primitives ──────────────────────────────────────────────────────────

    @staticmethod
    def _panel(frame, p1, p2, fill, alpha, border=None):
        """Translucent filled rectangle with an optional 1px border."""
        x1, y1 = p1; x2, y2 = p2
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return
        sub = frame[y1:y2, x1:x2]
        cv2.addWeighted(np.full_like(sub, fill), alpha, sub, 1 - alpha, 0, sub)
        if border is not None:
            cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), border, 1, cv2.LINE_AA)

    def _text(self, frame, txt, org, scale, color, thick=1, font=None):
        """Anti-aliased text with a soft shadow for legibility over video."""
        font = font or self._HUD_FONT
        cv2.putText(frame, txt, (org[0] + 1, org[1] + 1), font, scale,
                    (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(frame, txt, org, font, scale, color, thick, cv2.LINE_AA)

    @staticmethod
    def _corner_box(frame, x1, y1, x2, y2, color, length=16, thick=2):
        """Target bracket: only the four corners (targeting-system style)."""
        L = max(5, int(min(length, (x2 - x1) / 2, (y2 - y1) / 2)))
        for (cx, cy, sx, sy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                                 (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + sx * L, cy), color, thick, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + sy * L), color, thick, cv2.LINE_AA)

    def _reticle(self, frame, cx, cy, color):
        """Centre gunsight: gapped cross + dot."""
        gap, arm = 9, 16
        cv2.line(frame, (cx - gap - arm, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + gap, cy), (cx + gap + arm, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - gap - arm), (cx, cy - gap), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + gap), (cx, cy + gap + arm), color, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 2, color, -1, cv2.LINE_AA)

    @staticmethod
    def _dashed(frame, p1, p2, color, dash=9, gap=7, thick=1):
        p1 = np.array(p1, float); p2 = np.array(p2, float)
        d = p2 - p1; L = float(np.hypot(*d))
        if L < 1:
            return
        u = d / L
        n = int(L // (dash + gap)) + 1
        for i in range(n):
            a = p1 + u * (i * (dash + gap))
            b = p1 + u * min(i * (dash + gap) + dash, L)
            cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)),
                     color, thick, cv2.LINE_AA)

    @staticmethod
    def _dashed_poly(frame, pts, color, dash=8, gap=7, thick=1):
        """Dashed polyline that keeps a continuous dash pattern along a contour."""
        pts = pts.reshape(-1, 2).astype(float)
        if len(pts) < 2:
            return
        period = dash + gap
        carry = 0.0                      
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            d = b - a
            seg = float(np.hypot(*d))
            if seg < 1e-6:
                continue
            u = d / seg
            pos = 0.0
            while pos < seg:
                phase  = carry % period
                remain = (dash - phase) if phase < dash else (period - phase)
                step   = min(remain, seg - pos)
                if phase < dash:
                    p0 = a + u * pos
                    p1 = a + u * (pos + step)
                    cv2.line(frame, tuple(p0.astype(int)), tuple(p1.astype(int)),
                             color, thick, cv2.LINE_AA)
                pos   += step
                carry += step

    def _draw_frame_ticks(self, frame, color):
        """Subtle bracket in each viewport corner — frames the scene."""
        h, w = frame.shape[:2]
        m, L = 6, 22
        for (cx, cy, sx, sy) in ((m, m, 1, 1), (w - m, m, -1, 1),
                                 (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + sx * L, cy), color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + sy * L), color, 1, cv2.LINE_AA)

    # ── HUD panels ───────────────────────────────────────────────────────────────

    def _draw_header(self, frame, accent):
        w = frame.shape[1]
        self._panel(frame, (0, 0), (w, 30), self.C_INK, 0.45)
        cv2.line(frame, (0, 30), (w, 30), accent, 1, cv2.LINE_AA)

        title = "WASP  //  SKY-TRACK"
        self._text(frame, title, (12, 21), 0.5, self.C_HUD, 1)

        (tw, _), _ = cv2.getTextSize(title, self._HUD_FONT, 0.5, 1)
        self._text(frame, "IKH - DEFENSE LABS", (12 + tw + 20, 21),
                   0.5, self.C_DIM, 1)

        (sw, _), _ = cv2.getTextSize(self.state, self._HUD_FONT, 0.5, 1)
        x0 = w - sw - 30
        cv2.circle(frame, (x0 - 2, 15), 5, accent, -1, cv2.LINE_AA)
        self._text(frame, self.state, (x0 + 10, 21), 0.5, accent, 1)

        if self.conf is not None and self.state == "LOCK":
            ctxt = f"CONF {self.conf * 100:.0f}%"
            (cw, _), _ = cv2.getTextSize(ctxt, self._HUD_FONT, 0.5, 1)
            self._text(frame, ctxt, (x0 - cw - 18, 21), 0.5, accent, 1)

    def _draw_telemetry(self, frame, cmd, pos, cx_f, cy_f):
        h, w = frame.shape[:2]
        x1, y1 = 10, h - 72
        x2, y2 = min(w - 200, 360), h - 10   
        self._panel(frame, (x1, y1), (x2, y2), self.C_INK, 0.5, border=self.C_DIM)

        tx = x1 + 12
        self._text(frame, "TELEMETRY", (tx, y1 + 18), 0.42, self.C_DIM, 1)

        if pos:
            ex = pos[0] - cx_f; ey = pos[1] - cy_f
            az = np.degrees(ex * (self.px4.H_FOV / self.fw))
            el = np.degrees(-ey * (self.px4.V_FOV / self.fh))
            dev = f"DEV {ex:+04d},{ey:+04d}px   AZ {az:+05.1f}  EL {el:+05.1f}"
        else:
            dev = "DEV  ----,----px   AZ  ----  EL  ----"
        self._text(frame, dev, (tx, y1 + 38), 0.42, self.C_HUD, 1)

        px4 = (f"vx {cmd['vx']:+.2f}  vy {cmd['vy']:+.2f}  "
               f"vz {cmd['vz']:+.2f}  yaw {cmd['yaw_rate']:+.2f}")
        self._text(frame, px4, (tx, y1 + 58), 0.42, self.C_HUD, 1)

    # ── master draw ──────────────────────────────────────────────────────────────

    def _draw(self, frame, frame_full, pos, roi, cmd):
        h, w       = frame.shape[:2]
        cx_f, cy_f = w // 2, h // 2
        accent     = self._ACCENT.get(self.state, self.C_HUD)

        if self.show_sky:
            self._draw_sky(frame, accent)

        self._draw_frame_ticks(frame, self.C_DIM)
        self._reticle(frame, cx_f, cy_f, self.C_HUD)

        if roi and self.state == "PREDICT":
            self._corner_box(frame, roi[0], roi[1], roi[2], roi[3], self.C_DIM, 10, 1)

        if pos:
            px, py = int(pos[0]), int(pos[1])

            self._dashed(frame, (cx_f, cy_f), (px, py), accent, thick=1)

            if self.state == "PREDICT":
                sx, sy = self.kalman.pos_sigma()
                cv2.ellipse(frame, (px, py),
                            (max(12, int(sx * 3)), max(12, int(sy * 3))),
                            0, 0, 360, self.C_PREDICT, 1, cv2.LINE_AA)

            bw, bh = (self.last_bbox[2], self.last_bbox[3]) if self.last_bbox else (20, 20)
            half   = max(16, max(bw, bh) // 2 + 8)
            self._corner_box(frame, px - half, py - half, px + half, py + half,
                             accent, length=min(16, half // 2), thick=2)
            cv2.circle(frame, (px, py), 2, accent, -1, cv2.LINE_AA)

            vx, vy = self.kalman.velocity()
            if np.hypot(vx, vy) > 0.5:
                tip = (int(px + vx * 8), int(py + vy * 8))
                cv2.arrowedLine(frame, (px, py), tip, self.C_HUD, 1,
                                tipLength=0.3, line_type=cv2.LINE_AA)

        self._draw_header(frame, accent)
        self._draw_telemetry(frame, cmd, pos, cx_f, cy_f)

        self._draw_inset(frame, frame_full, pos)

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {self.video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps != src_fps or src_fps <= 1:  
            src_fps = 30.0

        rec_on       = self.record_dir is not None
        pad          = max(1, int(round(self.record_pad_sec * src_fps)))
        pre_buf      = deque(maxlen=pad)  
        writer       = None                
        post_left    = 0                   
        last_written = -1                 
        n_windows    = 0                   
        n_written    = 0                  
        fourcc       = None
        out_dir      = None
        out_path     = None
        if rec_on:
            out_dir  = Path(self.record_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = out_dir / f"{Path(self.video_path).stem}_detections.mp4"
            print(f"[REC] reel -> {out_path.resolve()}  | fps={src_fps:.2f}  "
                  f"pad={pad}f (~{self.record_pad_sec:.1f}s each side)")

        if not self.headless:
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
                fused, t_kf, t_yolo, roi, detected = self._process(frame_full, frame_disp)
                skip_tag = ""
            else:
                self.sky_mask = self.sky.mask(frame_disp)
                t0    = perf_counter()
                fused = self.kalman.predict() if self.kalman.initialized else None
                t_kf  = (perf_counter() - t0) * 1000
                t_yolo   = 0.0
                roi      = None
                detected = False
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

            self._draw(frame_disp, frame_full, fused, roi, cmd)

            if rec_on:
                if detected:
                    if post_left == 0:                      
                        n_windows += 1
                        if writer is None:                
                            writer = cv2.VideoWriter(
                                str(out_path), fourcc, src_fps,
                                (frame_disp.shape[1], frame_disp.shape[0]))
                        for i, f in pre_buf:                
                            if i > last_written:
                                writer.write(f); last_written = i; n_written += 1
                        print(f"[REC] window {n_windows} @ frame {self._frame_idx} "
                              f"(~{self._frame_idx / src_fps:.1f}s)")
                    writer.write(frame_disp)
                    last_written = self._frame_idx; n_written += 1
                    post_left = pad                         
                elif post_left > 0:                          
                    writer.write(frame_disp)
                    last_written = self._frame_idx; n_written += 1
                    post_left -= 1
                pre_buf.append((self._frame_idx, frame_disp.copy()))

            timing  = f"YOLO:{t_yolo:5.1f}ms  KF:{t_kf:.3f}ms  | total:{t_yolo+t_kf:5.1f}ms{skip_tag}"
            px4_str = PX4Commander.format(cmd, self.state)
            print(f"{timing}\n{px4_str}\n")

            if not self.headless:
                cv2.imshow(self._WIN, frame_disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if self.max_frames is not None and self._frame_idx >= self.max_frames:
                break

        if writer is not None:
            writer.release()
        cap.release()
        if not self.headless:
            cv2.destroyAllWindows()
        if rec_on:
            secs = n_written / src_fps
            print(f"[REC] done — {n_windows} detection window(s), {n_written} frames "
                  f"(~{secs:.1f}s) -> "
                  f"{out_path.resolve() if out_path else out_dir.resolve()}")


if __name__ == "__main__":
    DroneTracker(
        model_path     = "./runs/detect/v0/weights/best.pt",
        video_path     = "./data/videos_drones/field2.MP4",
        conf_thresh    = 0.01,
        display_size   = (640, 640),
        skip_frames    = 0,
        show_sky       = True,
        record_dir     = "./captures",  
        record_pad_sec = 5.0,
        headless       = os.environ.get("TRACKER_HEADLESS") == "1",
    ).run()
