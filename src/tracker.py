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
ground at all.

Flight control output
---------------------
Steering goes to the FC as STICK COMMANDS while the pilot flies in ACRO — see
fc_control.py for the full rationale.  In ACRO a stick deflection commands a
body angular RATE, which is exactly what "rotate until the target is centred"
needs: yaw rate nulls the horizontal pixel error and pitch rate nulls the
vertical one.  Roll, throttle and arming always stay with the pilot.

This replaces the previous GUIDED velocity setpoints, which could never move
this airframe: GUIDED is a mode the drone is never in (so nothing was ever
transmitted), and a velocity setpoint needs an EKF position estimate the
gimbal-mounted airframe does not have.

The commander streams the latest command at a fixed rate independent of the
detection FPS, and releases every channel back to the pilot whenever the target
is lost, the FC is disarmed, or the tracker exits.
"""

import math
import os
import queue
import sys
import threading
import collections
os.environ.pop("SESSION_MANAGER", None)

import cv2
import numpy as np
from collections import deque
from pathlib import Path
from time import perf_counter, sleep
from datetime import datetime
from ultralytics import YOLO
from dataclasses import dataclass
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fc_control import AcroCommander, ControlLaw, commander_from_env
from mjpeg_stream import MJPEGServer


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

    # The SAME three colours the phone dashboard paints its header with, so the
    # video and the app never disagree about which law is flying.  OpenCV is
    # BGR, so these are the #1e6fd9 / #e07b1f of the dashboard, reversed.
    C_ALT     = (217, 111,  30)   # ALT_HOLD  blue
    C_ACRO    = ( 31, 123, 224)   # ACRO      orange
    C_PILOT   = (190, 200, 195)   # neither: the pilot has every channel
    _HUD_FONT = cv2.FONT_HERSHEY_DUPLEX

    # ── zoom-inset (picture-in-picture) ─────────────────────────────────────────
    INSET_SRC  = 100   # native-px window cropped around the detection
    INSET_DISP = 180   # on-screen size of the inset (square, bottom-right)

    # ── zoom / enhancement parameters ──────────────────────────────────────────
    IMGSZ_SEARCH = 640   # high-res inference on the sky crop (SMALL/far target)
    IMGSZ_TRACK  = 640    # ROI inference for a SMALL target once locked
    IMGSZ_CLOSE  = 416    # cheap ROI inference for a BIG/close target (no zoom)
    CROP_PAD     = 24     # native px of padding added around the sky bbox

    # Size gate: the zoom (high imgsz) + CLAHE are only worth their cost on
    # small/distant drones.  Once the target is CLOSE it already has plenty of
    # pixels, so we drop to a plain low-res pass — cheaper, and it also avoids
    # CLAHE ever distorting an easy big target.  A target counts as "big" when
    # the longer side of its last box exceeds BIG_OBJ_FRAC of the frame height.
    BIG_OBJ_FRAC = 0.06   # ~38 px in a 640-tall display frame

    # Detection outlier gate — see _gate_detection().
    # The on-frame telemetry block.  Off: the dashboard owns the numbers.
    HUD_PANEL       = os.environ.get("HUD_PANEL", "0") == "1"

    GATE_ENABLE     = True
    # Tuned by replaying the real flights through the gate.  At 900 px/s + 26 px
    # it threw away 10-13% of detections on 20260909 — too eager, it would start
    # suppressing real motion.  At 1500 + 60 it rejects 2.4-4.4%, and the median
    # REJECTED jump is 140-224 px: 14-22 deg of bearing inside a single frame,
    # which no real target can do.  Catch the outliers, leave the rest alone.
    GATE_PX_PER_S   = 1500.0  # a real target's centre cannot cross faster
    GATE_FLOOR_PX   = 60.0    # plus a fixed allowance for centroid jitter
    GATE_MAX_REJECT = 3       # then yield, so a real move can re-acquire.
                              # MISS_TO_SEARCH is 20, so gating alone can never
                              # drop the lock.
    # CLAHE is OFF by default: this model was trained on un-equalised footage,
    # so histogram-equalising the crop shifts the pixel distribution and was
    # measured to SUPPRESS valid small-drone detections (e.g. a conf-0.37 hit
    # at imgsz=1280 vanished with CLAHE on).  The ZOOM (crop + high imgsz) is
    # what recovers small drones here.  Flip this to True only if you retrain /
    # fine-tune the model on contrast-enhanced data.
    USE_CLAHE    = False  # contrast-boost the crop before inference

    def __init__(
        self,
        model_path      = "/runs/detect/v0/weights/best.pt",
        # ── live camera (GStreamer pipeline, required) ────────────────────────
        camera_pipeline = None,   # GStreamer pipeline string
        camera_fps      = 30.0,   # fallback FPS when cap.get(CAP_PROP_FPS) returns 0
        # ─────────────────────────────────────────────────────────────────────
        conf_thresh    = 0.05,
        display_size   = (640, 640),
        skip_frames    = 0,
        show_sky       = True,
        use_sky_mask   = True,    # False → detect on the FULL frame (no sky gate/crop)
        record_dir     = None,
        record_pad_sec = 2.0,
        log_dir        = None,   # command CSV dir; None → ./data next to tracker.py
        run_stamp      = None,   # shared per-run timestamp (video/log/csv share it)
        headless       = False,
        max_frames     = None,
        # ── remote seeker view (MJPEG over WiFi) ──────────────────────────────
        # The Jetson has no screen in the field, so the annotated frame that
        # would have gone to cv2.imshow is also served over HTTP.  Any phone or
        # laptop on the same WiFi opens http://<jetson-ip>:<stream_port>/ .
        # None disables it entirely; see mjpeg_stream.py for the cost model
        # (nothing is encoded unless somebody is watching).
        stream_port    = None,
        stream_quality = 70,     # JPEG quality 1-100
        stream_fps     = 15.0,   # encode cap, independent of capture FPS
        # Flight-control link.  Pass a ready-made AcroCommander, or leave it None
        # and one is built from the environment (see fc_control.commander_from_env
        # for every knob: MAV_CONN, FC_TRANSPORT, CONTROL_AXES, KP_*, CAM_*FOV_DEG,
        # YAW_REVERSED / PITCH_REVERSED, STICK_LIMIT, ...).
        commander      = None,
    ):
        # An .engine carries no architecture, so ultralytics has to be told
        # what it is; a .pt knows already.
        self.model        = (YOLO(model_path, task="detect")
                             if str(model_path).endswith(".engine")
                             else YOLO(model_path))
        # Restrict detection to the "face" class only.  Resolve its class id from
        # the model's own label map so it works regardless of index order
        # (this model exposes e.g. {0: 'person', 1: 'face'}).
        names = self.model.names
        names = names if isinstance(names, dict) else dict(enumerate(names))
        # ONE class is tracked, whatever the model calls it.  The head model
        # exposes {0: 'person', 1: 'face'} and we want the face; the sky model
        # is single-class and calls it '0'.  Demanding a literal "face" label
        # made the second one impossible to load at all, so: an explicit
        # TARGET_CLASS wins, else a face-like label, else the only class there
        # is.  Anything ambiguous is an error that names the options rather
        # than a silent guess about what we are shooting at.
        want = os.environ.get("TARGET_CLASS")
        if want is not None and want != "":
            match = None
            if want.isdigit() and int(want) in names:
                match = int(want)
            else:
                match = next((i for i, n in names.items()
                              if str(n).lower() == want.lower()), None)
            if match is None:
                raise RuntimeError(
                    f"TARGET_CLASS={want!r} is not in this model's labels "
                    f"{names}.")
            self.target_class = match
        else:
            self.target_class = next(
                (i for i, n in names.items() if "face" in str(n).lower()), None)
            if self.target_class is None and len(names) == 1:
                self.target_class = next(iter(names))
            if self.target_class is None:
                raise RuntimeError(
                    f"Cannot tell which class to track in {names}. "
                    "Set TARGET_CLASS to the index or the label.")
        print(f"[MODEL] single-class detection: class {self.target_class} "
              f"('{names[self.target_class]}') of {dict(names)}, max 1 object")
        print(f"[MODEL] weights: {model_path}")
        self.camera_pipeline = camera_pipeline
        self.camera_fps      = float(camera_fps)        
        # A TensorRT engine is built for ONE input shape (640 here), so the
        # per-pass imgsz choices cannot apply — ultralytics letterboxes to the
        # engine's shape whatever is asked.  Say so once and align the constants
        # rather than leaving three numbers in the code that do nothing.  No
        # loss: inference measured flat across 640/512/416 anyway.
        self.using_engine = str(model_path).endswith(".engine")
        if self.using_engine:
            self.IMGSZ_SEARCH = self.IMGSZ_TRACK = self.IMGSZ_CLOSE = 640
            print("[MODEL] engine input is fixed at 640 — IMGSZ_* aligned")

        self.conf_thresh  = conf_thresh
        self.display_size = display_size
        self.fw, self.fh  = display_size
        self.skip_frames  = max(0, int(skip_frames))
        self.show_sky     = show_sky
        self.use_sky_mask = use_sky_mask
        self.record_dir      = record_dir
        self.record_pad_sec  = float(record_pad_sec)
        # One timestamp per run ties the video, log, and command CSV together.
        self.run_stamp = run_stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.headless     = headless
        self.max_frames   = max_frames
        self.stream_port    = int(stream_port) if stream_port else None
        self.stream_quality = int(stream_quality)
        self.stream_fps     = float(stream_fps)
        self.stream         = None      # MJPEGServer, created in _run()

        self.kalman  = KalmanTracker()
        self.ap      = commander or commander_from_env(
            log_dir=(str(log_dir) if log_dir else str(_DATA_DIR)))
        self.sky     = SkyDetector()
        self._clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self._last_cmd = self.ap.neutral_command()

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
        self._gate_rejects   = 0
        self.gated_total     = 0
        self.gate_note       = ""
        self._t_last_frame   = None
        self.mode_name       = os.environ.get("WASP_MODE_SPEED", "MEDIUM").upper()
        self.target_kind     = os.environ.get("TARGET_KIND", "DRONE").upper()
        self._pending_kind   = None
        self.model_path      = model_path
        self.model_paths     = {}
        self.target_widths   = {"HUMAN": 0.20, "DRONE": 0.30}
        self._last_cmd       = None
        self._last_sent      = False
        self._ui_fps         = 0.0
        self._t_ui_prev      = None
        # Per-stage timing.  "Optimise the pipeline" is unanswerable without
        # it: inference measured 34 ms flat at imgsz 640, 512 AND 416, which
        # says the cost is per-call overhead rather than pixels — so the rest
        # of the loop had to be measured instead of assumed.
        self._prof = {}
        self._prof_n = 0
        self.prof_ms = {}
        self.PROFILE = os.environ.get("PROFILE", "1") == "1"
        # Both detectors, so the operator can switch target type in the air.
        # WASP_MODEL is whatever flight_config picked for the flying mode;
        # WASP_MODEL_ALT is the other one.  Either may be absent, in which case
        # that direction of the switch simply refuses and says so.
        _alt = os.environ.get("WASP_MODEL_ALT", "").strip()
        _cur_kind = self.target_kind
        self.model_paths[_cur_kind] = model_path
        self.model_paths["HUMAN" if _cur_kind == "DRONE" else "DRONE"] = (
            _alt or None)
        for _k, _e in (("HUMAN", "TARGET_WIDTH_HUMAN"),
                       ("DRONE", "TARGET_WIDTH_DRONE")):
            try:
                self.target_widths[_k] = float(os.environ[_e])
            except (KeyError, ValueError):
                pass
        # The named mode is the operator-facing control; push it into the law
        # now so the banner and the dashboard agree from the first frame.
        self.set_mode(self.mode_name)
        self.MISS_TO_SEARCH  = 20
        self.MIN_ROI_HALF    = 50

        # Jetson camera bring-up resilience (see _open_capture).
        self.CAM_OPEN_RETRIES  = 5     # full pipeline re-opens on first-buffer failure
        self.CAM_WARMUP_READS  = 30    # reads per attempt to catch the first good frame
        self.CAM_RETRY_WAIT_S  = 0.8   # settle time between re-opens

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

        # one class only; max_det=1 → a single object.
        res = self.model(crop, imgsz=imgsz, classes=[self.target_class],
                         max_det=1, verbose=False)
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

    # ── the operator's control surface (driven from the phone dashboard) ─────

    # Three named settings instead of a continuous bar.  These map onto the
    # law's 0..1 sensitivity: below 0.5 trims the BANK CEILING only, above it
    # multiplies the turn gain.  The gain floor is never crossed in either
    # direction — see InterceptLaw._schedule_values.
    # ONE control.  SPEED, STEER and STABILISER were three separate toggles and
    # in the field that is three things to get wrong while a drone is closing.
    # A mode now sets all of it: how fast it flies, how hard it turns, and how
    # much the sticks are allowed to twitch.
    #
    #   climb_limit  ceiling under the FC's PILOT_SPEED_UP (set once to 25 m/s)
    #   max_lean     the forward-speed channel
    #   sens         steering authority, 0..1 (0.5 = the configured turn gain)
    #   slew/tau/deadband   the stabiliser
    #
    # Climb is NOT in here as a gain any more.  With CLIMB_LOS the vertical
    # command is v_forward * tan(elevation), so picking a faster mode buys a
    # proportionally faster climb and never a steeper one — which is the whole
    # point.  The ceiling is only a cap for the corrective term.
    # Retuned 2026-09-25 for the mission app (FC now 25 m/s up, 40 deg lean,
    # camera fixed 30 deg up) in the closed-loop sim: hover at 10 m, enemy
    # 40 m out on the aim mark, 6 directions x 5/10 m/s x 4 winds (calm,
    # 5 and 8 m/s crosswind, 8 m/s headwind), PUSH 30 then AI:
    #   stabiliser 14/.13 (old) 67/96 touches, 8/.18 66, 6/.22 65 -> 6/.22,
    #   the smoothest with no accuracy cost.  2.5/.60 ("SMOOTH") was too slow
    #   for a MOVING target: 35-39/48 vs 44-48/48.
    #   lean 20 (MEDIUM) and 22 (FAST) touch alike; 30 deg lean loses the
    #   target out of the 30-deg-up camera more often (27/36 vs 34-36/36).
    #   SLOW (12 deg) can NOT beat an 8 m/s headwind: 0/9.  Calm air only.
    _STAB = dict(slew_per_s=6.0, rate_tau=0.22, deadband_frac=0.050)
    MODES = {
        "SLOW":   dict(climb_limit_mps=8.0,  max_lean_deg=12.0, sens=0.35,
                       climb_trim=5.0, **_STAB),
        "MEDIUM": dict(climb_limit_mps=15.0, max_lean_deg=20.0, sens=0.50,
                       climb_trim=6.0, **_STAB),
        "FAST":   dict(climb_limit_mps=25.0, max_lean_deg=22.0, sens=0.65,
                       climb_trim=8.0, **_STAB),
    }
    _SPEED_KEYS = ("climb_limit_mps", "max_lean_deg")
    _STAB_KEYS  = ("slew_per_s", "rate_tau", "deadband_frac")

    def set_mode(self, name):
        """Apply one of MODES by name.  Returns the name actually set."""
        key = str(name).strip().upper()
        if key not in self.MODES:
            return self.mode_name
        law = getattr(getattr(self, "ap", None), "law", None)
        prof = self.MODES[key]
        if hasattr(law, "set_speed"):
            law.set_speed(**{k: prof[k] for k in self._SPEED_KEYS})
        if hasattr(law, "set_stability"):
            law.set_stability(**{k: prof[k] for k in self._STAB_KEYS})
        if hasattr(law, "set_sens"):
            law.set_sens(prof["sens"])
        if law is not None and hasattr(law, "climb_trim"):
            law.climb_trim = float(prof["climb_trim"])
        self.mode_name = key
        return key






    def request_target_kind(self, kind):
        """Ask for HUMAN or DRONE tracking.  Applied by the CONTROL LOOP.

        The HTTP thread must not load a model: ultralytics is not re-entrant
        against a running inference, and swapping self.model underneath a
        detect call is a crash waiting for the field.  So this only sets a flag
        and the loop performs the switch between frames.

        The camera TILT is deliberately NOT touched.  Tilt is a hardware fact —
        the angle the mount is physically bolted at — and the control law adds
        it to every elevation it computes.  Changing it from a web page would
        fly the airframe high or low by exactly that error with nothing in the
        log.  What this switches is the DETECTOR and the target width; the
        visible band still follows the mount you actually have.
        """
        key = str(kind).strip().upper()
        if key in ("HUMAN", "DRONE") and key != self.target_kind:
            self._pending_kind = key
        return self._pending_kind or self.target_kind

    def _apply_pending_kind(self):
        """Called from the control loop only."""
        key = self._pending_kind
        if not key:
            return
        self._pending_kind = None
        path = self.model_paths.get(key)
        if not path:
            print(f"[TARGET] no weights configured for {key} — staying on "
                  f"{self.target_kind}")
            return
        try:
            if path == self.model_path:
                model = self.model
            elif str(path).endswith(".engine"):
                model = YOLO(path, task="detect")
            else:
                model = YOLO(path)
            names = model.names
            if key == "HUMAN":
                cls = next((i for i, n in names.items()
                            if "face" in str(n).lower()), None)
            else:
                cls = None
            if cls is None:
                cls = next(iter(names)) if len(names) == 1 else 0
            self.model = model
            self.model_path = path
            self.target_class = cls
            self.target_kind = key
            width = self.target_widths.get(key)
            law = getattr(getattr(self, "ap", None), "law", None)
            if width and hasattr(law, "target_width_m"):
                law.target_width_m = float(width)
            self._reset()          # the old track belongs to the old detector
            print(f"[TARGET] now tracking {key}: class {cls} "
                  f"('{names[cls]}'), width {width} m, weights {path}")
        except Exception as exc:                     # never take the flight down
            print(f"[TARGET] switch to {key} FAILED, staying on "
                  f"{self.target_kind}: {exc}", file=sys.stderr)

    def state_snapshot(self):
        """Everything the dashboard shows, read off the live objects."""
        ap = getattr(self, "ap", None)
        law = getattr(ap, "law", None)
        cmd = self._last_cmd or {}
        out = dict(
            state=self.state, layer=self.layer,
            conf=(None if self.conf is None else round(self.conf, 3)),
            target=self.target_kind, mode=self.mode_name,
            gated=self.gated_total, gate_note=self.gate_note,
            prof=self.prof_ms,
            fps=round(self._ui_fps, 1),
            fc_mode=(getattr(ap, "mode_name", None) if ap else None),
            armed=(bool(getattr(ap, "armed", False)) if ap else False),
            linked=(bool(getattr(ap, "linked", False)) if ap else False),
            steering=bool(self._last_sent),
        )
        if hasattr(ap, "mission_state"):
            try:
                out["mission"] = ap.mission_state()
            except Exception as exc:              # never take the flight down
                out["mission"] = dict(error=str(exc))
        with _CONSOLE_LOCK:
            out["console_seq"] = _CONSOLE_SEQ[0]
        for k, dp in (("range_m", 1), ("bank_deg", 1), ("lean_deg", 1),
                      ("climb_mps", 2), ("az_deg", 1), ("el_deg", 1),
                      ("los_az_dps", 1), ("latency_s", 3),
                      ("pn_gain_eff", 1), ("bank_cap_deg", 1),
                      ("fill_frac", 4), ("lean_el_gate", 2),
                      ("v_fwd_mps", 1)):
            v = cmd.get(k)
            out[k] = None if v is None else round(float(v), dp)
        # The schedule comes STRAIGHT off the law, not from the last frame.
        # sens_state() is a pure recomputation, so a mode button confirms on the
        # very next poll instead of waiting for a detection frame — and while
        # the seeker is in SEARCH there are no detection frames at all.
        if law is not None and hasattr(law, "sens_state"):
            try:
                live = law.sens_state()
                out["pn_gain_eff"] = round(float(live["gain_eff"]), 1)
                out["bank_cap_deg"] = round(float(live["bank_cap_deg"]), 1)
                out["fill_frac"] = round(float(live["fill_frac"]), 4)
            except Exception:
                pass
        if law is not None and hasattr(law, "speed_state"):
            try:
                sp = law.speed_state()
                out["climb_limit"] = round(float(sp["climb_limit"]), 1)
                out["descent_limit"] = round(float(sp["descent_limit"]), 1)
                out["climb_gain"] = round(float(sp["climb_gain"]), 0)
                out["max_lean_deg"] = round(float(sp["max_lean_deg"]), 0)
            except Exception:
                pass
        if law is not None and hasattr(law, "stability_state"):
            try:
                st = law.stability_state()
                out["slew_per_s"] = round(float(st["slew_per_s"]), 1)
                out["rate_tau"] = round(float(st["rate_tau"]), 3)
                out["deadband_pct"] = round(float(st["deadband_frac"]) * 100, 1)
            except Exception:
                pass
        if law is not None:
            _mc = getattr(law, "max_climb_mps", None)
            _ct = getattr(law, "cam_tilt", None)
            if _mc is not None:
                out["climb_max"] = round(float(_mc), 1)
            if _ct is not None:
                out["tilt_deg"] = round(math.degrees(float(_ct)), 0)
        return out

    def _gate_detection(self, det, kf_pred, dt):
        """Reject a detection that has teleported.  True = accept.

        MEASURED, flight 20260909_110654 (3233 engaged frames): the bbox centre
        moved 0 px between frames at the median and 16-23 px at p90 — but 73-99
        px at p99, with single-frame jumps up to 500 px on a 640 px frame.  Those
        outliers are the detector latching onto something else for a frame, not
        the target moving.

        They matter far more than their 1% share suggests, because the LOS rate
        the guidance law banks on is a DERIVATIVE.  A 100 px jump at 30 Hz is
        ~10 deg in 33 ms, i.e. ~300 deg/s of apparent LOS rate — and the log
        shows exactly that: |los_az_dps| peaking at 159 and the commanded bank
        moving up to 27.8 deg in a single frame.  That is the twitch.

        NOT vibration: over the same frames the camera-frame elevation moved
        0.31 deg per frame on average (p95 1.40), which is smooth.  The airframe
        was steady; the detector was not.

        The gate is a plausibility check against the Kalman prediction, sized by
        how fast a real target can cross the frame plus a floor for centroid
        jitter and the target's own size.  A rejected frame counts as a miss, so
        the tracker coasts on the filter — and after GATE_MAX_REJECT consecutive
        rejections it yields, because a target that really has moved must be
        able to re-acquire rather than be gated out for ever.
        """
        if not self.GATE_ENABLE or kf_pred is None or det is None:
            return True
        dt = dt if (dt and dt > 1e-3) else (1.0 / 30.0)
        jump = math.hypot(det[0] - kf_pred[0], det[1] - kf_pred[1])
        size = max(self.last_bbox[2], self.last_bbox[3]) if self.last_bbox else 0
        allow = self.GATE_PX_PER_S * dt + self.GATE_FLOOR_PX + 0.5 * size
        if jump <= allow:
            self._gate_rejects = 0
            return True
        self._gate_rejects += 1
        if self._gate_rejects > self.GATE_MAX_REJECT:
            self._gate_rejects = 0      # it has insisted; believe it
            self.gate_note = "REACQ"
            return True
        self.gate_note = f"GATE {jump:.0f}px"
        self.gated_total += 1
        return False

    def _target_is_small(self):
        """True when the last known target is small/distant (→ zoom + CLAHE),
        False when it is big/close (→ plain cheap pass).  Unknown ⇒ small."""
        if self.last_bbox is None:
            return True
        bw, bh = self.last_bbox[2], self.last_bbox[3]
        return max(bw, bh) < self.fh * self.BIG_OBJ_FRAC

    def _sky_bbox_native(self):
        """Bounding box of the sky mask, in native coords, padded.

        With the sky mask disabled, the search region is the entire frame so
        YOLO detects anywhere in the image.
        """
        if not self.use_sky_mask:
            return (0, 0, self.full_w, self.full_h)
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
        # sky_mask None ⇒ no sky gate in _accept_box and full-frame region in
        # _sky_bbox_native, i.e. YOLO runs on the whole image.
        self.sky_mask = self.sky.mask(frame_disp) if self.use_sky_mask else None
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

        _now = perf_counter()
        _dt = None if self._t_last_frame is None else (_now - self._t_last_frame)
        self._t_last_frame = _now
        if yolo_det is not None and not self._gate_detection(yolo_det, kf_pred, _dt):
            yolo_det = None          # treat as a miss: coast on the filter

        if yolo_det:
            cx, cy, x1, y1, w, h, conf = yolo_det
            self.kalman.update(cx, cy)
            self.last_bbox = (x1, y1, w, h)
            self.missed    = 0
            self.state     = "LOCK"
            fused          = (cx, cy)
            self.conf      = float(conf)
            self.layer     = (f"ZOOM {conf:.2f}" if small else f"CLOSE {conf:.2f}")
            self.gate_note = ""
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
        self._gate_rejects = 0
        self.gate_note = ""

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

    def mode_color(self):
        """The flight mode's colour, identical to the phone dashboard's header.

        The video and the app must never disagree about which law is flying, so
        both read the same three states off the same source and paint the same
        three colours: blue ALT_HOLD, orange ACRO, neutral when the pilot has
        every channel.
        """
        m = getattr(getattr(self, "ap", None), "mode_name", None)
        if m == "ALT_HOLD":
            return self.C_ALT
        if m == "ACRO":
            return self.C_ACRO
        return self.C_PILOT

    def _draw_header(self, frame, accent):
        """A single bar in the flight-mode colour, and nothing else.

        The name of the product, the name of the company and the word SEARCH
        were three things on screen that never once changed what the operator
        should do.  The phone carries the state in words; the video carries it
        as a colour you can read at a glance without reading at all.
        """
        w = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (w, 4), self.mode_color(), -1)

    # ── flight-command panel ─────────────────────────────────────────────────────

    def _hbar(self, frame, x, y, w, h, frac, color, bidirectional=True):
        """Small horizontal gauge.  bidirectional: centre tick, fill grows left
        (frac<0) or right (frac>0).  Otherwise fills 0→right."""
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.C_DIM, 1, cv2.LINE_AA)
        frac = float(np.clip(frac, -1.0, 1.0))
        if bidirectional:
            cx = x + w // 2
            cv2.line(frame, (cx, y - 2), (cx, y + h + 2), self.C_DIM, 1, cv2.LINE_AA)
            fill = int(frac * (w // 2 - 2))
            if fill > 0:
                cv2.rectangle(frame, (cx + 1, y + 2), (cx + fill, y + h - 2),
                              color, -1, cv2.LINE_AA)
            elif fill < 0:
                cv2.rectangle(frame, (cx + fill, y + 2), (cx - 1, y + h - 2),
                              color, -1, cv2.LINE_AA)
        else:
            fill = int(max(0.0, frac) * (w - 4))
            if fill > 0:
                cv2.rectangle(frame, (x + 2, y + 2), (x + 2 + fill, y + h - 2),
                              color, -1, cv2.LINE_AA)

    def _draw_flightcmd(self, frame, cmd, pos, cx_f, cy_f, sent, accent):
        """Flight panel: target deviation, the STICK command going to the FC
        right now, and the FC's own reported state.

        The lamp reports what actually happened to the command, never merely
        that one was computed — the old HUD showed "TX" while the tracker was
        transmitting nothing at all, which is exactly how a dead control path
        went unnoticed:
            DRIVE   sticks sent AND the FC echoed them back (confirmed)
            TX      sticks sent, confirmation still pending
            REFUSED sticks sent but the FC is discarding them
            PILOT   the FC is not in ACRO — the pilot holds every channel
            ARM?    a target is locked but the FC is disarmed
            IDLE    channels released to the pilot (no target)
            SIM     no flight controller attached
        """
        # The numbers moved to the phone dashboard, which is where the operator
        # is actually looking and where they can be read without squinting at a
        # 640 px video over WiFi.  The frame keeps only what has to be spatial —
        # the reticle, the target bracket and the state lamp.  HUD_PANEL=1
        # brings the old block back for bench work.
        if not self.HUD_PANEL:
            return

        h, _ = frame.shape[:2]
        # The intercept law drives four channels instead of two, so it needs two
        # more gauge rows and the panel has to grow or the FC line collides.
        intercept = bool(getattr(self.ap, "is_intercept", False))
        pw, ph = 252, (188 if intercept else 130)
        x1 = 10
        y1 = h - 10 - ph
        x2, y2 = x1 + pw, y1 + ph
        self._panel(frame, (x1, y1), (x2, y2), self.C_INK, 0.5, border=self.C_DIM)

        tx = x1 + 12
        self._text(frame, "FLIGHT CMD", (tx, y1 + 18), 0.42, self.C_DIM, 1)

        confirmed = self.ap.override_confirmed
        if not self.ap.linked:
            lamp, tag = self.C_SEARCH, "SIM"
        elif sent and confirmed is False:
            lamp, tag = self.C_WARN, "REFUSED"
        elif sent:
            lamp, tag = self.C_LOCK, ("DRIVE" if confirmed else "TX")
        elif not getattr(self.ap, "rate_mode_ok", True):
            # The pilot's mode switch is out of ACRO, so we are deliberately
            # hands-off.  Distinct from IDLE (no target) and ARM? (disarmed).
            lamp, tag = self.C_WARN, "PILOT"
        elif pos is not None and not self.ap.armed:
            lamp, tag = self.C_SEARCH, "ARM?"
        else:
            lamp, tag = self.C_DIM, "IDLE"
        (tw, _), _ = cv2.getTextSize(tag, self._HUD_FONT, 0.42, 1)
        cv2.circle(frame, (x2 - tw - 24, y1 + 13), 4, lamp, -1, cv2.LINE_AA)
        self._text(frame, tag, (x2 - tw - 14, y1 + 18), 0.42, lamp, 1)

        if pos:
            ex = pos[0] - cx_f; ey = pos[1] - cy_f
            dev = (f"DEV {ex:+04d},{ey:+04d}px  "
                   f"AZ{cmd.get('az_deg', 0.0):+05.1f} EL{cmd.get('el_deg', 0.0):+05.1f}")
        else:
            dev = "DEV ----,----px  AZ----- EL-----"
        self._text(frame, dev, (tx, y1 + 36), 0.36, self.C_HUD, 1)

        # Gauges show the commanded BODY RATE, not the raw stick, so the bar and
        # the number always agree: with YAW_REVERSED/PITCH_REVERSED set the stick
        # sign is flipped to suit a reversed axis while the physical intent is
        # unchanged, and it is the intent the operator needs to read.
        lim = max(1e-3, self.ap.law.stick_limit)
        bar_x, bar_w, bar_h = tx + 42, 110, 8
        if intercept:
            # One row per driven channel, each showing the PHYSICAL quantity it
            # commands rather than the stick, and each scaled by its own ceiling:
            # bank and lean are angles, climb is a rate, yaw is a rate.
            law = self.ap.law
            max_bank = max(1e-3, np.degrees(law.max_bank))
            max_lean = max(1e-3, np.degrees(law.max_lean))
            max_climb = max(1e-3, law.max_climb_mps)
            rows = (
                ("YAW", cmd["yaw_rate"] / (lim * self.ap.max_yaw_rate),
                 f"{np.degrees(cmd['yaw_rate']):+04.0f}d/s"),
                ("BNK", cmd.get("bank_deg", 0.0) / max_bank,
                 f"{cmd.get('bank_deg', 0.0):+05.1f}°"),
                ("LEN", cmd.get("lean_deg", 0.0) / max_lean,
                 f"{cmd.get('lean_deg', 0.0):+05.1f}°"),
                ("CLB", cmd.get("climb_mps", 0.0) / max_climb,
                 f"{cmd.get('climb_mps', 0.0):+04.1f}m/s"),
            )
        else:
            rows = (
                ("YAW", cmd["yaw_rate"] / (lim * self.ap.max_yaw_rate),
                 f"{np.degrees(cmd['yaw_rate']):+04.0f}d/s"),
                ("PIT", cmd["pitch_rate"] / (lim * self.ap.max_pitch_rate),
                 f"{np.degrees(cmd['pitch_rate']):+04.0f}d/s"),
            )
        for i, (label, frac, val) in enumerate(rows):
            y = y1 + 48 + i * 20
            self._text(frame, label, (tx, y + bar_h), 0.38, self.C_HUD, 1)
            self._hbar(frame, bar_x, y, bar_w, bar_h, frac, accent, True)
            self._text(frame, val, (bar_x + bar_w + 8, y + bar_h), 0.38, self.C_HUD, 1)

        # FC truth: what mode it is really in and whether the motors are live.
        # With no link there is no arm state to report, so say that rather than
        # printing a stale flag next to "offline".
        fc_y = y1 + (144 if intercept else 104)
        if self.ap.linked:
            head = f"FC {self.ap.mode_name}"
            arm_txt = "ARMED" if self.ap.armed else "DISARMED"
            arm_col = self.C_LOCK if self.ap.armed else self.C_WARN
        else:
            head, arm_txt, arm_col = "FC", "not connected", self.C_SEARCH
        self._text(frame, head, (tx, fc_y), 0.36, self.C_HUD, 1)
        (mw, _), _ = cv2.getTextSize(head, self._HUD_FONT, 0.36, 1)
        self._text(frame, arm_txt, (tx + mw + 10, fc_y), 0.36, arm_col, 1)

        if intercept:
            # Range, latency and terminal state are the three things that decide
            # whether an intercept run is healthy, and none of them is visible
            # anywhere else in the HUD.
            rng = cmd.get("range_m")
            extra = (f"R {'--' if rng is None else f'{rng:.1f}m'}  "
                     f"lat {cmd.get('latency_s', 0.0) * 1000:.0f}ms  "
                     f"LOS{cmd.get('los_az_dps', 0.0):+.0f}d/s")
            col = self.C_HUD
            if cmd.get("terminal"):
                extra, col = "TERMINAL — guidance frozen, flying it in", self.C_WARN
            elif not cmd.get("have_att"):
                extra, col = "NO ATTITUDE — roll+climb are ZERO", self.C_WARN
            self._text(frame, extra, (tx, y1 + 124), 0.34, col, 1)

        if intercept:
            # The operator's sensitivity slider, read back out of the law and
            # drawn into the frame the slider's own page is watching.  This is
            # the loop that confirms a remote change actually landed.
            sens = cmd.get("sens")
            if sens is not None:
                s_txt = (f"SENS {sens:.2f}  gain {cmd.get('pn_gain_eff', 0.0):.0f}"
                         f"  cap {cmd.get('bank_cap_deg', 0.0):.1f}°"
                         f"  fill {cmd.get('fill_frac', 0.0) * 100:.1f}%")
                s_col = self.C_WARN if sens > 0.5 else self.C_HUD
                self._text(frame, s_txt, (tx, y1 + 162), 0.34, s_col, 1)

        self._text(frame, f"{self.ap.send_hz:.0f}Hz stream  (skip {self.skip_frames})",
                   (tx, y1 + ph - 8), 0.34, self.C_DIM, 1)

    # ── operator sensitivity, exposed to the HUD page ────────────────────────────

    # ── recording, off the control loop ──────────────────────────────────────
    #
    # MEASURED on this Jetson: cv2.VideoWriter.write() of a 640x640 mp4v frame
    # costs 9.42 ms.  A frame copy costs 0.10 ms.  So the recording cost was
    # never the buffering — it was the ENCODER, running on the control loop, and
    # 9.4 ms out of a 48 ms budget is the difference between 21 and 26 fps.
    #
    # The reel is evidence, not flight-critical, so it goes to a worker thread
    # behind a bounded queue and DROPS frames if the encoder cannot keep up.
    # Guidance must never wait on a video file.

    def _rec_start_thread(self):
        self._rec_q = queue.Queue(maxsize=90)
        self._rec_dropped = 0
        self._rec_stop = threading.Event()
        self._rec_thread = threading.Thread(target=self._rec_worker,
                                            name="wasp-rec", daemon=True)
        self._rec_thread.start()

    def _rec_worker(self):
        while True:
            try:
                frame = self._rec_q.get(timeout=0.25)
            except queue.Empty:
                if self._rec_stop.is_set():
                    return
                continue
            if frame is None:
                return
            try:
                if self._writer is not None:
                    self._writer.write(frame)
            except Exception:
                pass                      # a bad reel never stops the flight

    def _rec_submit(self, frame):
        """Hand a frame to the writer thread.  Never blocks."""
        try:
            self._rec_q.put_nowait(frame)
            return True
        except queue.Full:
            self._rec_dropped += 1
            return False

    def _rec_finish(self):
        if getattr(self, "_rec_thread", None) is None:
            return
        self._rec_stop.set()
        try:
            self._rec_q.put_nowait(None)
        except Exception:
            pass
        self._rec_thread.join(timeout=8.0)
        if self._rec_dropped:
            print(f"[REC] {self._rec_dropped} frame(s) dropped — the encoder "
                  f"could not keep up; guidance was not delayed for them.")

    def _lap(self, name, t_prev):
        """Accumulate one stage's time.  Returns the new mark."""
        if not self.PROFILE:
            return t_prev
        now = perf_counter()
        self._prof[name] = self._prof.get(name, 0.0) + (now - t_prev)
        return now

    def _prof_report(self, every=60):
        """Average the stage times over `every` frames and publish them."""
        if not self.PROFILE:
            return
        self._prof_n += 1
        if self._prof_n < every:
            return
        n = self._prof_n
        self.prof_ms = {k: round(v / n * 1000.0, 2)
                        for k, v in sorted(self._prof.items(),
                                           key=lambda kv: -kv[1])}
        # Keys indented with two spaces are BREAKDOWNS of a stage above them
        # (model/kalman live inside detect), so they must not be added again.
        total = sum(v for k, v in self.prof_ms.items()
                    if not k.startswith("  "))
        self.prof_ms["TOTAL"] = round(total, 2)
        print("[PROF] " + "  ".join(f"{k} {v:.1f}" for k, v in
                                    self.prof_ms.items())
              + f"  -> {1000.0 / total if total else 0:.1f} fps")
        self._prof.clear()
        self._prof_n = 0

    def _command(self, key, value):
        """One operator command from the dashboard.  Runs on the HTTP thread.

        Neither branch touches anything the control loop is mid-way through:
        the steering mode is a single clamped float store into the law, and the
        target switch only raises a flag for the loop to act on between frames.
        """
        if key in ("mode", "sens"):
            # "speed"/"sens" kept as aliases so an old bookmarked dashboard
            # still drives the one control instead of silently doing nothing.
            self.set_mode(value)
            m = getattr(self.ap, "mission", None)
            if m is not None:
                m.set_level("ai", self.mode_name)
        elif key == "target":
            self.request_target_kind(value)
        elif key in ("hold", "release", "step", "engage", "abort", "lvl"):
            return self._mission_command(key, str(value))
        else:
            raise ValueError("unknown command")
        return ""

    def _mission_command(self, key, value):
        """The phone's mission buttons.  Only POSTS a request to the mission;
        the commander's TX thread decides.  Returns '' or a refusal reason."""
        m = getattr(self.ap, "mission", None)
        if m is None:
            return "mission app not running (WASP_APP=mission, FC linked?)"
        now = perf_counter()
        if key == "hold":
            kind, _, session = value.partition(":")
            return m.request_hold(kind.strip().lower(), session.strip() or "?", now)
        if key == "release":
            m.request_release(value.strip())
            return ""
        if key == "step":
            return m.request_step(-1 if value.strip().lower().startswith("l") else 1)
        if key == "engage":
            return m.request_engage()
        if key == "abort":
            return m.request_abort()
        # lvl=<group>:<LEVEL>, groups vert / yaw / push / ai
        group, _, level = value.partition(":")
        group, level = group.strip().lower(), level.strip().upper()
        if group == "ai":
            if level not in self.MODES:
                return "unknown speed"
            self.set_mode(level)
        return m.set_level(group, level)

    # ── master draw ──────────────────────────────────────────────────────────────

    def _draw(self, frame, frame_full, pos, roi, cmd, sent):
        h, w       = frame.shape[:2]
        cx_f, cy_f = w // 2, h // 2
        accent     = self._ACCENT.get(self.state, self.C_HUD)

        if self.show_sky:
            self._draw_sky(frame, accent)

        self._draw_frame_ticks(frame, self.C_DIM)
        self._reticle(frame, cx_f, cy_f, self.mode_color())

        if roi and self.state == "PREDICT":
            self._corner_box(frame, roi[0], roi[1], roi[2], roi[3], self.C_DIM, 10, 1)

        if pos:
            px, py = int(pos[0]), int(pos[1])

            self._dashed(frame, (cx_f, cy_f), (px, py), self.mode_color(), thick=1)

            if self.state == "PREDICT":
                sx, sy = self.kalman.pos_sigma()
                cv2.ellipse(frame, (px, py),
                            (max(12, int(sx * 3)), max(12, int(sy * 3))),
                            0, 0, 360, self.C_PREDICT, 1, cv2.LINE_AA)

            # A RING in the flight-mode colour, sized to the target: a box says
            # "here is a detection", a ring says "this is the thing I am flying
            # at", and at 640 px a circle survives being small far better than
            # four corner brackets do.
            bw, bh = (self.last_bbox[2], self.last_bbox[3]) if self.last_bbox else (20, 20)
            rad = int(max(14, max(bw, bh) * 0.62 + 6))
            mc = self.mode_color()
            if self.state == "PREDICT":
                # Coasting on the filter: broken ring, so "I am guessing" is
                # visible without a word for it.
                for a0 in range(0, 360, 40):
                    cv2.ellipse(frame, (px, py), (rad, rad), 0, a0, a0 + 22,
                                mc, 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), rad, mc, 2, cv2.LINE_AA)
                # four short ticks on the ring -- a sight, not a rectangle
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cv2.line(frame, (px + dx * (rad - 5), py + dy * (rad - 5)),
                             (px + dx * (rad + 6), py + dy * (rad + 6)),
                             mc, 2, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 2, mc, -1, cv2.LINE_AA)

            vx, vy = self.kalman.velocity()
            if np.hypot(vx, vy) > 0.5:
                tip = (int(px + vx * 8), int(py + vy * 8))
                cv2.arrowedLine(frame, (px, py), tip, self.C_HUD, 1,
                                tipLength=0.3, line_type=cv2.LINE_AA)

        self._draw_header(frame, accent)
        self._draw_flightcmd(frame, cmd, pos, cx_f, cy_f, sent, accent)

        # (the zoom inset is deliberately not drawn: it covered a corner of the
        #  very image the operator is trying to read)
        
        
    # ──────────────────────────────────────────────────────────────────────────
    # CAPTURE FACTORY  ← new helper that opens the right source
    # ──────────────────────────────────────────────────────────────────────────

    def _open_capture(self):
        """Open the live camera from its GStreamer pipeline.

        Returns (cap, src_fps, source_label).
        """
        if self.camera_pipeline is None:
            raise RuntimeError("camera_pipeline is required (no capture source set).")

        print(f"[CAP] Opening GStreamer pipeline:\n      {self.camera_pipeline}")
        # On Jetson, nvarguscamerasrc intermittently fails to map the FIRST
        # capture buffer ("NvBufSurfaceFromFd Failed" / "dmabuf_fd -1 mapped
        # entry NOT found").  When it does, that error propagates and KILLS
        # the source thread, so no frame ever arrives — retrying read() on the
        # same handle is useless because the thread is already dead.  The only
        # recovery is to tear the pipeline down and open a FRESH one; because
        # the failure is a startup race it usually clears within a few tries.
        cap  = None
        warm = None
        for attempt in range(self.CAM_OPEN_RETRIES):
            cap = cv2.VideoCapture(self.camera_pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                # Warm up: pull frames until one arrives or the source dies.
                for _ in range(self.CAM_WARMUP_READS):
                    ok, warm = cap.read()
                    if ok and warm is not None:
                        break
            if warm is not None:
                if attempt:
                    print(f"[CAP] camera came up on attempt {attempt + 1}")
                break
            # This attempt is dead — release and try a clean re-open.
            cap.release()
            cap = None
            print(f"[CAP] no frames on attempt {attempt + 1}/"
                  f"{self.CAM_OPEN_RETRIES} (NvBufSurfaceFromFd?) — reopening")
            sleep(self.CAM_RETRY_WAIT_S)

        if warm is None:
            raise RuntimeError(
                "GStreamer pipeline opened but delivered no frames after "
                f"{self.CAM_OPEN_RETRIES} attempts.\n"
                "  nvarguscamerasrc failed to map capture buffers "
                "('NvBufSurfaceFromFd Failed').  Try, in order:\n"
                "      sudo systemctl restart nvargus-daemon\n"
                "      sudo reboot            # clears a wedged VI/NVCSI/Argus state\n"
                "  Also confirm no other process holds the camera and that the\n"
                "  CSI ribbon is seated.  Verify the stack independently with:\n"
                "      gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=30 ! "
                "fakesink\n"
                f"Pipeline: {self.camera_pipeline}"
            )

        # GStreamer appsink usually reports 0 for FPS — fall back to caller's value
        reported = cap.get(cv2.CAP_PROP_FPS)
        src_fps  = reported if (reported and reported == reported and reported > 1) \
                   else self.camera_fps
        label = f"camera (pipeline, ~{src_fps:.1f} fps)"

        return cap, src_fps, label

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        """Run the tracker, guaranteeing the sticks go back to the pilot.

        Whatever happens in the loop — Ctrl-C, a camera fault, an unhandled
        exception — the finally block releases every overridden channel and
        closes the link.  Without it a crash mid-steer would leave the last
        override latched on the FC until RC_OVERRIDE_TIME expired.
        """
        try:
            self._run()
        finally:
            self.ap.close()
            if self.stream is not None:
                self.stream.stop()      # also covers Ctrl-C and a mid-run raise
                self.stream = None

    def _run(self):
        # Identify the firmware, read the FC's own RC map/rates, and print every
        # blocker BEFORE a single frame is processed.  Fails loudly if the link
        # is dead or the board is not something we can steer.
        self.ap.connect()

        cap, src_fps, source_label = self._open_capture()
        print(f"[RUN] Source: {source_label}")
            
        # ── recording setup ────────────────────────────────────────────────────
        rec_on       = self.record_dir is not None
        pad          = max(1, int(round(self.record_pad_sec * src_fps)))
        pre_buf      = deque(maxlen=pad)
        fourcc       = None
        out_dir      = None
        out_path     = None            

        if rec_on:
            out_dir = Path(self.record_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = out_dir / f"{self.run_stamp}.mp4"   # new file per run
            print(f"[REC] reel -> {out_path.resolve()}  | fps={src_fps:.2f}  "
                  f"pad={pad}f (~{self.record_pad_sec:.1f}s each side)")            

        # ── remote seeker view ─────────────────────────────────────────────────
        # Independent of `headless`: on the Jetson there is no local window at
        # all, and the stream is the ONLY way anyone sees what the seeker sees.
        # A failure to bind must never take the flight down with it, so this is
        # best-effort and the tracker runs on without it.
        if self.stream_port:
            try:
                self.stream = MJPEGServer(port=self.stream_port,
                                          quality=self.stream_quality,
                                          fps=self.stream_fps,
                                          state_get=self.state_snapshot,
                                          command=self._command,
                                          console_get=console_since).start()
            except OSError as exc:
                print(f"[STREAM] disabled — cannot bind port "
                      f"{self.stream_port}: {exc}")
                self.stream = None

        if not self.headless:
            cv2.namedWindow(self._WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._WIN, *self.display_size)

        period = self.skip_frames + 1

        # The reel is only a playable mp4 once the writer is released — that is
        # what writes the moov atom.  Ctrl-C is the NORMAL way this program
        # ends, so releasing after the loop is not enough: the run that matters
        # most (the one you stopped to go look at) is exactly the one that would
        # be left as an unreadable stub.  Hence try/finally.
        try:
            self._loop(cap, src_fps, period, rec_on, pad, pre_buf, out_path,
                       fourcc)
        finally:
            self._finish_recording()
            cap.release()
            if not self.headless:
                cv2.destroyAllWindows()

    def _finish_recording(self):
        """Close the reel and report it.  Safe to call twice."""
        # Flush and stop the encoder thread FIRST, or the last queued frames
        # are written to a file that has already been released.
        self._rec_finish()
        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.release()
            self._writer = None
        if getattr(self, "_rec_on", False):
            src_fps = getattr(self, "_rec_fps", 0.0) or 0.0
            n_written = getattr(self, "_rec_frames", 0)
            n_windows = getattr(self, "_rec_windows", 0)
            secs = n_written / src_fps if src_fps else 0.0
            where = getattr(self, "_rec_path", None)
            print(f"[REC] done — {n_windows} detection window(s), {n_written} "
                  f"frames (~{secs:.1f}s) -> {where}")
            self._rec_on = False

    def _loop(self, cap, src_fps, period, rec_on, pad, pre_buf, out_path,
              fourcc):
        # Recording state lives on self so _finish_recording() can close the
        # reel from the finally block above, whatever ended the loop.
        self._writer      = None
        self._rec_on      = rec_on
        if rec_on:
            self._rec_start_thread()
        self._rec_fps     = src_fps
        self._rec_frames  = 0
        self._rec_windows = 0
        self._rec_path    = out_path.resolve() if out_path else None
        last_written      = -1
        post_left         = 0

        while cap.isOpened():
            ret, frame_full = cap.read()
            # Stamped the instant the frame lands, so the control law can measure
            # its own end-to-end latency instead of assuming the frame interval.
            # At intercept speed the difference is metres of stale geometry.
            t_cap = perf_counter()
            if not ret or frame_full is None:
                break

            _tp = perf_counter()
            frame_full           = np.ascontiguousarray(frame_full)
            self.full_h, self.full_w = frame_full.shape[:2]

            frame_disp = cv2.resize(frame_full, self.display_size)
            frame_disp = np.ascontiguousarray(frame_disp)
            _tp = self._lap("resize", _tp)
            cx_f, cy_f = self.fw // 2, self.fh // 2

            self._frame_idx += 1
            is_yolo_frame = (self._frame_idx % period == 0)

            # Between frames is the only safe place to swap the detector.
            if self._pending_kind:
                self._apply_pending_kind()

            # Display FPS, smoothed, for the dashboard.
            if self._t_ui_prev is not None:
                _d = t_cap - self._t_ui_prev
                if _d > 1e-6:
                    self._ui_fps = 0.9 * self._ui_fps + 0.1 * (1.0 / _d)
            self._t_ui_prev = t_cap

            if is_yolo_frame:
                fused, _t_kf, _t_yolo, roi, detected = self._process(
                    frame_full, frame_disp)
                _tp = self._lap("detect", _tp)
                if self.PROFILE:
                    # _process reports the model call separately, so "detect"
                    # minus this is the crop/Kalman/bookkeeping around it.
                    self._prof["  model"] = (self._prof.get("  model", 0.0)
                                             + _t_yolo / 1000.0)
                    self._prof["  kalman"] = (self._prof.get("  kalman", 0.0)
                                              + _t_kf / 1000.0)
            else:
                self.sky_mask = self.sky.mask(frame_disp) if self.use_sky_mask else None
                fused = self.kalman.predict() if self.kalman.initialized else None
                roi   = None
                if fused:
                    self.pos   = fused
                    self.layer = "KF-SKIP"

            # ── STEERING COMMAND ─────────────────────────────────────────────
            # Each detection frame recomputes the stick command that centres the
            # target; the commander's tx thread streams the latest one to the FC
            # at a FIXED rate (SEND_HZ, default 20 Hz) independent of the
            # detection FPS, so a slow YOLO frame never starves the FC and skip
            # frames do not create gaps in the stream.
            #   • target in view (LOCK/PREDICT) → yaw + pitch rates null the
            #     pixel offset, so the airframe rotates onto the target.
            #   • no target (SEARCH)            → sticks released to the pilot.
            # The sticks are driven ONLY while the FC is armed and a target is
            # locked; every other instant the channels go back to the pilot.
            target_off = None
            if is_yolo_frame:
                if fused:
                    target_off = (fused[0] - cx_f, fused[1] - cy_f)
                    # last_bbox and t_cap are ignored by the servo law and used by
                    # the intercept law for the range estimate, the terminal
                    # trigger and the latency measurement.
                    cmd = self.ap.compute(target_off[0], target_off[1],
                                          self.state, self.fw, self.fh,
                                          bbox=self.last_bbox, capture_t=t_cap)
                else:
                    cmd = self.ap.neutral_command()     # no target: centre + release
                self.ap.update(cmd, target_off, have_face=bool(fused),
                               detected=bool(detected),
                               conf=self.conf if detected else None,
                               bbox=self.last_bbox if detected else None)
                self._last_cmd = cmd
            else:
                cmd = self._last_cmd               # HUD display only
                if fused:
                    target_off = (fused[0] - cx_f, fused[1] - cy_f)

            # The commander's own live flag — true only when stick values are
            # really going out AND the FC is armed to act on them.
            actually_sent = self.ap.steering
            self._last_sent = bool(actually_sent)

            _tp = self._lap("control", _tp)
            self._draw(frame_disp, frame_full, fused, roi, cmd, actually_sent)
            _tp = self._lap("draw", _tp)

            # Same fully annotated frame the local window would show, pushed to
            # whoever is watching over WiFi.  One memcpy, no encode, no socket
            # work on this thread — see mjpeg_stream.publish().
            if self.stream is not None:
                self.stream.publish(frame_disp)
            _tp = self._lap("stream", _tp)

            # ── detection-gated recording ──────────────────────────────────────
            # This block CLAIMED to be detection-gated and was not: post_left
            # was re-armed to `pad` on every frame and never decremented, so the
            # gate opened once and stayed open.  The reel therefore ran
            # continuously (174 MB for one field session) AND a full 640x640
            # frame was deep-copied into a pre-roll buffer on every single frame
            # that nothing would ever read again.  Measured at 10.3 ms per frame
            # of the 80 ms budget — the second largest cost in the loop after
            # inference, spent on a buffer for a window that had already opened.
            #
            # Now: a detection arms the window and each frame without one spends
            # one frame of the post-roll.  The pre-roll copy happens ONLY while
            # idle, which is exactly when pre-roll is what it is for — so during
            # an engagement, when the frame rate actually matters, that memcpy
            # is gone.
            if rec_on:
               if fused:
                   if post_left == 0:
                       self._rec_windows += 1
                       if self._writer is None:
                           self._writer = cv2.VideoWriter(
                               str(out_path), fourcc, src_fps,
                               (frame_disp.shape[1], frame_disp.shape[0]))
                           if not self._writer.isOpened():
                               # Say so once, here, rather than leaving a stub
                               # file and a silent gap where the evidence
                               # should be.
                               print(f"[REC][WARN] could not open {out_path} "
                                     "for writing — this run will NOT be "
                                     "recorded.")
                               self._rec_on = rec_on = False
                       if rec_on:
                           for i, f in pre_buf:
                               if i > last_written:
                                   self._rec_submit(f); last_written = i
                                   self._rec_frames += 1
                           print(f"[REC] window {self._rec_windows} @ frame "
                                 f"{self._frame_idx} "
                                 f"(~{self._frame_idx / src_fps:.1f}s)")
                   post_left = pad
               if rec_on and post_left > 0:
                   # A copy, because the loop keeps drawing on frame_disp after
                   # this and the encoder may not have consumed it yet.  0.10 ms
                   # measured, against the 9.42 ms encode it displaces.
                   self._rec_submit(frame_disp.copy())
                   last_written = self._frame_idx; self._rec_frames += 1
                   post_left -= 1
               elif rec_on:
                   # Idle: keep a rolling pre-roll so the next window starts
                   # `pad` frames BEFORE the detection that opened it.
                   pre_buf.append((self._frame_idx, frame_disp.copy()))

            _tp = self._lap("record", _tp)
            # Print ONLY the real controller command steering the drone.
            print(self.ap.format_command(
                self.state, cmd, target_off, actually_sent,
                live=self.ap.linked, is_yolo=is_yolo_frame))

            _tp = self._lap("print", _tp)

            if not self.headless:
                cv2.imshow(self._WIN, frame_disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            _tp = self._lap("imshow", _tp)
            self._prof_report()

            if self.max_frames is not None and self._frame_idx >= self.max_frames:
                break

        # Releasing the channels, the reel, the camera and the window is the
        # caller's finally block — reaching the end of the loop normally is only
        # one of the ways out, and the others matter more.
        self.ap.close()        # releases every channel, then closes the link


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class CameraConfig:
    # Camera selection
    sensor_id: int = int(os.getenv("CAM_SENSOR_ID", "0"))
    # Sensor mode MUST be able to produce the width/height/fps requested below.
    # On the IMX219 the modes are:
    #   0: 3280x2464@21  1: 3280x1848@28  2: 1920x1080@30
    #   3: 1640x1232@30  4: 1280x720@60
    # Mode 0 (full-res, 21 fps) cannot output 1280x720@30 — that mismatch makes
    # nvarguscamerasrc fail with "Frame Rate specified is greater than supported"
    # / InvalidState.  Default to mode 4, which matches the 720p defaults below.
    # Set CAM_SENSOR_MODE=-1 to let Argus auto-select a mode from the caps.
    sensor_mode: int = int(os.getenv("CAM_SENSOR_MODE", "4"))

    # Video mode
    width: int = int(os.getenv("CAM_WIDTH", "1280"))
    height: int = int(os.getenv("CAM_HEIGHT", "720"))
    fps: int = int(os.getenv("CAM_FPS", "30"))

    # Image orientation
    flip_method: int = int(os.getenv("CAM_FLIP_METHOD", "0"))

    # Auto/manual controls
    ae_lock: bool = env_bool("CAM_AE_LOCK", False)
    awb_lock: bool = env_bool("CAM_AWB_LOCK", False)

    exposure_us: int = int(os.getenv("CAM_EXPOSURE_US", "0"))
    gain: float = float(os.getenv("CAM_GAIN", "0"))
    digital_gain: float = float(os.getenv("CAM_DIGITAL_GAIN", "0"))

    # Image processing
    tnr_mode: int = int(os.getenv("CAM_TNR_MODE", "1"))
    tnr_strength: float = float(os.getenv("CAM_TNR_STRENGTH", "0.5"))

    ee_mode: int = int(os.getenv("CAM_EE_MODE", "1"))
    ee_strength: float = float(os.getenv("CAM_EE_STRENGTH", "0.5"))

    # Appsink
    drop: bool = env_bool("CAM_DROP", True)
    max_buffers: int = int(os.getenv("CAM_MAX_BUFFERS", "1"))
    sync: bool = env_bool("CAM_SYNC", False)


def make_pipeline(cfg: CameraConfig) -> str:
    props = [
        f"sensor-id={cfg.sensor_id}",
        f"tnr-mode={cfg.tnr_mode}",
        f"tnr-strength={cfg.tnr_strength}",
        f"ee-mode={cfg.ee_mode}",
        f"ee-strength={cfg.ee_strength}",
    ]

    # Pin the sensor mode only when explicitly chosen (>= 0).  A negative value
    # means "let Argus auto-select the mode that satisfies the caps below".
    if cfg.sensor_mode >= 0:
        props.insert(1, f"sensor-mode={cfg.sensor_mode}")

    if cfg.ae_lock:
        props.append("aelock=true")

    if cfg.awb_lock:
        props.append("awblock=true")

    # Manual exposure
    if cfg.exposure_us > 0:
        props.append(
            f'exposuretimerange="{cfg.exposure_us} {cfg.exposure_us}"'
        )

    # Manual analog gain
    if cfg.gain > 0:
        props.append(f'gainrange="{cfg.gain} {cfg.gain}"')

    # Manual digital gain
    if cfg.digital_gain > 0:
        props.append(
            f'ispdigitalgainrange="{cfg.digital_gain} {cfg.digital_gain}"'
        )

    pipeline = (
        f"nvarguscamerasrc {' '.join(props)} ! "
        f"video/x-raw(memory:NVMM), "
        f"width={cfg.width}, "
        f"height={cfg.height}, "
        f"framerate={cfg.fps}/1 ! "
        f"nvvidconv flip-method={cfg.flip_method} ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink "
        f"drop={'1' if cfg.drop else '0'} "
        f"max-buffers={cfg.max_buffers} "
        f"sync={'true' if cfg.sync else 'false'}"
    )

    return pipeline


# -----------------------------------------------------------------------------

# All persistent run outputs (video, console log, command CSV) land here, next
# to tracker.py, so they survive across runs regardless of the launch cwd.
_DATA_DIR = Path(__file__).resolve().parent / "data"


# The last few hundred IMPORTANT console lines, for the phone's console panel.
# Per-frame chatter ([CTRL], [PROF]) and GStreamer's banner are left out: the
# point is to read what HAPPENED (phase changes, refusals, FC messages) at arm's
# length, not to scroll a firehose.
CONSOLE = collections.deque(maxlen=300)
_CONSOLE_SEQ = [0]
_CONSOLE_LOCK = threading.Lock()
_CONSOLE_SKIP = ("[CTRL]", "[PROF]", "GST_", "CONSUMER", "[ WARN", "   ",
                 "Camera index", "Camera mode", "Output Stream", "seconds to Run",
                 "Frame Rate")


def console_since(seq):
    """[(seq, 'HH:MM:SS', line)] newer than `seq`."""
    with _CONSOLE_LOCK:
        return [x for x in CONSOLE if x[0] > seq]


def _console_add(line):
    line = line.rstrip()
    if not line or line.startswith(_CONSOLE_SKIP):
        return
    with _CONSOLE_LOCK:
        _CONSOLE_SEQ[0] += 1
        CONSOLE.append((_CONSOLE_SEQ[0], datetime.now().strftime("%H:%M:%S"),
                        line[:220]))


class _Tee:
    """Duplicate a stream to the console AND a log file (line-buffered), so the
    full console output is persisted offline while still showing live.

    Every line written to the FILE is prefixed with a wall-clock timestamp
    (HH:MM:SS.mmm) so log entries can be lined up against the recorded video —
    the video file is named with this run's start time, so log-time minus that
    start is the position in the reel.  The live console stays unprefixed."""
    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh
        self._need_prefix = True          # at the start of a fresh line
        self._partial = ""                # for the phone's console buffer

    @staticmethod
    def _prefix():
        now = datetime.now()
        return f"[{now.strftime('%H:%M:%S')}.{now.microsecond // 1000:03d}] "

    def write(self, data):
        self._stream.write(data)          # console: unchanged
        out = []                          # file: timestamp each line start
        for ch in data:
            if self._need_prefix:
                out.append(self._prefix())
                self._need_prefix = False
            out.append(ch)
            if ch == "\n":
                self._need_prefix = True
        self._fh.write("".join(out))
        buf = self._partial + data
        *lines, self._partial = buf.split("\n")
        for ln in lines:
            _console_add(ln)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()


if __name__ == "__main__":
    # Single timestamp for this run: <stamp>.mp4 / <stamp>.log / commands_<stamp>.csv
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_fh = open(_DATA_DIR / f"{RUN_STAMP}.log", "w", buffering=1)
    sys.stdout = _Tee(sys.stdout, _log_fh)
    sys.stderr = _Tee(sys.stderr, _log_fh)
    print(f"[LOG] console mirrored to {(_DATA_DIR / f'{RUN_STAMP}.log')}")

    cam_cfg = CameraConfig()
    CAP_PIPELINE = make_pipeline(cam_cfg)
    # WASP_MODEL points at a local .pt and skips HuggingFace entirely — that is
    # how sky mode loads its own weights (runs/detect/v0/weights/best.pt), which
    # are not on the Hub at all.  Unset, we fall back to the face detector.
    # Checked here rather than at first inference: a missing weights file in the
    # field should stop the run at startup with the path in the message, not
    # three minutes later inside a detect call.
    _model_env = os.environ.get("WASP_MODEL", "").strip()
    if _model_env:
        model_pth = str(Path(_model_env).expanduser().resolve())
        if not Path(model_pth).is_file():
            print(f"[MODEL] WASP_MODEL is set but there is no file at "
                  f"{model_pth}", file=sys.stderr)
            sys.exit(1)
    else:
        model_pth = hf_hub_download(
            repo_id="iitolstykh/YOLO-Face-Person-Detector",
            filename="yolov8x_person_face.pt",
            repo_type="model"
        )

    # BOTH detectors, resolved now, so the operator can switch target type in
    # the air without the loop ever touching the network or the disk.  The
    # alternate is best-effort: if it cannot be resolved the switch simply
    # refuses at the dashboard and says so, which is far better than a failed
    # download inside the control loop.
    # PREFER A TENSORRT ENGINE next to the weights.  Measured on this Jetson,
    # same 1280x720 input the tracker actually feeds it:
    #
    #     PyTorch  best.pt      35.5 ms   28 fps
    #     TensorRT best.engine  28.1 ms   36 fps   (FP16, built here)
    #
    # and 21 ms of that 28 is inference — the other 7 is CPU letterboxing of the
    # full frame, which is why imgsz makes no difference (34 ms flat at 640, 512
    # AND 416 in PyTorch: the cost is per-call overhead, not pixels).
    #
    # An engine is tied to the exact GPU and TensorRT version that built it, so
    # this is best-effort with a hard fallback: a stale or foreign engine must
    # degrade to the .pt, not stop the flight.  Rebuild with
    #     yolo export model=runs/detect/v0/weights/best.pt format=engine half=True imgsz=640
    if os.environ.get("USE_ENGINE", "1") == "1":
        _eng = Path(model_pth).with_suffix(".engine")
        if _eng.is_file() and _eng.stat().st_mtime >= Path(model_pth).stat().st_mtime:
            try:
                _probe = YOLO(str(_eng), task="detect")
                if len(_probe.names) > 100:
                    # No embedded metadata: the class map is a placeholder
                    # ("class0".."class998"), so target-class resolution would be
                    # a guess.  Refuse it rather than track an arbitrary index.
                    print(f"[MODEL] ignoring {_eng.name}: no class metadata "
                          f"({len(_probe.names)} placeholder names) — "
                          f"re-export it with ultralytics.")
                else:
                    del _probe
                    model_pth = str(_eng)
                    print(f"[MODEL] TensorRT engine selected: {_eng.name}")
            except Exception as _exc:
                print(f"[MODEL] engine {_eng.name} unusable ({_exc}); "
                      f"falling back to {Path(model_pth).name}")

    _alt = ""
    try:
        if _model_env:                       # flying DRONE weights -> alt is the face detector
            _alt = hf_hub_download(
                repo_id="iitolstykh/YOLO-Face-Person-Detector",
                filename="yolov8x_person_face.pt", repo_type="model")
        else:                                # flying the face detector -> alt is the sky model
            _sky = Path(__file__).resolve().parent.parent / "runs/detect/v0/weights/best.pt"
            _alt = str(_sky) if _sky.is_file() else ""
    except Exception as _exc:
        print(f"[TARGET] alternate detector unavailable ({_exc}); "
              f"the in-air target switch will refuse.")
    os.environ["WASP_MODEL_ALT"] = _alt
    os.environ.setdefault("TARGET_KIND", "DRONE" if _model_env else "HUMAN")
    print(f"[TARGET] {os.environ['TARGET_KIND']} active; alternate: "
          f"{_alt or 'NONE'}")

    DroneTracker(
        #model_path   = "./runs/detect/v0/weights/best.pt",
        model_path   = model_pth,
        camera_pipeline=CAP_PIPELINE,
        camera_fps=cam_cfg.fps,
        conf_thresh  = 0.01,
        display_size = (640, 640),
        skip_frames  = 0,
        show_sky     = False,
        use_sky_mask = True,       # ground rejection: the SkyDetector sky gate
        record_dir      = str(_DATA_DIR),   # <stamp>.mp4 lands here
        record_pad_sec  = 1.5,
        log_dir         = str(_DATA_DIR),   # commands_<stamp>.csv lands here
        run_stamp       = RUN_STAMP,        # share the run timestamp
        headless     = os.environ.get("TRACKER_HEADLESS") == "1",
        # Remote seeker view over WiFi.  ON by default because the Jetson flies
        # without a screen and this is the only way to see the HUD; STREAM=0
        # turns it off, STREAM_PORT moves it off 8080.
        #   phone / laptop on the same WiFi -> http://<jetson-ip>:8080/
        stream_port    = (int(os.environ.get("STREAM_PORT", "8080"))
                          if env_bool("STREAM", True) else None),
        stream_quality = int(os.environ.get("STREAM_QUALITY", "70")),
        stream_fps     = float(os.environ.get("STREAM_FPS", "15")),
        # The flight-control link is configured entirely from the environment —
        # see fc_control.commander_from_env for the full list.  The defaults are
        # the deployed ones: talk to the FC through the local MAVProxy router on
        # udpin:127.0.0.1:14551, auto-detect the firmware, and drive yaw + pitch
        # in ACRO.  Useful overrides:
        #   MAV_CONN=/dev/ttyACM0   bypass the router, own the FC serial port
        #   MAV_CONN=none           offline: commands computed and printed only
        #   YAW_REVERSED=1 / PITCH_REVERSED=1   if an axis drives the wrong way
        #   KP_YAW=3.0 KP_PITCH=3.0             more aggressive centring
        #   STICK_LIMIT=0.35                    gentler authority ceiling
        commander    = commander_from_env(log_dir=str(_DATA_DIR)),
    ).run()
