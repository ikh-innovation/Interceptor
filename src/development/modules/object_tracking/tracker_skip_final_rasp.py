"""
tracker_skip_final_rasp.py  –  Drone Intercept Tracker  (PLAN B: Raspberry Pi 5 + Hailo)
=========================================================================================
Edge-portable twin of ``tracker_skip_final.py``.  Same detection *behaviour*
(sky-gated, small-drone-aware, Kalman-coasted, PX4-commanding) but re-architected
to hit real-time FPS on a **Raspberry Pi 5 + Hailo-8/8L AI HAT+** instead of a
Jetson Orin GPU.

Why a rewrite was needed
------------------------
Plan A (Orin) leans on the GPU to run YOLO at a *dynamic* ``imgsz`` — 1280 on the
wide sky crop during SEARCH, 640/416 once locked.  That dynamic high-res pass is
exactly what recovers a 1-px-far drone.  The Hailo NPU cannot do that:

  * There is **no CUDA** and no PyTorch inference on the NPU.
  * The model is compiled to a **fixed input size** (640) baked into the ``.hef``
    at compile time.  You cannot switch to 1280 at runtime.

So the "zoom" that kept far-drone pixels has to come from *how we feed the NPU*,
not from ``imgsz``:

  1. TILING (replaces imgsz=1280 SEARCH)
     The native sky region is split into 640-native-pixel tiles.  A tile maps
     ~1:1 into the model input, so a distant drone keeps as many pixels as it did
     under Plan A's 1280 crop — but each tile is a cheap fixed-640 NPU pass.

  2. MOTION-GATING (the FPS multiplier)
     Sky is mostly static; the enemy drone moves.  We frame-difference the sky
     and only infer tiles that contain motion.  On an empty sky that is *zero*
     NPU passes per frame → the loop runs at capture speed.  A moving far drone
     lights up exactly one tile → one cheap pass.

  3. CROP-SCALE ZOOM (unchanged idea, kept for LOCK/PREDICT)
     Once locked we crop the predicted ROI (small) and letterbox it to 640, so
     the target still fills the input — same "zoom" benefit as Plan A's ROI pass.

KalmanTracker, PX4Commander, the SEARCH→LOCK→PREDICT→LOST state machine and the
HUD are copied verbatim from Plan A so results stay consistent.  SkyDetector is
the one deliberate divergence — hardened for consistency (see its header).  The
detection-reel recorder from Plan A is intentionally REMOVED (Plan B is a live
edge detector, not a capture tool).

Backend abstraction (runs with OR without the NPU)
--------------------------------------------------
Detection is behind ``DetectorBackend`` with two implementations:

  * ``HailoBackend``       – uses ``hailo_platform`` (HailoRT).  Used on the Pi.
  * ``UltralyticsBackend`` – loads ``best.pt`` or an exported ``*_ncnn_model``
                             via Ultralytics.  Runs on this dev box / any CPU.

The backend is auto-selected: if ``hailo_platform`` imports and a ``.hef`` is
present, Hailo is used; otherwise it falls back to Ultralytics.  The tracker
logic is identical either way — develop/benchmark here, deploy on the Pi with no
code change.

Getting the .hef
----------------
The NPU needs a compiled ``.hef``.  ``best.pt`` cannot be compiled on the Pi;
run ``src/development/utils/export_rasp_model.py`` on an x86 box with the Hailo
Dataflow Compiler.  See that file for the full pt→onnx→hef recipe.
"""

import os
os.environ.pop("SESSION_MANAGER", None)

import cv2
import numpy as np
import threading
from pathlib import Path
from time import perf_counter


# ══════════════════════════════════════════════════════════════════════════════
# SKY DETECTOR   (Plan-B: robust + temporally stable — DIVERGES from Plan A)
# ------------------------------------------------------------------------------
# Plan A keyed the sky purely on FIXED colour thresholds (V>=150 AND blue/low-sat),
# decided independently every frame with no memory.  Measured on field2.MP4 that
# is inconsistent: exposure/haze shifts push pixels out of the fixed band so the
# whole mask blinks off (sky fraction seen swinging 0.2%..90%), and it flickers
# frame-to-frame (temporal IoU dipping to ~0).  The gate then intermittently
# rejects valid detections.
#
# This version fixes the two root causes while keeping the SAME public interface
# (mask() -> uint8 mask, background_is_sky()):
#
#   1. ADAPTIVE BRIGHTNESS  — the bright-pixel cut is Otsu-derived per frame
#      (floored/capped), so it tracks exposure instead of a hard 150.
#   2. LOW-TEXTURE CUE      — sky is smooth, ground is textured.  A low local-
#      gradient test flags grey/overcast/blown-out sky that COLOUR alone misses
#      (illumination-invariant), so the mask stops collapsing on white skies.
#   3. TEMPORAL EMA         — the per-frame mask is fused into a running average
#      and re-thresholded.  This bridges momentary dropouts and removes flicker,
#      which is the dominant "not consistent" complaint.  A moderate alpha keeps
#      it responsive to camera motion.
#
# The top-connected + ground-from-bottom structural logic is retained.
# ══════════════════════════════════════════════════════════════════════════════

class SkyDetector:
    # colour prior (kept, but brightness is now adaptive; V_MIN is only a floor)
    # Floor stays at Plan A's 150 ON PURPOSE: the adaptive Otsu cut can therefore
    # only move HIGHER (stricter) than Plan A, never lower — so it can never admit
    # bright ground that Plan A rejected.  Measured leak == Plan A's (12.5% vs
    # 12.4% on field2).  A lower floor caught dusk sky but leaked ~3% more ground,
    # which is the wrong trade for a targeting gate.
    V_MIN  = 150          # brightness FLOOR (adaptive Otsu can only go higher)
    S_MAX  = 60
    H_LO, H_HI = 90, 140

    # texture cue: sky is smooth → low gradient magnitude
    TEXTURE_MAX = 12      # mean |Laplacian| below this (per neighbourhood) = smooth
    TEXTURE_BLUR = 9      # neighbourhood size the gradient is averaged over
    # The smoothness cue must NOT admit bright smooth GROUND (roads, water, snow,
    # fields).  Sky — even grey/overcast — is low-saturation; saturated smooth
    # ground is not.  So the smooth cue is bounded by this saturation ceiling.
    SMOOTH_S_MAX = 90

    RING_PAD_FRAC = 0.6
    RING_PAD_MIN  = 12
    SKY_FRAC      = 0.60

    TOP_BAND = 3

    BOTTOM_BAND     = 3
    GROUND_MIN_FRAC = 0.02

    # temporal smoothing
    EMA_ALPHA   = 0.5     # weight of the current frame (higher = more responsive)
    EMA_ON_THR  = 0.45    # a pixel is "sky" once its smoothed score crosses this
    EMA_RESET_FRAC = 0.001  # if a raw frame is ~empty, don't let it wipe the EMA

    def __init__(self):
        self._k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._ema = None          # float32 [0,1] running sky-probability map

    def _raw_mask(self, frame_bgr):
        """Single-frame sky estimate: adaptive brightness ∧ (blue ∨ low-sat ∨ smooth)."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # adaptive brightness: Otsu split, floored so a dark frame can't call its
        # own brightest half "sky", capped so a washed-out frame stays permissive.
        otsu, _ = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        v_cut   = int(min(200, max(self.V_MIN, otsu)))
        bright  = V >= v_cut

        blue   = (H >= self.H_LO) & (H <= self.H_HI)
        lowsat = S <= self.S_MAX

        # low-texture (smoothness) — catches grey/overcast sky colour misses.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lap  = np.abs(cv2.Laplacian(gray, cv2.CV_16S, ksize=3)).astype(np.float32)
        lap  = cv2.blur(lap, (self.TEXTURE_BLUR, self.TEXTURE_BLUR))
        smooth = (lap <= self.TEXTURE_MAX) & (S <= self.SMOOTH_S_MAX)

        sky = (bright & (blue | lowsat | smooth)).astype(np.uint8) * 255
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

    def mask(self, frame_bgr):
        raw = self._raw_mask(frame_bgr)

        # temporal EMA: fuse this frame's estimate into the running probability.
        cur = (raw > 0).astype(np.float32)
        if self._ema is None or self._ema.shape != cur.shape:
            self._ema = cur.copy()
        elif cur.mean() < self.EMA_RESET_FRAC and self._ema.mean() > 0.1:
            # a momentary all-empty frame (glare/exposure blip) must NOT wipe a
            # well-established sky — let the EMA decay slowly instead of snapping.
            self._ema *= (1.0 - self.EMA_ALPHA * 0.25)
        else:
            self._ema = self.EMA_ALPHA * cur + (1.0 - self.EMA_ALPHA) * self._ema

        return (self._ema >= self.EMA_ON_THR).astype(np.uint8) * 255

    def _reclaim_embedded_blobs(self, sky):
        Hm, Wm  = sky.shape
        non_sky = (sky == 0).astype(np.uint8)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(non_sky)
        if n <= 1:
            return np.full_like(sky, 255)

        band          = max(1, self.BOTTOM_BAND)
        bottom_labels = np.unique(labels[Hm - band:, :])
        bottom_labels = bottom_labels[bottom_labels != 0]

        min_area = self.GROUND_MIN_FRAC * Hm * Wm
        keep     = [lab for lab in bottom_labels
                    if stats[lab, cv2.CC_STAT_AREA] >= min_area]

        ground = np.isin(labels, keep) if keep else np.zeros(sky.shape, bool)
        out    = np.where(ground, 0, 255).astype(np.uint8)
        out[sky > 0] = 255
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
# KALMAN   (copied verbatim from tracker_skip_final.py)
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
# PX4 COMMANDER   (copied verbatim from tracker_skip_final.py)
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
# DETECTION BACKENDS
# ------------------------------------------------------------------------------
# A backend takes a BGR image (any size) and returns a list of boxes in that
# image's own pixel coordinates:  [(x1, y1, x2, y2, conf), ...].
# Letterboxing to the model's fixed input size happens INSIDE the backend, so
# the tracker never has to know the network resolution.
# ══════════════════════════════════════════════════════════════════════════════

def _letterbox(img, new_size, color=(114, 114, 114)):
    """Resize keeping aspect ratio, pad to a square `new_size`.
    Returns (padded_img, scale, pad_x, pad_y) so boxes can be mapped back."""
    h, w = img.shape[:2]
    s = min(new_size / w, new_size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    px, py = (new_size - nw) // 2, (new_size - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, s, px, py


class DetectorBackend:
    """Interface. `input_size` is the network's fixed square input edge."""
    input_size = 640

    def infer(self, bgr):
        """-> list of (x1, y1, x2, y2, conf) in `bgr` pixel coordinates."""
        raise NotImplementedError

    def close(self):
        pass


class UltralyticsBackend(DetectorBackend):
    """Portable fallback: Ultralytics YOLO on CPU/GPU.

    Accepts a PyTorch checkpoint (``best.pt``) or an exported NCNN directory
    (``best_ncnn_model``).  Runs anywhere — used to develop/benchmark the
    pipeline off-target.  NOTE: this is NOT the Hailo path; on a bare Pi CPU it
    is the slow reference, not the deployment target.
    """
    def __init__(self, model_path, conf_thresh, input_size=640, class_id=None):
        from ultralytics import YOLO
        self.model      = YOLO(model_path)
        self.conf       = conf_thresh
        self.input_size = int(input_size)
        self.class_id   = class_id

    def infer(self, bgr):
        res = self.model(bgr, imgsz=self.input_size, conf=self.conf, verbose=False)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return []
        b     = res[0].boxes
        xyxy  = b.xyxy.cpu().numpy()
        confs = b.conf.cpu().numpy()
        clss  = b.cls.cpu().numpy() if b.cls is not None else np.zeros(len(confs))
        out = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            if c < self.conf:
                continue
            if self.class_id is not None and int(k) != self.class_id:
                continue
            out.append((float(x1), float(y1), float(x2), float(y2), float(c)))
        return out


class HailoBackend(DetectorBackend):
    """HailoRT inference on a compiled .hef (Hailo-8 / 8L, e.g. Pi AI HAT+).

    Assumes the .hef was compiled WITH on-chip NMS (``nms_postprocess`` in the
    model script — see export_rasp_model.py).  With on-chip NMS the output is
    already-decoded, per-class detections in NORMALISED [0,1] letterbox coords,
    which we un-letterbox back to `bgr` pixels.

    ┌─ ON-DEVICE VALIDATION NEEDED ───────────────────────────────────────────┐
    │ I cannot run Hailo hardware here, so three things must be confirmed once  │
    │ on the Pi (they vary with HailoRT version / how the .hef was compiled):   │
    │   1. Input dtype — most YOLO HEFs take UINT8 NHWC (quant baked in). If     │
    │      yours expects float32, set input_uint8=False.                        │
    │   2. NMS output layout — the HailoNMS format below is [y1,x1,y2,x2,score] │
    │      normalised, grouped per class. Print `raw` once to confirm.          │
    │   3. VStream names — pulled from the HEF automatically; no hardcoding.     │
    └───────────────────────────────────────────────────────────────────────┘
    """
    def __init__(self, hef_path, conf_thresh, class_id=0, input_uint8=True):
        from hailo_platform import (HEF, VDevice, ConfigureParams,
                                     InputVStreamParams, OutputVStreamParams,
                                     InferVStreams, HailoStreamInterface,
                                     FormatType)

        self.conf        = conf_thresh
        self.class_id    = class_id
        self.input_uint8 = input_uint8

        self._hef    = HEF(str(hef_path))
        self._device = VDevice()

        cfg = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe)
        self._ng = self._device.configure(self._hef, cfg)[0]
        self._ng_params = self._ng.create_params()

        in_info  = self._hef.get_input_vstream_infos()[0]
        self.input_name = in_info.name
        # vstream shape is (H, W, C); network is square so either dim works.
        self.input_size = int(in_info.shape[0])

        fmt = FormatType.UINT8 if input_uint8 else FormatType.FLOAT32
        self._in_params  = InputVStreamParams.make(self._ng, format_type=fmt)
        self._out_params = OutputVStreamParams.make(
            self._ng, format_type=FormatType.FLOAT32)

        # Keep the network group activated for the whole run (avoids per-frame
        # (de)activation overhead). Released in close().
        self._activation = self._ng.activate(self._ng_params)
        self._activation.__enter__()
        self._pipeline = InferVStreams(self._ng, self._in_params, self._out_params)
        self._pipeline.__enter__()

    def infer(self, bgr):
        lb, s, px, py = _letterbox(bgr, self.input_size)
        data = lb if self.input_uint8 else (lb.astype(np.float32) / 255.0)
        raw = self._pipeline.infer({self.input_name: data[np.newaxis, ...]})

        # On-chip NMS: raw[out_name] is per-batch, per-class list of detections
        # shaped (n, 5) as [y_min, x_min, y_max, x_max, score] in [0,1].
        out_name = next(iter(raw))
        dets = raw[out_name]
        boxes = self._parse_nms(dets)

        H, W = bgr.shape[:2]
        result = []
        for (ymin, xmin, ymax, xmax, score) in boxes:
            if score < self.conf:
                continue
            # normalised letterbox -> letterbox px -> un-pad -> original px
            x1 = (xmin * self.input_size - px) / s
            y1 = (ymin * self.input_size - py) / s
            x2 = (xmax * self.input_size - px) / s
            y2 = (ymax * self.input_size - py) / s
            x1 = float(np.clip(x1, 0, W)); x2 = float(np.clip(x2, 0, W))
            y1 = float(np.clip(y1, 0, H)); y2 = float(np.clip(y2, 0, H))
            if x2 > x1 and y2 > y1:
                result.append((x1, y1, x2, y2, float(score)))
        return result

    def _parse_nms(self, dets):
        """Flatten HailoRT on-chip NMS output to a list of (y1,x1,y2,x2,score).

        HailoRT hands NMS results back in a few shapes across versions; handle
        the common ones (batched list-of-per-class-arrays, or a single array).
        """
        out = []
        # unwrap batch dim
        if isinstance(dets, (list, tuple)):
            dets = dets[0]
        if isinstance(dets, (list, tuple)):
            # per-class: list where index == class id, each an (n,5) array
            for cls_id, arr in enumerate(dets):
                if self.class_id is not None and cls_id != self.class_id:
                    continue
                arr = np.asarray(arr)
                if arr.size == 0:
                    continue
                for row in arr.reshape(-1, arr.shape[-1]):
                    out.append(tuple(float(v) for v in row[:5]))
        else:
            arr = np.asarray(dets)
            for row in arr.reshape(-1, arr.shape[-1]):
                out.append(tuple(float(v) for v in row[:5]))
        return out

    def close(self):
        try:
            self._pipeline.__exit__(None, None, None)
            self._activation.__exit__(None, None, None)
        except Exception:
            pass


def build_backend(model_path=None, hef_path=None, conf_thresh=0.05,
                  input_size=640, class_id=0, force=None):
    """Auto-select the detection backend.

    force = "hailo" | "ultralytics" | None (auto).
    Auto: use Hailo if hailo_platform imports AND a .hef exists; else Ultralytics.
    """
    want_hailo = force == "hailo" or (force is None and hef_path and Path(hef_path).exists())
    if want_hailo:
        try:
            b = HailoBackend(hef_path, conf_thresh, class_id=class_id)
            print(f"[BACKEND] Hailo NPU  | hef={hef_path}  input={b.input_size}")
            return b
        except Exception as e:
            if force == "hailo":
                raise
            print(f"[BACKEND] Hailo unavailable ({e}); falling back to Ultralytics.")

    b = UltralyticsBackend(model_path, conf_thresh, input_size=input_size,
                           class_id=class_id)
    print(f"[BACKEND] Ultralytics  | model={model_path}  imgsz={b.input_size}  "
          f"(reference/off-target path — NOT the NPU)")
    return b


# ══════════════════════════════════════════════════════════════════════════════
# THREADED VIDEO SOURCE
# ------------------------------------------------------------------------------
# Decodes frames in a background thread so file I/O / demux overlaps with NPU
# inference.  Always keeps only the LATEST frame (drop-old policy) so the tracker
# never falls behind real time on the Pi.
# ══════════════════════════════════════════════════════════════════════════════

class ThreadedVideo:
    def __init__(self, path, drop_old=True):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps != self.fps or self.fps <= 1:
            self.fps = 30.0
        self.drop_old = drop_old
        self._lock    = threading.Lock()
        self._latest  = None       # (idx, frame)
        self._idx     = 0
        self._stopped = False
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self._stopped = True
                break
            self._idx += 1
            with self._lock:
                self._latest = (self._idx, np.ascontiguousarray(frame))

    def read(self):
        """Return (idx, frame) of the newest decoded frame, or (None, None)."""
        with self._lock:
            latest = self._latest
            self._latest = None if self.drop_old else latest
        return latest if latest is not None else (None, None)

    @property
    def finished(self):
        return self._stopped and self._latest is None

    def release(self):
        self._stopped = True
        self._t.join(timeout=1.0)
        self.cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# DRONE TRACKER  (Pi 5 + Hailo: tiled + motion-gated detection)
# ══════════════════════════════════════════════════════════════════════════════

class DroneTrackerRasp:
    _WIN = "WASP DRONE TRACKER  [PLAN B — PI5 + HAILO]"
    _COLOR = {
        "SEARCH":  (0,   0,   255),
        "LOCK":    (0,   255, 0),
        "PREDICT": (0,   165, 255),
    }

    # ── HUD palette (BGR) ────────────────────────────────────────────────────
    C_LOCK    = ( 96, 255, 120)
    C_SEARCH  = ( 70, 180, 255)
    C_PREDICT = (230, 200, 100)
    C_HUD     = (200, 230, 205)
    C_DIM     = (110, 120, 110)
    C_WARN    = ( 70,  70, 240)
    C_INK     = ( 22,  26,  22)
    _ACCENT   = {"SEARCH": C_SEARCH, "LOCK": C_LOCK, "PREDICT": C_PREDICT}
    _HUD_FONT = cv2.FONT_HERSHEY_DUPLEX

    INSET_SRC  = 100
    INSET_DISP = 180

    CROP_PAD   = 24     # native px padding around an ROI crop

    # ── tiling (replaces Plan A's imgsz=1280 search) ──────────────────────────
    TILE_OVERLAP    = 96     # native-px overlap between tiles (catch edge drones)
    MAX_TILES       = 6      # hard cap on tiles inferred per SEARCH frame
    FULL_SWEEP_EVERY = 60    # every N search frames, ignore motion & sweep all

    # ── motion gate ───────────────────────────────────────────────────────────
    MOTION_THRESH   = 18     # abs-diff threshold on the display-res grey frame
    MOTION_MIN_AREA = 4      # min moving px in a tile's footprint to infer it

    # Size gate (same meaning as Plan A): "big/close" target skips zoom effort.
    BIG_OBJ_FRAC = 0.06

    def __init__(
        self,
        backend,
        video_path     = "./data/videos_drones/field2.MP4",
        conf_thresh    = 0.05,
        display_size   = (640, 640),
        skip_frames    = 0,
        show_sky       = True,
        headless       = False,
        max_frames     = None,
        sky_every      = 4,      # recompute sky mask every N frames (CPU-bound)
        use_motion_gate = True,
        threaded_capture = True,
    ):
        self.backend      = backend
        self.video_path   = video_path
        self.conf_thresh  = conf_thresh
        self.display_size = display_size
        self.fw, self.fh  = display_size
        self.skip_frames  = max(0, int(skip_frames))
        self.show_sky     = show_sky
        self.headless       = headless
        self.max_frames     = max_frames
        self.sky_every      = max(1, int(sky_every))
        self.use_motion_gate = use_motion_gate
        self.threaded_capture = threaded_capture

        self.kalman  = KalmanTracker()
        self.px4     = PX4Commander()
        self.sky     = SkyDetector()

        self.state     = "SEARCH"
        self.missed    = 0
        self.last_bbox = None
        self.pos       = None
        self.layer     = "STANDBY"
        self.conf      = None
        self._frame_idx = 0
        self._search_frames = 0

        self.sky_mask         = None
        self._ground_rejected = False
        self._prev_gray       = None
        self._motion_mask     = None
        self._tiles_dropped   = 0

        self.full_w = self.full_h = None

        self.MISS_TO_PREDICT = 2
        self.MISS_TO_SEARCH  = 20
        self.MIN_ROI_HALF    = 50

    # ──────────────────────────────────────────────────────────────────────────
    # coordinate helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _accept_box(self, x1n, y1n, x2n, y2n, conf):
        """Native-space box → display space, with the sky gate applied.
        (Identical semantics to Plan A._accept_box.)"""
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

    def _infer_native(self, frame_full, region_native):
        """Run ONE backend pass on a native sub-region; return best sky-valid
        detection (or None).  The backend letterboxes the crop internally, so a
        small ROI crop is effectively 'zoomed' to the network input — same
        pixel-preservation trick as Plan A's high-imgsz ROI pass."""
        x1r, y1r, x2r, y2r = region_native
        x1r = max(0, int(x1r)); y1r = max(0, int(y1r))
        x2r = min(self.full_w, int(x2r)); y2r = min(self.full_h, int(y2r))
        if x2r - x1r < 2 or y2r - y1r < 2:
            return None

        crop = frame_full[y1r:y2r, x1r:x2r]
        boxes = self.backend.infer(crop)
        if not boxes:
            return None

        boxes.sort(key=lambda b: b[4], reverse=True)
        for (x1, y1, x2, y2, conf) in boxes:
            if conf < self.conf_thresh:
                break
            det = self._accept_box(x1 + x1r, y1 + y1r, x2 + x1r, y2 + y1r, conf)
            if det is not None:
                return det
        return None

    def _tiles(self, region_native):
        """Split a native region into <=input_size tiles with overlap.
        Yields (x1, y1, x2, y2) native tiles that keep far-drone pixel density."""
        x1, y1, x2, y2 = region_native
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(self.full_w, int(x2)); y2 = min(self.full_h, int(y2))
        T   = self.backend.input_size
        ov  = min(self.TILE_OVERLAP, T // 2)
        step = max(1, T - ov)
        xs = list(range(x1, max(x1 + 1, x2 - T + 1), step)) or [x1]
        ys = list(range(y1, max(y1 + 1, y2 - T + 1), step)) or [y1]
        if xs[-1] + T < x2:
            xs.append(max(x1, x2 - T))
        if ys[-1] + T < y2:
            ys.append(max(y1, y2 - T))
        for ty in ys:
            for tx in xs:
                yield (tx, ty, min(x2, tx + T), min(y2, ty + T))

    def _tile_has_motion(self, tile_native):
        """True if the tile's display-space footprint contains enough motion.
        With no motion mask yet (first frames) treat everything as active."""
        if self._motion_mask is None:
            return True
        x1, y1, x2, y2 = tile_native
        sx = self.fw / self.full_w
        sy = self.fh / self.full_h
        dx1, dy1 = int(x1 * sx), int(y1 * sy)
        dx2, dy2 = int(x2 * sx), int(y2 * sy)
        sub = self._motion_mask[dy1:dy2, dx1:dx2]
        if sub.size == 0:
            return False
        return cv2.countNonZero(sub) >= self.MOTION_MIN_AREA

    def _search_tiled(self, frame_full, force_full):
        """SEARCH pass: motion-gated tiling over the sky region.
        Returns the best sky-valid detection across inferred tiles (or None)."""
        sky_bbox = self._sky_bbox_native()
        if sky_bbox is None:
            return None

        tiles = list(self._tiles(sky_bbox))
        # Motion gate — unless a periodic full sweep is due or we have no history.
        gate = self.use_motion_gate and not force_full and self._motion_mask is not None
        if gate:
            active = [t for t in tiles if self._tile_has_motion(t)]
        else:
            active = tiles

        # Bound per-frame NPU work; prefer motion-richest tiles when over cap.
        self._tiles_dropped = 0
        if len(active) > self.MAX_TILES:
            if self._motion_mask is not None:
                active.sort(key=self._tile_motion_score, reverse=True)
            self._tiles_dropped = len(active) - self.MAX_TILES
            active = active[:self.MAX_TILES]

        best = None
        for tile in active:
            det = self._infer_native(frame_full, tile)
            if det is not None and (best is None or det[6] > best[6]):
                best = det
        return best

    def _tile_motion_score(self, tile_native):
        x1, y1, x2, y2 = tile_native
        sx = self.fw / self.full_w
        sy = self.fh / self.full_h
        sub = self._motion_mask[int(y1 * sy):int(y2 * sy), int(x1 * sx):int(x2 * sx)]
        return 0 if sub.size == 0 else cv2.countNonZero(sub)

    def _target_is_small(self):
        if self.last_bbox is None:
            return True
        bw, bh = self.last_bbox[2], self.last_bbox[3]
        return max(bw, bh) < self.fh * self.BIG_OBJ_FRAC

    def _sky_bbox_native(self):
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

    # ── velocity-adaptive ROI (display coords) — same as Plan A ────────────────

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
    # MOTION
    # ──────────────────────────────────────────────────────────────────────────

    def _update_motion(self, frame_disp):
        gray = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._prev_gray is None:
            self._prev_gray = gray
            self._motion_mask = None
            return
        diff = cv2.absdiff(gray, self._prev_gray)
        _, m = cv2.threshold(diff, self.MOTION_THRESH, 255, cv2.THRESH_BINARY)
        m = cv2.dilate(m, None, iterations=2)
        # only motion over sky counts (ground clutter / horizon shake ignored)
        if self.sky_mask is not None:
            m = cv2.bitwise_and(m, self.sky_mask)
        self._motion_mask = m
        self._prev_gray   = gray

    # ──────────────────────────────────────────────────────────────────────────
    # FRAME PROCESSING
    # ──────────────────────────────────────────────────────────────────────────

    def _process(self, frame_full, frame_disp):
        self._ground_rejected = False

        t0      = perf_counter()
        kf_pred = self.kalman.predict() if self.kalman.initialized else None
        t_kf    = (perf_counter() - t0) * 1000

        t0       = perf_counter()
        roi      = None
        yolo_det = None
        small    = self._target_is_small()

        if self.state in ("LOCK", "PREDICT") and kf_pred:
            # locked: cheap single-crop ROI pass (the crop 'zooms' via letterbox)
            roi      = self._predict_roi()
            yolo_det = self._infer_native(frame_full, self._roi_disp_to_native(roi))
            if yolo_det is None:
                # lost inside ROI → fall back to a tiled sky search this frame
                yolo_det = self._search_tiled(frame_full, force_full=True)
                roi = None
        else:
            self._search_frames += 1
            force_full = (self._search_frames % self.FULL_SWEEP_EVERY == 0)
            yolo_det = self._search_tiled(frame_full, force_full=force_full)

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
    # VISUALISATION  (copied from Plan A; skipped entirely when headless)
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

    @staticmethod
    def _panel(frame, p1, p2, fill, alpha, border=None):
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
        font = font or self._HUD_FONT
        cv2.putText(frame, txt, (org[0] + 1, org[1] + 1), font, scale,
                    (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(frame, txt, org, font, scale, color, thick, cv2.LINE_AA)

    @staticmethod
    def _corner_box(frame, x1, y1, x2, y2, color, length=16, thick=2):
        L = max(5, int(min(length, (x2 - x1) / 2, (y2 - y1) / 2)))
        for (cx, cy, sx, sy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                                 (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + sx * L, cy), color, thick, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + sy * L), color, thick, cv2.LINE_AA)

    def _reticle(self, frame, cx, cy, color):
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
        h, w = frame.shape[:2]
        m, L = 6, 22
        for (cx, cy, sx, sy) in ((m, m, 1, 1), (w - m, m, -1, 1),
                                 (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + sx * L, cy), color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + sy * L), color, 1, cv2.LINE_AA)

    def _draw_header(self, frame, accent, fps):
        w = frame.shape[1]
        self._panel(frame, (0, 0), (w, 30), self.C_INK, 0.45)
        cv2.line(frame, (0, 30), (w, 30), accent, 1, cv2.LINE_AA)

        title = "WASP  //  SKY-TRACK  [PI5+HAILO]"
        self._text(frame, title, (12, 21), 0.5, self.C_HUD, 1)

        (tw, _), _ = cv2.getTextSize(title, self._HUD_FONT, 0.5, 1)
        self._text(frame, f"IKH - DEFENSE LABS   {fps:4.1f} FPS",
                   (12 + tw + 20, 21), 0.5, self.C_DIM, 1)

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

    def _draw(self, frame, frame_full, pos, roi, cmd, fps):
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

        self._draw_header(frame, accent, fps)
        self._draw_telemetry(frame, cmd, pos, cx_f, cy_f)
        self._draw_inset(frame, frame_full, pos)

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        if self.threaded_capture:
            src = ThreadedVideo(self.video_path, drop_old=True)
            src_fps = src.fps
        else:
            src = None
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open: {self.video_path}")
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            if not src_fps or src_fps != src_fps or src_fps <= 1:
                src_fps = 30.0

        if not self.headless:
            cv2.namedWindow(self._WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._WIN, *self.display_size)

        period      = self.skip_frames + 1
        fps_ema     = 0.0
        loop_prev   = perf_counter()

        while True:
            # ── acquire next frame ────────────────────────────────────────────
            if src is not None:
                idx, frame_full = src.read()
                if frame_full is None:
                    if src.finished:
                        break
                    continue                       # decoder hasn't produced yet
                self._frame_idx = idx
            else:
                ret, frame_full = cap.read()
                if not ret or frame_full is None:
                    break
                frame_full = np.ascontiguousarray(frame_full)
                self._frame_idx += 1

            self.full_h, self.full_w = frame_full.shape[:2]
            frame_disp = np.ascontiguousarray(cv2.resize(frame_full, self.display_size))
            cx_f, cy_f = self.fw // 2, self.fh // 2

            # ── sky mask (cached; recomputed every sky_every frames) ──────────
            if self.sky_mask is None or self._frame_idx % self.sky_every == 0:
                self.sky_mask = self.sky.mask(frame_disp)
            if self.use_motion_gate:
                self._update_motion(frame_disp)

            is_yolo_frame = (self._frame_idx % period == 0)

            if is_yolo_frame:
                fused, t_kf, t_yolo, roi, detected = self._process(frame_full, frame_disp)
                skip_tag = ""
            else:
                t0    = perf_counter()
                fused = self.kalman.predict() if self.kalman.initialized else None
                t_kf  = (perf_counter() - t0) * 1000
                t_yolo, roi, detected = 0.0, None, False
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

            # ── FPS (whole-loop EMA) ──────────────────────────────────────────
            now = perf_counter()
            inst_fps = 1.0 / max(1e-6, now - loop_prev)
            loop_prev = now
            fps_ema = inst_fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst_fps

            if not self.headless:
                self._draw(frame_disp, frame_full, fused, roi, cmd, fps_ema)

            drop_tag = f"  drop:{self._tiles_dropped}t" if self._tiles_dropped else ""
            timing = (f"YOLO:{t_yolo:5.1f}ms  KF:{t_kf:.3f}ms  "
                      f"| loop:{fps_ema:4.1f}fps{skip_tag}{drop_tag}")
            print(f"{timing}\n{PX4Commander.format(cmd, self.state)}\n")

            if not self.headless:
                cv2.imshow(self._WIN, frame_disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if self.max_frames is not None and self._frame_idx >= self.max_frames:
                break

        if src is not None:
            src.release()
        else:
            cap.release()
        self.backend.close()
        if not self.headless:
            cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Backend auto-selects: Hailo NPU if the .hef exists and hailo_platform is
    # installed (on the Pi), otherwise the Ultralytics reference path (dev box).
    HEF_PATH   = "./runs/detect/v0/weights/best.hef"
    MODEL_PATH = "./runs/detect/v0/weights/best.pt"   # fallback / off-target

    backend = build_backend(
        model_path  = MODEL_PATH,
        hef_path    = HEF_PATH,
        conf_thresh = 0.05,
        input_size  = 640,
        class_id    = 0,
        force       = os.environ.get("TRACKER_BACKEND") or None,   # "hailo"/"ultralytics"
    )

    DroneTrackerRasp(
        backend        = backend,
        video_path     = "./data/videos_drones/field2.MP4",
        conf_thresh    = 0.05,
        display_size   = (640, 640),
        skip_frames    = 0,
        show_sky       = True,
        headless       = os.environ.get("TRACKER_HEADLESS") == "1",
        use_motion_gate = True,
        threaded_capture = True,
    ).run()
