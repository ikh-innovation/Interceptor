
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
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
cv2.setNumThreads(int(os.environ.get("CV_THREADS", "4")))
from dataclasses import dataclass, replace as _dc_replace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fc_control import AcroCommander, ControlLaw, commander_from_env
from mjpeg_stream import MJPEGServer
import mission as mission_mod
from follower import TargetFollower, search_radius



class SkyDetector:
    V_MIN  = 150
    S_MAX  = 60
    H_LO, H_HI = 90, 140

    RING_PAD_FRAC = 0.6
    RING_PAD_MIN  = 12
    SKY_FRAC      = 0.60

    TOP_BAND = 3

    BOTTOM_BAND     = 3
    GROUND_MIN_FRAC = 0.02

    def __init__(self, scale=1.0):
        self.scale = float(scale)
        k = 7 if self.scale >= 1.0 else max(3, int(round(7 * self.scale)) | 1)
        self._k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def mask(self, frame_bgr):
        if self.scale < 1.0:
            h, w = frame_bgr.shape[:2]
            small = cv2.resize(frame_bgr, (max(1, int(w * self.scale)),
                                           max(1, int(h * self.scale))),
                               interpolation=cv2.INTER_AREA)
            return cv2.resize(self._mask(small), (w, h),
                              interpolation=cv2.INTER_NEAREST)
        return self._mask(frame_bgr)

    def _mask(self, frame_bgr):
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
        return int(p[0, 0]), int(p[1, 0])

    def update(self, x, y):
        if not self.initialized:
            self.init(x, y)
            return
        self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))

    def velocity(self):
        if not self.initialized:
            return 0.0, 0.0
        return float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0])

    def pos_sigma(self):
        if not self.initialized:
            return 80.0, 80.0
        P = self.kf.errorCovPost
        return float(np.sqrt(abs(float(P[0, 0])))), float(np.sqrt(abs(float(P[1, 1]))))



class DroneTracker:
    _WIN = "WASP DRONE TRACKER  [SKY + ADAPTIVE ZOOM]"
    _COLOR = {
        "SEARCH":  (0,   0,   255),
        "LOCK":    (0,   255, 0),
        "PREDICT": (0,   165, 255),
    }

    C_LOCK    = ( 96, 255, 120)
    C_SEARCH  = ( 70, 180, 255)
    C_PREDICT = (230, 200, 100)
    C_HUD     = (200, 230, 205)
    C_DIM     = (110, 120, 110)
    C_WARN    = ( 70,  70, 240)
    C_INK     = ( 22,  26,  22)
    _ACCENT   = {"SEARCH": C_SEARCH, "LOCK": C_LOCK, "PREDICT": C_PREDICT}

    C_ALT     = (217, 111,  30)
    C_ACRO    = ( 31, 123, 224)
    C_PILOT   = (190, 200, 195)
    _HUD_FONT = cv2.FONT_HERSHEY_DUPLEX

    INSET_SRC  = 100
    INSET_DISP = 180

    IMGSZ_SEARCH = 640
    IMGSZ_TRACK  = 640
    IMGSZ_CLOSE  = 416
    CROP_PAD     = 24

    BIG_OBJ_FRAC = 0.06

    HUD_PANEL       = os.environ.get("HUD_PANEL", "0") == "1"

    GATE_ENABLE     = True
    GATE_PX_PER_S   = 1500.0
    GATE_FLOOR_PX   = 60.0
    GATE_MAX_REJECT = 3
    USE_CLAHE    = False

    REF_FPS     = 30.0
    cam_mode    = 30
    cam_aspect  = 4.0 / 3.0
    ref_display = None
    _ui_fps     = 0.0

    def __init__(
        self,
        model_path      = "/runs/detect/v0/weights/best.pt",
        camera_pipeline = None,
        camera_fps      = 30.0,
        camera_config   = None,
        conf_thresh    = 0.05,
        display_size   = (640, 640),
        skip_frames    = 0,
        show_sky       = True,
        use_sky_mask   = True,
        record_dir     = None,
        record_pad_sec = 2.0,
        log_dir        = None,
        run_stamp      = None,
        headless       = False,
        max_frames     = None,
        stream_port    = None,
        stream_quality = 70,
        stream_fps     = 15.0,
        commander      = None,
    ):
        from ultralytics import YOLO
        self.model        = (YOLO(model_path, task="detect")
                             if str(model_path).endswith(".engine")
                             else YOLO(model_path))
        names = self.model.names
        # Ultralytics changes OpenCV's global thread count while loading. Apply
        # the capture/resize setting after that initialization, not before it.
        cv2.setNumThreads(int(os.environ.get("CV_THREADS", "4")))
        print(f"[CPU] OpenCV threads: {cv2.getNumThreads()}")
        names = names if isinstance(names, dict) else dict(enumerate(names))
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
            self.target_class = (next(iter(names)) if len(names) == 1
                                 else None)
            if self.target_class is None:
                raise RuntimeError(
                    f"Cannot tell which class to track in {names}. "
                    "Set TARGET_CLASS to the index or the label.")
        print(f"[MODEL] single-class detection: class {self.target_class} "
              f"('{names[self.target_class]}') of {dict(names)}, max 1 object")
        print(f"[MODEL] weights: {model_path}")
        self.camera_pipeline = camera_pipeline
        self.camera_fps      = float(camera_fps)
        self.camera_config   = camera_config
        self.using_engine = str(model_path).endswith(".engine")
        if self.using_engine:
            self.IMGSZ_SEARCH = self.IMGSZ_TRACK = self.IMGSZ_CLOSE = 640
            print("[MODEL] engine input is fixed at 640 — IMGSZ_* aligned")

        self.conf_thresh  = conf_thresh
        self.display_size = display_size
        self.fw, self.fh  = display_size
        self.ref_display  = tuple(display_size)
        self.skip_frames  = max(0, int(skip_frames))
        self.show_sky     = show_sky
        self.use_sky_mask = use_sky_mask
        self.record_dir      = record_dir
        self.record_pad_sec  = float(record_pad_sec)
        self.run_stamp = run_stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.headless     = headless
        self.max_frames   = max_frames
        self.stream_port    = int(stream_port) if stream_port else None
        self.stream_quality = int(stream_quality)
        self.stream_fps     = float(stream_fps)
        self.stream         = None

        self.kalman  = KalmanTracker()
        self.ap      = commander or commander_from_env(
            log_dir=(str(log_dir) if log_dir else str(_DATA_DIR)))
        self.sky     = SkyDetector(
            scale=float(os.environ.get("SKY_MASK_SCALE", "0.5")))
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
        self.follower        = TargetFollower()
        self.use_follower    = os.environ.get("FOLLOW", "1") != "0"
        _saved = mission_mod.load_detection_defaults()
        if "follow" in _saved and "FOLLOW" not in os.environ:
            self.use_follower = _saved["follow"]
        self.follower_default = self.use_follower
        self.follow_hits     = 0
        self._gate_rejects   = 0
        self.gated_total     = 0
        self.gate_note       = ""
        self._t_last_frame   = None
        self._frame_dt       = None
        self.mode_name       = os.environ.get("WASP_MODE_SPEED", "MEDIUM").upper()
        self.target_kind     = os.environ.get("TARGET_KIND", "DRONE").upper()
        self.model_path      = model_path
        self._last_cmd       = None
        self._last_sent      = False
        self._ui_fps         = 0.0
        self._t_ui_prev      = None
        self._prof = {}
        self._prof_n = 0
        self.prof_ms = {}
        self.PROFILE = os.environ.get("PROFILE", "1") == "1"
        self.set_mode(self.mode_name)
        self.MISS_TO_SEARCH  = 20
        self.MIN_ROI_HALF    = 50

        self.CAM_OPEN_RETRIES  = 5
        self.CAM_WARMUP_READS  = 30
        self.CAM_RETRY_WAIT_S  = 0.8

        self.cam_default = 30
        start = 30
        if self.camera_config is not None:
            if _saved.get("cam") in CAM_MODES:
                self.cam_default = _saved["cam"]
            _env_cam = os.environ.get("CAM_MODE", "").strip()
            start = int(_env_cam) if _env_cam.isdigit() and int(_env_cam) in CAM_MODES \
                else self.cam_default
        self._cam_req  = None
        self._cam_busy = False
        self._apply_cam_mode(start)
        self._update_log_context()

    def _update_log_context(self):
        update = getattr(self.ap, "update_log_context", None)
        if callable(update):
            law = getattr(self.ap, "law", None)
            update(model=str(self.model_path), detector="TensorRT" if self.using_engine else "PyTorch",
                   cam_fps=self.cam_mode, frame_width=self.fw, frame_height=self.fh,
                   camera_pipeline=self.camera_pipeline, sky_gate=self.use_sky_mask,
                   follower=self.use_follower, sensitivity=self.mode_name,
                   cam_hfov_deg=None if law is None else math.degrees(law.h_fov),
                   cam_vfov_deg=None if law is None else math.degrees(law.v_fov),
                   processing_fps=round(self._ui_fps, 1), video_recording=self.record_dir is not None)


    def _enhance(self, crop):
        if not self.USE_CLAHE:
            return crop
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _accept_box(self, x1n, y1n, x2n, y2n, conf):
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
        x1r, y1r, x2r, y2r = region_native
        x1r = max(0, x1r); y1r = max(0, y1r)
        x2r = min(self.full_w, x2r); y2r = min(self.full_h, y2r)
        if x2r - x1r < 2 or y2r - y1r < 2:
            return None

        crop = frame_full[y1r:y2r, x1r:x2r]
        if enhance:
            crop = self._enhance(crop)

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


    _STAB = dict(slew_per_s=3.0, rate_tau=0.22, deadband_frac=0.050,
                 kp_yaw=2.0, yaw_cap_dps=40.0)
    MODES = {
        "SLOW":   dict(climb_limit_mps=8.0,  max_lean_deg=12.0, lean_near=1.0, sens=0.35,
                       climb_trim=5.0, **_STAB),
        "MEDIUM": dict(climb_limit_mps=15.0, max_lean_deg=20.0, lean_near=1.0, sens=0.50,
                       climb_trim=6.0, **_STAB),
        "FAST":   dict(climb_limit_mps=25.0, max_lean_deg=22.0, lean_near=1.0, sens=0.65,
                       climb_trim=8.0, **_STAB),
        "MAX":    dict(climb_limit_mps=25.0, max_lean_deg=35.0, lean_near=0.3,
                       sens=0.65, climb_trim=8.0, **_STAB),
    }
    _SPEED_KEYS = ("climb_limit_mps", "max_lean_deg", "lean_near")
    _STAB_KEYS  = ("slew_per_s", "rate_tau", "deadband_frac")

    def set_mode(self, name):
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
        if law is not None and "kp_yaw" in prof and hasattr(law, "kp_yaw"):
            law.kp_yaw = float(prof["kp_yaw"])
            full = math.degrees(getattr(law, "max_yaw_rate", math.radians(202.5)))
            law.stick_limit_yaw = min(0.90, float(prof["yaw_cap_dps"]) / max(1.0, full))
        self.mode_name = key
        return key

    def set_sky_gate(self, value):
        v = str(value).strip().lower()
        if v not in ("on", "off", "1", "0", "true", "false"):
            return "bad SKY GATE value"
        on = v in ("on", "1", "true")
        if on != self.use_sky_mask:
            self.use_sky_mask = on
            print(f"[SKY] sky gate {'ON' if on else 'OFF: detecting on the whole frame'}")
        return ""

    def set_follower(self, value):
        v = str(value).strip().lower()
        if v not in ("on", "off", "1", "0", "true", "false"):
            return "bad FOLLOWER value"
        on = v in ("on", "1", "true")
        if on != self.use_follower:
            self.use_follower = on
            print(f"[FOLLOW] follower {'ON' if on else 'OFF: detector only'}")
        return ""

    def set_camera(self, value):
        try:
            want = int(str(value).strip())
        except ValueError:
            return "bad CAMERA value"
        if want not in CAM_MODES:
            return "CAMERA: 30 or 60"
        if self.camera_config is None:
            return "this run cannot switch the camera (no camera config)"
        if want == (self._cam_req or self.cam_mode):
            return ""
        why = self._camera_locked()
        if why:
            return why
        self._cam_req = want
        print(f"[CAM] {want} fps asked for")
        return ""

    def _camera_locked(self):
        if self._cam_busy:
            return "the camera is switching"
        m = getattr(self.ap, "mission", None)
        if m is not None and m.phase in ("PUSH", "AI", "STOP", "WASP_READY", "WASP_FLIGHT", "IDLE") + mission_mod.AUTO_PHASES:
            return f"not during {m.phase}: the picture is dark a moment while it switches"
        if m is None and self._last_sent:
            return "not while the AI steers: the picture is dark a moment while it switches"
        return ""

    def state_snapshot(self):
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
            detector=("TensorRT" if str(getattr(self, "model_path", "")).endswith(".engine")
                      else "PyTorch"),
            lat_ms=(None if not self._last_cmd or not self._last_cmd.get("latency_s")
                    else round(1000 * float(self._last_cmd["latency_s"]))),
            fc_mode=(getattr(ap, "mode_name", None) if ap else None),
            armed=(bool(getattr(ap, "armed", False)) if ap else False),
            linked=(bool(getattr(ap, "linked", False)) if ap else False),
            steering=bool(self._last_sent),
            sky_gate=bool(self.use_sky_mask),
            follower=bool(self.use_follower),
            follower_default=bool(getattr(self, "follower_default", self.use_follower)),
            cam_fps=self.cam_mode, cam_default=getattr(self, "cam_default", 30),
            cam_busy=bool(getattr(self, "_cam_busy", False)
                          or getattr(self, "_cam_req", None) is not None),
            cam_aspect=round(self.cam_aspect, 4),
        )
        if law is not None and hasattr(law, "h_fov"):
            out["cam_hfov"] = round(math.degrees(law.h_fov), 2)
            out["cam_vfov"] = round(math.degrees(law.v_fov), 2)
        if hasattr(ap, "mission_state"):
            try:
                out["mission"] = ap.mission_state()
            except Exception as exc:
                out["mission"] = dict(error=str(exc))
        out["video_recording"] = getattr(self, "record_dir", None) is not None
        if hasattr(ap, "logging_state"):
            out["logs"] = ap.logging_state()
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
        m = getattr(ap, "mission", None)
        if m is not None:
            out["tilt_deg"] = m.camera_mount_deg
        return out

    def _gate_detection(self, det, kf_pred, dt):
        if not self.GATE_ENABLE or kf_pred is None or det is None:
            return True
        jump = math.hypot(det[0] - kf_pred[0], det[1] - kf_pred[1])
        allow = self._gate_allow(dt)
        m = getattr(self.ap, "mission", None)
        if m is not None and m.wasp.enabled:
            allow += max(self._rotation_margin())
        if jump <= allow:
            self._gate_rejects = 0
            return True
        self._gate_rejects += 1
        if self._gate_rejects > self._frames(self.GATE_MAX_REJECT):
            self._gate_rejects = 0
            self.gate_note = "REACQ"
            return True
        self.gate_note = f"GATE {jump:.0f}px"
        self.gated_total += 1
        return False

    def _frames(self, n):
        if self.cam_mode <= self.REF_FPS:
            return n
        return max(n, int(round(n * self._ui_fps / self.REF_FPS)))

    def _gate_allow(self, dt):
        dt = dt if (dt and dt > 1e-3) else (1.0 / 30.0)
        size = max(self.last_bbox[2], self.last_bbox[3]) if self.last_bbox else 0
        return self.GATE_PX_PER_S * dt + self.GATE_FLOOR_PX + 0.5 * size

    VERIFY_MIN    = 0.50
    VERIFY_BETTER = 0.10
    YIELD_MISSES  = 2

    def _follow(self, frame_disp, kf_pred, now):
        vx, vy = self.kalman.velocity()
        mx, my = self._rotation_margin()
        return self.follower.track(frame_disp, kf_pred[0], kf_pred[1],
                                   search_radius(self.missed, math.hypot(vx, vy), max(mx, my)),
                                   now)

    def _adjudicate(self, frame_disp, det, kf_pred, dt, now):
        size = max(self.last_bbox[2], self.last_bbox[3]) if self.last_bbox else 10
        agree = max(8.0, float(size))
        jump = math.hypot(det[0] - kf_pred[0], det[1] - kf_pred[1])
        if jump <= agree:
            self._gate_rejects = 0
            return det, None
        fol = self._follow(frame_disp, kf_pred, now)
        if fol is not None and math.hypot(det[0] - fol[0], det[1] - fol[1]) <= agree:
            self._gate_rejects = 0
            return det, None
        allow = self._gate_allow(dt) + max(self._rotation_margin())
        if jump > allow:
            self.gate_note = f"DISTRACT {jump:.0f}px"
            self.gated_total += 1
            return None, fol
        like = self.follower.verify(frame_disp, det[0], det[1])
        if fol is not None:
            if (like >= self.VERIFY_MIN
                    and like >= (self.follower.a_score or 0.0) + self.VERIFY_BETTER):
                self.gate_note = "DET>FOLLOW"
                return det, None
            self.gate_note = f"DISTRACT {jump:.0f}px"
            self.gated_total += 1
            return None, fol
        if like >= self.VERIFY_MIN or self.follower.misses >= self._frames(self.YIELD_MISSES):
            self._gate_rejects = 0
            return det, None
        self.gate_note = f"DISTRACT {jump:.0f}px"
        self.gated_total += 1
        return None, None

    def _target_is_small(self):
        if self.last_bbox is None:
            return True
        bw, bh = self.last_bbox[2], self.last_bbox[3]
        ref_h = self.ref_display[1] if self.ref_display else self.fh
        return max(bw, bh) < ref_h * self.BIG_OBJ_FRAC

    def _sky_bbox_native(self):
        if not self.use_sky_mask:
            return (0, 0, self.full_w, self.full_h)
        if self.sky_mask is None:
            return None
        x, y, w, h = cv2.boundingRect(self.sky_mask)
        if w == 0 or h == 0:
            return None
        sx = self.full_w / self.fw
        sy = self.full_h / self.fh
        x1 = int(x * sx) - self.CROP_PAD
        x2 = int((x + w - 1) * sx) + self.CROP_PAD
        y1 = int(y * sy) - self.CROP_PAD
        y2 = int((y + h - 1) * sy) + self.CROP_PAD
        return (x1, y1, x2, y2)


    def _predict_roi(self):
        px, py = self.kalman.predict()
        vx, vy = self.kalman.velocity()
        bw, bh = (self.last_bbox[2], self.last_bbox[3]) if self.last_bbox else (80, 80)

        base = max(self.MIN_ROI_HALF, int(max(bw, bh) * 0.6))
        mx, my = self._rotation_margin()
        hw   = min(base + int(abs(vx) * self.missed * 1.5) + mx, self.fw // 2)
        hh   = min(base + int(abs(vy) * self.missed * 1.5) + my, self.fh // 2)

        return (
            max(0, px - hw),        max(0, py - hh),
            min(self.fw, px + hw),  min(self.fh, py + hh),
        )

    def _rotation_margin(self):
        ap = getattr(self, "ap", None)
        b = getattr(ap, "backend", None)
        if b is None or not hasattr(b, "attitude"):
            return 0, 0
        try:
            att = b.attitude()
        except Exception:
            return 0, 0
        if not att:
            return 0, 0
        dt = self._frame_dt or 0.1
        fx = (self.fw / 2.0) / math.tan(getattr(ap, "H_FOV", math.radians(62.2)) / 2.0)
        fy = (self.fh / 2.0) / math.tan(getattr(ap, "V_FOV", math.radians(48.8)) / 2.0)
        m = getattr(ap, "mission", None)
        if m is not None and m.wasp.enabled:
            # With an upward camera roll/pitch move the target; yaw spins the
            # image. Bound all three axes, including arbitrary camera roll.
            rate = sum(abs(att.get(k) or 0.0) for k in ("rollspeed", "pitchspeed", "yawspeed"))
            angle = min(1.0, rate * min(dt, 0.25))
            return int(angle * fx * 2), int(angle * fy * 2)
        mx = abs(att.get("yawspeed") or 0.0) * dt * fx * 1.3
        my = abs(att.get("pitchspeed") or 0.0) * dt * fy * 1.3
        return int(mx), int(my)

    def _roi_disp_to_native(self, roi):
        sx = self.full_w / self.fw
        sy = self.full_h / self.fh
        return (int(roi[0] * sx), int(roi[1] * sy),
                int(roi[2] * sx), int(roi[3] * sy))


    def _process(self, frame_full, frame_disp):
        self.sky_mask = self.sky.mask(frame_disp) if self.use_sky_mask else None
        self._ground_rejected = False
        _m = getattr(self.ap, "mission", None)
        if _m is not None:
            _want = _m.ai_level_now()
            if _want != self.mode_name and _want in self.MODES:
                self.set_mode(_want)

        t0      = perf_counter()
        kf_pred = self.kalman.predict() if self.kalman.initialized else None
        t_kf    = (perf_counter() - t0) * 1000

        t0       = perf_counter()
        roi      = None
        yolo_det = None

        small = self._target_is_small()  

        _alt = (self.missed >= 1 and os.environ.get("ALT_SEARCH", "1") == "1")
        _sky_turn = _alt and (self._frame_idx % 2 == 1)
        if self.state in ("LOCK", "PREDICT") and kf_pred and _sky_turn:
            sky_bbox = self._sky_bbox_native()
            if sky_bbox is not None:
                srch_imgsz = self.IMGSZ_SEARCH if small else self.IMGSZ_TRACK
                yolo_det = self._detect_region(frame_full, sky_bbox,
                                               srch_imgsz, enhance=(self.USE_CLAHE and small))
        elif self.state in ("LOCK", "PREDICT") and kf_pred:
            roi         = self._predict_roi()
            track_imgsz = self.IMGSZ_TRACK if small else self.IMGSZ_CLOSE
            yolo_det = self._detect_region(frame_full, self._roi_disp_to_native(roi),
                                           track_imgsz, enhance=(self.USE_CLAHE and small))
            _hold = (self.use_follower and self.follower.armed and self.missed == 0)
            if yolo_det is None and not _alt and not _hold:
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
        if _dt is not None and 0.0 < _dt < 1.0:
            self._frame_dt = _dt
        fol_hit = None
        if not self.use_follower and self.follower.active:
            self.follower.reset()
        armed = (self.use_follower and self.follower.armed and kf_pred is not None
                 and self.state != "SEARCH")
        if armed and yolo_det is not None:
            yolo_det, fol_hit = self._adjudicate(frame_disp, yolo_det, kf_pred, _dt, _now)
        else:
            if yolo_det is not None and not self._gate_detection(yolo_det, kf_pred, _dt):
                yolo_det = None
            if armed and yolo_det is None:
                fol_hit = self._follow(frame_disp, kf_pred, _now)

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
            if self.use_follower:
                self.follower.seed(frame_disp, cx, cy, w, h, _now)
        elif fol_hit:
            fx, fy, fw_, fh_, score = fol_hit
            cx, cy = int(round(fx)), int(round(fy))
            self.kalman.update(cx, cy)
            self.last_bbox = (int(round(fx - fw_ / 2)), int(round(fy - fh_ / 2)),
                              max(1, int(round(fw_))), max(1, int(round(fh_))))
            self.missed    = 0
            self.state     = "LOCK"
            fused          = (cx, cy)
            self.follow_hits += 1
            self.layer     = f"FOLLOW {score:.2f}"
        elif kf_pred and self.state != "SEARCH":
            self.missed += 1
            if self.missed >= self._frames(self.MISS_TO_SEARCH):
                self._reset()
                fused      = None
                self.layer = "LOST"
            else:
                self.state = ("PREDICT" if self.missed >= self._frames(self.MISS_TO_PREDICT)
                              else self.state)
                fused      = kf_pred
                self.layer = f"KF  miss={self.missed}"
        else:
            self.missed += 1
            fused      = None
            self.layer = "GROUND-REJ" if self._ground_rejected else "LOST"

        self.pos = fused
        return fused, t_kf, t_yolo, roi, (yolo_det is not None or fol_hit is not None)

    def _reset(self):
        self.state     = "SEARCH"
        self.missed    = 0
        self.last_bbox = None
        self.kalman    = KalmanTracker()
        self.layer     = "STANDBY"
        self.conf      = None
        self._gate_rejects = 0
        self.gate_note = ""
        self.follower.reset()


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


    def mode_color(self):
        m = getattr(getattr(self, "ap", None), "mode_name", None)
        if m == "ALT_HOLD":
            return self.C_ALT
        if m == "ACRO":
            return self.C_ACRO
        return self.C_PILOT

    def _draw_header(self, frame, accent):
        w = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (w, 4), self.mode_color(), -1)


    def _hbar(self, frame, x, y, w, h, frac, color, bidirectional=True):
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
        if not self.HUD_PANEL:
            return

        h, _ = frame.shape[:2]
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

        lim = max(1e-3, self.ap.law.stick_limit)
        bar_x, bar_w, bar_h = tx + 42, 110, 8
        if intercept:
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
            sens = cmd.get("sens")
            if sens is not None:
                s_txt = (f"SENS {sens:.2f}  gain {cmd.get('pn_gain_eff', 0.0):.0f}"
                         f"  cap {cmd.get('bank_cap_deg', 0.0):.1f}°"
                         f"  fill {cmd.get('fill_frac', 0.0) * 100:.1f}%")
                s_col = self.C_WARN if sens > 0.5 else self.C_HUD
                self._text(frame, s_txt, (tx, y1 + 162), 0.34, s_col, 1)

        self._text(frame, f"{self.ap.send_hz:.0f}Hz stream  (skip {self.skip_frames})",
                   (tx, y1 + ph - 8), 0.34, self.C_DIM, 1)



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
                    size = (frame.shape[1], frame.shape[0])
                    if size != getattr(self, "_rec_size", size):
                        self._rec_rotate(frame)
                    self._writer.write(frame)
                    self._rec_seg_n = getattr(self, "_rec_seg_n", 0) + 1
                    if self._rec_seg_n >= self._rec_seg_frames():
                        self._rec_rotate(frame)
            except Exception:
                pass

    def _rec_seg_frames(self):
        fps = getattr(self, "_rec_fps", 0.0) or 30.0
        return max(30, int(float(os.environ.get("REC_SEGMENT_S", "20")) * fps))

    def _rec_rotate(self, frame):
        base = getattr(self, "_rec_base", None)
        fourcc = getattr(self, "_rec_fourcc", None)
        if base is None or fourcc is None:
            return
        self._rec_part = getattr(self, "_rec_part", 1) + 1
        nxt = base.with_name(f"{base.stem}_p{self._rec_part:02d}{base.suffix}")
        w = cv2.VideoWriter(str(nxt), fourcc, getattr(self, "_rec_fps", 30.0),
                            (frame.shape[1], frame.shape[0]))
        if not w.isOpened():
            return
        old, self._writer = self._writer, w
        self._rec_size = (frame.shape[1], frame.shape[0])
        old.release()
        self._rec_seg_n = 0

    def _rec_submit(self, frame):
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
        if not self.PROFILE:
            return t_prev
        now = perf_counter()
        self._prof[name] = self._prof.get(name, 0.0) + (now - t_prev)
        return now

    def _prof_report(self, every=60):
        if not self.PROFILE:
            return
        self._prof_n += 1
        if self._prof_n < every:
            return
        n = self._prof_n
        self.prof_ms = {k: round(v / n * 1000.0, 2)
                        for k, v in sorted(self._prof.items(),
                                           key=lambda kv: -kv[1])}
        total = sum(v for k, v in self.prof_ms.items()
                    if not k.startswith("  "))
        self.prof_ms["TOTAL"] = round(total, 2)
        print("[PROF] " + "  ".join(f"{k} {v:.1f}" for k, v in
                                    self.prof_ms.items())
              + f"  -> {1000.0 / total if total else 0:.1f} fps")
        self._prof.clear()
        self._prof_n = 0

    def _command(self, key, value):
        if key in ("mode", "sens"):
            self.set_mode(value)
            m = getattr(self.ap, "mission", None)
            if m is not None:
                m.set_level("ai", self.mode_name)
        elif key == "sky":
            return self.set_sky_gate(value)
        elif key == "follow":
            return self.set_follower(value)
        elif key == "cam":
            return self.set_camera(value)
        elif key == "def" and str(value).startswith("det:"):
            return self._save_detection_default(str(value)[4:].strip().lower())
        elif key in ("hold", "release", "step", "engage", "abort", "lvl", "lock", "auto", "autoen", "waspen", "waspp", "mount", "flightmode",
                     "autop", "autoreset", "def", "idles"):
            return self._mission_command(key, str(value))
        else:
            raise ValueError("unknown command")
        return ""

    def _mission_command(self, key, value):
        m = getattr(self.ap, "mission", None)
        if m is None:
            return "mission app not running (WASP_APP=mission, FC linked?)"
        if ((key == "engage" or (key == "auto" and str(value).strip().lower() in ("1", "on", "true")))
                and (self._cam_busy or self._cam_req is not None)):
            return "the camera is switching: ENGAGE / AUTO in a moment"
        if key == "lvl":
            group, _, level = str(value).partition(":")
            if group.strip().lower() == "ai":
                if level.strip().upper() not in self.MODES:
                    return "unknown speed"
                self.set_mode(level.strip().upper())
        if key == "def":
            r = mission_mod.save_default(m, value)
            if r == "":
                print(f"[PARAMS] default saved: {value}")
            return r
        return mission_mod.phone_command(m, key, value, perf_counter())

    def _save_detection_default(self, key):
        if key == "cam":
            want = self._cam_req or self.cam_mode
            r = mission_mod.save_detection_default("cam", want)
            if r == "":
                self.cam_default = want
                print(f"[PARAMS] default saved: camera {want} fps")
            return r
        if key != "follow":
            return mission_mod.save_detection_default(key, True)
        r = mission_mod.save_detection_default("follow", self.use_follower)
        if r == "":
            self.follower_default = self.use_follower
            print(f"[PARAMS] default saved: follower {'ON' if self.use_follower else 'OFF'}")
        return r


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

            bw, bh = (self.last_bbox[2], self.last_bbox[3]) if self.last_bbox else (20, 20)
            rad = int(max(14, max(bw, bh) * 0.62 + 6))
            mc = self.mode_color()
            if self.state == "PREDICT":
                for a0 in range(0, 360, 40):
                    cv2.ellipse(frame, (px, py), (rad, rad), 0, a0, a0 + 22,
                                mc, 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), rad, mc, 2, cv2.LINE_AA)
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

        
        

    def _open_capture(self):
        if self.camera_pipeline is None:
            raise RuntimeError("camera_pipeline is required (no capture source set).")

        print(f"[CAP] Opening GStreamer pipeline:\n      {self.camera_pipeline}")
        cap  = None
        warm = None
        for attempt in range(self.CAM_OPEN_RETRIES):
            cap = cv2.VideoCapture(self.camera_pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                for _ in range(self.CAM_WARMUP_READS):
                    ok, warm = cap.read()
                    if ok and warm is not None:
                        break
            if warm is not None:
                if attempt:
                    print(f"[CAP] camera came up on attempt {attempt + 1}")
                break
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

        reported = cap.get(cv2.CAP_PROP_FPS)
        src_fps  = reported if (reported and reported == reported and reported > 1) \
                   else self.camera_fps
        label = f"camera (pipeline, ~{src_fps:.1f} fps)"

        return cap, src_fps, label

    def _apply_cam_mode(self, mode):
        spec = CAM_MODES[mode]
        cx, cy = spec["crop"]
        self.cam_mode = mode
        if self.camera_config is not None:
            cfg = self.camera_config
            if "sensor_mode" in spec:
                cfg = _dc_replace(cfg, sensor_mode=spec["sensor_mode"], width=spec["width"],
                                  height=spec["height"], fps=spec["fps"])
            self.camera_pipeline = make_pipeline(cfg)
            self.camera_fps = float(cfg.fps)
            self.cam_aspect = cfg.width / float(cfg.height)
        rw, rh = self.ref_display
        self.display_size = ((rw, rh) if (cx, cy) == (1.0, 1.0) else
                             (2 * int(round(rw * cx / 2.0)), 2 * int(round(rh * cy / 2.0))))
        self.fw, self.fh = self.display_size
        law = getattr(getattr(self, "ap", None), "law", None)
        if hasattr(law, "set_crop"):
            law.set_crop(cx, cy)

    def _switch_camera(self, pre_buf):
        want = self._cam_req
        if want is None or want == self.cam_mode:
            self._cam_req = None
            return
        why = self._camera_locked()
        if why:
            self._cam_req = None
            print(f"[CAM] {want} fps not switched: {why}")
            return
        old = self.cam_mode
        self._cam_busy = True
        self._cam_req = None
        try:
            print(f"[CAM] {old} -> {want} fps: reopening the camera")
            self._apply_cam_mode(want)
            try:
                self._cap.release()
            except Exception:
                pass
            try:
                self._cap, _fps, _label = self._open_capture()
            except RuntimeError as exc:
                print(f"[CAM] {want} fps did not come up "
                      f"({str(exc).splitlines()[0]}) — back to {old} fps")
                self._apply_cam_mode(old)
                self._cap, _fps, _label = self._open_capture()
            self._reset()
            self.sky_mask = None
            self._t_last_frame = self._t_ui_prev = None
            pre_buf.clear()
            law = getattr(self.ap, "law", None)
            fov = (f"{math.degrees(law.h_fov):.1f} x {math.degrees(law.v_fov):.1f} deg, "
                   if law is not None else "")
            print(f"[CAM] camera at {self.cam_mode} fps: {fov}working frame "
                  f"{self.fw}x{self.fh}")
        finally:
            self._cam_busy = False


    def run(self):
        try:
            self._run()
        finally:
            self.ap.close()
            if self.stream is not None:
                self.stream.stop()
                self.stream = None

    def _run(self):
        self.ap.connect()

        try:
            cap, src_fps, source_label = self._open_capture()
        except RuntimeError as exc:
            if self.cam_mode == 30 or self.camera_config is None:
                raise
            print(f"[CAM] {self.cam_mode} fps did not come up "
                  f"({str(exc).splitlines()[0]}) — starting at 30 fps")
            self._apply_cam_mode(30)
            cap, src_fps, source_label = self._open_capture()
        self._cap = cap
        print(f"[RUN] Source: {source_label}")
        src_fps = min(src_fps, self.REF_FPS)
            
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
            out_path = out_dir / f"{self.run_stamp}.mp4"
            print(f"[REC] reel -> {out_path.resolve()}  | fps={src_fps:.2f}  "
                  f"pad={pad}f (~{self.record_pad_sec:.1f}s each side)")            

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

        try:
            self._loop(cap, src_fps, period, rec_on, pad, pre_buf, out_path,
                       fourcc)
        finally:
            self._finish_recording()
            self._cap.release()
            if not self.headless:
                cv2.destroyAllWindows()

    def _finish_recording(self):
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
        self._cap         = cap

        while self._cap.isOpened():
            if self._cam_req is not None:
                self._switch_camera(pre_buf)
            ret, frame_full = self._cap.read()
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

            target_off = None
            _mission_on = getattr(self.ap, "mission", None) is not None
            if is_yolo_frame:
                if fused and _mission_on and not detected:
                    target_off = (fused[0] - cx_f, fused[1] - cy_f)
                    cmd = self._last_cmd
                else:
                    if fused:
                        target_off = (fused[0] - cx_f, fused[1] - cy_f)
                        cmd = self.ap.compute(target_off[0], target_off[1],
                                              self.state, self.fw, self.fh,
                                              bbox=self.last_bbox, capture_t=t_cap)
                    else:
                        cmd = self.ap.neutral_command()
                    self.ap.update(cmd, target_off, have_target=bool(fused),
                                   detected=bool(detected),
                                   conf=self.conf if detected else None,
                                   bbox=self.last_bbox if detected else None)
                    self._last_cmd = cmd
            else:
                cmd = self._last_cmd
                if fused:
                    target_off = (fused[0] - cx_f, fused[1] - cy_f)

            actually_sent = self.ap.steering
            self._last_sent = bool(actually_sent)
            self._update_log_context()

            _tp = self._lap("control", _tp)
            self._draw(frame_disp, frame_full, fused, roi, cmd, actually_sent)
            _tp = self._lap("draw", _tp)

            if self.stream is not None:
                self.stream.publish(frame_disp)
            _tp = self._lap("stream", _tp)

            if rec_on:
               if fused:
                   if post_left == 0:
                       self._rec_windows += 1
                       if self._writer is None:
                           self._rec_base, self._rec_fourcc = Path(out_path), fourcc
                           self._rec_part, self._rec_seg_n = 1, 0
                           self._writer = cv2.VideoWriter(
                               str(out_path), fourcc, src_fps,
                               (frame_disp.shape[1], frame_disp.shape[0]))
                           self._rec_size = (frame_disp.shape[1], frame_disp.shape[0])
                           if not self._writer.isOpened():
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
                   self._rec_submit(frame_disp.copy())
                   last_written = self._frame_idx; self._rec_frames += 1
                   post_left -= 1
               elif rec_on:
                   pre_buf.append((self._frame_idx, frame_disp.copy()))

            _tp = self._lap("record", _tp)
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

        self.ap.close()


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class CameraConfig:
    sensor_id: int = int(os.getenv("CAM_SENSOR_ID", "0"))
    sensor_mode: int = int(os.getenv("CAM_SENSOR_MODE", "3"))

    width: int = int(os.getenv("CAM_WIDTH", "1280"))
    height: int = int(os.getenv("CAM_HEIGHT", "960"))
    fps: int = int(os.getenv("CAM_FPS", "30"))

    flip_method: int = int(os.getenv("CAM_FLIP_METHOD", "0"))

    ae_lock: bool = env_bool("CAM_AE_LOCK", False)
    awb_lock: bool = env_bool("CAM_AWB_LOCK", False)

    exposure_us: int = int(os.getenv("CAM_EXPOSURE_US", "0"))
    gain: float = float(os.getenv("CAM_GAIN", "0"))
    digital_gain: float = float(os.getenv("CAM_DIGITAL_GAIN", "0"))

    tnr_mode: int = int(os.getenv("CAM_TNR_MODE", "1"))
    tnr_strength: float = float(os.getenv("CAM_TNR_STRENGTH", "0.5"))

    ee_mode: int = int(os.getenv("CAM_EE_MODE", "1"))
    ee_strength: float = float(os.getenv("CAM_EE_STRENGTH", "0.5"))

    drop: bool = env_bool("CAM_DROP", True)
    max_buffers: int = int(os.getenv("CAM_MAX_BUFFERS", "1"))
    sync: bool = env_bool("CAM_SYNC", False)


CAM_MODES = {
    30: dict(crop=(1.0, 1.0)),
    60: dict(crop=(2560 / 3280, 1440 / 2464),
             sensor_mode=4, width=1280, height=720, fps=60),
}


def make_pipeline(cfg: CameraConfig) -> str:
    props = [
        f"sensor-id={cfg.sensor_id}",
        f"tnr-mode={cfg.tnr_mode}",
        f"tnr-strength={cfg.tnr_strength}",
        f"ee-mode={cfg.ee_mode}",
        f"ee-strength={cfg.ee_strength}",
    ]

    if cfg.sensor_mode >= 0:
        props.insert(1, f"sensor-mode={cfg.sensor_mode}")

    if cfg.ae_lock:
        props.append("aelock=true")

    if cfg.awb_lock:
        props.append("awblock=true")

    if cfg.exposure_us > 0:
        props.append(
            f'exposuretimerange="{cfg.exposure_us} {cfg.exposure_us}"'
        )

    if cfg.gain > 0:
        props.append(f'gainrange="{cfg.gain} {cfg.gain}"')

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



_DATA_DIR = Path(__file__).resolve().parent / "data"


CONSOLE = collections.deque(maxlen=300)
_CONSOLE_SEQ = [0]
_CONSOLE_LOCK = threading.Lock()
_CONSOLE_SKIP = ("[CTRL]", "[PROF]", "GST_", "CONSUMER", "[ WARN", "   ",
                 "Camera index", "Camera mode", "Output Stream", "seconds to Run",
                 "Frame Rate")


def console_since(seq):
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
    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh
        self._need_prefix = True
        self._partial = ""

    @staticmethod
    def _prefix():
        now = datetime.now()
        return f"[{now.strftime('%H:%M:%S')}.{now.microsecond // 1000:03d}] "

    def write(self, data):
        self._stream.write(data)
        out = []
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


def _run_stamp(data_dir):
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    if now.year < 2024:
        ctr = data_dir / ".run_counter"
        try:
            n = int(ctr.read_text().strip() or 0) + 1
        except (OSError, ValueError):
            n = 1
        try:
            ctr.write_text(f"{n}\n")
        except OSError:
            pass
        stamp = f"nosync_r{n:04d}_{now.strftime('%H%M%S')}"
    base, k = stamp, 2
    while any(data_dir.glob(f"{stamp}.*")) or any(data_dir.glob(f"control_{stamp}.*")):
        stamp, k = f"{base}_{k}", k + 1
    return stamp


if __name__ == "__main__":
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUN_STAMP = _run_stamp(_DATA_DIR)
    os.environ["WASP_RUN_STAMP"] = RUN_STAMP
    _log_fh = open(_DATA_DIR / f"{RUN_STAMP}.log", "w", buffering=1)
    sys.stdout = _Tee(sys.stdout, _log_fh)
    sys.stderr = _Tee(sys.stderr, _log_fh)
    print(f"[LOG] console mirrored to {(_DATA_DIR / f'{RUN_STAMP}.log')}")

    cam_cfg = CameraConfig()
    CAP_PIPELINE = make_pipeline(cam_cfg)
    _model_env = os.environ.get("WASP_MODEL", "").strip()
    model_pth = str(Path(_model_env).expanduser().resolve() if _model_env else
                    Path(__file__).resolve().parent.parent
                    / "runs/detect/v0/weights/best.pt")
    if not Path(model_pth).is_file():
        print(f"[MODEL] no weights file at {model_pth}", file=sys.stderr)
        sys.exit(1)

    if os.environ.get("USE_ENGINE", "1") == "1":
        _eng = Path(model_pth).with_suffix(".engine")
        if _eng.is_file() and _eng.stat().st_mtime >= Path(model_pth).stat().st_mtime:
            try:
                from ultralytics import YOLO
                _probe = YOLO(str(_eng), task="detect")
                _probe.predict(np.zeros((640, 640, 3), np.uint8), verbose=False)
                if len(_probe.names) > 100:
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

    os.environ.setdefault("TARGET_KIND", "DRONE")
    print(f"[TARGET] {os.environ['TARGET_KIND']} (fixed for this run)")
    print("[REC] disabled — logs only")

    DroneTracker(
        model_path   = model_pth,
        camera_pipeline=CAP_PIPELINE,
        camera_fps=cam_cfg.fps,
        camera_config=cam_cfg,
        conf_thresh  = 0.01,
        display_size = (640, 640),
        skip_frames  = 0,
        show_sky     = False,
        use_sky_mask = True,
        record_dir      = None,
        log_dir         = str(_DATA_DIR),
        run_stamp       = RUN_STAMP,
        headless     = os.environ.get("TRACKER_HEADLESS") == "1",
        stream_port    = (int(os.environ.get("STREAM_PORT", "8080"))
                          if env_bool("STREAM", True) else None),
        stream_quality = int(os.environ.get("STREAM_QUALITY", "70")),
        stream_fps     = float(os.environ.get("STREAM_FPS", "15")),
        commander    = commander_from_env(log_dir=str(_DATA_DIR)),
    ).run()
