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
Steering goes to an ArduPilot Copter FC (TBS Lucid H7 / STM32H743) over
MAVLink as GUIDED-mode velocity + yaw-rate setpoints
(SET_POSITION_TARGET_LOCAL_NED, MAV_FRAME_BODY_OFFSET_NED).  The control law
centres the detection in the frame: yaw rate nulls the horizontal pixel error,
climb/descend (NED vz) nulls the vertical error, and vx closes on the target
while locked.  One setpoint is sent per YOLO frame, so the command rate always
equals the detection rate set by skip_frames (camera_fps / (skip_frames + 1)).
"""

import os
os.environ.pop("SESSION_MANAGER", None) 

import cv2
import numpy as np
from collections import deque
from pathlib import Path
from time import perf_counter
from ultralytics import YOLO
from dataclasses import dataclass
from huggingface_hub import hf_hub_download

try:
    from pymavlink import mavutil
except ImportError:          # offline development box without pymavlink
    mavutil = None


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
# ARDUPILOT COMMANDER  (TBS Lucid H7 / STM32H743, ArduPilot Copter, MAVLink2)
# ══════════════════════════════════════════════════════════════════════════════

class ArduPilotCommander:
    """Centering control law + MAVLink link to an ArduPilot Copter FC.

    The detection's pixel offset from the frame centre is converted to an
    angular error through the camera FOV.  Yaw rate nulls the horizontal
    error and NED vertical velocity nulls the vertical error, so the airframe
    continuously steers to keep the target at the centre of the frame while
    vx (body-forward) closes on it.

    Transport: SET_POSITION_TARGET_LOCAL_NED in MAV_FRAME_BODY_OFFSET_NED —
    velocities are interpreted relative to the vehicle's current heading,
    which is what ArduPilot Copter GUIDED mode expects for body-frame
    velocity + yaw-rate control.  The FC must be in GUIDED mode to honour
    these setpoints, and ArduPilot stops trusting a stale velocity target
    after GUID_TIMEOUT (default 3 s) — so the send rate, which is locked to
    the detection rate (camera_fps / (skip_frames + 1)), must stay well above
    ~0.3 Hz.  At 30 fps even skip_frames = 10 gives ~2.7 Hz, still safe.

    Bench link: Jetson USB -> FC, enumerates as /dev/ttyACM0.
    Airframe link: spare UART with SERIALn_PROTOCOL = 2 (MAVLink2).
    """

    # Set these to the REAL lens FOV of the deployed camera before flight.
    H_FOV = np.radians(90.0)
    V_FOV = np.radians(90.0)

    K_YAW = 1.5
    K_VZ  = 2.0

    FWD = {"LOCK": 2.0, "PREDICT": 0.5, "SEARCH": 0.0}

    MAX_YAW  = 1.5
    MAX_VZ   = 2.0
    MAX_FWD  = 3.0

    GUIDED_MODE = 4                  # ArduPilot Copter custom_mode for GUIDED
    FRAME_BODY_OFFSET_NED = 9        # MAV_FRAME_BODY_OFFSET_NED
    # POSITION_TARGET_TYPEMASK: a SET bit means IGNORE that field.
    # Ignored: position x/y/z (1+2+4), accel (64+128+256), yaw angle (1024).
    # Active:  velocity vx/vy/vz (bits 3-5) and yaw_rate (bit 11).
    TYPE_MASK = 0b010111000111       # 1479

    def __init__(self, device=None, baud=115200):
        self.device = device         # e.g. "/dev/ttyACM0"; None = print-only
        self.baud   = baud
        self.master = None
        self._t0    = perf_counter()

    # ── link ──────────────────────────────────────────────────────────────────
    def connect(self):
        """Open the MAVLink link and verify the FC really runs ArduPilot."""
        if self.device is None:
            print("[MAV] no device configured — offline, commands are print-only")
            return
        if mavutil is None:
            raise RuntimeError("pymavlink not installed:  pip install pymavlink")

        print(f"[MAV] connecting {self.device} @ {self.baud} ...")
        self.master = mavutil.mavlink_connection(self.device, baud=self.baud)
        hb = self.master.wait_heartbeat(timeout=10)
        if hb is None:
            raise RuntimeError(f"[MAV] no heartbeat on {self.device} within 10 s")
        if hb.autopilot != mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
            raise RuntimeError(
                f"[MAV] FC is NOT running ArduPilot (autopilot id {hb.autopilot}). "
                "The Lucid H7 is likely flashed with Betaflight/INAV — MAVLink "
                "setpoints are the wrong transport there. Flash ArduPilot Copter "
                "or revisit the integration brief.")
        print(f"[MAV] ArduPilot heartbeat  sys={self.master.target_system} "
              f"comp={self.master.target_component}  mode={hb.custom_mode}")
        if hb.custom_mode != self.GUIDED_MODE:
            print("[MAV] WARNING: vehicle is not in GUIDED — velocity setpoints "
                  "are ignored until GUIDED is engaged (RC/GCS or set_guided()).")

    def set_guided(self):
        """Request GUIDED mode.  Deliberately NOT called automatically: engage
        from RC/GCS during bench + flight test, or call this explicitly."""
        if self.master is None:
            return
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            self.GUIDED_MODE, 0, 0, 0, 0, 0)

    def close(self):
        if self.master is not None:
            # final all-stop so the FC is not left chasing the last setpoint
            self.send(dict(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0))
            self.master.close()
            self.master = None

    # ── control law ───────────────────────────────────────────────────────────
    def compute(self, err_x, err_y, state, fw, fh):
        ang_x = err_x * (self.H_FOV / fw)
        ang_y = err_y * (self.V_FOV / fh)

        # err_x > 0  → target right of centre → yaw right (+ NED yaw rate)
        # err_y > 0  → target below centre    → descend   (+ NED vz is down)
        yaw_rate = float(np.clip( self.K_YAW * ang_x,  -self.MAX_YAW, self.MAX_YAW))
        vz       = float(np.clip( self.K_VZ  * ang_y,  -self.MAX_VZ,  self.MAX_VZ))
        vx       = float(np.clip( self.FWD.get(state, 0.0), 0.0, self.MAX_FWD))
        vy       = 0.0

        return dict(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)

    def search_command(self):
        """No target in view: don't idle — advance straight ahead at the
        controller's MAX forward speed, holding the current heading (yaw_rate 0)
        and altitude (vz 0), so the drone keeps closing/sweeping until the
        tracker re-acquires.  This is the SEARCH behaviour: fly forward in the
        direction the camera is currently looking, full speed."""
        return dict(vx=self.MAX_FWD, vy=0.0, vz=0.0, yaw_rate=0.0)

    def send(self, cmd):
        """One velocity + yaw-rate setpoint.  Called once per YOLO frame, so
        the MAVLink command rate always equals the detection rate.  Returns
        True if the message was written to the link, False if offline."""
        if self.master is None:
            return False
        self.master.mav.set_position_target_local_ned_send(
            int((perf_counter() - self._t0) * 1000),      # time_boot_ms
            self.master.target_system,
            self.master.target_component,
            self.FRAME_BODY_OFFSET_NED,
            self.TYPE_MASK,
            0.0, 0.0, 0.0,                                # position  (ignored)
            cmd["vx"], cmd["vy"], cmd["vz"],              # velocity  m/s, body
            0.0, 0.0, 0.0,                                # accel     (ignored)
            0.0, cmd["yaw_rate"])                         # yaw ign., yaw rate
        return True

    @staticmethod
    def format_command(state, cmd, target_off, sent, live, is_yolo):
        """One clean line describing the REAL steering command the controller
        is issuing, and where it went.

        target_off : (ex, ey) pixel offset of the target from frame centre when
                     a target is in view (→ centre on it); None while SEARCH
                     (→ advance straight ahead at max forward speed).
        sent       : True if this setpoint was transmitted to the flight
                     controller this frame.
        live       : True when a real MAVLink link is open (False = mp4/offline
                     simulation, nothing can be transmitted).
        is_yolo    : False on skip frames (Kalman coast, never transmitted).

        vy is omitted: horizontal centring is done with yaw, so lateral
        velocity is always 0 by design.
        """
        cmd_s = (f"fwd {cmd['vx']:+.2f}  vz {cmd['vz']:+.2f} m/s  "
                 f"yaw {cmd['yaw_rate']:+.3f} rad/s")
        if sent:
            dest = "TX->FC"
        elif not is_yolo:
            dest = "held (skip frame)"
        elif live:
            dest = f"not sent (state={state})"
        else:
            dest = "SIM (no FC attached)"

        if target_off is None:
            # SEARCH: no target → advance straight ahead at max forward speed.
            body = "no target — advancing straight ahead (max fwd, hold heading)"
        else:
            ex, ey = target_off
            body = f"target {ex:+05d},{ey:+05d}px -> centre"
        return f"[CTRL] {state:<7} {body} | {cmd_s} | {dest}"


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
    # CLAHE is OFF by default: this model was trained on un-equalised footage,
    # so histogram-equalising the crop shifts the pixel distribution and was
    # measured to SUPPRESS valid small-drone detections (e.g. a conf-0.37 hit
    # at imgsz=1280 vanished with CLAHE on).  The ZOOM (crop + high imgsz) is
    # what recovers small drones here.  Flip this to True only if you retrain /
    # fine-tune the model on contrast-enhanced data.
    USE_CLAHE    = False  # contrast-boost the crop before inference
    model_pth = hf_hub_download(
            repo_id="iitolstykh/YOLO-Face-Person-Detector",
            filename="yolov8x_person_face.pt",
            repo_type="model"
    )

    def __init__(
        self,
        #model_path      = "/runs/detect/v0/weights/best.pt",
        model_path = model_pth,
        video_path      = "/data/ClipExport.mp4",
        # ── NEW: live camera ──────────────────────────────────────────────────
        camera_pipeline = None,   # GStreamer pipeline string; if set, overrides video_path
        camera_fps      = 30.0,   # fallback FPS when cap.get(CAP_PROP_FPS) returns 0
        # ─────────────────────────────────────────────────────────────────────
        conf_thresh    = 0.05,
        display_size   = (640, 640),
        skip_frames    = 0,
        show_sky       = True,
        record_dir     = None,
        record_pad_sec = 2.0,
        headless       = False,
        max_frames     = None,
        mav_device     = None,      # e.g. "/dev/ttyACM0" (Jetson USB -> Lucid H7)
        mav_baud       = 115200,
    ):
        self.model        = YOLO(model_path)
        self.video_path   = video_path
        self.camera_pipeline = camera_pipeline
        self.camera_fps      = float(camera_fps)        
        self.conf_thresh  = conf_thresh
        self.display_size = display_size
        self.fw, self.fh  = display_size
        self.skip_frames  = max(0, int(skip_frames))
        self.show_sky     = show_sky
        self.record_dir      = record_dir
        self.record_pad_sec  = float(record_pad_sec)
        self.headless     = headless
        self.max_frames   = max_frames

        self.kalman  = KalmanTracker()
        self.ap      = ArduPilotCommander(mav_device, mav_baud)
        self.sky     = SkyDetector()
        self._clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self._last_cmd = dict(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)

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
        """Single flight panel: target deviation from centre + the ArduPilot
        command going out right now (yaw / climb / forward gauges).  Lamp:
        TX = setpoint sent this frame (YOLO frame), HOLD = skip frame (nothing
        sent, FC flies the last setpoint), SIM = no MAVLink device configured."""
        h, _ = frame.shape[:2]
        pw, ph = 246, 118
        x1 = 10
        y1 = h - 10 - ph
        x2, y2 = x1 + pw, y1 + ph
        self._panel(frame, (x1, y1), (x2, y2), self.C_INK, 0.5, border=self.C_DIM)

        tx = x1 + 12
        self._text(frame, "FLIGHT CMD", (tx, y1 + 18), 0.42, self.C_DIM, 1)

        if not sent:
            lamp, tag = self.C_DIM, "HOLD"
        elif self.ap.master is None:
            lamp, tag = self.C_SEARCH, "SIM"
        else:
            lamp, tag = self.C_LOCK, "TX"
        (tw, _), _ = cv2.getTextSize(tag, self._HUD_FONT, 0.42, 1)
        cv2.circle(frame, (x2 - tw - 24, y1 + 13), 4, lamp, -1, cv2.LINE_AA)
        self._text(frame, tag, (x2 - tw - 14, y1 + 18), 0.42, lamp, 1)

        if pos:
            ex = pos[0] - cx_f; ey = pos[1] - cy_f
            az = np.degrees(ex * (self.ap.H_FOV / self.fw))
            el = np.degrees(-ey * (self.ap.V_FOV / self.fh))
            dev = f"DEV {ex:+04d},{ey:+04d}px  AZ{az:+05.1f} EL{el:+05.1f}"
        else:
            dev = "DEV ----,----px  AZ----- EL-----"
        self._text(frame, dev, (tx, y1 + 36), 0.36, self.C_HUD, 1)

        bar_x, bar_w, bar_h = tx + 42, 110, 8
        rows = (
            # label   frac                                value      bidirectional
            ("YAW", cmd["yaw_rate"] / self.ap.MAX_YAW, f"{cmd['yaw_rate']:+.2f}r/s", True),
            ("CLB", -cmd["vz"]      / self.ap.MAX_VZ,  f"{-cmd['vz'] + 0.0:+.2f}m/s", True),
            ("FWD", cmd["vx"]       / self.ap.MAX_FWD, f"{cmd['vx']:.2f}m/s",        False),
        )
        for i, (label, frac, val, bidir) in enumerate(rows):
            y = y1 + 48 + i * 20
            self._text(frame, label, (tx, y + bar_h), 0.38, self.C_HUD, 1)
            self._hbar(frame, bar_x, y, bar_w, bar_h, frac, accent, bidir)
            self._text(frame, val, (bar_x + bar_w + 8, y + bar_h), 0.38, self.C_HUD, 1)

        rate = f"TX RATE = DET RATE  (skip {self.skip_frames})"
        self._text(frame, rate, (tx, y1 + ph - 8), 0.34, self.C_DIM, 1)

    # ── master draw ──────────────────────────────────────────────────────────────

    def _draw(self, frame, frame_full, pos, roi, cmd, sent):
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
        self._draw_flightcmd(frame, cmd, pos, cx_f, cy_f, sent, accent)

        self._draw_inset(frame, frame_full, pos)
        
        
    # ──────────────────────────────────────────────────────────────────────────
    # CAPTURE FACTORY  ← new helper that opens the right source
    # ──────────────────────────────────────────────────────────────────────────

    def _open_capture(self):
        """Open VideoCapture from either a GStreamer pipeline or a file path.

        Returns (cap, src_fps, source_label).
        """
        if self.camera_pipeline is not None:
            print(f"[CAP] Opening GStreamer pipeline:\n      {self.camera_pipeline}")
            cap = cv2.VideoCapture(self.camera_pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                raise RuntimeError(
                    "Cannot open GStreamer pipeline.  Check that:\n"
                    "  • OpenCV was built with GStreamer support\n"
                    "  • nvarguscamerasrc / nvvidconv are available on this device\n"
                    "  • sensor-id is correct\n"
                    f"Pipeline: {self.camera_pipeline}"
                )
            # GStreamer appsink usually reports 0 for FPS — fall back to caller's value
            reported = cap.get(cv2.CAP_PROP_FPS)
            src_fps  = reported if (reported and reported == reported and reported > 1) \
                       else self.camera_fps
            label = f"camera (pipeline, ~{src_fps:.1f} fps)"
        else:
            print(f"[CAP] Opening file: {self.video_path}")
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open: {self.video_path}")
            reported = cap.get(cv2.CAP_PROP_FPS)
            src_fps  = reported if (reported and reported == reported and reported > 1) \
                       else 30.0
            label = f"file ({self.video_path}, {src_fps:.1f} fps)"

        return cap, src_fps, label

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        # Fail loudly BEFORE flight if the link is dead or the firmware is wrong.
        self.ap.connect()

        # Controller calibration in effect.  The camera FOV converts pixel error
        # to an angular error, so it MUST match the real deployed lens or the
        # centring gains are wrong on the airframe — surface it every run.
        print(f"[CTRL CFG] camera FOV h/v = {np.degrees(self.ap.H_FOV):.1f}/"
              f"{np.degrees(self.ap.V_FOV):.1f} deg  (calibrate to the real lens!)  | "
              f"gains yaw={self.ap.K_YAW} vz={self.ap.K_VZ}  | "
              f"limits yaw<={self.ap.MAX_YAW} vz<={self.ap.MAX_VZ} "
              f"fwd<={self.ap.MAX_FWD} m/s")

        cap, src_fps, source_label = self._open_capture()
        print(f"[RUN] Source: {source_label}")
            
        # ── recording setup ────────────────────────────────────────────────────
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
            out_dir = Path(self.record_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # derive a sensible filename regardless of whether we have a file path
            stem = (Path(self.video_path).stem
                    if self.camera_pipeline is None
                    else "camera_live")
            out_path = out_dir / f"{stem}_detections.mp4"
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
                fused, _, _, roi, _ = self._process(frame_full, frame_disp)
            else:
                self.sky_mask = self.sky.mask(frame_disp)
                fused = self.kalman.predict() if self.kalman.initialized else None
                roi   = None
                if fused:
                    self.pos   = fused
                    self.layer = "KF-SKIP"

            # ── STEERING COMMAND ─────────────────────────────────────────────
            # A MAVLink setpoint is transmitted to the flight controller on
            # EVERY detection frame (rate = camera_fps / (skip_frames + 1)):
            #   • target in view (LOCK/PREDICT) → centre it: yaw + vz null the
            #     pixel offset so the drone converges on the target.
            #   • no target (SEARCH)            → advance straight ahead at MAX
            #     forward speed, holding heading & altitude, until re-acquired.
            # Skip frames send nothing; the FC coasts on its last setpoint
            # (GUID_TIMEOUT guards the link).
            # commanding: a setpoint is issued this frame (any detection frame).
            # actually_sent: it was really written to the FC (live link only;
            # offline it is computed for display/SIM but not transmitted).
            commanding    = is_yolo_frame
            actually_sent = False
            target_off    = None
            if is_yolo_frame:
                if fused:
                    target_off = (fused[0] - cx_f, fused[1] - cy_f)
                    cmd = self.ap.compute(target_off[0], target_off[1],
                                          self.state, self.fw, self.fh)
                else:
                    cmd = self.ap.search_command()   # SEARCH: forward at max
                actually_sent = self.ap.send(cmd)    # False when offline
                self._last_cmd = cmd
            else:
                cmd = self._last_cmd               # HUD display only, not sent
                if fused:
                    target_off = (fused[0] - cx_f, fused[1] - cy_f)

            # HUD lamp: TX (live+commanding) / SIM (offline+commanding) / HOLD.
            self._draw(frame_disp, frame_full, fused, roi, cmd, commanding)
          
            # ── detection-gated recording ──────────────────────────────────────
            if rec_on:
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
               pre_buf.append((self._frame_idx, frame_disp.copy()))            

            # Print ONLY the real controller command steering the drone.
            print(ArduPilotCommander.format_command(
                self.state, cmd, target_off, actually_sent,
                live=self.ap.master is not None, is_yolo=is_yolo_frame))

            if not self.headless:
                cv2.imshow(self._WIN, frame_disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if self.max_frames is not None and self._frame_idx >= self.max_frames:
                break

        self.ap.close()        # sends a final all-stop, then closes the link
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

def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


@dataclass
class CameraConfig:
    # Camera selection
    sensor_id: int = int(os.getenv("CAM_SENSOR_ID", "0"))
    sensor_mode: int = int(os.getenv("CAM_SENSOR_MODE", "0"))

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
        f"sensor-mode={cfg.sensor_mode}",
        f"tnr-mode={cfg.tnr_mode}",
        f"tnr-strength={cfg.tnr_strength}",
        f"ee-mode={cfg.ee_mode}",
        f"ee-strength={cfg.ee_strength}",
    ]

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

if __name__ == "__main__":
    cam_cfg = CameraConfig()
    CAP_PIPELINE = make_pipeline(cam_cfg)

    DroneTracker(
        model_path   = "./runs/detect/v0/weights/best.pt",
        camera_pipeline=CAP_PIPELINE,
        camera_fps=cam_cfg.fps,
        # --- FILE MODE (comment out camera_pipeline above, uncomment below) ---
        # video_path   = "./data/videos_drones/field.MP4",
        conf_thresh  = 0.01,
        display_size = (640, 640),
        skip_frames  = 0,
        show_sky     = True,
        record_dir      = "./captures",
        record_pad_sec  = 5.0,
        headless     = os.environ.get("TRACKER_HEADLESS") == "1",
        # On the Jetson set MAV_DEVICE=/dev/ttyACM0 (USB bench link to the
        # Lucid H7) or the UART device (FC side: SERIALn_PROTOCOL = 2).
        # Unset → offline: commands are computed and printed, nothing is sent.
        mav_device   = os.environ.get("MAV_DEVICE"),
    ).run()
