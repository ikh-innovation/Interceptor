"""Causal preview stabilization: sparse optical flow, optional camera-axis gyro.

The image is never cropped or scaled by stabilization. A fixed padded canvas
contains every source pixel even at the maximum translation/rotation. Only the
previous grayscale frame and feature points are retained; there is no frame queue.
"""

from dataclasses import dataclass
import math
import threading

import cv2
import numpy as np


@dataclass(frozen=True)
class PreviewTransform:
    matrix: np.ndarray
    size: tuple

    def geometry(self, frame_size):
        return dict(width=self.size[0], height=self.size[1],
                    frame_width=frame_size[0], frame_height=frame_size[1],
                    inverse=cv2.invertAffineTransform(self.matrix).tolist())


class VideoStabilizer:
    ANALYSIS_WIDTH = 320
    MAX_POINTS = 160
    MIN_POINTS = 12
    MAX_SHIFT = 0.06
    MAX_ANGLE = math.radians(5)
    SMOOTH_SECONDS = 0.12
    MAX_GAP = 0.15
    GYRO_HOLD_SECONDS = 0.20

    def __init__(self):
        self.reset()

    def reset(self):
        self.gray = self.points = self.last_t = self.signature = None
        self.path = np.zeros(3, np.float64)
        self.smooth = self.path.copy()
        self.frames = 0
        self.last_visual_t = None
        self.source = "warming up"

    @staticmethod
    def _rigid(dx, dy, angle, width, height):
        c, s = math.cos(angle), math.sin(angle)
        x, y = (width - 1) / 2, (height - 1) / 2
        return np.array([[c, -s, dx + x - c*x + s*y],
                         [s, c, dy + y - s*x - c*y]], np.float64)

    def _canvas(self, frame_size, correction):
        w, h = frame_size
        # Equal fractional padding preserves the existing preview aspect ratio.
        a = self.MAX_ANGLE
        factor = max(math.cos(a) + h/w*math.sin(a),
                     math.cos(a) + w/h*math.sin(a)) + 2*self.MAX_SHIFT
        ow, oh = 2*math.ceil(w*factor/2), 2*math.ceil(h*factor/2)
        matrix = self._rigid(*correction, w, h)
        matrix[:, 2] += ((ow-w)/2, (oh-h)/2)
        return PreviewTransform(matrix, (ow, oh))

    def _features(self, gray):
        return cv2.goodFeaturesToTrack(gray, self.MAX_POINTS, 0.015, 7, blockSize=5)

    @staticmethod
    def _gyro_motion(gyro, dt, w, h, tan_h, tan_v):
        if gyro is None:
            return None
        rates = np.asarray(gyro, np.float64)
        if rates.shape != (3,) or not np.isfinite(rates).all():
            return None
        wx, wy, wz = rates
        return np.array([-w/(2*tan_h)*wy*dt, h/(2*tan_v)*wx*dt, -wz*dt])

    def _flow(self, gray, initial):
        if self.points is None or len(self.points) < self.MIN_POINTS:
            return None
        options = dict(winSize=(15, 15), maxLevel=2,
                       criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 15, 0.03))
        nxt, status, error = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, self.points, initial,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW if initial is not None else 0, **options)
        if nxt is None or status is None or error is None:
            return None
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, nxt, None, **options)
        if back is None or back_status is None:
            return None
        p, q = self.points.reshape(-1, 2), nxt.reshape(-1, 2)
        h, w = gray.shape
        good = (status.ravel().astype(bool) & back_status.ravel().astype(bool)
                & (error.ravel() < 30) & np.isfinite(q).all(axis=1)
                & (np.linalg.norm(back.reshape(-1, 2)-p, axis=1) < 1.0)
                & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h))
        p, q = p[good], q[good]
        if len(p) < self.MIN_POINTS:
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(
            p, q, method=cv2.RANSAC, ransacReprojThreshold=1.5,
            maxIters=100, confidence=0.95, refineIters=5)
        if matrix is None or inliers is None or not np.isfinite(matrix).all():
            return None
        keep = inliers.ravel().astype(bool)
        if keep.sum() < self.MIN_POINTS or keep.mean() < 0.5:
            return None
        p, q = p[keep], q[keep]
        if np.any(np.ptp(p, axis=0) < np.array([w, h])*0.25):
            return None  # A small moving object is not global camera motion.
        scale = math.hypot(matrix[0, 0], matrix[1, 0])
        if not 0.97 <= scale <= 1.03:
            return None
        angle = math.atan2(matrix[1, 0], matrix[0, 0])
        rigid = self._rigid(0, 0, angle, w, h)
        shift = np.median(q - (p @ rigid[:, :2].T + rigid[:, 2]), axis=0)
        motion = np.array([*shift, angle])
        if (abs(angle) > math.radians(15)
                or np.any(np.abs(shift) > np.array([w, h])*0.15)):
            return None
        self.points = q.astype(np.float32).reshape(-1, 1, 2)
        return motion

    def update(self, frame, capture_t, gyro=None, tan_h=0.603, tan_v=0.454, context=None):
        """Estimate a preview transform from the unannotated current frame.

        gyro is (wx, wy, wz) in camera right/down/forward axes, rad/s.
        context invalidates history when mounting, flip, or camera mode changes.
        """
        h, w = frame.shape[:2]
        signature = (w, h, tan_h, tan_v, context)
        dt = None if self.last_t is None else capture_t-self.last_t
        if signature != self.signature or dt is None or not 0 < dt <= self.MAX_GAP:
            self.reset()
            self.signature = signature
            dt = None
        sw = min(w, self.ANALYSIS_WIDTH)
        sh = max(1, round(h*sw/w))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        correction = np.zeros(3)
        motion = None
        if dt is not None:
            predicted = self._gyro_motion(gyro, dt, sw, sh, tan_h, tan_v)
            initial = None
            if predicted is not None and self.points is not None:
                matrix = self._rigid(*predicted, sw, sh)
                initial = cv2.transform(self.points, matrix).astype(np.float32)
            motion = self._flow(gray, initial)
            if motion is None and initial is not None:
                motion = self._flow(gray, None)
            if motion is not None:
                self.last_visual_t = capture_t
                self.source = "gyro + optical flow" if initial is not None else "optical flow"
            elif (predicted is not None and self.last_visual_t is not None
                  and capture_t-self.last_visual_t <= self.GYRO_HOLD_SECONDS
                  and abs(predicted[2]) <= math.radians(15)
                  and np.all(np.abs(predicted[:2]) <= np.array([sw, sh])*0.15)):
                motion = predicted
                self.points = None
                self.source = "gyro"
            else:
                self.path[:] = self.smooth[:] = 0
                self.points = None
                self.source = "waiting for features"
        if motion is not None:
            motion[:2] *= (w/sw, h/sh)
            self.path += motion
            alpha = 1-math.exp(-dt/self.SMOOTH_SECONDS)
            self.smooth += alpha*(self.path-self.smooth)
            limit = np.array([w*self.MAX_SHIFT, h*self.MAX_SHIFT, self.MAX_ANGLE])
            correction = np.clip(self.smooth-self.path, -limit, limit)
            self.smooth = self.path+correction
            # Rebase the trajectory to avoid unbounded numbers during long pans.
            self.path[:] = 0
            self.smooth = correction.copy()
        self.frames += 1
        if self.points is None or len(self.points) < 80 or self.frames % 8 == 0:
            self.points = self._features(gray)
        self.gray, self.last_t = gray, capture_t
        return self._canvas((w, h), correction)

    @staticmethod
    def render(frame, transform):
        return cv2.warpAffine(frame, transform.matrix, transform.size,
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def camera_gyro(attitude, mount_deg, camera_roll=0):
    """Project existing FRD body rates onto the confirmed camera mounting axes."""
    if not isinstance(attitude, dict):
        return None
    try:
        age = float(attitude["age"])
        p, q, r = (float(attitude[k]) for k in ("rollspeed", "pitchspeed", "yawspeed"))
        tilt, roll = math.radians(mount_deg), math.radians(camera_roll)
    except (KeyError, ValueError, TypeError):
        return None
    if not all(math.isfinite(v) for v in (age, p, q, r, tilt, roll)) or not 0 <= age <= 0.15:
        return None
    x, y, z = q, math.sin(tilt)*p+math.cos(tilt)*r, math.cos(tilt)*p-math.sin(tilt)*r
    c, s = math.cos(roll), math.sin(roll)
    return (c*x+s*y, -s*x+c*y, z)


class PreviewWorker:
    """Single replaceable input slot: slow preview work cannot delay capture.

    The processor receives owned arrays. submit() never waits for OpenCV work;
    superseded inputs are discarded instead of accumulating camera latency.
    """

    def __init__(self, processor):
        self.processor = processor
        self._condition = threading.Condition()
        self._pending = None
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="video-stabilization", daemon=True)
        self._thread.start()

    def submit(self, raw, annotated, capture_t, generation):
        packet = (raw, annotated.copy(), capture_t, generation)
        with self._condition:
            if not self._stopped:
                self._pending = packet
                self._condition.notify()

    def discard(self):
        with self._condition:
            self._pending = None

    def close(self):
        with self._condition:
            self._stopped = True
            self._pending = None
            self._condition.notify()
        self._thread.join(timeout=2)

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._stopped)
                if self._stopped:
                    return
                packet, self._pending = self._pending, None
            self.processor(*packet)
