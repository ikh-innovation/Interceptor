

from collections import deque
import math

import cv2
import numpy as np

from wasp import camera_ray, rotate, rotation, unit


def project_bearing(bearing, att, width, height, tan_h, tan_v, camera_roll=0):
    down, right, forward = rotate(rotation(att), bearing, inverse=True)
    if forward >= -0.15:
        return None
    a = math.radians(camera_roll)
    x, y = right / -forward, down / -forward
    return (width / 2 + (x * math.cos(a) + y * math.sin(a)) * width / (2 * tan_h),
            height / 2 + (-x * math.sin(a) + y * math.cos(a)) * height / (2 * tan_v))


class WaspMotionEstimator:
    DETECTOR_TTL = 0.20
    FLOW_TTL = 0.15
    HISTORY_S = 0.22
    MIN_CUTOFF_HZ = 5.0
    SPEED_CUTOFF = 18.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.bearing = None
        self.velocity = np.zeros(3)
        self.history = deque()
        self.last_t = self.detector_t = None
        self.confidence = 0.0
        self.motion_confidence = 0.0
        self.filter_lag_s = 0.0
        self.noise_rad = 0.0
        self.box = None
        self.gray = self.points = self.template = None
        self.frame_t = None
        self.signature = None
        self.pending = None
        self.needs_confirmation = False
        self.rejected = 0
        self.flow_quality = 0.0
        self.source = "SEARCH"

    @staticmethod
    def _patch(gray, box):
        cx, cy, w, h = box
        if min(w, h) < 3 or not (w / 2 <= cx < gray.shape[1] - w / 2
                                and h / 2 <= cy < gray.shape[0] - h / 2):
            return None
        return cv2.getRectSubPix(gray, (max(3, round(w)), max(3, round(h))), (float(cx), float(cy)))

    def _seed(self, gray):
        self.points = None
        self.template = self._patch(gray, self.box)
        if self.template is None or self.template.std() < 4:
            return
        cx, cy, w, h = self.box
        x, y = max(0, math.floor(cx - w / 2)), max(0, math.floor(cy - h / 2))
        crop = gray[y:min(gray.shape[0], math.ceil(cy + h / 2)),
                    x:min(gray.shape[1], math.ceil(cx + w / 2))]
        if min(crop.shape) < 3:
            return
        pts = cv2.goodFeaturesToTrack(crop, 20, 0.02, 2, blockSize=3)
        if pts is not None and len(pts) >= 3:
            self.points = pts + np.array([[[x, y]]], dtype=np.float32)

    def _flow(self, gray, expected, now):
        self.flow_quality = 0.0
        if (self.gray is None or self.points is None or len(self.points) < 3
                or self.gray.shape != gray.shape or self.frame_t is None
                or not 0 < now - self.frame_t <= self.FLOW_TTL
                or self.detector_t is None or now - self.detector_t > self.FLOW_TTL):
            return None
        options = dict(winSize=(9, 9), maxLevel=2,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
        initial = None
        if expected is not None:
            initial = self.points + np.array([[[expected[0] - self.box[0],
                                                expected[1] - self.box[1]]]], np.float32)
        nxt, status, error = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, self.points, initial,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW if initial is not None else 0, **options)
        if nxt is None:
            return None
        back, bstatus, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, nxt, None, **options)
        if back is None:
            return None
        fb = np.linalg.norm(back - self.points, axis=2).reshape(-1)
        good = ((status.reshape(-1) == 1) & (bstatus.reshape(-1) == 1)
                & (fb < 0.8) & (error.reshape(-1) < 35)
                & np.isfinite(nxt.reshape(-1, 2)).all(axis=1))
        if good.sum() < 3:
            return None
        shifts = (nxt - self.points).reshape(-1, 2)[good]
        shift = np.median(shifts, axis=0)
        agree = np.linalg.norm(shifts - shift, axis=1) < 1.2
        if agree.sum() < 3:
            return None
        cx, cy, w, h = self.box
        box = (cx + float(shift[0]), cy + float(shift[1]), w, h)
        patch = self._patch(gray, box)
        if patch is None or patch.std() < 4 or self.template is None:
            return None
        patch = cv2.resize(patch, (self.template.shape[1], self.template.shape[0]))
        a, b = patch.astype(float).ravel(), self.template.astype(float).ravel()
        a, b = a - a.mean(), b - b.mean()
        corr = float(a @ b / max(1e-9, np.linalg.norm(a) * np.linalg.norm(b)))
        if corr < 0.55:
            return None
        self.points = nxt[good][agree].reshape(-1, 1, 2)
        self.flow_quality = min(1.0, corr * float(agree.mean()) * max(0.0, 1 - float(np.median(fb[good]))))
        return box

    def predicted_center(self, att, width, height, tan_h, tan_v, camera_roll, now):
        if self.bearing is None or self.last_t is None or now - self.last_t > self.DETECTOR_TTL:
            return None
        future = unit(self.bearing + self.velocity * (max(0.0, now - self.last_t) + self.filter_lag_s))
        return project_bearing(future, att, width, height, tan_h, tan_v, camera_roll)

    def _motion(self, world, now, quality, pixel_rad):
        self.history.append((now, np.array(world), quality))
        while self.history and now - self.history[0][0] > self.HISTORY_S:
            self.history.popleft()


        window = max(0.08, self.HISTORY_S / (1 + 6 * float(np.linalg.norm(self.velocity))))
        recent = [row for row in self.history if now - row[0] <= window]
        velocity, trust, residual = np.zeros(3), 0.0, pixel_rad
        if len(recent) >= 3 and now - recent[0][0] >= 0.045:
            times = np.array([r[0] - now for r in recent])
            values = np.array([r[1] for r in recent])
            weights = np.array([max(0.05, r[2]) for r in recent])

            for _ in range(2):
                mt = np.average(times, weights=weights)
                mean = np.average(values, axis=0, weights=weights)
                denom = float(np.sum(weights * (times - mt) ** 2))
                velocity = np.sum(weights[:, None] * (times - mt)[:, None] * (values - mean), axis=0) / max(1e-9, denom)
                errors = np.linalg.norm(values - mean - (times - mt)[:, None] * velocity, axis=1)
                residual = max(pixel_rad * 0.6, float(np.median(errors)))
                weights *= np.minimum(1.0, 2.5 * residual / np.maximum(errors, 1e-9))
            velocity -= np.array(world) * float(velocity @ world)
            speed = float(np.linalg.norm(velocity))
            span = now - recent[0][0]
            movement = speed * span
            trust = max(0.0, min(1.0, (movement / max(1e-9, 2 * residual) - 0.5) / 1.5))
            trust *= min(1.0, span / 0.08) * quality
            velocity *= min(1.0, 2.0 / max(speed, 1e-9)) * trust
        dt = 0.0 if self.last_t is None else now - self.last_t
        self.noise_rad = max(pixel_rad * 0.6, min(pixel_rad * 4, residual))
        self.motion_confidence = trust
        self.velocity = velocity
        if self.bearing is None:
            self.bearing = np.array(world)
        else:
            cutoff = self.MIN_CUTOFF_HZ + self.SPEED_CUTOFF * float(np.linalg.norm(velocity))
            self.filter_lag_s = 1 / (2 * math.pi * cutoff)
            alpha = dt / (dt + self.filter_lag_s)
            self.bearing = np.array(unit(self.bearing + alpha * (np.array(world) - self.bearing)))
        self.last_t = now

    def update(self, frame, detection, att, now, tan_h, tan_v, camera_roll=0,
               min_conf=0.2, synthetic=False):

        height, width = frame.shape[:2]
        signature = (width, height, tan_h, tan_v, camera_roll, synthetic)
        expired = self.detector_t is not None and now - self.detector_t > self.DETECTOR_TTL
        if (signature != self.signature or self.last_t is not None
                and (now <= self.last_t or now - self.last_t > self.DETECTOR_TTL) or expired):
            self.reset()
            self.needs_confirmation = expired and not synthetic
        self.signature = signature
        if not math.isfinite(now) or att is None:
            self.reset()
            return None
        try:
            mat = np.asarray(rotation(att), dtype=float)
            valid = (mat.shape == (3, 3) and np.isfinite(mat).all()
                     and np.allclose(mat.T @ mat, np.eye(3), atol=0.01)
                     and abs(np.linalg.det(mat)-1) < 0.01
                     and all(math.isfinite(v) and v > 0 for v in (tan_h, tan_v)))
        except (TypeError, ValueError, KeyError, OverflowError):
            valid = False
        if not valid:
            self.reset()
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        expected = self.predicted_center(att, width, height, tan_h, tan_v, camera_roll, now)
        flow = None if synthetic else self._flow(gray, expected, now)
        pixel_rad = max(2 * tan_h / width, 2 * tan_v / height)

        def world_at(box):
            return np.array(rotate(rotation(att), camera_ray(
                box[0] - width / 2, box[1] - height / 2, width, height, tan_h, tan_v, camera_roll)))

        accepted = False
        if detection is not None:
            try:
                vals = tuple(float(v) for v in detection)
            except (ValueError, TypeError, OverflowError):
                vals = ()
            if (len(vals) != 5 or not all(math.isfinite(v) for v in vals)
                    or not 0 <= vals[0] < width or not 0 <= vals[1] < height
                    or min(vals[2:4]) < 1 or max(vals[2:4]) > max(width, height)
                    or not min_conf <= vals[4] <= 1):
                detection = None
            else:
                candidate = world_at(vals)
                dt = 0 if self.last_t is None else now - self.last_t
                reference = (world_at(flow) if flow is not None else
                             None if self.bearing is None else np.array(unit(self.bearing + self.velocity * (dt + self.filter_lag_s))))
                size = max(vals[2:4])
                gate = max(pixel_rad * (3.0 + min(8.0, size * 0.2)),
                           3 * self.noise_rad) + max(0.10, float(np.linalg.norm(self.velocity)) * 1.5) * dt
                jump = 0 if reference is None else float(np.linalg.norm(candidate - reference))
                accepted = synthetic or not self.needs_confirmation and (reference is None or jump <= gate)


                if not accepted and flow is None:
                    if self.pending is not None:
                        pt, pending = self.pending
                        accepted = 0 < now - pt <= 0.10 and np.linalg.norm(candidate - pending) <= max(gate, 0.7 * (now - pt))
                        if accepted and self.template is not None and self.template.std() >= 4:
                            patch = self._patch(gray, vals[:4])
                            if patch is None:
                                accepted = False
                            else:
                                patch = cv2.resize(patch, (self.template.shape[1], self.template.shape[0]))
                                a, b = patch.astype(float).ravel(), self.template.astype(float).ravel()
                                a, b = a-a.mean(), b-b.mean()
                                accepted = a @ b / max(1e-9, np.linalg.norm(a)*np.linalg.norm(b)) >= 0.35
                    self.pending = (now, candidate)
                if accepted:
                    self.pending = None
                    self.needs_confirmation = False
                    self.box = vals[:4]
                    self.detector_t, self.confidence = now, vals[4]
                    self.source = "DETECTOR"
                else:
                    self.rejected += 1

        if not accepted:
            if flow is None or self.detector_t is None or now - self.detector_t > self.FLOW_TTL:
                self.gray, self.frame_t, self.points = gray, now, None
                self.source = "REJECTED" if detection is not None else "SEARCH"
                return None
            self.box, self.source = flow, "FLOW"
        quality = min(1.0, self.confidence / max(min_conf, 0.5)) if accepted else self.flow_quality
        world = world_at(self.box)
        self._motion(world, now, quality, pixel_rad)
        if accepted and not synthetic:
            self._seed(gray)
        self.gray, self.frame_t = gray, now


        current_bearing = unit(self.bearing + self.velocity * self.filter_lag_s)
        centre = project_bearing(current_bearing, att, width, height, tan_h, tan_v, camera_roll)
        if centre is None:
            return None
        cx, cy = centre
        return dict(wasp=True, have_att=True, motion_assist=True,
                    bearing=current_bearing, bearing_rate=tuple(self.velocity),
                    motion_confidence=self.motion_confidence, noise_rad=self.noise_rad,
                    capture_t=now, detector_capture_t=self.detector_t, camera_roll=camera_roll,
                    az_deg=math.degrees(math.atan(2 * (cx - width / 2) / width * tan_h)),
                    el_cam_deg=-math.degrees(math.atan(2 * (cy - height / 2) / height * tan_v)),
                    box=(cx, cy, self.box[2], self.box[3]), confidence=self.confidence,
                    source=self.source, flow_quality=self.flow_quality, rejected=self.rejected)
