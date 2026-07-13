from .kalman import KalmanTracker


class TargetTracker:
    def __init__(self, config, big_obj_frac):
        self.miss_to_predict = config.miss_to_predict
        self.miss_to_search = config.miss_to_search
        self.big_obj_frac = big_obj_frac
        self.kalman = KalmanTracker()
        self.state = "SEARCH"
        self.missed = 0
        self.last_bbox = None
        self.conf = None
        self.layer = "STANDBY"

    def predict(self):
        return self.kalman.predict() if self.kalman.initialized else None

    def target_is_small(self, disp_h):
        if self.last_bbox is None:
            return True
        bw, bh = self.last_bbox[2], self.last_bbox[3]
        return max(bw, bh) < disp_h * self.big_obj_frac

    def on_result(self, detection, kf_pred, small, ground_rejected):
        if detection is not None:
            self.kalman.update(detection.cx, detection.cy)
            self.last_bbox = (detection.x1, detection.y1, detection.w, detection.h)
            self.missed = 0
            self.state = "LOCK"
            self.conf = float(detection.conf)
            self.layer = f"ZOOM {detection.conf:.2f}" if small else f"CLOSE {detection.conf:.2f}"
            return (detection.cx, detection.cy)

        if kf_pred and self.state != "SEARCH":
            self.missed += 1
            if self.missed >= self.miss_to_search:
                self.reset()
                self.layer = "LOST"
                return None
            if self.missed >= self.miss_to_predict:
                self.state = "PREDICT"
            self.layer = f"KF  miss={self.missed}"
            return kf_pred

        self.missed += 1
        self.layer = "GROUND-REJ" if ground_rejected else "LOST"
        return None

    def mark_skip(self, has_prediction):
        if has_prediction:
            self.layer = "KF-SKIP"

    def reset(self):
        self.state = "SEARCH"
        self.missed = 0
        self.last_bbox = None
        self.kalman = KalmanTracker()
        self.conf = None
        self.layer = "STANDBY"
