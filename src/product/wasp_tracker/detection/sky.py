import cv2
import numpy as np


class SkyDetector:
    def __init__(self, config):
        self.v_min = config.v_min
        self.s_max = config.s_max
        self.h_lo = config.h_lo
        self.h_hi = config.h_hi
        self.ring_pad_frac = config.ring_pad_frac
        self.ring_pad_min = config.ring_pad_min
        self.sky_frac = config.sky_frac
        self.top_band = config.top_band
        self.bottom_band = config.bottom_band
        self.ground_min_frac = config.ground_min_frac
        self._k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def mask(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        bright = V >= self.v_min
        blue = (H >= self.h_lo) & (H <= self.h_hi)
        lowsat = S <= self.s_max
        sky = (bright & (blue | lowsat)).astype(np.uint8) * 255

        sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN, self._k)
        sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, self._k)

        n, labels = cv2.connectedComponents(sky)
        if n <= 1:
            top_sky = np.zeros_like(sky)
        else:
            top_labels = np.unique(labels[: self.top_band, :])
            top_labels = top_labels[top_labels != 0]
            top_sky = (np.isin(labels, top_labels).astype(np.uint8) * 255
                       if top_labels.size else np.zeros_like(sky))

        return self._reclaim_embedded_blobs(top_sky)

    def _reclaim_embedded_blobs(self, sky):
        Hm, Wm = sky.shape
        non_sky = (sky == 0).astype(np.uint8)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(non_sky)
        if n <= 1:
            return np.full_like(sky, 255)

        band = max(1, self.bottom_band)
        bottom_labels = np.unique(labels[Hm - band:, :])
        bottom_labels = bottom_labels[bottom_labels != 0]

        min_area = self.ground_min_frac * Hm * Wm
        keep = [lab for lab in bottom_labels
                if stats[lab, cv2.CC_STAT_AREA] >= min_area]

        ground = np.isin(labels, keep) if keep else np.zeros(sky.shape, bool)
        out = np.where(ground, 0, 255).astype(np.uint8)
        out[sky > 0] = 255
        return out

    def background_is_sky(self, mask, bbox):
        x, y, w, h = bbox
        pad = max(self.ring_pad_min, int(max(w, h) * self.ring_pad_frac))
        Hm, Wm = mask.shape

        ox1 = max(0, x - pad); oy1 = max(0, y - pad)
        ox2 = min(Wm, x + w + pad); oy2 = min(Hm, y + h + pad)
        outer = mask[oy1:oy2, ox1:ox2]
        if outer.size == 0:
            return False

        ix1 = max(0, x - ox1); iy1 = max(0, y - oy1)
        ix2 = min(outer.shape[1], (x + w) - ox1)
        iy2 = min(outer.shape[0], (y + h) - oy1)
        inner = outer[iy1:iy2, ix1:ix2]

        ring_total = outer.size - inner.size
        if ring_total <= 0:
            return False
        ring_sky = cv2.countNonZero(outer) - (cv2.countNonZero(inner) if inner.size else 0)
        return (ring_sky / ring_total) >= self.sky_frac
