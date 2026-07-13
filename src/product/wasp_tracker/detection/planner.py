import numpy as np


class DetectionPlanner:
    def __init__(self, detector, config):
        self.detector = detector
        self.imgsz_search = config.imgsz_search
        self.imgsz_track = config.imgsz_track
        self.imgsz_close = config.imgsz_close
        self.crop_pad = config.crop_pad
        self.min_roi_half = config.min_roi_half
        self.use_clahe = config.use_clahe

    def plan(self, ctx, tracker, kf_pred, small):
        if tracker.state in ("LOCK", "PREDICT") and kf_pred:
            roi = self._predict_roi(ctx, tracker)
            track_imgsz = self.imgsz_track if small else self.imgsz_close
            det = self.detector.detect(ctx, self._roi_to_native(ctx, roi),
                                       track_imgsz, self.use_clahe and small)
            if det is None:
                sky_bbox = self._sky_bbox_native(ctx)
                if sky_bbox is not None:
                    srch_imgsz = self.imgsz_search if small else self.imgsz_track
                    det = self.detector.detect(ctx, sky_bbox, srch_imgsz,
                                               self.use_clahe and small)
                roi = None
            return det, roi

        sky_bbox = self._sky_bbox_native(ctx)
        det = None
        if sky_bbox is not None:
            det = self.detector.detect(ctx, sky_bbox, self.imgsz_search, self.use_clahe)
        return det, None

    def _predict_roi(self, ctx, tracker):
        px, py = tracker.kalman.predict()
        vx, vy = tracker.kalman.velocity()
        bw, bh = (tracker.last_bbox[2], tracker.last_bbox[3]) if tracker.last_bbox else (80, 80)

        base = max(self.min_roi_half, int(max(bw, bh) * 0.6))
        hw = min(base + int(abs(vx) * tracker.missed * 1.5), ctx.disp_w // 2)
        hh = min(base + int(abs(vy) * tracker.missed * 1.5), ctx.disp_h // 2)

        return (
            max(0, px - hw), max(0, py - hh),
            min(ctx.disp_w, px + hw), min(ctx.disp_h, py + hh),
        )

    def _roi_to_native(self, ctx, roi):
        sx = ctx.full_w / ctx.disp_w
        sy = ctx.full_h / ctx.disp_h
        return (int(roi[0] * sx), int(roi[1] * sy),
                int(roi[2] * sx), int(roi[3] * sy))

    def _sky_bbox_native(self, ctx):
        if ctx.sky_mask is None:
            return None
        ys, xs = np.where(ctx.sky_mask > 0)
        if xs.size == 0:
            return None
        sx = ctx.full_w / ctx.disp_w
        sy = ctx.full_h / ctx.disp_h
        x1 = int(xs.min() * sx) - self.crop_pad
        x2 = int(xs.max() * sx) + self.crop_pad
        y1 = int(ys.min() * sy) - self.crop_pad
        y2 = int(ys.max() * sy) + self.crop_pad
        return (x1, y1, x2, y2)
