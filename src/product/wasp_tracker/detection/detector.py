import cv2
import numpy as np

from ..context import Detection


class SkyGatedDetector:
    def __init__(self, model, sky, config, conf_thresh):
        self.model = model
        self.sky = sky
        self.conf_thresh = conf_thresh
        self._use_clahe = config.use_clahe
        self._clahe = cv2.createCLAHE(clipLimit=config.clahe_clip,
                                      tileGridSize=(config.clahe_grid, config.clahe_grid))
        self.ground_rejected = False

    def reset_frame(self):
        self.ground_rejected = False

    def warmup(self, imgsz):
        blank = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.model(blank, imgsz=imgsz, verbose=False)

    def _enhance(self, crop):
        if not self._use_clahe:
            return crop
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _accept_box(self, ctx, x1n, y1n, x2n, y2n, conf):
        sx = ctx.disp_w / ctx.full_w
        sy = ctx.disp_h / ctx.full_h
        dx1, dy1 = x1n * sx, y1n * sy
        dx2, dy2 = x2n * sx, y2n * sy
        bw, bh = dx2 - dx1, dy2 - dy1
        bbox = (int(dx1), int(dy1), max(1, int(round(bw))), max(1, int(round(bh))))

        if ctx.sky_mask is not None and not self.sky.background_is_sky(ctx.sky_mask, bbox):
            self.ground_rejected = True
            return None

        cx = int((dx1 + dx2) / 2)
        cy = int((dy1 + dy2) / 2)
        return Detection(cx, cy, int(dx1), int(dy1), bbox[2], bbox[3], float(conf))

    def detect(self, ctx, region_native, imgsz, enhance):
        x1r, y1r, x2r, y2r = region_native
        x1r = max(0, x1r); y1r = max(0, y1r)
        x2r = min(ctx.full_w, x2r); y2r = min(ctx.full_h, y2r)
        if x2r - x1r < 2 or y2r - y1r < 2:
            return None

        crop = ctx.frame_full[y1r:y2r, x1r:x2r]
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
            det = self._accept_box(ctx, x1 + x1r, y1 + y1r, x2 + x1r, y2 + y1r, confs[idx])
            if det is not None:
                return det
        return None
