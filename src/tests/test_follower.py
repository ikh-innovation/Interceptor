#!/usr/bin/env python3
import contextlib
import io
import math
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

with contextlib.redirect_stdout(io.StringIO()):
    import tracker as T
from follower import TargetFollower

FAILED = []
W = H = 640


def check(label, ok, detail=""):

    if not ok:
        FAILED.append(label)


def cloud_field(rng, n=1024):
    f = np.zeros((n, n), np.float32)
    for sc, amp in ((96, 1.0), (32, 0.5), (12, 0.25), (4, 0.10)):
        g = rng.standard_normal((n // sc + 2, n // sc + 2)).astype(np.float32)
        f += amp * cv2.resize(g, (n, n), interpolation=cv2.INTER_CUBIC)
    f = (f - f.min()) / (f.max() - f.min())
    return (150 + 90 * f).astype(np.float32)


def drone_sprite(size):
    s = max(3, int(size))
    k = s * 4 + 1
    img = np.zeros((k, k), np.float32)
    c = k // 2
    cv2.ellipse(img, (c, c), (max(1, s), max(1, s // 2)), 0, 0, 360, 1.0, -1, lineType=cv2.LINE_AA)
    if s >= 5:
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            cv2.circle(img, (c + dx * s, c + dy * (s // 2 + 1)), max(1, s // 3), 1.0, -1, lineType=cv2.LINE_AA)
    img = cv2.resize(img, (k // 2 | 1, k // 2 | 1), interpolation=cv2.INTER_AREA)
    return img


def render(field, ox, oy, sprite, tx, ty, rng, dark=70.0):
    fr = field[int(oy):int(oy) + H, int(ox):int(ox) + W].copy()
    if sprite is not None:
        k = sprite.shape[0]
        x0, y0 = int(round(tx)) - k // 2, int(round(ty)) - k // 2
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + k), min(H, y0 + k)
        if xb > xa and yb > ya:
            a = sprite[ya - y0:yb - y0, xa - x0:xb - x0]
            reg = fr[ya:yb, xa:xb]
            fr[ya:yb, xa:xb] = reg * (1 - a) + dark * a
    fr += rng.normal(0, 2.0, fr.shape).astype(np.float32)
    fr = np.clip(fr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)


def make_tracker(follow=True):
    t = T.DroneTracker.__new__(T.DroneTracker)
    t.fw, t.fh = W, H
    t.full_w, t.full_h = W, H
    t.kalman = T.KalmanTracker()
    t.state, t.missed, t.last_bbox, t.pos = "SEARCH", 0, None, None
    t.layer, t.conf, t.gate_note = "STANDBY", None, ""
    t._frame_idx, t._t_last_frame, t._frame_dt = 0, None, None
    t._gate_rejects, t.gated_total = 0, 0
    t.MISS_TO_PREDICT, t.MISS_TO_SEARCH, t.MIN_ROI_HALF = 2, 20, 50
    t.use_sky_mask, t.sky_mask, t._ground_rejected = False, None, False
    t.follower, t.use_follower, t.follow_hits, t._yolo_disagree = TargetFollower(), follow, 0, 0
    t.ap = None
    return t


class FakeDetector:

    def __init__(self, rng, p, p_false=0.0):
        self.rng, self.p, self.p_false = rng, p, p_false
        self.truth = None
        self.distractor = None
        self.calls = 0

    def __call__(self, frame_full, region, imgsz, enhance):
        x1, y1, x2, y2 = region
        r = self.rng
        self.calls += 1
        if self.distractor is not None:
            dx, dy, dw, dh, dp, dc = self.distractor
            if x1 <= dx < x2 and y1 <= dy < y2 and r.random() < dp:
                return (int(dx), int(dy), int(dx - dw / 2), int(dy - dh / 2), int(dw), int(dh), dc)
        if self.truth is not None:
            cx, cy, w, h = self.truth
            if x1 <= cx < x2 and y1 <= cy < y2 and r.random() < self.p:
                jx, jy = r.gauss(0, 0.7), r.gauss(0, 0.7)
                return (int(cx + jx), int(cy + jy), int(cx - w / 2), int(cy - h / 2),
                        max(1, int(w)), max(1, int(h)), 0.4 + 0.4 * r.random())
        if r.random() < self.p_false:
            fx, fy = r.uniform(max(x1, 30), min(x2, W - 30)), r.uniform(max(y1, 30), min(y2, H - 30))
            return (int(fx), int(fy), int(fx - 5), int(fy - 3), 10, 6, 0.25)
        return None


def run(size, follow, seed, p_det=0.15, p_false=0.0, frames=900, leave_at=None, vanish_at=None,
        distract=None):
    rng = np.random.default_rng(seed)
    pr = random.Random(seed)
    field = cloud_field(rng)
    sprite = drone_sprite(size)
    t = make_tracker(follow)
    det = FakeDetector(pr, p_det, p_false)
    t._detect_region = det
    x, y = 180.0, 330.0
    vx, vy = 2.2, -0.6
    ox, oy = 150.0, 150.0
    out = []
    first = None
    for i in range(frames):
        ang = 0.02 * math.sin(i / 40.0)
        vx, vy = vx * math.cos(ang) - vy * math.sin(ang), vx * math.sin(ang) + vy * math.cos(ang)
        if leave_at is not None and i >= leave_at:
            vx, vy = 9.0, 0.0
        x, y = x + vx, y + vy
        if leave_at is None and (x < 60 or x > W - 60):
            vx = -vx
        if leave_at is None and (y < 60 or y > H - 60):
            vy = -vy
        ox += 0.4 + pr.gauss(0, 1.2)
        oy += 0.15 + pr.gauss(0, 1.2)
        ox, oy = float(np.clip(ox, 0, 1024 - W - 1)), float(np.clip(oy, 0, 1024 - H - 1))
        visible = (vanish_at is None or i < vanish_at) and 0 <= x < W and 0 <= y < H
        fr = render(field, ox, oy, sprite if visible else None, x, y, rng)
        on = distract is not None and distract[0] <= i < distract[1]
        if on and distract[3]:
            g = render(field, ox, oy, sprite, distract[2][0], distract[2][1], rng)
            k = sprite.shape[0] + 6
            dx0, dy0 = int(distract[2][0]) - k // 2, int(distract[2][1]) - k // 2
            fr[dy0:dy0 + k, dx0:dx0 + k] = g[dy0:dy0 + k, dx0:dx0 + k]
        det.distractor = distract[2] if on else None
        bw, bh = 2.0 * size, 1.2 * size
        det.truth = (x, y, bw, bh) if visible and not (on and distract[4]) else None
        t._frame_idx = i
        fused, _a, _b, _roi, detected = t._process(fr, fr)
        if detected and first is None:
            first = i
        err = None if fused is None else math.hypot(fused[0] - x, fused[1] - y)
        out.append((bool(detected), err, visible, t.layer, fused))
    t.calls = det.calls
    return out, first, t



for size in (3, 5, 8, 14):
    res = {}
    for follow in (False, True):
        cov, errs = [], []
        for seed in (1, 2, 3):
            out, first, t = run(size, follow, seed)
            seg = out[first:] if first is not None else []
            if not seg:
                cov.append(0.0)
                continue
            cov.append(sum(1 for d, e, v, _l, _f in seg if d and e is not None and e < max(4.0, size)) / len(seg))
            errs += [e for d, e, v, _l, _f in seg if d and e is not None]
        res[follow] = (sum(cov) / len(cov), float(np.percentile(errs, 95)) if errs else 99.0)
    (c0, _e0), (c1, e1) = res[False], res[True]
    check(f"{size:2d} px drone, detector on 15% of frames: tracked {100 * c1:.0f}% of frames "
          f"(detector alone {100 * c0:.0f}%), p95 error {e1:.1f} px",
          c1 >= 0.93 and e1 <= max(3.0, 0.6 * size), f"follower off {100 * c0:.0f}%")


for size in (4, 10):
    covs = []
    for seed in (4, 5):
        out, first, t = run(size, True, seed, p_det=0.06)
        seg = out[first:]
        covs.append(sum(1 for d, e, v, _l, _f in seg if d and e is not None and e < max(4.0, size)) / len(seg))
    check(f"{size:2d} px, detector 6%: tracked {100 * min(covs):.0f}% of frames at worst", min(covs) >= 0.85,
          str([round(100 * c) for c in covs]))


for size in (5, 12):
    out, first, t = run(size, True, 7, leave_at=500)
    gone = [o for o in out[500:] if not o[2]]
    fake = sum(1 for d, e, v, _l, _f in gone if d)
    check(f"{size:2d} px leaves the frame: no 'detection' after it is gone ({fake} of {len(gone)} frames)",
          len(gone) > 50 and fake == 0, f"final state {t.state}")
    out, first, t = run(size, True, 8, vanish_at=450)
    gone = out[450:]
    fake = sum(1 for d, e, v, _l, _f in gone if d)
    check(f"{size:2d} px vanishes in the middle of the sky: at most a few frames of follower "
          f"({fake} of {len(gone)}), then let go", fake <= 3 and t.state == "SEARCH",
          f"final {t.state}, layer {t.layer}")


for size in (5, 10):
    out, first, t = run(size, True, 9, p_false=0.02)
    seg = out[first:]
    off = sum(1 for d, e, v, _l, _f in seg if d and e is not None and e > 3 * max(4.0, size))
    good = sum(1 for d, e, v, _l, _f in seg if d and e is not None and e < max(4.0, size)) / len(seg)
    check(f"{size:2d} px with cloud false alarms: tracked {100 * good:.0f}%, measurements far off "
          f"the target {off} of {len(seg)}", good >= 0.90 and off <= 0.01 * len(seg))


CORNER = (70.0, 60.0)
for size in (5, 10):
    for label, twin, blind in (("a cloud speck at 0.9 on every frame", False, False),
                               ("...and the detector blind to the target meanwhile", False, True),
                               ("a second, identical drone detected there", True, False)):
        for follow in (True, False):
            spot = (CORNER[0], CORNER[1], 2.0 * size, 1.2 * size, 1.0, 0.9)
            out, first, t = run(size, follow, 12, frames=600, distract=(300, 450, spot, twin, blind))
            seg = out[300:450]
            stolen = sum(1 for _d, _e, _v, _l, f in seg
                         if f is not None and math.hypot(f[0] - CORNER[0], f[1] - CORNER[1]) < 30)
            held = sum(1 for d, e, v, _l, _f in seg if d and e is not None and e < max(4.0, size)) / len(seg)
            after = sum(1 for d, e, v, _l, _f in out[450:] if d and e is not None and e < max(4.0, size)) / 150
            if follow:
                res_on = (stolen, held, after)
            else:
                res_off = stolen
        stolen, held, after = res_on
        check(f"{size:2d} px, {label}: 0 frames on the corner ({stolen} of 150), target held "
              f"{100 * held:.0f}%, after it {100 * after:.0f}%",
              stolen == 0 and held >= 0.90 and after >= 0.90,
              f"today's tracker: {res_off} of 150 frames on the corner")


for size in (5, 10):
    out, first, t = run(size, True, 13, frames=600)
    per = t.calls / 600.0
    _o, _f, t0 = run(size, False, 13, frames=600)
    check(f"{size:2d} px: {per:.2f} detector calls per frame (follower off: {t0.calls / 600.0:.2f})",
          per <= 1.02)


import time
rng = np.random.default_rng(11)
field = cloud_field(rng)
fr = render(field, 100, 100, drone_sprite(6), 320, 320, rng)
g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
f = TargetFollower()
f.seed(fr, 320, 320, 12, 7, 0.0)
f.seed(fr, 320, 320, 12, 7, 0.03)
for radius in (60, 120):
    t0 = time.perf_counter()
    for i in range(200):
        hit = f.track(fr, 322, 318, radius, 0.05)
    ms = (time.perf_counter() - t0) / 200 * 1000
    check(f"one follower search ({radius} px radius, three scales) takes {ms:.2f} ms on this CPU",
          hit is not None and ms < 8.0)


if FAILED:

    for x in FAILED:
        pass
    sys.exit(1)
