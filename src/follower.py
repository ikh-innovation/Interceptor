"""
follower.py  -  keep a tiny target once the detector has found it
==================================================================
Field, 2026-09-30 (Bf 3-4, a drone 6-20 px across in a 640 px frame): the
detector fired on only a fraction of the frames that showed the target, the
tracker coasted on its Kalman guess, and after 20 misses it dropped the
track and started again from nothing.  The drone was still in the picture.

The follower closes those gaps.  Whenever the DETECTOR confirms the target,
the follower keeps a small picture of it (the ANCHOR template).  On every
frame the detector misses, it looks for the target near where the Kalman
filter says it should be (normalised cross-correlation) and, if it is found
clearly enough, reports it as a measurement.  The detector keeps running
every frame; its next hit re-seeds the anchor.

HARD TO TAKE, IMPOSSIBLE TO STEAL.  Holding on hard amplifies whatever the
tracker latched onto, so the follower only ARMS on a confirmed target (two
detector hits within CONFIRM_S), and once armed a detection elsewhere never
takes the track away just by insisting: the tracker asks the follower
(verify) whether that detection LOOKS like the target it is holding.  Replay
of the 2026-09-30 field video: the top detection jumped > 90 px between two
detections 1173 times -- a second detection elsewhere is the rule, not the
exception.

Replay of the 2026-09-30 flight videos showed what a follower must cope with:
  * the drone moves 20-100 px between frames (gusts, the airframe's own
    shake and yaw): the search window must be wide -- 40 px at least, more
    with the target's speed and the airframe's rotation;
  * its look changes as it turns (the anchor alone matched at ~0.5): a
    second, RECENT template follows the change, updated only from strong
    matches that ALSO still resemble the anchor, so it cannot wander off;
  * it is often near the edge of the picture: windows are edge-padded.

What stops it from following the sky instead of the drone:
  * a match must score well (SCORE_MIN) AND be clearly better than the best
    match elsewhere in the window (PEAK_MARGIN), however high it scores: a
    smooth cloud gradient matches a smooth template at 0.99 -- everywhere
    (replay 2026-09-30: the follower slid onto bare cloud that way);
  * the matched spot must hold a blob that stands out from its surround
    (TEX_MIN, the same test as for seeding), as the drone does;
  * it must still resemble the detector's own picture (ANCHOR_MIN);
  * a template must stand out from its surroundings when seeded (TEX_MIN),
    and a match must have some structure (FLAT_MIN);
  * VIS_MAX_S: without a detector confirmation for this long it lets go.

Cost: only the search window is cut out, turned grey and matched, in 8-bit
(OpenCV's SIMD path); the frame itself is never copied.  matchTemplate is
single-threaded, and on the Jetson at 15 W (6 cores, 1.5 GHz) one 25 px
template over a 120 px radius window takes 8.2 ms -- the two templates were
~20 ms a frame, a third of the frame budget, exactly when the target moves
fast.  A wide window is therefore cut into horizontal strips (overlapping by
the template's height, so every position sees the same pixels and scores
the same) matched on several cores at once.
"""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


class TargetFollower:
    SCORE_MIN = 0.55         # NCC a match must reach (best of the two templates)
    ANCHOR_MIN = 0.35        # ...and it must still resemble the detector's picture
    PEAK_MARGIN = 0.10       # ...and beat the best match elsewhere by this much
    ADAPT_MIN = 0.80         # only matches this strong update the RECENT template
    ADAPT_RATE = 0.25
    TEX_MIN = 6.0            # grey levels: the target must differ from its surround
    FLAT_MIN = 3.0           # a matched patch flatter than this is empty sky
    MIN_SIDE = 13            # px, smallest template (a 4 px target still gets context)
    MAX_SIDE = 73
    PAD = 1.6                # template side = PAD x the box's larger side (+ context)
    SCALES = (0.85, 1.18)    # tried around the best position found at scale 1
    VIS_MAX_S = 4.0          # longest the follower holds without a detector hit
                             # (field 2026-09-30: the longest same-target gap 3.6 s)
    SEED_BLEND = 0.35        # a new detector hit is blended into the anchor
    CONFIRM_HITS = 2         # detector hits before the follower arms...
    CONFIRM_S = 0.7          # ...each within this long of the one before
    MAX_RADIUS = 120         # px, the widest search around the prediction
    VERIFY_R = 4             # px, slack around a detection when verifying it

    def __init__(self):
        self.reset()

    def reset(self):
        self.anchor = None       # float32, from DETECTOR hits only
        self.recent = None       # float32, follows the target's changing look
        self._a8 = self._r8 = None   # their 8-bit copies, what is matched
        self.side = 0
        self.box = None          # (w, h) of the last detector box
        self.t_seed = None       # time of the last detector confirmation
        self.n_conf = 0          # detector hits in the confirmation run
        self.score = None
        self.a_score = None      # the last hit's likeness to the anchor
        self.hits = 0            # follower hits since the last detector hit
        self.misses = 0          # follower searches in a row that found nothing
        self.why = ""            # why the last seed / track failed (for the logs)

    @property
    def active(self):
        return self.anchor is not None

    @property
    def armed(self):
        """A CONFIRMED target: the follower tracks it and guards it."""
        return self.anchor is not None and self.n_conf >= self.CONFIRM_HITS

    # ── the detector confirmed the target ────────────────────────────────
    def seed(self, frame, cx, cy, w, h, now):
        side = int(np.clip(self.PAD * max(w, h) + 6, self.MIN_SIDE, self.MAX_SIDE)) | 1
        patch = _crop(frame, int(round(cx)) - side // 2, int(round(cy)) - side // 2, side, side)
        if patch is None:
            self.why = "seed: off the frame"
            return False
        patch = patch.astype(np.float32)
        if _texture(patch) < self.TEX_MIN:
            self.why = "seed: no contrast"
            return False
        fresh = self.t_seed is not None and now - self.t_seed <= (
            self.VIS_MAX_S if self.armed else self.CONFIRM_S)
        if not fresh:
            self.anchor, self.n_conf = None, 0
        if self.anchor is not None and self.anchor.shape == patch.shape:
            self.anchor = (1.0 - self.SEED_BLEND) * self.anchor + self.SEED_BLEND * patch
        else:
            self.anchor = patch
        self.n_conf = min(self.CONFIRM_HITS, self.n_conf + 1)
        self.recent = self.anchor.copy()
        self._a8 = self._r8 = _u8(self.anchor)
        self.side = side
        self.box = (w, h)
        self.t_seed = now
        self.hits = 0
        self.misses = 0
        self.why = ""
        return True

    # ── the detector missed: look for the target near the prediction ──────
    def track(self, frame, px, py, radius, now):
        """Return (cx, cy, w, h, score) in frame coordinates, or None."""
        if not self.armed:
            return None
        if now - self.t_seed > self.VIS_MAX_S:
            self.reset()
            self.why = "track: too long without the detector"
            return None
        hit = self._track(frame, px, py, radius)
        if hit is None:
            self.misses += 1
        else:
            self.misses = 0
            self.hits += 1
        return hit

    def _track(self, frame, px, py, radius):
        H, W = frame.shape[:2]
        side = self.side
        r = int(min(radius, self.MAX_RADIUS))
        # 1. scale 1, both templates, the whole window
        x0 = int(round(px)) - r - side // 2
        y0 = int(round(py)) - r - side // 2
        win = _crop(frame, x0, y0, 2 * r + side, 2 * r + side)
        if win is None:
            self.why = "track: window off the frame"
            return None
        if self._r8 is not self._a8:
            ra, rr = _ncc_many(win, (self._a8, self._r8))
            res = np.maximum(ra, rr)
        else:
            ra = res = _ncc(win, self._a8)
        _mn, peak, _ml, (mx, my) = cv2.minMaxLoc(res)
        k = max(2, side // 2)
        saved = res[max(0, my - k):my + k + 1, max(0, mx - k):mx + k + 1].copy()
        res[max(0, my - k):my + k + 1, max(0, mx - k):mx + k + 1] = -1.0
        second = float(res.max())
        res[max(0, my - k):my + k + 1, max(0, mx - k):mx + k + 1] = saved
        a_score = float(ra[my, mx])
        dx, dy = _subpix(res, mx, my, 1), _subpix(res, mx, my, 0)
        cx = x0 + mx + dx + side / 2.0
        cy = y0 + my + dy + side / 2.0
        scale = 1.0
        # 2. other scales, only around that spot (the target grew or shrank)
        for s in self.SCALES:
            ss = int(round(side * s)) | 1
            if ss < 7 or ss > self.MAX_SIDE + 20:
                continue
            m = 4
            sx0, sy0 = int(round(cx)) - ss // 2 - m, int(round(cy)) - ss // 2 - m
            w2 = _crop(frame, sx0, sy0, ss + 2 * m, ss + 2 * m)
            if w2 is None:
                continue
            ta = cv2.resize(self._a8, (ss, ss), interpolation=cv2.INTER_LINEAR)
            ra2 = _ncc(w2, ta)
            if self._r8 is not self._a8:
                res2 = np.maximum(ra2, _ncc(w2, cv2.resize(self._r8, (ss, ss),
                                                            interpolation=cv2.INTER_LINEAR)))
            else:
                res2 = ra2
            _n, p2, _l, (ux, uy) = cv2.minMaxLoc(res2)
            if p2 > peak + 0.02:
                peak, a_score, scale = float(p2), float(ra2[uy, ux]), s
                cx = sx0 + ux + ss / 2.0
                cy = sy0 + uy + ss / 2.0
        if peak < self.SCORE_MIN:
            self.why = f"track: low score {peak:.2f}"
            return None
        if peak - second < self.PEAK_MARGIN:
            self.why = f"track: ambiguous {peak:.2f}/{second:.2f}"
            return None                      # ambiguous: sky texture, not the drone
        if a_score < self.ANCHOR_MIN:
            self.why = f"track: no longer like the detector's picture {a_score:.2f}"
            return None
        if not (0 <= cx < W and 0 <= cy < H):
            self.why = "track: left the picture"
            return None
        patch = _crop(frame, int(round(cx)) - side // 2, int(round(cy)) - side // 2, side, side)
        patch = patch.astype(np.float32)
        if float(patch.std()) < self.FLAT_MIN:
            self.why = "track: flat patch"
            return None
        if _texture(patch) < self.TEX_MIN:
            self.why = "track: no blob at the match (bare sky / cloud)"
            return None
        if peak >= self.ADAPT_MIN and a_score >= 0.5 and scale == 1.0:
            self.recent = (1.0 - self.ADAPT_RATE) * self.recent + self.ADAPT_RATE * patch
            self._r8 = _u8(self.recent)
        self.score = peak
        self.a_score = a_score
        self.why = ""
        w, h = self.box
        return (cx, cy, w * scale, h * scale, peak)

    # ── is a detection over there the target the follower holds? ──────────
    def verify(self, frame, cx, cy):
        """The anchor's best likeness within VERIFY_R px of (cx, cy), -1..1.
        -1 when there is no anchor or no blob there."""
        if self._a8 is None:
            return -1.0
        side, m = self.side, self.VERIFY_R
        x0, y0 = int(round(cx)) - side // 2 - m, int(round(cy)) - side // 2 - m
        win = _crop(frame, x0, y0, side + 2 * m, side + 2 * m)
        if win is None:
            return -1.0
        res = _ncc(win, self._a8)
        _n, best, _l, (ux, uy) = cv2.minMaxLoc(res)
        patch = win[uy:uy + side, ux:ux + side].astype(np.float32)
        if _texture(patch) < self.TEX_MIN:
            return -1.0
        return float(best)


# ── helpers ────────────────────────────────────────────────────────────────
def _crop(img, x0, y0, w, h):
    """img[y0:y0+h, x0:x0+w] as 8-bit grey, the frame's edge repeated where
    the window hangs over it; None when it misses the frame entirely."""
    H, W = img.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xb <= xa or yb <= ya:
        return None
    c = img[ya:yb, xa:xb]
    if c.ndim == 3:
        c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    if (xa, ya, xb, yb) != (x0, y0, x0 + w, y0 + h):
        c = cv2.copyMakeBorder(c, ya - y0, y0 + h - yb, xa - x0, x0 + w - xb,
                               cv2.BORDER_REPLICATE)
    return c


def _u8(tpl):
    return np.clip(tpl + 0.5, 0, 255).astype(np.uint8)


# matchTemplate releases the GIL, so plain threads run the strips in parallel.
# Four workers: the camera, recorder, stream and TensorRT threads keep the
# other two cores.
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="follow")
_SPLIT_MIN = 90 * 90        # result positions below which one call is cheaper


def _match(win, tpl):
    res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    # a flat window divides by ~0: NaN / inf there, never a match
    return np.nan_to_num(res, copy=False, nan=-1.0, posinf=-1.0, neginf=-1.0)


def _strips(win, tpl, n):
    """(window slice, template) jobs whose results, stacked, are the full map."""
    rows = win.shape[0] - tpl.shape[0] + 1
    cuts = np.linspace(0, rows, n + 1).astype(int)
    return [(win[a:b + tpl.shape[0] - 1], tpl) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def _ncc_many(win, tpls):
    """_ncc of one window against several templates, strips in parallel."""
    tpls = list(tpls)
    size = (win.shape[0] - tpls[0].shape[0] + 1) * (win.shape[1] - tpls[0].shape[1] + 1)
    if size < _SPLIT_MIN:
        return [_match(win, t) for t in tpls]
    n = max(1, 4 // len(tpls))
    jobs = [_strips(win, t, n) for t in tpls]
    futs = [[_POOL.submit(_match, w, t) for w, t in js] for js in jobs]
    return [np.vstack([f.result() for f in fs]) for fs in futs]


def _ncc(win, tpl):
    return _ncc_many(win, (tpl,))[0]


def _texture(patch):
    """How much the centre (the target) differs from the patch's border."""
    n = patch.shape[0]
    c = max(2, n // 4)
    mid = patch[n // 2 - c // 2:n // 2 + c // 2 + 1, n // 2 - c // 2:n // 2 + c // 2 + 1]
    border = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])
    return abs(float(mid.mean()) - float(np.median(border))) + 0.5 * float(mid.std())


def _subpix(res, x, y, axis):
    if axis == 1:
        if x <= 0 or x >= res.shape[1] - 1:
            return 0.0
        a, b, c = res[y, x - 1], res[y, x], res[y, x + 1]
    else:
        if y <= 0 or y >= res.shape[0] - 1:
            return 0.0
        a, b, c = res[y - 1, x], res[y, x], res[y + 1, x]
    d = a - 2 * b + c
    return 0.0 if abs(d) < 1e-9 else float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))


def search_radius(missed, speed_px, rot_px, base=40.0):
    """How far from the prediction to look: a floor for the airframe's shake
    and gusts (the field showed 20-100 px jumps), the target's own motion
    over the frames since the last fix, and the airframe's rotation."""
    return float(min(TargetFollower.MAX_RADIUS, base + speed_px * (1.0 + 0.5 * missed) + rot_px))
