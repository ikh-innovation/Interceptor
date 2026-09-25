#!/usr/bin/env python3
"""The sky gate (SkyDetector) at half resolution must decide like full resolution.

Ground rejection stays the SAME module; SKY_MASK_SCALE=0.5 only computes its
mask on a half-size frame (21 -> ~7 ms on this Jetson).  Checked on recorded
field frames when any are readable, and always on synthetic sky/ground scenes.

    ../.env/yolo/bin/python tests/test_sky_mask.py
"""
import contextlib, glob, io, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import tracker

FAILED = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def synthetic(seed):
    rng = np.random.default_rng(seed)
    img = np.zeros((640, 640, 3), np.uint8)
    img[:] = (235, 200, 150)                          # sky, BGR
    hz = int(rng.integers(300, 520))
    img[hz:] = (40, 90, 60)                           # ground from the bottom
    for _ in range(6):                                # trees rising from the ground
        x = int(rng.integers(0, 600)); w = int(rng.integers(20, 60)); top = int(rng.integers(150, hz))
        img[top:hz, x:x + w] = (30, 70, 40)
    for _ in range(4):                                # dark blobs floating in the sky
        cv2.circle(img, (int(rng.integers(40, 600)), int(rng.integers(20, 200))), int(rng.integers(4, 14)), (60, 60, 60), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


frames = [synthetic(s) for s in range(12)]
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for v in sorted(glob.glob(os.path.join(here, "data", "*.mp4")))[-8:]:
    cap = cv2.VideoCapture(v)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for k in range(3 if n > 0 else 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / 3))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(f, (640, 640)))
full, half = tracker.SkyDetector(), tracker.SkyDetector(scale=0.5)
t = time.perf_counter(); A = [full.mask(f) for f in frames]; ta = (time.perf_counter() - t) / len(frames) * 1000
t = time.perf_counter(); B = [half.mask(f) for f in frames]; tb = (time.perf_counter() - t) / len(frames) * 1000
agree = [np.mean((a > 0) == (b > 0)) for a, b in zip(A, B)]
rng = np.random.default_rng(1); dis = tot = 0
for a, b in zip(A, B):
    for _ in range(60):
        w = int(rng.integers(8, 100)); x = int(rng.integers(0, 640 - w)); y = int(rng.integers(0, 640 - w))
        tot += 1; dis += full.background_is_sky(a, (x, y, w, w)) != full.background_is_sky(b, (x, y, w, w))
print(f"  {len(frames)} frames: full {ta:.1f} ms, half {tb:.1f} ms")
check("half resolution is at least 2x faster", tb * 2 <= ta, f"{ta:.1f} -> {tb:.1f} ms")
check("pixel agreement: mean >= 99%, worst frame >= 97%", np.mean(agree) >= 0.99 and min(agree) >= 0.97,
      f"mean {np.mean(agree):.4f}, worst {min(agree):.4f}")
check("sky-gate verdict differs on <= 1% of boxes", dis <= 0.01 * tot, f"{dis}/{tot}")
check("the shipped tracker builds the gate at 0.5 by default",
      'os.environ.get("SKY_MASK_SCALE", "0.5")' in open(os.path.join(here, "tracker.py")).read())
print()
if FAILED:
    print(f"{len(FAILED)} FAILED:")
    for f in FAILED:
        print("  - " + f)
    sys.exit(1)
print("all passed")
