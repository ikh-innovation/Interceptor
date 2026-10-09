#!/usr/bin/env python3
import argparse
import contextlib
import io
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

with contextlib.redirect_stdout(io.StringIO()):
    import test_follower as TF

MODEL = os.path.join(os.path.dirname(SRC), "runs/detect/v0/weights/best.engine")
CONF = 0.20
GREEN, ORANGE, GREY = (60, 200, 60), (0, 140, 255), (170, 170, 170)


class Recorded:

    def __init__(self):
        self.d = None

    def __call__(self, frame_full, region, imgsz, enhance):
        if self.d is None:
            return None
        x1, y1, x2, y2, c = self.d
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if not (region[0] <= cx < region[2] and region[1] <= cy < region[3]):
            return None
        return (int(cx), int(cy), int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1)), c)


def detect_all(path):
    from ultralytics import YOLO
    m = YOLO(MODEL, task="detect")
    c = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = c.read()
        if not ok:
            break
        r = m(fr, imgsz=640, max_det=3, conf=0.05, verbose=False)
        ds = []
        if r and len(r[0].boxes):
            for b, cf in zip(r[0].boxes.xyxy.tolist(), r[0].boxes.conf.tolist()):
                ds.append([round(v, 1) for v in b] + [round(cf, 3)])
        out.append(ds)
    return out


def draw(fr, t, detected, fused, stats, title):
    img = fr.copy()
    if fused is not None and t.state != "SEARCH":
        x, y = int(fused[0]), int(fused[1])
        if detected:
            col = ORANGE if t.layer.startswith("FOLLOW") else GREEN
            bw, bh = (t.last_bbox[2], t.last_bbox[3]) if t.last_bbox else (12, 8)
            m = 4
            cv2.rectangle(img, (int(x - bw / 2 - m), int(y - bh / 2 - m)),
                          (int(x + bw / 2 + m), int(y + bh / 2 + m)), col, 2)
        else:
            col = GREY
            cv2.drawMarker(img, (x, y), col, cv2.MARKER_CROSS, 18, 1)
        R = 24
        g = cv2.copyMakeBorder(fr, R, R, R, R, cv2.BORDER_CONSTANT)
        crop = g[y:y + 2 * R, x:x + 2 * R]
        if crop.shape[:2] == (2 * R, 2 * R):
            z = cv2.resize(crop, (8 * R, 8 * R), interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(z, (0, 0), (8 * R - 1, 8 * R - 1), col, 3)
            img[640 - 8 * R - 8:640 - 8, 640 - 8 * R - 8:640 - 8] = z
    cv2.rectangle(img, (0, 0), (640, 34), (30, 30, 30), -1)
    pct = 100.0 * stats[0] / max(1, stats[1])
    cv2.putText(img, f"{title}   tracked {pct:4.0f}%", (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render(path, dets, out_path):
    c = cv2.VideoCapture(path)
    fps = c.get(cv2.CAP_PROP_FPS) or 30.0
    pipe = (f"appsrc ! videoconvert ! video/x-raw,format=I420 ! x264enc speed-preset=veryfast "
            f"bitrate=3000 key-int-max=30 ! video/x-h264,profile=baseline ! mp4mux ! "
            f"filesink location={out_path}")
    w = cv2.VideoWriter(pipe, cv2.CAP_GSTREAMER, 0, fps, (1280, 640), True)
    if not w.isOpened():
        raise SystemExit("cannot open the H.264 writer (gstreamer x264enc)")
    trk = {f: TF.make_tracker(f) for f in (False, True)}
    rec = {f: Recorded() for f in (False, True)}
    for f in trk:
        trk[f]._detect_region = rec[f]
    stats = {False: [0, 0], True: [0, 0]}
    started = False
    for i in range(len(dets)):
        ok, fr = c.read()
        if not ok:
            break
        ds = [d for d in dets[i] if d[4] >= CONF]
        started = started or bool(ds)
        halves = []
        for f in (False, True):
            rec[f].d = ds[0] if ds else None
            trk[f]._frame_idx = i
            fused, _a, _b, _r, det = trk[f]._process(fr, fr)
            if started:
                stats[f][1] += 1
                stats[f][0] += int(bool(det))
            halves.append(draw(fr, trk[f], det, fused, stats[f],
                               "NEW: detector + follower" if f else "TODAY: detector only"))
        w.write(np.hstack(halves))
    w.release()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--dets", help="a JSON cache of detections {video: [[x1,y1,x2,y2,conf],...] per frame}")
    a = ap.parse_args()
    cache = json.load(open(a.dets)) if a.dets and os.path.exists(a.dets) else {}
    out_dir = os.path.join(os.path.dirname(SRC), "replays")
    os.makedirs(out_dir, exist_ok=True)
    for v in a.videos:
        key = next((k for k in cache if os.path.basename(k) == os.path.basename(v)), None)
        dets = cache[key] if key else detect_all(v)
        out = os.path.join(out_dir, os.path.basename(v).replace(".mp4", "_replay.mp4"))
        st = render(v, dets, out)




if __name__ == "__main__":
    main()
