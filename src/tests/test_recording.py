#!/usr/bin/env python3
import contextlib, io, os, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["REC_SEGMENT_S"] = "1"
import cv2
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import tracker

FAILED = []


def check(label, ok, detail=""):

    if not ok:
        FAILED.append(label)


d = Path(tempfile.mkdtemp())
base = d / "20260926_120000.mp4"
t = tracker.DroneTracker.__new__(tracker.DroneTracker)
t._rec_fps = 30.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
t._rec_base, t._rec_fourcc, t._rec_part, t._rec_seg_n = base, fourcc, 1, 0
t._writer = cv2.VideoWriter(str(base), fourcc, 30.0, (320, 320))
t._rec_start_thread()
for i in range(100):
    f = np.full((320, 320, 3), i % 255, np.uint8)
    cv2.putText(f, str(i), (40, 160), 0, 2, (255, 255, 255), 3)
    t._rec_submit(f)
    time.sleep(0.004)
time.sleep(1.0)
files = sorted(d.glob("20260926_120000*.mp4"))
names = [p.name for p in files]
counts = {}
for p in files[:-1]:
    c = cv2.VideoCapture(str(p)); counts[p.name] = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release()

check("100 frames at 1 s segments -> 4 files: the base name, then _p02, _p03, _p04",
      names == ["20260926_120000.mp4", "20260926_120000_p02.mp4", "20260926_120000_p03.mp4",
                "20260926_120000_p04.mp4"], str(names))
check("every CLOSED segment is playable after the 'power cut', 30 frames each",
      counts and all(v == 30 for v in counts.values()), str(counts))

if FAILED:

    for f in FAILED:
        pass
    sys.exit(1)
