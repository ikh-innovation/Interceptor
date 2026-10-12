
import contextlib
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from tracker import CameraConfig, DroneTracker, shrink_to


class _Val:
    def __init__(self, v):
        self.v = v
    def cpu(self):
        return self
    def numpy(self):
        return np.asarray(self.v, np.float32)
    def tolist(self):
        return list(self.v)


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy, self.conf = [_Val(xyxy)], _Val([conf])
    def __len__(self):
        return 1


class _Result:
    def __init__(self, xyxy, conf=0.9):
        self.boxes = _Boxes(xyxy, conf)


class DetectionInputTests(unittest.TestCase):
    def tracker(self, width, height, fw, fh, seen):
        t = DroneTracker.__new__(DroneTracker)
        t.full_w, t.full_h, t.fw, t.fh = width, height, fw, fh
        t.sky_mask, t.target_class, t.conf_thresh = None, 0, 0.25
        def model(img, imgsz, **_):
            seen.append(img)
            h, w = img.shape[:2]
            return [_Result((w * 0.25, h * 0.5, w * 0.25 + 8, h * 0.5 + 8))]
        t.model = model
        return t

    def test_big_regions_reach_the_model_already_shrunk_and_map_back_in_every_mode(self):
        # 8 MP / 21 FPS, 30 FPS and 60 FPS frames, each into its own preview size.
        for width, height, fw, fh in ((3280, 2464, 640, 640), (1280, 960, 640, 640),
                                      (1280, 720, 500, 374)):
            seen = []
            t = self.tracker(width, height, fw, fh, seen)
            frame = np.zeros((height, width, 3), np.uint8)
            det = t._detect_region(frame, (0, 0, width, height), 640, False)
            self.assertEqual(max(seen[-1].shape[:2]), 640, (width, height))
            self.assertLessEqual(abs(det[2] - fw * 0.25), 1, (width, height, det))
            self.assertLessEqual(abs(det[3] - fh * 0.5), 1, (width, height, det))

    def test_small_regions_reach_the_model_untouched(self):
        seen = []
        t = self.tracker(3280, 2464, 640, 640, seen)
        frame = np.zeros((2464, 3280, 3), np.uint8)
        det = t._detect_region(frame, (1000, 800, 1400, 1100), 640, False)
        self.assertEqual(seen[-1].shape[:2], (300, 400))
        self.assertTrue(np.shares_memory(seen[-1], frame))
        self.assertEqual(det[2:4], (int(1100 * 640 / 3280), int(950 * 640 / 2464)))

    def test_shrink_keeps_a_small_target_wherever_it_lands(self):
        # INTER_LINEAR alone lets a 2 px target vanish at some offsets; the
        # shrink must keep its strength, so the detector sees it every frame.
        for width, height in ((3280, 2464), (3280, 1400), (1100, 800), (1280, 720)):
            r = 640 / max(width, height)
            w, h = round(width * r), round(height * r)
            mass = []
            for off in range(8):
                img = np.zeros((height, width), np.float32)
                img[height // 2:height // 2 + 2, width // 2 + off:width // 2 + off + 2] = 200
                mass.append(float(shrink_to(img, w, h).sum()))
            self.assertLess(max(mass) - min(mass), 0.05 * max(mass), (width, height, mass))


class SearchBoundsTests(unittest.TestCase):
    def test_search_bounds_preserve_inclusive_pixels_and_padding(self):
        rng = np.random.default_rng(11)
        tracker = DroneTracker.__new__(DroneTracker)
        tracker.use_sky_mask = True
        for fw, fh, width, height in ((640, 640, 1280, 960), (500, 374, 1280, 720)):
            tracker.fw, tracker.fh = fw, fh
            tracker.full_w, tracker.full_h = width, height
            masks = [np.zeros((fh, fw), np.uint8), np.full((fh, fw), 255, np.uint8)]
            for x, y in ((0, 0), (fw-1, fh-1), (fw//2, fh//2)):
                mask = np.zeros((fh, fw), np.uint8)
                mask[y, x] = 255
                masks.append(mask)
            masks += [(rng.random((fh, fw)) < fraction).astype(np.uint8)*255
                      for fraction in (.001, .1, .9)]
            masks.append(masks[-1][:, ::-1])
            for mask in masks:
                tracker.sky_mask = mask
                ys, xs = np.where(mask > 0)
                expected = None if not xs.size else (
                    int(xs.min()*width/fw)-tracker.CROP_PAD,
                    int(ys.min()*height/fh)-tracker.CROP_PAD,
                    int(xs.max()*width/fw)+tracker.CROP_PAD,
                    int(ys.max()*height/fh)+tracker.CROP_PAD)
                self.assertEqual(tracker._sky_bbox_native(), expected)
        tracker.sky_mask = None
        self.assertIsNone(tracker._sky_bbox_native())
        tracker.use_sky_mask = False
        self.assertEqual(tracker._sky_bbox_native(), (0, 0, width, height))

    def test_configured_threads_survive_detector_initialization(self):
        class Model:
            @property
            def names(self):
                cv2.setNumThreads(0)
                return {0: 'target'}

        class AP:
            mission = None
            def neutral_command(self):
                return {}

        original = cv2.getNumThreads()
        try:
            with patch('ultralytics.YOLO', return_value=Model()), \
                 patch.dict(os.environ, {'CV_THREADS': '3', 'TARGET_CLASS': '0', 'CAM_MODE': '30'}), \
                 contextlib.redirect_stdout(io.StringIO()):
                DroneTracker(model_path='test.engine', camera_config=CameraConfig(),
                             commander=AP(), headless=True)
            self.assertEqual(cv2.getNumThreads(), 3)
        finally:
            cv2.setNumThreads(original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
