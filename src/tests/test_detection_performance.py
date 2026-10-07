"""Check optimized search bounds and library initialization without hardware."""
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
from tracker import CameraConfig, DroneTracker


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
            masks.append(masks[-1][:, ::-1])  # non-contiguous view
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
                cv2.setNumThreads(0)  # library initialization resets this global
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
