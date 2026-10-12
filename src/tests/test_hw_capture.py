#!/usr/bin/env python3
"""VIC capture: preview and detection crops come from the native NVMM frame, never a CPU copy.

The hardware tests need a Jetson (nvvidconv); they feed NVMM frames from a test
source, the same memory nvarguscamerasrc delivers, so the camera stays free.
"""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import hw_capture as hc
with contextlib.redirect_stdout(io.StringIO()):
    import tracker as T

HW = hc.available()


def nvmm_source(w, h, pattern="pinwheel"):
    return (f"videotestsrc is-live=true pattern={pattern} ! "
            f"video/x-raw,format=NV12,width={w},height={h},framerate=21/1 ! "
            "nvvidconv ! video/x-raw(memory:NVMM),format=NV12")


def nvmm_sample(gray):
    """One NVMM frame holding this grayscale picture."""
    h, w = gray.shape
    p = hc.Gst.parse_launch(
        f"appsrc name=a format=time caps=\"video/x-raw,format=BGRx,width={w},height={h},framerate=21/1\" ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! appsink name=o sync=false")
    p.set_state(hc.Gst.State.PLAYING)
    p.get_by_name("a").emit("push-buffer", hc.Gst.Buffer.new_wrapped(
        np.dstack([gray, gray, gray, np.zeros_like(gray)]).tobytes()))
    return p, p.get_by_name("o").emit("try-pull-sample", 2 * hc.Gst.SECOND)


def converter():
    cap = hc.HwCapture.__new__(hc.HwCapture)
    cap._pipe, cap._convs, cap._last = Mock(), {}, None
    return cap


class PureTests(unittest.TestCase):
    def test_filter_follows_the_shrink(self):
        self.assertEqual(hc.vic_filter(3280, 2464, 640, 480), hc.TAP10)
        self.assertEqual(hc.vic_filter(3280, 2464, 640, 640), hc.TAP10)
        self.assertEqual(hc.vic_filter(1280, 960, 640, 640), hc.TAP5)
        self.assertEqual(hc.vic_filter(1280, 720, 500, 374), hc.TAP5)
        self.assertEqual(hc.vic_filter(200, 150, 200, 150), hc.TAP5)

    def test_output_sizes_snap_to_few_converters(self):
        self.assertEqual(hc.bucket(640, 481), (640, 480))
        self.assertEqual(hc.bucket(640, 427), (640, 400))
        self.assertEqual(hc.bucket(640, 20), (640, 80))
        self.assertEqual(hc.bucket(300, 640), (320, 640))
        self.assertEqual(hc.bucket(640, 640), (640, 640))

    def test_crops_are_even_inside_and_never_empty(self):
        self.assertEqual(hc.even_rect(427, 321, 827, 621, 1280, 960), (426, 320, 828, 622))
        self.assertEqual(hc.even_rect(-5, -5, 5000, 5000, 3280, 2464), (0, 0, 3280, 2464))
        self.assertEqual(hc.even_rect(3279, 2463, 3280, 2464, 3280, 2464), (3278, 2462, 3280, 2464))
        self.assertEqual(hc.even_rect(10, 10, 10, 10, 640, 480), (10, 10, 12, 12))


@unittest.skipUnless(HW, "needs a Jetson with nvvidconv")
class HardwareTests(unittest.TestCase):
    def test_every_mode_reads_paired_preview_and_native_frames(self):
        # 30 FPS, 60 FPS and 8 MP / 21 FPS frame and preview sizes.
        for (w, h), disp in (((1280, 960), (640, 640)), ((1280, 720), (500, 374)),
                             ((3280, 2464), (640, 640))):
            cap = hc.HwCapture(nvmm_source(w, h, "ball"), (w, h), disp, 21)
            try:
                for _ in range(3):
                    ok, f = cap.read()
                    self.assertTrue(ok, cap.error)
                    self.assertEqual(f.shape, (h, w, 3))
                    self.assertEqual(f.disp.shape, (disp[1], disp[0], 3))
                self.assertEqual(cap.unpaired, 0)
                self.assertTrue(cap.isOpened())
                self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 21.0)
            finally:
                cap.release()
            self.assertFalse(cap.isOpened())

    def test_crops_land_where_asked(self):
        w, h = 3280, 2464
        rng = np.random.default_rng(3)
        tex = cv2.resize(rng.integers(0, 255, (h // 8, w // 8), np.uint8), (w, h),
                         interpolation=cv2.INTER_CUBIC)
        keep, s = nvmm_sample(tex)
        cap = converter()
        try:
            full, used = cap._crop(s, (0, 0, w, h), w, h)
            self.assertEqual(used, (0, 0, w, h))
            ref = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY).astype(np.float32)
            for rect, size in (((820, 616, 1914, 1438), (640, 481)), ((w - 512, h - 384, w, h), (512, 384)),
                               ((0, 0, 1094, 820), (640, 480)), ((0, 0, w, h), (640, 481))):
                img, used = cap._crop(s, rect, *size)
                self.assertEqual(used, rect)
                self.assertEqual(img.shape, (size[1], size[0], 3))
                want = cv2.resize(ref[rect[1]:rect[3], rect[0]:rect[2]], size, interpolation=cv2.INTER_AREA)
                (dx, dy), _ = cv2.phaseCorrelate(want, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32))
                self.assertLess(max(abs(dx), abs(dy)), 0.5, (rect, dx, dy))
        finally:
            keep.set_state(hc.Gst.State.NULL)
            cap.release()

    def test_a_small_target_stays_visible_wherever_it_lands(self):
        # cv2 INTER_LINEAR loses a 2 px target completely at some offsets (8 MP) or
        # swings it 56 % (60 FPS preview); the VIC filter must keep it steady.
        cap = converter()
        try:
            for (w, h), size in (((3280, 2464), (640, 481)), ((3280, 2464), (640, 640)),
                                 ((1280, 720), (500, 374)), ((1280, 960), (640, 640))):
                mass = []
                for off in range(6):
                    img = np.full((h, w), 40, np.uint8)
                    img[h // 2:h // 2 + 2, w // 2 + off:w // 2 + off + 2] = 240
                    keep, s = nvmm_sample(img)
                    out, _ = cap._crop(s, (0, 0, w, h), *size)
                    keep.set_state(hc.Gst.State.NULL)
                    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    mass.append(float((g - np.median(g)).clip(0).sum()))
                self.assertGreater(min(mass), 0.0, ((w, h), size, mass))
                self.assertLess((max(mass) - min(mass)) / max(mass), 0.25, ((w, h), size, mass))
        finally:
            cap.release()


class _Frame:
    """An NVMM-style frame: crop() even-aligns the rect like the VIC path does."""

    def __init__(self, w, h):
        self.shape, self.calls = (h, w, 3), []

    def crop(self, x1, y1, x2, y2, ow, oh):
        self.calls.append(((x1, y1, x2, y2), (ow, oh)))
        return np.zeros((oh, ow, 3), np.uint8), hc.even_rect(x1, y1, x2, y2, self.shape[1], self.shape[0])


class _Boxes:
    def __init__(self, box):
        class V:
            def __init__(s, v): s.v = v
            def cpu(s): return s
            def numpy(s): return np.asarray(s.v, np.float32)
            def tolist(s): return list(s.v)
        self.xyxy, self.conf = [V(box)], V([0.9])

    def __len__(self):
        return 1


class TrackerTests(unittest.TestCase):
    def tracker(self, w, h, fw, fh):
        t = T.DroneTracker.__new__(T.DroneTracker)
        t.full_w, t.full_h, t.fw, t.fh = w, h, fw, fh
        t.sky_mask, t.target_class, t.conf_thresh = None, 0, 0.25
        return t

    def test_nvmm_frames_are_cropped_on_the_vic_and_map_back(self):
        for (w, h), (fw, fh) in (((3280, 2464), (640, 640)), ((1280, 960), (640, 640)), ((1280, 720), (500, 374))):
            t = self.tracker(w, h, fw, fh)
            seen = []
            def model(img, imgsz, **_):
                seen.append(img.shape)
                ih, iw = img.shape[:2]
                return [type("R", (), {"boxes": _Boxes((iw * 0.25, ih * 0.5, iw * 0.25 + 8, ih * 0.5 + 8))})()]
            t.model = model
            f = _Frame(w, h)
            det = t._detect_region(f, (0, 0, w, h), 640, False)
            self.assertEqual(f.calls[-1][1][0], 640, (w, h))
            self.assertLessEqual(abs(det[2] - fw * 0.25), 1, (w, h, det))
            self.assertLessEqual(abs(det[3] - fh * 0.5), 1, (w, h, det))
            # A lock-on crop smaller than the model input is scaled up on the VIC, as YOLO would.
            f = _Frame(w, h)
            det = t._detect_region(f, (101, 101, 401, 301), 640, False)
            self.assertEqual(f.calls[-1], ((101, 101, 401, 301), (640, 427)))
            self.assertLessEqual(abs(det[2] - (100 + 300 * 0.25) * fw / w), 1, (w, h, det))

    def cam(self, mode, captures):
        t = T.DroneTracker.__new__(T.DroneTracker)
        t.camera_config, t.ref_display, t.ap = T.CameraConfig(), (640, 640), None
        t.CAM_OPEN_RETRIES, t.CAM_WARMUP_READS, t.CAM_RETRY_WAIT_S = 2, 2, 0.0
        t._apply_cam_mode(mode)
        return t

    def test_the_vic_path_is_used_and_checked_against_the_mode(self):
        t = self.cam(21, None)
        good = Mock(); good.read.return_value = (True, _Frame(3280, 2464))
        with patch.object(T.hw_capture, "available", return_value=True), \
             patch.object(T.hw_capture, "HwCapture", return_value=good) as hw:
            cap, fps, _ = t._open_capture()
        self.assertIs(cap, good)
        self.assertEqual((fps, t.cam_capture), (21.0, "VIC"))
        self.assertIn("sensor-mode=0", hw.call_args.args[0])
        self.assertNotIn("videoconvert", hw.call_args.args[0])
        self.assertEqual(hw.call_args.args[1:3], ((3280, 2464), (640, 640)))
        # A camera that ignores the mode is a failed switch, not a silent 30 FPS picture.
        wrong = Mock(); wrong.read.return_value = (True, _Frame(1280, 960))
        with patch.object(T.hw_capture, "available", return_value=True), \
             patch.object(T.hw_capture, "HwCapture", return_value=wrong):
            with self.assertRaisesRegex(RuntimeError, "delivered 1280x960, mode 21 needs 3280x2464"):
                t._open_capture()
        wrong.release.assert_called_once()

    def test_the_cpu_pipeline_takes_over_when_the_vic_path_fails(self):
        t = self.cam(30, None)
        cpu = Mock(); cpu.isOpened.return_value = True
        cpu.read.return_value = (True, np.zeros((960, 1280, 3), np.uint8)); cpu.get.return_value = 30.0
        with patch.object(T.hw_capture, "available", return_value=True), \
             patch.object(T.hw_capture, "HwCapture", side_effect=RuntimeError("hardware capture: boom")), \
             patch.object(T.cv2, "VideoCapture", return_value=cpu):
            cap, fps, _ = t._open_capture()
        self.assertIs(cap, cpu)
        self.assertTrue(t.cam_capture.startswith("CPU (VIC failed"), t.cam_capture)
        with patch.dict("os.environ", {"HW_CAPTURE": "0"}), \
             patch.object(T.hw_capture, "HwCapture") as hw, \
             patch.object(T.cv2, "VideoCapture", return_value=cpu):
            t._open_capture()
        hw.assert_not_called()
        self.assertEqual(t.cam_capture, "CPU")

    def test_a_failed_switch_reverts_and_says_why(self):
        t = self.cam(30, None)
        t._cam_req, t._cam_busy, t._cap = 21, False, Mock()
        t._camera_locked = lambda: ""
        t._reset = lambda: None
        t.ap = Mock(law=None)
        opened = []
        def fake_open():
            opened.append(t.cam_mode)
            if t.cam_mode == 21:
                raise RuntimeError("camera delivered 1280x960, mode 21 needs 3280x2464")
            return Mock(), 30.0, "camera"
        t._open_capture = fake_open
        t._switch_camera([])
        self.assertEqual(opened, [21, 30])
        self.assertEqual(t.cam_mode, 30)
        self.assertIn("21 failed, back to 30", t.cam_error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
