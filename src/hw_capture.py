"""Camera capture that keeps the full-resolution frame in the Jetson's NVMM memory.

nvarguscamerasrc hands over every frame in NVMM (hardware) memory. One tee branch
scales the preview on the VIC; the other keeps the native frame there, untouched.
Detection crops are cut and scaled from that native frame on the VIC on demand, so
no full-resolution pixel is ever copied to the CPU: 8 MP costs the CPU what the
30 / 60 FPS modes do.
"""

import cv2
import numpy as np

try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstVideo
    Gst.init(None)
except (ImportError, ValueError):
    Gst = GstVideo = None

TAP5, TAP10 = 2, 3
_SLICE_NS = 250_000_000
_STEP = 80          # converter output short sides snap to this
_CONVERTERS = 6     # cached converter pipelines, least recently used dropped


def available():
    return Gst is not None and all(Gst.ElementFactory.find(e) is not None
                                   for e in ("nvarguscamerasrc", "nvvidconv", "appsink", "appsrc"))


def vic_filter(src_w, src_h, dst_w, dst_h):
    # VIC bilinear aliases like cv2 INTER_LINEAR. 5-tap keeps a 2 px target steadiest
    # up to ~3x shrink; past that (8 MP -> 640) only 10-tap stops it flickering.
    return TAP10 if max(src_w / float(dst_w), src_h / float(dst_h)) > 3.0 else TAP5


def even_rect(x1, y1, x2, y2, w, h):
    # NV12 chroma is 2x2: crops start and end on even pixels, inside the frame, >= 2 px.
    x1 = max(0, min(int(x1) & ~1, w - 2))
    y1 = max(0, min(int(y1) & ~1, h - 2))
    x2 = min(w, max(int(x2) + (int(x2) & 1), x1 + 2))
    y2 = min(h, max(int(y2) + (int(y2) & 1), y1 + 2))
    return x1, y1, x2, y2


def bucket(w, h):
    # nvvidconv cannot change its output size mid-stream, so each output size is its own
    # cached converter: the short side snaps to _STEP and a small CPU resize fixes the rest.
    long_, short = max(w, h), min(w, h)
    short = min(long_, max(_STEP, int(round(short / float(_STEP))) * _STEP))
    return (long_, short) if w >= h else (short, long_)


def _bgr(sample):
    info = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
    w, h, stride = info.width, info.height, info.stride[0]
    buf = sample.get_buffer()
    ok, m = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    try:
        rows = np.frombuffer(m.data, np.uint8, count=stride * h).reshape(h, stride)
        return cv2.cvtColor(rows[:, :w * 4].reshape(h, w, 4), cv2.COLOR_BGRA2BGR)
    finally:
        buf.unmap(m)


def _size(sample):
    s = sample.get_caps().get_structure(0)
    return s.get_value("width"), s.get_value("height")


class HwFrame:
    """A camera frame: `disp` is the BGR preview; the native frame stays in NVMM memory."""

    def __init__(self, cap, sample, disp):
        w, h = _size(sample)
        self.shape = (h, w, 3)
        self.disp = disp
        self.pts = sample.get_buffer().pts
        self._cap, self._sample = cap, sample

    def crop(self, x1, y1, x2, y2, out_w, out_h):
        """Native region scaled to out_w x out_h on the VIC: (BGR image, rect used) or (None, None)."""
        return self._cap._crop(self._sample, (x1, y1, x2, y2), out_w, out_h)


class HwCapture:
    """cv2.VideoCapture-shaped reader: read() -> (ok, HwFrame)."""

    def __init__(self, source, native_size, display_size, fps, open_timeout=5.0):
        nw, nh = native_size
        dw, dh = display_size
        self.fps = float(fps)
        self.error = None
        self.unpaired = 0
        self._last = None
        self._convs = {}
        self._pipe = Gst.parse_launch(
            f"{source} ! tee name=t "
            "t. ! queue max-size-buffers=1 leaky=downstream ! "
            f"nvvidconv interpolation-method={vic_filter(nw, nh, dw, dh)} ! "
            f"video/x-raw, format=BGRx, width={dw}, height={dh} ! "
            "appsink name=disp drop=true max-buffers=1 sync=false "
            "t. ! queue max-size-buffers=1 leaky=downstream ! "
            "appsink name=native drop=true max-buffers=1 sync=false")
        self._disp = self._pipe.get_by_name("disp")
        self._native = self._pipe.get_by_name("native")
        self._bus = self._pipe.get_bus()
        self._pipe.set_state(Gst.State.PLAYING)
        ok, _ = self.read(timeout=open_timeout)
        if not ok:
            why = self.error or "no frames"
            self.release()
            raise RuntimeError(f"hardware capture: {why}")

    def isOpened(self):
        return self._pipe is not None

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def read(self, timeout=5.0):
        if self._pipe is None:
            return False, None
        d = self._pull(self._disp, timeout)
        n = self._pull(self._native, timeout) if d is not None else None
        # Both branches carry the same buffers; the preview lags by the VIC scale.
        for _ in range(6):
            if d is None or n is None:
                return False, None
            pd, pn = d.get_buffer().pts, n.get_buffer().pts
            if pd == pn:
                break
            if pd < pn:
                d = self._pull(self._disp, timeout)
            else:
                n = self._pull(self._native, timeout)
        else:
            self.unpaired += 1
        disp = _bgr(d)
        if disp is None:
            self.error = "could not map the preview frame"
            return False, None
        if self._last is not None:
            self._last._sample = None
        self._last = HwFrame(self, n, disp)
        return True, self._last

    def release(self):
        if self._last is not None:
            self._last._sample = None
            self._last = None
        for conv in self._convs.values():
            conv["pipe"].set_state(Gst.State.NULL)
        if self._pipe is not None:
            self._pipe.set_state(Gst.State.NULL)
        self._convs, self._pipe = {}, None

    def _pull(self, sink, timeout):
        waited = 0.0
        while waited < timeout:
            s = sink.emit("try-pull-sample", _SLICE_NS)
            if s is not None:
                return s
            waited += _SLICE_NS / 1e9
            msg = self._bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg is not None:
                self.error = (msg.parse_error()[0].message if msg.type == Gst.MessageType.ERROR
                              else "camera stream ended")
                return None
        self.error = f"no frame in {timeout:.1f} s"
        return None

    def _crop(self, sample, rect, out_w, out_h):
        if sample is None or self._pipe is None:
            return None, None
        w, h = _size(sample)
        x1, y1, x2, y2 = even_rect(*rect, w, h)
        out_w, out_h = max(2, int(out_w)), max(2, int(out_h))
        bw, bh = bucket(out_w, out_h)
        conv = self._converter(sample.get_caps(), bw, bh)
        el = conv["conv"]
        for k, v in (("left", x1), ("top", y1), ("right", x2), ("bottom", y2)):
            el.set_property(k, v)
        el.set_property("interpolation-method", vic_filter(x2 - x1, y2 - y1, bw, bh))
        if conv["src"].emit("push-buffer", sample.get_buffer()) != Gst.FlowReturn.OK:
            return None, None
        out = conv["sink"].emit("try-pull-sample", 4 * _SLICE_NS)
        img = _bgr(out) if out is not None else None
        if img is None:
            return None, None
        if (bw, bh) != (out_w, out_h):
            img = cv2.resize(img, (out_w, out_h), interpolation=(
                cv2.INTER_AREA if out_w * out_h < bw * bh else cv2.INTER_LINEAR))
        return img, (x1, y1, x2, y2)

    def _converter(self, caps, bw, bh):
        conv = self._convs.pop((bw, bh), None)
        if conv is not None and not conv["caps"].is_equal(caps):
            conv["pipe"].set_state(Gst.State.NULL)
            conv = None
        if conv is None:
            if len(self._convs) >= _CONVERTERS:
                oldest = next(iter(self._convs))
                self._convs.pop(oldest)["pipe"].set_state(Gst.State.NULL)
            pipe = Gst.parse_launch(
                "appsrc name=src format=time ! nvvidconv name=conv output-buffers=2 ! "
                f"video/x-raw, format=BGRx, width={bw}, height={bh} ! "
                "appsink name=out drop=false max-buffers=1 sync=false")
            conv = dict(pipe=pipe, caps=caps, src=pipe.get_by_name("src"),
                        conv=pipe.get_by_name("conv"), sink=pipe.get_by_name("out"))
            conv["src"].set_property("caps", caps)
            pipe.set_state(Gst.State.PLAYING)
        self._convs[(bw, bh)] = conv
        return conv
