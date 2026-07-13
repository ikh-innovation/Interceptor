import cv2


class FrameSource:
    def __init__(self, capture, label):
        self.capture = capture
        self.label = label
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open source: {label}")

    def frames(self):
        while self.capture.isOpened():
            ok, frame = self.capture.read()
            if not ok or frame is None:
                break
            yield frame

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class FrameSourceFactory:
    _GST_HINTS = ("!", "appsink", "nvarguscamerasrc", "v4l2src")
    _STREAM_PREFIXES = ("rtsp://", "http://", "https://", "udp://", "tcp://")

    @classmethod
    def create(cls, config):
        spec = config.path

        if isinstance(spec, int):
            return cls._camera(spec, config, str(spec))

        text = str(spec)
        if text.isdigit():
            return cls._camera(int(text), config, text)
        if any(hint in text for hint in cls._GST_HINTS):
            return cls._camera(text, config, text, api=cv2.CAP_GSTREAMER)
        if text.startswith("/dev/video") or text.startswith(cls._STREAM_PREFIXES):
            return cls._camera(text, config, text)
        return FrameSource(cv2.VideoCapture(text), text)

    @classmethod
    def _camera(cls, target, config, label, api=None):
        capture = cv2.VideoCapture(target, api) if api is not None else cv2.VideoCapture(target)
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, config.camera_buffer_size)
        except cv2.error:
            pass
        return FrameSource(capture, label)
