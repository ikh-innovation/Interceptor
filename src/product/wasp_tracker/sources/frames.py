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
    def make_pipeline(cls, cfg):
        props = [
            f"sensor-id={cfg.sensor_id}",
            f"sensor-mode={cfg.sensor_mode}",
            f"tnr-mode={cfg.tnr_mode}",
            f"tnr-strength={cfg.tnr_strength}",
            f"ee-mode={cfg.ee_mode}",
            f"ee-strength={cfg.ee_strength}",
        ]

        if cfg.ae_lock:
            props.append("aelock=true")

        if cfg.awb_lock:
            props.append("awblock=true")

        if cfg.exposure_us > 0:
            props.append(f'exposuretimerange="{cfg.exposure_us} {cfg.exposure_us}"')

        if cfg.gain > 0:
            props.append(f'gainrange="{cfg.gain} {cfg.gain}"')

        if cfg.digital_gain > 0:
            props.append(
                f'ispdigitalgainrange="{cfg.digital_gain} {cfg.digital_gain}"'
            )

        return (
            f"nvarguscamerasrc {' '.join(props)} ! "
            f"video/x-raw(memory:NVMM), "
            f"width={cfg.width}, "
            f"height={cfg.height}, "
            f"framerate={cfg.fps}/1 ! "
            f"nvvidconv flip-method={cfg.flip_method} ! "
            f"video/x-raw, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink "
            f"drop={'1' if cfg.drop else '0'} "
            f"max-buffers={cfg.max_buffers} "
            f"sync={'true' if cfg.sync else 'false'}"
        )

    @classmethod
    def create(cls, config, camera):
        spec = config.path

        # CSI camera
        if spec == "csi":
            pipeline = cls.make_pipeline(camera)
            return cls._camera(
                pipeline,
                config,
                "CSI Camera",
                api=cv2.CAP_GSTREAMER,
            )

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
