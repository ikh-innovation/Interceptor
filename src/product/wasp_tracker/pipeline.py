from time import perf_counter

import cv2
import numpy as np
from ultralytics import YOLO

from .context import FrameContext
from .control.ardupilot import ArduPilotCommander
from .detection.detector import SkyGatedDetector
from .detection.planner import DetectionPlanner
from .detection.sky import SkyDetector
from .sources.frames import FrameSourceFactory
from .telemetry import FrameRecord, LoggerFactory
from .tracking.state import TargetTracker
from .visualization.renderer import RendererFactory


class TrackingPipeline:
    def __init__(self, config):
        self.config = config
        self.model = YOLO(config.model.path)
        self.sky = SkyDetector(config.sky)
        self.detector = SkyGatedDetector(self.model, self.sky, config.detection,
                                         config.model.conf_thresh)
        self.planner = DetectionPlanner(self.detector, config.detection)
        self.tracker = TargetTracker(config.tracking, config.detection.big_obj_frac)
        self.commander = ArduPilotCommander(config.control)
        self.source = FrameSourceFactory.create(config.source)
        self.renderer = RendererFactory.create(config.visualization, self.commander,
                                               config.source.skip_frames)
        self.logger = LoggerFactory.create(config.logging)

        self.display_size = (config.source.display_width, config.source.display_height)
        self.period = config.source.skip_frames + 1
        self.max_frames = config.source.max_frames
        self._last_cmd = ArduPilotCommander.zero_command()
        self._frame_idx = 0

    def run(self):
        self.commander.connect()
        if self.config.detection.warmup:
            self.detector.warmup(self.config.detection.imgsz_search)
        self.renderer.open(self.display_size)
        self.logger.start(dict(
            source=self.config.source.path,
            model=self.config.model.path,
            display_w=self.display_size[0],
            display_h=self.display_size[1],
            skip_frames=self.config.source.skip_frames,
        ))
        try:
            for frame_full in self.source.frames():
                if self._step(frame_full):
                    break
                if self.max_frames is not None and self._frame_idx >= self.max_frames:
                    break
        finally:
            self.commander.close()
            self.source.release()
            self.renderer.close()
            self.logger.summary()

    def _step(self, frame_full):
        frame_full = np.ascontiguousarray(frame_full)
        full_h, full_w = frame_full.shape[:2]
        frame_disp = np.ascontiguousarray(cv2.resize(frame_full, self.display_size))
        disp_w, disp_h = self.display_size
        cx_f, cy_f = disp_w // 2, disp_h // 2

        self._frame_idx += 1
        is_yolo = (self._frame_idx % self.period == 0)

        sky_mask = self.sky.mask(frame_disp)
        ctx = FrameContext(frame_full, frame_disp, full_w, full_h, disp_w, disp_h, sky_mask)

        if is_yolo:
            fused, t_kf, t_yolo, roi, detected = self._detect(ctx)
        else:
            t0 = perf_counter()
            fused = self.tracker.predict()
            t_kf = (perf_counter() - t0) * 1000
            t_yolo, roi, detected = 0.0, None, False
            self.tracker.mark_skip(fused is not None)

        cmd = self._command(fused, cx_f, cy_f, disp_w, disp_h, is_yolo)

        self.renderer.draw(ctx, self.tracker, fused, roi, cmd, is_yolo)
        self.logger.frame(FrameRecord(
            frame_idx=self._frame_idx, is_yolo=is_yolo, state=self.tracker.state,
            layer=self.tracker.layer, conf=self.tracker.conf, t_kf_ms=t_kf,
            t_yolo_ms=t_yolo, pos=fused, cmd=cmd, sent=is_yolo, detected=detected))
        return self.renderer.present(frame_disp)

    def _detect(self, ctx):
        self.detector.reset_frame()
        t0 = perf_counter()
        kf_pred = self.tracker.predict()
        t_kf = (perf_counter() - t0) * 1000

        small = self.tracker.target_is_small(ctx.disp_h)
        t0 = perf_counter()
        det, roi = self.planner.plan(ctx, self.tracker, kf_pred, small)
        t_yolo = (perf_counter() - t0) * 1000

        fused = self.tracker.on_result(det, kf_pred, small, self.detector.ground_rejected)
        return fused, t_kf, t_yolo, roi, (det is not None)

    def _command(self, fused, cx_f, cy_f, disp_w, disp_h, is_yolo):
        if not is_yolo:
            return self._last_cmd
        if fused:
            cmd = self.commander.compute(fused[0] - cx_f, fused[1] - cy_f,
                                         self.tracker.state, disp_w, disp_h)
        else:
            cmd = ArduPilotCommander.zero_command()
        self.commander.send(cmd)
        self._last_cmd = cmd
        return cmd
