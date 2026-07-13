import logging
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

_LOGGER_NAME = "wasp_tracker"


@dataclass
class FrameRecord:
    frame_idx: int
    is_yolo: bool
    state: str
    layer: str
    conf: Optional[float]
    t_kf_ms: float
    t_yolo_ms: float
    pos: Optional[tuple]
    cmd: dict
    sent: bool
    detected: bool


class TrackerLogger:
    def start(self, meta):
        raise NotImplementedError

    def frame(self, record):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError


class NullTrackerLogger(TrackerLogger):
    def start(self, meta):
        pass

    def frame(self, record):
        pass

    def summary(self):
        pass


class ConsoleTrackerLogger(TrackerLogger):
    def __init__(self, per_frame, level):
        self.per_frame = per_frame
        self._log = _stdout_logger(level)
        self._meta = {}
        self._t0 = perf_counter()
        self._frames = 0
        self._yolo = 0
        self._detected = 0
        self._sent = 0
        self._states = {}
        self._yolo_ms = [0, 0.0, 0.0, 0.0, 0.0]

    def start(self, meta):
        self._meta = meta
        self._t0 = perf_counter()
        self._frames = self._yolo = self._detected = self._sent = 0
        self._states = {}
        self._yolo_ms = [0, 0.0, 0.0, 0.0, 0.0]
        self._log.info(
            "stream start | source=%s model=%s display=%sx%s skip=%s",
            meta.get("source"), meta.get("model"),
            meta.get("display_w"), meta.get("display_h"), meta.get("skip_frames"))

    def frame(self, record):
        self._frames += 1
        self._states[record.state] = self._states.get(record.state, 0) + 1
        if record.sent:
            self._sent += 1
        if record.is_yolo:
            self._yolo += 1
            _welford(self._yolo_ms, record.t_yolo_ms)
            if record.detected:
                self._detected += 1
        if self.per_frame:
            self._log.info(self._format(record))

    def summary(self):
        elapsed = max(1e-6, perf_counter() - self._t0)
        fps = self._frames / elapsed
        det_rate = (self._detected / self._yolo * 100.0) if self._yolo else 0.0
        count, mean, _, lo, hi = self._yolo_ms
        std = _std(self._yolo_ms)
        states = "  ".join(f"{k}:{v}" for k, v in sorted(self._states.items()))
        lines = [
            "──────── WASP TRACK SUMMARY ────────",
            f"source           : {self._meta.get('source')}",
            f"frames processed : {self._frames}",
            f"detect frames    : {self._yolo}  (skip_frames={self._meta.get('skip_frames')})",
            f"detections       : {self._detected}  (rate {det_rate:.1f}%)",
            f"setpoints sent   : {self._sent}",
            f"wall time        : {elapsed:.2f} s",
            f"throughput       : {fps:.1f} fps",
            f"yolo latency ms  : mean {mean:.1f}  min {lo:.1f}  max {hi:.1f}  std {std:.1f}  (n={count})",
            f"state histogram  : {states}",
            "────────────────────────────────────",
        ]
        self._log.info("\n".join(lines))

    def _format(self, record):
        tag = "YOLO" if record.is_yolo else "SKIP"
        pos = f"{record.pos[0]:>4},{record.pos[1]:>4}" if record.pos else "----,----"
        cmd = record.cmd
        tx = "TX" if record.sent else "HOLD"
        return (
            f"f{record.frame_idx:06d} {tag} {record.state:<7} {record.layer:<14} "
            f"yolo={record.t_yolo_ms:6.1f}ms kf={record.t_kf_ms:.3f}ms "
            f"pos={pos} "
            f"cmd[fwd{cmd['vx']:+.2f} lat{cmd['vy']:+.2f} vz{cmd['vz']:+.2f} yaw{cmd['yaw_rate']:+.3f}] {tx}"
        )


class LoggerFactory:
    @staticmethod
    def create(config):
        if not config.enabled:
            return NullTrackerLogger()
        return ConsoleTrackerLogger(config.per_frame, config.level)


def _stdout_logger(level):
    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d WASP %(message)s",
                                            datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    return logger


def _welford(agg, x):
    agg[0] += 1
    delta = x - agg[1]
    agg[1] += delta / agg[0]
    agg[2] += delta * (x - agg[1])
    agg[3] = x if agg[0] == 1 else min(agg[3], x)
    agg[4] = x if agg[0] == 1 else max(agg[4], x)


def _std(agg):
    return (agg[2] / agg[0]) ** 0.5 if agg[0] > 0 else 0.0
