from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml


@dataclass
class ModelConfig:
    path: str = "./runs/detect/v0/weights/best.pt"
    conf_thresh: float = 0.01


@dataclass
class SourceConfig:
    path: Union[str, int] = "./data/videos_drones/field.MP4"
    display_width: int = 640
    display_height: int = 640
    skip_frames: int = 0
    max_frames: Optional[int] = None
    camera_buffer_size: int = 1


@dataclass
class DetectionConfig:
    imgsz_search: int = 1280
    imgsz_track: int = 640
    imgsz_close: int = 416
    crop_pad: int = 24
    big_obj_frac: float = 0.06
    use_clahe: bool = False
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    min_roi_half: int = 50
    warmup: bool = True


@dataclass
class SkyConfig:
    show: bool = True
    v_min: int = 150
    s_max: int = 60
    h_lo: int = 90
    h_hi: int = 140
    ring_pad_frac: float = 0.6
    ring_pad_min: int = 12
    sky_frac: float = 0.60
    top_band: int = 3
    bottom_band: int = 3
    ground_min_frac: float = 0.02


@dataclass
class TrackingConfig:
    miss_to_predict: int = 2
    miss_to_search: int = 20


@dataclass
class ControlConfig:
    mav_device: Optional[str] = None
    mav_baud: int = 115200
    h_fov_deg: float = 90.0
    v_fov_deg: float = 90.0
    k_yaw: float = 1.5
    k_vz: float = 2.0
    fwd_lock: float = 2.0
    fwd_predict: float = 0.5
    fwd_search: float = 0.0
    max_yaw: float = 1.5
    max_vz: float = 2.0
    max_fwd: float = 3.0


@dataclass
class VisualizationConfig:
    enabled: bool = True
    show_window: bool = True
    show_sky: bool = True
    show_inset: bool = True


@dataclass
class LoggingConfig:
    enabled: bool = True
    per_frame: bool = True
    level: str = "INFO"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    sky: SkyConfig = field(default_factory=SkyConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path=None, overrides=None):
        raw = {}
        if path is not None:
            resolved = Path(path)
            if resolved.exists():
                loaded = yaml.safe_load(resolved.read_text()) or {}
                if not isinstance(loaded, dict):
                    raise ValueError(f"config root must be a mapping: {resolved}")
                raw = loaded
        cfg = _build(cls, raw)
        if overrides:
            _apply_overrides(cfg, overrides)
        return cfg


def _build(dataclass_type, values):
    kwargs = {}
    for f in fields(dataclass_type):
        if f.name not in values:
            continue
        given = values[f.name]
        if is_dataclass(f.type) and isinstance(given, dict):
            kwargs[f.name] = _build(f.type, given)
        else:
            kwargs[f.name] = given
    return dataclass_type(**kwargs)


def _apply_overrides(cfg, overrides):
    routes = {
        "source": ("source", "path"),
        "skip_frames": ("source", "skip_frames"),
        "max_frames": ("source", "max_frames"),
        "model": ("model", "path"),
        "conf_thresh": ("model", "conf_thresh"),
        "mav_device": ("control", "mav_device"),
        "mav_baud": ("control", "mav_baud"),
        "viz_enabled": ("visualization", "enabled"),
        "show_window": ("visualization", "show_window"),
        "log_enabled": ("logging", "enabled"),
        "log_per_frame": ("logging", "per_frame"),
        "log_level": ("logging", "level"),
    }
    for key, value in overrides.items():
        if value is None or key not in routes:
            continue
        section, attr = routes[key]
        setattr(getattr(cfg, section), attr, value)
