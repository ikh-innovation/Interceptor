import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wasp_tracker import Config, TrackingPipeline

CONFIG_PATH = str(Path(__file__).with_name("config.yaml"))


class Client:
    def __init__(
        self,
        config=CONFIG_PATH,
        source=None,
        model=None,
        conf_thresh=None,
        skip_frames=None,
        max_frames=None,
        mav_device=None,
        mav_baud=None,
        viz_enabled=None,
        show_window=None,
        log_enabled=None,
        log_per_frame=None,
        log_level=None,
        log_sink=None,
        log_dir=None,
    ):
        self.config = Config.load(config, {
            "source": source,
            "model": model,
            "conf_thresh": conf_thresh,
            "skip_frames": skip_frames,
            "max_frames": max_frames,
            "mav_device": mav_device,
            "mav_baud": mav_baud,
            "viz_enabled": viz_enabled,
            "show_window": show_window,
            "log_enabled": log_enabled,
            "log_per_frame": log_per_frame,
            "log_level": log_level,
            "log_sink": log_sink,
            "log_dir": log_dir,
        })
        self.pipeline = TrackingPipeline(self.config)

    def run(self):
        self.pipeline.run()


if __name__ == "__main__":
    Client(
        source="./data/videos_drones/field.MP4",
        conf_thresh=0.01,
        skip_frames=0,
        viz_enabled=True,
        mav_device=None,
    ).run()
