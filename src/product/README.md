# WASP Tracker

Production build of the sky‑gated, adaptive‑zoom drone interceptor tracker
(refactored from `src/development/modules/object_tracking/tracker.py`).

It detects small/distant drones against the sky, tracks them with a Kalman
filter, and streams GUIDED‑mode velocity + yaw‑rate setpoints to an ArduPilot
Copter flight controller over MAVLink — one setpoint per detection frame.

The detection maths are **identical** to the development tracker; this build only
reorganises the code into modules, makes the visualization and logging optional,
and adds a typed config + CLI so it can be deployed and tuned without editing code.

---

## What changed vs. the dev script

| Concern            | Dev `tracker.py`                    | Production `wasp_tracker`                                  |
| ------------------ | ----------------------------------- | --------------------------------------------------------- |
| Structure          | one 1000‑line file                  | modular package, one responsibility per module            |
| Visualization      | always drawn, `imshow` toggled      | fully optional Null‑renderer (zero cost when off)         |
| Logging            | `print()` per frame                 | custom logger to a per‑run file (or stdout), per‑frame **and** end‑of‑stream summary, optional |
| Config             | constants in the class              | `config.yaml` + typed dataclasses + CLI overrides         |
| Input              | video file only                     | video file, webcam, `/dev/videoN`, RTSP, GStreamer        |
| Detection results  | —                                   | **unchanged** (same model call, thresholds, and pass order) |

Latency win on Jetson: run with `viz_enabled=False`. All HUD drawing is skipped entirely
(not just `imshow`), while detection/tracking/flight output stay bit‑for‑bit the same.

---

## Install

```bash
cd <repo-root>
python -m venv .venv
. .venv/bin/activate
pip install -r src/product/requirements.txt
```

Validated on Python 3.11 with the pinned versions in `requirements.txt`.

## Quick start

The entrypoint is the `Client` class in [`run.py`](run.py). Set the parameters you
want in its `__main__` block and launch from the **repository root** (asset paths
in `config.yaml` are relative to it):

```bash
python src/product/run.py
```

```python
# Video file, with the targeting HUD window:
Client(source="./data/videos_drones/field.MP4").run()

# Live camera, headless + max performance (typical Jetson deployment):
Client(source=0, viz_enabled=False).run()

# Short offline smoke test, no window, summary only:
Client(max_frames=60, show_window=False, log_per_frame=False).run()
```

`Client` can also be imported and driven from your own script. Press `q` in the
HUD window to quit.

## Client parameters

Every parameter left as `None` falls back to its `config.yaml` value; anything you
pass overrides it. `config` selects the base config file.

| Parameter        | Effect                                                          |
| ---------------- | --------------------------------------------------------------- |
| `config`         | Path to the base config file                                    |
| `source`         | Video file / `0` / `/dev/videoN` / `rtsp://…` / GStreamer string |
| `model`          | YOLO weights                                                    |
| `conf_thresh`    | Confidence acceptance threshold                                 |
| `skip_frames`    | Detection rate = `fps / (N + 1)`                                |
| `max_frames`     | Stop after N frames                                             |
| `mav_device`     | e.g. `/dev/ttyACM0`; `None` ⇒ offline, commands print‑only      |
| `mav_baud`       | MAVLink baud rate                                               |
| `viz_enabled`    | `False` disables HUD + window (fastest)                         |
| `show_window`    | `False` keeps the HUD but skips `imshow`                        |
| `log_enabled`    | `False` disables all logging                                    |
| `log_per_frame`  | `False` keeps only the end‑of‑stream summary                    |
| `log_level`      | `INFO` (default), `DEBUG`, …                                    |
| `log_sink`       | `file` (default) / `console` / `both`                           |
| `log_dir`        | Directory for per‑run log files (default `logs`)                |

## Logging

By default each run writes its own timestamped file, **`logs/run_<timestamp>.log`**
(the `logs/` directory is git‑ignored; the path is printed at start‑up). Per frame
it records the frame index, `DET`/`---` detection flag, state, layer/confidence,
YOLO + Kalman timing in ms, fused pixel position, and the outgoing flight command;
at end of stream it appends a summary (frames, detection rate, throughput, YOLO
latency mean/min/max/std, state histogram) — so the file tells the whole story of
the video. Stats use O(1) memory, so it is safe on an indefinitely long live stream.

- `logging.sink`: `file` (default) · `console` (stdout) · `both`
- `logging.log_dir`: where per‑run files go (default `logs`)
- `logging.per_frame`: `false` keeps only the summary
- `logging.enabled`: `false` disables logging entirely

Override per run on `Client` with `log_sink`, `log_dir`, `log_per_frame`,
`log_enabled`, `log_level`.

## Camera sources

`source` accepts:

- a **file path** (`./data/videos_drones/field.MP4`)
- a **webcam index** (`0`)
- a **V4L2 device** (`/dev/video0`)
- an **RTSP/HTTP/UDP** URL
- a **GStreamer pipeline** (auto‑detected when the string contains `!`, e.g. a
  Jetson CSI camera via `nvarguscamerasrc ... ! appsink`)

Live sources use a 1‑frame capture buffer for lowest latency.

## Flight / MAVLink

Steering goes to an **ArduPilot Copter** FC as `SET_POSITION_TARGET_LOCAL_NED`
in `MAV_FRAME_BODY_OFFSET_NED` (body‑frame velocity + yaw‑rate). The FC must be in
**GUIDED** mode. On connect the link is verified to be ArduPilot and fails loudly
otherwise (the Lucid H7 must be flashed to ArduPilot Copter — Betaflight/INAV will
not honour these setpoints). Set the **real lens FOV** (`control.h_fov_deg` /
`v_fov_deg`) before flight. With no `mav_device`, commands are computed and logged
but nothing is transmitted.

## Jetson deployment

- Install **torch/torchvision from NVIDIA's JetPack wheels** — do not pip‑install
  the desktop pins. Use the CUDA‑enabled OpenCV bundled with JetPack.
- Run with `viz_enabled=False` (and `log_per_frame=False` to skip per‑frame disk writes).
- Keep `detection.warmup: true` so the first frame does not pay CUDA init latency.
- For the CSI camera pass an `nvarguscamerasrc` GStreamer pipeline as `source`.

## Layout

```
src/product/
├── run.py            CLI entrypoint
├── config.yaml       runtime configuration
├── requirements.txt  pinned dependencies
├── README.md
├── ARCHITECTURE.md   design patterns + data flow
└── wasp_tracker/
    ├── config.py         typed config + YAML factory
    ├── context.py        per-frame context + Detection type
    ├── pipeline.py        orchestrator (main loop)
    ├── telemetry.py      logger (console / null) + factory
    ├── detection/        sky mask, sky-gated YOLO detector, ROI planner
    ├── tracking/         Kalman filter + target state machine
    ├── control/          ArduPilot MAVLink commander
    ├── sources/          frame-source factory (file / camera / stream)
    └── visualization/    HUD renderer + null renderer + factory
```

See `ARCHITECTURE.md` for the design patterns and how "same detections" is guaranteed.
