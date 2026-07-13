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
| Logging            | `print()` per frame                 | custom stdout logger, per‑frame **and** end‑of‑stream summary, optional |
| Config             | constants in the class              | `config.yaml` + typed dataclasses + CLI overrides         |
| Input              | video file only                     | video file, webcam, `/dev/videoN`, RTSP, GStreamer        |
| Detection results  | —                                   | **unchanged** (same model call, thresholds, and pass order) |

Latency win on Jetson: run with `--no-viz`. All HUD drawing is skipped entirely
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

Run from the **repository root** (asset paths in `config.yaml` are relative to it):

```bash
# Video file, with the targeting HUD window:
python src/product/run.py

# Live camera, headless + max performance (typical Jetson deployment):
python src/product/run.py --source 0 --no-viz

# Short offline smoke test, no window, summary only:
python src/product/run.py --max-frames 60 --no-window --quiet
```

Press `q` in the HUD window to quit.

## CLI

Every flag overrides the matching `config.yaml` value:

| Flag             | Effect                                                           |
| ---------------- | --------------------------------------------------------------- |
| `--config PATH`  | Use a different config file                                      |
| `--source S`     | Video file / `0` / `/dev/videoN` / `rtsp://…` / GStreamer string |
| `--model PATH`   | YOLO weights                                                    |
| `--conf F`       | Confidence acceptance threshold                                 |
| `--skip-frames N`| Detection rate = `fps / (N + 1)`                                |
| `--max-frames N` | Stop after N frames                                             |
| `--mav-device D` | e.g. `/dev/ttyACM0`; unset ⇒ offline, commands print‑only       |
| `--no-viz`       | Disable HUD + window (fastest)                                  |
| `--no-window`    | Keep HUD, skip `imshow`                                         |
| `--no-log`       | Disable all logging                                             |
| `--quiet`        | Per‑frame logs off, keep the summary                           |
| `--log-level L`  | `INFO` (default), `DEBUG`, …                                    |

## Logging

The logger writes to **stdout only — never to a file**. Per frame it emits state,
layer, YOLO/Kalman timing, fused pixel position, and the outgoing flight command;
at end of stream it prints a summary (frames, detection rate, throughput, YOLO
latency mean/min/max/std, state histogram). Stats are computed with O(1) memory,
so it is safe on an indefinitely long live stream. Toggle with `logging.enabled`
/ `logging.per_frame` in the config or `--no-log` / `--quiet`.

## Camera sources

`--source` accepts:

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
- Run with `--no-viz` (and `--quiet` if stdout is slow).
- Keep `detection.warmup: true` so the first frame does not pay CUDA init latency.
- For the CSI camera pass an `nvarguscamerasrc` GStreamer pipeline to `--source`.

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
