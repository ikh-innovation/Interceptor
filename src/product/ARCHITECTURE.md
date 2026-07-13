# Architecture

The tracker is a per‑frame pipeline. Each responsibility from the original
monolith is isolated behind a small interface so it can be swapped, disabled, or
tuned from config without touching the hot path.

## Data flow (one frame)

```
FrameSource ──▶ resize ──▶ SkyDetector.mask ──▶ FrameContext
                                                    │
                          ┌─────────────────────────┴─────────────┐
                    YOLO frame?                                 skip frame
                          │                                        │
              DetectionPlanner.plan                        TargetTracker.predict
              (ROI pass → sky-crop pass)                   (Kalman coast)
                          │
              SkyGatedDetector.detect  ── sky gate ──▶ Detection | None
                          │
              TargetTracker.on_result  (FSM + Kalman update)
                          │
                     fused position
                          │
   ArduPilotCommander.compute + send   (only on YOLO frames)
                          │
   Renderer.draw ──▶ Renderer.present        TrackerLogger.frame
                                             (end): TrackerLogger.summary
```

## Design patterns

- **Factory** — `FrameSourceFactory`, `RendererFactory`, `LoggerFactory`, and
  `Config.load` build the right concrete object from config (file vs camera vs
  stream; HUD vs null; console vs null). The pipeline never hard‑codes a variant.
- **Strategy** — `DetectionPlanner` encapsulates the interchangeable search
  policy: velocity‑adaptive ROI first, sky‑crop fallback, and the small‑vs‑close
  image‑size selection.
- **Null Object** — `NullRenderer` and `NullTrackerLogger` make visualization and
  logging *free* when disabled: no branches on the hot path, just a no‑op call.
  This is the primary Jetson latency lever.
- **Template Method** — `TrackingPipeline.run` is the fixed skeleton
  (connect → warmup → loop → teardown); `_step` fixes the per‑frame order
  (context → detect/coast → command → render → log) and delegates each stage.
- **State machine** — `TargetTracker` owns the `SEARCH → LOCK → PREDICT` FSM and
  the Kalman lifecycle, keeping transition rules in one place.
- **Observer** — `TrackerLogger` receives a `FrameRecord` each frame and a
  summary at end of stream, decoupled from control flow.

## Module map

| Module                        | Responsibility                                             |
| ----------------------------- | ---------------------------------------------------------- |
| `config.py`                   | Typed dataclasses; YAML + override factory                 |
| `context.py`                  | `FrameContext`, `Detection`                                |
| `sources/frames.py`           | `FrameSource` + factory (file / webcam / dev / stream / gst) |
| `detection/sky.py`            | HSV sky mask + ground‑from‑bottom reclaim                  |
| `detection/detector.py`       | Sky‑gated YOLO region inference                            |
| `detection/planner.py`        | ROI / sky‑crop pass selection (Strategy)                   |
| `tracking/kalman.py`          | Constant‑velocity Kalman filter                            |
| `tracking/state.py`           | Target FSM + Kalman lifecycle                              |
| `control/ardupilot.py`        | Centering control law + MAVLink link                       |
| `visualization/renderer.py`   | Renderer interface, null renderer, factory                 |
| `visualization/hud.py`        | Targeting HUD                                              |
| `telemetry.py`                | Console / null logger + factory                            |
| `pipeline.py`                 | Orchestrator (Template Method)                             |

## "Same detections" guarantees

The refactor preserves detection behaviour exactly:

- **Model call is verbatim** — `model(crop, imgsz=imgsz, verbose=False)` with no
  `conf` argument, so Ultralytics' default confidence floor is unchanged. Passing
  a `conf` here would change results, so it is deliberately omitted.
- **Kalman predict cadence is preserved** — on a locked/predicting YOLO frame the
  filter is advanced once for the fused estimate and once inside the ROI planner,
  exactly as the original did; this is intentional, not simplified away.
- **Pass order, image sizes, sky gate, thresholds, and the size gate**
  (`big_obj_frac`) are byte‑for‑byte the same, now sourced from config with the
  original values as defaults.
- **Command rate stays locked to the detection rate** — one setpoint per YOLO
  frame; skip frames only coast the Kalman filter and never transmit.
