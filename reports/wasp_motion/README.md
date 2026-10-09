# WASP motion assist and comparison videos

Motion assist is available in **Parameters → WASP Flight parameters → Motion assist**.
It defaults to **OFF**, saves automatically, and applies across Smooth, Standard,
Responsive and Extra. It cannot change during countdown or flight.

OFF retains the existing detector, Kalman, follower and bbox steering path.
ON uses a separate estimator and bypasses the legacy Kalman and template follower
for WASP. Seeker mode is unchanged.

## What ON does

- Filters noisy target bearings using actual frame timestamps and capture-time
  aircraft attitude, so aircraft rotation is not counted as target motion.
- Rejects isolated position jumps and checks sustained changes against target
  appearance when a usable template exists.
- Runs sparse optical flow inside the target box, with forward/backward checks,
  feature agreement and appearance validation. It follows one target; it does
  not manufacture additional neural detections.
- Uses confident motion to compensate filtering delay and the configured target
  lead, while reducing small alternating corrections near the camera centre.
- Associates up to six genuine detector candidates with the current target.
- Bridges detection gaps with validated flow for at most **150 ms** from the last
  detector confirmation. The detector anchor expires at **200 ms**; guidance also
  checks its age against the existing **250 ms** command freshness limit.
- Requires two consistent detections to reacquire after an expired anchor.
  Initial acquisition remains immediate.
- Falls back to temporal filtering when tiny targets have too few reliable
  features. Synthetic test-mode boxes bypass flow and detection rejection.

Launch countdown, throttle, straight launch, steering blend, centre timer, pilot
gates and Abort/Idle behavior retain their existing meanings. This change does
not add a throttle failsafe: losing a target stops fresh steering corrections,
but the existing selected flight throttle continues until the existing flight
logic, Abort/Idle or pilot takeover changes it.

## Videos

All clips have OFF on the left and ON on the right, with an equal target-detail
zoom. MP4 files use H.264, yuv420p and fast-start metadata.

### Actual TensorRT, random motion

- [Random directions and speeds, 30 FPS](wasp-tensorrt-random-cruise-30fps.mp4)
- [Fast random bursts, vibration, blur and tiny targets, 60 FPS](wasp-tensorrt-random-bursts-60fps.mp4)

These two 20-second clips genuinely run
`runs/detect/v0/weights/best.engine` through `TensorRTBackend` on `cuda:0`.
The engine SHA-256 is
`1d0533efaeb571e1d4bd50a52248ea4a40879bccf2d0d5b06c96277bd6fd3e1a`.
The unchanged model confidence threshold is 0.25.

The photographed drone and sky come from frame 2430 of
`/home/iknowhow/Documents/Interceptor/data/ClipExport.mp4`; the original full-frame
model detection had confidence 0.875. See [asset provenance](assets/source.json).
This is a real drone cutout, not an insect drawing. Its random movement and
compositing are artificial; **the model boxes are genuine inference results**.
Neither clip is recorded flight footage or a forecast of the airframe's motion.

Both modes receive identical native 1280×960 frames, mapped to 640×480 for
tracking. The 60 FPS clip tests 60 Hz sampling at this fixed geometry; it does
not reproduce the camera sensor's separate 60 FPS crop. Every generated frame
gets a detector call. Playback FPS is not a claim about real-time inference
throughput. The tests use the existing sky gate and enable the existing follower
on the OFF side. The ON side uses its flow estimator.

Random headings/speeds, target size changes, brief image occlusions, blur and a
second drone distractor are rendered into the images. No detector boxes or
detector misses are injected. Ground truth is used only for drawing and scoring;
it never seeds the tracker, selects its crops or controls guidance.

A model hit requires centre distance at most `max(3 px, half target width)` and
bbox IoU at least 0.10. Coverage includes every video frame, including deliberate
occlusion frames. **Fresh model coverage** counts matching neural detections.
**Measured tracking coverage** counts matching boxes refreshed by a detector,
follower or flow; a merely held/predicted box is logged separately.

| Scenario | Fresh model OFF | Fresh model ON | Measured tracking OFF | Measured tracking ON |
| --- | ---: | ---: | ---: | ---: |
| Random cruise, 30 FPS, 600 frames | 91.83% | 92.33% | 98.83% | 93.50% |
| Random bursts, 60 FPS, 1200 frames | 88.42% | 87.25% | 98.42% | 87.25% |

Motion assist did **not** improve tracking coverage in these random-motion tests.
The existing follower recovered more gaps, particularly on tiny targets. ON also
had a larger RMS estimated-centre error: 1.14 vs 0.81 pixels in cruise and 1.87 vs
0.81 pixels in bursts. Different search crops account for small differences in
fresh neural hits; the neural engine and weights were never modified. ON should
remain an optional, default-OFF choice rather than an assumed upgrade for every
target or trajectory.

Each clip has a `*-metrics.json`, `*-off.csv`, `*-on.csv`, and separate
`*-off-detections.jsonl` / `*-on-detections.jsonl` files. The JSONL files preserve
actual model confidences, boxes and search regions for every frame.

### Synthetic steering demonstration

- [Synthetic disturbances and steering, 30 FPS](wasp-motion-side-by-side-30fps.mp4)
- [Synthetic disturbances and steering, 60 FPS](wasp-motion-side-by-side-60fps.mp4)

These earlier 30-second clips use **synthetic detector observations**, the
production trackers/guidance, and an assumed angular camera response. They show
jitter, vibration, isolated false boxes, target turns, short detection gaps and
a 5-pixel target. They assume a 60 ms frame delay and an 80 ms angular-rate
response, with Extra steering settings. They are not TensorRT benchmarks or
validated predictions of the physical drone's flight.

| Sampling | Pointing RMS OFF | Pointing RMS ON | Command-change RMS OFF | Command-change RMS ON |
| --- | ---: | ---: | ---: | ---: |
| 30 FPS | 4.54 px | 4.47 px | 5.17 deg/s | 2.70 deg/s |
| 60 FPS | 4.14 px | 4.53 px | 5.78 deg/s | 1.70 deg/s |

ON reduced command jitter by approximately 48% and 71%, respectively. It did
not universally improve centring. Both modes rejected the isolated false boxes
in these scenarios. Metrics and per-frame CSVs accompany both clips.

## Reproduce

From the repository root:

```bash
python3 src/tests/render_wasp_motion.py --output reports/wasp_motion --fps 30
python3 src/tests/render_wasp_motion.py --output reports/wasp_motion --fps 60
.env/yolo/bin/python src/tests/render_wasp_motion_tensorrt.py --output reports/wasp_motion
```

The renderers need ffmpeg or `imageio-ffmpeg`. The TensorRT renderer also accepts
`--ffmpeg /path/to/ffmpeg` and requires the genuine TensorRT backend; it refuses
to use another inference backend. Its deterministic seeds are 91027 and 58041.
The original synthetic renderer uses seed 7284.

Verification covers estimator noise reduction at 30/60 FPS, aircraft rotation,
outliers, appearance changes, bounded flow, stale anchors/reacquisition, no-texture
targets, geometry changes, test-mode boxes, settings persistence, profile
independence, flight locks and the controller integration. Simulated MAVLink and
browser tests exercise countdown, throttle, centre timer and Abort/Idle with
motion assistance. These checks are not physical flight qualification.
