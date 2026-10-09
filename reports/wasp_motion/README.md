# WASP: five hard 30 FPS TensorRT comparisons

Motion assist OFF is on the left; ON is on the right. These are offline vision stress tests.

The target and sky are photographs from frame 2430 of `/home/iknowhow/Documents/Interceptor/data/ClipExport.mp4`. The original target box was `[455, 348, 508, 378]`, with TensorRT confidence 0.875. Movement, size changes and camera motion are synthetic. Neural detections are genuine.

The suite uses one photographed target pose. It is a stress comparison of the current pipelines, not a model-accuracy benchmark on unseen flight footage.

Engine: `runs/detect/v0/weights/best.engine`; SHA-256 `1d0533efaeb571e1d4bd50a52248ea4a40879bccf2d0d5b06c96277bd6fd3e1a`. Backend: `TensorRTBackend` on `cuda:0`. Model confidence threshold: 0.25.

Every video uses 30 FPS, native 1280×960 pixels and 640×480 tracking coordinates. Each contains 20 seconds of motion plus a 2-second results card. Playback FPS does not establish real-time inference throughput.

Both production trackers receive identical frames and capture-time camera attitude. OFF retains the existing Kalman/follower path; ON uses the current motion-assist estimator. Existing sky gating and detector settings remain enabled. No tracker tuning or model weight changes were made for these tests. Ground truth never seeds a detector or tracker.

The camera follows the same prescribed path in both modes, independently of tracking outputs. These videos isolate vision behavior rather than simulate closed-loop steering.

The camera translates forward at an assumed 25 m/s and rotates with the simulated body, without a gimbal. A cloud plane at 900m supplies translation parallax; bank, pitch, yaw and 9–13Hz vibration supply rigid-camera motion. The photographed target is projected as a 0.6m planar object whose range changes repeatedly, producing apparent zoom without changing the fixed lens calibration. Exposure blur is an approximate directional image filter, capped at 21 native pixels. This kinematic model is not calibrated against the physical drone, and camera angles/accelerations are deliberately aggressive.

Coverage denominators include only frames where the entire target is in view and not deliberately occluded. Each video also shows visible-frame counts, so leaving the camera view cannot masquerade as a detector failure or improvement. A correct box requires centre distance <= max(3px, half target width) and bbox IoU >= 0.10. Measured tracking can come from a detector, follower or validated flow. Held/predicted boxes are excluded. The centre-error RMS includes all measured estimates while the target is visible, including wrong associations; it does not penalize missing estimates.

| Scenario | Fresh model OFF / ON | Correct tracking OFF / ON | Centre RMS OFF / ON | Target visible |
| --- | ---: | ---: | ---: | ---: |
| [01-zigzag](01-zigzag-30fps.mp4) | 61.7% / 72.1% | 68.5% / 10.2% | 1.06 / 7.53 px | 587/600 |
| [02-hairpins](02-hairpins-30fps.mp4) | 50.6% / 65.3% | 53.1% / 16.1% | 1.33 / 2.75 px | 522/600 |
| [03-spirals](03-spirals-30fps.mp4) | 50.7% / 67.4% | 59.1% / 4.4% | 1.44 / 2.59 px | 596/600 |
| [04-crossings](04-crossings-30fps.mp4) | 36.5% / 58.6% | 54.9% / 3.7% | 1.13 / 0.96 px | 430/600 |
| [05-chaos](05-chaos-30fps.mp4) | 70.0% / 74.6% | 81.5% / 7.8% | 22.20 / 113.90 px | 536/600 |

| Scenario | Apparent speed p95 / max | Apparent width min–max | Max body rotation | Seed |
| --- | ---: | ---: | ---: | ---: |
| 01-zigzag | 2098 / 2606 px/s | 3.5–66.7 px | 262 deg/s | 27091 |
| 02-hairpins | 3277 / 5097 px/s | 3.2–61.1 px | 282 deg/s | 38214 |
| 03-spirals | 2773 / 3519 px/s | 3.4–57.9 px | 262 deg/s | 65128 |
| 04-crossings | 3149 / 4491 px/s | 3.4–58.7 px | 262 deg/s | 70439 |
| 05-chaos | 1996 / 2879 px/s | 3.8–56.2 px | 282 deg/s | 91572 |

The earlier videos, previews and JSON/CSV files were removed. Only these five MP4 comparisons, this report and the two real photographic assets remain.

Reproduce from the repository root:

```bash
.env/yolo/bin/python src/tests/render_wasp_motion_stress.py
```

Use `--ffmpeg /path/to/ffmpeg` if ffmpeg is not on PATH. The script accepts `--scenario 1` through `--scenario 5` for a single scenario and requires the genuine TensorRT backend. It opens no FC connection and sends no flight commands.

Verdict: OFF has higher correct-tracking coverage in all five scenarios. ON obtains more fresh neural target hits in these runs, but its current estimator accepts far fewer correct tracks. The chaotic distractor scenario also produces large measured-centre errors. Motion assist remains OFF by default; these results do not support treating ON as a flight improvement.
