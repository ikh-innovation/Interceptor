# WASP: move to another Jetson

Restored from the earlier `instructions.md` / `setup.md`, updated for the current
application. Keep the controller disarmed during installation and bench checks.

## 1. Clone the complete project

```bash
mkdir -p "$HOME/Documents"
git clone --branch virgin git@github.com:ikh-innovation/Interceptor.git "$HOME/Documents/wasp"
cd "$HOME/Documents/wasp"
PROJECT_ROOT="$PWD"
WASP_PY="$PROJECT_ROOT/.env/yolo/bin/python"
sha256sum -c models.sha256
```

The trained `runs/detect/v0/weights/best.pt` and its `best.engine` are both in Git.
The repository-root engines are older alternatives; the configured model is the
pair under `runs/detect/v0/weights`. Code, UI, fonts, logo, tests, launchers and
service template are included. Python environments and flight logs are local.

## 2. Install the matching Jetson environment

Reference system: Orin Nano Super; Jetson Linux **R36.5.2**; Ubuntu 22.04;
Python 3.10; CUDA 12.6; TensorRT 10.3; PyTorch 2.8.0 / Torchvision 0.23.0;
OpenCV 4.5.4 with GStreamer; Ultralytics 8.4.67.

Flash JetPack for the destination board. Install its system camera/TensorRT
packages, then create a local Python environment:

```bash
sudo apt update
sudo apt install -y nvidia-jetpack python3-venv python3-pip python3-opencv \
  python3-libnvinfer python3-scipy python3-matplotlib python3-yaml \
  python3-requests python3-pil python3-lxml python3-future libopenblas-dev \
  git v4l-utils gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
sudo usermod -aG dialout,video,render "$USER"
python3 -m venv --system-site-packages .env/yolo
"$WASP_PY" -m pip install --upgrade pip
"$WASP_PY" -m pip install --no-deps -r requirements-jetson.txt
# Reference JetPack 6 / CUDA 12.6 wheels; use the destination's matching builds.
"$WASP_PY" -m pip install --no-deps torch==2.8.0 torchvision==0.23.0 \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

Log out/in after changing groups. Keep system OpenCV and TensorRT; pip OpenCV
wheels can replace the GStreamer build. For a different JetPack, select matching
CUDA PyTorch/Torchvision builds using the [Jetson installation guide](https://docs.ultralytics.com/guides/nvidia-jetson/).

```bash
"$WASP_PY" - <<'PY'
import re, cv2, torch, torchvision, tensorrt, ultralytics, pymavlink
assert torch.cuda.is_available(), 'CUDA unavailable'
assert re.search(r'GStreamer:\s+YES', cv2.getBuildInformation()), 'GStreamer unavailable'
print(torch.cuda.get_device_name(0), torch.__version__, torchvision.__version__)
print('TensorRT', tensorrt.__version__, 'OpenCV', cv2.__version__, 'YOLO', ultralytics.__version__)
PY
```

## 3. Verify or rebuild TensorRT

First try the shipped engine on a matching Jetson/JetPack installation:

```bash
"$WASP_PY" scripts/export-engine.py --verify-only
```

If the GPU or TensorRT differs, rebuild **on the destination** from the tracked
`.pt`. Serialized engines have hardware/version compatibility limits; see
[NVIDIA's engine portability notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html#engine-portability).

```bash
"$WASP_PY" -m pip install --no-deps -r requirements-export.txt
"$WASP_PY" scripts/export-engine.py
```

Export uses FP16, 640 × 640, batch 1, static shapes, workspace 2 GiB.
The rebuilt engine will have a different checksum. Verify the shipped checksums
before rebuilding; the source `.pt` stays unchanged. Select the verified engine
explicitly with `WASP_MODEL=/absolute/path/to/best.engine`. For a CPU/PyTorch
diagnostic run, set `WASP_MODEL` to `best.pt` and `USE_ENGINE=0`.

## 4. Tune for the new controller and camera

| Item | Where / what to set |
| --- | --- |
| FC USB and baud | `FC_DEV=/dev/serial/by-id/...`, `FC_BAUD=115200`; check the actual port and firmware |
| Camera connection | CSI sensor ID, ribbon/driver; `CAM_DEV`, `CAM_SENSOR_ID` |
| Camera modes | `CameraConfig` and `CAM_MODES` in `src/tracker.py`; current IMX219: mode 3 / 1280×960 / 30 FPS, mode 4 / 1280×720 / 60 FPS |
| Camera field of view | Measure lens/sensor FOV; set `CAM_HFOV_DEG`, `CAM_VFOV_DEG`, and the 60 FPS crop fractions in `CAM_MODES` |
| Physical mounting | `MOUNT_TILT_DEG` in `src/flight_config.py`, `CAMERA_MOUNT_DEG`; confirm 30° for Seeker or 90° for Wasp in the UI |
| Image rotation | `CAM_FLIP_METHOD`; Wasp Advanced → Camera alignment must match the image/body axes |
| Detector | `WASP_MODEL`, `TARGET_CLASS`; the shipped model has one class named `0` |
| Target size | `TARGET_WIDTH_M` in `src/flight_config.py`; match the real target width |
| Seeker tuning | `MAX_LEAN_DEG`, `ROLL_PN_GAIN` and `MODES['sky']` in `src/flight_config.py` |
| Wasp responses | Parameters → steering responses / Advanced; the matrix shows each saved profile's values |
| Processing / network | `CV_THREADS` (reference 4), `STREAM_PORT` (8080), `STREAM_FPS` (15; use 30 for a 30 FPS web preview), `STREAM_QUALITY` (70) |
| Jetson performance | Choose a supported `nvpmodel` power mode and cooling; run `sudo jetson_clocks` if appropriate for the board |

Calibrate the new FC's sensors, RC channels, motor/frame setup and failsafes using
its normal setup procedure. The app checks Wasp ACRO calibration, fresh attitude,
`ACRO_TRAINER=0`, and `0 < RC_OVERRIDE_TIME <= 1`; review its setup status before
using controls. Keep controller-specific tuning out of the shared model files.

Stabilization starts OFF. It uses full-frame black padding and fresh existing
gyro rates when mounting/axes are known; otherwise optical flow. It changes the
preview only. Wasp motion assist has been removed. See [stabilization details](src/VIDEO_STABILIZATION.md).

## 5. Start manually, then install automatic startup

```bash
cd "$PROJECT_ROOT"
WASP_YOLO_PY="$WASP_PY" src/start-wasp.sh
```

Phone on the same network → `http://JETSON_IP:8080/` (`hostname -I` shows the IP).
Confirm live video and controller status, then stop the manual process before
installing the service:

```bash
mkdir -p "$HOME/.config/systemd/user/wasp-tracker.service.d" "$HOME/wasp-logs"
cp src/wasp-tracker.service "$HOME/.config/systemd/user/wasp-tracker.service"
cat > "$HOME/.config/systemd/user/wasp-tracker.service.d/runtime.conf" <<CONF
[Service]
Environment="WASP_YOLO_PY=$WASP_PY"
Environment="WASP_APP=mission"
Environment="CONTROL_LAW=intercept"
Environment="CONTROL_AXES=roll,pitch,yaw,throttle"
Environment="FC_STEER_MODES=ALT_HOLD,ACRO"
Environment="TRACKER_HEADLESS=1"
Environment="WASP_MODEL=$PROJECT_ROOT/runs/detect/v0/weights/best.engine"
Environment="CV_THREADS=4"
Environment="STREAM_FPS=15"
CONF
sudo systemctl enable --now nvargus-daemon.service
sudo loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now wasp-tracker.service
systemctl --user status wasp-tracker.service --no-pager
```

The template expects `~/Documents/wasp`; adjust `WorkingDirectory` and `ExecStart`
if you clone elsewhere. Add the destination's serial/camera/FOV settings to
`runtime.conf`. Check existing overrides with `systemctl --user cat wasp-tracker`.
Logs go to `~/wasp-logs/` and `src/data/`. Saved UI defaults live in
`~/.config/wasp/`: copy that folder separately **only if** those settings match
the new hardware; otherwise configure defaults in the UI. Local logs, captures,
credentials, and Python environments are intentionally not committed.

## 6. Verify the transfer

```bash
"$WASP_PY" src/tests/test_video_stabilization.py
"$WASP_PY" src/tests/test_wasp.py
"$WASP_PY" src/tests/test_camera_modes.py
"$WASP_PY" src/tests/test_wasp_ui.py
curl --fail http://127.0.0.1:8080/state | python3 -m json.tool
```

Camera-only diagnostics (stop the service so the diagnostic owns the camera):

```bash
systemctl --user stop wasp-tracker
CAM_SENSOR_MODE=3 CAM_WIDTH=1280 CAM_HEIGHT=960 CAM_FPS=30 \
  "$WASP_PY" src/tests/cameraTest.py --headless --seconds 15
systemctl --user start wasp-tracker
```

Refresh the phone UI after updating. Verify startup after reboot while disarmed.
