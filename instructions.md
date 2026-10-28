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

The trained `runs/detect/v0/weights/best.pt` and its `best.engine` (640 px) and
`best_960.engine` (960 px) are in Git, built on an Orin Nano Super with TensorRT 10.3.
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
A larger input for far, small targets is built the same way next to it, without touching
`best.engine`: `"$WASP_PY" scripts/export-engine.py --imgsz 960` writes `best_960.engine`.
The tracker loads every `best_<size>.engine` beside the selected engine at start; the app's
Detection > Engine row switches between them between frames and saves the choice as the default.
The rebuilt engine will have a different checksum. Verify the shipped checksums
before rebuilding; the source `.pt` stays unchanged. Select the verified engine
explicitly with `WASP_MODEL=/absolute/path/to/best.engine`. For a CPU/PyTorch
diagnostic run, set `WASP_MODEL` to `best.pt` and `USE_ENGINE=0`.

## 4. Tune for the new controller and camera

| Item | Where / what to set |
| --- | --- |
| FC USB and baud | `FC_DEV=/dev/serial/by-id/...`, `FC_BAUD=115200`; check the actual port and firmware |
| Camera connection | CSI sensor ID, ribbon/driver; `CAM_DEV`, `CAM_SENSOR_ID` |
| Camera modes | `CameraConfig` and `CAM_MODES` in `src/tracker.py`; current IMX219: mode 3 / 1280×960 / 30 FPS, mode 4 / 1280×720 / 60 FPS, mode 0 / 3280×2464 / 21 FPS (8 MP). Frames stay in NVMM memory: `src/hw_capture.py` scales the preview and detection crops on the VIC (`HW_CAPTURE=0` forces the old CPU pipeline, which is also the automatic fallback). A switch only counts once the camera delivers the mode's frame size; the app's Camera row shows what it is delivering |
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

## 4b. Replacement flight controller, camera or settings

Everything needed to rebuild the drone is in the repository:

| What | Where |
| --- | --- |
| FC parameters (all 1271, read 2026-10-28, GPS and compass on) | `deploy/fc/TBS_LUCID_H7_2026-10-28.param` |
| What each WASP-critical parameter does | `controller-settings.md` |
| App settings (WASP profiles, throttle, ramp, timers, camera, engine) | `deploy/wasp-config/*.json` |
| Pilot procedure (Greek) | `pilot-instructions-el.md` |

**New FC (same model, ArduCopter):**
1. Flash the same ArduCopter version.
2. Stop the app (`systemctl --user stop wasp-tracker`) and run
   `"$WASP_PY" src/tests/fc_param_check.py --conn /dev/serial/by-id/usb-ArduPilot_*-if00 --apply`:
   it lists every difference from `deploy/fc/TBS_LUCID_H7_2026-10-28.param` and writes the ones
   you accept, one by one, read back (sensor calibrations and IDs are never copied). Or in
   Mission Planner: Full Parameter List → Load from file → Write Params. Reboot the FC, start
   the app again.
3. **Recalibrate on the new board.** The file carries the old board's sensor
   offsets. Do the accelerometer, compass, radio (RC) and ESC calibration, and
   check motor order and direction with Motor Test, props off.
4. Run `fc_param_check.py` again without `--apply`: every line under "App, SafeDown, GPS and
   ARM checks" must read OK (`ACRO_TRAINER=0`, `RC_OVERRIDE_TIME=1`, `FLTMODE_CH=6` with switch
   low ACRO / middle STABILIZE / high ALT_HOLD, `SYSID_MYGCS=255`, `RC5_OPTION=154`, GPS in the
   EKF, compass on and detected). All the values and why are in `controller-settings.md`.
5. For a different FC model or frame, don't load the whole file. Set only
   section 1 of `controller-settings.md` by hand.

**New camera:** same IMX219 module, nothing to change. For another sensor or
lens, measure the field of view and set `CAM_HFOV_DEG`, `CAM_VFOV_DEG` and the
`CAM_MODES` table (section 4). Then confirm the 90° mount and the camera alignment
in the app (Parameters → WASP → Advanced).

**App settings:** with identical hardware, copy them before the first start:
`mkdir -p ~/.config/wasp && cp deploy/wasp-config/*.json ~/.config/wasp/`.
Otherwise leave them out and set the values in the UI.

**GPS with compass:** a module with 6 wires (5V, GND, TX, RX, SDA, SCL): GPS TX to the FC's
R2, RX to T2 (SERIAL2), SDA / SCL to the FC's I2C. Calibrate the compass in Mission Planner.
The phone's GPS/Compass tile must read YES / YES outdoors.

**After any replacement:** do a props-off bench check, outdoors (ARM needs a GPS position):
power up flat, mode switch DOWN, throttle and arm switch down. Phone **ARM** (the app arms the
FC; amber 2.5 s countdown), then the blue **LAUNCH WASP** (full throttle at once). Switch up
(SafeDown: ALT_HOLD, then RTL). Disarm (arm switch up, then down), ARM and LAUNCH again, switch
to the middle (catch: STABILIZE, sticks back to the pilot after 1–3 s), disarm. Last: ARM, then
SAFEDOWN on the phone must disarm it.

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
"$WASP_PY" src/tests/test_safedown.py
"$WASP_PY" src/tests/test_wasp_mavlink.py
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

### SafeDown in simulation (optional, about 1 hour)

`src/tests/sitl_safedown.py` flies the real app against real ArduCopter SITL with
the FC parameters from `deploy/fc/`: launch from the tilted 4-leg stand, then
SafeDown under different masses, CG offsets, wind, gusts, upsets, dives and
faults. Build SITL once (about 7 minutes on the Jetson):

```bash
git clone --depth 1 --branch Copter-4.6.2 --recurse-submodules --shallow-submodules \
  https://github.com/ArduPilot/ardupilot.git ~/ardupilot
cd ~/ardupilot && python3 ./waf configure --board sitl && python3 ./waf copter
cd "$PROJECT_ROOT/src/tests"
"$WASP_PY" sitl_safedown.py --list          # the scenarios
"$WASP_PY" sitl_safedown.py -k nominal -j 1 # one or a few
"$WASP_PY" sitl_safedown.py -j 4            # all of them, 4 at a time
```

It looks for `~/ardupilot/build/sitl/bin/arducopter`, or `WASP_SITL_BIN`. It
skips if neither exists.
