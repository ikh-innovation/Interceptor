# Setup

## 1. Hardware

- Remove propellers. Keep the FC disarmed.
- Power off the Jetson before connecting the CSI ribbon.
- Connect the IMX219 camera to CSI camera 0 with the correct ribbon/adapter.
- Connect the FC to the Jetson by USB.
- Connect the receiver to the FC.
- Fit the Jetson fan and a suitable power supply.
- Flash the matching JetPack 6 image with [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager).

| Reference installation | Version |
| --- | --- |
| Jetson | Orin Nano Super, 8 GB |
| Jetson Linux | R36.5.2 |
| Ubuntu / Python | 22.04 / 3.10 |
| CUDA / TensorRT | 12.6 / 10.3 |
| PyTorch / Torchvision | 2.8.0 / 0.23.0 |
| Ultralytics / NumPy | 8.4.67 / 1.26.1 |
| OpenCV | 4.5.4, GStreamer enabled |
| FC | TBS LUCID H7, ArduCopter 4.6.3 |

## 2. Jetson packages and repository

```bash
sudo apt update
sudo apt install -y nvidia-jetpack python3-venv python3-pip python3-opencv \
  python3-libnvinfer python3-matplotlib python3-scipy python3-yaml \
  python3-requests python3-pil libopenblas-dev git v4l-utils \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
sudo usermod -aG dialout,video,render "$USER"
sudo systemctl enable --now nvargus-daemon.service
sudo reboot
```

```bash
mkdir -p "$HOME/Documents"
git clone --branch virgin https://github.com/ikh-innovation/Interceptor.git \
  "$HOME/Documents/Interceptor-1"
cd "$HOME/Documents/Interceptor-1"
PROJECT_ROOT="$PWD"
WASP_PY="$PROJECT_ROOT/.env/yolo/bin/python"
python3 -m venv --system-site-packages .env/yolo
"$WASP_PY" -m pip install --upgrade pip
"$WASP_PY" -m pip install 'numpy==1.26.1' filelock 'typing-extensions>=4.10' \
  'sympy>=1.13.3' networkx jinja2 fsspec psutil packaging
"$WASP_PY" -m pip install --no-deps torch==2.8.0 torchvision==0.23.0 \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
"$WASP_PY" -m pip install --no-deps ultralytics==8.4.67 ultralytics-thop==2.0.20
"$WASP_PY" -m pip install pymavlink==2.4.49 pyserial==3.5 polars==1.41.2 nvidia-ml-py
```

Use the system OpenCV and TensorRT packages. Do not install pip `opencv-python`
or replace the Jetson CUDA PyTorch wheels with CPU wheels.

```bash
cat /etc/nv_tegra_release
"$WASP_PY" - <<'PY'
import re, cv2, torch, torchvision, tensorrt, ultralytics, pymavlink
assert torch.cuda.is_available(), 'CUDA unavailable'
assert re.search(r'GStreamer:\s+YES', cv2.getBuildInformation()), 'GStreamer unavailable'
print('GPU:', torch.cuda.get_device_name(0))
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('TensorRT:', tensorrt.__version__)
print('OpenCV:', cv2.__version__, 'Ultralytics:', ultralytics.__version__)
PY
```

Orin Nano Super power mode:

```bash
sudo nvpmodel -m 2
nvpmodel -q
sudo jetson_clocks
sudo jetson_clocks --fan
```

Expected: `MAXN_SUPER`. On another Jetson, choose its supported performance mode.
Repeat `jetson_clocks` after reboot, or install the startup unit below.

```bash
sudo tee /etc/systemd/system/jetson-clocks.service >/dev/null <<'UNIT'
[Unit]
Description=Jetson performance clocks
After=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now jetson-clocks.service
```

## 3. Camera

```bash
ls -l /dev/video0
v4l2-ctl --list-devices
cat /sys/class/video4linux/video0/name
systemctl is-active nvargus-daemon.service
```

| IMX219 choice | Sensor mode | Output | Sensor crop |
| --- | --- | --- | --- |
| 30 FPS | 3 | 1280 × 960 | Full view |
| 60 FPS | 4 | 1280 × 720 | Narrower view |

```bash
systemctl --user stop wasp-tracker.service
CAM_SENSOR_ID=0 CAM_SENSOR_MODE=3 CAM_WIDTH=1280 CAM_HEIGHT=960 CAM_FPS=30 \
  "$WASP_PY" src/tests/cameraTest.py --headless --seconds 15
CAM_SENSOR_ID=0 CAM_SENSOR_MODE=4 CAM_WIDTH=1280 CAM_HEIGHT=720 CAM_FPS=60 \
  "$WASP_PY" src/tests/cameraTest.py --headless --seconds 15
```

For another sensor, use its documented modes. Update `CameraConfig` and
`CAM_MODES` in `src/tracker.py`: sensor mode, dimensions, FPS, crop fractions and
camera field of view. Repeat both camera tests.

## 4. Controller

In Mission Planner or another ArduPilot GCS:

1. Install firmware for the exact FC board.
2. Calibrate accelerometer, compass and radio.
3. Verify motor order, motor direction and frame type.
4. Assign transmitter modes: STABILIZE, ALT_HOLD and ACRO.
5. Verify transmitter arming/disarming and takeover.
6. Set/read the following parameters. Save and reboot the FC.

| Parameter | Reference value |
| --- | ---: |
| SYSID_MYGCS | 255 |
| RC_OPTIONS | 32; ignore-overrides bit 1 clear |
| RC_OVERRIDE_TIME | 1 |
| ACRO_TRAINER | 0 |
| ACRO_RP_RATE | 360 |
| ACRO_Y_RATE | 202.5 |
| ACRO_RP_EXPO | 0 |
| ACRO_Y_EXPO | 0 |
| ANGLE_MAX | 3000 |
| PILOT_SPEED_UP | 2500 |
| PILOT_SPEED_DN | 400 |
| PILOT_ACCEL_Z | 1500 |
| PSC_JERK_Z | 5 |
| ATC_INPUT_TC | 0.15 |

Reference airframe values; calibrate/tune before applying to another aircraft:

| Parameter | Recorded value |
| --- | ---: |
| PSC_ACCZ_P / PSC_ACCZ_I | 0.5 / 1 |
| ATC_ACCEL_R_MAX / ATC_ACCEL_P_MAX | 146100 / 146100 |
| FS_THR_ENABLE / FS_EKF_ACTION | 3 / 1 |
| FS_GCS_ENABLE | 0 |

Match RC mapping to the actual receiver. Reference transmitter:

| Axis | Mapping | MIN | TRIM | MAX | DZ | REVERSED |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Throttle | RCMAP_THROTTLE=1 | 988 | 990 | 2010 | 30 | 0 |
| Roll | RCMAP_ROLL=2 | 996 | 1499 | 2010 | 20 | 0 |
| Pitch | RCMAP_PITCH=3 | 988 | 1500 | 2007 | 20 | 0 |
| Yaw | RCMAP_YAW=4 | 988 | 1499 | 2010 | 20 | 0 |

Calibrate your own `RCn_MIN/TRIM/MAX/DZ/REVERSED`; do not copy another radio's
endpoints. Tune your own airframe. Verify radio, battery and EKF failsafes.

```bash
ls -l /dev/serial/by-id/
"$WASP_PY" src/tests/fc_check.py --help
"$WASP_PY" src/tests/fc_check.py --connect /dev/serial/by-id/YOUR_FC-if00
```

Keep the tracker stopped while another application owns the FC serial port.

## 5. TensorRT engine

Export on the destination Jetson. Use the trained weights under `runs/detect/v0`.

```bash
"$WASP_PY" -m pip install onnx onnxslim
YOLO_AUTOINSTALL=false "$WASP_PY" - <<'PY'
from ultralytics import YOLO
model = YOLO('runs/detect/v0/weights/best.pt')
print(model.export(format='engine', imgsz=640, half=True, device=0,
                   batch=1, dynamic=False, workspace=2))
PY
YOLO_AUTOINSTALL=false "$WASP_PY" - <<'PY'
import numpy as np
from ultralytics import YOLO
model = YOLO('runs/detect/v0/weights/best.engine', task='detect')
model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=640, verbose=False)
print(model.names)
assert len(model.names) == 1
PY
```

Use `runs/detect/v0/weights/best.engine`. Rebuild after changing GPU or TensorRT
version. The repository-root `.engine` files are not the configured runtime model.

## 6. Saved phone settings

```bash
mkdir -p "$HOME/.config/wasp"
cat > "$HOME/.config/wasp/detection.json" <<'JSON'
{"cam":60,"follow":true}
JSON
cat > "$HOME/.config/wasp/auto.json" <<'JSON'
{"height":15,"climb":3,"turn":30,"adjust":5,"window":5,"ready":0.25,"giveup":2,"closing":3,"floor":3,"line":30,"ai":"MAX","steer":0.3}
JSON
cat > "$HOME/.config/wasp/levels.json" <<'JSON'
{"vert":"MEDIUM","yaw":"MEDIUM","push":"MEDIUM","ai":"MEDIUM","search":"MEDIUM"}
JSON
```

## 7. Automatic startup

```bash
mkdir -p "$HOME/.config/systemd/user/wasp-tracker.service.d" "$HOME/wasp-logs"
SERVICE_FILE="$HOME/.config/systemd/user/wasp-tracker.service"
sed "s|/home/iknowhow/Documents/Interceptor-1|$PROJECT_ROOT|g; s|/home/iknowhow/wasp-logs|$HOME/wasp-logs|g" \
  src/wasp-tracker.service > "$SERVICE_FILE.tmp"
mv "$SERVICE_FILE.tmp" "$SERVICE_FILE"
chmod +x src/start-wasp.sh src/run.py
cat > "$HOME/.config/systemd/user/wasp-tracker.service.d/runtime.conf" <<CONF
[Service]
Environment="WASP_YOLO_PY=$WASP_PY"
Environment="WASP_APP=mission"
Environment="CONTROL_LAW=intercept"
Environment="CONTROL_AXES=roll,pitch,yaw,throttle"
Environment="FC_STEER_MODES=ALT_HOLD,ACRO"
Environment="CAMERA_MOUNT_DEG=90"
Environment="CAM_TILT_DEG=30"
Environment="TRACKER_HEADLESS=1"
Environment="CV_THREADS=4"
Environment="REQUIRE_ARMED=1"
Environment="REQUIRE_RATE_MODE=1"
Environment="STREAM=1"
Environment="STREAM_PORT=8080"
Environment="STREAM_FPS=15"
Environment="WASP_MODEL=$PROJECT_ROOT/runs/detect/v0/weights/best.engine"
Environment="USE_ENGINE=1"
CONF
sudo loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now wasp-tracker.service
systemctl --user status wasp-tracker.service --no-pager
loginctl show-user "$USER" -p Linger
```

Expected: `enabled`, `active (running)`, `Linger=yes`.

## 8. Network and phone

```bash
nmcli --ask device wifi connect "YOUR_WIFI_SSID"
nmcli connection modify "YOUR_WIFI_SSID" connection.autoconnect yes
hostname -I
curl --fail http://127.0.0.1:8080/state | python3 -m json.tool
```

Phone → same Wi-Fi → `http://JETSON_IP:8080/`.

| Phone setting | Set |
| --- | --- |
| Parameters → Flight mode | Wasp drone |
| Physical camera / confirmation popup | 90°, upward → Confirm |
| Detection → Sky gate | OFF for the current reference setting; ON at startup |
| Detection → Follower | ON |
| Detection → Camera | 60 FPS |
| Launch throttle | 100% |
| Flight time | 10 s |
| Flight time no det | 3 s; choices 3 / 5 / 7 / 10 s |
| Straight launch | 1 s |
| Steering blend | 0.5 s |
| Steering rate | 60°/s; 45°/s also available |
| Steering gain | 2 |
| Target lead | 80 ms |
| Camera alignment | 0° only when image bottom points toward the nose |

Launch → hold attitude at launch throttle until the first valid detection.
First detection → complete any remaining Straight launch → Steering blend → AI.
Flight time counts from launch, including the initial no-detection wait.
No detection timeout / Flight time / target loss after acquisition / ABORT →
release all Jetson overrides to the pilot; disarm before another launch.

Accepted Seeker ENGAGE / AUTO or Wasp LAUNCH → continues with Wi-Fi disconnected.
Jetson → FC remains wired USB. Transmitter mode switch → pilot control.
Held phone movement buttons → expire after 0.3 s without fresh requests.

Camera alignment check, disarmed and propellers removed: move an object above the
camera toward the aircraft nose.

| Object moves in image | Camera alignment |
| --- | ---: |
| Down | 0° |
| Right | 90° |
| Up | 180° |
| Left | 270° |

Seeker drone → physical camera 30° → Confirm. Refresh retains the active mode;
after a service restart, select the mode and confirm the camera again.

## 9. Checks

```bash
"$WASP_PY" src/flight_config.py
"$WASP_PY" src/tests/test_wasp.py
"$WASP_PY" src/tests/test_camera_modes.py
"$WASP_PY" src/tests/test_detection_performance.py
"$WASP_PY" src/tests/test_wasp_mavlink.py
"$WASP_PY" src/tests/test_wifi_autonomy.py
python3 src/tests/test_wasp_ui.py
curl --fail http://127.0.0.1:8080/state | python3 -m json.tool
tail -n 80 "$HOME/wasp-logs/wasp-tracker.log"
```

| `/state` field | Expected while setting up |
| --- | --- |
| detector | TensorRT |
| cam_fps | 60 |
| armed / steering | false / false |
| linked | true |
| mission.wasp_enabled | true |
| mission.wasp_camera_confirmed | true |
| mission.wasp_setup_why | empty string |
| mission.wasp_attitude_fresh | true |
| mission.wasp_params.duration / gain | 10 / 2 |

Browser tests require Chromium, Google Chrome or Firefox.

FPS checks:

```bash
nvpmodel -q
sudo jetson_clocks --show
timeout 10s tegrastats --interval 1000
curl --fail http://127.0.0.1:8080/state | python3 -m json.tool
```

Camera tests measure source FPS. `/state.fps` measures the processing loop.
The phone video stream is capped at 15 FPS. Keep `CV_THREADS=4`; model input
remains 640 with one detection pass per processed frame.

## 10. Power-up check

Keep the FC disarmed.

```bash
sudo reboot
```

After reboot:

```bash
systemctl --user is-enabled wasp-tracker.service
systemctl --user is-active wasp-tracker.service
loginctl show-user "$USER" -p Linger
hostname -I
curl --fail http://127.0.0.1:8080/state | python3 -m json.tool
```

Phone → select Wasp drone → confirm physical 90° camera → verify live video and
TensorRT. Verify no automatic arming or launch.

Before flight: verify detection on the actual intended target, command direction
with propellers removed, ABORT, transmitter takeover and telemetry/target-loss
release. Timeout/ABORT releases Jetson overrides; it does not land or disarm.

[Jetson PyTorch packages](https://forums.developer.nvidia.com/t/install-pytorch-for-cuda-12-6-jetpack-6-2/348456)
· [TensorRT export](https://docs.ultralytics.com/guides/nvidia-jetson/)
· [ArduPilot RC input](https://ardupilot.org/dev/docs/mavlink-rcinput.html)
