# Jetson Setup: Complete Guide (for Claude or a human)

**Purpose:** Set up a brand-new Jetson Orin Nano Super to fly the Interceptor
drone — WASP mode, SEEKER mode, ASTERIX integration, GCP cloud dashboard, and
Firestore flight logging. Every step is here. Follow in order.

**Reference hardware:**
- Jetson Orin Nano Super (JetPack 6 / Jetson Linux R36.5.2)
- TBS LUCID H7 flight controller (ArduCopter) on USB → `/dev/ttyACM0`
- IMX219 CSI camera (ribbon cable, sensor-id 0)
- RFD 868x EU radio (4-wire UART, optional)
- CAN GPS+compass module on FC CAN port

**Network:**
- Jetson WiFi IP: `192.168.205.160` (field network)
- Laptop IP: `192.168.205.172`
- GCP relay: `34-76-37-31.sslip.io`
- Dashboard password: ask the team (user: `pilot`)

---

## Phase 1: Flash and base OS

1. Flash **JetPack 6** (R36.5.2) using NVIDIA SDK Manager on a host PC.
2. Create user `iknowhow` during OOBE.
3. Connect to WiFi / Ethernet for internet access.
4. After first boot:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y nvidia-jetpack python3-venv python3-pip python3-opencv \
  python3-libnvinfer python3-scipy python3-matplotlib python3-yaml \
  python3-requests python3-pil python3-lxml python3-future libopenblas-dev \
  git v4l-utils gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good
sudo usermod -aG dialout,video,render "$USER"
```

5. **Log out and log back in** (group changes need a new session).

---

## Phase 2: Clone the repository

```bash
mkdir -p "$HOME/Documents"
git clone --branch virgin https://github.com/ikh-innovation/Interceptor.git \
  "$HOME/Documents/Interceptor"
cd "$HOME/Documents/Interceptor"
```

**Create the compatibility symlink** — the systemd service template references
`~/Documents/wasp`; the repo is now named `Interceptor`:

```bash
ln -sfn "$HOME/Documents/Interceptor" "$HOME/Documents/wasp"
```

Set shorthand variables used throughout:

```bash
export PROJECT_ROOT="$HOME/Documents/Interceptor"
export WASP_PY="$PROJECT_ROOT/.env/yolo/bin/python"
```

---

## Phase 3: Python environment

```bash
cd "$PROJECT_ROOT"
python3 -m venv --system-site-packages .env/yolo
"$WASP_PY" -m pip install --upgrade pip
"$WASP_PY" -m pip install --no-deps -r requirements-jetson.txt
"$WASP_PY" -m pip install --no-deps torch==2.8.0 torchvision==0.23.0 \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

**Verify everything is installed:**

```bash
"$WASP_PY" - <<'PY'
import re, cv2, torch, torchvision, tensorrt, ultralytics, pymavlink
assert torch.cuda.is_available(), 'CUDA unavailable'
assert re.search(r'GStreamer:\s+YES', cv2.getBuildInformation()), 'GStreamer unavailable'
print(torch.cuda.get_device_name(0), torch.__version__, torchvision.__version__)
print('TensorRT', tensorrt.__version__, 'OpenCV', cv2.__version__, 'YOLO', ultralytics.__version__)
PY
```

Expected: prints GPU name, versions, no errors.

---

## Phase 4: Model integrity and TensorRT engine

Verify the shipped model weights are intact:

```bash
cd "$PROJECT_ROOT"
sha256sum -c models.sha256
```

All lines must say OK. If any fail, the repo is corrupt — re-clone.

The repo ships pre-built engines. Verify they match this Jetson:

```bash
"$WASP_PY" scripts/export-engine.py --verify-only
```

If it fails (different GPU/TensorRT), rebuild:

```bash
"$WASP_PY" -m pip install --no-deps -r requirements-export.txt
"$WASP_PY" scripts/export-engine.py
"$WASP_PY" scripts/export-engine.py --imgsz 960
```

---

## Phase 5: Flight controller check

1. Plug the FC into a Jetson USB port. Verify it appears:

```bash
ls /dev/ttyACM* /dev/serial/by-id/usb-ArduPilot*
```

Expected: `/dev/ttyACM0` (or similar).

2. Load the reference parameters (props OFF, outdoors):

```bash
"$WASP_PY" src/tests/fc_param_check.py \
  --conn /dev/serial/by-id/usb-ArduPilot_*-if00 --apply
```

This compares against `deploy/fc/TBS_LUCID_H7_2026-10-28.param` and writes
accepted values. **Never copy sensor calibration IDs.**

3. After loading params, **recalibrate on the new board:**
   - Accelerometer calibration (level)
   - Compass calibration (outdoors, rotate in all axes)
   - RC calibration (all sticks full range)
   - ESC calibration
   - Motor test (props OFF! check order + direction)

4. Verify critical params:

```bash
"$WASP_PY" src/tests/fc_param_check.py \
  --conn /dev/serial/by-id/usb-ArduPilot_*-if00
```

Every line under "App, SafeDown, GPS and ARM checks" must read **OK**:
- `ACRO_TRAINER=0`
- `RC_OVERRIDE_TIME=1`
- `FLTMODE_CH=6` (switch: low=ACRO, middle=STABILIZE, high=ALT_HOLD)
- `SYSID_MYGCS=255`
- `RC5_OPTION=154`
- GPS in EKF, compass on and detected

See `controller-settings.md` for full details on every critical parameter.

---

## Phase 6: Camera check

The IMX219 CSI camera needs `nvargus-daemon`:

```bash
sudo systemctl enable --now nvargus-daemon.service
```

Quick test (stop any running tracker first):

```bash
systemctl --user stop wasp-tracker 2>/dev/null || true
CAM_SENSOR_MODE=3 CAM_WIDTH=1280 CAM_HEIGHT=960 CAM_FPS=30 \
  "$WASP_PY" src/tests/cameraTest.py --headless --seconds 15
```

Expected: prints frame stats, no errors. If it fails:
- Check ribbon cable is seated properly
- `sudo systemctl restart nvargus-daemon`
- `ls /dev/video*` should show at least `video0`

---

## Phase 7: App settings (optional, saves time)

If hardware is identical to the reference drone, copy saved configs:

```bash
mkdir -p ~/.config/wasp
cp deploy/wasp-config/*.json ~/.config/wasp/
```

Otherwise, skip this and configure in the UI after first start.

---

## Phase 8: Manual test run (before service install)

Start the app once manually to confirm camera + FC work together:

```bash
cd "$PROJECT_ROOT/src"
WASP_YOLO_PY="$WASP_PY" ./start-wasp.sh
```

Open `http://<JETSON_IP>:8080/` on a phone/laptop on the same network. You
should see live video and controller status. Press Ctrl-C to stop.

If manual start fails, fix the issue before proceeding — the service will have
the same problem but with less-visible error output.

---

## Phase 9: Install the wasp-tracker service

```bash
cd "$PROJECT_ROOT"
mkdir -p "$HOME/.config/systemd/user/wasp-tracker.service.d" "$HOME/wasp-logs"

# Copy the service file (template uses ~/Documents/wasp via the symlink from Phase 2)
cp src/wasp-tracker.service "$HOME/.config/systemd/user/wasp-tracker.service"
```

**Create the runtime config override:**

```bash
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
```

**Optional env vars** (add to `runtime.conf` only if non-default):

| Variable | Default | When to set |
|----------|---------|-------------|
| `FC_DEV` | auto-detect USB | Non-standard FC port |
| `FC_BAUD` | `115200` | Different baud rate |
| `CAM_SENSOR_ID` | `0` | Multiple CSI cameras |
| `CAM_HFOV_DEG` / `CAM_VFOV_DEG` | IMX219 defaults | Different lens/sensor |
| `MOUNT_TILT_DEG` | `30` (Seeker) / `90` (Wasp) | Different camera angle |
| `TARGET_CLASS` | `0` | Different YOLO class |
| `TARGET_WIDTH_M` | configured in UI | Different target size |
| `STREAM_QUALITY` | `70` | Trade bandwidth for quality |

**Set Jetson power mode** (do once, persists across reboots):

```bash
sudo nvpmodel -m 0          # max performance
sudo jetson_clocks           # lock clocks (optional, run after each boot or add to rc.local)
```

**Enable linger (so user services start at boot, not just at login):**

```bash
sudo loginctl enable-linger "$USER"
```

**Start the service:**

```bash
systemctl --user daemon-reload
systemctl --user enable --now wasp-tracker.service
systemctl --user status wasp-tracker.service --no-pager
```

**Verify the HTTP server is up:**

```bash
curl --fail http://127.0.0.1:8080/state | python3 -m json.tool
```

Expected: JSON with `linked`, `fc_mode`, `cam_frame`, etc.

---

## Phase 10: GCP cloud dashboard registration

Register this Jetson with the cloud relay so it appears on the dashboard.
Pick a unique name (lowercase, hyphens OK, e.g. `wasp-01`, `wasp-02`):

```bash
cd "$PROJECT_ROOT"
sudo deploy/gcp/jetson/install-frpc.sh --name wasp-01
```

It will prompt for the dashboard password. Expected output:

```
[ok] Jetson 'wasp-01' registered and the Interceptor UI is already up.
     Dashboard: https://34-76-37-31.sslip.io/
     Direct:    https://wasp-01.34-76-37-31.sslip.io/
```

---

## Phase 11: Firestore flight logging (GCP)

This uploads battery telemetry and flight metadata to GCP Firestore in real time.

### 11a. One-time setup (done once for the whole project, not per Jetson)

**These two commands only need to run once, from any machine with gcloud:**

```bash
# Grant Firestore access to the Jetson service account
gcloud projects add-iam-policy-binding ikh-drone-cloud \
  --member="serviceAccount:jetson-log-writer@ikh-drone-cloud.iam.gserviceaccount.com" \
  --role="roles/datastore.user"

# Create the key file
gcloud iam service-accounts keys create jetson-key.json \
  --iam-account=jetson-log-writer@ikh-drone-cloud.iam.gserviceaccount.com
```

**DONE ALREADY** as of 2026-11-11. The key file is at
`$PROJECT_ROOT/jetson-key.json` on the current Jetson. Copy it to new Jetsons
(it is gitignored — never commit it).

### 11b. Per-Jetson setup

Copy the key file to the new Jetson:

```bash
# From laptop or another Jetson:
scp jetson-key.json iknowhow@<NEW_JETSON_IP>:~/Documents/Interceptor/
```

Add the GCP environment variables to the service override:

```bash
cat >> "$HOME/.config/systemd/user/wasp-tracker.service.d/runtime.conf" <<CONF
Environment="GCP_PROJECT=ikh-drone-cloud"
Environment="JETSON_ID=wasp-01"
Environment="GOOGLE_APPLICATION_CREDENTIALS=$PROJECT_ROOT/jetson-key.json"
CONF
```

**Change `JETSON_ID=wasp-01` to this drone's unique name** (same name used in
`install-frpc.sh`).

Reload and restart:

```bash
systemctl --user daemon-reload
systemctl --user restart wasp-tracker.service
```

---

## Phase 12: Reboot test

This is the most important step. **Reboot the Jetson:**

```bash
sudo reboot
```

After reboot, verify **without logging in** (services must auto-start):

1. **From the laptop** (same WiFi network):

```bash
# HTTP server up?
curl --fail http://192.168.205.160:8080/state | python3 -m json.tool

# Camera delivering frames?
curl -s http://192.168.205.160:8080/state | python3 -c "
import json,sys; j=json.load(sys.stdin)
print('Camera:', 'OK' if j.get('cam_frame') and not j.get('cam_error') else j.get('cam_error','no frames'))
print('FC:', 'linked' if j.get('linked') else 'NOT linked')
print('Mode:', j.get('fc_mode','--'))
"
```

2. **From the cloud dashboard**: open `https://34-76-37-31.sslip.io/` — the
   drone card should appear within ~30 seconds of the Jetson being up. Camera
   and Controller should show "connected".

3. **From a phone on any network**: open the cockpit URL
   `https://wasp-01.34-76-37-31.sslip.io/` — live video, controller status.

---

## Phase 13: Run the test suite

```bash
cd "$PROJECT_ROOT"
"$WASP_PY" -m unittest discover -s src/tests -v
```

Or individually:

```bash
"$WASP_PY" src/tests/test_safedown.py
"$WASP_PY" src/tests/test_seeker_missions.py
"$WASP_PY" src/tests/test_asterix_eketa.py
```

All must pass.

---

## Phase 14: Bench test (props OFF!)

**Outdoors, props OFF, controller disarmed:**

1. Power up flat on a level surface.
2. Mode switch **DOWN** (ACRO). Throttle stick down. Arm switch down.
3. On the phone UI, tap **ARM** (the app arms the FC; amber 2.5 s countdown).
4. Tap blue **LAUNCH WASP** (full throttle, no props = safe).
5. Mode switch **UP** (ALT_HOLD) → SafeDown should engage → RTL → disarm.
6. Repeat: ARM, LAUNCH, mode switch to **MIDDLE** (STABILIZE) → pilot catch
   mode, sticks return to pilot. Disarm.
7. ARM, then tap **SAFEDOWN** on phone → must disarm immediately.

If any step fails, **do not fly.** Debug with:

```bash
journalctl --user -u wasp-tracker -f
```

---

## Phase 15: SafeDown SITL simulation (optional, ~1 hour)

Fly the real app against ArduCopter SITL to verify SafeDown under different
conditions. Build SITL once:

```bash
git clone --depth 1 --branch Copter-4.6.2 --recurse-submodules --shallow-submodules \
  https://github.com/ArduPilot/ardupilot.git ~/ardupilot
cd ~/ardupilot && python3 ./waf configure --board sitl && python3 ./waf copter
cd "$PROJECT_ROOT/src/tests"
"$WASP_PY" sitl_safedown.py --list          # list scenarios
"$WASP_PY" sitl_safedown.py -k nominal -j 1 # one scenario
"$WASP_PY" sitl_safedown.py -j 4            # all, 4 at a time
```

---

## Quick reference: services on this Jetson

| Service | Purpose | Check |
|---------|---------|-------|
| `nvargus-daemon` (system) | CSI camera driver | `sudo systemctl status nvargus-daemon` |
| `wasp-tracker` (user) | Interceptor app | `systemctl --user status wasp-tracker` |
| `frpc` (system) | Cloud tunnel | `sudo systemctl status frpc` |

**Restart everything after code changes:**

```bash
cd ~/Documents/Interceptor && git pull
systemctl --user restart wasp-tracker.service
```

**Logs:**

```bash
# App logs
tail -f ~/wasp-logs/wasp-tracker.log

# Cloud tunnel
sudo journalctl -u frpc -f

# Camera daemon
sudo journalctl -u nvargus-daemon -n 50
```

---

## Laptop: ground planner (for ASTERIX field testing)

On the laptop (same WiFi as the Jetson):

```bash
cd ~/Documents/Interceptor
git pull
python src/ground_planner.py --jetson 192.168.205.160
```

Open `http://localhost:9000`. Draw targets on the satellite map, hit PLAY to
send fake ASTERIX CAT062 packets to the Jetson. The drone's cockpit UI shows
the targets; the pilot picks one and taps SEEK.

---

## Safety checklist before every flight

- [ ] Props tight, arms locked
- [ ] Battery charged, voltage above 12.0 V (3S) / 16.0 V (4S)
- [ ] GPS lock: phone UI shows YES / > 8 sats
- [ ] Compass: phone UI shows YES
- [ ] Camera: live video in the phone UI
- [ ] FC linked: phone UI shows mode (ACRO / STABILIZE / ALT_HOLD)
- [ ] Mode switch DOWN = ACRO, MIDDLE = STABILIZE, UP = ALT_HOLD
- [ ] Geofence drawn on the phone (Geofence button)
- [ ] Pilot can take over at any time: ACRO + throttle > 70%
- [ ] SAFEDOWN button tested on ground (should disarm)
- [ ] Wind < 5 m/s for first flights

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Camera black | `sudo systemctl restart nvargus-daemon && systemctl --user restart wasp-tracker` |
| FC not linked | Check USB cable, `ls /dev/ttyACM*`, restart service |
| Not on cloud dashboard | `sudo journalctl -u frpc -n 50` — check token/network |
| ASTERIX no data | Check Jetson WiFi IP, UDP port 8600, run `ground_planner.py` |
| "Address in use" on laptop | `pkill -f ground_planner` then rerun |
| Engage question not appearing | Pull latest code, restart service |
| Pilot can't take over | Mode switch must be in ACRO + throttle > 70% |

---

## Related documents

| Document | What it covers |
|----------|---------------|
| `instructions.md` | Detailed hardware tuning (camera modes, FOV, FC env vars, SITL) |
| `controller-settings.md` | Every FC parameter the app depends on and why |
| `pilot-instructions-el.md` | Pilot procedure (Greek) |
| `deploy/gcp/README.md` | First-time GCP relay VM setup (Terraform, Caddy, frps) |
| `deploy/gcp/NEW_JETSON.md` | Quick 3-step cloud registration (subset of this doc) |
| `deploy/fc/TBS_LUCID_H7_2026-10-28.param` | Reference FC parameter file |
| `deploy/wasp-config/*.json` | App settings (profiles, throttle, timers, camera, engine) |
| `src/VIDEO_STABILIZATION.md` | Video stabilization details |
