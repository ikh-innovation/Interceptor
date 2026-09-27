# INSTALL_PROMPT — bring WASP up on a new Jetson

**For the human:** clone the repo on the new Jetson, open Claude Code in the repo folder, and paste this one line:

> Read INSTALL_PROMPT.md and do everything in it on this Jetson. Check every item, fix what you are allowed to fix, ask me for the physical facts, and end with the final report.

Everything below is written **for the Claude Code session running on the new Jetson**.

---

## Your job

Make this Jetson run WASP exactly like the reference Jetson: same camera mode, same detector engine, same service, same checks passing. Work through the phases **in order**. For every item, **check first, fix only if it is wrong, then check again**. At the end, print the final report (template at the bottom).

### Hard rules
1. **Never write a flight-controller parameter.** Read them and report them. The human changes them in Mission Planner.
2. **Never arm, never spin motors, never run `fc_check.py --wiggle`.** Anything that talks to the FC runs with **props off** and the FC disarmed. Ask the human to confirm props are off before phase 6.
3. **Do not change control-law gains, speeds or mission timings** (`fc_control.py`, `mission.py`, `flight_config.py` modes, `tracker.py` `MODES`). They were tuned in simulation and in the field. Only machine-specific settings are yours to fix: paths, packages, engines, camera mode, service.
4. **Physical facts come from the human, never from a default.** These are the camera tilt angle and the target width. Ask.
5. If a check fails and the fix is not listed here, **stop and report it**. Do not improvise around it.
6. Do not commit or push unless the human asks.

---

## Reference machine (what "correct" looks like)

| Item | Reference value |
|---|---|
| Board | NVIDIA Orin Nano Developer Kit |
| L4T / JetPack | R35.5.0 (JetPack 5.1.3) |
| Python | 3.8.10, venv at `<repo>/.env/yolo`, `include-system-site-packages = true` |
| torch | 2.2.0, **`torch.cuda.is_available() == True`** |
| ultralytics | 8.4.160 |
| OpenCV | 4.5.4 from JetPack, **built with GStreamer** (comes via system site-packages) |
| TensorRT | 8.5.2.2 (JetPack) |
| numpy / pymavlink / pyserial / huggingface_hub | 1.23.5 / 2.4.49 / 3.5 / 0.36.2 |
| Camera | IMX219 on CSI `sensor-id=0`, `/dev/video0` |
| Camera mode | **sensor-mode 3, 1280×960 @ 30 fps** (full sensor, 62.2° × 48.8°) |
| Detector (sky) | `runs/detect/v0/weights/best.pt` (in git) → `best.engine` built **on this machine** |
| Detector (human) | `iitolstykh/YOLO-Face-Person-Detector` / `yolov8x_person_face.pt` in the HF cache |
| Tracker frame rate | ~30 fps, detector reported as `TensorRT` |
| User | `iknowhow` (uid 1000), groups include `dialout video i2c gpio render` |
| Repo path | `/home/iknowhow/Documents/Interceptor` |
| Service | systemd **user** unit `wasp-tracker.service`, GDM autologin |
| Logs | `~/wasp-logs/wasp-tracker.log`, per-run files in `src/data/` |
| Phone app | `http://<jetson-ip>:8080/`, PIN in `~/.wasp_ui_token` |

Different versions are **not automatically wrong**, but every difference goes in the final report. JetPack and TensorRT decide whether an engine from elsewhere can run. You rebuild it here anyway (phase 4).

---

## Phase 1 — Machine and OS

Check:
```bash
cat /proc/device-tree/model; echo; head -1 /etc/nv_tegra_release
whoami; id -u; id -nG
pwd                      # must be the repo root
nvpmodel -q 2>/dev/null | head -2
df -h ~ | tail -1        # want > 10 GB free (engines, videos)
```
Fix / decide:
- **Repo path or user differ from the reference?** Five files hard-code `/home/iknowhow/Documents/Interceptor` or `/home/iknowhow/wasp-logs`:
  - `src/start-wasp.sh`
  - `src/start-wasp-sky.sh`
  - `src/wasp-tracker.service` (`WorkingDirectory`, `ExecStart`, `StandardOutput`, `StandardError`)
  - `XAUTHORITY=/run/user/1000/...` in both launch scripts, which assumes uid 1000

  Show the human the exact edits, apply them after they agree, and list them in the report.
- Add the user to the groups the hardware needs, if missing: `sudo usermod -aG dialout,video,i2c,gpio,render $USER`. A re-login is needed; tell the human.
- Set max performance, and report it: `sudo nvpmodel -m 0` (or the board's MAXN mode) and `sudo jetson_clocks`.

## Phase 2 — Python environment

Check:
```bash
ls .env/yolo/bin/python && grep include-system .env/yolo/pyvenv.cfg
.env/yolo/bin/python - <<'EOF'
import sys, torch, cv2, numpy, ultralytics, pymavlink, serial, huggingface_hub, tensorrt
print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available())
print("cv2", cv2.__version__, "GStreamer" if "GStreamer:                   YES" in cv2.getBuildInformation() else "NO GSTREAMER")
print("ultralytics", ultralytics.__version__, "trt", tensorrt.__version__, "numpy", numpy.__version__)
EOF
```
Fix:
- **No venv:** `python3 -m venv --system-site-packages .env/yolo`. Without `--system-site-packages`, it cannot see JetPack's GStreamer OpenCV or TensorRT.
- **torch without CUDA** (the PyPI wheel is CPU-only on Jetson): install NVIDIA's CUDA wheel that matches this JetPack. Everything still "works" on CPU, but the detector drops to ~0.3 fps and the seeker is blind. This check is mandatory.
- **Missing packages:** `.env/yolo/bin/pip install ultralytics==8.4.160 pymavlink pyserial huggingface_hub "numpy<1.24"`. Then **check that pip did not replace JetPack's OpenCV or numpy** with a PyPI build. `cv2` must still report GStreamer. If ultralytics pulled in `opencv-python`, uninstall it so the system cv2 wins.

## Phase 3 — Repository sanity

```bash
git status --short
ls src/mission_ui.html runs/detect/v0/weights/best.pt src/flight_config.py
```
`src/mission_ui.html` is the phone app. If it is missing, the phone shows "UI file missing" and has **no ENGAGE / STOP / ABORT**. Restore it: `git checkout -- src/mission_ui.html`.

## Phase 4 — Detector models and TensorRT

1. **Sky weights** (`runs/detect/v0/weights/best.pt`) come with the repo. Check that the file exists and loads:
   `.env/yolo/bin/python -c "from ultralytics import YOLO; m=YOLO('runs/detect/v0/weights/best.pt'); print(m.names)"`, which prints `{0: '0'}`.
2. **Build the TensorRT engine on this machine.** Engines are tied to one GPU and one TensorRT version, and `.engine` files are git-ignored on purpose:
   ```bash
   cd runs/detect/v0/weights
   ../../../../.env/yolo/bin/yolo export model=best.pt format=engine half=True imgsz=640 device=0
   ```
   - It takes several minutes and produces `best.engine` next to `best.pt`. The tracker picks it up automatically.
   - It must be **exported by ultralytics**, so the class metadata is embedded. The tracker refuses an engine with placeholder class names.
   - The input is fixed at 640, which is expected.
3. **Face/person model**: only needed if `flight_config.py` is ever set to `MODE = "human"` (the in-air HUMAN/DRONE switch was removed; sky mode never loads it). Download it into the HF cache **while online**; the field has no internet and the launchers set `HF_HUB_OFFLINE=1`:
   ```bash
   .env/yolo/bin/python -c "from huggingface_hub import hf_hub_download as d; print(d('iitolstykh/YOLO-Face-Person-Detector','yolov8x_person_face.pt'))"
   ```
   Then check that it loads with `HF_HUB_OFFLINE=1` set.

## Phase 5 — Camera

1. **Presence:**
   - `ls /dev/video0 /sys/class/video4linux/` and `systemctl is-active nvargus-daemon`.
   - **No `/dev/video0`:** it is almost always the CSI ribbon. Ask the human to reseat it at both ends. `sudo dmesg | grep -i imx219` shows `i2c read probe (-121)` when the sensor does not answer.
   - **No IMX219 overlay:** `sudo /opt/nvidia/jetson-io/jetson-io.py` → Configure for compatible hardware → the IMX219 camera on the correct port → reboot.
2. **Confirm mode 3 is the full sensor on THIS driver.** This is the check that matters. On the reference machine, mode 4 (1280×720) turned out to be a centre crop (50.4° × 29.7°) while the code assumes 62.2° × 48.8°. Every angle was wrong, and in simulation touches fell from ~80% to ~45–65%. Read the crop windows straight from the driver:
   ```bash
   python3 - <<'EOF'
   import os, struct
   b = open(f"/lib/modules/{os.uname().release}/kernel/drivers/media/i2c/nv_imx219.ko", "rb").read()
   i = -1
   while (i := b.find(b"\x64\x01", i + 1)) >= 0:           # register 0x0164, little endian
       if i % 2 or struct.unpack_from("<H", b, i + 4)[0] != 0x0165:
           continue
       r, p = {}, i
       while (a := struct.unpack_from("<H", b, p)[0]) != 0x0001:   # table end marker
           r[a] = b[p + 2]; p += 4
       g = lambda h, l: (r.get(h, 0) << 8) | r.get(l, 0)
       print(f"x {g(0x164,0x165)}-{g(0x166,0x167)}  y {g(0x168,0x169)}-{g(0x16a,0x16b)}  "
             f"out {g(0x16c,0x16d)}x{g(0x16e,0x16f)}  binning {r.get(0x174)},{r.get(0x175)}")
   EOF
   ```
   - **Correct result:** the table with `out 1640x1232` shows `x 0-3279 y 0-2463 binning 1,1`. That is the full sensor, 2×2 binned, so mode 3 at 1280×960 gives 62.2° × 48.8°.
   - **Anything else** (a different sensor, driver or crop): **stop and report it**. `CAM_HFOV_DEG` and `CAM_VFOV_DEG` would then need to be set to the real field of view. Compute it with focal = 1640 / tan(31.1°) = 2718.8 px (full-array pixels): FOV = 2·atan(half-window / 2718.8).
3. **Capture test** (the service must be stopped; the camera has a single owner):
   ```bash
   systemctl --user stop wasp-tracker 2>/dev/null
   cd src && timeout -s INT 12 ../.env/yolo/bin/python -u tests/cameraTest.py --headless
   ```
   You want `sensor-mode=3 ... width=1280, height=960` in the pipeline and **~30 fps average**. Mode 3 and 1280×960 are the code defaults (`tracker.CameraConfig`). Check that nothing overrides them: `grep -rn CAM_SENSOR_MODE\|CAM_HEIGHT src/*.sh src/*.service`.
4. **Tilt:** ask the human for the **measured** camera angle (degrees up from horizontal, inclinometer on the camera body, drone on level ground). Sky mode needs **30**. Do not continue with sky mode unless they confirm it.

## Phase 6 — Flight controller link (props OFF, disarmed)

Ask the human: *"Props off, FC powered over USB, disarmed?"* Wait for a yes.

```bash
ls -l /dev/serial/by-id/ /dev/ttyACM*          # the port is auto-discovered; do not hard-code it
cd src
../.env/yolo/bin/python tests/fc_check.py --connect /dev/ttyACM0 --verify
../.env/yolo/bin/python tests/check_modes.py --connect /dev/ttyACM0
```
Checks:
- **Firmware:** it should be ArduPilot Copter.
- **RC override:** `--verify` must prove the override landed (the echo in RC_CHANNELS matches). A "silently dropped" result means the source sysid is not `SYSID_MYGCS` (or `MAV_GCS_SYSID` on 4.6+). `run.py` adopts it automatically, so report it but do not change the FC.
- **Modes:** ALT_HOLD must be reachable from the pilot's mode switch.

Then report these parameters against the values the app expects. `run.py` prints the same checklist at connect, under `MISSION APP ... FC`:

| Parameter | Wanted | Why |
|---|---|---|
| `PILOT_SPEED_UP` | 2500 | the AI's climb authority (25 m/s) |
| `PILOT_SPEED_DN` | 250 | slow descents (0 would copy UP = 25 m/s down) |
| `PILOT_ACCEL_Z` | 500 | used for stop distance |
| `ANGLE_MAX` | 4000 | 40° lean for PUSH and the AI |
| `RC_OVERRIDE_TIME` | 1 | a hung Jetson's last command dies in 1 s |
| `ATC_INPUT_TC` | 0.2–0.25 | softer response to a new command |
| `LOG_BACKEND_TYPE` | 1 + SD card | otherwise no FC logs exist |
| `INS_LOG_BAT_MASK` | 1 | IMU batch log for vibration FFT |
| `SYSID_MYGCS` | read, not changed | the override source sysid |

**Never write these.** List the mismatches for the human to fix in Mission Planner.

## Phase 7 — Flight configuration

Show `src/flight_config.py` settings to the human and confirm each one:
- `MODE = "sky"` (drone target) or `"human"`
- `MOUNT_TILT_DEG` = the measured angle from phase 5
- `TARGET_WIDTH_M` = the target drone's **measured** motor-tip-to-motor-tip width (the placeholder is 0.30)
- `COMMIT_CHAN` = a spare RC channel if the pilot wired one, otherwise 0

Then check it: `.env/yolo/bin/python src/flight_config.py`. It refuses to run if the mode and tilt disagree, and that refusal is correct.

## Phase 8 — Auto-start service

```bash
grep -i automatic /etc/gdm3/custom.conf          # want AutomaticLoginEnable=true, AutomaticLogin=<user>
mkdir -p ~/wasp-logs ~/.config/systemd/user
ln -sf "$PWD/src/wasp-tracker.service" ~/.config/systemd/user/wasp-tracker.service
systemctl --user daemon-reload
systemctl --user enable wasp-tracker.service
systemctl --user is-enabled wasp-tracker.service   # enabled
```
- **Autologin off:** the unit is a user service tied to the graphical session, so without autologin it never starts at power-up. Set this in `/etc/gdm3/custom.conf` under `[daemon]` (needs sudo; tell the human):
  ```
  AutomaticLoginEnable=true
  AutomaticLogin=<user>
  ```
- **Phone PIN:** created on the first run in `~/.wasp_ui_token`. Give it to the human.

## Phase 9 — Tests (no hardware in the loop)

```bash
cd src
for t in test_control_law test_intercept test_acro_dual test_schedule test_yaw_circles \
         test_mission test_mission_mavlink test_field_sequence test_simulated_fc \
         test_recording test_sky_mask test_hold_wifi test_lock; do
  printf "%-22s " $t; timeout 600 ../.env/yolo/bin/python tests/$t.py 2>&1 | tail -1
done
timeout 300 ../.env/yolo/bin/python tests/test_ui_browser.py 2>&1 | tail -1     # needs firefox
../.env/yolo/bin/python tests/sim_sky_wind.py --quick                            # the odds, not pass/fail
../.env/yolo/bin/python tests/sim_lock_wind.py --quick                           # LOCK vs gusts, the odds
```
- Every test should end with `all passed`. `test_field_sequence` ends with `PASSES END TO END`.
- For reference, the full `sim_sky_wind.py` run (defaults: AI MEDIUM, PUSH MEDIUM 30°, 12 target motions including 45° ones) on the reference machine gives:
  - touch 73 / 75 / 74 / 66% at Bf 0 / 1 / 2 / 3
  - AI takes over 99%
  - no ground contact
  - peak yaw rate ≤ 40°/s
- The full `sim_lock_wind.py` gives:
  - LOCK and the FC alone within ~1.5° in normal conditions
  - at Bf 3 / HIGH sensitivity, "ends >10°": FC only 15/25, LOCK 3/25
  - LOCK's own command never above 30°/s

  Numbers far from these mean the environment differs. Report them.

## Phase 10 — Live start, then reboot

1. **Live start:**
   - Run `systemctl --user restart wasp-tracker` and wait until `[CTRL]` lines appear in `~/wasp-logs/wasp-tracker.log`.
   - Check the log shows:
     - `sensor-mode=3 ... 1280, height=960`
     - `[MODEL] TensorRT engine selected: best.engine`
     - `[STREAM] seeker view -> http://<ip>:8080/`
     - the FC being found
   - Read `http://127.0.0.1:8080/state` with Python (`curl` may not be installed). It should show `fps` ≈ 30 and `detector` = `TensorRT`.
   - Fetch `http://127.0.0.1:8080/`. It must be the app, not "UI file missing".
2. **Reboot:** ask the human to reboot (do not do it yourself; it ends your session). In the next session, check:
   - `systemctl --user status wasp-tracker` shows it active
   - the newest `src/data/*.log` started within ~1 minute of boot (`uptime -s`)

   There is no RTC battery. Without a network the clock starts at 1970, and files are named `nosync_rNNNN_HHMMSS`. That is expected.

---

## Final report (print exactly this shape)

```
WASP INSTALL REPORT — <hostname>, <date>
Board / L4T:        <model>, <R35.x>         [same | DIFFERENT: ...]
Python env:         torch <v> cuda <yes/no>, ultralytics <v>, cv2 <v> gst <yes/no>, trt <v>
Repo path / user:   <path>, <user> uid <n>    [edits made: ...]
Sky engine:         built <yes/no>, loads <yes/no>, tracker fps <n>
Face model cached:  <yes/no>, loads offline <yes/no>
Camera:             /dev/video0 <yes/no>, mode 3 = full sensor <yes/no>, 1280x960 @ <fps>
Camera tilt:        <deg> (confirmed by human: <yes/no>)
FC:                 <firmware> <version>, override verified <yes/no>, ALT_HOLD reachable <yes/no>
FC params to fix:   <list, or none>
flight_config:      MODE <..>, tilt <..>, target width <..>, commit chan <..>
Service:            enabled <yes/no>, autologin <yes/no>, starts at boot <verified / awaiting reboot>
Phone:              http://<ip>:8080/?k=<PIN>
Tests:              <n>/<n> passed  [failures: ...]
Sim (quick):        touch Bf0-3 <..>%, AI takeover <..>%, crashes <..>
Open items:         <anything that needs the human>
```
