# Jetson setup and Wasp operation

## 1. Pull on the Jetson

Use the Jetson account that owns the project. Keep the FC disarmed and remove
propellers for setup and initial command checks.

```bash
systemctl --user stop wasp-tracker.service
PROJECT_ROOT="$HOME/Documents/Interceptor-1"  # adjust to your checkout
cd "$PROJECT_ROOT"
git status --short
git switch virgin
git pull --ff-only origin virgin
```

Preserve any Jetson edits before pulling. Then ask Codex on the Jetson:
**“Read instructions.md, check my Jetson setup, and install and verify the boot service.”**

## 2. Check the existing Jetson environment

Use your existing JetPack/CUDA Python environment; adjust `WASP_PY` if necessary.
Keep these variables in the same terminal for the remaining commands.

```bash
WASP_PY="$PROJECT_ROOT/.env/yolo/bin/python"
test -x "$WASP_PY"
"$WASP_PY" -c 'import re, cv2, torch, ultralytics, pymavlink; assert torch.cuda.is_available(), "CUDA unavailable"; assert re.search(r"GStreamer:\s+YES", cv2.getBuildInformation()), "Check OpenCV GStreamer support"; print("CUDA and camera dependencies OK")'
"$WASP_PY" src/flight_config.py
ls -l /dev/video0 /dev/serial/by-id/
"$WASP_PY" src/tests/test_wasp.py
"$WASP_PY" src/tests/test_wasp_mavlink.py
```

Stop and resolve failed checks before continuing. Keep `MOUNT_TILT_DEG=90` and
`CAM_TILT_DEG=30`. Check that the actual mosquito is detected: the repository's
model metadata describes drone training, not a verified mosquito model.

In the FC configuration, verify `ACRO_TRAINER=0`, `0 < RC_OVERRIDE_TIME <= 1`,
readable ACRO rate/expo parameters, calibrated roll/pitch/yaw/throttle channels,
and working transmitter takeover/failsafes. Do not disable arming or mode checks.

## 3. Install automatic startup without desktop login

Connect the camera and FC. Configure the Jetson's Wi-Fi to reconnect at boot.
Run as the project owner, using `sudo` only where shown:

```bash
cd "$PROJECT_ROOT"
systemctl --user disable --now wasp-tracker.service
mkdir -p "$HOME/.config/systemd/user/wasp-tracker.service.d" "$HOME/wasp-logs"
sed -e "s|/home/iknowhow/Documents/Interceptor-1|$PROJECT_ROOT|g" \
    -e "s|/home/iknowhow/wasp-logs|$HOME/wasp-logs|g" \
    src/wasp-tracker.service > "$HOME/.config/systemd/user/wasp-tracker.service"
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
Environment="REQUIRE_ARMED=1"
Environment="REQUIRE_RATE_MODE=1"
Environment="STREAM=1"
Environment="STREAM_PORT=8080"
CONF
sudo systemctl enable --now nvargus-daemon.service
sudo loginctl enable-linger "$(id -un)"
systemctl --user daemon-reload
systemctl --user enable --now wasp-tracker.service
systemctl --user status wasp-tracker.service --no-pager
systemctl --user cat wasp-tracker.service
tail -n 80 "$HOME/wasp-logs/wasp-tracker.log"
```

Check for old service overrides in `systemctl --user cat`; reconcile conflicting
paths/settings. If this is a first install, “unit does not exist” from the initial
stop/disable command is expected. Camera/model/FC startup can take a few minutes.

```bash
curl --fail http://127.0.0.1:8080/state | python3 -m json.tool
hostname -I
loginctl show-user "$(id -un)" -p Linger
```

Require `Linger=yes`, live video, and a reachable `/state` before rebooting.
Keep the FC disarmed:

```bash
sudo reboot
```

## 4. Open the phone app after reboot

1. Connect the phone to the same reachable Wi-Fi network.
2. Open **http://JETSON_IP:8080/** using the address from `hostname -I`.
3. Choose **WASP**. The chooser appears again on every refresh.
4. Open **Parameters**. Confirm the fixed **90° camera**, image rotation and live video.
5. Check the Wasp FC setup status. `wasp_setup_why` must be empty and
   `wasp_attitude_fresh` true in `/state`. “Disarmed” is expected before arming.

If the app does not appear, on the Jetson run:

```bash
systemctl --user status wasp-tracker.service --no-pager
journalctl --user -u wasp-tracker.service -n 80 --no-pager
tail -n 80 "$HOME/wasp-logs/wasp-tracker.log"
curl --fail http://127.0.0.1:8080/state
hostname -I
```

If local `/state` works, check the phone's address/network, Wi-Fi client isolation
and firewall. Hard-refresh the browser after updates. Use a trusted local network.

## 5. Set Wasp parameters while disarmed

| Control | Choices | Default |
| --- | --- | --- |
| Launch throttle | 25%, 50%, 100% | 100% |
| Straight launch | Off, 0.5 s, 1 s, 2 s | 1 s |
| Steering blend | 0.25 s, 0.5 s, 1 s | 0.5 s |
| Maximum total flight time | 1 s, 2 s, 3 s, 5 s | 2 s |
| Maximum steering rate | 30, 60, 90°/s | 60°/s |
| Steering gain | 2, 4, 6 | 4 |
| Target-motion lead | 0, 80, 150 ms | 80 ms |
| Camera image rotation | 0, 90, 180, 270° | 0° |
| Detection → Ground rejection (sky gate) | On / Off | On at startup |
| Detection → Follower | On / Off | Saved setting |

Total flight time must exceed straight launch plus blend time. Increase total time
first for longer intervals. Wasp launch settings last until the service restarts;
refreshing the page preserves them. Select the mode and confirm the camera after
restarting the service.

## 6. Check the launch sequence with propellers removed

1. Select Wasp **while disarmed**. Confirm 90°. Set throttle to **25%** for this
   command check, straight launch to **1 s**, blend to **0.5 s**, and total to **2 s**.
2. Secure the assembly and aim at a target the detector recognises near image centre.
3. Keep transmitter throttle low, select **ACRO**, then arm using the transmitter.
   Arming may idle the motors. It must not start a Jetson launch.
4. Wait for **LAUNCH WASP** to become available. Do not apply full throttle manually.
5. Press **LAUNCH WASP** once. Check this sequence:
   - Immediately: selected throttle request, zero commanded turn rate.
   - First 1 s: **STRAIGHT** countdown.
   - Next 0.5 s: **BLENDING STEERING** toward the current tracked target.
   - Then: **GUIDING**, with the same throttle request.
   - At 2 s total: overrides released, **DONE · DISARM**.
6. Verify steering direction. At image rotation 0°, image right corresponds to body
   right and image bottom to the nose. A right-side target requests positive roll;
   a below-centre target requests negative pitch.
7. Repeat the check for **ABORT**, leaving ACRO, and target/telemetry loss, including
   during the straight interval. All overrides must release without relaunching.
8. Disarm before another attempt. Verify reboot starts the UI but never arms or launches.

## 7. Before a flight attempt

- Complete the actual-airframe checks above and establish reliable mosquito detection.
- Verify motor directions, camera alignment, platform clearance, FC tune and pilot recovery.
- Use a clear test area and validate takeoff/recovery before attempting pursuit.
- Keep the transmitter ready. Abort/timeout returns control to the transmitter;
  it does **not** automatically disarm, hover, brake or land.
- Throttle percentages are commands, not measured thrust/speed. 25% may not lift
  the aircraft. Zero commanded turn rate does not guarantee a straight ground path.
- Fresh detections are required before launch and throughout the attempt. Loss for
  more than 250 ms ends it; Follower alone cannot keep it running.

**PC software tests are complete; Jetson boot and real flight remain unverified
until checked on your hardware.**
