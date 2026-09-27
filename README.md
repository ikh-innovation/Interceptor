# WASP — vision-guided interceptor drone

WASP runs on an NVIDIA Jetson mounted on a multirotor. It watches a forward camera, detects a target with YOLO, and flies the drone into it. It sends stick commands to an ArduPilot flight controller over MAVLink.

It can chase two kinds of target:

| Mode | Target | Camera mount | Detector |
|---|---|---|---|
| `human` | a person's head/face on the ground | 10° up | `iitolstykh/YOLO-Face-Person-Detector` (HuggingFace, cached) |
| `sky` | another drone in the air, above | 30° up | local single-class model, `runs/detect/v0/weights/best.pt` |

The airframe has no GPS. Guidance uses only the camera image, the flight controller's attitude and its barometric altitude.

> **Safety.** The intercept law is built to make contact. Near the target it deliberately freezes the controls and flies through. The pilot keeps arming, the mode switch and the transmitter the whole time, and the software drives nothing unless the pilot has selected the steering mode (see [Safety model](#safety-model)).

---

## How it works

```
IMX219 camera ──► tracker.py ──► detection (bbox, confidence)
                    │
                    ▼
               fc_control.py ──► InterceptLaw: yaw / roll / pitch / throttle
                    │
                    ▼
                mission.py   ──► HOVER → SCOUT → PUSH → AI → STOP
                    │
                    ▼
         RC_CHANNELS_OVERRIDE @ 50 Hz ──► ArduPilot (ALT_HOLD)

mjpeg_stream.py ──► phone browser: live seeker view + mission controls
```

### Detection — `tracker.py`
- Captures the CSI camera through GStreamer (`nvarguscamerasrc`): IMX219 **sensor mode 3, 1280×960 @ 30 fps**. That is the full sensor, 62.2° × 48.8°, which is what the control law assumes. Mode 4 (1280×720) is a centre crop and must not be used.
- Runs YOLO. A TensorRT `.engine` next to the weights is used when present (≈28 ms/frame against ≈35 ms for PyTorch on this Jetson).
- Sky gate: a colour-based sky mask, so detections on the ground are rejected in sky mode.
- Adaptive zoom: a small, distant target is searched in a native-resolution crop of the sky at a higher inference size. A close target drops to a cheap low-resolution ROI pass.
- The detector is called once per frame. While coasting, the ROI search and the sky search take turns.
- Recording: detection-gated MP4 written in 20 s segments (`REC_SEGMENT_S`), so pulling the battery loses at most the last segment. Everything is written to `src/data/`: `<stamp>.mp4`, `<stamp>.log` and `control_<stamp>.csv`. The Jetson has no RTC, so without network time the stamp is `nosync_rNNNN_HHMMSS` (a persistent run counter), so runs never overwrite each other.

### Guidance — `fc_control.py`
There are two control laws. They solve different problems and are not retunes of each other:

- **`InterceptLaw`** (default, free flight, ALT_HOLD). Each channel has exactly one job:

  | Channel | Job |
  |---|---|
  | yaw | keep the camera pointed at the target |
  | roll | proportional navigation on the line-of-sight rate: bank to turn the velocity vector and lead a crossing target |
  | pitch | closing speed; nose-down lean only, never positive |
  | throttle | close the vertical gap (elevation error, plus a lead term in sky mode) |

  ALT_HOLD is used because a centred throttle there means "hold altitude". That makes it safe for the software to own the throttle and hand it back.
- **`ControlLaw`** (`--gimbal`, ACRO). A visual servo that nulls the pixel error with yaw and pitch body rates. It is only correct on the bench gimbal rig, where the airframe cannot move.

Two implementation details matter a lot:
- The line-of-sight bearing is one world-referenced signal (`yaw + azimuth`), differentiated once.
- Attitude is interpolated to the frame's capture timestamp.

Both are there so the seeker's own rotation is never mistaken for target motion.

### Mission flow — `mission.py`
The pilot arms and takes off in ALT_HOLD, then hovers with the throttle centred. From there the phone app drives it:

| Phase | What happens |
|---|---|
| **HOVER** | Jetson sends nothing; the pilot's sticks are live |
| **CLIMB / DESCEND** | while ▲/▼ is held: throttle only |
| **SCOUT** | while ⟲/⟳ is held (a tap = a small step): yaw only |
| **LOCK** *(a flag, not a phase)* | the heading is held where LOCK was pressed (or at a heading picked on the degree scroll), with no time limit, until ENGAGE, ABORT, LOCK again or the pilot's switch |
| **PUSH** | ENGAGE: fixed nose-down lean, heading and height held. From the first confirmed detection the AI steers yaw/roll/throttle |
| **AI** | after 1.5 s of PUSH the InterceptLaw also takes the lean (blended in over 0.4 s) |
| **STOP** | soft stop: eased back to level over 1 s, then every channel is released |

Behaviour worth knowing:
- **Bumpless takeover.** Each channel starts from the pilot's own stick value, and every change is a smoothstep ramp.
- **Dead-man.** A held button is a stream of keepalives. If none arrives for 0.3 s, the axis ramps back to zero. A lost message stops the drone instead of flying it away.
- **Target lost during AI.** The drone goes back to the straight PUSH and searches again for 2.5 s. If nothing is confirmed within the search time of PUSH (4 s by default, ☰ → PARAMETERS), it does a soft stop.
- **Terminal commit.** In the last metre the target fills the frame and the detector drops out. The last command is held for about 1 s, then the drone does a soft stop.
- **AI floor.** PUSH and AI never descend below 3 m above the arming point.
- **Speed levels** (SLOW / MEDIUM / FAST) are set separately for vertical, yaw, push and AI, in ☰ → PARAMETERS.
- **Search time after ENGAGE** (☰ → PARAMETERS → SEARCH): **4 s** (default), 6 s or 8 s. This is how long the PUSH looks for a target before the soft stop. Longer means farther out: at PUSH 20° that is about 19 / 42 / 69 m (at 30°, 28 / 61 / 97 m).
  - **Defaults:** vertical SLOW, yaw MEDIUM, **PUSH SLOW (20°)**, AI MEDIUM.
  - **AI levels in m/s:** SLOW / MEDIUM / FAST mean ≈11 / 15 / 16 m/s of cruise (the max lean of 12° / 20° / 22°) and a climb ceiling of 8 / 15 / 25 m/s.
- **What the settings buy** (`tests/sim_sky_wind.py --speed all --push …`, a moving target, incl. at 45°, in gusty wind): touch rate on moving targets in Bf 2–3.

  | AI → | SLOW | MEDIUM | FAST |
  |---|---|---|---|
  | PUSH 20° (default) | 48% | **73%** | 69% |
  | PUSH 30° | 44% | 66% | 61% |
  | PUSH 40° | 32% | 48% | 45% |

  AI MEDIUM is the best level at every push. A gentler push leaves the AI more frames. The weak spots at any setting: a head-on target at 10 m/s, and a zigzagging one.

#### LOCK — hold the heading
LOCK holds the **heading**, meaning which way the nose points. It does **not** hold the position: there is no GPS, so in ALT_HOLD the wind still pushes the drone sideways. It keeps looking north while it drifts.

How it behaves:
- **Gentle on purpose.** The loop runs on the FC's `ATTITUDE.yaw` at 50 Hz. A correction is capped at **15°/s**, and LOCK changes its own turn rate by at most **30°/s²**, so it never steers quickly. In the sim the drone's yaw motion is the wind's, not LOCK's: the RMS yaw rate is the same with a 30°/s or a 15°/s cap, and the heading holds just as well.
- **The FC gets the first go.** ArduPilot's own heading hold already pulls back from a gust. LOCK acts only once the heading has been **still for 1 s at the wrong value**. That is the case the FC cannot fix: a gust stronger than the yaw authority drags the FC's own heading target, and the FC then holds the wrong heading. Correcting during the swing instead fights the FC and overshoots.
- **No chatter.** On a noisy heading, the smoothed error decides whether to correct, with hysteresis (on at 1°, off at 0.4°). The actual error sets how much.
- **Moving the drone while locked.** ▲▼ work while locked, and the heading stays. ⟲/⟳ (held, or a tap) also turn the drone. Let go and it turns back slowly to the **locked** heading. The lock itself never moves with them.
- **Changing the lock.** Use the **heading bar** under the picture. Drag it; it follows your finger like the compass tape, so dragging left = more degrees. Nothing is sent while you drag. Lift your finger and a pop-up asks **LOCK AT 313°?**
  - **YES** sends `lock=313`. The drone turns there slowly (15°/s at most) and holds it. This works whether or not it was already locked.
  - **NO** sends nothing.
- **Repeats are safe.** `lock=on` while already locked changes nothing, so a double tap or a WiFi retry cannot move it.
- **Clearing.** ENGAGE clears LOCK, and the AI takes the yaw bumplessly. ABORT and the pilot's mode switch clear it too.
- **Heading lost.** If ATTITUDE goes stale for more than 0.5 s, the yaw command is exactly zero until it returns. LOCK never corrects on an old heading.
- **The pilot's yaw stick.** While LOCK is on, the Jetson holds the yaw channel, so the pilot's yaw stick does nothing. The mode switch still takes everything back.
- **Heading estimate.** LOCK holds what the FC *believes* the heading is. Without a compass, that estimate can drift slowly, and nothing on the Jetson can see that.

What the simulation shows (`tests/sim_lock_wind.py`, 600 hovers of 60 s, gusty Bf 0–3, three airframe yaw sensitivities):
- **Normal conditions.** ArduPilot alone already holds within ~1.5°, and LOCK stays out of the way (same numbers).
- **Stronger gusts than the yaw motors.** At Bf 3 on a weathervane-prone airframe, the FC alone finished more than 10° off in **15/25** runs. With LOCK it was **3/25**. These gusts overpower the motors, so no controller can hold through them. LOCK walks the heading back once the gust lets go.

### Phone app — `mjpeg_stream.py` + `mission_ui.html`
The Jetson flies headless. A phone or laptop on the field WiFi opens:

| URL | What |
|---|---|
| `http://<jetson-ip>:8080/` | live annotated seeker view + mission controls |
| `/snapshot.jpg` | a single frame |
| `/state` | JSON state (phase, detector, fps, latency) |
| `/cmd` | commands (no PIN; one phone at a time holds control) |

The look is light grey and white with black controls; orange is used only as the accent. The page layout, top to bottom:
- **Top bar:** ☰ top-left opens the **MENU** with three sections, each a full-screen page with ✕: **PARAMETERS** (all the SLOW / MEDIUM / FAST settings and the search time), **MESSAGES** (what the Jetson reports) and **TEST**. The whole app is in English. There is no message bar. A refusal shows for 4 s over the bottom of the picture. The only pop-up is the heading bar's YES / NO.
  - ☰ → **TEST** → START TEST (only while the drone is disarmed): the same screen drives a pretend drone on the Jetson. It runs the real mission code (`desk_sim.py`) against a model of ArduPilot's yaw hold and gusty wind, and never touches the FC. A striped bar offers GUST ⟲/⟳ (a gust stronger than the motors), WIND 0–3, ↺ and EXIT. If the real drone arms, the page leaves test mode at once.
  - The lock state is always shown at the top: **UNLOCKED** (grey) or **LOCKED 313°** (orange). The phone never vibrates.
- **Status:** the status tiles (FC mode, arming, heading, target, vibration, detector fps). There is no phase card, so the picture gets the room. The phase shows on ENGAGE (PUSH / AI ✓ / DONE ✓) and the lock state in the top bar.
- **Picture:** the 4:3 picture, as wide as the screen. Under it are the **heading bar** and, on its right, a box with the **height** and climb rate, in the same style. When idle, the bar follows the heading and shows the locked heading as a white notch. While you drag it, the picked heading also shows as an orange line on the compass tape.
  - A compass tape runs along its top. Its width is the camera's 62° field of view, and it carries a white mark at the LOCKed heading.
  - A thin dashed aim arc runs corner to corner and turns green when the target is on it.
- **Controls:**
  - The ▲▼⟲⟳ pad sits at the left. A held button turns orange, and pulses while the Jetson is accepting its keepalives, so you can see the drone is being told. It turns red if the command is refused.
  - On the right, set on a diagonal like a Game Boy's B / A, are **LOCK** (low-left; white, with an orange ring and the heading when on) and **ENGAGE** (high-right; red).

The target is always a drone; there is no in-air HUMAN/DRONE switch. The Ubuntu typeface is served from `src/fonts/` by the Jetson itself, since the field has no internet.

There is **no PIN**: any phone on the field WiFi can command, one at a time (the controller lease). The field WiFi is the only gate, so keep it private. Frames are JPEG-encoded only while someone is watching.

---

## Hardware

- NVIDIA Jetson Orin Nano (JetPack: CUDA, TensorRT, GStreamer OpenCV)
- IMX219 CSI camera on `/dev/video0`, sensor mode 3, on a **measured, fixed up-tilt** (30° for sky mode)
- ArduPilot Copter flight controller over USB serial. The port is auto-discovered, and `SYSID_MYGCS` is read from the FC at connect. Betaflight/INAV over MSP is also supported by `fc_control.py`.
- RC transmitter. An optional spare channel (`COMMIT_CHAN`) gives a lean dial and a ~100 ms abort.

## Repository layout

```
src/
  run.py              single entry point: preflight, FC discovery, presets, launches tracker
  tracker.py          camera, detection, sky gate, zoom, recording, main loop
  fc_control.py       control laws, MAVLink/MSP transports, stick/PWM mapping
  mission.py          Hover → Scout → Engage state machine (pure logic, testable)
  mjpeg_stream.py     HTTP server: seeker stream, state, phone commands
  mission_ui.html     the phone app
  flight_config.py    THE file to edit: mode, mount tilt, target width, gains
  flight_report.py    post-flight analysis from the control CSV
  start-wasp.sh       boot launcher (reads flight_config.py)
  start-wasp-sky.sh   manual sky-mode launcher with a mount-tilt confirmation
  wasp-tracker.service  systemd user unit: starts at power-up
  tests/              offline simulations, fake FC, bench diagnostics
  data/               run logs, videos, CSVs (generated)
runs/detect/          sky-detector training run and weights
docs/                 non-technical explainer (Greek)
*.pt / *.engine / *.onnx   model weights
```

## Setup

On a new Jetson, open Claude Code in the repo and have it follow **[INSTALL_PROMPT.md](INSTALL_PROMPT.md)**. It checks and fixes the environment, TensorRT engine, camera mode, FC link and service, then prints a report.


Python dependencies live in the virtualenv `.env/yolo`. It must see the system site-packages so that JetPack's GStreamer OpenCV and TensorRT are available. It needs:

- `torch` (CUDA build — the CPU build runs but gives ~0.3 fps)
- `ultralytics`, `huggingface_hub`, `numpy`
- `pymavlink`, `pyserial`

The field has no internet. Download the face detector into the HuggingFace cache once, while online. The launchers set `HF_HUB_OFFLINE=1` and `YOLO_AUTOINSTALL=false`.

Enable auto-start at power-up:

```bash
systemctl --user enable --now wasp-tracker.service
tail -f ~/wasp-logs/wasp-tracker.log
```

Starting the service does **not** give the AI control. It waits, idle, until the pilot arms and selects ALT_HOLD.

## Configuration

Edit **`src/flight_config.py`** and reboot:

```python
MODE = "sky"            # or "human"
MOUNT_TILT_DEG = 30     # the camera's REAL measured angle
TARGET_WIDTH_M = None   # real target width in metres (drives range + terminal freeze)
MAX_LEAN_DEG = None     # None = mode default; moves together with ROLL_PN_GAIN
ROLL_PN_GAIN = None     # ≈ 3 × cruise speed
COMMIT_CHAN  = 0        # spare RC channel for lean dial + fast abort, 0 = none
```

```bash
python3 src/flight_config.py            # show and check the active mode
```

The file **refuses to start** if the mode's required tilt disagrees with `MOUNT_TILT_DEG`. A wrong tilt raises no error; the drone just flies consistently high or low by exactly that error.

For a one-off override, export a variable instead of editing the file:

```bash
MAX_LEAN_DEG=16 ROLL_PN_GAIN=48 ./src/start-wasp.sh
```

## Running by hand

```bash
cd src
./run.py                  # intercept preset (default)
./run.py --gimbal         # bench gimbal preset (ACRO visual servo)
./run.py --no-wait        # fail at once if the FC is not up
./run.py --router         # put MAVProxy in the path (UDP 14550 / 14551)
./run.py --no-verify      # skip the RC-override acceptance check
./run.py --stream-port N  # move the phone app off 8080
./start-wasp-sky.sh       # sky mode, asks you to confirm the camera tilt
```

`run.py` re-execs itself under the right interpreter and waits for an FC heartbeat, so it can start before the drone is powered. It identifies the firmware, checks that RC overrides are really accepted (by watching the FC's own `RC_CHANNELS` echo), and stops on the first preflight failure with the reason.

`TRACKER_HEADLESS=0` brings back the local OpenCV window for bench work.

## Safety model

- **Mode gate.** Nothing is driven unless the FC reports a mode in `FC_STEER_MODES` (ALT_HOLD for intercept). The pilot's mode switch is the AI's on/off switch.
- **Arm gate.** Nothing is driven while disarmed (`REQUIRE_ARMED=1`), or before the FC reports the drone in the air.
- **Commit channel.** If `COMMIT_CHAN` is wired, it scales the lean and aborts in ~100 ms. Until the first RC frame is read, commit is forced to 0.
- **Release.** On target loss, stop, crash or exit, every overridden channel goes back to the pilot.
- **Throttle at centre.** The pilot's throttle stick must stay centred while the software drives. In ALT_HOLD a stick left at the bottom means "descend at max" the moment control is handed back.
- **Sky mode is blind below the horizon.** It closes up to about 35° of elevation and does not close above ~45°.

## Tests

Most tests are plain scripts that print PASS/FAIL. They need no FC and no camera. Run them from `src/`:

```bash
cd src
../.env/yolo/bin/python tests/test_intercept.py       # closed-loop intercept sim
../.env/yolo/bin/python tests/test_mission.py         # mission state machine + PWM map
../.env/yolo/bin/python tests/test_mission_mavlink.py # end to end against a fake ArduPilot
```

| Test | Covers |
|---|---|
| `test_control_law.py`, `test_intercept.py`, `test_acro_dual.py`, `test_schedule.py`, `test_yaw_circles.py` | control-law signs, limits, closed-loop sims, wind, gain schedule |
| `test_mission.py`, `test_mission_mavlink.py`, `test_field_sequence.py`, `test_simulated_fc.py` | mission logic and MAVLink plumbing against `fake_fc.py` |
| `test_recording.py`, `test_sky_mask.py` | segmented recording, sky gate at half resolution |
| `test_ui_browser.py` | the phone app in headless Firefox |
| `test_hold_wifi.py` | a held ▲▼⟲⟳ over realistic field WiFi: no stutter, and letting go stops |
| `test_lock.py` | LOCK under bad conditions: noisy/late heading, across north, heading lost mid-lock, 10 min in gusts, weak yaw motors; locked at 313° through a gust stronger than the motors, steady torque, ⟲/⟳/▲ while locked; never over 15°/s or 30°/s²; the phone's command path |
| `ui_demo.py` | the phone app on a desk (no drone): `tests/ui_demo.py`, open the printed link, then ☰ → TEST |
| `sim_lock_wind.py` | LOCK vs the FC alone, 600 gusty hovers (not pass/fail) |
| `sim_sky_wind.py` | sky-mode campaign: moving target × gusty wind Bf 0–3, touch odds and turn overshoot (not pass/fail) |
| `fc_check.py`, `check_modes.py`, `cameraTest.py` | bench diagnostics against real hardware |

## After a flight

```bash
cd src
../.env/yolo/bin/python flight_report.py              # newest control CSV
```

It gives a one-page answer to three questions:
- Did the FC fly what was commanded (tracking error, lag)?
- How old was the picture each command was built from?
- Was vibration shaking the camera?
