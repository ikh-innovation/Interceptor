# WASP — vision-guided interceptor drone

WASP runs on an NVIDIA Jetson mounted on a multirotor. It watches a forward camera, detects a target with YOLO, and flies the drone into it. It sends stick commands to an ArduPilot flight controller over MAVLink.

It has one mode:

| Mode | Target | Camera mount | Detector |
|---|---|---|---|
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
| **AUTO_CLIMB** | AUTO: straight up at 3 m/s to 10 m, heading locked, watching for a target |
| **AUTO_HOLD** | at 10 m (or where it was, if already higher), heading locked, watching, with no time limit |
| **AUTO_ALIGN** | the first confirmed detection: yaw and height put the target on the aim mark |
| **PUSH** | ENGAGE: fixed nose-down lean, heading and height held. From the first confirmed detection the AI steers yaw/roll/throttle |
| **AI** | after 1.5 s of PUSH the InterceptLaw also takes the lean (blended in over 0.4 s) |
| **STOP** | soft stop: eased back to level over 1 s, then every channel is released |

#### AUTO — up, find, line up, go
One press. Leg one goes straight up; leg two is a straight line at the target, then the AI.
- **Leg one.** It climbs at 3 m/s and slows into 10 m, with the heading locked by the same loop as LOCK. If nothing is detected it stays there, heading locked, until you act. Pressed above 10 m, it holds where it is. There is no GPS, so it holds heading and height; the wind still drifts it, as with LOCK.
- **Lining up.** The first confirmed detection stops the climb. Yaw turns the target to the centre (at up to 30°/s, rate-limited, and also following the target's own drift), and height puts it on the aim mark: −min(PUSH, 20)° in the picture, so it is centred once the push lean is on. Nothing moves on a detection older than 0.15 s. Lining up never descends below 3 m. If the target is lost, AUTO goes back to watching (climbing on to 10 m if it was not there yet).
- **Ready.** It goes when the target has been within 5° left/right and 5° above or 12° below the mark for 0.25 s. It also goes early when the target is closing fast (under 3 s to go, held for 0.5 s; this uses the range estimate, so measure `TARGET_WIDTH_M`), or after 2 s of lining up if the target is within 15° of the nose.
- **Leg two.** A straight line at AUTO's own lean (default 30°), starting from the yaw and throttle it was already flying: the lean ramps in over 1.2 s; the AI blends in yaw / roll / throttle over 0.4 s, then takes the lean at 1.5 s, at AUTO's own AI level if one is set (MAX = the fastest, ~21 m/s). Holding the AI off for longer bought no smoothness in the sim and cost hits.
- **Looking around.** While it climbs or watches, ⟲ / ⟳ turn it (held, or a tap = one step); the heading lock follows and holds wherever it stops. ▲▼ and LOCK are refused during AUTO, and ⟲ / ⟳ too once it is lining up.
- **Stopping it.** AUTO again = a soft stop, then the pilot's sticks. ENGAGE during AUTO = go now. ABORT and the pilot's mode switch work as everywhere else.
- **ON / OFF.** ☰ → PARAMETERS → AUTO. **OFF at every power-up**; while OFF the AUTO button is greyed out. Switching it OFF while AUTO flies stops AUTO (soft).
- **Settings** (☰ → PARAMETERS → AUTO; saved in `~/.config/wasp/auto.json`, kept after a reboot, refused while AUTO flies; `mission.Mission.AUTO_SETTINGS` is the list):

  | Setting | Choices (default **bold**) | What it does |
  |---|---|---|
  | HEIGHT | 5 · 10 · **15** · 20 m | leg one climbs to this |
  | CLIMB SPEED | 2 · **3** · 5 m/s | leg one's climb |
  | TURN SPEED | 20 · **30** · 45 °/s | lining up: the fastest turn |
  | HEIGHT ADJUST | 3 · **5** · 8 m/s | lining up: the fastest climb |
  | READY WINDOW | ±3 · **±5** · ±8° | how close to the aim mark counts as lined up |
  | HOLD, THEN GO | **0.25** · 0.5 · 1 s | lined up this long, then go |
  | GO ANYWAY AFTER | 1 · **2** · 3 · 5 s | not lined up by then but near the nose: go |
  | GO AT ONCE IF CLOSING | OFF · 2 · **3** · 4 s | that close in time: go at once |
  | LOWEST | 2 · **3** · 5 m | lining up never descends below |
  | STRAIGHT-LINE LEAN | = PUSH · 20 · **30** · 40° | leg two's lean |
  | AI SPEED | = AI · MEDIUM · FAST · **MAX** | the AI level the run hands over to |
  | AI STEERS AFTER | 0 · 0.3 · **0.6** s | the AI's first steering after the line starts |
- **In TEST** (☰ → TEST) AUTO is ON from the start and one press shows the whole thing on the drawn world. The enemy flies in from the right (an arrow at the edge shows where), so pressed early AUTO is seen to climb, then WATCH, then find it, line it up, fly the straight line, hand over to the AI, and hit (~19 s). A numbered caption over the picture says which step it is on (1 climbing · 2 watching · 3 lining up · 4 straight line · 5 the AI · HIT), and the AUTO button stays orange through the whole run (UP 10 m / WATCHING / LINING UP / GO / AI). The captions are TEST only; the real picture never shows them. The pretend detector sees the enemy whenever it is in view, with the flight sim's odds, so ENGAGE in TEST now goes PUSH → AI too.
- **From the horizon** (`SIM_START=horizon`, 864 engagements per row: the enemy within 15° of the nose, coming in from 60–80 m at 0 / 30 / 60 / 90° off the line to us at 5 / 10 / 15 m/s, or going away from 20–40 m at 0 / 30 / 60° at 5 / 10 m/s, low 5–12 m / mid 15–25 m / high 30–45 m up, Bf 0–3; a hit is < 0.5 m). Against the pilot's own flow, hovering at 10 m and pressing ENGAGE 0.7 s after it shows on screen:

  | | pilot ENGAGE, AI MEDIUM | pilot ENGAGE, AI MAX | AUTO line 30, AI MEDIUM | **AUTO line 20, AI MAX** | AUTO line 40, AI MAX |
  |---|---|---|---|---|---|
  | all | 30.6% | 34.0% | 37.8% | **39.7%** | 34.0% |
  | coming straight in (0°) | 26% | 38% | 38% | **50%** | 41% |
  | coming in at 30° | 22% | 20% | 37% | 36% | 29% |
  | coming in at 60° | **25%** | 22% | 15% | 14% | 14% |
  | crossing (90°) | **15%** | 15% | 1% | 1% | 1% |
  | going away (0 / 30 / 60°) | 47 / 47 / 51% | 54 / 55 / 53% | 68 / 66 / 72% | **64 / 66 / 77%** | 58 / 53 / 68% |
  | low target (5–12 m) | 6% | 5% | 34% | **38%** | 34% |
  | mid (15–25 m) | 55% | **57%** | 51% | 48% | 51% |
  | high (30–45 m) | 31% | **40%** | 29% | 34% | 17% |
  | coming at 5 / 10 / 15 m/s | 51 / 12 / 2% | 50 / 17 / 5% | 45 / 20 / 2% | 45 / 22 / 8% | 42 / 17 / 5% |

  **With the defaults (15 m, line 30°, AI MAX, AI after 0.6 s)** against no AUTO at AI MAX (`SIM_START` horizon / hover 4–8 m / ground 1.5 m): horizon **35.0%** vs 34.0% (39.0% with the AI at 0 s); hover 66.0% vs **67.7%**; ground **68.4%** vs 63.6%. AUTO wins going away (68 / 61 / 73% vs 54 / 55 / 53%) and against low drones (34% vs 5%); no AUTO wins crossing (15% vs 1%), mid / high drones coming in, and a drone coming at 10 m/s from a hover (52% vs 15%). AUTO hits ~2 s later and hit the ground in 1.8–2.4% of runs (AI MAX chasing a drone down; no AUTO 0.5–0.9%).

  Most of AUTO's gain is **where it starts**: from the ground the 30°-up camera sees a low drone that a pilot hovering at 10 m is blind to (seen 72% vs 14%). Started at 10 m like the pilot, AUTO hits 28.4%. Its weak spot is the **crossing** drone: lining up chases one that keeps sliding sideways, and the pilot's ENGAGE (straight at once, the AI steering from the first frame) does far better there. Nothing catches a drone coming in at 15 m/s often (≤ 8%). Settings: PRECISE (±3°, 0.5 s, 3 s) 29.1%, QUICK (±8°, 1 s, closing 4 s) 35.8%, height 5 / 20 m 38.0%, same as 10.
- **In the sim** (`SIM_AUTO=1`, 960 engagements, AI MEDIUM, PUSH 30°): hovering 4–8 m up, AUTO hits **76%** against ENGAGE's 70%. From 1.5 m, with the target 25–50 m out and 6–25 m up, it hits **79%** against 61% (`SIM_START=ground`). Head-on targets improve most (71% against 49 / 29%). The cost is time: the hit comes at ~7–8 s, not 5–6.

Behaviour worth knowing:
- **Bumpless takeover.** Each channel starts from the pilot's own stick value, and every change is a smoothstep ramp.
- **Dead-man.** A held button is a stream of keepalives. If none arrives for 0.3 s, the axis ramps back to zero. A lost message stops the drone instead of flying it away.
- **Target lost during AI.** The drone goes back to the straight PUSH and searches again for 2.5 s. If nothing is confirmed within the search time of PUSH (6 s by default, ☰ → PARAMETERS), it does a soft stop.
- **Terminal commit.** In the last metre the target fills the frame and the detector drops out. The last command is held for about 1 s, then the drone does a soft stop.
- **No ground check.** Neither ▼ nor PUSH / AI is stopped near the ground (removed 2026-09-30: the field flights were low and the old floors blocked the descent). The pilot's mode switch is the only guard against the ground.
- **Speed levels** (SLOW / MEDIUM / FAST, and MAX for the AI) are set separately for vertical, yaw, push and AI, in ☰ → PARAMETERS.
- **Search time after ENGAGE** (☰ → PARAMETERS → SEARCH): 4 s, **6 s** (default) or 8 s. This is how long the PUSH looks for a target before the soft stop. Longer means farther out: at PUSH 20° that is about 19 / 42 / 69 m (at 30°, 28 / 61 / 97 m).
  - **Defaults: everything MEDIUM** (the field's best, 2026-09-30): height 3 / 1.5 m/s, turn 20°/s, **PUSH 30°**, AI MEDIUM, **search 6 s**.
  - **AI levels in m/s:** SLOW / MEDIUM / FAST mean ≈11 / 15 / 16 m/s of cruise (the max lean of 12° / 20° / 22°) and a climb ceiling of 8 / 15 / 25 m/s.
  - **AI MAX** (AI only, a 4th button): 35° max lean, ≈21 m/s in the sim, climb ceiling 25 m/s. It eases off to 30% of that lean as the target grows from ~0.7% to 3% of the frame width (~35 m to ~8 m for a 0.30 m quad), because at 21 m/s the detector's ~0.1 s delay is ~2 m of miss. That ease-off took it from 62% to 68% hits. It is the best level against targets coming at you (sim, PUSH 30°: 82% against MEDIUM's 49%) and the fastest (hit at ~4.5 s), but overall it is not more accurate than MEDIUM.
  - **Why not 25 m/s:** in the sim's drag model, 25 m/s needs about 45° of lean, more than ANGLE_MAX 40 allows. ANGLE_MAX 50 bought nothing in the sim; a bigger turn gain hurt (55 → 59%, 62 → 54%); a camera tilted further up (40–45°) made every AI level worse (−15 to −25 points), because the targets start 5–25° above the horizon.
- **What the settings buy** (`tests/sim_sky_wind.py`, 960 engagements per cell: 12 target motions × Bf 0–3, total tilt held to ANGLE_MAX as the FC does, 2026-10-06). Touch rate:

  | ENGAGE, hovering 4–8 m, target in frame | AI MEDIUM | AI FAST | AI MAX |
  |---|---|---|---|
  | PUSH 20° | **77%** | 73% | 75% (head-on 92%) |
  | PUSH 30° (default) | 70% | 63% | 68% |
  | PUSH 40° | 60% | 58% | 56% |

  | AUTO, from 1.5 m, target 25–50 m out, 6–25 m up | AI MEDIUM | AI FAST | AI MAX |
  |---|---|---|---|
  | PUSH 20° | **80%** | 76% | 75% |
  | PUSH 30° (default) | 78% | 75% | 74% |
  | PUSH 40° | 69% | 69% | 73% |

  A gentler push leaves the AI more frames: PUSH 20° is the best at every AI level. AI MEDIUM is the most accurate; MAX is the fastest and the best head-on. The weak spot at any setting is a zigzagging target (~15%).

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

The look is light grey and white with black controls; orange is used only as the accent. The same page fits every screen:
- **Phone:** everything in one column, as described below. The pad and LOCK / ENGAGE each sit on a dark plate.
- **Tablet or desktop (700 px wide and up):** one line on top with ☰, the name and the six status cells. The picture sits under it, as big as it fits, with the heading bar and height box under the picture. The controls are **inside** the picture: the ▲▼⟲⟳ pad bottom-left and LOCK above ENGAGE bottom-right, each on a dark glass plate so they read against any sky. The four arrows are equal squares, sized to a tenth of the picture height. The script moves the controls into the picture only at these widths; they behave exactly as on the phone.

The page layout on a phone, top to bottom:
- **Top bar:** ☰ top-left opens the **MENU** with three sections, each a full-screen page with ✕: **PARAMETERS** (four cards, each showing what is set and opening its own page: FLIGHT ▲▼ ⟲⟳, ENGAGE push / AI / search, AUTO with all its settings, DETECTION sky gate / follower), **MESSAGES** (what the Jetson reports) and **TEST**. The whole app is in English. There is no message bar. A refusal shows for 4 s over the bottom of the picture. The only pop-up is the heading bar's YES / NO.
  - ☰ → **TEST** → START TEST (only while the drone is disarmed): the same screen drives a pretend drone on the Jetson. It runs the real mission code (`desk_sim.py`) against a model of ArduPilot's yaw hold and gusty wind, and never touches the FC. The picture is a drawn world with no words in it, so a newcomer gets it at a glance:
    1. You start **on the ground**, facing 110°, with tree tops at the bottom of the picture. ▲ climbs toward the clouds, and the trees sink away.
    2. The **enemy drone** hangs at 140°, 60 m out and higher up: a dark quad with spinning rotors, a blinking red light and red brackets. While it is out of view, a red arrow at the frame's edge points to it. Turn (⟲⟳, or the heading bar) to find it.
    3. A dotted line runs from the aim point to it. Climb until the enemy sits **under the aim line**: the line and the brackets turn green. Then press **ENGAGE**.
    4. The nose tips down, the ground rushes by, and the enemy grows as you close. While it is in view, the pretend airframe chases it the way the AI would. (The real mission never gets a detection from it, so the phone shows PUSH.)
    5. It ends in a **burst** and a green ✓. A few seconds later the test starts over from the ground. A miss starts over too.

    The geometry is real (`desk_sim.py`: position, height, lean, the camera's 30° tilt and FOV), so climbing moves the enemy and the clouds down the picture, turning moves them across, and the PUSH lean tilts the whole view. The test controls are inside the picture: two small wind keys top-left (a gust stronger than the motors, pushing the nose left or right) and a black ✕ EXIT top-right. The picture also gets a black frame, so the test is never mistaken for the real drone. If the real drone arms, the page leaves test mode at once.
- **Status:** the status tiles (FC mode, arming, heading, target, vibration, detector fps). There is no phase card, so the picture gets the room. The phase shows on ENGAGE (PUSH / AI ✓ / DONE ✓), and the lock state on the LOCK button and as a white notch on the heading bar. The top bar holds only ☰ and the name.
- **Picture:** the 4:3 picture, as wide as the screen. Under it are the **heading bar** and, on its right, a box with the **height** and climb rate, in the same style. When idle, the bar follows the heading and shows the locked heading as a white notch. While you drag it, the picked heading also shows as an orange line on the compass tape.
  - A compass tape runs along its top. Its width is the camera's 62° field of view, and it carries a white mark at the LOCKed heading.
  - A thin dashed aim arc runs corner to corner and turns green when the target is on it.
- **Controls:**
  - The ▲▼⟲⟳ pad sits at the left. A held button turns orange, and pulses while the Jetson is accepting its keepalives, so you can see the drone is being told. It turns red if the command is refused.
  - On the right, one above the other, are **AUTO** (a grey pill; orange while AUTO flies, with UP 10 m / WATCHING / LINING UP under it), **LOCK** (white; an orange ring and the heading when on) and **ENGAGE** (red), 74 px round.

The target is always a drone. The Ubuntu typeface is served from `src/fonts/` by the Jetson itself, since the field has no internet. So is the IKnowHow Defence logo (`src/img/ikd-logo.svg`, at `/img/ikd-logo.svg`). It sits at the right end of the top line, and is also the browser-tab icon and the sign-off at the bottom of the MENU page.

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
  img/                the IKnowHow Defence logo, served to the app
  flight_config.py    THE file to edit: mode, mount tilt, target width, gains
  flight_report.py    post-flight analysis from the control CSV
  start-wasp.sh       boot launcher (reads flight_config.py)
  start-wasp-sky.sh   manual sky-mode launcher with a mount-tilt confirmation
  wasp-tracker.service  systemd user unit: starts at power-up
  tests/              offline simulations, fake FC, bench diagnostics
  data/               run logs, videos, CSVs (generated)
runs/detect/          sky-detector training run and weights
*.pt / *.engine / *.onnx   model weights
```

## Setup

On a new Jetson, open Claude Code in the repo and have it follow **[INSTALL_PROMPT.md](INSTALL_PROMPT.md)**. It checks and fixes the environment, TensorRT engine, camera mode, FC link and service, then prints a report.


Python dependencies live in the virtualenv `.env/yolo`. It must see the system site-packages so that JetPack's GStreamer OpenCV and TensorRT are available. It needs:

- `torch` (CUDA build — the CPU build runs but gives ~0.3 fps)
- `ultralytics`, `numpy`
- `pymavlink`, `pyserial`

The field has no internet. The sky weights are in the repo, and the launchers set `YOLO_AUTOINSTALL=false` so ultralytics never tries pip at boot.

Enable auto-start at power-up:

```bash
systemctl --user enable --now wasp-tracker.service
tail -f ~/wasp-logs/wasp-tracker.log
```

Starting the service does **not** give the AI control. It waits, idle, until the pilot arms and selects ALT_HOLD.

## Configuration

Edit **`src/flight_config.py`** and reboot:

```python
MODE = "sky"            # the only mode
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
| `test_mission.py` A12 | AUTO: climb ramps in and stops at 10 m, holds with no detection, lines up rate-limited, goes bumplessly, lost / cancel / ABORT / mode switch / ENGAGE, the 3 m floor, the 2 s and closing-speed early goes, refusals |
| `test_lock.py` | LOCK under bad conditions: noisy/late heading, across north, heading lost mid-lock, 10 min in gusts, weak yaw motors; locked at 313° through a gust stronger than the motors, steady torque, ⟲/⟳/▲ while locked; never over 15°/s or 30°/s²; the phone's command path |
| `ui_demo.py` | the phone app on a desk (no drone): `tests/ui_demo.py`, open the printed link, then ☰ → TEST |
| `sim_lock_wind.py` | LOCK vs the FC alone, 600 gusty hovers (not pass/fail) |
| `sim_sky_wind.py` | sky-mode campaign: moving target × gusty wind Bf 0–3, touch odds and turn overshoot (not pass/fail). `--speed all` compares the four AI levels; `SIM_AUTO=1` presses AUTO instead of ENGAGE; `SIM_START=ground` starts at 1.5 m with the target 25–50 m out and 6–25 m up. Total tilt is held to `ANGLE_MAX` as the FC does (`SIM_TILT_LIMIT=0` = the old per-axis model) |
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
