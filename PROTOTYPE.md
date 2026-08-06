# PROTOTYPE — WASP head tracker

Living status doc for the `prototype` branch. **Update this whenever the code
changes.** It replaces the old `HANDOFF.md` (still readable on `WASP-45`, commit
`f8a1a2c`, which carries the full history of the two faults that killed demo 1).

Last updated: 2026-08-06.

---

## 1. Layout

```
src/
  run.py              entry point: interpreter, preset, FC discovery, preflight, tracker
  tracker.py          camera, YOLO, Kalman, HUD, per-frame steering call
  fc_control.py       two control laws + FC transports + transmit thread + safety + CSV
  wasp-tracker.service  systemd user unit, auto-starts run.py at Jetson power-up
  tests/
    fc_check.py       bench diagnostic: identify firmware, verify overrides, wiggle sticks
    fake_fc.py        simulated ArduPilot (sysid gating, RC_CHANNELS echo, ATTITUDE)
    test_control_law.py     the GIMBAL law
    test_intercept.py       the FREE-FLIGHT law, incl. a closed-loop intercept sim
    test_simulated_fc.py    transports, gating and both laws end to end
```

Run all three after **any** change to the control path. None needs an FC or a
camera. All three pass as of this commit.

```bash
cd src
python3 tests/test_control_law.py
python3 tests/test_intercept.py
python3 tests/test_simulated_fc.py
```

---

## 2. The rig

* **Jetson Orin** on the airframe, CSI camera (IMX219, sensor mode 4 → 1280x720@30).
* **TBS Lucid H7**, **ArduPilot Copter** — confirmed on the bench 2026-08-04.
  `SYSID_MYGCS=255`, overrides **ACCEPTED**. `ACRO_TRAINER` and `ACRO_RP_EXPO`
  both zeroed on the board.
* `ACRO_Y_RATE=202` deg/s, `ACRO_RP_RATE=360` deg/s.
* **NO GPS.** This rules out GUIDED/LOITER and any position hold. `ALT_HOLD` needs
  only the barometer, which is why the free-flight law targets it.
* A **real RC transmitter + receiver**. The pilot arms and flies.

---

## 3. TWO CONTROL LAWS — pick with `CONTROL_LAW`

| | `servo` (`ControlLaw`) | `intercept` (`InterceptLaw`) |
|---|---|---|
| flight mode | **ACRO** (stick = body rate) | **ALT_HOLD** (stick = angle / climb rate) |
| driven | yaw, pitch | **roll, pitch, yaw, throttle** |
| law | pursuit on pixel error | **proportional navigation** |
| correct for | **the gimbal** — cannot translate | **free flight** |
| preset | `run.py --gimbal` | `run.py` (**default**) |

**They are not retunes of each other.** On a gimbal, pointing the camera *is* the
whole job, and the servo law did it well: 656/656 frames LOCK on the bench. In
free flight, pointing the camera and moving the airframe are the *same two
actuators*, so aiming and intercepting compete for them. That is why the servo
law tracked beautifully on the rig and then flew **backwards** in the air.

### 3a. The gimbal law (`servo`) — unchanged, still verified

```
pixel offset -> normalized offset -> angular error -> body rate -> stick -> PWM
```

Pinhole/tangent angular error; stick = `rate / ACRO_*_RATE` read off the FC;
deadband, per-axis ceiling, slew limiter. `PITCH_MODE=aim` nulls vertical pixel
error (correct on the gimbal); `PITCH_MODE=approach` holds a nose-down lean via an
outer loop on measured `ATTITUDE` (the free-flight stopgap that `intercept`
replaces).

### 3b. The free-flight law (`intercept`) — NEW

Four channels, one job each:

| Channel | Job | Driven by |
|---|---|---|
| **yaw** | keep the seeker on target | pursuit on azimuth error |
| **roll** | lateral acceleration — turn the velocity vector | **PN on the LOS rate** |
| **pitch** | closing speed | nose-down lean × commit |
| **throttle** | vertical closure | elevation error + LOS-rate lead |

**Proportional navigation.** `a_lat = ROLL_PN_GAIN * λ̇`, then `bank = atan(a_lat/g)`.
Driving the *rate of rotation of the line of sight* to zero holds the target at a
**constant bearing**, which is a collision course. Pure pursuit (turn ∝ current
bearing error) instead trails a crossing target for ever — that is the
carrot-on-a-stick behaviour reported from the field.

**The LOS rate is one signal, and that detail is load-bearing:**

```
λ      = yaw_measured + azimuth_in_frame        (unwrapped)
λ̇      = d(λ)/dt                                (low-pass filtered)
```

The obvious form, `d(az_body)/dt + yawspeed`, is algebraically identical and
**dynamically unstable**: it mixes a heavily filtered term with an instantaneous
one, so every seeker yaw transiently reads as LOS rotation, PN banks on the
airframe's own yaw, and it limit-cycles. It was *measured* doing exactly that —
bank pinned at 30°, heading winding through 180°+, range never closing. Locked by
`test_intercept.py` §3 ("the SEEKER's own yaw cannot drive PN") and §9.

With the single-signal form the seeker cannot drive PN for a structural reason
rather than a tuned one: λ is a *world* bearing, so rotating the airframe does not
change it. Only relative translation does.

**And both halves of λ must come from the same instant.** `azimuth_in_frame` comes
from a video frame ~100 ms old; `ATTITUDE` arrives on its own schedule. Adding two
numbers from different times injects a bearing error that varies with yaw
*acceleration*, and PN multiplies it by a large gain. Measured end to end against
the simulated FC: a nose swinging on a **constant** target bearing produced **27°
of phantom bank**, pinned at the ceiling. `ArduPilotAcro.attitude(at_t=…)` now
interpolates the attitude history to the frame's capture time (yaw kept unwrapped
so it can be interpolated at all), and the same test reads **0.06°**. ATTITUDE is
requested at 50 Hz rather than 20 to keep the interpolation error small. Locked by
`test_simulated_fc.py` case J.

**Vertical.** `climb = CLIMB_GAIN * el_world + CLIMB_LEAD * λ̇_el`, commanded as an
ALT_HOLD climb rate. Elevation is resolved into the **world** frame,
`el_world = pitch + cam_tilt + el_camera`, so the command follows the target's real
elevation rather than where it lands on the sensor.

**Roll de-rotation.** Pixel error is de-rotated by measured roll before anything
else. Once roll is driven the horizon tilts in the image, and without this a bank
leaks error straight into the other axis.

**Missing attitude ⇒ zero, never a guess.** No fresh `ATTITUDE` and roll and
throttle command exactly 0 — the airframe still aims and flies forward but will
not manoeuvre or change altitude. Preflight names it as a blocker.

**Range and closing speed** come from bbox width against `TARGET_WIDTH_M` (0.20 m,
a head). Used for the terminal trigger and the log, deliberately **not** in the
loop gain: the estimate is noisy, and textbook PN's `V_closing` term collapses to
zero when the target runs away at a similar speed — commanding no turn exactly
when the chase needs one.

**Terminal commit.** Past `TERMINAL_BBOX_FRAC` of frame height or
`TERMINAL_RANGE_M`, guidance **freezes** and the airframe flies the trajectory it
already had, for at most `TERMINAL_HOLD_S`. In the last metre the head fills the
frame, the detector (trained on distant heads) drops out and the pixel error goes
meaningless — so this is where a live loop veers off. Latched (one lost detection
cannot un-commit) and time-limited (a false trigger cannot fly on for ever).

**Latency.** The target is extrapolated forward by the measured
capture→command delay (`tracker.py` stamps the frame at `cap.read()`), capped at
`LATENCY_MAX_S`. Measured stable to 150 ms, degrading but not diverging at 200 ms.
Latency remains the binding constraint — the fix for more is a faster pipeline, not
more gain.

### 3c. THE CAMERA TILT IS NOT OPTIONAL

At nose-down lean θ, a target at the airframe's **own** altitude appears at
elevation **+θ** in a body-fixed camera. The lens is 48.8° vertical (**±24.4°**),
and mode 4 is a *cropped* IMX219 mode so the truth is narrower. At 12° of lean a
level target is already halfway to the top edge; past ~24° it leaves the frame
**entirely, out of the top**.

This is why `aim` and `approach` were both unwinnable: `aim` kept the target in
frame by pitching up and flying backwards; `approach` flew forwards and lost it out
of the top. **Tilt the camera mount UP by roughly the cruise lean** (the preset
assumes 15° for `MAX_LEAN_DEG=20`) and set `CAM_TILT_DEG` to the real angle.
`CAM_TILT_DEG=0` raises a preflight blocker.

### 3d. Gain scaling — the one coupling to remember

`ROLL_PN_GAIN` is dimensionally an **own-speed in m/s**: `gain = N × airspeed`,
where `N` is PN's navigation constant. `N=1` is exactly a *coordinated turn*;
classical PN uses `N=3..5`.

The default **45** is `N=3` × the ~15 m/s the airframe settles at with
`MAX_LEAN_DEG=20`. **Raise `ROLL_PN_GAIN` if you raise `MAX_LEAN_DEG`**, or the
turn gain no longer matches the speed it flies. Measured (§9):

| `ROLL_PN_GAIN` | hard zigzag | target in frame |
|---|---|---|
| 0 (no bank) | never closes | — |
| 25 | **no hit**, 2.64 m | 37 % |
| **45 (default)** | **hit, 0.57 m** | **100 %** |
| 80 | 0.60 m | 100 % |

Not monotonic — over-gaining is worse again.

---

## 4. Safety model

* Sticks are driven **only** while the FC is in a **steer mode** (`FC_STEER_MODES`:
  ALT_HOLD for intercept, ACRO for the gimbal) AND **armed** AND a target is
  locked. Re-evaluated every transmit cycle, so it toggles indefinitely.
* **In ALT_HOLD a centred throttle HOLDS ALTITUDE.** That is what makes it
  defensible for the AI to hold the throttle at all — a release or a freeze keeps
  flying level, where in ACRO the same PWM is an arbitrary thrust setting.
* Target stale after 0.5 s ⇒ release. `NO_TARGET_BEHAVIOR=release`.
* `tracker.run()`'s `finally` calls `ap.close()`, so Ctrl-C, a camera fault or any
  unhandled exception releases the sticks. Release is sent 5× before the link
  closes. ArduPilot's `RC_OVERRIDE_TIME` (3 s) is the FC-side backstop.
* **Props off for all bench work.**

### The abort path, and its honest latency

The mode gate can only be as fast as ArduPilot's **1 Hz `HEARTBEAT`** — up to a
second of travel at intercept speed. With the AI holding all four sticks that is
not good enough on its own, so:

* **`COMMIT_CHAN` — wire this if you possibly can.** Any spare channel. It is
  never overridden, so it arrives on the 10 Hz `RC_CHANNELS` stream: a **live
  speed dial** (it scales the forward lean 0..1) *and* a **~100 ms abort** (below
  `COMMIT_ABORT_BELOW`, every channel is released). Without it the lean is fixed
  at `COMMIT_DEFAULT` whenever a target is locked and the mode switch is the only
  abort.
* **`ENGAGE_MAX_S`** can cap one continuous engagement, but is **0 (off) by
  default**: the pilot is the abort, and a timer that drops the sticks
  mid-engagement is its own surprise.

**Why the pilot's throttle can no longer be the "go" command:** `throttle_frac()`
reads the pilot's stick out of `RC_CHANNELS`, which is genuinely their input *only
because throttle was never overridden*. ArduPilot copies an active override into
`radio_in` — the very mechanism the confirmation check relies on — so once the AI
drives throttle, that read returns **our own command**. It becomes a feedback
loop. Hence the commit channel.

---

## 5. Verification

**Gimbal law** (`test_control_law.py`): sign conventions, deadband continuity,
saturation, slew, inversion flags, RC PWM mapping with asymmetric trim, MSP CRC
framing, closed-loop convergence, a negative control proving a genuinely reversed
axis **diverges**, and §12 proving `approach` closes from every quadrant while
`aim` with the target above **retreats −40.9 m**.

**Free-flight law** (`test_intercept.py`): roll de-rotation round-trips −40..+60°;
the camera-tilt geometry analytically; PN's defining properties (constant bearing ⇒
no bank, correct direction, correct magnitude, **the seeker's own yaw cannot drive
it**, heading-wrap safe); the vertical channel commanding the same climb at every
lean angle; lean **never** positive anywhere in the frame; range inversion;
terminal latch + expiry; latency measurement and lead; the throttle PWM map
(0 ⇒ exactly mid, `THR_DZ` escape, trim-at-bottom ignored).

§9 is a **closed-loop sim that models translation** — lean → forward accel, bank →
lateral accel, throttle → climb, drag → terminal speed — with the camera
measurement generated by inverting exactly the geometry the law resolves. Results:

| scenario | with PN (45) | without bank |
|---|---|---|
| sprint straight across | **hit at 4.9 s**, 0.58 m | **never closes**, 7.44 m |
| heading swept getting there | 30° (constant bearing) | 156° (tail chase) |
| hard zigzag, close range | **hit at 8.9 s**, 0.57 m, 100 % in frame | — |
| stern chase, mean range | 16.9 m | 21.2 m |
| 8 px jitter, sprint, 5 seeds | 0.57–0.60 m | — |
| 8 px jitter, hard zigzag | 0.57–3.13 m (**the weak spot**) | — |
| latency 50/100/150 ms | 0.57 / 0.57 / 0.59 m | — |
| latency 200 ms | 1.28 m (degrades, no divergence) | — |

**Transports and gating end to end** (`test_simulated_fc.py`): cases A–G as before
(sysid adoption, the demo-1 regression, `RC_OPTIONS`, disarmed, the mode gate
toggling both ways), plus **H** the intercept path in ALT_HOLD driving all four
channels with the commit channel read and never written and the commit abort
releasing everything, and **I** no `ATTITUDE` ⇒ roll and throttle exactly zero
while yaw and pitch still drive.

**Verified on real hardware 2026-08-04** (gimbal law only): 656/656 LOCK, 619
confirmed, 0 refused, 2260 direction checks with 0 wrong.

### Not yet verified on hardware — read this before flying

* **The entire intercept law.** Simulated only.
* **Props-on behaviour**, for either law.
* **Roll and throttle polarity on the real airframe.** Check on the bench: standing
  to the drone's LEFT must command LEFT bank; a target above must command CLIMB.
  Use `ROLL_REVERSED` / `THROTTLE_REVERSED`, never a sign in the law.
* **`ANGLE_MAX`, `PILOT_Y_RATE`, `PILOT_SPEED_UP`, `THR_DZ`** are read off the FC
  but have never been confirmed to read correctly on this board — `fc_check.py`
  and the connect banner both print them.
* **`PILOT_SPEED_UP` is 2.5 m/s by default**, which caps the vertical chase. Raise
  it on the FC if the target changes height fast.
* **True lens FOV is assumed** (§3c).
* The mode gate against the real 1 Hz heartbeat.

---

## 6. Bring-up order for the intercept law

Nothing that can move is powered until the previous step passes. **Props off
through step 4.**

1. `python3 tests/fc_check.py --verify` — firmware, gating params, override accepted.
   FC on USB, no battery.
2. Tilt the camera mount up and measure the angle. Set `CAM_TILT_DEG`.
3. `CONTROL_LAW=intercept REQUIRE_ARMED=0 ./run.py` on the bench, disarmed, props
   off. Walk around the camera and read the console: `bank` must go **right** when
   you move right, `climb` **positive** when you stand above the lens. Fix polarity
   with `ROLL_REVERSED` / `THROTTLE_REVERSED`.
4. Battery in, props **off**, pilot arms in **ALT_HOLD**. Confirm `DRIVE` on the
   HUD and a non-zero `lat` reading. Check the CSV's `latency_s` column — if it is
   above ~0.15 s, fix the pipeline before flying (TensorRT engine, smaller imgsz).
5. Props on, open field, **`MAX_LEAN_DEG=10 ROLL_PN_GAIN=22`** for the first
   flight — half speed, gain scaled to match — then work up. Wire `COMMIT_CHAN`
   first if at all possible.

Reading the console:

```
[CTRL] LOCK  target +0142,-0071px -> centre | yaw +0.11(+22d/s) bank +12.4 lean -18.0
       climb +1.2m/s | LOS +14.2,+1.1d/s R 18.3m lat 105ms | TX->FC (confirmed)
```

HUD lamps: `DRIVE` confirmed / `TX` pending / `REFUSED` discarded / `PILOT` not in
the steer mode / `ARM?` disarmed / `IDLE` no target / `SIM` no FC. Plus
`TERMINAL — guidance frozen` and `NO ATTITUDE — roll+climb are ZERO`.

Every run writes `src/data/control_<stamp>.csv`. The intercept law adds
`roll_stick, throttle_stick, el_cam_deg, bank_deg, lean_deg, climb_mps,
los_az_dps, los_el_dps, range_m, closing_mps, latency_s, commit, have_att,
terminal`. **`los_az_dps` is what PN is actually reacting to and `latency_s`
decides whether any of it is stable — plot those two first.**

---

## 7. Hard rules for anyone editing this

1. **Do not reintroduce GUIDED velocity setpoints.** There is no GPS.
2. **Do not hardcode a MAVLink source system id.** Read `SYSID_MYGCS`.
3. **Do not make the HUD claim `TX` for a command that was merely computed.** That
   single lie cost a whole demo.
4. **Do not fix a wrong-direction axis by flipping a sign in a control law.** Use
   the `*_REVERSED` flags. The signs are test-locked.
5. **Do not remove the release-on-exit path** in `tracker.run()`'s `finally`.
6. **Do not restore "no target ⇒ fly forward at max speed".**
7. **Do not compute the LOS rate as `d(az_body)/dt + yawspeed`.** It is
   algebraically right and dynamically unstable — see §3b. Differentiate the world
   bearing as one signal, and take the attitude **at the frame's capture time**.
8. **Do not judge the override echo against only the latest sent value.**
   `RC_CHANNELS` arrives at ~10 Hz against a 50 Hz command stream, so mid-manoeuvre
   the frame you get back was generated against an older command. Matching only the
   newest one reports `REFUSED` during a healthy engagement — it did. Match against
   the last `ECHO_HISTORY_S` of sent values.
9. **Do not blend the aim term back into pitch in intercept mode.** Vertical
   framing is the throttle's job now.
10. **`throttle_frac()` is meaningless once throttle is driven** (§4). Use
   `commit()`.
11. **Report the post-clamp, post-slew command**, not the raw law output. The HUD
    gauges, the CSV and the sim's plant all read these numbers.
12. Run all three suites in `tests/` after any change to the control path.
13. **Props off** for every bench test.

---

## 8. Env reference

**Shared:** `CONTROL_LAW` `MAV_CONN` `MAV_BAUD` `FC_TRANSPORT` `CONTROL_AXES`
`FC_STEER_MODES` `MAV_SOURCE_SYSTEM` `CAM_HFOV_DEG` `CAM_VFOV_DEG` `STICK_LIMIT`
`DEADBAND_FRAC` `SLEW_PER_S` `SEND_HZ` `REQUIRE_ARMED` `REQUIRE_RATE_MODE`
`NO_TARGET_BEHAVIOR` `FC_DEV` `FC_BAUD` `CAM_DEV` `YAW_REVERSED` `PITCH_REVERSED`

**Gimbal only:** `KP_YAW` `KP_PITCH` `KD_YAW` `KD_PITCH` `STICK_LIMIT_YAW`
`STICK_LIMIT_PITCH` `CONTROL_SHAPE` `SHAPE_SAT` `SHAPE_GAMMA` `PITCH_MODE`
`APPROACH_DEG` `APPROACH_KP` `APPROACH_THR_DEADBAND`

**Intercept only:** `CAM_TILT_DEG` `ROLL_PN_GAIN` `MAX_BANK_DEG` `MAX_LEAN_DEG`
`CLIMB_GAIN` `CLIMB_LEAD` `COMMIT_CHAN` `COMMIT_DEFAULT` `COMMIT_ABORT_BELOW`
`ENGAGE_MAX_S` `TARGET_WIDTH_M` `TERMINAL_BBOX_FRAC` `TERMINAL_RANGE_M`
`TERMINAL_HOLD_S` `CONTROL_LATENCY_S` `LATENCY_MAX_S` `LOS_TAU`
`ROLL_REVERSED` `THROTTLE_REVERSED` `STICK_LIMIT_ROLL` `STICK_LIMIT_THROTTLE`

Full docs and defaults live in `commander_from_env` and `_intercept_from_env`.

**Intercept preset** (`run.py`, the default): `CAM_TILT_DEG 15, KP_YAW 4.5,
ROLL_PN_GAIN 45, MAX_BANK_DEG 30, MAX_LEAN_DEG 20, CLIMB_GAIN 8, CLIMB_LEAD 3,
STICK_LIMIT 0.90, SLEW_PER_S 11, DEADBAND_FRAC 0.02, LOS_TAU 0.12,
TERMINAL_BBOX_FRAC 0.35, TERMINAL_RANGE_M 2.5, SEND_HZ 50`

**Gimbal preset** (`run.py --gimbal`): `KP_YAW 4.5, KP_PITCH 6.0, KD 0.10,
STICK_LIMIT_YAW 0.65, STICK_LIMIT_PITCH 0.22, SLEW_PER_S 11, PITCH_MODE approach,
APPROACH_DEG 12`

Launcher flags: `--intercept` (default), `--gimbal`, `--tuned` (no-op),
`--no-wait`, `--router`, `--no-mp`, `--no-verify`.

---

## 9. Open items

* Everything in §5 "not yet verified on hardware".
* **8 px of jitter on a hard zigzag is the weak spot** (0.57–3.13 m across seeds).
  The lever is detection quality and latency, not gain.
* `src/data/` CSVs are gitignored; nothing plots them yet.
* `tracker.py`'s Kalman still runs in pixel/frame units and is used only to coast
  through skip frames. The intercept law does its own filtering and latency
  extrapolation in angle space, so the two are independent — worth unifying.
