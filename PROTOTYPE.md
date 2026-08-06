# PROTOTYPE — WASP head tracker

Living status doc for the `prototype` branch. **Update this whenever the code
changes.** It replaces the old `HANDOFF.md` (still readable on `WASP-45`, commit
`f8a1a2c`, which carries the full history of the two faults that killed demo 1).

Last updated: 2026-08-06.

---

## 1. Layout

```
src/
  run.py              entry point: interpreter, gains, FC discovery, preflight, tracker
  tracker.py          camera, YOLO, Kalman, HUD, per-frame steering call
  fc_control.py       control law + FC transports + transmit thread + safety + CSV
  wasp-tracker.service  systemd user unit, auto-starts run.py at Jetson power-up
  tests/
    fc_check.py       bench diagnostic: identify firmware, verify overrides, wiggle sticks
    fake_fc.py        simulated ArduPilot (reproduces sysid gating + RC_CHANNELS echo)
    test_control_law.py     control-law checks
    test_simulated_fc.py    transport / gating checks against fake_fc
```

Run the suites after **any** change to the control path:

```bash
cd src
python3 tests/test_control_law.py
python3 tests/test_simulated_fc.py
```

Both need no FC and no camera. Both pass as of this commit.

---

## 2. The rig

* **Jetson Orin** on the airframe, CSI camera (IMX219, sensor mode 4 → 1280x720@30),
  runs the tracker.
* **TBS Lucid H7**, **ArduPilot Copter** — confirmed on the bench 2026-08-04.
  `SYSID_MYGCS=255`, overrides **ACCEPTED**. `ACRO_TRAINER` and `ACRO_RP_EXPO`
  were both zeroed on the board (the law's stick→rate map is linear).
* `ACRO_Y_RATE=202` deg/s, `ACRO_RP_RATE=360` deg/s. The asymmetry drives the
  per-axis stick ceilings.
* A **real RC transmitter + receiver** is connected. The pilot arms and flies.
* Two configurations have been flown:
  * **on a stationary gimbal** — the airframe can rotate but not translate.
  * **free flight on a field** — the case that exposed the open problem in §5.

## 3. How it works today

ACRO body-rate control over `RC_CHANNELS_OVERRIDE`.

```
pixel offset -> normalized offset -> angular error -> body rate -> stick -> PWM
```

* Normalized offset is `2*err/frame_size` (±1 at the frame edge). The 1280x720 →
  640x640 resize is anisotropic but **linear**, so the normalized offset survives
  it and pairing it with the true lens FOV is correct.
* Angular error uses the **pinhole/tangent** model, `atan(n*tan(FOV/2))`.
* `yaw_rate = KP_YAW * az_error` — target right of centre ⇒ positive yaw rate.
* Stick = `rate / acro_full_stick_rate`, the full-stick rate read from the FC's
  own `ACRO_Y_RATE` / `ACRO_RP_RATE`.
* Then deadband (rescaled, no step at the edge), per-axis `STICK_LIMIT` clamp,
  slew-rate limiter.
* **Nothing about the transmitter is hardcoded.** Channel map, calibration, ACRO
  rates and `SYSID_MYGCS` are all read off the FC at connect. Channels we do not
  drive are sent as `0`, which ArduPilot reads as "no override".
* **Override confirmation:** ArduPilot copies an active override into `radio_in`,
  which `RC_CHANNELS` reports, so the backend compares what came back to what it
  sent. Judged only on channels commanded >40 µs off trim — a stick at trim reads
  the same either way and matching on it would manufacture a false confirm.
  HUD: `DRIVE` confirmed / `TX` pending / `REFUSED` discarded / `PILOT` not in
  ACRO / `ARM?` disarmed / `IDLE` no target / `SIM` no FC.

### Driven channels

| Channel | Driven? | By what |
|---|---|---|
| **yaw** | yes | nulls horizontal pixel error |
| **pitch** | yes | `PITCH_MODE` — see below |
| **roll** | **no** | never touched, in any mode |
| **throttle** | **no** | pilot's, and *read* as the approach command |
| arming | **no** | pilot's |

### `PITCH_MODE`

| Mode | Pitch does | Correct when |
|---|---|---|
| `aim` | nulls the target's vertical pixel error | **on the gimbal** |
| `approach` (current default) | holds a nose-DOWN lean, ignores vertical pixel error | **in free flight** |

`approach` exists because off the gimbal a multirotor's **pitch attitude IS its
direction of travel**, so "aim the camera up at a target above me" and "fly
towards it" are opposite commands on one axis. Aiming up is what made the
airframe retreat.

```
target_lean = -APPROACH_DEG * pilot_throttle_above_mid     (never positive)
pitch_rate  = APPROACH_KP * (target_lean - measured_pitch)
```

* **The pilot's throttle is the approach command.** Backing it off is an instant
  abort with the stick they already hold.
* It closes on measured `ATTITUDE`, not on a rate — in ACRO a stick is a rate, so
  an open-loop nose-down command keeps rotating past vertical. ArduPilot sends no
  `ATTITUDE` until asked, so the backend requests it at 20 Hz on connect.
* `target_lean` can never be positive ⇒ can never command nose-up ⇒ can never fly
  backwards. Lean hard-clamped to 30°.
* A missing or stale `ATTITUDE` commands **zero** pitch, never a last-known value.
  Preflight waits 3 s for the first sample and lists a blocker if none arrives.

### Rates

Two **deliberately decoupled** rates:

* **Detection rate** — `tracker.py` recomputes the stick once per YOLO frame
  (`skip_frames=0`, so every camera frame). `ControlLaw.compute()` uses the
  wall-clock gap between detection frames as its `dt`, which is correct.
* **Transmit rate** — a separate thread streams the *latest* command at a fixed
  `SEND_HZ` (default **20**), independent of detection FPS, so a slow YOLO frame
  never starves the FC and never creates a gap in the stream.

### Safety model

* Sticks are driven **only** while the FC is **in ACRO** AND **armed** AND a
  target is locked. The pilot's mode switch is the handover: STABILIZE ⇒ the pilot
  has every channel, ACRO ⇒ the AI drives yaw+pitch. Re-evaluated every transmit
  cycle, so it toggles indefinitely. Handover detection is bounded by ArduPilot's
  1 Hz `HEARTBEAT`, so a handback can take up to ~1 s to be noticed;
  `RC_OVERRIDE_TIME` (3 s) is the FC-side backstop.
* Roll, throttle and arming are **never** touched.
* Target stale after 0.5 s with no detection ⇒ release.
* No target ⇒ `NO_TARGET_BEHAVIOR=release`. The old "no face ⇒ fly forward at max
  speed" behaviour is **gone** and must not return.
* `tracker.run()`'s `finally` calls `ap.close()`, so Ctrl-C, a camera fault or any
  unhandled exception still releases the sticks. The release is sent 5× before the
  link closes.
* **Props off for all bench work.**

Every run writes `src/data/control_<stamp>.csv`:
`t_s,mode,armed,steering,written,confirmed,ex,ey,az_deg,el_deg,yaw_stick,pitch_stick,yaw_rate_dps,pitch_rate_dps`.

---

## 4. Verified

**Simulated** (`tests/`, no hardware): sign conventions, deadband continuity,
saturation, slew, inversion flags, RC PWM mapping with asymmetric trim, MSP CRC
framing, closed-loop convergence, a negative control proving a genuinely reversed
axis **diverges**, the sysid gating rule including the demo-1 regression, the mode
gate toggling both ways, and §12 proving `approach` closes from above/below/
left/right while `aim` with the target above **retreats −40.9 m**.

**Real hardware, 2026-08-04**, props off, pilot armed in ACRO at mid throttle:
656/656 frames LOCK, 619 `TX->FC (confirmed)`, 0 `REFUSED`, 2260 direction checks
with 0 wrong. Sticks released immediately on disarm.

**Still unverified:** props-on behaviour, and the mode gate against the real 1 Hz
heartbeat (simulated only).

---

## 5. THE OPEN PROBLEM — free-flight intercept

On the gimbal the tracker works: it rotates onto the target and holds it centred.
**In free flight it does not intercept**, and the reasons are structural, not
tuning:

1. **Pure pursuit lags a crossing target.** The law is a carrot-on-a-stick: turn
   rate ∝ *current* pixel error. Against a target running across the field this
   produces a trailing tail-chase that never closes. The standard fix is
   **proportional navigation** — command lateral acceleration ∝ line-of-sight
   *rate*, which flies a constant-bearing collision course.

2. **Nothing controls the vertical.** `aim` used pitch and flew backwards;
   `approach` correctly stopped using pitch but put nothing in its place. On a
   multirotor the only channel that changes altitude is **throttle**, which the AI
   does not touch. So a target above or below the flight line is never closed on.

3. **Roll is unused, so there is no lateral acceleration channel at all.** A
   multirotor turns its velocity vector by banking. Yaw only rotates the camera.
   (For the record: roll was **never** driven on the gimbal either — the
   yaw+pitch combination just looks like it.)

4. **A fixed forward camera cannot both fly fast and keep the target in frame.**
   At nose-down lean θ, a target at the drone's own altitude sits at elevation
   **+θ** in the image. The lens is 48.8° vertical, i.e. ±24.4°, and mode 4 is a
   *cropped* IMX219 mode so the true figure is narrower. At θ=12° a level target
   is already halfway to the top edge; past ~24° it leaves the frame entirely.
   This is a hardware limit, not a gain.

5. **Latency, not rate, is the binding constraint.** The grid search found every
   configuration goes unstable beyond ~0.2 s of detection latency. At intercept
   closing speeds 0.2 s is metres of stale geometry. End-to-end latency has never
   been measured, only the frame interval.

Unused state already on hand, which the fixes need: `ATTITUDE` is streaming at
20 Hz and `fc_control.py` stores `_roll_rad` but **exposes no reader and never
uses it**; `msg.yawspeed`/`pitchspeed` are discarded entirely; and `tracker.py`
already runs a Kalman filter (currently only to coast through skip frames) that
could predict the target forward by the measured latency.

Design discussion in progress — nothing in §3 has changed yet.

---

## 6. Hard rules for anyone editing this

1. **Do not reintroduce GUIDED velocity setpoints** for the gimbal rig. They
   cannot steer an airframe that cannot translate and has no EKF position.
2. **Do not hardcode a MAVLink source system id.** Read `SYSID_MYGCS` from the FC.
3. **Do not make the HUD claim `TX` for a command that was merely computed.** That
   single lie cost a whole demo.
4. **Do not fix a wrong-direction axis by flipping a sign in `ControlLaw`.** Use
   `YAW_REVERSED` / `PITCH_REVERSED`. The law's signs are test-locked.
5. **Do not remove the release-on-exit path** in `tracker.run()`'s `finally`.
6. **Do not restore "no target ⇒ fly forward at max speed".**
7. **If the AI is ever given the throttle channel, `throttle_frac()` breaks.** It
   reads the pilot's stick out of `RC_CHANNELS`, which is only genuinely the
   pilot's input *because throttle is never overridden*. Once overridden,
   `RC_CHANNELS` reports our own value back and the read becomes a feedback loop.
8. Run `tests/` after any change to the control path.
9. **Props off** for every bench test.

---

## 7. Env reference

`MAV_CONN` `FC_TRANSPORT` `CONTROL_AXES` `MAV_SOURCE_SYSTEM` `CAM_HFOV_DEG`
`CAM_VFOV_DEG` `KP_YAW` `KP_PITCH` `KD_YAW` `KD_PITCH` `STICK_LIMIT`
`STICK_LIMIT_YAW` `STICK_LIMIT_PITCH` `DEADBAND_FRAC` `SLEW_PER_S`
`CONTROL_SHAPE` `SHAPE_SAT` `SHAPE_GAMMA` `PITCH_MODE` `APPROACH_DEG`
`APPROACH_KP` `APPROACH_THR_DEADBAND` `YAW_REVERSED` `PITCH_REVERSED` `SEND_HZ`
`REQUIRE_ARMED` `REQUIRE_RATE_MODE` `NO_TARGET_BEHAVIOR` `FC_DEV` `FC_BAUD`
`CAM_DEV`

Full docs and defaults live in `fc_control.commander_from_env`'s docstring.

Current preset (`run.py`, one preset only, raised 2026-08-05):
`KP_YAW 4.5, KP_PITCH 6.0, KD 0.10, STICK_LIMIT_YAW 0.65,
STICK_LIMIT_PITCH 0.22, SLEW_PER_S 11.0, PITCH_MODE approach, APPROACH_DEG 12`

Launcher flags: `--tuned` (no-op), `--no-wait`, `--router`, `--no-mp`,
`--no-verify`.
