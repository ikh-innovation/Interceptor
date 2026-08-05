# HANDOFF — WASP head tracker, flight-control path

Written 2026-08-03 on branch `WASP-45`, for whoever (human or agent) picks this
up next on the Jetson. This is the state of the flight-control path immediately
before the demo flight, and the reasoning behind every decision in it, so the
same mistakes are not made twice.

---

## 1. The rig

* **Jetson Orin dev kit**, mounted on the drone, CSI camera, runs the tracker.
* **Flight controller: TBS Lucid H7 (STM32H743VIH6).** Which firmware is on it
  right now is **NOT CONFIRMED** — run `fc_check.py` to find out. ArduPilot
  Copter is the intended and recommended firmware.
* **The airframe is bolted to a stationary gimbal.** It cannot translate. It can
  rotate in **yaw and pitch** (and roll, unused).
* **A real RC transmitter + receiver is connected.** The pilot arms the airframe
  and flies it in **ACRO**. The pilot keeps throttle, roll and arming at all times.

## 2. The goal

Detect a human head, and rotate the airframe until the detection's bounding-box
centre coincides with the camera-frame centre. Yaw nulls horizontal error, pitch
nulls vertical error. Detection already worked fine; **only the control output
was broken.**

---

## 3. What was broken (do not reintroduce either of these)

### Fault 1 — the tracker transmitted literally nothing

`run_pipeline.py` never set `CONTROL_MODE`, so `tracker.py` fell back to its
default `"guided"`, and the transmit loop was gated on the FC being in GUIDED
mode:

```python
if self.linked and guided and not stopped:      # old tracker.py
    self._tx_once(cmd, self.master)
```

The drone is flown in ACRO, so `guided` was `False` on every frame and **not one
byte ever left the Jetson.** The HUD lamp still read `TX`, because it was driven
by `commanding = is_yolo_frame` — i.e. "a setpoint was computed", not "a setpoint
was sent". A dead control path looked healthy for an entire demo.

### Fault 2 — even the RC path was silently discarded by the FC

The alternative `CONTROL_MODE=rc` path (which `wasp-pipeline.service` did enable)
sent `RC_CHANNELS_OVERRIDE` from `source_system=1`. ArduPilot accepts overrides
**only** from the system id held in `SYSID_MYGCS` (renamed `MAV_GCS_SYSID` on
4.6+), whose default is **255**:

```cpp
void GCS_MAVLINK::handle_rc_channels_override(const mavlink_message_t &msg)
{
    if (msg.sysid != sysid_my_gcs()) {
        return;        // dropped: no ACK, no STATUSTEXT, no log entry, nothing
    }
```

So every override was thrown away with zero diagnostics. This is the most likely
cause of the commit named "Fail demo 1".

### Why GUIDED was the wrong design anyway

A GUIDED velocity setpoint (`SET_POSITION_TARGET_LOCAL_NED`) requires a healthy
EKF **position** estimate — GPS or optical flow. A gimbal-mounted airframe
indoors has neither, so ArduPilot would have refused the setpoints even in
GUIDED. Velocity control was never the right abstraction for a rig that cannot
translate.

---

## 4. What replaced it

**ACRO is the correct mode.** In ACRO a stick deflection *is* a body angular-rate
command, and "rotate until the target is centred" is exactly a body-rate
command. No GPS, no EKF position, works on a stationary gimbal.

The old `ArduPilotCommander` class (566 lines inside `tracker.py`) was deleted
and replaced by:

| File | Role |
|---|---|
| `fc_control.py` | **NEW.** Control law + FC transports + transmit thread + safety gating + CSV log |
| `run.py` | **NEW.** One-command launcher: picks the yolo-env interpreter, applies a gain preset (`--max` default, `--fast`, `--tuned`, `--gentle`), waits for the FC so it can be started before the drone is powered, then execs `run_pipeline.py --direct` |
| `fc_check.py` | **NEW.** Bench diagnostic: identify firmware, report gating settings, prove an override is accepted, wiggle the sticks |
| `run_pipeline.py` | **REWRITTEN.** Identifies firmware before anything claims the port, wires the right transport, verifies overrides, then runs the tracker |
| `tracker.py` | **MODIFIED.** Old commander removed; builds `AcroCommander` from env; HUD and console line now report the truth |
| `wasp-pipeline.service` | **UPDATED** to the new env vars |
| `README.md` | Bring-up runbook + tuning table |
| `tests/` | **NEW.** Self-test suite; needs no FC and no camera |
| `run_video_pilot.py` | Marked **STALE — does not run**; it predates the GStreamer camera pipeline |

### Control law (`fc_control.ControlLaw`)

```
pixel offset -> normalized offset -> angular error -> body rate -> stick -> PWM
```

* Normalized offset is `2*err/frame_size`, i.e. ±1 at the frame edge. The tracker
  resizes 1280x720 → 640x640, which is anisotropic but **linear**, so the
  normalized offset is preserved and pairing it with the true lens FOV is correct.
* Angular error uses the **pinhole/tangent** model, `atan(n * tan(FOV/2))`, not a
  linear px→deg ratio. Matters on wide CSI lenses.
* `yaw_rate = KP_YAW * az_error` — target right of centre ⇒ **positive** yaw rate
  (yaw right).
* `pitch_rate = -(KP_PITCH * el_error)` — target below centre ⇒ **negative**
  pitch rate (**nose down**). Positive body pitch rate is nose UP, by the
  right-hand rule about the body Y axis which points right. **This sign is the
  one people get wrong.**
* Stick = `rate / acro_full_stick_rate`, where the full-stick rate is read from
  the FC's own `ACRO_Y_RATE` / `ACRO_RP_RATE`, so the mapping matches this
  airframe's tuning rather than a guess.
* Then: deadband (rescaled, no step at the edge), `STICK_LIMIT` clamp, and a
  slew-rate limiter.
* `compute()` takes an optional `dt`; production leaves it `None` (wall-clock gap
  between detection frames is the correct timestep), the sim passes it explicitly.
* The returned `yaw_rate` / `pitch_rate` are the **physical** rates after clamp
  and slew — inversion flags do not flip them, because they compensate a reversed
  axis and the physical intent is unchanged. The HUD gauges read these, so bar
  and number always agree.

### Transports

* **ArduPilot** (`ArduPilotAcro`) — `RC_CHANNELS_OVERRIDE`. At connect it reads
  `SYSID_MYGCS`/`MAV_GCS_SYSID` and **adopts that system id** (the Fault-2 fix),
  plus `RC_OPTIONS`, `ACRO_TRAINER`, `ACRO_Y_RATE`, `ACRO_RP_RATE`,
  `RCMAP_YAW`/`RCMAP_PITCH` and each channel's `RC{n}_MIN/MAX/TRIM/REVERSED`.
  **Nothing about the transmitter is hardcoded.** Channels we do not drive are
  sent as `0`, which ArduPilot reads as "no override", so the pilot keeps them.
* **Betaflight/INAV** (`BetaflightAcro`) — `MSP_SET_RAW_RC` over the serial port,
  seeded each cycle from `MSP_RC` so undriven channels echo the pilot's real
  sticks. Reads `MSP_RX_MAP` for channel order. Needs `pyserial`.
* **Offline** (`OfflineBackend`) — `MAV_CONN=none`: commands computed and printed,
  never transmitted.
* `build_backend(transport="auto")` tries MAVLink then MSP.

### Override confirmation — the thing that was missing

ArduPilot copies an active override into `radio_in`, which is what `RC_CHANNELS`
reports. So the backend watches `RC_CHANNELS` and compares it to what it sent:

* match ⇒ `override_confirmed = True`, HUD lamp `DRIVE`, console `TX->FC (confirmed)`
* no match after 2 s of steering ⇒ `override_confirmed = False`, HUD lamp
  `REFUSED`, and a loud multi-line diagnostic naming the likely causes in order.

Confirmation is judged **only** on channels commanded clearly away from trim
(>40 µs). A stick sitting at trim reads the same whether or not the override was
applied, so matching on it would manufacture a false "confirmed".

### Safety model

* Sticks are driven **only** while the FC is **in ACRO** AND **armed** AND a target
  is locked. The pilot's mode switch is the handover control: in STABILIZE (or any
  non-rate mode) the tracker holds nothing and the pilot has every channel; flipping
  to ACRO engages the AI; flipping back releases within one transmit cycle. It
  toggles indefinitely — the gate is re-evaluated every cycle, not latched. Governed
  by `REQUIRE_RATE_MODE` (default `1`); each handover prints a `[FC][MODE]` line.
  Detection latency is bounded by ArduPilot's 1 Hz `HEARTBEAT`, so a handback can
  take up to ~1 s to be noticed; `RC_OVERRIDE_TIME` (3 s) is the FC-side backstop.
  Only `ArduPilotAcro` reports a live mode — `FcBackend.in_rate_mode()` returns
  `True` for the MSP and offline backends so they are not gated off by a mode they
  never report.
* Roll, throttle and arming are **never** touched.
* Target goes stale after `target_timeout` = 0.5 s with no detection ⇒ release.
* No target ⇒ `NO_TARGET_BEHAVIOR=release` (default): all channels back to the pilot.
  The old "no face ⇒ fly forward at max speed" behaviour is **gone** and must not
  return; it is meaningless on a gimbal and dangerous off one.
* `tracker.py`'s `run()` is a thin wrapper whose `finally` calls `ap.close()`, so
  Ctrl-C, a camera fault or any unhandled exception still releases the sticks.
  The release is sent 5× (a datagram can be dropped) before the link closes.
* ArduPilot independently drops a stale override after `RC_OVERRIDE_TIME`
  (default 3 s), so a hung Jetson cannot hold the sticks indefinitely.
* **Props off for all bench work.**

---

## 5. Bring-up procedure

Each step needs a different physical rig state, escalating deliberately: nothing
that can move is powered until the previous step has passed.

| Step | Battery / ESCs | Props | Pilot + TX | FC | Can it move? |
|---|---|---|---|---|---|
| 1 identify + verify | **not needed** (FC on Jetson USB) | off | not needed | disarmed | no |
| 2 wiggle | connected | **off** | yes | armed, mid throttle | motors spin, no rotation |
| 3 tracker, dry | connected | **off** | yes | disarm → arm *after* preflight | motors spin, no rotation |
| 4 direction + tune | connected | **off** | yes | armed, mid throttle | motors spin, no rotation |
| 5 the demo | connected | **on** | yes | armed, mid throttle | **yes** |

```bash
cd src/development/modules/object_tracking

# 1. Which firmware, and will it accept a stick command?  Safe while disarmed:
#    an override changes the FC's stick INPUT, not the motors.  USB power to the
#    FC is enough here — the ESCs are not involved.
python3 fc_check.py --verify

# 2. Prove the FC acts on the sticks, no YOLO.  Battery IN, PROPS OFF, armed in
#    ACRO at mid throttle.  Watch the motor SPREAD number and listen to the
#    motors — with no props the airframe will NOT rotate.
python3 fc_check.py --wiggle 8 --yes

# 3. Full pipeline.  Start DISARMED (preflight skips the override check if the FC
#    is already armed), let preflight pass, confirm the HUD is tracking, THEN arm.
python3 run_pipeline.py --no-mp

# 5. Props on, the real demo.  Only after 1-4 pass.  Start conservative.
STICK_LIMIT=0.35 python3 run_pipeline.py --no-mp
```

**Two traps.** USB power alone does not power the ESCs — from step 2 on the
flight battery must be connected. And the airframe cannot rotate with props off,
because rotation comes from differential prop thrust; bare motors produce
negligible torque. Props off you verify by the motor-spread number and the sound,
and check axis direction from the sign of the console `yaw`/`pitch` values
(stand right of boresight ⇒ yaw must be positive; stand below ⇒ pitch must be
negative).

`fc_check.py --verify` sweeps candidate system ids and tells you which one the FC
accepts. `--wiggle` also prints the motor output spread from `SERVO_OUTPUT_RAW`,
which is the real proof the FC turned sticks into differential thrust.

A running MAVProxy router owns the FC serial port exclusively. Either probe
through it (`--connect udpin:127.0.0.1:14551`) or stop it (`pkill -f mavproxy.py`)
and probe the port directly.

### Reading the running tracker

```
[CTRL] LOCK    target +0142,-0071px -> centre | yaw +0.11 ( +22.3 d/s)  pitch +0.03 (+10.1 d/s) | TX->FC (confirmed)
```

| HUD lamp | Meaning |
|---|---|
| `DRIVE` | sticks sent **and** confirmed by the FC |
| `TX` | sticks sent, confirmation still pending |
| `REFUSED` | sticks sent but the FC is discarding them |
| `PILOT` | FC is not in ACRO — the pilot holds every channel, by design |
| `ARM?` | target locked, FC disarmed — nothing will move |
| `IDLE` | no target: channels released to the pilot |
| `SIM` | no FC attached |

Every run writes `data/control_<stamp>.csv`:
`t_s,mode,armed,steering,written,confirmed,ex,ey,az_deg,el_deg,yaw_stick,pitch_stick,yaw_rate_dps,pitch_rate_dps`.
Plot that instead of guessing.

---

## 6. If it does not move — decision tree

1. **`fc_check.py --verify` says `nothing answered`** → power, USB cable, port
   ownership (router?), or `dialout` group membership.
2. **Says `BTFL`/`INAV`** → not ArduPilot. Either flash ArduPilot Copter (target
   `TBS_LUCID_H7`) — recommended — or configure Betaflight's MSP override:
   `feature MSP_OVERRIDE`, `set msp_override_channels_mask = 10` (ch2 pitch +
   ch4 yaw), `save`, then assign **MSP RC OVERRIDE** to an aux switch and turn it
   on. `run_pipeline.py` detects Betaflight and wires `FC_TRANSPORT=msp` itself
   (no router, and Mission Planner cannot share the port).
3. **Says `overrides: REFUSED`** → the printout names causes in priority order:
   wrong sysid, `RC_OPTIONS` bit 1 (value 2 = ignore overrides), router not
   forwarding (try `run_pipeline.py --direct`).
4. **Says `ACCEPTED` but nothing turns** → almost certainly **THROTTLE**. In ACRO
   the attitude controller's authority scales with throttle: at minimum throttle
   the mixer has no headroom to differentiate the motors, so yaw/pitch commands
   produce almost no torque. **The pilot must hold mid throttle.** This is
   firmware behaviour and cannot be fixed in software.
5. **It turns the WRONG WAY** — away from the head, into the gimbal stop. That
   axis is reversed: set `YAW_REVERSED=1` or `PITCH_REVERSED=1`. **Do not "fix"
   it by flipping a sign in the control law** — the law's signs are verified by
   the test suite.
6. **`ACRO_TRAINER` != 0** auto-levels and limits lean angle, fighting a
   sustained pitch command. Set it to 0 for the bench.
7. **yaw/pitch on a channel > 8** — `RC_CHANNELS_OVERRIDE` only reaches channels
   1–8. Remap `RCMAP_YAW`/`RCMAP_PITCH`.

---

## 7. Tuning

One knob at a time, and read the CSV between changes.

| Symptom | Change |
|---|---|
| Too slow to centre | `KP_YAW` / `KP_PITCH` up (2.0 → 3.0) |
| Oscillates around centre | `KP_*` down, or `KD_YAW`/`KD_PITCH` up to 0.1–0.3 |
| Twitches on a centred target | `DEADBAND_FRAC` up (0.03 → 0.05) |
| Too violent overall | `STICK_LIMIT` down (0.55 → 0.35) |
| Jerky stick steps | `SLEW_PER_S` down (3.0 → 1.5) |

### Full env reference (defaults from `fc_control.commander_from_env`)

| Variable | Default | Meaning |
|---|---|---|
| `MAV_CONN` | `udpin:127.0.0.1:14551` | FC link; a serial path works; `none` = offline |
| `FC_TRANSPORT` | `auto` | `auto` \| `mavlink` \| `msp` \| `none` |
| `CONTROL_AXES` | `yaw,pitch` | Axes the tracker drives |
| `MAV_SOURCE_SYSTEM` | *read from FC* | Override the sysid we send from. Leave unset. |
| `CAM_HFOV_DEG` / `CAM_VFOV_DEG` | `62.2` / `48.8` | True lens FOV |
| `KP_YAW` / `KP_PITCH` | `2.0` | rad/s of body rate per rad of angular error |
| `KD_YAW` / `KD_PITCH` | `0.0` | Damping for latency-induced overshoot |
| `STICK_LIMIT` | `0.55` | Hard ceiling on deflection |
| `DEADBAND_FRAC` | `0.03` | Centre deadband, fraction of half-frame |
| `SLEW_PER_S` | `3.0` | Max stick change per second |
| `YAW_REVERSED` / `PITCH_REVERSED` | `0` | `1` if an axis drives the wrong way |
| `SEND_HZ` | `20` | Stream rate, independent of detection FPS |
| `REQUIRE_ARMED` | `1` | `0` drives sticks while disarmed (bench only) |
| `REQUIRE_RATE_MODE` | `1` | Steer only in ACRO; the pilot's mode switch is the handover. `0` = ungated |
| `NO_TARGET_BEHAVIOR` | `release` | `release` \| `hold` |

Launcher flags: `--no-mp`, `--direct` (no router; removes it as a failure mode),
`--no-verify`.

**On FOV:** `62.2 x 48.8` is the Pi-cam-v2 / IMX219 full-frame spec. The pipeline
runs sensor mode 4 (1280x720), which on IMX219 is a **cropped** mode, so the true
FOV is likely narrower. A wrong FOV only **rescales the loop gain**, so do not
chase the optics — tune `KP_*` instead.

---

## 8. Verification status

There is no FC on the development laptop, so the control path was validated
against a **simulated ArduPilot** (`tests/fake_fc.py`) that reproduces the real
sysid-gating rule and the `RC_CHANNELS` override echo.

```bash
python3 tests/test_control_law.py     # 36 checks
python3 tests/test_simulated_fc.py    # 43 checks
```

* `test_control_law.py` — sign conventions, deadband continuity, saturation,
  slew, inversion flags, RC PWM mapping with asymmetric trim, MSP CRC framing, a
  closed-loop kinematic sim proving convergence at KP 1/2/3, and a negative
  control proving a genuinely reversed axis **diverges** (so a real polarity
  error would be caught).
* `test_simulated_fc.py` — case **A** default `SYSID_MYGCS=255`; **B** non-default
  `42` (proves it is read, not assumed); **C** *the regression*: sending from
  sysid 1 against `SYSID_MYGCS=255` must be refused **and reported**; **D**
  `RC_OPTIONS=2` detected; **E** disarmed ⇒ sticks stay released; **F** *the mode
  gate*: STABILIZE ⇒ nothing driven, ACRO ⇒ driven, back to STABILIZE ⇒ released,
  ACRO again ⇒ driven again (proves the gate is not a one-shot latch); **G**
  `REQUIRE_RATE_MODE=0` restores the ungated behaviour.
* A separate check rendered 72 HUD/console permutations without exception.

### Verified on real hardware 2026-08-04

Bench session on the Jetson with the real TBS Lucid H7, **props off**, pilot armed in
ACRO at mid throttle:

* Firmware confirmed **ArduPilot Copter**; `SYSID_MYGCS=255`; overrides **ACCEPTED**.
* Fixed on the board: `ACRO_TRAINER` 2 → **0**, `ACRO_RP_EXPO` 0.3 → **0** (the
  control law's stick→rate map is linear and assumes no expo), `ACRO_Y_EXPO` 0.
* Live run: **656/656 frames LOCK**, **619 `TX->FC (confirmed)`, 0 `REFUSED`**, and
  **2260 direction checks with 0 wrong** on all four of right/left/above/below.
  Physical yaw-right confirmed by eye.
* On disarm mid-run the sticks released immediately.
* At `KP 3.5/6.0`, `KD 0.15`: mean error 30 px, 37 % of frames centred, **3 stick
  reversals in 78 s** (no oscillation), peak stick **0.16 yaw / 0.10 pitch** against
  a `STICK_LIMIT` of 0.35 — i.e. large unused authority, so the gains can go higher.
  Pitch is the weak axis because `ACRO_RP_RATE=360` makes stick = rate/360; raise
  `KP_PITCH`, do **not** lower `ACRO_RP_RATE` (that changes the pilot's own feel).

Still unverified: **behaviour with props on**, and the mode gate against the real
1 Hz `HEARTBEAT` (simulated only, cases F/G).

---

## 9. Open items / known gaps

* **Firmware on the board is unconfirmed.** `fc_check.py` answers it in seconds.
* **True lens FOV is assumed**, see §7.
* **`src/product/wasp_tracker/control/ardupilot.py` still has the identical
  GUIDED-only defect.** Only the development pipeline (`src/development/modules/
  object_tracking/`) was fixed. The product package is a separate
  config.yaml-driven architecture and needs the same treatment before it ships.
* `run_video_pilot.py` is stale and does not run (see its header).
* `pyserial` is required for the MSP/Betaflight path — confirm it is installed in
  `yolo-env` if the board turns out to be on Betaflight.

## 10. Hard rules for anyone editing this

1. **Do not reintroduce GUIDED velocity setpoints.** They cannot steer this rig.
2. **Do not hardcode a MAVLink source system id.** Read `SYSID_MYGCS` from the FC.
3. **Do not make the HUD claim `TX` for a command that was merely computed.**
   That single lie cost a whole demo.
4. **Do not fix a wrong-direction axis by flipping a sign in `ControlLaw`.** Use
   `YAW_REVERSED` / `PITCH_REVERSED`. The law's signs are test-locked.
5. **Do not remove the release-on-exit path** in `tracker.run()`'s `finally`.
6. **Do not restore "no target ⇒ fly forward at max speed".**
7. Run `tests/` after any change to the control path.
8. **Props off** for every bench test.
