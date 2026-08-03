# WASP head tracker — flight-control runbook

Detect a human head, and rotate the airframe until the bounding-box centre sits
on the frame centre. The drone is on a stationary gimbal, free in yaw and pitch,
flown by a pilot in **ACRO**.

---

## Why the drone was not moving

Two independent faults, both silent:

1. **The tracker transmitted nothing at all.** `run_pipeline.py` never set
   `CONTROL_MODE`, so `tracker.py` defaulted to `"guided"` and its transmit loop
   was gated on the FC being in GUIDED mode. The drone is flown in ACRO, so that
   gate was `False` on every frame and not one byte ever left the Jetson. The HUD
   still lit "TX", because it reported "a setpoint was computed", not "a setpoint
   was sent".

2. **Even the RC path was being discarded by the FC.** The tracker sent
   `RC_CHANNELS_OVERRIDE` from `source_system=1`. ArduPilot accepts overrides
   *only* from the system id in `SYSID_MYGCS` (default **255**):

   ```cpp
   void GCS_MAVLINK::handle_rc_channels_override(const mavlink_message_t &msg)
   {
       if (msg.sysid != sysid_my_gcs()) {
           return;            // dropped: no ACK, no STATUSTEXT, no log entry
       }
   ```

   So every override was thrown away with no diagnostic anywhere.

Beyond that, GUIDED velocity setpoints were the wrong idea for this airframe
regardless: they need an EKF position estimate (GPS / optical flow), which a
gimbal-mounted drone indoors does not have.

**What replaced it.** ACRO is the correct mode: a stick deflection *is* a body
angular-rate command, which is exactly what "rotate until centred" needs. The
tracker now reads `SYSID_MYGCS` off the FC and adopts it, then **verifies** the
override landed by watching the FC echo the value back in `RC_CHANNELS`. If it
does not land, it says so loudly instead of pretending to fly.

---

## Files

| File | Role |
|---|---|
| `fc_control.py` | Control law + FC transports (ArduPilot RC override, Betaflight MSP), transmit thread, safety gating, CSV log |
| `fc_check.py` | **Bench diagnostic.** Identifies the firmware, reports every setting that gates an override, proves an override is accepted, and can wiggle the sticks |
| `run_pipeline.py` | Launcher: identifies the firmware, wires the right transport, verifies overrides, then runs the tracker |
| `tracker.py` | Detection, Kalman tracking, HUD, and the per-frame control update |
| `wasp-pipeline.service` | systemd user unit for boot-time start |
| `tests/` | Self-test suite. Needs no flight controller and no camera. |

### Self-test (no hardware needed)

```bash
python3 tests/test_control_law.py     # signs, deadband, limits, slew, PWM mapping,
                                      # and a closed-loop sim proving convergence
python3 tests/test_simulated_fc.py    # full stack against a simulated ArduPilot that
                                      # reproduces the real sysid-gated override rule
```

`tests/fake_fc.py` is that simulator: it speaks MAVLink, serves parameters, and
drops overrides from the wrong system id exactly as ArduPilot does. Case **C**
reproduces the original bug (`source_system=1` vs `SYSID_MYGCS=255`) and asserts
it is now both refused *and* reported — so the regression cannot come back
silently. Run these after any change to the control path.

---

## Bring-up procedure

Run these in order. Do not skip step 1 — it answers the only question that
matters before YOLO is even involved.

### 1. What is on the board, and will it accept a stick command?

```bash
cd src/development/modules/object_tracking
python3 fc_check.py --verify
```

Safe with the FC disarmed: an override changes the FC's *stick input*, not the
motors. Reports the firmware, mode, arm state, `SYSID_MYGCS`, `RC_OPTIONS`,
`ACRO_TRAINER`, the ACRO full-stick rates, the RC channel map and calibration,
and the live receiver values — then sends a real override and confirms the FC
applied it.

Read the verdict:

* **`overrides : ACCEPTED`** → the transport works. Go to step 2.
* **`overrides : REFUSED`** → the printout names the cause in priority order.
* **`VERDICT: the board is running BTFL …`** → it is Betaflight, not ArduPilot.
  See *Betaflight* below.
* **`nothing answered`** → power, cable, port ownership, or `dialout` group.

If a MAVProxy router is already running it owns the FC serial port. Either probe
through it (`--connect udpin:127.0.0.1:14551`) or stop it (`pkill -f
mavproxy.py`) and probe the port directly.

### 2. Make it visibly move, without YOLO

**PROPS OFF.** Arm from the transmitter in ACRO and hold mid throttle, then:

```bash
python3 fc_check.py --wiggle 8 --yes
```

The airframe should sweep in yaw and pitch. The printout includes the motor
output spread, which is the real proof the FC turned the sticks into
differential thrust.

If the override is accepted but nothing turns, it is **throttle** — see
*Throttle authority* below.

### 3. Run the tracker

```bash
python3 run_pipeline.py --no-mp
```

Preflight identifies the firmware, starts the router, verifies overrides, and
only then loads YOLO. Watch the console:

```
[CTRL] LOCK    target +0142,-0071px -> centre | yaw +0.11 ( +22.3 d/s)  pitch +0.03 (+10.1 d/s) | TX->FC (confirmed)
```

`TX->FC (confirmed)` means bytes left the Jetson **and** the FC echoed them
back. The HUD lamp says the same thing:

| Lamp | Meaning |
|---|---|
| `DRIVE` | sticks sent and confirmed by the FC |
| `TX` | sticks sent, confirmation still pending |
| `REFUSED` | sticks sent but the FC is discarding them |
| `ARM?` | target locked, FC disarmed — nothing will move |
| `IDLE` | no target: channels released to the pilot |
| `SIM` | no FC attached |

### 4. Check the direction, then tune

If the drone turns **away** from the head and runs to the gimbal stop, that axis
is reversed. Set the flag and restart — do not change the gains:

```bash
YAW_REVERSED=1 python3 run_pipeline.py --no-mp     # or PITCH_REVERSED=1
```

Then tune, one knob at a time:

| Symptom | Change |
|---|---|
| Too slow to centre | `KP_YAW` / `KP_PITCH` up (2.0 → 3.0) |
| Oscillates around centre | `KP_*` down, or `KD_YAW`/`KD_PITCH` up to 0.1–0.3 |
| Twitches on a centred target | `DEADBAND_FRAC` up (0.03 → 0.05) |
| Too violent overall | `STICK_LIMIT` down (0.55 → 0.35) |
| Jerky stick steps | `SLEW_PER_S` down (3.0 → 1.5) |

Every run writes `data/control_<stamp>.csv` with the pixel error, the angular
error, both stick values, both commanded rates, and whether the override was
confirmed — plot that rather than guessing.

---

## Safety model

* Sticks are driven **only** while the FC is armed **and** a target is locked.
* Roll, throttle and arming are **never** touched — they stay with the pilot.
* Channels we do not drive are sent as `0`, which ArduPilot reads as "no
  override", so the pilot keeps them.
* The target goes stale after 0.5 s with no detection, and every channel is
  released.
* On exit — clean, Ctrl-C, or an unhandled exception — the release is sent
  repeatedly before the link closes.
* ArduPilot independently drops a stale override after `RC_OVERRIDE_TIME`
  (default 3 s), so a hung Jetson cannot hold the sticks indefinitely.

**Keep the props off for all bench work.**

---

## Throttle authority — the most likely remaining cause

In ACRO the attitude controller's authority scales with throttle. At minimum
throttle the mixer has no headroom to differentiate the motors, so a yaw or
pitch command produces almost no torque and the airframe will barely move even
though the override was accepted.

**The pilot must hold mid throttle for the airframe to respond.** This is
firmware behaviour, not something the tracker can work around.

Other non-software blockers, all reported by `fc_check.py`:

| Setting | Problem |
|---|---|
| `RC_OPTIONS` bit 1 (value 2) | FC ignores all GCS overrides. Clear it. |
| `ACRO_TRAINER` = 1 or 2 | Auto-levels and limits lean angle, fighting a sustained pitch command. Set to 0. |
| yaw/pitch on a channel > 8 | `RC_CHANNELS_OVERRIDE` only reaches channels 1–8. Remap `RCMAP_YAW` / `RCMAP_PITCH`. |
| No receiver signal | The pilot cannot arm, and a released channel has no source. |

---

## Betaflight

If `fc_check.py` reports `BTFL`, the board is not running ArduPilot and cannot be
driven over MAVLink at all.

**Recommended:** flash ArduPilot Copter (target `TBS_LUCID_H7`) and fly ACRO.
That gives parameter access, telemetry, Mission Planner, and the path this
pipeline is built around.

**Or** stay on Betaflight — the tracker supports MSP. `run_pipeline.py` detects
it and wires `FC_TRANSPORT=msp` straight at the serial port (no router, and
Mission Planner cannot share the port). Betaflight ignores MSP sticks until you
configure the override:

```
feature MSP_OVERRIDE
set msp_override_channels_mask = 10      # ch2 pitch + ch4 yaw
save
```

then assign **MSP RC OVERRIDE** to a spare aux switch in the Modes tab and turn
it on. That switch is also the pilot's kill switch for the tracker's authority.

---

## Configuration reference

All of it is environment variables, so the service unit and a manual run behave
identically. Defaults are in `fc_control.commander_from_env`.

| Variable | Default | Meaning |
|---|---|---|
| `MAV_CONN` | `udpin:127.0.0.1:14551` | FC link. A serial path works too. `none` = offline, compute and print only. |
| `FC_TRANSPORT` | `auto` | `auto` \| `mavlink` \| `msp` \| `none` |
| `CONTROL_AXES` | `yaw,pitch` | Which axes the tracker drives |
| `MAV_SOURCE_SYSTEM` | *(read from FC)* | Override the sysid we send from. Leave unset. |
| `CAM_HFOV_DEG` / `CAM_VFOV_DEG` | `62.2` / `48.8` | True lens FOV. A wrong value only rescales the gain. |
| `KP_YAW` / `KP_PITCH` | `2.0` | rad/s of body rate per rad of angular error |
| `KD_YAW` / `KD_PITCH` | `0.0` | Damping, for overshoot from detection latency |
| `STICK_LIMIT` | `0.55` | Never deflect a stick further than this |
| `DEADBAND_FRAC` | `0.03` | Centre deadband, fraction of the half-frame |
| `SLEW_PER_S` | `3.0` | Max stick change per second |
| `YAW_REVERSED` / `PITCH_REVERSED` | `0` | Set to `1` if an axis drives the wrong way |
| `SEND_HZ` | `20` | Stream rate, independent of detection FPS |
| `REQUIRE_ARMED` | `1` | `0` drives the sticks even while disarmed (bench only) |
| `NO_TARGET_BEHAVIOR` | `release` | `release` \| `hold` |

Launcher flags: `--no-mp` (skip Mission Planner), `--direct` (no router, tracker
owns the FC serial — removes the router as a failure mode), `--no-verify` (skip
the override check).
