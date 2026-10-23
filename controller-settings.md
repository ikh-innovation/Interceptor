# WASP: flight controller settings

ArduCopter settings for a WASP drone: the values the app requires, the values it
recommends, and the values the pilot has to decide. "Now" is what the TBS LUCID H7
connected to this Jetson read back on 2026-10-22. Change parameters in Mission
Planner or QGC (Full Parameter List → Write Params → Refresh Params, and check that
the value stuck). Then restart the app (`systemctl --user restart wasp-tracker`),
because it reads the parameters only once, when it connects.

## 1. Required: WASP will not launch without these

The app checks all of these before GO and shows the first one that fails as the
WASP setup reason in the UI.

| Parameter | Needed | Now | Why |
| --- | --- | --- | --- |
| `ACRO_TRAINER` | **0** | 0 | WASP sends body rates in ACRO. Values 1 and 2 level the drone and limit the lean, which fights the launch. Required since WASP was added (2026-10-07). |
| `RC_OVERRIDE_TIME` | **> 0 and ≤ 1** | 1 | If the Jetson hangs, its last command lapses after 1 s, not 3. |
| `RC_OPTIONS` bit 1 | **clear** | 32 (clear) | Bit 1 makes the FC ignore every MAVLink override. |
| `ACRO_RP_RATE`, `ACRO_Y_RATE` | > 0 | 360, 202.5 | The app converts its rate commands to stick positions with these. |
| `ACRO_RP_EXPO`, `ACRO_Y_EXPO` | 0 to 1 | 0.3, 0 | Same reason. |
| `RC1..RC4_MIN/MAX/TRIM/DZ`, `RCMAP_*` | calibrated, 4 different channels within 1–8 | RC3 1100/1500/1900 | The app maps 0–100 % throttle onto RC3_MIN..RC3_MAX. |
| `SYSID_MYGCS` | the Jetson's system ID | 255 | Overrides from any other system ID are ignored. |
| Flight mode | **ACRO**, armed | n/a | GO is refused in any other mode or while disarmed. |
| `FLTMODE_GCSBLOCK` | bits 0 and 2 **clear** | 0 | SafeDown switches the FC to STABILIZE and ALT_HOLD over MAVLink. LAND can't be blocked. |
| `FLTMODE_CH` / `FLTMODE1..6` | ch6: low ACRO, middle STABILIZE, high ALT_HOLD | 6 / 1,0,0,0,0,2 | The middle position is the pilot's SafeDown button. The app reads the switch at connect. |

## 2. Recommended by the app's checklist (not blocking)

| Parameter | Recommended | Now | Why |
| --- | --- | --- | --- |
| `ATC_INPUT_TC` | 0.2–0.25 | **0.15** | Softer response to a new command, so less kick. |
| `INS_LOG_BAT_MASK` | 1 | **0** | IMU batch log, needed to check vibration (FFT). |
| `LOG_BACKEND_TYPE` | 1, with an SD card fitted | 1 | Without FC logs a crash can't be diagnosed. |
| `ANGLE_MAX` | 4000 | 4000 | ok |
| `PILOT_SPEED_UP` / `PILOT_SPEED_DN` / `PILOT_ACCEL_Z` | 2500 / 250 / 500 | 2500 / 250 / 500 | ok (only used by the ALT_HOLD modes) |

## 3. Throttle curve: check before the next flight

| Parameter | Now | Note |
| --- | --- | --- |
| `MOT_THST_HOVER` | **0.125** | This is ArduCopter's **minimum** allowed value. A learned hover stuck at the floor is suspicious. |
| `MOT_HOVER_LEARN` | 2 | Learns the hover throttle and saves it (it only learns in ALT_HOLD-type modes). |
| `ACRO_THR_MID` | 0 | 0 = ACRO uses `MOT_THST_HOVER` for its mid-stick. |
| `MOT_THST_EXPO`, `MOT_SPIN_ARM/MIN/MAX` | 0.62, 0.10 / 0.13 / 0.95 | |

In STABILIZE and ACRO, ArduCopter bends the throttle stick so that **mid-stick
equals the hover throttle**. With the hover throttle at 0.125 the curve is fully
cubic (motor output = stick³):

| Stick (or WASP throttle) | 50 % | 75 % | 85 % | 100 % |
| --- | --- | --- | --- | --- |
| Motor output | 12.5 % | 42 % | 61 % | 100 % |

If the drone really needs about 45–60 % to hover, then:
- WASP at 75 % or 85 % can't climb, and only 100 % takes off.
- In ACRO and STABILIZE the pilot has to hold the stick far above mid just to
  hover, and a centred stick drops the drone.
- **At takeover, putting the stick at mid in STABILIZE gives 12.5 % thrust.**

**Check:** in a steady STABILIZE hover, where is the pilot's throttle stick?
- Near mid: the drone really does hover at about 12.5 %, and this section isn't
  the problem.
- Around 75–85 %: the hover value is wrong. Find the real hover throttle in the FC
  log (`CTUN.ThO` during a steady hover). Set `MOT_THST_HOVER` to it, or let it
  re-learn in ALT_HOLD with `MOT_HOVER_LEARN = 2`. Set `ACRO_THR_MID` to the same
  value so ACRO matches. Then the stick at mid hovers in both modes, and the WASP
  75 / 85 / 100 % values will mean what they say.

## 4. Launch from a tripod or slope: the angle limit

During the straight phase WASP holds the launch angle with zero rates. Nothing
holds height then, so only the vertical part of the thrust (thrust × cos angle)
lifts the drone. It climbs only if that is more than its weight:

| Real hover thrust | 15° | 20° | 45° | 70° |
| --- | --- | --- | --- | --- |
| 0.25 (thrust-to-weight 4:1) | climbs | climbs | climbs | climbs |
| 0.35 (about 3:1) | climbs | climbs | climbs | **sinks** |
| 0.50 (2:1) | climbs | climbs | climbs, slowly | **sinks** |
| 0.70 (about 1.4:1) | climbs | climbs, slowly | **sinks** | **sinks** |

So the steepest usable launch angle is about arccos(real hover thrust), minus a
margin. At 70° the drone must hover at well under 34 % thrust. If it doesn't, it
goes forward and drops however the app is set. The real hover thrust is also what
tells you how hard the drone can accelerate.

- **Launch throttle:** use 100 %. At full stick the motors get full output no
  matter what `MOT_THST_HOVER` says, so the hover value doesn't change the launch
  itself.
- **Throttle ramp:** OFF (the default). The FC already spools up from ground idle
  over `MOT_SPOOL_TIME` (0.5 s on this FC), and a quick launch clears the tripod or
  slope sooner. A slow ramp leaves one leg in contact longer: in ACRO the FC keeps
  correcting against the contact, the correction builds up, and the drone can flip
  when it lets go. Only try 0.1 s if testing shows a problem with the motor spool.
- **Straight time:** at steep angles, use a shorter straight phase (0.5 s), or
  accept the height it loses before steering starts.
- **Measure the hover thrust before choosing angles.** Hover for 30–60 s in
  ALT_HOLD; with `MOT_HOVER_LEARN = 2` the FC learns `MOT_THST_HOVER` and saves it.
  Or read `CTUN.ThO` in a steady hover from the FC log. Don't type a value in by
  guesswork.

## 5. Failsafes: the pilot decides these

| Parameter | Now | What it does with WASP |
| --- | --- | --- |
| `FS_THR_ENABLE` | 3 (LAND) | On radio loss in ACRO or STABILIZE with the throttle at zero, ArduCopter **disarms immediately, even in the air**. Keep the transmitter on and bound for the whole flight. |
| `FS_THR_VALUE` | 975 | Below RC3_MIN (1100), so ok. |
| `FS_GCS_ENABLE` | 0 (off) | The Jetson isn't treated as a GCS for failsafe. Keep it 0. |
| `BATT_FS_LOW_ACT` | 1 (LAND) at `BATT_LOW_VOLT` 10.5 V | Voltage sag at 100 % throttle can trigger LAND. The mode change out of ACRO ends WASP and the FC starts landing. Check the setting against your pack. |
| `FS_CRASH_CHECK` | 1 | Not active in ACRO. |
| `DISARM_DELAY` | 10 s | Auto-disarms on the ground after 10 s at zero throttle. Press GO within 10 s of arming. |

## 6. SafeDown: settings it uses

| Parameter | Now | What it does in SafeDown |
| --- | --- | --- |
| `LAND_SPEED` | 50 | Descent in the last `LAND_ALT_LOW` (10 m), cm/s. |
| `LAND_SPEED_HIGH` / `WPNAV_SPEED_DN` | 0 / 250 | Above 10 m LAND comes down at 2.5 m/s (2.8 m/s measured in SITL). Set `LAND_SPEED_HIGH` lower for a gentler but longer descent. |
| `PILOT_ACCEL_Z` | 500 | ALT_HOLD stops a climb or sink at only 5 m/s². That is why SafeDown first stops fast climbs and dives itself, in STABILIZE. |
| `MOT_THST_HOVER` | 0.125 | The app's STABILIZE throttle starts from it, then learns the real hover from the IMU. |
| (none: the IMU) | | Air brake: without GPS the app leans the drone against the accelerometer's drag reading (at most 12°) so it stops relative to the air even when the EKF's "level" is off. It can't cancel the wind itself. |
| `DISARM_DELAY`, land detector | 10 s | LAND disarms on touchdown by itself. |
| `EK3_SRC1_POSXY`, `EK3_SRC1_VELXY` | 0, 0 | **No GPS in the EKF.** SafeDown levels the drone and stops it vertically, but can't hold position: it drifts with the wind. After a hard launch or a tumble the EKF's attitude and vertical speed can also be wrong (seen in SITL), which SafeDown checks against the barometer. To use the fitted GPS: `COMPASS_ENABLE = 1` and a compass calibration first (now 0: without a heading the EKF can't fuse GPS at all, even with a fix), then `EK3_SRC1_POSXY = 3` and `EK3_SRC1_VELXY = 3`. With a GPS position the app's air brake steps aside and LAND holds position itself. |

## 7. How WASP uses these settings

1. **Arm.** The pilot arms in ACRO with the throttle at zero. The app sends nothing.
2. **Launch.** Pressing GO (timer 0 s) sends full launch throttle at once. After
   the straight phase the app steers with body rates in ACRO.
3. **Hold.** WASP keeps the sticks until SafeDown, a mode change or a disarm.
   There is no time limit. Short telemetry gaps keep the launch throttle with zero
   steering.
4. **SafeDown.** The pilot's switch to the middle, or the phone's SAFEDOWN button,
   lands it on its own: level, stop, then LAND. See `pilot-instructions-el.md`.
5. **Plan B.** The pilot sets the throttle stick to mid, then switches **up** to
   ALT_HOLD. All channels return to the transmitter at once.
6. **Next launch.** Disarm before the next GO; WASP stays off until then.

Pure ACRO with `ACRO_TRAINER = 0` doesn't level itself. Don't hand-fly it unless
the pilot is trained for it. For hand-flying practice in ACRO, set it back to 2, and
return it to 0 before a WASP flight.
