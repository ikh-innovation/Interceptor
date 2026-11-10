# WASP: flight controller settings

ArduCopter settings for a WASP drone: the values the app requires, the values it
recommends, and the values the pilot has to decide. "Now" is what the TBS LUCID H7
connected to this Jetson read back on 2026-10-28 (the new FC, after the one-by-one check and
with GPS and compass switched on). The full set is in `deploy/fc/TBS_LUCID_H7_2026-10-28.param`;
`src/tests/fc_param_check.py` compares an FC with it and writes the differences one by one. Change parameters in Mission
Planner or QGC (Full Parameter List → Write Params → Refresh Params, and check that
the value stuck). Then restart the app (`systemctl --user restart wasp-tracker`),
because it reads the parameters only once, when it connects.

## 1. Required: WASP will not launch without these

The app checks all of these before ARM and LAUNCH WASP and shows the first one that fails
under the ARM button.

| Parameter | Needed | Now | Why |
| --- | --- | --- | --- |
| `ACRO_TRAINER` | **0** | 0 | WASP sends body rates in ACRO. Values 1 and 2 level the drone and limit the lean, which fights the launch. Required since WASP was added (2026-10-07). |
| `RC_OVERRIDE_TIME` | **> 0 and ≤ 1** | 1 | If the Jetson hangs, its last command lapses after 1 s, not 3. |
| `RC_OPTIONS` bit 1 | **clear** | 32 (clear) | Bit 1 makes the FC ignore every MAVLink override. |
| `ACRO_RP_RATE`, `ACRO_Y_RATE` | > 0 | 360, 202.5 | The app converts its rate commands to stick positions with these. |
| `ACRO_RP_EXPO`, `ACRO_Y_EXPO` | 0 to 1 | 0.3, 0 | Same reason. |
| `RC1..RC4_MIN/MAX/TRIM/DZ`, `RCMAP_*` | calibrated, 4 different channels within 1–8 | `RCMAP_THROTTLE` 1, ROLL 2, PITCH 3, YAW 4; throttle RC1 988 / 990 / 2010 | The app maps 0–100 % throttle onto RC1_MIN..RC1_MAX (the throttle channel `RCMAP_THROTTLE` names). Recalibrate the radio on a new FC or transmitter. |
| `SYSID_MYGCS` | the Jetson's system ID | 255 | Overrides from any other system ID are ignored. |
| Flight mode | **ACRO** (switch down) | n/a | The phone's ARM is refused in any other mode; LAUNCH WASP needs the FC armed for 2.5 s. |
| `FLTMODE_GCSBLOCK` | bits 0 and 2 **clear** | 0 | SafeDown switches the FC to STABILIZE / ALT_HOLD / LAND over MAVLink. |
| `FLTMODE_CH` / `FLTMODE1..6` | ch6: low ACRO | 6 / 1,0,0,0,0,2 | **Low (ACRO) is required for launch and for all of WASP.** Middle/high no longer have app-level meaning — if moved in flight, the Jetson releases every channel and the FC flies stock in whatever mode is selected. No more "catch via STABILIZE" or "SafeDown via ALT_HOLD". |
| `FS_GCS_ENABLE` | **1** (RTL) or 2 (LAND) | 1 | **New requirement.** If the Jetson dies mid-air (cable, power, SoC crash), the FC falls back to RTL (1) or LAND (2) on its own. ARM is refused if this is 0. |
| `FS_GCS_TIMEOUT` | **≤ 5** seconds | 3 | How long the FC waits without a Jetson heartbeat before the failsafe fires. 3 s is a good balance between false triggers and reaction speed. |
| GPS and compass | **Optional** — only needed if geofence is used | `COMPASS_ENABLE 1`, `EK3_SRC1_POSXY 3`, `EK3_SRC1_VELXY 3` | **Changed.** WASP no longer requires GPS to launch. GPS + compass are only needed if you arm a geofence (so the drone can RTL to the launch point when it breaches the fence) or want the Jetson-failsafe to RTL instead of just LAND. Without them, WASP works and the SAFEDOWN button lands in place. |

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
| `MOT_THST_HOVER` | **0.152** | Learned in flight (`MOT_HOVER_LEARN` 2); it was 0.125, ArduCopter's minimum, before. Real now. |
| `MOT_HOVER_LEARN` | 2 | Learns the hover throttle and saves it (it only learns in ALT_HOLD-type modes). |
| `ACRO_THR_MID` | 0 | 0 = ACRO uses `MOT_THST_HOVER` for its mid-stick. |
| `MOT_THST_EXPO`, `MOT_SPIN_ARM/MIN/MAX` | 0.62, 0.10 / 0.13 / 0.95 | |

In STABILIZE and ACRO, ArduCopter bends the throttle stick so that **mid-stick
equals the hover throttle**. With the hover throttle at 0.152 the curve is nearly
cubic:

| Stick (or WASP throttle) | 50 % | 75 % | 85 % | 100 % |
| --- | --- | --- | --- | --- |
| Motor output | 15 % | 45 % | 63 % | 100 % |

So in STABILIZE (the catch) a centred throttle stick is about a hover. Before
the app gives the sticks back it writes the thrust it just measured into
`MOT_THST_HOVER` on this FC (the FC's own limits, 0.125 to 0.6875) and the FC
saves it. Do not copy this board's number onto another drone. The rest of this
section is how to check it again after a change of battery, props or payload.

If the drone really needs about 45–60 % to hover, then:
- WASP at 75 % or 85 % can't climb, and only 100 % takes off.
- In ACRO and STABILIZE the pilot has to hold the stick far above mid just to
  hover, and a centred stick drops the drone.
- **At takeover the app has already set mid-stick to this drone's hover**, not to
  another board's saved number.

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
| `FS_THR_ENABLE` | **1 (RTL)**, written 2026-10-30 (was 3, LAND) | On radio loss the FC flies home (the GO spot: the app sets home at LAUNCH WASP) and lands; LAND where it is without a GPS position. **It works only while the app is not sending sticks**: after the CATCH (the pilot flying), or before launch. During a WASP flight the app's RC override counts as radio input for ArduCopter, so the FC never sees the loss (shown in SITL and in ArduCopter's `radio.cpp`). Until the app watches the receiver itself, a saved geofence is what brings a WASP flight home after a radio loss. In ACRO or STABILIZE with the throttle at zero ArduCopter can also **disarm at once, even in the air**: keep the transmitter on and bound for the whole flight. |
| `FS_THR_VALUE` | 975 | Must stay more than 10 below the throttle's RC1_MIN (988): it is, by 3. Recalibrating the radio lower than 986 makes the FC refuse to arm; then set `FS_THR_VALUE` to 950. |
| `FS_GCS_ENABLE` | 0 (off) | The Jetson isn't treated as a GCS for failsafe. Keep it 0. |
| `BATT_FS_LOW_ACT` | 1 (LAND) at `BATT_LOW_VOLT` 10.5 V | Voltage sag at 100 % throttle can trigger LAND. The mode change out of ACRO ends WASP and the FC starts landing. Check the setting against your pack. |
| `FS_CRASH_CHECK` | 1 | Not active in ACRO. |
| `DISARM_DELAY` | 10 s | Auto-disarms on the ground after 10 s at zero throttle. Press LAUNCH WASP within 10 s of ARM. |
| `FENCE_ACTION` | **0 (none)**, changed 2026-11-09 (was 1, RTL) | **Must be 0.** The geofence is the app's, not the FC's. With `FENCE_ACTION = 1` the FC forces RTL whenever the altitude exceeds `FENCE_ALT_MAX` (100 m), which fights SafeDown's own mode changes and caused the drone to land far away on 2026-11-09. The app's geofence handles the boundary; the FC's fence must not interfere. |
| `FENCE_ENABLE` | 0 | The app's geofence (section 7) is used instead. |

## 6. SafeDown: settings it uses

SafeDown flies the drone home. **Pilot switch up (ALT_HOLD) or geofence breach:**
the app requests RTL directly to the launch-point GPS coordinates (the GO spot,
saved at LAUNCH WASP with `DO_SET_HOME`). **SAFEDOWN button:** levels and stops
the drone (in STABILIZE, then ALT_HOLD), then RTL.
With no GPS position, or below 5 m, it lands where it is (LAND).

| Parameter | Now | What it does in SafeDown |
| --- | --- | --- |
| `RTL_ALT` | 1500 | Climbs to at least 15 m before flying home. Raise it if there are trees or buildings on the way. |
| `RTL_SPEED` | 0 | 0 = `WPNAV_SPEED` (2000 = 20 m/s). Fast; 1000 (10 m/s) is gentler and still quick. |
| `RTL_LOIT_TIME` | 5000 | Hovers 5 s over the GO spot before it descends. |
| `RTL_ALT_FINAL` | 0 | Lands and disarms at home. |
| `FS_EKF_ACTION` | 1 (LAND) | If the EKF fails the FC lands on its own; SafeDown then never tries RTL again. |
| `LAND_SPEED` | 50 | Descent in the last `LAND_ALT_LOW` (10 m), cm/s. |
| `LAND_SPEED_HIGH` / `WPNAV_SPEED_DN` | 0 / 250 | Above 10 m LAND comes down at 2.5 m/s (2.8 m/s measured in SITL). Set `LAND_SPEED_HIGH` lower for a gentler but longer descent. |
| `PILOT_ACCEL_Z` | 500 | ALT_HOLD stops a climb or sink at only 5 m/s². That is why SafeDown first stops fast climbs and dives itself, in STABILIZE. |
| `MOT_THST_HOVER` | 0.152 on this board | Do not copy it. On a STAB catch the app writes the thrust it measured and the FC saves that. |
| (none: the IMU) | | Air brake: without GPS the app leans the drone against the accelerometer's drag reading (at most 12°) so it stops relative to the air even when the EKF's "level" is off. It can't cancel the wind itself. |
| `DISARM_DELAY`, land detector | 10 s | LAND disarms on touchdown by itself. |
| `COMPASS_ENABLE`, `EK3_SRC1_POSXY`, `EK3_SRC1_VELXY` | 1, 3, 3 | **Required.** Without a heading the EKF can't fuse GPS at all, even with a fix, so the GPS module must have a compass and it must be calibrated. With a GPS position the app's air brake steps aside and RTL or LAND hold position themselves. If GPS is lost on the way home it lands where it is. |

## 7. How WASP uses these settings

1. **Arm.** The phone's **ARM**: in ACRO with the throttle stick at zero, the app runs the arm
   switch's own function over MAVLink (`MAV_CMD_DO_AUX_FUNCTION` with `RC5_OPTION` = 154,
   switch HIGH), so the FC arms exactly as from the switch, airmode included. The physical
   switch stays low: an emergency disarm in the air is the switch UP and back DOWN.
   **LAUNCH WASP** is offered 2.5 s after arming, once the FC's 2 s arming delay is over.
2. **Launch.** Pressing LAUNCH WASP (there is no timer any more) sends full launch throttle at once. After
   the straight phase the app steers with body rates in ACRO.
3. **Hold.** WASP keeps the sticks until SafeDown, a mode change or a disarm.
   There is no time limit. Short telemetry gaps keep the launch throttle with zero
   steering.
4. **SafeDown (switch up / ALT_HOLD).** The pilot's switch **up** (ALT_HOLD):
   the app requests **RTL directly** to the GO spot (the launch-point GPS
   coordinates) and the FC flies home and lands. No levelling or stopping
   first — RTL handles the braking itself. See `pilot-instructions-el.md`.
   The phone's **SAFEDOWN button** does the same except it levels and stops
   (STABILIZE → ALT_HOLD) before the RTL, in case the pilot's switch is not
   on ALT_HOLD.
5. **Plan B (catch).** The pilot sets the throttle stick to mid, then switches to
   the **middle** (STABILIZE). The app levels it and brakes to a hover (barometer
   for the climb, air drag for the lean) and returns every channel once it is
   level and the height is held — or **after 5 seconds at most**, whichever comes
   first. The pilot is never locked out longer than that.
   Switch up again to send it home (RTL).
   The pilot can also switch to STABILIZE during a SafeDown that is already
   running; the app turns it into a catch and hands the sticks back when
   it hovers (same 5 s limit).
6. **Next launch.** Disarm before the next ARM; WASP stays off until then.
7. **Geofence.** A polygon drawn or typed on the phone (`~/.config/wasp/geofence.json`). While it
   is saved (there is no on / off: Clear and Save removes it), ARM needs the drone inside it,
   and leaving it during a WASP flight triggers **RTL directly** to the GO spot (the launch-point
   GPS coordinates), the same as the pilot's switch up.
   It is the app's fence, not the FC's. **Set `FENCE_ACTION = 0` on the FC** (section 5):
   the FC's own fence must not fight the app's mode changes.

Pure ACRO with `ACRO_TRAINER = 0` doesn't level itself. Don't hand-fly it unless
the pilot is trained for it. For hand-flying practice in ACRO, set it back to 2, and
return it to 0 before a WASP flight.

## 8. GPS and compass: needed for SafeDown home, ARM and the geofence

"On the FC" is what the FC has on 2026-10-30, with the GPS and compass module fitted. It is a
**CAN (DroneCAN) module**: GPS and compass both come over CAN, not SERIAL2. Calibrate the
compass in Mission Planner after fitting it (Setup → Compass → Onboard Mag Calibration,
outdoors, then reboot); until then its offsets are 0.

| Parameter | Set to | On the FC | Why |
| --- | --- | --- | --- |
| `GPS1_TYPE` | **9** (DroneCAN) for a CAN module, 1 (auto, serial) for a serial one | 9 | Where the FC looks for the GPS. |
| `CAN_P1_DRIVER` / `CAN_P1_BITRATE` / `CAN_D1_PROTOCOL` | 1 / 1000000 / 1 (DroneCAN) | 1 / 1000000 / 1 | The CAN port the module is on. Needed for the CAN GPS and compass. |
| `SERIAL2_PROTOCOL` / `SERIAL2_BAUD` | 5 / 57 | 5 / 57 | Only for a serial GPS (TX to R2, RX to T2). Unused with the CAN module; harmless. |
| `GPS_AUTO_CONFIG` | 1 | 1 | The FC sets up the receiver itself. |
| `COMPASS_ENABLE` / `COMPASS_USE` | 1 / 1 | 1 / 1 | **Required.** Without a heading the EKF never uses the GPS. |
| `COMPASS_AUTODEC` | **1** | 1 (written 2026-10-30) | Magnetic declination from the GPS position (about 5° in Greece). At 0 the heading is off by that much. |
| `COMPASS_ORIENT` | as Mission Planner detects | 0 | Set by the calibration for an external module. |
| `COMPASS_DEV_ID` / `COMPASS_EXTERNAL` | not 0 (set by the FC) / 1 | 97539 (DroneCAN) / 1 | 0 = no compass found. |
| `COMPASS_OFS_X/Y/Z` | set by the calibration | **0, 0, 0** | Not calibrated yet. |
| `EK3_SRC1_POSXY` / `EK3_SRC1_VELXY` | 3 / 3 (GPS) | 3 / 3 | Position and velocity from the GPS. |
| `EK3_SRC1_POSZ` / `EK3_SRC1_YAW` | 1 (baro) / 1 (compass) | 1 / 1 | Height from the barometer, heading from the compass. |
| `FS_EKF_ACTION` | 1 (LAND) | 1 | A failed EKF lands; SafeDown then never fights it with RTL. |
| `RTL_ALT`, `RTL_SPEED`, `RTL_LOIT_TIME`, `RTL_ALT_FINAL` | see section 6 | 1500, 0, 5000, 0 | How SafeDown flies home. |
| `RC5_OPTION` | **154** (arm / disarm switch) | 154 | The phone's ARM runs this switch's function. Without it the app sends a plain arm command (no airmode). |
| `ARMING_CHECK` | 8524 recommended (adds compass and GPS checks) | 8512 | The FC itself then refuses to arm without a healthy compass and a GPS lock. The app checks both anyway. |
| `FENCE_ENABLE` | 0 | 0 | The geofence is the app's (section 7), not the FC's. `FENCE_ACTION` must also be 0 (section 5). |

The phone's tile **GPS/Compass** reads `YES / YES` only with a 3D fix, at least 6 satellites,
and a compass the FC reports present, enabled and healthy. ARM needs both **only if a geofence
is armed** — without a geofence, WASP launches without GPS and the SAFEDOWN button lands in
place.

---

## Complete Mission Planner parameter list (write these to the FC)

Open Mission Planner → **CONFIG → Full Parameter List** (or **Full Parameter Tree**), paste the
values below, click **Write Params**, then **Refresh Params** and verify each value stuck.
Restart the app on the Jetson afterwards:
```
sudo systemctl --user restart wasp-tracker
```

Alternative: run [scripts/set-fc-failsafe.py](scripts/set-fc-failsafe.py) with wasp-tracker
stopped — it sets `FS_GCS_ENABLE` and `FS_GCS_TIMEOUT` over MAVLink in one shot. The rest
of the parameters you still set in Mission Planner.

### Mandatory (WASP will not arm without these)

| Parameter | Value | Note |
| --- | --- | --- |
| `ACRO_TRAINER` | `0` | ACRO sends raw body rates — no auto-level, no lean limit. |
| `ACRO_RP_RATE` | `360` | Roll/pitch rate the app assumes. |
| `ACRO_Y_RATE` | `202.5` | Yaw rate the app assumes. |
| `ACRO_RP_EXPO` | `0.3` | Must be 0..1. |
| `ACRO_Y_EXPO` | `0` | Must be 0..1. |
| `RC_OVERRIDE_TIME` | `1` | Override lapses after 1 s of Jetson silence (FC falls back to pilot sticks). |
| `RC_OPTIONS` | `32` (any value with bit 1 clear) | Bit 1 set = FC ignores all GCS overrides — DO NOT set it. |
| `FLTMODE_GCSBLOCK` | `0` | SafeDown needs to put the FC into STABILIZE / ALT_HOLD / LAND. |
| `FLTMODE_CH` | `6` | Pilot mode switch channel. |
| `FLTMODE1` | `1` (ACRO) | Switch DOWN — required for WASP launch. |
| `FS_GCS_ENABLE` | `1` (RTL) or `2` (LAND) | **NEW:** Jetson-dead failsafe. RTL needs GPS+compass; LAND works without. **The pilot can override the failsafe mid-flight by flipping the FC mode switch on the transmitter to STAB or ACRO — ArduPilot drops RTL immediately when the pilot picks a new mode.** |
| `FS_GCS_TIMEOUT` | `2` (seconds, ArduPilot minimum) | How fast the FC reacts when Jetson heartbeats stop. The Jetson sends heartbeats at 5 Hz, so worst case Jetson-death → RTL is ~2 s. Max allowed by the app: 3. |
| `SYSID_MYGCS` | `255` | Must match the Jetson's sysid (hard-coded default). |
| `RC5_OPTION` | `154` | Channel 5 is the arm switch. |

### Optional (recommended by the app's checklist)

| Parameter | Value | Note |
| --- | --- | --- |
| `ATC_INPUT_TC` | `0.15`–`0.25` | Softer FC response to commanded attitude. |
| `INS_LOG_BAT_MASK` | `1` | Needed for FFT vibration analysis. |
| `LOG_BACKEND_TYPE` | `1` | SD card logs. Crash diagnosis impossible without this. |
| `ANGLE_MAX` | `4000` | Centi-degrees. 40° max lean in ALT_HOLD / STABILIZE. |
| `PILOT_SPEED_UP` | `2500` | 25 m/s climb limit (ALT_HOLD only). |
| `PILOT_SPEED_DN` | `250` | 2.5 m/s descent limit (ALT_HOLD only). |
| `PILOT_ACCEL_Z` | `500` | 5 m/s² vertical accel limit. |
| `MOT_HOVER_LEARN` | `2` | Learns and saves hover throttle. |
| `ACRO_THR_MID` | `0` | 0 = ACRO mid-stick uses MOT_THST_HOVER. |

### GPS / compass (write these only if using geofence or RTL-on-Jetson-loss)

| Parameter | Value | Note |
| --- | --- | --- |
| `COMPASS_ENABLE` | `1` | Required for EKF to use GPS. |
| `COMPASS_USE` | `1` | |
| `COMPASS_AUTODEC` | `1` | Auto magnetic declination from GPS. |
| `GPS1_TYPE` | `9` (DroneCAN) or `1` (serial auto) | Depending on your GPS wiring. |
| `GPS_AUTO_CONFIG` | `1` | |
| `EK3_SRC1_POSXY` | `3` | GPS as horizontal position source. |
| `EK3_SRC1_VELXY` | `3` | GPS as velocity source. |
| `EK3_SRC1_POSZ` | `1` | Baro as altitude source. |
| `EK3_SRC1_YAW` | `1` | Compass as heading source. |
| `FS_EKF_ACTION` | `1` (LAND) | EKF failure lands; prevents RTL fighting a bad EKF. |
| `FENCE_ENABLE` | `0` | We use the app's geofence, not the FC's. |

### Return-to-launch tuning (if FS_GCS_ENABLE=1 or you use geofence)

| Parameter | Value | Note |
| --- | --- | --- |
| `RTL_ALT` | `1500` (15 m) | RTL cruise altitude above home. |
| `RTL_SPEED` | `0` (use WPNAV_SPEED) | 0 means "use the waypoint-nav speed". |
| `RTL_LOIT_TIME` | `5000` (5 s) | Loiter time over home before descent. |
| `RTL_ALT_FINAL` | `0` (land) | 0 = land at home. |

### Changing parameters from the Jetson (no Mission Planner needed)

For the two new mandatory failsafe params:
```bash
sudo systemctl stop wasp-tracker        # release the serial port
~/Documents/Interceptor/.env/yolo/bin/python3 \
  ~/Documents/Interceptor/scripts/set-fc-failsafe.py
sudo systemctl start wasp-tracker
```

For everything else — easiest is Mission Planner over USB from a laptop.

### After writing parameters

1. Verify in Mission Planner: Refresh Params → search for each name → confirm the value.
2. Restart the app on the Jetson so it re-reads the new values:
   ```bash
   sudo systemctl --user restart wasp-tracker
   ```
3. On the phone, try to ARM. If a mandatory param is missing or wrong, the ARM button
   stays grey and shows the specific parameter that failed under it.
