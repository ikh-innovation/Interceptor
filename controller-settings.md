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
| `FLTMODE_GCSBLOCK` | bits 0 and 2 **clear** | 0 | SafeDown switches the FC to STABILIZE and ALT_HOLD over MAVLink, then to RTL or LAND. |
| `FLTMODE_CH` / `FLTMODE1..6` | ch6: low ACRO, middle STABILIZE, high ALT_HOLD | 6 / 1,0,0,0,0,2 | Low = WASP launch. High = SafeDown (fly back to the GO spot and land). Middle = catch, then the pilot flies in STABILIZE. The app reads the switch at connect. |
| GPS and compass | 3D fix with 6+ satellites, EKF using GPS | `COMPASS_ENABLE 1`, `EK3_SRC1_POSXY 3`, `EK3_SRC1_VELXY 3` | ARM is refused until the FC has a GPS position: SafeDown needs it to fly home. Power up flat (the EKF does not lock GPS on the tilted stand), then put it on the stand. `WASP_REQUIRE_GPS=0` turns the gate off for bench work. |

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

So in STABILIZE (the catch) a centred throttle stick is about a hover. The rest of this
section is how to check it again after a change of battery, props or payload.

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
| `FS_THR_VALUE` | 975 | Must stay more than 10 below the throttle's RC1_MIN (988): it is, by 3. Recalibrating the radio lower than 986 makes the FC refuse to arm; then set `FS_THR_VALUE` to 950. |
| `FS_GCS_ENABLE` | 0 (off) | The Jetson isn't treated as a GCS for failsafe. Keep it 0. |
| `BATT_FS_LOW_ACT` | 1 (LAND) at `BATT_LOW_VOLT` 10.5 V | Voltage sag at 100 % throttle can trigger LAND. The mode change out of ACRO ends WASP and the FC starts landing. Check the setting against your pack. |
| `FS_CRASH_CHECK` | 1 | Not active in ACRO. |
| `DISARM_DELAY` | 10 s | Auto-disarms on the ground after 10 s at zero throttle. Press LAUNCH WASP within 10 s of ARM. |

## 6. SafeDown: settings it uses

SafeDown levels and stops the drone itself (in STABILIZE, then ALT_HOLD), then
hands it to the FC's own **RTL**. At LAUNCH WASP the app saves the GPS spot and makes it
the FC's home (`DO_SET_HOME`), so RTL flies back to the stand, not to where it
was armed. With no GPS position, or below 5 m, it lands where it is (LAND).

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
| `MOT_THST_HOVER` | 0.152 | The app's STABILIZE throttle starts from it, then learns the real hover from the IMU. |
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
4. **SafeDown.** The pilot's switch **up** (ALT_HOLD), or the phone's SAFEDOWN
   button: level, stop, then RTL to the GO spot and land there. See
   `pilot-instructions-el.md`.
5. **Plan B (catch).** The pilot sets the throttle stick to mid, then switches to
   the **middle** (STABILIZE). The app levels and steadies it for 1–3 s, then all
   channels return to the transmitter. Switch up again to send it home.
6. **Next launch.** Disarm before the next ARM; WASP stays off until then.
7. **Geofence.** A polygon drawn or typed on the phone (`~/.config/wasp/geofence.json`). While it
   is saved (there is no on / off: Clear and Save removes it), ARM needs the drone inside it, and leaving it during a WASP flight starts SafeDown home.
   It is the app's fence, not the FC's (`FENCE_ENABLE` stays as it is).

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
| `FENCE_ENABLE` | 0 | 0 | The geofence is the app's (section 7), not the FC's. |

The phone's tile **GPS/Compass** reads `YES / YES` only with a 3D fix, at least 6 satellites,
and a compass the FC reports present, enabled and healthy. ARM needs both, and the EKF using
the GPS.
