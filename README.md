# WASP — Interceptor prototype

**Branch `prototype`. Read this file first.**

This is the entry point for two readers:

* **The pilot / operator in the field** — go straight to [§4, ΤΑ 10 ΒΗΜΑΤΑ](#4-τα-10-βήματα-στο-πεδίο). It is in Greek and non-technical, on purpose.
* **A fresh Claude Code session (e.g. running on the Jetson)** — read §1 through §3 and §5 through §8 before touching anything. They contain the problem statement, the design, what has been measured, what has *not*, and the traps that have already been paid for once.

Companion documents, both still current:

| File | What it is |
|---|---|
| [PROTOTYPE.md](PROTOTYPE.md) | Full technical status: both control laws, the safety model, the verification record, the env reference, the hard rules. The authoritative source when this README and it disagree. |
| [SCENARIO.md](SCENARIO.md) | The Greek field procedure in depth — the pre-battery checks, the tilt table, the 9-row scenario, what the pilot keeps hold of. |
| [docs/wasp-explained-gr.html](docs/wasp-explained-gr.html) | Source of the 5-page Greek explainer for a non-technical reader — no acronyms, read before the flight. **The built `WASP_EXPLAINED_GR.pdf` is deliberately not tracked** (removed in `48ce562`); print the HTML to PDF, 5 pages. |

---

## 1. The problem this prototype exists to solve

### What worked

On the **gimbal** — the airframe bolted down, unable to translate — the tracker pointed the camera at a person and held them. Yaw and pitch body rates in ACRO, nulling the pixel error. Conservative, but correct. That code still exists and is still test-locked (`CONTROL_LAW=servo`, the `GIMBAL` preset).

### What failed, in free flight

With a pilot holding the transmitter and the mode switch moved from STABILIZE to ACRO, the same code did the wrong thing repeatedly. The user's own description:

> *"it tracked me ok and turned to see me but I don't want that. It messed up a lot of times — when I was higher than the camera, the drone looked up to see me and suddenly flew backwards."*

That is not a tuning failure. It is three structural faults, and no gain value reaches any of them:

1. **Pointing is not intercepting.** Yaw rotates the camera. It does *not* move the airframe. A multirotor turns its **velocity vector** by **banking**, and roll was never driven at all.
2. **Pitch was being used to aim.** A target above the camera produced a nose-**up** pitch command — and on a multirotor nose-up *is reverse thrust*. That is the "suddenly flew backwards". The frame's vertical axis must never touch pitch.
3. **Pure pursuit lags a crossing target.** Turn rate proportional to the *current* bearing error — the user's own words, *"carrot on a stick"* — settles into a trailing tail-chase against someone cutting left and right, and never closes the range.

### What the requirement actually is

> *"Regardless of whether I am higher or lower than the camera, I want to run on the field and the drone to use its max power to follow me correctly, without drifting."*

So: a **moving human target**, **arbitrary relative height**, **hard lateral cuts**, and the drone works its own throttle. **No GPS on this airframe** — barometer and camera only.

---

## 2. How it is solved

Four channels, each with exactly one job. This is the whole design:

| Channel | Job | Law |
|---|---|---|
| **yaw** | keep the seeker pointed at the target | pursuit on azimuth, `KP_YAW` |
| **roll** | lateral acceleration — actually go where they went | **proportional navigation** on the azimuth LOS rate |
| **pitch** | closing speed | nose-**down** lean × commit. Never positive, ever. |
| **throttle** | vertical closure | elevation error + LOS-rate lead |

Mode is **ALT_HOLD**, not ACRO. In ALT_HOLD a roll/pitch stick *is* a lean angle and the throttle stick *is* a climb rate, so the outer loop ACRO needed just to hold a lean disappears — and a **centred throttle means hold altitude**, which is the only reason it is safe for this code to hold the throttle at all. Horizontal authority is identical to STABILIZE (same `ANGLE_MAX`); ALT_HOLD gives up nothing in power.

### 2a. Proportional navigation, and why it replaces pursuit

```
a_lat = ROLL_PN_GAIN × λ̇          bank = atan(a_lat / g)
```

PN drives the *rate* the line of sight rotates to zero. A constant bearing **is** a collision course, so PN leads a crossing target instead of trailing it. Measured against a hard zigzag: `ROLL_PN_GAIN=25` never hits and holds the target in frame 37% of the time; `45` hits with 100% in frame. 45 ≈ N=3 × ~15 m/s cruise.

### 2b. The single most important implementation detail

The LOS bearing must be **one world-referenced signal**, differentiated once:

```python
los_az = self._az_rate.update(self._unwrap(yaw + az), dt)
```

The algebraically identical `d(az_body)/dt + yawspeed` is **dynamically unstable** — it mixes a filtered term with an instantaneous one, so every yaw of the seeker itself transiently reads as line-of-sight rotation. That bug produced a limit cycle: bank pinned at ±27–30°, heading winding through 180°, range never closing. Written this way the seeker's own rotation *structurally cannot* drive PN. There is a test named for exactly this.

### 2c. Time alignment is not optional either

Both halves of `λ = yaw + az` must come from the **same instant**. The frame is ~100 ms old by the time it is processed; ATTITUDE arrives on its own schedule. Mixing them produced **27° of phantom bank** on a nose swinging along a constant bearing. Fixed by requesting ATTITUDE at 50 Hz and interpolating it to the frame's capture timestamp (`attitude(at_t=capture_t)`). Now **0.000°**.

### 2d. The camera tilt is hardware, and it is mandatory

At nose-down lean θ, a target at the airframe's *own* altitude appears at **+θ** in a body-fixed camera. The lens is only ±24.4° vertically. So a **level-mounted camera loses the target out of the top of the frame the moment the drone flies fast** — measured: at most 48% of frames on target, and it *never* closes, at any lean from 8° to 20°.

`run.py` therefore **refuses to start the intercept law without `CAM_TILT_DEG`** and exits 2. There is no safe default. Recommended mount: **10°**, giving a visible band of −34°…+14° about the horizon at 20° of lean.

### 2e. Gains that must move together

`ROLL_PN_GAIN = N × airspeed` with N=3, and airspeed follows the lean. So `MAX_LEAN_DEG` and `ROLL_PN_GAIN` are **not** independent knobs — `pn_gain_for()` prints a warning if one is changed without the other. Shipped pair: **20° / 45**, the best of everything measured.

### 2f. Terminal commit

In the last metre the head fills the frame and the detector drops out. Guidance is **deliberately frozen** there and the airframe flies the trajectory it already had (`TERMINAL_BBOX_FRAC`, `TERMINAL_RANGE_M`, `TERMINAL_HOLD_S`). **It is built to hit.** This matters for who you point it at on the first flight.

### 2g. Command rate

The user asked whether command rate should follow detection rate. It should not be *limited* to it. `SEND_HZ=50`: at 20 Hz the stick is quantised to 50 ms, which is 10° of attitude at 200 °/s. Detections arrive slower and are interpolated/predicted between. **Latency is the binding constraint** — the loop is unstable beyond ~0.2 s, and the fix for that is prediction and a faster pipeline, never more gain.

---

## 3. Safety model

* **Mode gate** — the tracker drives nothing unless the FC reports a mode in `FC_STEER_MODES` (`ALT_HOLD` for intercept). The pilot's mode switch is the on/off control for the AI.
* **Arm gate** — nothing is driven while disarmed (`REQUIRE_ARMED=1`).
* **Commit channel** — `COMMIT_CHAN=6` scales the lean *and* is the fast abort, ~100 ms. If the switch is down the law **does not steer at all**. If `COMMIT_CHAN` is set but no RC frame has been read yet, commit is forced to **0.0** and steering is refused — an earlier version fell back to `1.0` and would have gone to full lean with the pilot's stick at idle.
* **No engagement timer.** `ENGAGE_MAX_S=0` by default, at the user's instruction: *"the pilot is experienced and will stop the drone."* The mechanism is still in the code if it is ever wanted back.
* **Loss of target, stop, or crash** → every overridden channel is released.
* `RC_CHANNELS_OVERRIDE` is only accepted by ArduPilot from `SYSID_MYGCS` (default 255) and is **dropped silently** otherwise. `run.py` reads it off the FC at connect rather than assuming.

### The throttle-centre rule — the one that bites

Once this code drives throttle, **the pilot's throttle stick must sit at CENTRE**. Centre = hold altitude. If it is left at the bottom, then at the instant the code hands the channels back the FC reads "descend at maximum" and **the drone falls**. This is stated in SCENARIO.md, in the PDF, and in step 7 below.

---

## 4. ΤΑ 10 ΒΗΜΑΤΑ ΣΤΟ ΠΕΔΙΟ

*Απλά λόγια, χωρίς τεχνικούς όρους. Αναλυτικά: [SCENARIO.md](SCENARIO.md) και το PDF.*

### Βήμα 1 — Μέτρα την κάμερα
Βάλε το drone στο **πάτωμα**. Στάσου **10 μέτρα** μπροστά του. **Το κεφάλι σου πρέπει να πέφτει στο κέντρο της εικόνας.** Αν πέφτει, η κάμερα είναι στις ~10 μοίρες πάνω. Σημείωσε τη γωνία που πέτυχες — τη χρειάζεσαι στο βήμα 4.

### Βήμα 2 — Ο διακόπτης «πόσο δυνατά»
Πες στον πιλότο να συνδέσει **έναν ελεύθερο διακόπτη** στο τηλεκοντρόλ (κανάλι 6). Ρυθμίζει πόσο δυνατά πάει, **και το σταματάει σε ένα δέκατο του δευτερολέπτου.** Αν δεν έχει ελεύθερο κανάλι, μη τρέξεις την εντολή με `COMMIT_CHAN`.

### Βήμα 3 — Έλεγχος στον πάγκο, **ΕΛΙΚΕΣ ΕΞΩ**

```bash
cd src
CAM_TILT_DEG=0 REQUIRE_ARMED=0 ./run.py --no-wait
```

Περπάτα μπροστά στην κάμερα και διάβασε την οθόνη:

| Πού στέκεσαι | Τι πρέπει να δεις |
|---|---|
| **δεξιά** του drone | `bank` **θετικό** |
| **αριστερά** | `bank` **αρνητικό** |
| **πάνω** από τον φακό | `climb` **θετικό** |
| **κάτω** | `climb` **αρνητικό** |

**Αν κάτι είναι ανάποδα, μη πετάξεις.** Διορθώνεται με `ROLL_REVERSED=1` / `THROTTLE_REVERSED=1`. Αυτό είναι το μόνο βήμα που πιάνει ανάποδη σύνδεση — και ανάποδη κλίση σημαίνει ότι το drone γέρνει **μακριά** από τον στόχο με όλη τη δύναμη.

### Βήμα 4 — Ξεκίνα το πρόγραμμα

```bash
cd src
CAM_TILT_DEG=10 COMMIT_CHAN=6 ./run.py
```

Βάλε **τη γωνία που μέτρησες** στο βήμα 1, όχι το 10. Θα τυπώσει τη ζώνη ορατότητας — αν βγάλει προειδοποίηση με `!!`, σταμάτα.

**Για να βλέπεις τι βλέπει η κάμερα, από το κινητό σου:** στην αρχή θα τυπώσει μια γραμμή `seeker view : http://192.168.x.x:8080/`. Σύνδεσε το κινητό στο **ίδιο WiFi** με το Jetson και άνοιξε αυτή τη διεύθυνση στον browser. Δεν χρειάζεται καμία εφαρμογή. Είναι **μόνο για να βλέπεις** — δεν υπάρχει κανένα κουμπί εκεί μέσα, και ο πιλότος με το τηλεχειριστήριο παραμένει ο μόνος τρόπος να σταματήσει.

### Βήμα 5 — Πες στον πιλότο να σηκώσει το drone **κανονικά, σε STABILIZE**
Τίποτα ασυνήθιστο. Το πρόγραμμα **δεν αγγίζει τίποτα** σε αυτό το σημείο.

### Βήμα 6 — Ύψος περίπου **2 μέτρα**, με την κάμερα **στραμμένη αλλού**
Όχι προς άνθρωπο. Αν τη στιγμή που αλλάξει τρόπο βλέπει κάποιον, **ξεκινάει αμέσως**.

### Βήμα 7 — Πες στον πιλότο να φέρει το **γκάζι στο ΚΕΝΤΡΟ**
Και να το **αφήσει εκεί**. Κέντρο σημαίνει «κρατά ύψος». Αν το αφήσει κάτω, τη στιγμή που το πρόγραμμα παραδώσει τα χειριστήρια το drone θα διαβάσει «κατέβα με το μέγιστο» και **θα πέσει**.

### Βήμα 8 — Ο διακόπτης του βήματος 2 **ΠΑΝΩ**
Αν είναι κάτω, το πρόγραμμα **δεν οδηγεί καθόλου**. Είναι η νούμερο ένα αιτία για «γιατί δεν κάνει τίποτα;».

### Βήμα 9 — Διακόπτης σε **ALT_HOLD**, μετά **στρέψε το drone προς τον στόχο**
Μόλις μπει το κεφάλι στο κάδρο, **αναλαμβάνει και τα τέσσερα χειριστήρια**. Στην οθόνη ανάβει `DRIVE`. Από εκεί **τρέχεις** — δεξιά, αριστερά, όπου θέλεις. Το drone γέρνει στο πλάι και **κόβει δρόμο**, και ανεβοκατεβαίνει μόνο του για να έρθει στο ύψος του κεφαλιού.

### Βήμα 10 — Το ΣΤΟΠ
Δύο τρόποι, και οι δύο δουλεμένοι:

* **Διακόπτης του βήματος 2 κάτω** → όλα πίσω στον πιλότο σε **~0,1 δευτ.** ← χρησιμοποίησε αυτόν
* **Διακόπτης πίσω σε STABILIZE** → όλα πίσω σε **έως 1 δευτ.**

### ⚠ Την πρώτη φορά, ο στόχος να **μην** είναι άνθρωπος

Κράνος σε στύλο, κούκλα, ό,τι έχεις σε ύψος κεφαλιού.

**Τίποτα από αυτά δεν έχει τρέξει σε πραγματικό drone** — όλα είναι προσομοίωση. Και το τελευταίο μέτρο είναι **σκόπιμα κλειδωμένο**: σταματάει να διορθώνει και συνεχίζει την πορεία που είχε. **Είναι φτιαγμένο να χτυπήσει.**

Πιο ήπια πρώτη πτήση:

```bash
CAM_TILT_DEG=10 MAX_LEAN_DEG=12 ROLL_PN_GAIN=35 COMMIT_CHAN=6 ./run.py
```

---

## 5. What is verified, and what is not

### Verified — 6 suites, 285 checks, 0 failures

```bash
cd src
python3 tests/test_control_law.py      # 76 — the gimbal law, test-locked
python3 tests/test_intercept.py        # 79 — the intercept law + closed-loop sim
python3 tests/test_simulated_fc.py     # 66 — MAVLink against a fake FC
python3 tests/test_field_sequence.py   # 22 — TODAY'S engagement, shipped config
python3 tests/test_schedule.py         # 23 — range schedule + operator sensitivity
python3 tests/test_acro_dual.py        # 19 — ACRO rate translation + wind rejection
```

`test_field_sequence.py` walks the exact sequence of §4 on the shipped preset and asserts the three requested behaviours **by name**:

| Requirement | Measured |
|---|---|
| Flies **at** the target | lean **−19.8°** (never positive), `ch2=1236` |
| Follows a **moving** target | peak bank **+27.0°**, `ch1=1860`; own-yaw phantom bank **0.000°** (was 27.0° before the time-alignment fix) |
| Works the throttle **itself** | **+1.64 / −1.64 / 0.00 m/s** → `ch3 = 1862 / 1138 / 1500` |

Plus every abort path in both directions.

### What two real flights changed

Both flights flew. Neither hit. The logs said why, and both fixes came out of
measured numbers rather than a guess:

| Flight | What the log showed | Fix |
|---|---|---|
| `20260909_110654` | climb commanded **mean 1.11 m/s, never above 4**, against a 6 m/s ceiling the FC already had. Lean pinned at its 12° limit for **85%** of frames. | Gains sized to the ceiling (`ceiling/0.31`); FC `PILOT_SPEED_UP` 600 → 2500 |
| | bbox centre jumping **73–99 px at p99**, up to 501 px on a 640 px frame — while the camera itself moved only 0.31°/frame. Not vibration: the detector. | Track gate rejects the physically impossible (2.4–4.4% of frames) |
| `20260914_111228` | throttle stick saturated at **both** ends; world elevation went **negative on 17%** of frames and below the visible band on 9% — it climbed straight over the target and lost it underneath. | `CLIMB_LOS`: climb = `v_forward × tan(elevation)` |

### The demo scenario, simulated end to end

Own drone 2 m and stationary; enemy from 50 m at 10 m height doing 10 m/s at us.
72 combinations — approach angle 0/10/20°, four winds, three speed modes, both
flight modes. **FAST intercepts in all of them, 0.52–0.63 m**, target in frame
96–100%. Approach angle barely matters; a *headwind* is the hard case, not a
crosswind.

ACRO matches ALT_HOLD **only with a perfect thumb on the throttle**, because in
ACRO the AI never touches it:

| ACRO, pilot model | SLOW | MEDIUM | FAST |
|---|---|---|---|
| perfect | 0.90 | 0.64 | 0.52 |
| realistic (0.6 s behind) | 2.27 | 1.96 | 1.50 |
| fixed 5 m/s climb | 5.48 | 6.45 | 6.11 |
| **ALT_HOLD, for comparison** | 0.86 | 0.60 | **0.60** |

**Fly ALT_HOLD + FAST.** ACRO is there to compare, not to demo.

### NOT verified — read before flying

**Nothing in the intercept path has ever run on real hardware.** Every number above is simulation. A test is not a flight.

The single highest risk is **roll and throttle polarity on the actual airframe**. If roll is inverted the drone banks *away* from the target at full deflection. **Step 3 is the only thing that catches it and it must not be skipped.**

Second weak spot: ~8 px of detection jitter on a hard zigzag gives a miss distance spread of **0.57–3.13 m** across random seeds. It is not deterministic.

---

## 5b. The phone — this is the operator surface now

`http://<jetson-ip>:8080/` on the same WiFi. The console prints the address at
every boot; it enumerates real interfaces, so it is right on any network.

Shaped like an app: a bottom nav (`TEST` / `ΤΑΧΥΤΗΤΑ` / `ΣΤΟΧΟΣ`) and a header
whose **colour is the flight mode** — black when the pilot has every channel,
**blue in ALT_HOLD, orange in ACRO**. The video overlay uses the same three
colours for the target ring, so the app and the picture can never disagree.

| Control | Options | Sets |
|---|---|---|
| ΤΑΧΥΤΗΤΑ | SLOW / MEDIUM / FAST | climb ceiling, lean, turn gain and stick damping, together |
| ΣΤΟΧΟΣ | HUMAN / DRONE | swaps the detector between frames (both load at startup) |
| TEST | forces a mode's look | bench check only; never reaches the tracker |

**ALT_HOLD ↔ ACRO is deliberately NOT on the phone.** That gate is the interlock
that stops the AI touching the sticks; putting it on an unauthenticated page on
the field WiFi would remove the property it exists to provide. The pilot's own
mode switch decides, and the header reports it.

## 6. Running it

```bash
cd src
CAM_TILT_DEG=<measured> COMMIT_CHAN=6 ./run.py        # intercept, the default
./run.py --gimbal                                     # the old bench rig
./run.py --no-wait                                    # fail fast if no FC
./run.py --tuned                                      # no-op, kept for systemd
```

`run.py` re-execs into the right interpreter, applies the preset as **defaults** (anything already in the environment wins), discovers the FC serial port (the by-id descriptor *and* the ACM number both change across reboots — never hardcode it; `FC_DEV` overrides), reads firmware ID / channel map / calibration / `SYSID_MYGCS` / `ANGLE_MAX` / `PILOT_SPEED_UP` / `THR_DZ` off the FC, runs preflight, then launches the tracker.

### 6a. Watching the seeker from a phone — the detection window over WiFi

The Orin Nano flies with no screen, so the annotated detection frame — boxes, HUD, the stick command, everything `cv2.imshow` would have drawn — is **also served over HTTP as MJPEG**. Any browser on the same WiFi opens it. No app, no VLC, no GStreamer on the viewing device.

```
http://<jetson-ip>:8080/               the live seeker view
http://<jetson-ip>:8080/snapshot.jpg   one frame, for a bug report
```

`run.py` prints the exact address in the banner **before** it waits for the FC, so the page can already be open on the phone when the first frame lands. `hostname -I` on the Jetson gives the address if you need it by hand.

It is **on by default**, and it costs the flight computer nothing while nobody is watching — `publish()` returns without even copying the frame until a viewer connects. When one does, a separate daemon thread encodes at `STREAM_FPS` (15, half the capture rate), latest-frame-wins. A phone on a weak signal skips frames; it can never slow the detection loop or the 20 Hz command stream.

| Knob | Default | What it does |
|---|---|---|
| `./run.py --no-stream` / `STREAM=0` | on | serve nothing |
| `./run.py --stream-port N` / `STREAM_PORT` | 8080 | move it off 8080 |
| `STREAM_QUALITY` | 70 | JPEG quality; drop to ~50 on a weak link |
| `STREAM_FPS` | 15 | encode cap, independent of capture FPS |

Prove the network path before a flight, with no camera and no FC:

```bash
cd src && python3 mjpeg_stream.py     # test pattern on :8080
```

Three things to know before trusting it:

* **The page is a HUD, not a control surface.** There is nothing to click. The pilot's transmitter is still the only abort, and the mode switch is still the only handover.
* **It is unauthenticated and bound on all interfaces.** That is right for a closed field WiFi or the Jetson's own hotspot, and wrong for anything else. `--no-stream` on a network you do not own.
* **Do not use it to fly.** It is a spotter's view of what the seeker thinks, with WiFi latency on top of the pipeline latency. `latency_s` in the CSV is the number that matters for control; the stream's own delay is not in it and never affects it.

If you ever need lower latency than MJPEG gives — a real FPV feed rather than a HUD — that is a GStreamer `udpsink` off the camera, a separate path from this one, and it does not carry the annotations.

### Files

| Path | Role |
|---|---|
| [src/run.py](src/run.py) | The only entry point. Presets, geometry check, FC discovery, preflight, banner. |
| [src/fc_control.py](src/fc_control.py) | Both control laws, MAVLink/MSP transports, the TX thread, safety gates, CSV logging. |
| [src/tracker.py](src/tracker.py) | Capture, detection, tracking, HUD. Stamps `t_cap` right after `cap.read()`. |
| [src/mjpeg_stream.py](src/mjpeg_stream.py) | The seeker view over WiFi (§6a). Standalone: run it directly for a test pattern. |
| [src/tests/](src/tests/) | The four suites plus `fake_fc.py` and `fc_check.py`. |
| [src/wasp-tracker.service](src/wasp-tracker.service) | systemd **user** unit for auto-start on the Jetson. |

### What to watch in the log

The intercept CSV adds these to the base columns:

```
roll_stick, throttle_stick, el_cam_deg, bank_deg, lean_deg, climb_mps,
los_az_dps, los_el_dps, range_m, closing_mps, latency_s, commit, have_att, terminal
```

Two matter most:

* **`latency_s`** — above **0.15** fix the pipeline before raising any speed. This is the constraint that decides whether the rest is stable.
* **`los_az_dps`** — what PN is actually reacting to. If bank is doing something surprising, look here first.

Also `have_att`: with no ATTITUDE, roll and throttle are forced to **exactly zero** by design.

---

## 7. Hard rules — do not undo these

These were each paid for with a real bug. [PROTOTYPE.md §7](PROTOTYPE.md) has the full list.

1. **`λ = yaw + az`, unwrapped, differentiated once.** Never `d(az_body)/dt + yawspeed`. §2b.
2. **Attitude must be interpolated to the frame's capture time.** Never use the latest sample. §2c.
3. **Pitch is never positive.** The frame's vertical axis touches throttle, never pitch. Nose-up is reverse thrust.
4. **`MAX_LEAN_DEG` and `ROLL_PN_GAIN` move together.** §2e.
5. **`CAM_TILT_DEG` has no default.** Refusing to start is correct behaviour, not a bug to fix.
6. **Report values post-clamp and post-slew**, with inversion undone. Reporting raw law output made the HUD, the CSV and the sim's own plant all consume overstated numbers.
7. **The echo check needs history.** The FC's `RC_CHANNELS` comes back at ~10 Hz against commands sent at 50 Hz; comparing against only the latest command produced false `REFUSED` in healthy flight. There is a 0.7 s per-channel history for this.
8. **Commit defaults to 0.0, not 1.0**, when the channel is configured but unread. §3.
9. **Never hardcode "ACRO" in a message.** Four messages used to send the pilot to the one mode where the intercept law does nothing. Print `steer_modes`.
10. **The terminal hold counts accumulated `dt`, not wall clock**, or it is untestable.
11. **Do not tune on single sim runs.** The simulator is chaotic — lean 14 failed while 12 and 16 passed, non-monotonically. Only claim what is robust across a sweep.

### Two of my own proposals that were measured and rejected

* **Raising the vertical gain / ceiling** — no measurable effect in any geometry. The height error is absorbed long before the endgame. Don't.
* **Deriving lean and gain from tilt** — the derived tilt-0 config measured 33% in frame and no closure. Backed out.

---

## 8. Open items

* **Hardware bring-up of the intercept path.** Everything else is downstream of this.
* **`48ce562` added `docs/` to `.gitignore`** — dropped into the middle of the Python "Distribution / packaging" block, which looks unintended. `docs/wasp-explained-gr.html` survives because ignore rules do not apply to already-tracked files, but **any new file under `docs/` will now be ignored silently.** The same commit untracked `WASP_EXPLAINED_GR.pdf`, so the PDF has to be regenerated from the HTML locally — it will not arrive with a `git pull`.
* **`src/wasp-tracker.service` carries a stale comment** claiming `run.py` ships exactly one preset; there are two, and the unit does not set `CONTROL_LAW`, so it gets the intercept default. Its paths are the Jetson's (`/home/iknowhow/Documents/Interceptor/src`), and both launchers run from the venv at `.env/yolo`.
* **Detection jitter on hard zigzags** (§5) is the remaining accuracy limit.
* Camera shortlist for longer-range work was narrowed to three options under €200 for the Jetson Orin dev kit; not ordered.
