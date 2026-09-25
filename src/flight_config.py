#!/usr/bin/env python3
"""ONE file that decides which WASP flies at power-up.

Edit this, reboot, and the Jetson comes up in the mode you chose.  It is read
by start-wasp.sh, which is what the systemd unit runs, so there is exactly one
place to change and nothing else to remember.

    MODE = "human"   ->  head tracking      (camera mount at 10 deg)
    MODE = "sky"     ->  airborne target    (camera mount at 30 deg)

Pick ONE below by commenting out the other.

    python3 flight_config.py            show the active mode and check it
    python3 flight_config.py --shell    emit the env for start-wasp.sh

WHY THIS FILE REFUSES TO GUESS ABOUT THE MOUNT
----------------------------------------------
The camera angle is a HARDWARE FACT.  The control law adds it to every
elevation it computes, so a number here that disagrees with the real mount
flies the airframe high or low by exactly that error, consistently, forever,
with nothing in the log to say why.  It never errors — it just flies wrong.

At the bench a human can be asked.  At power-up nobody is there.  So the
physical angle is declared ONCE, in MOUNT_TILT_DEG below, and a mode whose
required tilt does not match it REFUSES TO START.  Changing the mode without
touching the mount is meant to be loud, not silent.
"""

# ═════════════════════════════════════════════════════════════════════════════
# 1. THE MODE.  Comment out the one you are NOT flying.
#    (If you leave both uncommented, the LAST one wins — Python is just
#    assigning a variable twice.  Comment one out properly.)
# ═════════════════════════════════════════════════════════════════════════════

# MODE = "human"          # head tracking — needs the mount at 10 deg
MODE = "sky"        # drone in the sky — needs the mount at 30 deg


# ═════════════════════════════════════════════════════════════════════════════
# 2. THE PHYSICAL MOUNT ANGLE, IN DEGREES UP.  Measure it, do not assume it.
#    This is what the camera ACTUALLY is, not what you want it to be.
#    Change the mount FIRST, then change this number, then change MODE.
# ═════════════════════════════════════════════════════════════════════════════

MOUNT_TILT_DEG = 30


# ═════════════════════════════════════════════════════════════════════════════
# 3. TARGET WIDTH, METRES — the target's REAL width, not its apparent size.
#    Range = focal_px * this / bbox_px, so it drives the range readout AND the
#    terminal freeze.  Leave None to take the mode's default.
#    Measure the target drone motor-tip to motor-tip and put it here.
# ═════════════════════════════════════════════════════════════════════════════

TARGET_WIDTH_M = None      # e.g. 0.35


# ═════════════════════════════════════════════════════════════════════════════
# 4. Lean and turn gain.  These two ALWAYS move together (gain ~= 3 x cruise
#    speed) — raising the lean alone makes it trail.  None = mode default.
# ═════════════════════════════════════════════════════════════════════════════

MAX_LEAN_DEG  = None       # e.g. 16
ROLL_PN_GAIN  = None       # e.g. 48

# Spare RC channel: scales the lean AND aborts in ~100 ms.  0 = none wired,
# which means lean is pinned at 100% while locked and the mode switch is the
# only abort (~1 s).  Wire one if you can.
COMMIT_CHAN = 0


# ═════════════════════════════════════════════════════════════════════════════
# Below here is the mode table and the checks.  You should not need to edit it.
# ═════════════════════════════════════════════════════════════════════════════

MODES = {
    "human": {
        "label": "HEAD TRACKING (human face)",
        "cam_tilt_deg": 10,
        "target_width_m": 0.20,        # a human face
        "max_lean_deg": 12,
        "roll_pn_gain": 35,
        "band": "-26 .. +22 deg at 12 deg lean  (sees the ground)",
        # None = the HuggingFace face/person detector, pulled from the cache.
        "model": None,
        "target_class": "",            # "" = auto (the 'face' class)
        # NO range schedule here.  This is the scored scenario, and section 9 of
        # tests/test_intercept.py stops hitting the close-range zigzag the
        # moment the far-field bank ceiling is tightened: a human face fills so
        # little of the frame that the schedule reads "far" for the whole
        # engagement and never hands the authority back.  Bit-for-bit the
        # behaviour that was measured.
        "climb_gain": 8,
        "stick_limit_throttle": 0.90,
        "lean_el_guard": 0.0,
        # Human mode keeps the measured behaviour: no LOS climb, no guard.
        "climb_los": False,
        "edge_guard": 0.0,
        "bank_far_deg": None,
        "sens_default": 0.5,
        "sens_mode": "STANDARD",
        "speed_mode": "SLOW",
        "pn_near_mult": 1.0,
        "lock_ramp_s": 0.0,
        "note": "Scored and flown.  Closes on a level target.",
    },
    "sky": {
        "label": "AIRBORNE TARGET (drone above)",
        "cam_tilt_deg": 30,
        "target_width_m": 0.30,        # placeholder: ~7-inch quad. MEASURE IT.
        "max_lean_deg": 12,
        "roll_pn_gain": 35,
        "band": ("-6 .. +42 deg near level; full 20 deg lean would drop it to "
                 "-14 .. +34.  EDGE_GUARD gives speed back to the seeker "
                 "whenever the target nears ANY edge.  BLIND BELOW THE HORIZON."),
        # The sky weights are LOCAL — trained here, not on the Hub.  Relative to
        # the repo root, resolved to an absolute path below, because tracker.py
        # runs with its cwd in src/ and a relative path would miss by one level.
        "model": "runs/detect/v0/weights/best.pt",
        # That model is single-class and calls its class '0', not 'face'.
        "target_class": "",            # "" = auto: the only class there is
        # THE RANGE SCHEDULE, and it is on only here.  A drone in the sky is
        # acquired at long range, where the LOS rate is mostly bbox jitter and
        # the seeker's own yaw slaving injects a phantom rate of the same order
        # (see fc_control.InterceptLaw's docstring).  Reacting hard to that
        # throws the airframe around on nothing.
        #
        # Note WHAT is scheduled: the BANK CEILING, not the turn gain.  Gain 25
        # was measured losing a hard zigzag outright, so ROLL_PN_GAIN is a floor
        # the schedule may only ever add to — never a number to scale down.
        # Authority is the part that is safe to hold back early.
        # ── SPEED.  Measured from flight 20260909_110654 (3233 engaged frames):
        #   climb commanded  mean 1.11 m/s, max 3.97, and NOT ONE frame above
        #                    4 m/s — against a 6.0 m/s ceiling the FC was
        #                    already configured to give (PILOT_SPEED_UP=600).
        #                    Throttle stick averaged 0.26 of 0.90 available.
        #   lean commanded   pinned at the 12 deg limit for 85% of frames.
        #
        # So the airframe was not slow because it was working hard — it was slow
        # because it was barely asking.  climb = CLIMB_GAIN * elevation_error,
        # and at the observed median error of 10 deg (0.178 rad) a gain of 8
        # requests 1.4 m/s.  That is the 1.11 m/s in the log, exactly.
        #
        # SCENARIO.md's rule is CLIMB_GAIN ~= ceiling / 0.31, so 6.0 m/s wants
        # ~19.  THIS is the knob that makes it climb, and climbing is what
        # closes the range on a target above you.
        "climb_gain": 19,
        "stick_limit_throttle": 1.0,   # was 0.90 = a 10% ceiling for nothing
        # Lean is the forward-speed channel and it WAS saturated, so it does
        # need raising.  But raising it alone would have blinded the seeker:
        # the visible band is (tilt - lean) +/- 24.4, so lean 20 drops the top
        # edge from 42 to 34 deg while the measured elevation p95 was 38.9.
        # lean_el_guard hands the lean back to the seeker near the top of the
        # frame, which makes the higher ceiling safe to carry.
        "max_lean_deg": 20,
        # THE CLIMB/FORWARD TRADE-OFF.  Flight 20260914 measured the old law
        # climbing bang-bang (throttle stick saturated at BOTH ends) and
        # arriving ABOVE the target: world elevation went negative on 17% of
        # frames and below the -6 deg edge of the visible band on 9%, at which
        # point the target is under the nose and sky mode cannot see it.
        # CLIMB_LOS points the velocity vector AT the target instead:
        # climb = forward_speed * tan(elevation).  In sim that takes FAST from
        # 55% of the run above the target (35% out of band) to 2% / 0%, at the
        # same intercept time.
        "climb_los": True,
        "climb_trim": 6,
        "cruise_mps": 15,          # what this airframe settles at, 20 deg lean
        # Symmetric frame-edge guard REPLACES the old top-only lean_el_guard,
        # which the same flight showed cutting forward lean to 14% while the
        # target was actually escaping out of the BOTTOM.
        "edge_guard": 0.55,
        "lean_el_guard": 0.0,
        # PN gain tracks airspeed (gain ~= 3 x cruise).  At 20 deg of lean this
        # airframe settles near 15 m/s, so 45 — the documented default — not 35.
        "roll_pn_gain": 45,
        # WIND.  Measured in sim (tests/test_acro_dual.py, section 6): a
        # stationary target 20 m out / 10 m up, 12 m/s crosswind.  At the old
        # bank_far=12 the airframe MISSES (3.05 m, 16% in frame) because the
        # "calm at acquisition" ceiling is still binding when the wind-induced
        # LOS rate needs more authority than that.  At bank_far=24 it HITS
        # cleanly (0.56 m, 100% in frame) — same still-air result either way
        # (0.60 m), so this costs nothing when there is no wind.  Field flights
        # are not indoors: raised for that reason, not for a faster acquisition.
        # 30/36 since 2026-09-25: the miss traces showed the bank PINNED at the
        # 24 deg ceiling in the last second while the target slid sideways.
        # ANGLE_MAX is 40 on this FC now, so there is room.
        "bank_far_deg": 30,        # ...opening to MAX_BANK_DEG as it closes
        "max_bank_deg": 36,
        # Vertical LOS-rate lead: 3 -> 8 cut the vertical miss (0.66 -> 0.57 m
        # at 10 m/s) and raised touches 13 -> 19% in the scenarios.
        "climb_lead": 8,
        "sens_default": 0.5,       # STANDARD on the dashboard = the plain gain
        "sens_mode": "STANDARD",
        # MEDIUM at power-up: FAST is one tap away on the phone, but a
        # 20 m/s climb should be a decision someone makes, not a default.
        "speed_mode": "MEDIUM",
        "pn_near_mult": 1.4,       # and 40% more gain once it fills the frame
        # Ramp the bank ceiling in over 1 s after lock (was 0.5).  With the
        # mission's 1.5 s blend this halves the takeover kick in sim (mean
        # bank in the first second 11.1 -> 5.7 deg) at no accuracy cost.
        # 0.3 s since 2026-09-25 (with the mission's 0.4 s blend): the AI must
        # reach authority inside a 1-2 s encounter.  See mission.BLEND_S.
        "lock_ramp_s": 0.3,        # ignore the first-lock differentiator spike
        "note": ("Sim: closes to ~35 deg elevation.  Nothing closes above ~45 "
                 "deg in any configuration."),
        # BOTH flight modes drive the SAME guidance law.  ALT_HOLD commands
        # angles onto the FC's own ~400 Hz attitude controller; ACRO has none,
        # so an outer P loop (ATT_KP) reaches the same bank/lean target as a
        # RATE, over a much slower link — expect ACRO to reject a gust WORSE,
        # not better, than ALT_HOLD.  Throttle never moves in ACRO (see
        # ArduPilotAcro.steer): the pilot always keeps it there.  Whichever one
        # the pilot's transmitter is actually in decides, cycle to cycle — no
        # separate switch to remember.
        "steer_modes": "ALT_HOLD,ACRO",
    },
}


class ConfigError(Exception):
    pass


def resolve():
    """The active mode as a dict, or raise ConfigError with the reason."""
    if MODE not in MODES:
        raise ConfigError(
            f"MODE is {MODE!r}, which is not one of {sorted(MODES)}.\n"
            "  Edit flight_config.py and uncomment exactly one MODE line.")

    cfg = dict(MODES[MODE])
    want = cfg["cam_tilt_deg"]

    if int(MOUNT_TILT_DEG) != int(want):
        raise ConfigError(
            f"MODE is {MODE!r}, which needs the camera at {want} deg,\n"
            f"  but MOUNT_TILT_DEG says the camera is physically at "
            f"{MOUNT_TILT_DEG} deg.\n\n"
            "  These must agree.  The control law adds the tilt to every\n"
            "  elevation it computes: a wrong number does not error, it flies\n"
            "  high or low by exactly that error, permanently.\n\n"
            f"  Either physically move the mount to {want} deg and set\n"
            f"  MOUNT_TILT_DEG = {want}, or put MODE back to the mode that\n"
            f"  matches the mount you actually have.")

    if TARGET_WIDTH_M is not None:
        cfg["target_width_m"] = float(TARGET_WIDTH_M)
    if MAX_LEAN_DEG is not None:
        cfg["max_lean_deg"] = float(MAX_LEAN_DEG)
    if ROLL_PN_GAIN is not None:
        cfg["roll_pn_gain"] = float(ROLL_PN_GAIN)
    cfg["mode"] = MODE
    cfg["commit_chan"] = int(COMMIT_CHAN)

    # A mode that names its own weights must actually have them.  Better to say
    # so here, with the path, than to let the run reach the field and die.
    if cfg.get("model"):
        from pathlib import Path
        p = (Path(__file__).resolve().parent.parent / cfg["model"]).resolve()
        if not p.is_file():
            raise ConfigError(
                f"MODE is {MODE!r}, whose weights are {cfg['model']},\n"
                f"  but there is no file at {p}.")
        cfg["model"] = str(p)
    return cfg


def as_shell(cfg):
    """The env start-wasp.sh evals.  Every value the run depends on."""
    return "\n".join([
        f"WASP_MODE={cfg['mode']}",
        f"WASP_MODE_LABEL='{cfg['label']}'",
        f"WASP_MODE_BAND='{cfg['band']}'",
        f"CAM_TILT_DEG={cfg['cam_tilt_deg']}",
        f"TARGET_WIDTH_M={cfg['target_width_m']}",
        f"MAX_LEAN_DEG={cfg['max_lean_deg']}",
        f"ROLL_PN_GAIN={cfg['roll_pn_gain']}",
        f"COMMIT_CHAN={cfg['commit_chan']}",
        # The range schedule.  Empty BANK_FAR_DEG means "same as MAX_BANK_DEG",
        # which is how a mode says it does not want the schedule at all.
        f"BANK_FAR_DEG={'' if cfg.get('bank_far_deg') is None else cfg['bank_far_deg']}",
        f"SENS_DEFAULT={cfg.get('sens_default', 0.5)}",
        f"PN_NEAR_MULT={cfg.get('pn_near_mult', 1.0)}",
        f"LOCK_RAMP_S={cfg.get('lock_ramp_s', 0.0)}",
        f"MAX_BANK_DEG={cfg.get('max_bank_deg', 30)}",
        f"CLIMB_LEAD={cfg.get('climb_lead', 3)}",
        f"CLIMB_GAIN={cfg.get('climb_gain', 8)}",
        f"STICK_LIMIT_THROTTLE={cfg.get('stick_limit_throttle', 0.90)}",
        f"LEAN_EL_GUARD={cfg.get('lean_el_guard', 0.0)}",
        # The dashboard's named steering modes and the two target widths, so
        # switching target type in the air also switches the range scale.
        f"WASP_MODE_SPEED={cfg.get('speed_mode', 'MEDIUM')}",
        f"CLIMB_LOS={1 if cfg.get('climb_los') else 0}",
        f"EDGE_GUARD={cfg.get('edge_guard', 0.0)}",
        f"CRUISE_MPS={cfg.get('cruise_mps', 15)}",
        f"CLIMB_TRIM={cfg.get('climb_trim', 6)}",
        # ALT_HOLD only unless a mode explicitly asks for both — the pilot's
        # transmitter mode switch is what actually arbitrates between them.
        f"FC_STEER_MODES={cfg.get('steer_modes', 'ALT_HOLD')}",
        # The dashboard SPEED profile owns climb_gain and max_lean at run
        # time, so CLIMB_GAIN / MAX_LEAN_DEG above are only the values the
        # law is built with before the profile is applied at startup.
        f"TARGET_WIDTH_HUMAN={MODES['human']['target_width_m']}",
        f"TARGET_WIDTH_DRONE={TARGET_WIDTH_M if TARGET_WIDTH_M is not None else MODES['sky']['target_width_m']}",
        # Empty is meaningful: it tells tracker.py to fall back to the Hub
        # model / auto class rather than inheriting a stale export.
        f"WASP_MODEL={cfg.get('model') or ''}",
        f"TARGET_CLASS={cfg.get('target_class') or ''}",
    ])


if __name__ == "__main__":
    import sys
    shell = "--shell" in sys.argv
    try:
        cfg = resolve()
    except ConfigError as exc:
        # stderr either way, so a boot failure lands in the log with the reason.
        print(f"\n[flight_config] REFUSING TO START\n\n  {exc}\n",
              file=sys.stderr)
        sys.exit(1)

    if shell:
        print(as_shell(cfg))
    else:
        print(f"  mode          : {cfg['mode']}  —  {cfg['label']}")
        print(f"  camera tilt   : {cfg['cam_tilt_deg']} deg "
              f"(mount declared at {MOUNT_TILT_DEG})")
        print(f"  visible band  : {cfg['band']}")
        print(f"  target width  : {cfg['target_width_m']} m")
        print(f"  lean / gain   : {cfg['max_lean_deg']} deg / "
              f"{cfg['roll_pn_gain']}")
        print(f"  climb         : gain {cfg.get('climb_gain', 8)} m/s/rad, "
              f"stick <= {cfg.get('stick_limit_throttle', 0.9) * 100:.0f}%"
              + (f"   lean eases above {cfg['lean_el_guard'] * 100:.0f}% "
                 f"of the half-frame" if cfg.get('lean_el_guard') else "")
              + ("   CLIMB_LOS (v_forward * tan(elevation))"
                 if cfg.get('climb_los') else ""))
        if cfg.get('edge_guard'):
            print(f"  edge guard    : speed eases back above "
                  f"{cfg['edge_guard'] * 100:.0f}% toward ANY frame edge")
        print(f"  commit chan   : {cfg['commit_chan']}")
        print(f"  FC mode gate  : {cfg.get('steer_modes', 'ALT_HOLD')}"
              + ("  — pilot's transmitter switch decides which control law runs"
                 if ',' in cfg.get('steer_modes', '') else ""))
        if cfg.get("bank_far_deg") is None:
            print(f"  turn schedule : OFF — bank fixed at MAX_BANK_DEG")
        else:
            print(f"  turn schedule : bank {cfg['bank_far_deg']} deg far -> "
                  f"MAX_BANK_DEG near, gain x1 -> x{cfg['pn_near_mult']}")
            print(f"  sensitivity   : {cfg['sens_default']} at start "
                  f"(live on the HUD slider)  lock ramp "
                  f"{cfg['lock_ramp_s']}s")
        print(f"  weights       : {cfg.get('model') or 'HuggingFace face detector (cached)'}")
        print(f"  note          : {cfg['note']}")
