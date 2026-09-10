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
        "bank_far_deg": None,
        "sens_default": 0.5,
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
        "band": "-6 .. +42 deg at 12 deg lean  (BLIND BELOW THE HORIZON)",
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
        "bank_far_deg": 12,        # ...opening to MAX_BANK_DEG as it closes
        "sens_default": 0.2,       # calm at acquisition; 0.5 = the plain gain
        "pn_near_mult": 1.4,       # and 40% more gain once it fills the frame
        "lock_ramp_s": 0.5,        # ignore the first-lock differentiator spike
        "note": ("Sim: closes to ~35 deg elevation.  Nothing closes above ~45 "
                 "deg in any configuration."),
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
        print(f"  commit chan   : {cfg['commit_chan']}")
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
