#!/usr/bin/env python3


MODE = "sky"





MOUNT_TILT_DEG = 90



TARGET_WIDTH_M = None



MAX_LEAN_DEG  = None
ROLL_PN_GAIN  = None

COMMIT_CHAN = 0



MODES = {
    "sky": {
        "label": "AIRBORNE TARGET (drone above)",
        "cam_tilt_deg": 30,
        "target_width_m": 0.30,
        "max_lean_deg": 12,
        "roll_pn_gain": 35,
        "band": ("-6 .. +42 deg near level; full 20 deg lean would drop it to "
                 "-14 .. +34.  EDGE_GUARD gives speed back to the seeker "
                 "whenever the target nears ANY edge.  BLIND BELOW THE HORIZON."),
        "model": "runs/detect/v0/weights/best.pt",
        "target_class": "",
        "climb_gain": 19,
        "stick_limit_throttle": 1.0,
        "max_lean_deg": 20,
        "climb_los": True,
        "climb_trim": 6,
        "cruise_mps": 15,
        "edge_guard": 0.55,
        "lean_el_guard": 0.0,
        "roll_pn_gain": 45,
        "bank_far_deg": 30,
        "max_bank_deg": 36,
        "climb_lead": 8,
        "sens_default": 0.5,
        "sens_mode": "STANDARD",
        "speed_mode": "MEDIUM",
        "pn_near_mult": 1.4,
        "lock_ramp_s": 0.3,
        "note": ("Sim: closes to ~35 deg elevation.  Nothing closes above ~45 "
                 "deg in any configuration."),
        "steer_modes": "ALT_HOLD,ACRO",
    },
}


class ConfigError(Exception):
    pass


def resolve():
    if MODE not in MODES:
        raise ConfigError(
            f"MODE is {MODE!r}, which is not one of {sorted(MODES)}.\n"
            "  Edit flight_config.py and uncomment exactly one MODE line.")

    cfg = dict(MODES[MODE])
    want = cfg["cam_tilt_deg"]

    if MOUNT_TILT_DEG not in (want, 90):
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
    cfg["camera_mount_deg"] = MOUNT_TILT_DEG
    cfg["commit_chan"] = int(COMMIT_CHAN)

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
    return "\n".join([
        f"WASP_MODE={cfg['mode']}",
        f"WASP_MODE_LABEL='{cfg['label']}'",
        f"WASP_MODE_BAND='{cfg['band']}'",
        f"CAM_TILT_DEG={cfg['cam_tilt_deg']}",
        f"CAMERA_MOUNT_DEG={cfg['camera_mount_deg']}",
        f"TARGET_WIDTH_M={cfg['target_width_m']}",
        f"MAX_LEAN_DEG={cfg['max_lean_deg']}",
        f"ROLL_PN_GAIN={cfg['roll_pn_gain']}",
        f"COMMIT_CHAN={cfg['commit_chan']}",
        f"BANK_FAR_DEG={'' if cfg.get('bank_far_deg') is None else cfg['bank_far_deg']}",
        f"SENS_DEFAULT={cfg.get('sens_default', 0.5)}",
        f"PN_NEAR_MULT={cfg.get('pn_near_mult', 1.0)}",
        f"LOCK_RAMP_S={cfg.get('lock_ramp_s', 0.0)}",
        f"MAX_BANK_DEG={cfg.get('max_bank_deg', 30)}",
        f"CLIMB_LEAD={cfg.get('climb_lead', 3)}",
        f"CLIMB_GAIN={cfg.get('climb_gain', 8)}",
        f"STICK_LIMIT_THROTTLE={cfg.get('stick_limit_throttle', 0.90)}",
        f"LEAN_EL_GUARD={cfg.get('lean_el_guard', 0.0)}",
        f"WASP_MODE_SPEED={cfg.get('speed_mode', 'MEDIUM')}",
        f"CLIMB_LOS={1 if cfg.get('climb_los') else 0}",
        f"EDGE_GUARD={cfg.get('edge_guard', 0.0)}",
        f"CRUISE_MPS={cfg.get('cruise_mps', 15)}",
        f"CLIMB_TRIM={cfg.get('climb_trim', 6)}",
        f"FC_STEER_MODES={cfg.get('steer_modes', 'ALT_HOLD')}",
        f"TARGET_WIDTH_DRONE={TARGET_WIDTH_M if TARGET_WIDTH_M is not None else MODES['sky']['target_width_m']}",
        f"WASP_MODEL={cfg.get('model') or ''}",
        f"TARGET_CLASS={cfg.get('target_class') or ''}",
    ])


if __name__ == "__main__":
    import sys
    shell = "--shell" in sys.argv
    try:
        cfg = resolve()
    except ConfigError as exc:


        sys.exit(1)

    if shell:
        print(as_shell(cfg))
    else:













        if cfg.get('edge_guard'):
            pass





        if cfg.get("bank_far_deg") is None:
            pass
        else:
            pass
