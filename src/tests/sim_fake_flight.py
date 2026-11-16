#!/usr/bin/env python3
"""Drive FlightLogs + LogUploader through a synthetic flight so Firestore sees
rows / events / metadata under flight_logs/{JETSON_ID}/flights/<flight_id> AND
the /logs convert step works end-to-end.

The upstream row schema matches what fc_control.py writes on a real flight
(see fc_control.py `_open_log`), so convert_logs_to_standard can parse it.
"""
import os
import sys
import csv
import io
import math
import time
import random
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC  = HERE.parent
sys.path.insert(0, str(SRC))

from flight_logging import FlightLogs
from log_uploader  import LogUploader


# Upstream CSV columns the real tracker writes (minimum the converter needs).
UPSTREAM_COLS = [
    "t_s", "mode", "armed",
    "meas_roll_deg", "meas_pitch_deg", "meas_yaw_deg",
    "rollspeed_dps", "pitchspeed_dps", "yawspeed_dps",
    "height_m", "climb_mps",
    "vibe_x", "vibe_y", "vibe_z", "clip",
    "pwm_roll", "pwm_pitch", "pwm_yaw", "pwm_thr",
    "lat", "lon", "gps_fix", "gps_sats",
]
UPSTREAM_HEADER = ",".join(UPSTREAM_COLS)


def row_csv(values):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(
        values.get(k, "") if values.get(k) is not None else "" for k in UPSTREAM_COLS)
    return buf.getvalue()


def make_upstream(t_s, phase, armed, fc_mode):
    roll  = 10 * math.sin(t_s * 0.3)
    pitch = 5 * math.sin(t_s * 0.5)
    yaw   = (t_s * 8) % 360
    return {
        "t_s":  round(t_s, 3),
        "mode": fc_mode,
        "armed": 1 if armed else 0,
        "meas_roll_deg":  round(roll, 2),
        "meas_pitch_deg": round(pitch, 2),
        "meas_yaw_deg":   round(yaw, 2),
        "rollspeed_dps":  round(20 * math.sin(t_s), 2),
        "pitchspeed_dps": round(15 * math.sin(t_s * 1.1), 2),
        "yawspeed_dps":   round(10 * math.sin(t_s * 0.7), 2),
        "height_m": round(2 + 4 * math.sin(t_s * 0.1), 2),
        "climb_mps": round(0.5 * math.cos(t_s * 0.1), 2),
        "vibe_x": round(random.uniform(0.0, 0.5), 2),
        "vibe_y": round(random.uniform(0.0, 0.5), 2),
        "vibe_z": round(random.uniform(0.0, 0.5), 2),
        "clip":   0,
        "pwm_roll":  1500 + int(roll  * 5),
        "pwm_pitch": 1500 + int(pitch * 5),
        "pwm_yaw":   1500,
        "pwm_thr":   1200 + int(80 * max(0.0, math.sin(t_s * 0.2))),
        "lat": 37.9755, "lon": 23.7348,
        "gps_fix": 3, "gps_sats": 14,
    }


def make_state(t_s, phase, armed):
    return dict(
        phase=phase, armed=armed, linked=True,
        idle_limit_s=60.0, t_s=t_s,
        wasp_enabled=(phase == "WASP_FLIGHT"),
        auto_run=False, gate_reason="",
        detection_fresh=bool(random.getrandbits(1)),
        wasp_stage="HUNT" if phase == "WASP_FLIGHT" else "OFF",
        wasp_steering_blend=0.0 if phase != "WASP_FLIGHT" else min(1.0, t_s / 3.0),
        wasp_target_acquired=(phase == "WASP_FLIGHT" and t_s > 2.0),
        idle_stage="stabilize" if phase == "IDLE" else "",
        idle_height_target_m=5.0 if phase == "IDLE" else "",
        idle_heading_target_deg=90.0 if phase == "IDLE" else "",
        cmd_roll_rate_dps=round(20 * math.sin(t_s * 0.5), 2),
        cmd_pitch_rate_dps=round(15 * math.sin(t_s * 0.7), 2),
        cmd_yaw_rate_dps=round(10 * math.sin(t_s * 0.3), 2),
        cam_fps=30.0, processing_fps=20.0,
    )


def main():
    project  = os.environ.get("GCP_PROJECT", "ikh-drone-cloud")
    jetson   = os.environ.get("JETSON_ID",  "wasp-03")
    data_dir = Path(os.environ.get("WASP_DATA_DIR", SRC / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    uploader = LogUploader(project=project, jetson_id=jetson)
    uploader.start()

    stamp  = datetime.now().strftime("%Y%m%d_%H%M%S_sim")
    logger = FlightLogs(directory=data_dir,
                        stamp=stamp,
                        header=UPSTREAM_HEADER,
                        uploader=uploader)

    def metadata():
        return {"sim": True, "note": "firestore end-to-end smoke test",
                "src": "sim_fake_flight.py",
                "jetson_id": jetson,
                "fc_mode": "STABILIZE"}

    phases = [
        ("IDLE",         3.0, True,  "STABILIZE"),
        ("WASP_FLIGHT", 15.0, True,  "ALT_HOLD"),
        ("SAFEDOWN",     2.0, True,  "ALT_HOLD"),
        ("STANDBY",      0.1, False, "STABILIZE"),
    ]
    tick_s = 0.05
    t = 0.0
    for phase, dur, armed, fc_mode in phases:
        n = int(dur / tick_s)
        print(f"-- phase {phase} armed={armed} for {dur:.1f}s ({n} ticks)")
        for _ in range(n):
            row   = row_csv(make_upstream(t, phase, armed, fc_mode))
            state = make_state(t, phase, armed)
            logger.record(t, row, state, metadata)
            t    += tick_s
            time.sleep(tick_s)

    logger.close(now=t, reason="sim complete", phase="STANDBY")
    uploader.stop()
    print("uploader snapshot:", uploader.snapshot())
    if logger.last:
        print("flight_id:", logger.last["flight_id"])


if __name__ == "__main__":
    main()
