






import argparse
import csv
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wasp import Wasp, WaspGuidance

DT = .01
SEND_DT = .02
CAM_DT = 1/30
FW = FH = 640
TH = math.tan(math.radians(62.2/2))
TV = math.tan(math.radians(48.8/2))


@dataclass(frozen=True)
class Condition:
    name: str
    latency_s: float
    rate_tau_s: float
    noise_px: float
    dropout: bool = False


CONDITIONS = (
    Condition("nominal", .04, .08, .5),
    Condition("delay_noise", .12, .18, 1.5),
    Condition("detection_gap", .08, .12, 1.0, True),
)


class Mapper:
    @staticmethod
    def wasp_pwm(rates, throttle):
        return rates


def simulate(params, moving=False, condition=CONDITIONS[0], seed=0,
             speed_mps=20.0, distance_m=250.0, initial_error_deg=0.0):
    rng = random.Random(seed)
    w, guidance = Wasp(), WaspGuidance()
    w.params.update(params)
    w.begin(0, target_acquired=False)
    angle = -math.radians(initial_error_deg)
    actual_rate = 0.0
    py = pz = 0.0
    vy, vz = speed_mps*math.sin(angle), -speed_mps*math.cos(angle)
    pending = []
    obs = None
    det_t = None
    next_capture = next_send = 0.0
    command = 0.0
    previous_command = 0.0
    rows = []
    mapper = Mapper()
    target_speed = 10.0 if moving else 0.0
    horizon = 25.0
    for step in range(round(horizon/DT)+1):
        t = round(step*DT, 8)
        dy, dz = target_speed*t-py, -distance_m-pz
        distance = math.hypot(dy, dz)
        bearing = math.atan2(dy, -dz)
        error = (bearing-angle+math.pi) % (2*math.pi)-math.pi
        att = dict(roll=angle, pitch=0.0, yaw=0.0)
        if t >= next_capture-1e-9:
            next_capture += CAM_DT
            body_y = math.cos(angle)*dy+math.sin(angle)*dz
            body_z = -math.sin(angle)*dy+math.cos(angle)*dz
            visible = body_z < 0 and abs(body_y/-body_z) <= TH
            gap = condition.dropout and 6.0 <= t < 6.35
            if visible and not gap:
                ex = FW/2/TH*(body_y/-body_z)+rng.gauss(0, condition.noise_px)
                pending.append((t+condition.latency_s, t, ex, att))
        while pending and pending[0][0] <= t+1e-9:
            _, capture_t, ex, capture_att = pending.pop(0)
            obs = guidance.observe(ex, 0, FW, FH, TH, TV, capture_att, capture_t)
            obs["camera_roll"] = 0.0
            det_t = t
        acceleration = 0.0
        if t >= next_send-1e-9:
            next_send += SEND_DT
            ctx = dict(att=att, ai_cmd=obs, det_t=det_t, det_conf=.9 if obs else None)
            tracking = not w.fault(t, ctx, .2)
            if tracking:
                try:
                    if w.acquired_t is None:
                        w.acquire(t, ctx)
                    else:
                        WaspGuidance.rates(obs, att, t, gain=w.params["gain"],
                                           lead_s=w.params["lead"], max_rate_dps=w.params["rate"])
                except ValueError:
                    tracking = False
            if not tracking:
                w.lose_target()
            command = w.output(mapper, t, ctx)[0]
            acceleration = math.degrees(abs(command-previous_command)/SEND_DT)
            previous_command = command
            progress = w.progress(t)
            rows.append(dict(t_s=t, range_m=distance,
                             error_deg=math.degrees(error),
                             command_dps=math.degrees(command),
                             command_accel_dps2=acceleration,
                             measured_dps=math.degrees(actual_rate),
                             stage=progress["stage"], blend=progress["steering_blend"],
                             target_acquired=progress["target_acquired"]))
        if distance <= 10:
            break
        actual_rate += DT/(condition.rate_tau_s+DT)*(command-actual_rate)
        angle += actual_rate*DT

        vy += DT/(.35+DT)*(speed_mps*math.sin(angle)-vy)
        vz += DT/(.35+DT)*(-speed_mps*math.cos(angle)-vz)
        py += vy*DT
        pz += vz*DT
    tracked = [r for r in rows if r["t_s"] >= 2 and r["range_m"] >= 20]
    errors = [r["error_deg"] for r in tracked]
    metrics = dict(motion="crossing_10mps" if moving else "stationary",
                   condition=condition.name, seed=seed, speed_mps=speed_mps,
                   initial_range_m=distance_m, initial_error_deg=initial_error_deg,
                   **{k:params[k] for k in ("straight", "blend", "rate", "gain")},
                   reached_10m=distance <= 10,
                   time_to_10m_s=round(t, 2) if distance <= 10 else None,
                   min_range_m=round(min(r["range_m"] for r in rows), 3),
                   rms_error_deg=round(math.sqrt(sum(e*e for e in errors)/max(1,len(errors))), 3),
                   max_error_deg=round(max(abs(r["error_deg"]) for r in rows if r["range_m"] >= 20), 3),
                   max_command_dps=round(max(abs(r["command_dps"]) for r in rows), 3),
                   rms_command_accel_dps2=round(math.sqrt(sum(r["command_accel_dps2"]**2 for r in tracked)/max(1,len(tracked))), 3),
                   max_command_accel_dps2=round(max(r["command_accel_dps2"] for r in rows), 3),
                   missing_target_s=round(SEND_DT*sum(not r["target_acquired"] for r in rows), 3))
    return metrics, rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("reports/wasp_scenarios"))
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    defaults = dict(Wasp().params)
    candidates = {name.title(): dict(settings) for name, settings in Wasp.STEERING_PROFILES.items()}
    candidates["Experimental"] = dict(blend=.25, rate=90.0, gain=6.0)
    results, traces = [], {}
    for name, settings in candidates.items():
        for moving, condition, seed in itertools.product((False, True), CONDITIONS, range(3)):
            metrics, rows = simulate(dict(defaults, **settings), moving, condition, seed)
            results.append(dict(profile=name, **metrics))
            if moving and seed == 0:
                traces[(name, condition.name)] = rows
                write_csv(args.output/f"trace_{name.lower().replace(' ', '_')}_{condition.name}.csv", rows)
    write_csv(args.output/"profiles.csv", results)
    if args.grid:
        grid = []
        for straight, blend, rate, gain in itertools.product((0.0,.5,1.0), (.25,.5,1.0), (45.0,60.0,90.0), (2.0,4.0,6.0)):
            params = dict(defaults, straight=straight, blend=blend, rate=rate, gain=gain)
            for moving, condition in itertools.product((False,True), CONDITIONS):
                metrics, _ = simulate(params, moving, condition, seed=0)
                grid.append(metrics)
        write_csv(args.output/"grid.csv", grid)
    assumptions = dict(speed_mps=20, initial_range_m=250, lateral_target_speeds_mps=[0,10],
                       camera_fps=30, frame_size=[640,640], hfov_deg=62.2, vfov_deg=48.8,
                       straight_s=defaults["straight"], lead_s=defaults["lead"], velocity_tau_s=.35,
                       conditions=[c.__dict__ for c in CONDITIONS],
                       limitations=["Synthetic accepted detections, no detector resolution/performance model",
                                    "Assumed rate and translational dynamics, not measured airframe parameters",
                                    "No wind, gravity, thrust, drag, battery, or throttle-to-speed model",
                                    "10 m approach boundary; no contact, capture, or collision prediction"])
    (args.output/"assumptions.json").write_text(json.dumps(assumptions, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name in candidates:
        for col, condition_name in enumerate(("nominal", "detection_gap")):
            rows = traces[(name, condition_name)]
            axes[0,col].plot([r["t_s"] for r in rows], [r["error_deg"] for r in rows], label=name)
            axes[1,col].plot([r["t_s"] for r in rows], [r["command_dps"] for r in rows])
    for col, title in enumerate(("Accepted detections, 40 ms delay", "350 ms detection gap, 80 ms delay")):
        axes[0,col].set_title(title)
        axes[0,col].set_ylabel("Pointing error (degrees)")
        axes[1,col].set_ylabel("Requested body rate (degrees/s)")
        axes[1,col].set_xlabel("Time since launch (s)")
        for row in range(2):
            axes[row,col].grid(alpha=.25)
    axes[0,0].legend()
    fig.suptitle("Synthetic WASP tracking: 250 m start, 20 m/s own speed, 10 m/s lateral target\nAssumed dynamics and valid detections; stops at 10 m, does not predict capture", fontsize=12)
    fig.savefig(args.output/"comparison.png", dpi=170)
    plt.close(fig)
    for profile in candidates:
        selected = [r for r in results if r["profile"] == profile and r["motion"] == "crossing_10mps"]






if __name__ == "__main__":
    main()
