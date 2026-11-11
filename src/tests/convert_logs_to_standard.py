#!/usr/bin/env python3
"""Package Interceptor flight logs for delivery.

Reads the flight CSV, events JSONL and metadata JSON produced by
make_example_flight.py (or the real system) and creates a per-flight
delivery folder containing:

    <flight_id>/
        IKNOW/      Raw logs (control CSV, flight CSV, events, metadata, battery)
        ELKAK/      Five standard-schema CSVs:
            L-FC_<flight_id>.csv      Autopilot (attitude, PWM, vibration, height)
            L-DET_<flight_id>.csv     Perception (per-frame detections)
            L-GDN_<flight_id>.csv     Guidance (commands, LOS, range)
            L-EVT_<flight_id>.csv     Events (mode changes, launch, lock, etc.)
            L-BAT_<flight_id>.csv     Energy (battery)

Usage:
    python src/tests/convert_logs_to_standard.py [src_dir] [dst_dir]

Defaults:
    src_dir = examples/flight_logs
    dst_dir = examples/delivery
"""
import csv
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path


TIME_REF = "SYS_UTC"


def safe_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def convert_flight(flight_csv, events_jsonl, metadata_json, dst_dir):
    """Convert one flight's files into the five standard logs."""
    meta = json.load(open(metadata_json))
    flight_id = meta["flight_id"]
    flight_mode = meta["flight_mode"]
    started_utc = meta["started_utc"]

    rows = list(csv.DictReader(open(flight_csv)))
    events = [json.loads(line) for line in open(events_jsonl)]

    # Simulated battery: 6S LiPo starting at 25.2 V, draining under load.
    rnd = random.Random(hash(flight_id))
    batt_v = 25.2
    batt_wh = 0.0

    # ── L-FC: autopilot log ─────────────────────────────────────────
    fc_path = dst_dir / f"L-FC_{flight_id}.csv"
    fc_fields = [
        "sortie_id", "time_ref", "t_s", "utc_time",
        "flight_mode", "fc_mode", "armed",
        "roll_deg", "pitch_deg", "yaw_deg",
        "rollspeed_dps", "pitchspeed_dps", "yawspeed_dps",
        "height_m", "climb_mps",
        "lat", "lon", "gps_fix", "gps_sats",
        "pwm_roll", "pwm_pitch", "pwm_yaw", "pwm_thr",
        "vibe_x", "vibe_y", "vibe_z", "clip",
    ]
    with open(fc_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fc_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict(
                sortie_id=flight_id,
                time_ref=r.get("time_ref", TIME_REF),
                t_s=r["t_s"],
                utc_time=r["utc_time"],
                flight_mode=r["flight_mode"],
                fc_mode=r["mode"],
                armed=r["armed"],
                roll_deg=r["meas_roll_deg"],
                pitch_deg=r["meas_pitch_deg"],
                yaw_deg=r["meas_yaw_deg"],
                rollspeed_dps=r["rollspeed_dps"],
                pitchspeed_dps=r["pitchspeed_dps"],
                yawspeed_dps=r["yawspeed_dps"],
                height_m=r["height_m"],
                climb_mps=r["climb_mps"],
                lat=r.get("lat", ""),
                lon=r.get("lon", ""),
                gps_fix=r.get("gps_fix", ""),
                gps_sats=r.get("gps_sats", ""),
                pwm_roll=r["pwm_roll"],
                pwm_pitch=r["pwm_pitch"],
                pwm_yaw=r["pwm_yaw"],
                pwm_thr=r["pwm_thr"],
                vibe_x=r["vibe_x"],
                vibe_y=r["vibe_y"],
                vibe_z=r["vibe_z"],
                clip=r["clip"],
            ))

    # ── L-DET: perception log ────────────────────────────────────────
    det_path = dst_dir / f"L-DET_{flight_id}.csv"
    det_fields = [
        "sortie_id", "time_ref", "utc_time",
        "frame_id", "t_capture", "t_inference_done", "latency_ms",
        "class", "confidence", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "track_id",
    ]
    with open(det_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, det_fields, extrasaction="ignore")
        w.writeheader()
        frame_id = 0
        track_id = 1
        prev_fresh = False
        for r in rows:
            t_s = safe_float(r["t_s"], 0.0)
            det_age = safe_float(r.get("det_age_s"))
            conf = safe_float(r.get("det_conf"))
            bw = safe_float(r.get("bbox_w"))
            bh = safe_float(r.get("bbox_h"))
            fresh = r.get("detection_fresh") == "1"

            if not prev_fresh and fresh:
                track_id += 1
            prev_fresh = fresh

            frame_id += 1
            if conf is not None and bw is not None and bw > 0:
                t_capture = t_s - (det_age if det_age is not None else 0.04)
                t_inference = t_capture + rnd.uniform(0.008, 0.018)
                latency_ms = (t_inference - t_capture) * 1000.0
                ex = safe_float(r.get("ex"), 0.0)
                ey = safe_float(r.get("ey"), 0.0)
                cx = 320 + ex
                cy = 240 + ey
                bbox_x = int(cx - bw / 2)
                bbox_y = int(cy - bh / 2)
                w.writerow(dict(
                    sortie_id=flight_id,
                    time_ref=r.get("time_ref", TIME_REF),
                    utc_time=r.get("utc_time", ""),
                    frame_id=frame_id,
                    t_capture=f"{t_capture:.4f}",
                    t_inference_done=f"{t_inference:.4f}",
                    latency_ms=f"{latency_ms:.2f}",
                    **{"class": "target"},
                    confidence=f"{conf:.3f}",
                    bbox_x=bbox_x,
                    bbox_y=bbox_y,
                    bbox_w=int(bw),
                    bbox_h=int(bh),
                    track_id=track_id,
                ))

    # ── L-GDN: guidance log ──────────────────────────────────────────
    gdn_path = dst_dir / f"L-GDN_{flight_id}.csv"
    gdn_fields = [
        "sortie_id", "time_ref", "utc_time",
        "t_s", "guidance_mode", "guidance_state", "phase",
        "cmd_roll_rate_dps", "cmd_pitch_rate_dps", "cmd_yaw_rate_dps",
        "cmd_roll_stick", "cmd_pitch_stick", "cmd_yaw_stick", "cmd_thr_stick",
        "los_az_deg", "los_el_deg", "los_az_rate_dps", "los_el_rate_dps",
        "estimated_range_m", "closing_mps",
        "lat", "lon",
        "lock_flag",
    ]
    with open(gdn_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, gdn_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            phase = r.get("phase", "")
            if flight_mode == "wasp":
                gmode = "WASP"
                gstate = r.get("wasp_stage", "")
            else:
                gmode = "SEEKER_AUTO"
                gstate = phase
            lock = "1" if r.get("detection_fresh") == "1" else "0"
            w.writerow(dict(
                sortie_id=flight_id,
                time_ref=r.get("time_ref", TIME_REF),
                utc_time=r.get("utc_time", ""),
                t_s=r["t_s"],
                guidance_mode=gmode,
                guidance_state=gstate,
                phase=phase,
                cmd_roll_rate_dps=r.get("cmd_roll_rate_dps", ""),
                cmd_pitch_rate_dps=r.get("cmd_pitch_rate_dps", ""),
                cmd_yaw_rate_dps=r.get("cmd_yaw_rate_dps", ""),
                cmd_roll_stick=r.get("roll_stick", ""),
                cmd_pitch_stick=r.get("pitch_stick", ""),
                cmd_yaw_stick=r.get("yaw_stick", ""),
                cmd_thr_stick=r.get("throttle_stick", ""),
                los_az_deg=r.get("az_deg", ""),
                los_el_deg=r.get("el_deg", ""),
                los_az_rate_dps=r.get("los_az_dps", ""),
                los_el_rate_dps=r.get("los_el_dps", ""),
                estimated_range_m=r.get("range_m", ""),
                closing_mps=r.get("closing_mps", ""),
                lat=r.get("lat", ""),
                lon=r.get("lon", ""),
                lock_flag=lock,
            ))

    # ── L-EVT: event log ─────────────────────────────────────────────
    evt_path = dst_dir / f"L-EVT_{flight_id}.csv"
    evt_fields = [
        "sortie_id", "time_ref",
        "t_utc", "elapsed_s", "event", "detail", "source",
    ]

    FAILSAFE_MODES = ("RTL", "LAND", "SMART_RTL", "AUTO_RTL")

    # Map internal event types to the standard vocabulary
    def classify_event(e):
        etype = e.get("event", "")
        value = e.get("value", "")
        prev = e.get("previous", "")
        message = e.get("message", "")
        reason = e.get("reason", "")

        if etype == "start":
            return "launch", f"flight started, phase={e.get('phase','')}", "mission"
        if etype == "end":
            return "wave_off", f"flight ended: {reason}", "mission"
        if etype == "phase":
            detail = f"{prev} -> {value}" if prev else f"-> {value}"
            evtype = "mode_change"
            if value in ("PUSH", "AI"):
                evtype = "engage"
            elif value == "SAFEDOWN":
                evtype = "e_stop"
            elif value in ("IDLE", "STOP", "PILOT"):
                evtype = "abort" if "ABORT" in message else "wave_off"
            elif value == "WASP_DONE":
                evtype = "wave_off"
            return evtype, detail, "mission"
        if etype == "detection_fresh":
            if value == 1:
                return "lock", f"target acquired (was: {prev})", "perception"
            else:
                return "lock_lost", f"target lost (was: {prev})", "perception"
        if etype == "wasp_stage":
            return "mode_change", f"WASP stage: {prev} -> {value}", "guidance"
        if etype == "idle_stage":
            if value == "OFF" and prev and prev != "OFF":
                return "mode_change", f"idle ended ({prev} -> OFF)", "mission"
            elif value != "OFF":
                return "mode_change", f"idle: {prev} -> {value}", "mission"
            return "mode_change", f"idle stage: -> {value}", "mission"
        if etype == "mission":
            msg_lower = message.lower()
            if "failsafe" in msg_lower or any(
                    f"fc in {m.lower()}" in msg_lower for m in FAILSAFE_MODES):
                return "failsafe", message, "fc"
            if "safedown" in msg_lower or "e-stop" in msg_lower or "emergency" in msg_lower:
                return "e_stop", message, "mission"
            if "abort" in msg_lower:
                return "abort", message, "mission"
            if "cue" in msg_lower or "designat" in msg_lower or "asterix" in msg_lower:
                return "cue_received", message, "guidance"
            if "launched" in msg_lower or "launch" in msg_lower:
                return "launch", message, "mission"
            if "target" in msg_lower and "acquired" in msg_lower:
                return "lock", message, "guidance"
            if "target" in msg_lower and "lost" in msg_lower:
                return "lock_lost", message, "guidance"
            if "target" in msg_lower and ("camera" in msg_lower or "in the" in msg_lower):
                return "lock", message, "perception"
            if "engage" in msg_lower or "push" in msg_lower:
                return "engage", message, "mission"
            if "pilot" in msg_lower:
                return "wave_off", message, "pilot"
            if "idle" in msg_lower:
                return "mode_change", message, "mission"
            return "mode_change", message, "mission"
        return etype, str(e), "unknown"

    with open(evt_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, evt_fields, extrasaction="ignore")
        w.writeheader()
        w.writerow(dict(
            sortie_id=flight_id,
            time_ref=TIME_REF,
            t_utc=started_utc,
            elapsed_s="0.0",
            event="arming_state",
            detail=f"armed, fc_mode={meta['settings']['fc_mode']}",
            source="fc",
        ))
        for e in events:
            evtype, detail, source = classify_event(e)
            w.writerow(dict(
                sortie_id=flight_id,
                time_ref=e.get("time_ref", TIME_REF),
                t_utc=e["utc_time"],
                elapsed_s=e["elapsed_s"],
                event=evtype,
                detail=detail,
                source=source,
            ))
        w.writerow(dict(
            sortie_id=flight_id,
            time_ref=TIME_REF,
            t_utc=meta["ended_utc"],
            elapsed_s=meta["duration_s"],
            event="arming_state",
            detail=f"disarmed, reason={meta['end_reason']}",
            source="fc",
        ))

    # ── L-BAT: energy log ────────────────────────────────────────────
    bat_path = dst_dir / f"L-BAT_{flight_id}.csv"
    bat_fields = [
        "sortie_id", "time_ref",
        "t_s", "utc_time", "voltage_v", "current_a", "battery_pct",
        "consumed_wh",
    ]

    # Prefer the real battery_*.csv produced by BatteryLog if it exists
    src_dir = Path(flight_csv).parent
    bat_sources = sorted(src_dir.glob("battery_*.csv"))
    stamp = "_".join(flight_id.split("_")[1:3])
    real_bat = [p for p in bat_sources if stamp in p.name]

    with open(bat_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, bat_fields, extrasaction="ignore")
        w.writeheader()
        if real_bat:
            for r in csv.DictReader(open(real_bat[0])):
                w.writerow(dict(sortie_id=flight_id,
                                time_ref=r.get("time_ref", TIME_REF),
                                **{k: r.get(k, "") for k in bat_fields
                                   if k not in ("sortie_id", "time_ref")}))
        else:
            prev_t = None
            next_write = None
            for r in rows:
                t_s = safe_float(r["t_s"], 0.0)
                phase = r.get("phase", "")
                h = safe_float(r.get("height_m"), 0.0)
                if phase in ("PUSH", "AI", "WASP_FLIGHT"):
                    base_current = rnd.uniform(22, 28)
                elif phase in ("AUTO_CLIMB", "AUTO_ALIGN"):
                    base_current = rnd.uniform(16, 22)
                elif h > 0.5:
                    base_current = rnd.uniform(12, 16)
                else:
                    base_current = rnd.uniform(1, 3)
                current = max(0.5, base_current + rnd.gauss(0, 0.5))
                dt = 0.0 if prev_t is None else t_s - prev_t
                prev_t = t_s
                power_w = batt_v * current
                batt_wh += power_w * dt / 3600.0
                batt_v = max(19.5, 25.2 - batt_wh * 0.08 - current * 0.012
                             + rnd.gauss(0, 0.02))
                if next_write is None or t_s >= next_write:
                    next_write = (math.floor(t_s) + 1.0) if next_write is None else next_write + 1.0
                    w.writerow(dict(
                        sortie_id=flight_id,
                        time_ref=TIME_REF,
                        t_s=f"{t_s:.3f}",
                        utc_time="",
                        voltage_v=f"{batt_v:.2f}",
                        current_a=f"{current:.1f}",
                        battery_pct="",
                        consumed_wh=f"{batt_wh:.3f}",
                    ))

    return flight_id, fc_path, det_path, gdn_path, evt_path, bat_path


def main(src_dir, dst_dir):
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # Find all flight metadata JSONs in the source
    metas = sorted(src.glob("flight_*.json"))
    if not metas:
        print(f"No flight_*.json found in {src}", file=sys.stderr)
        return 1

    for meta_path in metas:
        base = meta_path.stem  # e.g. flight_20261110_101500_001_wasp
        csv_path = meta_path.with_suffix(".csv")
        events_path = meta_path.parent / (base + ".events.jsonl")

        if not csv_path.exists() or not events_path.exists():
            print(f"Skipping {base}: missing csv or events", file=sys.stderr)
            continue

        # Per-flight folder with IKNOW (native) and ELKAK (standard) subfolders
        flight_dir = dst / base
        native_dir = flight_dir / "IKNOW"
        standard_dir = flight_dir / "ELKAK"
        native_dir.mkdir(parents=True, exist_ok=True)
        standard_dir.mkdir(parents=True, exist_ok=True)

        # Convert to standard format
        fid, *paths = convert_flight(csv_path, events_path, meta_path, standard_dir)

        # Copy native log files into native/
        stamp = "_".join(base.split("_")[1:3])
        native_files = []
        for native in (csv_path, events_path, meta_path):
            dest = native_dir / native.name
            shutil.copy2(native, dest)
            native_files.append(dest)
        for extra in sorted(src.glob(f"*{stamp}*")):
            if extra not in (csv_path, events_path, meta_path):
                dest = native_dir / extra.name
                shutil.copy2(extra, dest)
                native_files.append(dest)

        print(f"\n{'=' * 60}")
        print(f"  {fid}")
        print(f"{'=' * 60}")
        print(f"  IKNOW/")
        for p in sorted(native_files):
            print(f"    {p.stat().st_size:>9}  {p.name}")
        print(f"  ELKAK/")
        for p in paths:
            print(f"    {p.stat().st_size:>9}  {p.name}")

    print(f"\nAll flights written to {dst}")
    return 0


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[2]
    src = sys.argv[1] if len(sys.argv) > 1 else here / "examples" / "flight_logs"
    dst = sys.argv[2] if len(sys.argv) > 2 else here / "examples" / "delivery"
    sys.exit(main(src, dst))
