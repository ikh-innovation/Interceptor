#!/usr/bin/env python3
import csv
import glob
import io
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    txt = open(path, "rb").read().replace(b"\x00", b"").decode(errors="ignore")
    return list(csv.DictReader(io.StringIO(txt)))


def col(rows, k):
    out = []
    for r in rows:
        v = r.get(k, "")
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return np.array(out)


def decode_angle(pwm, tr=1500, dz=20, lo=1100, hi=1900, amax=40.0):
    out = np.zeros_like(pwm, dtype=float)
    lo_m = pwm < tr - dz
    hi_m = pwm > tr + dz
    out[lo_m] = (pwm[lo_m] - (tr - dz)) / ((tr - dz) - lo)
    out[hi_m] = (pwm[hi_m] - (tr + dz)) / (hi - (tr + dz))
    return np.clip(out, -1, 1) * amax


def lag_s(cmd, meas, dt=0.02, max_lag=25):
    ok = ~(np.isnan(cmd) | np.isnan(meas))
    c, m = cmd[ok] - np.nanmean(cmd[ok]), meas[ok] - np.nanmean(meas[ok])
    if len(c) < 60 or np.std(c) < 0.3:
        return None
    best, arg = -1e9, 0
    for k in range(0, max_lag):
        v = float(np.dot(c[:len(c) - k], m[k:]))
        if v > best:
            best, arg = v, k
    return arg * dt


def band_share(x, fs=50.0, lo=2.0, hi=10.0):
    x = x[~np.isnan(x)]
    if len(x) < 64:
        return None
    x = x - x.mean()
    F = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    tot = F[f > 0].sum()
    return None if tot <= 0 else float(F[(f > lo) & (f < hi)].sum() / tot)


def main(path):
    rows = load(path)
    if not rows or "phase" not in rows[0]:
        print(f"{path}: no mission columns (flown with WASP_APP=auto?)")
        return 1
    ph = np.array([r["phase"] for r in rows])
    t = col(rows, "t_s")
    print(f"# Flight report: {os.path.basename(path)}")
    dur = {p: float(np.sum(ph == p)) * 0.02 for p in sorted(set(ph))}
    print("phases (s): " + ", ".join(f"{k} {v:.1f}" for k, v in dur.items()))

    drive = np.isin(ph, ["PUSH", "AI", "STOP"])
    print("\n## 1. CONTROLLER: does the FC fly what was sent?")
    if drive.sum() < 50:
        print("   not enough PUSH/AI time to judge")
    else:
        for axis, pwm_k, meas_k, sign in (("pitch", "pwm_pitch", "meas_pitch_deg", 1.0),
                                          ("roll", "pwm_roll", "meas_roll_deg", 1.0)):
            pwm = col(rows, pwm_k)[drive]
            cmd = decode_angle(pwm) * sign
            meas = col(rows, meas_k)[drive]
            err = meas - cmd
            lg = lag_s(cmd, meas)
            print(f"   {axis:5s}: commanded {np.nanmin(cmd):+.0f}..{np.nanmax(cmd):+.0f} deg, "
                  f"tracking error median {np.nanmedian(np.abs(err)):.1f} deg, "
                  f"p95 {np.nanpercentile(np.abs(err), 95):.1f} deg, lag "
                  + ("n/a" if lg is None else f"{lg * 1000:.0f} ms"))
        r_share = band_share(col(rows, "pwm_roll")[ph == "AI"])
        if r_share is not None:
            print(f"   roll stick energy at 2-10 Hz during AI: {r_share * 100:.0f}% "
                  f"(chatter; under ~15% is smooth)")

    print("\n## 2. JETSON: how old is the picture?")
    age = col(rows, "cap_age_s")
    age = age[~np.isnan(age)]
    da = col(rows, "det_age_s")
    if age.size:
        print(f"   capture -> command age: median {np.median(age) * 1000:.0f} ms, "
              f"p95 {np.percentile(age, 95) * 1000:.0f} ms (above ~150 ms, fix the pipeline "
              f"before raising speed)")
    fresh = int(np.sum(np.diff(np.nan_to_num(da, nan=1e9)) < -0.005))
    span = (t[-1] - t[0]) if len(t) > 1 else 0
    if span > 0:
        print(f"   detections: ~{fresh / span:.1f} per second over the flight")
    conf = col(rows, "det_conf")
    conf = conf[~np.isnan(conf)]
    if conf.size:
        print(f"   detection confidence: median {np.median(conf):.2f}, "
              f"{np.mean(conf >= 0.2) * 100:.0f}% of logged cycles >= 0.20")

    print("\n## 3. VIBRATION: is the airframe shaking the camera?")
    vx, vy, vz = col(rows, "vibe_x"), col(rows, "vibe_y"), col(rows, "vibe_z")
    clip = col(rows, "clip")
    air = np.isin(ph, ["HOVER", "CLIMB", "DESCEND", "SCOUT", "PUSH", "AI", "STOP"])
    if np.isnan(vz[air]).all():
        print("   no VIBRATION data")
    else:
        vm = np.nanmax(np.vstack([vx, vy, vz])[:, air], axis=0)
        cl = np.nanmax(clip) - np.nanmin(clip) if not np.isnan(clip).all() else 0
        verdict = ("GOOD" if np.nanpercentile(vm, 95) < 30 else
                   "MARGINAL" if np.nanpercentile(vm, 95) < 60 else "BAD")
        print(f"   vibration (max axis) median {np.nanmedian(vm):.0f}, p95 "
              f"{np.nanpercentile(vm, 95):.0f} m/s^2 -> {verdict} (<30 good, >60 bad); "
              f"new clipping events {int(cl)}")
    rs = np.abs(col(rows, "rollspeed_dps")) + np.abs(col(rows, "pitchspeed_dps"))
    el = col(rows, "el_cam_deg")
    new = np.r_[False, np.diff(da) < -0.01] & air
    if new.sum() > 20:
        eln = el[new]
        jit = np.abs(np.diff(eln))
        rsn = rs[new][1:]
        ok = ~(np.isnan(jit) | np.isnan(rsn))
        if ok.sum() > 10 and np.std(rsn[ok]) > 0:
            cc = float(np.corrcoef(jit[ok], rsn[ok])[0, 1])
            print(f"   detection jitter frame-to-frame: median {np.median(jit[ok]):.2f} deg; "
                  f"correlation with body rates {cc:+.2f} "
                  f"({'follows the airframe: SHAKE / camera mount' if cc > 0.4 else 'does not follow the airframe: the DETECTOR'})")

    print("\n## Verdict")
    print("   read the three sections: the largest number relative to its threshold is the culprit.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        p = sys.argv[1]
    else:
        files = sorted(glob.glob(os.path.join(HERE, "data", "control_*.csv")),
                       key=os.path.getmtime)
        if not files:
            print("no control_*.csv in data/")
            sys.exit(1)
        p = files[-1]
    sys.exit(main(p))
