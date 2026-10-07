#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import types
if "ultralytics" not in sys.modules:
    sys.modules["ultralytics"] = types.ModuleType("ultralytics")
    sys.modules["ultralytics"].YOLO = None
import mission as M
import sim_auto_scenarios as A

WHYS = ("never_seen", "no_go", "lost_line", "lost_chase", "near_miss", "crash")
SCALE = 10


def pct(xs):
    xs = list(xs)
    return 100.0 * sum(xs) / max(1, len(xs))


def summarize(res):
    out = dict(hit={}, why={})
    for fam in ("come", "go", "other"):
        rs = [r for r in res if r["family"] == fam]
        out["hit"][fam] = pct(r["hit"] for r in rs)
        out["why"][fam] = {w: pct(r["why"] == w for r in rs) for w in WHYS}
    cg = [r for r in res if r["family"] in ("come", "go")]
    out["hit"]["score"] = pct(r["hit"] for r in cg)
    out["n"] = len(res)
    return out


def cell(rs):
    return dict(n=len(rs), hit=pct(r["hit"] for r in rs), near=pct(r["best"] < 1.5 for r in rs),
                why={w: pct(r["why"] == w for r in rs) for w in WHYS})


def turned(trace):
    if not trace:
        return []
    y0 = math.radians(trace[0][4])
    c, s = math.cos(y0), math.sin(y0)
    flat = []
    for row in trace:
        t, x, y, z, yaw, lean, bank, tx, ty, tz, ph, det, az, el, spd = row
        out = (t, x * c + y * s, -x * s + y * c, z, ((yaw - trace[0][4] + 180) % 360) - 180, lean, bank,
               tx * c + ty * s, -tx * s + ty * c, tz, ph, det, az, el, spd)
        flat.extend(int(round(v * SCALE)) for v in out)
    return flat


def outcome(r):
    if r is None:
        return None
    o = {k: r.get(k) for k in ("hit", "best", "why", "t_best", "t_seen", "t_go", "t_ai", "n_go")}
    o["trace"] = turned(r.get("trace") or [])
    return o


def build(after, before=None, after_h=None, before_h=None):
    ra = after["results"]
    rb = before["results"] if before else None
    key = lambda r: (r["scenario"], r["speed"], r["seed"])
    bmap = {key(r): r for r in rb} if rb else {}
    scen = []
    for name, sc in A.SCENARIOS.items():
        rows = []
        for sp in sc["speeds"]:
            a = [r for r in ra if r["scenario"] == name and r["speed"] == sp]
            if not a:
                continue
            b = [r for r in (rb or []) if r["scenario"] == name and r["speed"] == sp]
            rows.append(dict(speed=sp, after=cell(a), before=cell(b) if b else None))
        if rows:
            scen.append(dict(key=name, family=sc["family"], desc=sc["desc"], rows=rows))
    runs = []
    for r in ra:
        if not r.get("trace"):
            continue
        b = bmap.get(key(r))
        runs.append(dict(id=len(runs), scenario=r["scenario"], speed=r["speed"], start=r["start"], bf=r["bf"],
                         seed=r["seed"], r0=r["r0"], az0=r["az0"], dz0=r["dz0"],
                         after=outcome(r), before=outcome(b) if (b and b.get("trace")) else None))
    sa = summarize(ra)
    sb = summarize(rb) if rb else None
    n_cg = sum(1 for r in ra if r["family"] in ("come", "go"))
    verdict = [
        dict(label="Coming at us + going away", main=True, before=sb and sb["hit"]["score"], after=sa["hit"]["score"],
             sub=f"{n_cg} engagements, every enemy above us"),
        dict(label="Coming at us", before=sb and sb["hit"]["come"], after=sa["hit"]["come"],
             sub="straight, past our side, angled, overhead, diving, climbing, weaving, homing, stopping short"),
        dict(label="Going away", before=sb and sb["hit"]["go"], after=sa["hit"]["go"],
             sub="straight, angled, climbing, turning, weaving, running once we close"),
    ]
    if after_h:
        hb = summarize(before_h["results"])["hit"]["score"] if before_h else None
        verdict.append(dict(label="Held-out engagements", before=hb, after=summarize(after_h["results"])["hit"]["score"],
                            sub="a second, fresh set of seeds the tuning never saw"))
    verdict.append(dict(label="Other: hovering, crossing, out of view", before=sb and sb["hit"]["other"],
                        after=sa["hit"]["other"], sub="reported, not in the score"))
    lede = ("AUTO, pressed once, against enemy drones that start 6 to 45 m above the interceptor and either "
            "come at it or go away. Every engagement runs the real mission and guidance code against a "
            "modelled airframe, wind, camera and detector.")
    foot = ("Simulation, not flight. Real: mission.py (AUTO_CLIMB, AUTO_HOLD, AUTO_ALIGN, the straight line, "
            "the AI, the soft stop) and fc_control.InterceptLaw as start-wasp.sh builds it, with the tracker's "
            "AI levels. Modelled (tests/sim_sky_wind.py): a point-mass airframe in ALT_HOLD, total tilt held "
            "to 40°, climb changes limited to 15 m/s² (PILOT_ACCEL_Z), gusty wind Beaufort 0-3, a 30° up camera "
            "62.2 × 48.8°, a 25 Hz detector 120 ms late that misses more as the 0.30 m target shrinks (nothing "
            "beyond ~80 m), the barometer's noise. A hit is a closest approach under 0.5 m between centres. "
            f"{sa['n']} engagements per version, from the ground (1.5 m) and from a 5-10 m hover, AUTO "
            "pressed at t = 0, 50 s at most.")
    return dict(phases=list(M.PHASES), trace_scale=SCALE, lede=lede, foot=foot, verdict=verdict,
                summary=dict(after=sa, before=sb), scenarios=scen, runs=runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("after", help="sim_auto_scenarios --json output (with --traces)")
    ap.add_argument("--before", default="", help="the same campaign on older code")
    ap.add_argument("--after-heldout", default="", help="a --master-seed run of the new code")
    ap.add_argument("--before-heldout", default="", help="...and of the old code")
    ap.add_argument("-o", "--out", default="auto_trials.html")
    ap.add_argument("--bare", action="store_true",
                    help="no <!doctype>/<html> wrapper (for a host that adds its own)")
    a = ap.parse_args()
    load = lambda p: json.load(open(p)) if p else None
    data = build(load(a.after), load(a.before), load(a.after_heldout), load(a.before_heldout))
    tpl = open(os.path.join(HERE, "sim_auto_report.html")).read()
    page = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    if not a.bare:
        page = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
                + page + "\n</html>\n")
    with open(a.out, "w") as f:
        f.write(page)
    print(f"wrote {a.out}  ({len(page) / 1e6:.1f} MB, {len(data['runs'])} replays)")


if __name__ == "__main__":
    main()
