#!/usr/bin/env python3
"""
End-to-end test against a SIMULATED ArduPilot Copter that reproduces the real
firmware's override-gating rule:

    void GCS_MAVLINK::handle_rc_channels_override(const mavlink_message_t &msg)
    {
        if (msg.sysid != sysid_my_gcs()) return;   // silently dropped
        ...
    }

and echoes an accepted override back in RC_CHANNELS (as RC_Channel::update()
copies override_value into radio_in).  That lets us prove, without hardware:

  A. the commander adopts the FC's SYSID_MYGCS and its overrides are accepted
  B. it works for a NON-default SYSID_MYGCS too (read, not assumed)
  C. the old behaviour (source_system=1 vs SYSID_MYGCS=255) is refused
  D. RC_OPTIONS bit 1 (ignore overrides) is detected and reported
"""
import sys
from time import perf_counter, sleep

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymavlink import mavutil
from fc_control import AcroCommander, ControlLaw

fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


from fake_fc import FakeArduPilot


def run_case(name, port, *, sysid_mygcs=255, rc_options=0, force_sysid=None,
             armed=True, expect_confirmed=True, settle=4.0):
    print(f"\n=== {name} ===")
    fc = FakeArduPilot(port, sysid_mygcs=sysid_mygcs, rc_options=rc_options,
                       armed=armed)
    fc.start()
    sleep(0.3)

    law = ControlLaw(kp_yaw=2.0, kp_pitch=2.0, slew_per_s=1e9)
    cmdr = AcroCommander(conn=f"udpin:127.0.0.1:{port}", transport="mavlink",
                         law=law, send_hz=20.0, force_sysid=force_sysid,
                         log_dir=None, verbose=False)
    try:
        cmdr.connect()
    except Exception as e:
        check(f"{name}: connect", False, str(e))
        fc.shutdown()
        return None

    check("linked to the simulated FC", cmdr.linked)
    check("read SYSID_MYGCS off the FC",
          cmdr.backend.sysid_mygcs == sysid_mygcs,
          f"got {cmdr.backend.sysid_mygcs}")
    check("read the ACRO full-stick yaw rate off the FC",
          abs(cmdr.law.max_yaw_rate - 3.5343) < 0.01,
          f"{cmdr.law.max_yaw_rate:.4f} rad/s (202.5 deg/s)")
    check("read the RC map off the FC (yaw=ch4, pitch=ch2)",
          cmdr.backend.rc["yaw"]["chan"] == 4
          and cmdr.backend.rc["pitch"]["chan"] == 2)
    check("mode reported as ACRO", cmdr.mode_name == "ACRO", cmdr.mode_name)

    # Drive a target hard right of centre for a few seconds.
    t0 = perf_counter()
    while perf_counter() - t0 < settle:
        c = cmdr.compute(+200, -80, "LOCK", 640, 640)
        cmdr.update(c, (200, -80), have_face=True)
        sleep(0.05)

    confirmed = cmdr.override_confirmed
    applied = dict(fc.overrides)
    rejected = set(fc.rejected_from)
    sent_from = cmdr.backend.master.mav.srcSystem
    cmdr.close()
    sleep(0.2)
    released = dict(fc.overrides)
    fc.shutdown()

    print(f"    sent from sysid={sent_from}  FC applied={applied}  "
          f"FC rejected from={rejected or '{}'}  confirmed={confirmed}")
    return dict(confirmed=confirmed, applied=applied, rejected=rejected,
                sent_from=sent_from, released=released)


# ── A. default SYSID_MYGCS=255 ────────────────────────────────────────────────
r = run_case("A. ArduPilot default SYSID_MYGCS=255", 14601, sysid_mygcs=255)
if r:
    check("override CONFIRMED via the RC_CHANNELS echo", r["confirmed"] is True)
    check("we sent from sysid 255", r["sent_from"] == 255)
    check("FC applied an override on the yaw channel (ch4)", 4 in r["applied"],
          str(r["applied"]))
    check("yaw override is right of trim (target was right of centre)",
          r["applied"].get(4, 0) > 1500, f"ch4={r['applied'].get(4)}")
    check("FC applied an override on the pitch channel (ch2)", 2 in r["applied"])
    check("target above centre -> pitch above trim (nose up)",
          r["applied"].get(2, 0) > 1500, f"ch2={r['applied'].get(2)}")
    check("close() released every channel back to the pilot",
          r["released"] == {}, str(r["released"]))
    check("nothing was rejected", not r["rejected"], str(r["rejected"]))

# ── B. non-default SYSID_MYGCS: must be READ, not assumed ────────────────────
r = run_case("B. non-default SYSID_MYGCS=42 (must be read from the FC)",
             14602, sysid_mygcs=42)
if r:
    check("adopted the FC's non-default sysid", r["sent_from"] == 42)
    check("override CONFIRMED with a non-default SYSID_MYGCS",
          r["confirmed"] is True)
    check("FC applied the override", bool(r["applied"]), str(r["applied"]))

# ── C. the OLD bug: sending from sysid 1 against SYSID_MYGCS=255 ─────────────
r = run_case("C. regression: old behaviour (source_system=1) must be REFUSED",
             14603, sysid_mygcs=255, force_sysid=1, settle=5.0)
if r:
    check("we sent from sysid 1 (the old hardcoded value)", r["sent_from"] == 1)
    check("FC applied NOTHING (this was the silent failure)", not r["applied"],
          str(r["applied"]))
    check("FC recorded the drop from sysid 1", 1 in r["rejected"])
    check("commander reports override_confirmed=False (no longer silent)",
          r["confirmed"] is False)

# ── D. RC_OPTIONS bit 1 = ignore overrides ───────────────────────────────────
r = run_case("D. RC_OPTIONS=2 (IGNORE_OVERRIDES) is detected", 14604,
             sysid_mygcs=255, rc_options=2, settle=5.0)
if r:
    check("FC applied nothing with overrides disabled", not r["applied"])
    check("commander reports override_confirmed=False", r["confirmed"] is False)

# ── E. disarmed: must NOT drive the sticks ───────────────────────────────────
r = run_case("E. DISARMED -> sticks stay released (safety gate)", 14605,
             sysid_mygcs=255, armed=False, settle=3.0)
if r:
    check("no override sent while disarmed", not r["applied"], str(r["applied"]))

print("\n" + "=" * 62)
if fails:
    print(f"{len(fails)} FAILURE(S):")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("ALL SIMULATED-FC CHECKS PASSED")
