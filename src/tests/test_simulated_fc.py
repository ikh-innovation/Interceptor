#!/usr/bin/env python3
import math
import sys
from time import perf_counter, sleep

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymavlink import mavutil
from fc_control import AcroCommander, ControlLaw, InterceptLaw

fails = []


def check(name, cond, detail=""):

    if not cond:
        fails.append(name)


from fake_fc import FakeArduPilot


def run_case(name, port, *, sysid_mygcs=255, rc_options=0, force_sysid=None,
             armed=True, expect_confirmed=True, settle=4.0):

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

    t0 = perf_counter()
    while perf_counter() - t0 < settle:
        c = cmdr.compute(+200, -80, "LOCK", 640, 640)
        cmdr.update(c, (200, -80), have_target=True)
        sleep(0.05)

    confirmed = cmdr.override_confirmed
    applied = dict(fc.overrides)
    rejected = set(fc.rejected_from)
    sent_from = cmdr.backend.master.mav.srcSystem
    cmdr.close()
    sleep(0.2)
    released = dict(fc.overrides)
    fc.shutdown()



    return dict(confirmed=confirmed, applied=applied, rejected=rejected,
                sent_from=sent_from, released=released)


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

r = run_case("B. non-default SYSID_MYGCS=42 (must be read from the FC)",
             14602, sysid_mygcs=42)
if r:
    check("adopted the FC's non-default sysid", r["sent_from"] == 42)
    check("override CONFIRMED with a non-default SYSID_MYGCS",
          r["confirmed"] is True)
    check("FC applied the override", bool(r["applied"]), str(r["applied"]))

r = run_case("C. regression: old behaviour (source_system=1) must be REFUSED",
             14603, sysid_mygcs=255, force_sysid=1, settle=5.0)
if r:
    check("we sent from sysid 1 (the old hardcoded value)", r["sent_from"] == 1)
    check("FC applied NOTHING (this was the silent failure)", not r["applied"],
          str(r["applied"]))
    check("FC recorded the drop from sysid 1", 1 in r["rejected"])
    check("commander reports override_confirmed=False (no longer silent)",
          r["confirmed"] is False)

r = run_case("D. RC_OPTIONS=2 (IGNORE_OVERRIDES) is detected", 14604,
             sysid_mygcs=255, rc_options=2, settle=5.0)
if r:
    check("FC applied nothing with overrides disabled", not r["applied"])
    check("commander reports override_confirmed=False", r["confirmed"] is False)

r = run_case("E. DISARMED -> sticks stay released (safety gate)", 14605,
             sysid_mygcs=255, armed=False, settle=3.0)
if r:
    check("no override sent while disarmed", not r["applied"], str(r["applied"]))

def run_mode_gate_case(port):

    fc = FakeArduPilot(port, sysid_mygcs=255, armed=True, mode=0)
    fc.start()
    sleep(0.3)

    law = ControlLaw(kp_yaw=2.0, kp_pitch=2.0, slew_per_s=1e9)
    cmdr = AcroCommander(conn=f"udpin:127.0.0.1:{port}", transport="mavlink",
                         law=law, send_hz=20.0, log_dir=None, verbose=False)
    try:
        cmdr.connect()
    except Exception as e:
        check("F: connect", False, str(e))
        fc.shutdown()
        return

    check("gate is on by default (REQUIRE_RATE_MODE default)",
          cmdr.require_rate_mode is True)

    def drive(seconds):
        t0 = perf_counter()
        while perf_counter() - t0 < seconds:
            c = cmdr.compute(+200, -80, "LOCK", 640, 640)
            cmdr.update(c, (200, -80), have_target=True)
            sleep(0.05)

    results = []
    for i, (mode, label) in enumerate([(0, "STABILIZE"), (1, "ACRO"),
                                       (0, "STABILIZE"), (1, "ACRO"),
                                       (0, "STABILIZE")]):
        fc.mode = mode
        fc.overrides.clear()
        drive(1.5)
        applied = dict(fc.overrides)
        results.append((label, applied, cmdr.rate_mode_ok))



    cmdr.close()
    sleep(0.2)
    fc.shutdown()

    check("1st STABILIZE: AI drives NOTHING, pilot has every channel",
          not results[0][1], str(results[0][1]))
    check("1st STABILIZE: rate_mode_ok is False", results[0][2] is False)
    check("switch to ACRO: AI takes yaw/pitch",
          bool(results[1][1]), str(results[1][1]))
    check("in ACRO the yaw override is right of trim (target right of centre)",
          results[1][1].get(4, 0) > 1500, f"ch4={results[1][1].get(4)}")
    check("1st ACRO: rate_mode_ok is True", results[1][2] is True)
    check("switch BACK to STABILIZE: every channel released to the pilot",
          not results[2][1], str(results[2][1]))
    check("2nd ACRO: AI takes the sticks AGAIN (gate is not a one-shot latch)",
          bool(results[3][1]), str(results[3][1]))
    check("2nd STABILIZE: released AGAIN", not results[4][1], str(results[4][1]))


run_mode_gate_case(14606)


def run_gate_off_case(port):

    fc = FakeArduPilot(port, sysid_mygcs=255, armed=True, mode=0)
    fc.start()
    sleep(0.3)
    law = ControlLaw(kp_yaw=2.0, kp_pitch=2.0, slew_per_s=1e9)
    cmdr = AcroCommander(conn=f"udpin:127.0.0.1:{port}", transport="mavlink",
                         law=law, send_hz=20.0, log_dir=None, verbose=False,
                         require_rate_mode=False)
    try:
        cmdr.connect()
    except Exception as e:
        check("G: connect", False, str(e))
        fc.shutdown()
        return
    t0 = perf_counter()
    while perf_counter() - t0 < 2.0:
        c = cmdr.compute(+200, -80, "LOCK", 640, 640)
        cmdr.update(c, (200, -80), have_target=True)
        sleep(0.05)
    applied = dict(fc.overrides)
    cmdr.close()
    sleep(0.2)
    fc.shutdown()

    check("with the gate off, STABILIZE still gets driven (old behaviour)",
          bool(applied), str(applied))


run_gate_off_case(14607)


def run_intercept_case(port):

    fc = FakeArduPilot(port, sysid_mygcs=255, armed=True, mode=0)
    fc.start()
    sleep(0.3)
    fc.radio[6] = 2000
    law = InterceptLaw(cam_tilt_deg=15.0, roll_pn_gain=45.0, slew_per_s=1e9,
                       max_lean_deg=20.0, climb_gain=8.0)
    cmdr = AcroCommander(conn=f"udpin:127.0.0.1:{port}", transport="mavlink",
                         axes=("roll", "pitch", "yaw", "throttle"), law=law,
                         send_hz=50.0, log_dir=None, verbose=False,
                         steer_modes=frozenset({"ALT_HOLD"}),
                         commit_chan=6, commit_abort_below=0.05)
    try:
        cmdr.connect()
    except Exception as e:
        check("H: connect", False, str(e))
        fc.shutdown()
        return

    def drive(seconds):
        t0 = perf_counter()
        while perf_counter() - t0 < seconds:
            c = cmdr.compute(+120, -60, "LOCK", 640, 640, bbox=(0, 0, 30, 30))
            cmdr.update(c, (120, -60), have_target=True)
            sleep(0.02)

    check("H: angle-mode scaling was read off the FC",
          abs(law.angle_max - __import__("math").radians(30.0)) < 1e-6
          and abs(law.max_climb_mps - 2.5) < 1e-6,
          f"ANGLE_MAX={__import__('math').degrees(law.angle_max):.0f} deg, "
          f"climb<={law.max_climb_mps:.1f} m/s")

    drive(1.5)
    in_stab = dict(fc.overrides)
    check("H: STABILIZE is NOT driven when the gate wants ALT_HOLD",
          not in_stab, str(in_stab or "{}"))

    fc.mode = 2
    sleep(1.2)
    drive(2.0)
    in_ah = dict(fc.overrides)
    check("H: ALT_HOLD IS driven, on all four channels",
          set(in_ah) == {1, 2, 3, 4}, f"channels {sorted(in_ah)}")
    check("H: the FC confirmed the override echo", cmdr.override_confirmed is True,
          f"confirmed={cmdr.override_confirmed}")
    check("H: throttle is driven around MID (1500), not around TRIM (1002)",
          in_ah.get(3, 0) > 1500, f"ch3={in_ah.get(3)}")

    check("H: the commit channel is never written", 6 not in in_ah,
          f"channels {sorted(in_ah)}")

    fc.radio[6] = 1000
    sleep(0.4)
    drive(0.6)
    after = dict(fc.overrides)
    check("H: dropping the commit stick releases EVERY channel", not after,
          str(after or "{}"))
    check("H: ...and it is reported as a commit abort", cmdr._commit_aborted)

    cmdr.close()
    sleep(0.2)
    fc.shutdown()


run_intercept_case(14608)


def run_no_attitude_case(port):

    fc = FakeArduPilot(port, sysid_mygcs=255, armed=True, mode=2, attitude=False)
    fc.start()
    sleep(0.3)
    law = InterceptLaw(cam_tilt_deg=15.0, roll_pn_gain=45.0, slew_per_s=1e9)
    cmdr = AcroCommander(conn=f"udpin:127.0.0.1:{port}", transport="mavlink",
                         axes=("roll", "pitch", "yaw", "throttle"), law=law,
                         send_hz=50.0, log_dir=None, verbose=False,
                         steer_modes=frozenset({"ALT_HOLD"}))
    try:
        cmdr.connect()
    except Exception as e:
        check("I: connect", False, str(e))
        fc.shutdown()
        return
    check("I: preflight names the missing ATTITUDE as a blocker",
          any("ATTITUDE" in b for b in cmdr.backend.blockers)
          or law.pitch_mode == "intercept")
    t0 = perf_counter()
    last = None
    while perf_counter() - t0 < 1.5:
        last = cmdr.compute(+120, -60, "LOCK", 640, 640, bbox=(0, 0, 30, 30))
        cmdr.update(last, (120, -60), have_target=True)
        sleep(0.02)
    sleep(0.3)
    applied = dict(fc.overrides)
    cmdr.close()
    sleep(0.2)
    fc.shutdown()
    check("I: roll and throttle are EXACTLY zero without attitude",
          last["roll_stick"] == 0.0 and last["throttle_stick"] == 0.0,
          f"roll={last['roll_stick']:+.4f} thr={last['throttle_stick']:+.4f}")
    check("I: yaw and pitch still drive, so it aims and flies forward",
          last["yaw_stick"] != 0.0 and last["pitch_stick"] < 0.0,
          f"yaw={last['yaw_stick']:+.3f} pitch={last['pitch_stick']:+.3f}")
    check("I: roll sits at TRIM on the wire (a centred stick, not a lean)",
          applied.get(1) == 1500, f"ch1={applied.get(1)}")


run_no_attitude_case(14609)


def run_time_alignment_case(port):

    fc = FakeArduPilot(port, sysid_mygcs=255, armed=True, mode=2)
    fc.start()
    sleep(0.3)
    law = InterceptLaw(cam_tilt_deg=15.0, roll_pn_gain=45.0, slew_per_s=1e9,
                       deadband_frac=0.0)
    cmdr = AcroCommander(conn=f"udpin:127.0.0.1:{port}", transport="mavlink",
                         axes=("roll", "pitch", "yaw", "throttle"), law=law,
                         send_hz=50.0, log_dir=None, verbose=False,
                         steer_modes=frozenset({"ALT_HOLD"}))
    try:
        cmdr.connect()
    except Exception as e:
        check("J: connect", False, str(e))
        fc.shutdown()
        return

    half = math.tan(math.radians(62.2 / 2))
    worst = 0.0
    for k in range(60):
        yaw = math.radians(6.0) * math.sin(k * 0.25)
        fc.att.update(yaw=yaw, yawspeed=math.radians(90.0))
        sleep(0.06)
        az_px = -(320.0) * math.tan(yaw) / half
        cmd = cmdr.compute(az_px, 0, "LOCK", 640, 640, bbox=(0, 0, 30, 30),
                           capture_t=perf_counter())
        cmdr.update(cmd, (int(az_px), 0), have_target=True)
        if k > 25:
            worst = max(worst, abs(cmd["bank_deg"]))

    cmdr.close()
    sleep(0.2)
    fc.shutdown()
    check("J: a nose swinging on a constant bearing commands ~no bank",
          worst < 1.0, f"worst {worst:.3f} deg (was 27.0 before time alignment)")


run_time_alignment_case(14610)



if fails:

    for f in fails:
        pass
    sys.exit(1)
