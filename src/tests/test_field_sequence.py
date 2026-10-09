import sys, math
from time import perf_counter, sleep
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fc_control import AcroCommander, InterceptLaw
from fake_fc import FakeArduPilot

PORT = 14801
bad = []
def ck(n, c, d=""):

    if not c: bad.append(n)

fc = FakeArduPilot(PORT, sysid_mygcs=255, armed=False, mode=0)
fc.start(); sleep(0.4)
fc.att.update(roll=0.0, pitch=0.0, yaw=0.0, yawspeed=0.0)
fc.radio[6] = 2000

law = InterceptLaw(cam_tilt_deg=10.0, roll_pn_gain=45.0, max_lean_deg=20.0,
                   max_bank_deg=30.0, climb_gain=8.0, climb_lead=3.0,
                   kp_yaw=4.5, stick_limit=0.90, slew_per_s=11.0,
                   deadband_frac=0.02, terminal_bbox_frac=0.35,
                   terminal_range_m=2.5, terminal_hold_s=1.0)
c = AcroCommander(conn=f"udpin:127.0.0.1:{PORT}", transport="mavlink",
                  axes=("roll","pitch","yaw","throttle"), law=law, send_hz=50.0,
                  steer_modes=frozenset({"ALT_HOLD"}), commit_chan=6,
                  commit_abort_below=0.05, log_dir=None, verbose=False)
c.connect()
HALF_H = math.tan(math.radians(62.2/2)); HALF_V = math.tan(math.radians(48.8/2))
def px(az_deg, el_deg=0.0):
    return (320.0*math.tan(math.radians(az_deg))/HALF_H,
            -320.0*math.tan(math.radians(el_deg))/HALF_V)

TILT = 10.0
def drive(sec, az=8.0, el_world=None, el=0.0):
    if el_world is not None:
        el = el_world - TILT
    t0 = perf_counter(); cmd = None
    while perf_counter()-t0 < sec:
        ex, ey = px(az, el)
        cmd = c.compute(ex, ey, "LOCK", 640, 640, bbox=(0,0,30,30),
                        capture_t=perf_counter()-0.10)
        c.update(cmd, (int(ex),int(ey)), have_target=True); sleep(0.02)
    return cmd


last = drive(1.0)
ck("nothing driven while disarmed + STABILIZE", not fc.overrides, str(fc.overrides or "{}"))


fc.armed = True; sleep(1.3); drive(1.0)
ck("armed but STABILIZE -> pilot keeps every stick", not fc.overrides, str(fc.overrides or "{}"))


fc.mode = 2; sleep(1.4)
last = drive(2.0)
ck("all FOUR sticks now driven", set(fc.overrides)=={1,2,3,4}, f"ch {sorted(fc.overrides)}")
ck("the drone confirms it applied them", c.override_confirmed is True)


ck("nose leans DOWN = flies forward (never backwards)", last["lean_deg"] < -5,
   f"{last['lean_deg']:.1f} deg")
ck("yaw turns toward a target on the RIGHT", last["yaw_stick"] > 0,
   f"yaw {last['yaw_stick']:+.3f}")
ck("forward lean reaches the drone as real stick travel",
   abs(fc.overrides[2]-1500) > 40, f"ch2={fc.overrides[2]}")


yaw = 0.0; peak = 0.0
for k in range(70):
    yaw += math.radians(14.0)*0.06
    fc.att.update(yaw=yaw, yawspeed=math.radians(14.0)); sleep(0.06)
    ex, ey = px(3.0)
    last = c.compute(ex, ey, "LOCK", 640, 640, bbox=(0,0,30,30), capture_t=perf_counter())
    c.update(last, (int(ex),int(ey)), have_target=True)
    peak = max(peak, last["bank_deg"])
ck("target crossing -> BANKS to cut the corner", peak > 8.0, f"peak bank {peak:+.1f} deg")
ck("bank stays inside its ceiling", peak <= 30.0+1e-9)
ck("bank reaches the drone as real stick travel", abs(fc.overrides[1]-1500) > 40,
   f"ch1={fc.overrides[1]}")
c.update(c.neutral_command(), None, have_target=False); sleep(0.15)
worst = 0.0
for k in range(70):
    fixed = math.radians(6.0)*math.sin(k*0.25)
    fc.att.update(yaw=fixed, yawspeed=math.radians(90.0)); sleep(0.06)
    ex, ey = px(-math.degrees(fixed))
    last = c.compute(ex, ey, "LOCK", 640, 640, bbox=(0,0,30,30), capture_t=perf_counter())
    c.update(last, (int(ex),int(ey)), have_target=True)
    if k > 25: worst = max(worst, abs(last["bank_deg"]))
ck("its OWN turning does NOT fake a bank (no spin-out)", worst < 1.0,
   f"worst {worst:.3f} deg  (27.0 before the fix)")


c.update(c.neutral_command(), None, have_target=False); sleep(0.15)
up = drive(1.2, az=2.0, el_world=+12.0)
ch3_up = fc.overrides.get(3, 0)
ck("target ABOVE -> commands CLIMB", up["climb_mps"] > 0.2, f"{up['climb_mps']:+.2f} m/s")
ck("...and the throttle stick goes ABOVE mid", ch3_up > 1540, f"ch3={ch3_up}")
c.update(c.neutral_command(), None, have_target=False); sleep(0.15)
dn = drive(1.2, az=2.0, el_world=-12.0)
ch3_dn = fc.overrides.get(3, 0)
ck("target BELOW -> commands DESCEND", dn["climb_mps"] < -0.2, f"{dn['climb_mps']:+.2f} m/s")
ck("...and the throttle stick goes BELOW mid", 0 < ch3_dn < 1460, f"ch3={ch3_dn}")
c.update(c.neutral_command(), None, have_target=False); sleep(0.15)
lv = drive(1.2, az=2.0, el_world=0.0)
ck("target LEVEL -> holds altitude (throttle at mid)",
   abs(lv["climb_mps"]) < 0.35 and abs(fc.overrides.get(3,0)-1500) < 130,
   f"{lv['climb_mps']:+.2f} m/s, ch3={fc.overrides.get(3)}")


fc.radio[6] = 1000; sleep(0.5); drive(0.7)
ck("commit switch DOWN -> every stick released (fast abort)", not fc.overrides,
   str(fc.overrides or "{}"))
fc.radio[6] = 2000; sleep(0.5); drive(1.0)
ck("commit switch back UP -> AI drives again", set(fc.overrides)=={1,2,3,4})
fc.mode = 0; sleep(1.5); drive(0.7)
ck("mode switch to STABILIZE -> every stick released", not fc.overrides,
   str(fc.overrides or "{}"))
fc.mode = 2; sleep(1.4); drive(0.8)
ck("back to ALT_HOLD -> re-engages by itself", set(fc.overrides)=={1,2,3,4})
t0 = perf_counter()
while perf_counter()-t0 < 1.2:
    c.update(c.neutral_command(), None, have_target=False); sleep(0.02)
ck("target lost -> every stick released", not fc.overrides, str(fc.overrides or "{}"))
c.close(); sleep(0.4)
ck("Ctrl-C / shutdown -> every stick released", not fc.overrides, str(fc.overrides or "{}"))
fc.shutdown()


if bad:
    pass; [None for b in bad]; sys.exit(1)
