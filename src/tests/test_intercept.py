#!/usr/bin/env python3
"""Offline verification of the FREE-FLIGHT intercept law.

Needs no FC and no camera. Covers the law's geometry, PN, the vertical channel,
range and the terminal commit, latency, limits, and that the HUD renders.
The closed-loop sky-target campaign lives in tests/sim_sky_wind.py.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fc_control import ArduPilotAcro, InterceptLaw, _Rate

FW = FH = 640
HFOV, VFOV = 62.2, 48.8
G = 9.80665
fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def law(**kw):
    kw.setdefault("h_fov_deg", HFOV)
    kw.setdefault("v_fov_deg", VFOV)
    kw.setdefault("slew_per_s", 1e9)      # unless a test is about the slew
    kw.setdefault("deadband_frac", 0.0)
    return InterceptLaw(**kw)


def att(roll=0.0, pitch=0.0, yaw=0.0, yawspeed=0.0, pitchspeed=0.0):
    return dict(roll=roll, pitch=pitch, yaw=yaw, yawspeed=yawspeed,
                pitchspeed=pitchspeed, rollspeed=0.0)


def px_az(deg):
    """Pixel offset for a target `deg` right of the boresight."""
    return (FW / 2.0) * math.tan(math.radians(deg)) / math.tan(math.radians(HFOV / 2))


def px_el(deg):
    """Pixel offset (image y, DOWN positive) for a target `deg` above boresight."""
    return -(FH / 2.0) * math.tan(math.radians(deg)) / math.tan(math.radians(VFOV / 2))


# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 1. roll de-rotation: a bank must not leak into the other axis ===")

L = law()
# At +90 deg of bank (right wing straight down) the body's "up" points to world
# LEFT, so a target straight up in the world must appear on the image LEFT.
ex, ey = L._de_rotate(-100.0, 0.0, math.radians(90.0))
check("roll +90: image-left maps to world-UP",
      abs(ex) < 1e-6 and ey < -1e-6, f"H={ex:+.1f} V={ey:+.1f}")
ex, ey = L._de_rotate(50.0, -30.0, 0.0)
check("zero roll is the identity", abs(ex - 50.0) < 1e-9 and abs(ey + 30.0) < 1e-9)
# Round-trip: rotate a level-frame offset into the body frame by hand, then let
# the law take it back out.
H0, V0 = 80.0, -40.0
for deg in (-40, -12, 0, 7, 25, 60):
    r = math.radians(deg)
    bx = H0 * math.cos(r) + V0 * math.sin(r)
    by = -H0 * math.sin(r) + V0 * math.cos(r)
    gx, gy = L._de_rotate(bx, by, r)
    if abs(gx - H0) > 1e-6 or abs(gy - V0) > 1e-6:
        fails.append(f"de-rotation round trip at {deg} deg")
check("de-rotation inverts the body rotation at -40..+60 deg",
      not any("round trip" in f for f in fails))

# The reason it matters: banked, a purely horizontal miss must produce no
# vertical command at all.
Lb = law(climb_gain=8.0, climb_lead=0.0, cam_tilt_deg=0.0)
r = math.radians(25.0)
bx = 120.0 * math.cos(r)
by = -120.0 * math.sin(r)
c = Lb.compute(bx, by, FW, FH, dt=0.05, att=att(roll=r))
check("banked 25 deg, purely horizontal miss -> ZERO climb",
      abs(c["climb_mps"]) < 1e-6, f"climb={c['climb_mps']:+.4f} m/s")
Lb.neutral()
c = Lb.compute(120.0, 0.0, FW, FH, dt=0.05, att=att(roll=0.0))
base_climb = c["climb_mps"]
check("...and without de-rotation the same geometry WOULD leak (control check)",
      abs(base_climb) < 1e-6, f"level climb={base_climb:+.4f} m/s")


# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 2. the camera-tilt coupling (why aim and approach both lost) ===")

# Nose-down 20 deg, level-mounted camera: a target at the airframe's OWN altitude
# sits 20 deg high in frame. With +/-24.4 deg of lens, that is nearly the edge.
L0 = law(cam_tilt_deg=0.0)
lean = math.radians(-20.0)
# pixel offset for a target 20 deg ABOVE the boresight
ey_px = -(FH / 2.0) * math.tan(math.radians(20.0)) / math.tan(math.radians(VFOV / 2))
c = L0.compute(0.0, ey_px, FW, FH, dt=0.05, att=att(pitch=lean))
check("level camera, nose-down 20: a LEVEL target resolves to ~0 world elevation",
      abs(c["el_deg"]) < 0.5, f"el_world={c['el_deg']:+.2f} deg "
      f"(el_cam={c['el_cam_deg']:+.1f})")
frac = abs(ey_px) / (FH / 2.0)
check("...and it sits at 82% of the way to the top edge of the frame",
      frac > 0.75, f"{frac * 100:.0f}% of the half-frame")
check("...so past ~24 deg of lean a level target is OUT of frame",
      math.tan(math.radians(24.4)) <= math.tan(math.radians(VFOV / 2)) + 1e-9)

# With the mount tilted up by the cruise lean, the same target sits mid-frame.
L15 = law(cam_tilt_deg=20.0)
c = L15.compute(0.0, 0.0, FW, FH, dt=0.05, att=att(pitch=lean))
check("camera tilted +20 up at 20 deg nose-down: boresight is LEVEL",
      abs(c["el_deg"]) < 1e-6, f"el_world={c['el_deg']:+.4f} deg at centre-frame")

L15.neutral()
c = L15.compute(0.0, 0.0, FW, FH, dt=0.05, att=att(pitch=0.0))
check("...and level airframe with a +20 tilt looks 20 deg UP",
      abs(c["el_deg"] - 20.0) < 1e-6, f"el_world={c['el_deg']:+.2f} deg")


# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 3. proportional navigation on roll ===")

def crossing(rate_dps, n=400, hz=20.0, gain=25.0, az_hold_deg=2.0, **kw):
    """A CROSSING target with the seeker tracking it.

    The seeker holds the target at a near-constant small azimuth while the nose
    follows it round, so the world bearing lambda = yaw + az rotates at the yaw
    rate. Everything stays well inside the frame, which matters: the pixel offset
    is clamped at the frame edge, so a test that sweeps the target out of view
    measures the clamp instead of the law.
    """
    L = law(roll_pn_gain=gain, **kw)
    dt = 1.0 / hz
    yaw = 0.0
    out = None
    for _ in range(n):
        yaw += math.radians(rate_dps) * dt
        out = L.compute(px_az(az_hold_deg), 0.0, FW, FH, dt=dt, att=att(yaw=yaw))
    return out


def own_yaw_only(rate_dps, n=200, hz=20.0, gain=25.0, amp_deg=10.0):
    """A target on a FIXED world bearing while the airframe yaws under it.

    az moves exactly opposite to yaw, so lambda is constant and PN must command
    nothing. This is the case the first formulation got wrong, and getting it
    wrong is what produced the full-bank limit cycle in an earlier closed-loop sim.
    """
    L = law(roll_pn_gain=gain)
    dt = 1.0 / hz
    t = 0.0
    worst = 0.0
    for _ in range(n):
        t += dt
        # Oscillate the heading rather than integrate it, so az stays in frame.
        yaw = math.radians(amp_deg) * math.sin(math.radians(rate_dps) * t)
        azd = -math.degrees(yaw)                 # perfect tracking: lambda == 0
        c = L.compute(px_az(azd), 0.0, FW, FH, dt=dt, att=att(yaw=yaw))
        worst = max(worst, abs(c["bank_deg"]))
    return worst


c = crossing(+10.0)
check("world LOS rotating RIGHT -> bank RIGHT", c["bank_deg"] > 1.0,
      f"bank={c['bank_deg']:+.1f} deg at 10 deg/s of LOS rotation")
c = crossing(-10.0)
check("world LOS rotating LEFT  -> bank LEFT", c["bank_deg"] < -1.0,
      f"bank={c['bank_deg']:+.1f} deg")

# THE defining property of PN: a constant bearing commands nothing. Pure pursuit
# would still be turning here, which is exactly why it trails a crossing target.
c = crossing(0.0, az_hold_deg=25.0)
check("target 25 deg off boresight but LOS rate ZERO -> no bank",
      abs(c["bank_deg"]) < 1e-6, f"bank={c['bank_deg']:+.6f} deg")

# The regression the closed-loop sim caught.
worst = own_yaw_only(120.0)
check("the SEEKER's own yaw cannot drive PN (this was the limit cycle)",
      worst < 1e-6, f"largest bank {worst:.6f} deg while the nose swings +/-10 deg")

# a_lat = gain * los_rate, and bank = atan(a_lat/g).
c = crossing(+10.0, n=600)
want = math.degrees(math.atan2(25.0 * math.radians(10.0), G))
check("bank = atan(gain * LOS_rate / g) once the filter has settled",
      abs(c["bank_deg"] - want) < 0.05, f"{c['bank_deg']:+.3f} vs {want:+.3f} deg")
check("...and that value is inside the bank ceiling, so it is not the clamp",
      want < 30.0, f"{want:.1f} deg < 30 deg")
# A coordinated turn needs a_lat = V*omega, so gain == airspeed is exactly
# coordinated. That is the sanity check on the gain's units.
check("gain is dimensionally an own-speed (coordinated turn at gain==V)",
      abs(25.0 * 0.2 - 5.0) < 1e-9, "25 m/s * 0.2 rad/s = 5.0 m/s^2")

c = crossing(+300.0, n=300)
check("bank clamped to max_bank_deg", abs(c["bank_deg"]) <= 30.0 + 1e-9,
      f"bank={c['bank_deg']:+.2f} deg at 300 deg/s")

# Heading unwrapping: crossing the +/-pi seam must not spike the roll command.
# Settle the filter FIRST, well away from the seam, so what is measured is the
# seam and not the filter's start-up transient.
Lw = law(roll_pn_gain=25.0)
yaw = 0.0
for _ in range(400):
    yaw += 0.01
    Lw.compute(0.0, 0.0, FW, FH, dt=0.05,
               att=att(yaw=math.atan2(math.sin(yaw), math.cos(yaw))))
settled = None
spike = 0.0
for _ in range(60):
    yaw += 0.01                      # walks straight through +pi
    c = Lw.compute(0.0, 0.0, FW, FH, dt=0.05,
                   att=att(yaw=math.atan2(math.sin(yaw), math.cos(yaw))))
    if settled is not None:
        spike = max(spike, abs(c["bank_deg"] - settled))
    settled = c["bank_deg"]
check("crossing the +/-180 deg heading seam does not spike the bank command",
      spike < 0.5, f"largest step {spike:.5f} deg through the wrap")

# No attitude, no PN: the world LOS bearing cannot be formed from pixels alone,
# and the body-frame rate is not a substitute.
Lna = law(roll_pn_gain=25.0, climb_gain=8.0)
c = Lna.compute(+150.0, -80.0, FW, FH, dt=0.05, att=None)
check("no ATTITUDE -> roll EXACTLY zero", c["roll_stick"] == 0.0)
check("no ATTITUDE -> throttle EXACTLY zero", c["throttle_stick"] == 0.0)
check("no ATTITUDE -> yaw still steers (seeker survives)", c["yaw_stick"] != 0.0,
      f"yaw={c['yaw_stick']:+.3f}")
check("no ATTITUDE -> pitch still drives forward", c["pitch_stick"] < 0.0,
      f"pitch={c['pitch_stick']:+.3f}")
check("no ATTITUDE is reported, not hidden", c["have_att"] is False)


# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 4. the vertical channel (the 'higher or lower than it' fix) ===")

Lv = law(climb_gain=8.0, climb_lead=0.0, cam_tilt_deg=0.0)
c = Lv.compute(0.0, -100.0, FW, FH, dt=0.05, att=att(pitch=0.0))
check("target ABOVE -> CLIMB", c["climb_mps"] > 0, f"{c['climb_mps']:+.2f} m/s")
Lv.neutral()
c = Lv.compute(0.0, +100.0, FW, FH, dt=0.05, att=att(pitch=0.0))
check("target BELOW -> DESCEND", c["climb_mps"] < 0, f"{c['climb_mps']:+.2f} m/s")

# The point of resolving into the world frame: the climb command must follow the
# TARGET's real elevation, not where it happens to land on the sensor.
Lv2 = law(climb_gain=8.0, climb_lead=0.0, cam_tilt_deg=20.0)
ups = []
for lean_deg in (0.0, -10.0, -20.0, -30.0):
    Lv2.neutral()
    # A target 10 deg above the airframe, whatever the airframe is doing.
    el_cam = math.radians(10.0) - (math.radians(lean_deg) + math.radians(20.0))
    ey_px = -(FH / 2.0) * math.tan(el_cam) / math.tan(math.radians(VFOV / 2))
    c = Lv2.compute(0.0, ey_px, FW, FH, dt=0.05,
                    att=att(pitch=math.radians(lean_deg)))
    ups.append(c["climb_mps"])
check("a target 10 deg above commands the SAME climb at every lean angle",
      max(ups) - min(ups) < 1e-6 and all(u > 0 for u in ups),
      f"climb {['%+.2f' % u for u in ups]} m/s at lean 0/-10/-20/-30")

Lv3 = law(climb_gain=0.0, climb_lead=3.0, cam_tilt_deg=0.0)
el = 0.0
for _ in range(80):                   # target rising through the frame
    el += 0.2 * 0.05
    c = Lv3.compute(0.0, px_el(math.degrees(el)), FW, FH, dt=0.05, att=att())
check("a RISING vertical LOS leads the target upward (climb_gain is 0 here)",
      c["climb_mps"] > 0, f"{c['climb_mps']:+.2f} m/s at 0.2 rad/s of LOS rate")
Lv3.neutral()
c = Lv3.compute(0.0, px_el(20.0), FW, FH, dt=0.05, att=att())
check("...and a STATIC elevation error commands nothing through the lead term",
      abs(c["climb_mps"]) < 1e-9, f"{c['climb_mps']:+.4f} m/s")

Lv4 = law(climb_gain=1e6, climb_lead=0.0, max_climb_mps=2.5,
          stick_limit=1.0, cam_tilt_deg=0.0)
c = Lv4.compute(0.0, -100.0, FW, FH, dt=0.05, att=att())
check("throttle stick saturates at 1.0, not beyond",
      abs(c["throttle_stick"]) <= 1.0 + 1e-9, f"{c['throttle_stick']:+.4f}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 5. pitch can never fly the airframe backwards ===")

Lf = law(max_lean_deg=20.0)
worst = -1.0
for ex_deg in range(-31, 32, 2):
    for ey_deg in range(-24, 25, 2):
        Lf.neutral()
        exp = (FW / 2.0) * math.tan(math.radians(ex_deg)) / math.tan(math.radians(HFOV / 2))
        eyp = (FH / 2.0) * math.tan(math.radians(ey_deg)) / math.tan(math.radians(VFOV / 2))
        c = Lf.compute(exp, eyp, FW, FH, dt=0.05, att=att(pitch=math.radians(-15)))
        worst = max(worst, c["lean_deg"])
check("commanded lean is NEVER positive (nose-up) anywhere in the frame",
      worst <= 0.0 + 1e-12, f"most nose-up commanded = {worst:+.6f} deg")

Lf.neutral()
c = Lf.compute(0.0, 0.0, FW, FH, dt=0.05, att=att(), commit=1.0)
check("full commit -> full lean", abs(c["lean_deg"] + 20.0) < 1e-6,
      f"{c['lean_deg']:+.2f} deg")
Lf.neutral()
c = Lf.compute(0.0, 0.0, FW, FH, dt=0.05, att=att(), commit=0.0)
check("zero commit -> zero lean (the pilot's abort)", abs(c["lean_deg"]) < 1e-9)
Lf.neutral()
c = Lf.compute(0.0, 0.0, FW, FH, dt=0.05, att=att(), commit=0.5)
check("half commit -> half lean", abs(c["lean_deg"] + 10.0) < 1e-6,
      f"{c['lean_deg']:+.2f} deg")

# Turn first, accelerate as the bearing closes.
Lf.neutral()
far = (FW / 2.0) * math.tan(math.radians(31.0)) / math.tan(math.radians(HFOV / 2))
c_far = Lf.compute(far, 0.0, FW, FH, dt=0.05, att=att(), commit=1.0)
Lf.neutral()
c_on = Lf.compute(0.0, 0.0, FW, FH, dt=0.05, att=att(), commit=1.0)
check("lean is reduced while the nose is off the target",
      abs(c_far["lean_deg"]) < abs(c_on["lean_deg"]),
      f"{c_far['lean_deg']:+.1f} deg at 31 deg off vs {c_on['lean_deg']:+.1f} on target")


# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 6. range, closing speed, and the terminal commit ===")

Lr = law(target_width_m=0.20)
f_px = (FW / 2.0) / math.tan(math.radians(HFOV / 2))
for want_m in (30.0, 10.0, 3.0):
    bw = f_px * 0.20 / want_m
    got = Lr.range_m(bw, FW)
    if abs(got - want_m) > 1e-6:
        fails.append(f"range at {want_m} m")
check("range from bbox width inverts cleanly at 30/10/3 m",
      not any("range at" in f for f in fails))
check("range is None without a bbox", Lr.range_m(None, FW) is None)

Lt = law(terminal_bbox_frac=0.35, terminal_range_m=0.0, terminal_hold_s=1.0)
c = Lt.compute(0.0, 0.0, FW, FH, dt=0.05, bbox_w=40, bbox_h=100, att=att())
check("bbox below the terminal fraction -> normal guidance", not c["terminal"])
c = Lt.compute(+200.0, 0.0, FW, FH, dt=0.05, bbox_w=90, bbox_h=int(FH * 0.4),
               att=att())
check("bbox at 40% of frame -> TERMINAL", c["terminal"])
frozen = (c["yaw_stick"], c["roll_stick"], c["pitch_stick"], c["throttle_stick"])
c2 = Lt.compute(-300.0, +200.0, FW, FH, dt=0.05, bbox_w=20, bbox_h=20, att=att())
check("terminal LATCHES: a lost/large detection cannot un-commit it", c2["terminal"])
check("...and the sticks are frozen at their last pre-terminal value",
      (c2["yaw_stick"], c2["roll_stick"], c2["pitch_stick"],
       c2["throttle_stick"]) == frozen,
      f"yaw {frozen[0]:+.3f} roll {frozen[1]:+.3f} pitch {frozen[2]:+.3f}")
# Time-limited, so a FALSE trigger cannot fly on holding a stale lean for ever.
c3 = None
for _ in range(30):
    c3 = Lt.compute(0.0, 0.0, FW, FH, dt=0.05, bbox_w=20, bbox_h=20, att=att())
check("terminal hold EXPIRES — the latch clears instead of holding for ever",
      not c3["terminal"], f"after 1.5 s: terminal={c3['terminal']}")
# ...and once it has cleared, guidance is LIVE again rather than frozen: a fresh
# off-centre target must move the yaw command.
c4 = Lt.compute(px_az(20.0), 0.0, FW, FH, dt=0.05, bbox_w=20, bbox_h=20, att=att())
check("...and guidance resumes live, not stuck at the frozen sticks",
      c4["yaw_stick"] > 0.0 and c4["yaw_stick"] != frozen[0],
      f"yaw {c4['yaw_stick']:+.3f} responds again (frozen was {frozen[0]:+.3f})")

Lc = law(terminal_bbox_frac=1.0, terminal_range_m=2.5)
c = Lc.compute(0.0, 0.0, FW, FH, dt=0.05, bbox_w=f_px * 0.30 / 2.0, bbox_h=10,
               att=att())
check("range under TERMINAL_RANGE_M also triggers", c["terminal"],
      f"range={c['range_m']:.2f} m")


# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 7. latency measurement and lead ===")

Ll = law(latency_s=0.20, latency_max_s=0.35, kp_yaw=4.5, deadband_frac=0.0)
c = Ll.compute(0.0, 0.0, FW, FH, dt=0.05, att=att())
check("a fixed latency is reported as configured", abs(c["latency_s"] - 0.20) < 1e-9)
Lcap = law(latency_s=1.0, latency_max_s=0.35)
c = Lcap.compute(0.0, 0.0, FW, FH, dt=0.05, att=att())
check("latency is capped (never extrapolate a jittering track far)",
      abs(c["latency_s"] - 0.35) < 1e-9, f"{c['latency_s']:.3f} s")

Lm = law(latency_s=0.0)
for _ in range(60):
    c = Lm.compute(0.0, 0.0, FW, FH, dt=0.05, att=att(), capture_age=0.11)
check("a measured capture age converges to the real latency",
      abs(c["latency_s"] - 0.11) < 0.01, f"{c['latency_s'] * 1000:.0f} ms")

# The lead must actually move the aim point: a target sweeping right should be
# aimed at further right than where it was seen.
def yaw_for(latency):
    Lx = law(latency_s=latency, kp_yaw=4.5, deadband_frac=0.0)
    out = 0.0
    az = 0.0
    for _ in range(40):
        az += 0.25 * 0.05                     # LOS sweeping right at 0.25 rad/s
        px = (FW / 2.0) * math.tan(az) / math.tan(math.radians(HFOV / 2))
        out = Lx.compute(px, 0.0, FW, FH, dt=0.05, att=att())["yaw_stick"]
    return out


check("latency lead aims AHEAD of a sweeping target",
      yaw_for(0.20) > yaw_for(0.0) > 0,
      f"yaw {yaw_for(0.0):+.4f} at 0 ms -> {yaw_for(0.20):+.4f} at 200 ms")


# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 8. limits, slew, polarity, and the throttle PWM map ===")

Ls = law(slew_per_s=2.0, max_lean_deg=30.0, stick_limit=1.0)
Ls.neutral()
c = Ls.compute(0.0, 0.0, FW, FH, dt=0.1, att=att(), commit=1.0)
check("slew limits the first step", abs(c["pitch_stick"] + 0.2) < 1e-9,
      f"pitch={c['pitch_stick']:+.3f} after 0.1 s at 2.0/s (unlimited: -1.0)")


def drive(L, axis):
    """Produce a non-zero command on one channel, however that channel is driven.

    roll needs the WORLD LOS bearing to rotate, which takes a run of frames with an
    advancing heading — a single call carrying a body rate cannot do it, and that
    is exactly the property that keeps the seeker from driving PN.
    """
    if axis == "roll":
        yaw, out = 0.0, None
        for _ in range(200):
            yaw += math.radians(10.0) * 0.05
            out = L.compute(px_az(2.0), 0.0, FW, FH, dt=0.05, att=att(yaw=yaw))
        return out
    if axis == "yaw":
        return L.compute(px_az(15.0), 0.0, FW, FH, dt=0.05, att=att())
    if axis == "throttle":
        return L.compute(0.0, px_el(15.0), FW, FH, dt=0.05, att=att())
    return L.compute(0.0, 0.0, FW, FH, dt=0.05, att=att(), commit=1.0)


for axis, kw, field in (("yaw", dict(kp_yaw=4.5), "yaw_stick"),
                        ("roll", dict(roll_pn_gain=25.0), "roll_stick"),
                        ("pitch", dict(max_lean_deg=20.0), "pitch_stick"),
                        ("throttle", dict(climb_gain=8.0), "throttle_stick")):
    ca = drive(law(**kw), axis)
    cb = drive(law(**dict(kw, **{f"invert_{axis}": True})), axis)
    if not (abs(ca[field] + cb[field]) < 1e-12 and ca[field] != 0.0):
        fails.append(f"{axis} reversal flag")
    # The reported PHYSICAL value must NOT flip: an inversion flag compensates a
    # miswired axis, so the physical intent is unchanged and it is the intent the
    # HUD, the console and the CSV have to show.
    phys = {"yaw": "yaw_rate", "roll": "bank_deg", "pitch": "lean_deg",
            "throttle": "climb_mps"}[axis]
    if abs(ca[phys] - cb[phys]) > 1e-9:
        fails.append(f"{axis} reported physical value flipped")
check("every reversal flag flips exactly its own channel",
      not any("reversal flag" in f for f in fails))
check("...while the REPORTED physical command stays the physical intent",
      not any("reported physical" in f for f in fails))

Lz = law(deadband_frac=0.02, kp_yaw=4.5, climb_gain=8.0, cam_tilt_deg=0.0)
tiny = (FW / 2.0) * 0.005
c = Lz.compute(tiny, tiny, FW, FH, dt=0.05, att=att())
check("inside the angular deadband -> exactly zero yaw and climb",
      c["yaw_stick"] == 0.0 and c["climb_mps"] == 0.0)

# The throttle map cannot pivot on trim (trim sits at the bottom of a throttle
# channel) and has to escape THR_DZ, inside which ArduPilot just holds altitude.
be = ArduPilotAcro.__new__(ArduPilotAcro)
be.rc = {"throttle": dict(chan=3, min=1000, max=2000, trim=1002, reversed=False)}
be.thr_deadzone = 100
check("zero climb -> EXACTLY mid PWM (hold altitude)",
      be._throttle_pwm(0.0) == 1500, f"{be._throttle_pwm(0.0)}")
check("the smallest climb command escapes THR_DZ",
      be._throttle_pwm(1e-6) > 1500 + 100 - 1, f"{be._throttle_pwm(1e-6)}")
check("full climb -> max PWM", be._throttle_pwm(1.0) == 2000)
check("full descend -> min PWM", be._throttle_pwm(-1.0) == 1000)
check("the map is monotonic and clamped",
      be._throttle_pwm(0.5) < be._throttle_pwm(0.9) < be._throttle_pwm(2.0) == 2000)
check("trim at the bottom of the travel is IGNORED, not used as centre",
      be._throttle_pwm(0.0) != 1002)
ber = ArduPilotAcro.__new__(ArduPilotAcro)
ber.rc = {"throttle": dict(chan=3, min=1000, max=2000, trim=1002, reversed=True)}
ber.thr_deadzone = 100
check("a reversed throttle channel is honoured",
      ber._throttle_pwm(1.0) == 1000 and ber._throttle_pwm(0.0) == 1500)

r = _Rate(tau=0.1)
for _ in range(200):
    r.update(r.value * 0 + 0.0, 0.01)
check("the LOS-rate filter sits at zero for a stationary input",
      abs(r.value) < 1e-9)
r2 = _Rate(tau=0.05)
x = 0.0
for _ in range(400):
    x += 1.0 * 0.01
    r2.update(x, 0.01)
check("the LOS-rate filter converges on a constant slope",
      abs(r2.value - 1.0) < 0.02, f"{r2.value:.4f} vs 1.0")


# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 9. the HUD actually renders (it had never been executed) ===")
# The flight panel gained two gauge rows, a status line and a taller panel for the
# intercept law. None of that code had ever run: a KeyError in there would take
# the tracker down mid-flight. It releases the sticks on the way out, so it is
# safe — but the run is over. So render every state once and see.
try:
    import numpy as np
    import cv2                                                   # noqa: F401
    from tracker import DroneTracker
except Exception as e:                                           # pragma: no cover
    check(f"HUD smoke test skipped (no cv2/ultralytics: {e.__class__.__name__})",
          True, "run this suite in yolo-env to cover the HUD")
else:
    class _FakeLaw:
        stick_limit = 0.90
        max_bank = math.radians(30.0)
        max_lean = math.radians(20.0)
        max_climb_mps = 2.5

    class _FakeAp:
        """The surface the HUD reads. Deliberately hand-written rather than a real
        commander, so the test needs no FC and no threads."""
        def __init__(self, intercept, linked=True, armed=True, conf=True,
                     rate_ok=True):
            self.is_intercept = intercept
            self.law = _FakeLaw()
            self.linked = linked
            self.armed = armed
            self.override_confirmed = conf
            self.rate_mode_ok = rate_ok
            self.mode_name = "ALT_HOLD" if intercept else "ACRO"
            self.send_hz = 50.0
            self.max_yaw_rate = math.radians(202.0)
            self.max_pitch_rate = math.radians(360.0)

    hud = DroneTracker.__new__(DroneTracker)
    hud.skip_frames = 0
    frame = np.zeros((640, 640, 3), np.uint8)

    icmd = InterceptLaw(h_fov_deg=HFOV, v_fov_deg=VFOV, cam_tilt_deg=15.0).compute(
        +120, -60, FW, FH, dt=0.05, bbox_w=30, bbox_h=30, att=att(yaw=0.2))
    variants = {
        "intercept, driving": dict(cmd=icmd, sent=True),
        "intercept, terminal": dict(cmd=dict(icmd, terminal=True), sent=True),
        "intercept, no attitude": dict(cmd=dict(icmd, have_att=False), sent=True),
        "intercept, range unknown": dict(cmd=dict(icmd, range_m=None), sent=True),
        "intercept, neutral/idle": dict(cmd=InterceptLaw().neutral(), sent=False),
    }
    rendered = 0
    for name, v in variants.items():
        for linked in (True, False):
            for armed in (True, False):
                for conf in (True, False, None):
                    for rate_ok in (True, False):
                        for pos in ((320, 300), None):
                            ap = _FakeAp(True, linked, armed, conf, rate_ok)
                            hud.ap = ap
                            try:
                                hud._draw_flightcmd(frame.copy(), v["cmd"], pos,
                                                    320, 320, v["sent"],
                                                    (0, 255, 0))
                                rendered += 1
                            except Exception as e:
                                fails.append(f"HUD raised on {name}: "
                                             f"{e.__class__.__name__}: {e}")
    check(f"intercept HUD renders every state without raising ({rendered} "
          f"permutations)", not any("HUD raised" in f for f in fails))

    # The servo law's panel must keep working untouched.
    scmd = __import__("fc_control").ControlLaw().compute(+120, -60, FW, FH, dt=0.05)
    ok_servo = True
    for linked in (True, False):
        for sent in (True, False):
            ap = _FakeAp(False, linked, True, True, True)
            hud.ap = ap
            try:
                hud._draw_flightcmd(frame.copy(), scmd, (320, 300), 320, 320, sent,
                                    (0, 255, 0))
            except Exception as e:
                ok_servo = False
                fails.append(f"servo HUD raised: {e.__class__.__name__}: {e}")
    check("the gimbal HUD panel still renders too", ok_servo)

    # And the panel must fit inside the frame it is drawn on.
    check("the taller intercept panel fits in a 640x640 frame", 170 + 20 < 640)


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("ALL INTERCEPT CHECKS PASSED")
