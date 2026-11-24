"""SEEK -> camera lock -> ALT_HOLD attack -> hover -> RTL, through the real Mission
and SeekerMissions, with the pilot's throttle stick DOWN (as SEEK requires at ARM).

The simulated FC's ALT_HOLD follows the pilot's throttle stick whenever no
override comes from the Jetson (stick down: descend at PILOT_SPEED_DN, 2.5 m/s),
so any moment the software leaves the FC in ALT_HOLD on the pilot's stick shows
up as a descent.  The attack's overrides drive climb and lean like ArduCopter's
ALT_HOLD (PILOT_SPEED_UP 25 m/s, PILOT_ACCEL_Z 5 m/s^2, lean with a lag).

    python src/tests/test_seek_transition.py
"""
import ast
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import mission as MI
import seeker_missions as S
from asterix_cue import rfd_line

FAILED = []


def check(label, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(label)


# The simulated copter of test_seeker_missions.py, without running its tests.
_src = open(os.path.join(HERE, "test_seeker_missions.py")).read()
_tree = ast.parse(_src)
_tree.body = [n for n in _tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
              or (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets))]
H = {"__file__": os.path.join(HERE, "test_seeker_missions.py"), "__name__": "seeker_harness"}
exec(compile(_tree, "test_seeker_missions.py", "exec"), H)
HOME = H["HOME"]
G = 9.81


class Copter(H["FakeCopter"]):
    """ALT_HOLD that obeys the Jetson's stick overrides, or the pilot's stick without them."""

    def __init__(self, clock, pilot_thr):
        super().__init__(clock)
        self.pilot_thr = pilot_thr          # -1 = stick down, 0 = middle
        self.override = None
        self.roll = self.pitch = 0.0        # degrees
        self.on_pilot_stick = 0.0           # seconds spent in ALT_HOLD on the pilot's stick, in the air
        self.min_vz_alt_hold = 0.0
        self.modes = []

    def request_mode(self, name, allow_disarmed=False):
        ok = super().request_mode(name, allow_disarmed)
        if ok:
            self.modes.append((round(self.k.t, 2), name))
        return ok

    def step(self, dt):
        if not (self.armed and self.mode_name == "ALT_HOLD"):
            self.roll = self.pitch = 0.0
            return super().step(dt)
        o = self.override
        if o:
            thr, roll_cmd, pitch_cmd = o["throttle"], o["roll"], o["pitch"]
        else:
            thr, roll_cmd, pitch_cmd = self.pilot_thr, 0.0, 0.0
            if self.alt > 0.5:
                self.on_pilot_stick += dt
        want_vz = thr * 25.0 if thr > 0 else thr * 2.5
        self.vz += max(-5.0*dt, min(5.0*dt, want_vz - self.vz))
        self.roll += (roll_cmd - self.roll) * min(1.0, dt/0.15)
        self.pitch += (pitch_cmd - self.pitch) * min(1.0, dt/0.15)
        cy, sy = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        af, ar = G*math.tan(math.radians(-self.pitch)), G*math.tan(math.radians(self.roll))
        self.vn += (af*cy - ar*sy - 0.3*self.vn) * dt
        self.ve += (af*sy + ar*cy - 0.3*self.ve) * dt
        self.n += self.vn*dt
        self.e += self.ve*dt
        self.alt = max(0.0, self.alt + self.vz*dt)
        if self.alt > 0.5:
            self.min_vz_alt_hold = min(self.min_vz_alt_hold, self.vz)


class Mapper(H["FakeMapper"]):
    """What the Mission calls on the controller; sticks as physical values for the simulation."""

    def idle_pwm(self, yaw_dps, climb_mps):
        return dict(roll=0.0, pitch=0.0, yaw=yaw_dps, throttle=climb_mps/25.0)

    def lean_pwm(self, roll_deg, pitch_deg):
        return dict(roll=roll_deg, pitch=pitch_deg)


def bearing_of(fc, tgt):
    """World NED unit vector from the drone to the target, and the angle off the camera axis."""
    dn = (tgt[0] - fc.n)
    de = (tgt[1] - fc.e)
    du = (tgt[2] - fc.alt)
    d = math.sqrt(dn*dn + de*de + du*du)
    cr, sr = math.cos(math.radians(fc.roll)), math.sin(math.radians(fc.roll))
    cp, sp = math.cos(math.radians(fc.pitch)), math.sin(math.radians(fc.pitch))
    cy, sy = math.cos(math.radians(fc.yaw)), math.sin(math.radians(fc.yaw))
    ax_f, ax_r, ax_u = -sp*cr, sr, cp*cr                 # the camera (body up) axis, yaw frame
    ax = (ax_f*cy - ax_r*sy, ax_f*sy + ax_r*cy, ax_u)
    off = math.degrees(math.acos(max(-1.0, min(1.0, (ax[0]*dn + ax[1]*de + ax[2]*du)/max(d, 1e-9)))))
    return (dn/d, de/d, -du/d), off, d


def run(pilot_thr=-1.0, tgt_vel=(0.0, 0.0), abort_after=None, pilot_stab_after=None, until=400.0):
    k = H["Clock"]()
    fc = Copter(k, pilot_thr)
    mp = Mapper(fc, k)
    tmp = tempfile.mkdtemp(prefix="seek-transition-")
    m = MI.Mission(mp, log=lambda *_a, **_k: None, seeker_path=os.path.join(tmp, "seeker.json"))
    m.fence.load(dict(points=H["square"](3000)))
    m.camera_mount_deg = 90.0
    sm = m.seeker
    sm.log_dir = tmp
    sm._cue_started = True                       # no UDP / radio: the cue goes in directly
    st = dict(attack_t=None, rtl_t=None, closest=1e9, hit=False, aborted=False, stab_t=None, modes_after_stab=[])
    tgt = [120.0, 40.0, 40.0]                    # north, east (m from home), height above home
    hist = []
    last_tx = -1e9

    def ctx_for():
        b, off, d = bearing_of(fc, tgt)
        seen = off < 31.0 and d < 80.0
        hist.append((k.t, b, seen))
        ctx = dict(att=dict(roll=math.radians(fc.roll), pitch=math.radians(fc.pitch), yaw=math.radians(fc.yaw)),
                   position=fc.pos(), height=fc.alt, airborne=fc.alt > 0.5)
        old = [h for h in hist if h[0] <= k.t - 0.05]
        if old and old[-1][2]:
            t0, b0, _ = old[-1]
            prev = [h for h in hist if h[0] <= t0 - 0.05]
            rate = tuple((x - y)/max(t0 - prev[-1][0], 1e-3) for x, y in zip(b0, prev[-1][1])) if prev else (0, 0, 0)
            ctx.update(det_t=k.t, det_conf=0.8, ai_cmd=dict(bearing=b0, bearing_rate=rate, capture_t=t0))
        return ctx, d

    def tick(ctx):
        nonlocal last_tx
        if k.t - last_tx >= 0.5:
            last_tx = k.t
            lat, lon = S._offset(HOME, tgt[0], tgt[1])
            sm.cue.ingest_rfd(rfd_line(lat, lon, tgt[2], "REL", "1"), now=k.t)
        out = m.cycle(k.t, ctx)
        # fc_control's steer_attack_pwm: overrides only armed in ALT_HOLD / GUIDED; else released.
        fc.override = out if out and fc.armed and fc.mode_name in ("ALT_HOLD", "GUIDED") else None

    ctx, _ = ctx_for()
    tick(ctx)
    k.t += 0.6
    tick(ctx)
    why = sm.command("start:SEEK")
    assert not why, why
    t_start = k.t
    while k.t < t_start + until:
        dt = 0.02
        k.t += dt
        fc.step(dt)
        mp.frame_t = k.t
        if st["attack_t"] is not None:           # the target moves once the attack starts
            tgt[0] += tgt_vel[0]*dt
            tgt[1] += tgt_vel[1]*dt
        ctx, d = ctx_for()
        r = sm.run
        if r and r.get("state") == "ATTACK" and st["attack_t"] is None:
            st["attack_t"] = k.t
        if st["attack_t"] is not None and not st["hit"]:
            st["closest"] = min(st["closest"], d)
            st["hit"] = st["hit"] or d < 1.5
        if fc.mode_name == "RTL" and st["rtl_t"] is None:
            st["rtl_t"] = k.t
        if abort_after is not None and st["attack_t"] is not None and not st["aborted"] \
                and k.t - st["attack_t"] > abort_after:
            st["aborted"] = True
            sm.command("abort")
        if pilot_stab_after is not None and st["attack_t"] is not None and st["stab_t"] is None \
                and k.t - st["attack_t"] > pilot_stab_after:
            st["stab_t"] = k.t
            fc.switch, fc.mode_name = 2, "STABILIZE"     # the pilot flips the switch: the FC follows it
            fc.modes.append((round(k.t, 2), "STABILIZE (pilot)"))
        tick(ctx)
        if not sm.active and st["attack_t"] is not None:
            break
        if st["stab_t"] is not None and k.t - st["stab_t"] > 20:
            break
    if st["stab_t"] is not None:
        st["modes_after_stab"] = [mo for t, mo in fc.modes if t > st["stab_t"] and "pilot" not in mo]
    return fc, sm, m, st


# ---- A: throttle DOWN, standing target ----------------------------------------
fc, sm, m, st = run(pilot_thr=-1.0)
check("A. camera lock -> ATTACK in ALT_HOLD (never ACRO)", st["attack_t"] is not None
      and "ACRO" not in [mo for _, mo in fc.modes], fc.modes)
check("A. reaches the target (10 m above)", st["hit"], f"closest {st['closest']:.2f} m")
check("A. never left in ALT_HOLD on the pilot's stick (down)", fc.on_pilot_stick == 0.0,
      f"{fc.on_pilot_stick:.2f} s")
check("A. no descent at any point in ALT_HOLD", fc.min_vz_alt_hold > -0.3, f"min vz {fc.min_vz_alt_hold:.2f} m/s")
check("A. then RTL and lands at home, disarmed", st["rtl_t"] is not None and not fc.armed
      and math.hypot(fc.n, fc.e) < 2.0, f"armed {fc.armed}, {math.hypot(fc.n, fc.e):.1f} m from home")

# ---- B: throttle DOWN, ABORT one second into the attack -------------------------
fc, sm, m, st = run(pilot_thr=-1.0, abort_after=1.0)
check("B. ABORT during the attack -> RTL", st["aborted"] and st["rtl_t"] is not None, fc.modes)
check("B. never left in ALT_HOLD on the pilot's stick (down)", fc.on_pilot_stick == 0.0,
      f"{fc.on_pilot_stick:.2f} s")
check("B. no descent at any point in ALT_HOLD", fc.min_vz_alt_hold > -0.3, f"min vz {fc.min_vz_alt_hold:.2f} m/s")
check("B. lands at home, disarmed", not fc.armed and math.hypot(fc.n, fc.e) < 2.0)

# ---- C: the pilot flips the switch to STABILIZE during the attack ------------------
fc, sm, m, st = run(pilot_thr=0.0, pilot_stab_after=0.8)
check("C. pilot to STABILIZE: SEEK lets go (PILOT), sticks released", sm.run is not None
      and sm.run.get("hands_off") and not sm.attacking and not m.seek_attack.active,
      f"state {sm.run and sm.run.get('state')}")
check("C. and never asks for any mode behind the pilot's back (no RTL)", st["modes_after_stab"] == [],
      st["modes_after_stab"])

# ---- D: throttle in the MIDDLE, moving target -------------------------------------
fc, sm, m, st = run(pilot_thr=0.0, tgt_vel=(0.0, 5.0))
check("D. throttle middle, target moving 5 m/s during the attack: reaches it", st["hit"], f"closest {st['closest']:.2f} m")
check("D. never on the pilot's stick, then RTL home", fc.on_pilot_stick == 0.0 and not fc.armed
      and math.hypot(fc.n, fc.e) < 2.0, f"{fc.on_pilot_stick:.2f} s on the stick")

# ---- E: throttle DOWN, moving target 8 m/s ------------------------------------------
fc, sm, m, st = run(pilot_thr=-1.0, tgt_vel=(-8.0, 0.0))
check("E. throttle down, target moving 8 m/s during the attack: reaches it, no descent", st["hit"] and fc.min_vz_alt_hold > -0.3
      and fc.on_pilot_stick == 0.0, f"closest {st['closest']:.2f} m, min vz {fc.min_vz_alt_hold:.2f}")

print("\nALL PASS" if not FAILED else f"\n{len(FAILED)} FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
