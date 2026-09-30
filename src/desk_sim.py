"""
desk_sim.py  -  a pretend drone for the phone app's ΔΟΚΙΜΗ (test) mode
======================================================================
The REAL mission.Mission (LOCK, ramps, refusals: the exact code that flies),
driven against a model of ArduPilot's yaw hold in ALT_HOLD and gusty wind.
It never touches the FC: its "sticks" go to the model and nowhere else.

The server (mjpeg_stream) serves it at /sim/state and /sim/cmd, and refuses
/sim/cmd while the REAL drone is armed.  It runs only while someone uses it:
idle for IDLE_S and it stops; the next request starts a fresh one.

The model is the one in tests/sim_lock_wind.py (heading target, the
attitude-error limit that lets a strong gust drag it, a rate loop, finite yaw
authority), cut down to what the phone needs.
"""
import math
import random
import threading
import time

import mission as M

DT = 0.002
ANG_P, ERR_MAX, RATE_MAX = 4.5, 120.0 / 4.5, 202.5
RATE_BW, RATE_I, I_MAX, ACC_MAX = 8.0, 4.0, 60.0, 150.0
BEAUFORT = {0: (0.0, 0.0), 1: (1.0, 0.4), 2: (2.5, 0.8), 3: (4.4, 1.3)}   # mean, sigma m/s
SENS = 40.0                     # deg/s^2 of yaw push from a 5 m/s broadside wind
# A pretend drone for the picture: a fixed point in the world at this compass
# bearing and this elevation below the camera's axis when level, so in HOVER it
# sits just under the aim line.  ENGAGE leans the nose down by the PUSH angle,
# the camera tilts with it, and the target climbs the picture by exactly the
# lean.  Shown only: the mission never gets a detection from it, so nothing
# the AI would do is faked.
TGT_BRG, TGT_EL0 = 140.0, -18.0
LEAN_TAU = 0.25                 # s, the airframe settling onto a lean command


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


class _Yaw:
    """ArduPilot's yaw in ALT_HOLD: the stick is a rate into a heading target."""

    def __init__(self, psi):
        self.psi, self.r, self.tgt, self.i = psi, 0.0, psi, 0.0

    def step(self, stick, dist, dt):
        self.tgt = (self.tgt + stick * dt) % 360.0
        e = wrap180(self.tgt - self.psi)
        if abs(e) > ERR_MAX:                     # a strong gust drags the target
            self.tgt = (self.psi + math.copysign(ERR_MAX, e)) % 360.0
            e = math.copysign(ERR_MAX, e)
        er = max(-RATE_MAX, min(RATE_MAX, ANG_P * e + stick)) - self.r
        self.i = max(-I_MAX, min(I_MAX, self.i + RATE_I * er * dt))
        self.r += (max(-ACC_MAX, min(ACC_MAX, RATE_BW * er + self.i)) + dist) * dt
        self.psi = (self.psi + self.r * dt) % 360.0


class _Mapper:
    def phys_to_pwm(self, axis, v):
        if axis == "yaw":
            return int(round(1500 + 400 * max(-1.0, min(1.0, v / RATE_MAX))))
        return int(round(1500 + 10 * v))

    def pwm_to_phys(self, axis, pwm):
        return (pwm - 1500) / 400.0 * RATE_MAX if axis == "yaw" else (pwm - 1500) / 10.0

    def pilot_pwm(self, axis):
        return 1500

    def ai_pwm(self, cmd):
        return {}


class DeskSim:
    IDLE_S = 60.0

    def __init__(self, bf=2, seed=None):
        self.bf = bf
        self.sky_gate = True                # the SKY GATE switch (no camera here: shown only)
        self.follower = True                # the FOLLOWER switch (shown only, like the sky gate)
        self._lk = threading.Lock()
        self._thread = None
        self._touch = 0.0
        self._reset(seed)

    # ── life cycle ─────────────────────────────────────────────────────────
    def _reset(self, seed=None):
        self.rng = random.Random(seed)
        self.yaw = _Yaw(TGT_BRG)            # facing the pretend drone
        self.lean = 0.0                     # deg of pitch, negative = nose down
        self.mp = _Mapper()
        self.lines, self.seq = [], 0
        self.m = M.Mission(self.mp, log=self._log)
        self.h, self.vz = 6.0, 0.0
        self.wdir = self.rng.uniform(0, 2 * math.pi)
        self.g = self.gd = self.turb = 0.0
        self.kick_until, self.kick_acc = 0.0, 0.0
        self.out = {}
        self.t0 = time.perf_counter()
        self._log("[SIM] ΔΟΚΙΜΗ: a pretend drone, hovering in ALT_HOLD. Nothing reaches the FC.")

    def touch(self):
        """Every request: keeps it running (starts a fresh one if it slept)."""
        with self._lk:
            self._touch = time.perf_counter()
            if self._thread is None or not self._thread.is_alive():
                self._reset()
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

    def _log(self, s):
        self.seq += 1
        self.lines.append((self.seq, time.strftime("%H:%M:%S"), str(s)))
        del self.lines[:-200]

    def console_get(self, since):
        with self._lk:
            return [r for r in self.lines if r[0] > since]

    # ── 50 Hz, like the Jetson's TX thread ─────────────────────────────────
    def _run(self):
        t_next = time.perf_counter()
        while time.perf_counter() - self._touch < self.IDLE_S:
            with self._lk:
                self._cycle(time.perf_counter() - self.t0)
            t_next += 0.02
            time.sleep(max(0.0, t_next - time.perf_counter()))

    def _cycle(self, now):
        ctx = dict(gate_ok=True, gate_why="", att=dict(yaw=math.radians(self.yaw.psi + self.rng.gauss(0, 0.3))),
                   height=self.h, climb=self.vz, ai_cmd=None, ai_t=None, det_t=None, det_conf=None)
        self.out = self.m.cycle(now, ctx)
        stick = self.mp.pwm_to_phys("yaw", self.out["yaw"]) if "yaw" in self.out else 0.0
        vz_cmd = self.mp.pwm_to_phys("throttle", self.out["throttle"]) if "throttle" in self.out else 0.0
        lean_cmd = self.mp.pwm_to_phys("pitch", self.out["pitch"]) if "pitch" in self.out else 0.0
        self.lean += (lean_cmd - self.lean) * 0.02 / LEAN_TAU
        mean, sig = BEAUFORT[self.bf]
        r = self.rng
        for i in range(10):
            dist = 0.0
            if mean > 0:
                a = DT / 2.0
                self.g += -a * self.g + sig * math.sqrt(2 * a) * r.gauss(0, 1)
                self.gd += -a * self.gd + math.radians(15) * math.sqrt(2 * a) * r.gauss(0, 1)
                v = max(0.0, mean + self.g)
                beta = wrap180(math.degrees(self.wdir + self.gd + math.pi) - self.yaw.psi)
                k = SENS * (v / 5.0) ** 2
                b = DT / 0.5
                self.turb += -b * self.turb + math.sqrt(2 * b) * r.gauss(0, 1)
                dist = k * math.sin(math.radians(beta)) + 0.35 * k * self.turb
            if now + i * DT < self.kick_until:
                dist += self.kick_acc
            self.yaw.step(stick, dist, DT)
        self.vz += (vz_cmd - self.vz) * 0.1
        self.h = max(0.5, self.h + self.vz * 0.02)

    # ── what the phone sees / sends ────────────────────────────────────────
    def state(self):
        self.touch()
        with self._lk:
            snap = self.m.snapshot(time.perf_counter() - self.t0)
            snap.update(app="mission", gate_why="", height_m=round(self.h, 2), climb_mps=round(self.vz, 2),
                        heading_deg=round(self.yaw.psi % 360.0), vibe=dict(x=8.0, y=9.0, z=12.0, clip=0),
                        det_age_s=None, det_conf=None, target_fresh=True,
                        target_az_deg=round(wrap180(TGT_BRG - self.yaw.psi), 1),
                        target_el_deg=round(TGT_EL0 - self.lean, 1), pwm=dict(self.out))
            seq = self.seq
        return dict(linked=True, armed=True, fc_mode="ALT_HOLD", detector="TensorRT", fps=15.0,
                    lat_ms=None, tilt_deg=30.0, target="DRONE", mission=snap, console_seq=seq,
                    sky_gate=self.sky_gate, follower=self.follower,
                    sim=dict(bf=self.bf))

    def command(self, key, value):
        """The phone's commands, plus the test-only ones: gust=l|r, wind=0..3,
        reset=1."""
        self.touch()
        with self._lk:
            now = time.perf_counter() - self.t0
            if key == "gust":
                self.kick_until = now + 0.35
                self.kick_acc = (-1 if str(value).lower().startswith("l") else 1) * self.rng.uniform(300, 450)
                self._log(f"[SIM] a sharp gust, {'left' if self.kick_acc < 0 else 'right'} (stronger than the motors)")
                return ""
            if key == "wind":
                try:
                    bf = int(value)
                except ValueError:
                    return "bad wind"
                if bf not in BEAUFORT:
                    return "bad wind"
                self.bf = bf
                self._log(f"[SIM] wind Bf {bf}")
                return ""
            if key == "sky":
                v = str(value).strip().lower()
                if v not in ("on", "off", "1", "0", "true", "false"):
                    return "bad SKY GATE value"
                self.sky_gate = v in ("on", "1", "true")
                self._log(f"[SIM] sky gate {'ON' if self.sky_gate else 'OFF'}")
                return ""
            if key == "follow":
                v = str(value).strip().lower()
                if v not in ("on", "off", "1", "0", "true", "false"):
                    return "bad FOLLOWER value"
                self.follower = v in ("on", "1", "true")
                self._log(f"[SIM] follower {'ON' if self.follower else 'OFF'}")
                return ""
            if key == "reset":
                bf = self.bf
                self._reset()
                self.bf = bf
                return ""
            return M.phone_command(self.m, key, value, now)
