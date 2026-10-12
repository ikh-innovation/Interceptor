import math
import random
import threading
import time

import mission as M

DT = 0.002
ANG_P, ERR_MAX, RATE_MAX = 4.5, 120.0 / 4.5, 202.5
RATE_BW, RATE_I, I_MAX, ACC_MAX = 8.0, 4.0, 60.0, 150.0
BEAUFORT = {0: (0.0, 0.0), 1: (1.0, 0.4), 2: (2.5, 0.8), 3: (4.4, 1.3)}
SENS = 40.0
TGT_BRG, TGT_RANGE, TGT_ELEV0, TGT_SPAN = 140.0, 60.0, 20.0, 0.8
TGT_BRG0, TGT_V = 178.0, 3.5
START_HDG, START_H = 110.0, 0.5
TILT = 30.0
HFOV, VFOV = 62.2, 48.8
CAM_CROP = {21: (1.0, 1.0), 30: (1.0, 1.0), 60: (2560 / 3280, 1440 / 2464)}
CAM_FRAME = {21: (3280, 2464), 30: (1280, 960), 60: (1280, 720)}
HIT_R, RESTART_S = 2.0, 3.5
G, DRAG = 9.80665, 0.0159
FOCAL_PX, FRAME_W = 640.0 / math.tan(math.radians(HFOV / 2)), 1280.0
AI_LEAN = {"SLOW": 12.0, "MEDIUM": 20.0, "FAST": 22.0, "MAX": 35.0}
MAX_NEAR, NEAR_FAR, NEAR_NEAR = 0.3, 0.007, 0.03
LEAN_TAU = 0.25


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


class _Yaw:

    def __init__(self, psi):
        self.psi, self.r, self.tgt, self.i = psi, 0.0, psi, 0.0

    def step(self, stick, dist, dt):
        self.tgt = (self.tgt + stick * dt) % 360.0
        e = wrap180(self.tgt - self.psi)
        if abs(e) > ERR_MAX:
            self.tgt = (self.psi + math.copysign(ERR_MAX, e)) % 360.0
            e = math.copysign(ERR_MAX, e)
        er = max(-RATE_MAX, min(RATE_MAX, ANG_P * e + stick)) - self.r
        self.i = max(-I_MAX, min(I_MAX, self.i + RATE_I * er * dt))
        self.r += (max(-ACC_MAX, min(ACC_MAX, RATE_BW * er + self.i)) + dist) * dt
        self.psi = (self.psi + self.r * dt) % 360.0


class _Mapper:
    V_FOV = math.radians(VFOV)

    def phys_to_pwm(self, axis, v):
        if axis == "yaw":
            return int(round(1500 + 400 * max(-1.0, min(1.0, v / RATE_MAX))))
        return int(round(1500 + 10 * v))

    def pwm_to_phys(self, axis, pwm):
        return (pwm - 1500) / 400.0 * RATE_MAX if axis == "yaw" else (pwm - 1500) / 10.0

    def pilot_pwm(self, axis):
        return 1500

    def ai_pwm(self, cmd):
        return dict(cmd["pwm"])


class DeskSim:
    IDLE_S = 60.0

    def __init__(self, bf=2, seed=None):
        self.bf = bf
        self.sky_gate = True
        self.follower = True
        self.cam = 30
        self.video_stabilization = False
        self._lk = threading.Lock()
        self._thread = None
        self._touch = 0.0
        self._reset(seed)

    def _reset(self, seed=None):
        self.rng = random.Random(seed)
        self.yaw = _Yaw(START_HDG)
        self.lean = 0.0
        self.mp = _Mapper()
        self._set_cam(self.cam)
        self.lines, self.seq = [], 0
        self.m = M.Mission(self.mp, log=self._log)
        self.m.auto_enabled = True
        M.load_auto_params(self.m)
        self.h, self.vz = START_H, 0.0
        self.x = self.y = self.v = self.odo = 0.0
        b, b0 = math.radians(TGT_BRG), math.radians(TGT_BRG0)
        self.tx, self.ty = TGT_RANGE * math.sin(b0), TGT_RANGE * math.cos(b0)
        self.tx1, self.ty1 = TGT_RANGE * math.sin(b), TGT_RANGE * math.cos(b)
        self.talt = START_H + TGT_RANGE * math.tan(math.radians(TGT_ELEV0))
        self.hit_t = self.end_t = None
        self.attempt = False
        self.geo = self._geo(0.0)
        self.rate = dict(brg=0.0, elw=0.0, rng=0.0)
        self.wdir = self.rng.uniform(0, 2 * math.pi)
        self.g = self.gd = self.turb = 0.0
        self.kick_until, self.kick_acc = 0.0, 0.0
        self.out = {}
        self.det = None
        self.t0 = time.perf_counter()
        self._log("[SIM] ΔΟΚΙΜΗ: a pretend drone, hovering in ALT_HOLD. Nothing reaches the FC.")

    def _set_cam(self, cam):
        cx, cy = CAM_CROP[cam]
        self.cam = cam
        self.hfov = 2.0 * math.degrees(math.atan(cx * math.tan(math.radians(HFOV / 2.0))))
        self.vfov = 2.0 * math.degrees(math.atan(cy * math.tan(math.radians(VFOV / 2.0))))
        self.mp.V_FOV = math.radians(self.vfov)

    def touch(self):
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

    def _geo(self, now):
        dx, dy = self.tx - self.x, self.ty - self.y
        dz = self.talt + 0.4 * math.sin(1.3 * now) - self.h
        dh = math.hypot(dx, dy)
        brg = math.degrees(math.atan2(dx, dy)) % 360.0
        elw = math.degrees(math.atan2(dz, dh))
        return dict(brg=brg, elw=elw, rng=math.sqrt(dh * dh + dz * dz), dz=dz,
                    az=wrap180(brg - self.yaw.psi), el=elw - TILT - self.lean)

    def _run(self):
        t_next = time.perf_counter()
        while time.perf_counter() - self._touch < self.IDLE_S:
            with self._lk:
                self._cycle(time.perf_counter() - self.t0)
            t_next += 0.02
            time.sleep(max(0.0, t_next - time.perf_counter()))

    def _detect(self, now):
        g = self.geo
        if self.hit_t is not None or abs(g["az"]) >= self.hfov / 2 or abs(g["el"]) >= self.vfov / 2:
            return
        bw = FOCAL_PX * TGT_SPAN / max(0.5, g["rng"])
        if self.rng.random() >= 0.93 * max(0.0, min(1.0, (bw - 4.0) / 10.0)) * 0.5:
            return
        lv = self.m.ai_level_now()
        lean = AI_LEAN.get(lv, 20.0)
        if lv == "MAX":
            near = max(0.0, min(1.0, (bw / FRAME_W - NEAR_FAR) / (NEAR_NEAR - NEAR_FAR)))
            lean *= 1.0 - near * (1.0 - MAX_NEAR)
        tgo = g["rng"] / max(self.v, 3.0)
        yaw = max(-40.0, min(40.0, 3.0 * g["az"]))
        climb = max(-2.5, min(6.0, g["dz"] / max(tgo, 0.5)))
        mp = self.mp
        cmd = dict(az_deg=g["az"], el_cam_deg=g["el"], el_deg=g["elw"],
                   los_az_dps=self.rate["brg"], range_m=g["rng"], closing_mps=-self.rate["rng"],
                   pwm=dict(yaw=mp.phys_to_pwm("yaw", yaw), throttle=mp.phys_to_pwm("throttle", climb),
                            pitch=mp.phys_to_pwm("pitch", -lean), roll=1500))
        self.det = (cmd, now)

    def _cycle(self, now):
        self._detect(now)
        cmd, t_det = self.det if self.det else (None, None)
        ctx = dict(gate_ok=True, gate_why="", att=dict(yaw=math.radians(self.yaw.psi + self.rng.gauss(0, 0.3))),
                   height=self.h, climb=self.vz, ai_cmd=cmd, ai_t=t_det, det_t=t_det,
                   det_conf=None if cmd is None else 0.8)
        self.out = self.m.cycle(now, ctx)
        stick = self.mp.pwm_to_phys("yaw", self.out["yaw"]) if "yaw" in self.out else 0.0
        vz_cmd = self.mp.pwm_to_phys("throttle", self.out["throttle"]) if "throttle" in self.out else 0.0
        lean_cmd = self.mp.pwm_to_phys("pitch", self.out["pitch"]) if "pitch" in self.out else 0.0
        self.lean += (lean_cmd - self.lean) * 0.02 / LEAN_TAU
        g = self.geo
        ph = self.m.phase
        if ph in ("PUSH", "AI"):
            self.attempt = True
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
        a = G * math.tan(math.radians(max(0.0, -self.lean))) - DRAG * self.v * self.v
        self.v = max(0.0, self.v + a * 0.02)
        psi = math.radians(self.yaw.psi)
        self.x += self.v * math.sin(psi) * 0.02
        self.y += self.v * math.cos(psi) * 0.02
        self.odo += self.v * 0.02
        if self.hit_t is None:
            ex, ey = self.tx1 - self.tx, self.ty1 - self.ty
            d = math.hypot(ex, ey)
            if d > 0.05:
                s = min(d, TGT_V * 0.02) / d
                self.tx += ex * s
                self.ty += ey * s
        g2 = self._geo(now)
        for k in self.rate:
            d = (wrap180(g2[k] - g[k]) if k == "brg" else g2[k] - g[k]) / 0.02
            self.rate[k] += (d - self.rate[k]) * 0.2
        self.geo = g2
        if self.hit_t is None and g2["rng"] < HIT_R:
            self.hit_t = now
            self._log("[SIM] HIT: the pretend enemy drone is down")
        if self.attempt and self.end_t is None and self.hit_t is None and ph not in ("PUSH", "AI", "STOP"):
            self.end_t = now
        t_end = self.hit_t if self.hit_t is not None else self.end_t
        if t_end is not None and now - t_end > RESTART_S:
            self._restart()

    def _restart(self):
        bf, lv, ae, ap = self.bf, dict(self.m.level), self.m.auto_enabled, self.m.auto_params()
        self._reset()
        self.bf = bf
        self.m.level.update(lv)
        self.m.auto_enabled = ae
        for k, v in ap.items():
            self.m.set_auto_param(k, v)
        self._log("[SIM] again: facing 110, the enemy flying in from the right")

    def state(self):
        self.touch()
        with self._lk:
            snow = time.perf_counter() - self.t0
            snap = self.m.snapshot(snow)
            snap.update(app="mission", gate_why="", height_m=round(self.h, 2), climb_mps=round(self.vz, 2),
                        heading_deg=round(self.yaw.psi % 360.0), vibe=dict(x=8.0, y=9.0, z=12.0, clip=0),
                        det_age_s=None if self.det is None else round(snow - self.det[1], 2),
                        det_conf=None if self.det is None else 0.8,
                        target_fresh=(self.hit_t is None and self.det is not None
                                      and snow - self.det[1] < 0.5),
                        target_az_deg=round(self.geo["az"], 1),
                        target_el_deg=round(self.geo["el"], 1), pwm=dict(self.out))
            now = time.perf_counter() - self.t0
            g, r = self.geo, self.rate
            scene = dict(brg=round(g["brg"], 2), elw=round(g["elw"], 2), rng=round(g["rng"], 2),
                         brg_r=round(r["brg"], 2), elw_r=round(r["elw"], 2), rng_r=round(r["rng"], 2),
                         span=TGT_SPAN, lean=round(self.lean, 2), spd=round(self.v, 2), odo=round(self.odo, 2),
                         x=round(self.x, 2), y=round(self.y, 2),
                         hit=None if self.hit_t is None else round(now - self.hit_t, 2),
                         gust=(0 if now >= self.kick_until else (-1 if self.kick_acc < 0 else 1)))
            seq = self.seq
        return dict(linked=True, armed=True, fc_mode="ALT_HOLD", detector="TensorRT", fps=15.0,
                    lat_ms=None, tilt_deg=30.0, target="DRONE", mission=snap, console_seq=seq,
                    sky_gate=self.sky_gate, follower=self.follower,
                    cam_fps=self.cam, cam_default=30, cam_busy=False,
                    video_stabilization=self.video_stabilization,
                    cam_hfov=round(self.hfov, 2), cam_vfov=round(self.vfov, 2),
                    cam_aspect=round(16.0 / 9.0 if self.cam == 60 else 4.0 / 3.0, 4),
                    cam_frame=list(CAM_FRAME[self.cam]), cam_capture="SIM", cam_error="",
                    sim=dict(bf=self.bf, scene=scene))

    def command(self, key, value):
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
            if key == "stabilization":
                v = str(value).strip().lower()
                if v not in ("on", "off", "1", "0", "true", "false"):
                    return "bad VIDEO STABILIZATION value"
                self.video_stabilization = v in ("on", "1", "true")
                return ""
            if key == "cam":
                try:
                    cam = int(str(value).strip())
                except ValueError:
                    return "bad CAMERA value"
                if cam not in CAM_CROP:
                    return "CAMERA: 21, 30 or 60"
                if self.m.phase in ("PUSH", "AI", "STOP") + M.AUTO_PHASES:
                    return f"not during {self.m.phase}: the picture is dark a moment while it switches"
                if cam != self.cam:
                    self._set_cam(cam)
                    self._log(f"[SIM] camera {cam} fps: {self.hfov:.1f} x {self.vfov:.1f} deg")
                return ""
            if key == "def":
                return "TEST: settings are not saved as defaults"
            if key == "reset":
                bf = self.bf
                self._reset()
                self.bf = bf
                return ""
            return M.phone_command(self.m, key, value, now)
