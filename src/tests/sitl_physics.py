#!/usr/bin/env python3
"""6-DOF quad physics for ArduCopter SITL's JSON backend (`--model JSON`).

A WASP-class quad (BF_X_REV motor order, as FRAME_TYPE 18 on the real FC) on
four landing legs, launched from a tilted stand and landing on flat ground.
The model runs in its own process, in lockstep with SITL: every servo packet
advances the physics by one SITL frame and answers with the new IMU and pose.

Frames: world NED with the origin on the ground under the stand top, body FRD.
"""
from dataclasses import dataclass, asdict
import math
import multiprocessing as mp
import random
import socket
import struct
import threading
import time

G = 9.80665
RHO = 1.225
NOMINAL_MASS = 0.9

# BF_X_REV, from AP_MotorsMatrix.cpp: (angle clockwise from the nose, yaw factor).
# Index i is motor i+1 = SERVOi+1 (SERVO1..4_FUNCTION 33..36). A yaw factor of
# +1 (CCW prop) yaws the body clockwise (positive NED yaw) as it speeds up.
MOTORS = ((135, +1), (45, -1), (-135, -1), (-45, +1))


@dataclass
class PhysicsConfig:
    mass_kg: float = NOMINAL_MASS
    cg_offset_m: tuple = (0.0, 0.0, 0.01)   # CG from the motor centre, body FRD
    inertia_scale: float = 1.0
    arm_m: float = 0.13                     # motor centre to motor axis (7" class)
    max_thrust_n: float = 17.65             # per motor: hover at 12.5 % for 0.9 kg
    motor_tau_s: float = 0.025
    thrust_expo: float = 0.62               # thrust = (1-e)x + e x^2, x = 0..1
    yaw_coef_m: float = 0.016               # prop drag torque per newton of thrust
    rotor_inertia_kgm2: float = 3e-5        # prop + bell: spin-up reaction torque
    rotor_max_rad_s: float = 2200.0         # prop speed at full thrust
    rotor_drag_kgps: float = 0.27           # in-plane rotor drag at hover thrust
    body_cda_m2: tuple = (0.03, 0.03, 0.05)
    wind_ned: tuple = (0.0, 0.0, 0.0)
    gust_sigma: float = 0.0
    gust_tau: float = 1.5
    stand_pitch_deg: float = 45.0           # positive: the drone sits nose-down
    stand_roll_deg: float = 0.0
    stand_yaw_deg: float = 0.0
    stand_height_m: float = 0.4
    stand_half_m: float = 0.16              # stand top is a 0.32 m square
    leg_k: float = 4000.0                   # per leg spring, N/m (stiff carbon legs)
    leg_c: float = 60.0                     # per leg damper, N s/m
    mu_ground: float = 0.8
    mu_stand: float = 2.0                   # cradle sockets hold the legs on a slope
    imu_noise: tuple = (0.05, 0.002)        # accel m/s^2, gyro rad/s (1 sigma)
    seed: int = 1
    instance: int = 0

    @property
    def port(self):
        return 9002 + 10*self.instance


def _euler_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return (cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy)


def _quat_to_dcm(q):
    w, x, y, z = q
    return ((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
            (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)),
            (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))


def _quat_to_euler(q):
    w, x, y, z = q
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y-z*x))))
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return roll, pitch, yaw


def _mv(R, v):
    return (R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2],
            R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2],
            R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2])


def _mtv(R, v):
    return (R[0][0]*v[0]+R[1][0]*v[1]+R[2][0]*v[2],
            R[0][1]*v[0]+R[1][1]*v[1]+R[2][1]*v[2],
            R[0][2]*v[0]+R[1][2]*v[1]+R[2][2]*v[2])


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


class Quad:
    """The rigid body, its motors and its contacts. Pure physics, no I/O."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        s = cfg.mass_kg / NOMINAL_MASS * cfg.inertia_scale
        self.I = (0.0055*s, 0.0055*s, 0.0095*s)
        cg = cfg.cg_offset_m
        L = cfg.arm_m
        rel = lambda p: (p[0]-cg[0], p[1]-cg[1], p[2]-cg[2])
        self.motor_r = [rel((L*math.cos(math.radians(a)), L*math.sin(math.radians(a)), 0.0))
                        for a, _ in MOTORS]
        self.leg_geom = [(0.85*L*math.cos(math.radians(a)), 0.85*L*math.sin(math.radians(a)), 0.07)
                         for a, _ in MOTORS]
        self.leg_r = [rel(p) for p in self.leg_geom]
        # Anything but a leg touching the ground is a crash: motor bells, prop
        # tips and the top plate.
        self.body_r = ([rel((L*math.cos(math.radians(a)), L*math.sin(math.radians(a)), -0.015))
                        for a, _ in MOTORS]
                       + [rel((1.4*L*math.cos(math.radians(a)), 1.4*L*math.sin(math.radians(a)), -0.02))
                          for a, _ in MOTORS]
                       + [rel((0.0, 0.0, -0.035)), rel((0.09, 0.0, 0.0)), rel((-0.09, 0.0, 0.0))])
        self.rotor_r = rel((0.0, 0.0, 0.0))
        self.hover_n = cfg.mass_kg*G
        self.tilt_to = None
        self._set_stand(cfg.stand_pitch_deg)
        self.reset()

    def _set_stand(self, pitch_deg):
        cfg = self.cfg
        self.stand_pitch = pitch_deg
        sp, sr, sy = (math.radians(pitch_deg), math.radians(cfg.stand_roll_deg),
                      math.radians(cfg.stand_yaw_deg))
        self.stand_q = _euler_to_quat(sr, -sp, sy)
        self.stand_R = _quat_to_dcm(self.stand_q)
        self.stand_up = _mv(self.stand_R, (0.0, 0.0, -1.0))
        self.stand_o = (0.0, 0.0, -cfg.stand_height_m)

    def _seat(self):
        # Geometric centre above the stand top so the leg tips just touch it,
        # loaded by the weight's share of a leg-spring compression.
        cfg = self.cfg
        lift = 0.07 - cfg.mass_kg*G/(4*cfg.leg_k)
        gc = tuple(self.stand_o[i] + lift*self.stand_up[i] for i in range(3))
        cgw = _mv(self.stand_R, cfg.cg_offset_m)
        self.q = self.stand_q
        self.p = tuple(gc[i]+cgw[i] for i in range(3))

    def _tilting(self, dt):
        """Someone tilting the drone in its cradle to a new stand angle: the
        pose is prescribed, the gyro sees the turn, the IMU sees 1 g."""
        target, rate = self.tilt_to
        step = max(-rate*dt, min(rate*dt, target - self.stand_pitch))
        self._set_stand(self.stand_pitch + step)
        self._seat()
        self.v = (0.0, 0.0, 0.0)
        self.w = (0.0, -math.radians(step/dt) if dt else 0.0, 0.0)
        self.spec = _mtv(self.stand_R, (0.0, 0.0, -G))
        self.t += dt
        self.on_stand, self.legs_stand = True, 4
        if abs(target - self.stand_pitch) < 1e-6:
            self.tilt_to, self.w = None, (0.0, 0.0, 0.0)

    def reset(self):
        cfg = self.cfg
        self.t = 0.0
        self.q = self.stand_q
        R = self.stand_R
        # Geometric centre above the stand top so the leg tips just touch it,
        # loaded by the weight's share of a leg-spring compression.
        lift = 0.07 - cfg.mass_kg*G/(4*cfg.leg_k)
        gc = tuple(self.stand_o[i] + lift*self.stand_up[i] for i in range(3))
        cgw = _mv(R, cfg.cg_offset_m)
        self.p = tuple(gc[i]+cgw[i] for i in range(3))
        self.v = (0.0, 0.0, 0.0)
        self.w = (0.0, 0.0, 0.0)
        self.x = [0.0]*4
        self.omega = [0.0]*4
        self.motor_scale = [1.0]*4
        self.torque = (0.0, 0.0, 0.0)
        self.torque_until = -1.0
        self.wind = tuple(cfg.wind_ned)
        self.gust = [0.0, 0.0, 0.0]
        self.gust_sigma, self.gust_tau = cfg.gust_sigma, cfg.gust_tau
        self.anchor = {}
        self.stand_on = True
        self.stand_left_t = None
        self.on_stand = self.on_ground = False
        self.legs_ground = self.legs_stand = 0
        self.last_ground_t = 0.0
        self.left_stand = False
        self.touchdowns = []
        self.hard_landings = []
        self.max_td_vz = self.max_td_vh = 0.0
        self.crashed = None
        self.tip_since = None
        self.spec = (0.0, 0.0, -G)
        self.thrust = [0.0]*4

    # -- forces ---------------------------------------------------------------
    def _contact(self, key, P, V, n, depth, mu, R, r_b, F, T):
        """Spring-damper normal force along n plus stick-slip friction."""
        cfg = self.cfg
        vn = V[0]*n[0]+V[1]*n[1]+V[2]*n[2]
        fn = cfg.leg_k*depth - cfg.leg_c*vn
        if fn <= 0.0:
            self.anchor.pop(key, None)
            return False
        a = self.anchor.get(key)
        if a is None:
            a = P
        d = (P[0]-a[0], P[1]-a[1], P[2]-a[2])
        dn = d[0]*n[0]+d[1]*n[1]+d[2]*n[2]
        d = (d[0]-dn*n[0], d[1]-dn*n[1], d[2]-dn*n[2])
        vt = (V[0]-vn*n[0], V[1]-vn*n[1], V[2]-vn*n[2])
        kt, ct = 4.0*cfg.leg_k, 0.5*cfg.leg_c
        ft = (-kt*d[0]-ct*vt[0], -kt*d[1]-ct*vt[1], -kt*d[2]-ct*vt[2])
        mag = math.sqrt(ft[0]**2+ft[1]**2+ft[2]**2)
        lim = mu*fn
        if mag > lim:
            # Slipping: friction at its Coulomb limit and the anchor dragged
            # along so the spring never stores more than that.
            sc = lim/mag
            ft = (ft[0]*sc, ft[1]*sc, ft[2]*sc)
            back = (ft[0]+ct*vt[0], ft[1]+ct*vt[1], ft[2]+ct*vt[2])
            a = (P[0]+back[0]/kt, P[1]+back[1]/kt, P[2]+back[2]/kt)
        self.anchor[key] = a
        fw = (fn*n[0]+ft[0], fn*n[1]+ft[1], fn*n[2]+ft[2])
        F[0] += fw[0]; F[1] += fw[1]; F[2] += fw[2]
        tb = _cross(r_b, _mtv(R, fw))
        T[0] += tb[0]; T[1] += tb[1]; T[2] += tb[2]
        return True

    def step(self, pwm, dt):
        if self.tilt_to is not None:
            return self._tilting(dt)
        cfg = self.cfg
        m = cfg.mass_kg
        R = _quat_to_dcm(self.q)
        p, v, w = self.p, self.v, self.w
        F = [0.0, 0.0, 0.0]   # world, everything but gravity
        T = [0.0, 0.0, 0.0]   # body, about the CG

        # Motors: SITL sends 1000..2000 us; idle motors (SPIN_ARM 0.10) give no
        # thrust and full thrust is reached at SPIN_MAX 0.95.
        a = dt/(cfg.motor_tau_s+dt)
        e = cfg.thrust_expo
        thrust = 0.0
        for i, (_ang, yf) in enumerate(MOTORS):
            c = (pwm[i]-1000)/1000.0 if pwm[i] >= 1000 else 0.0
            xc = min(1.0, max(0.0, (c-0.10)/0.85))
            self.x[i] += a*(xc-self.x[i])
            x = self.x[i]
            f = cfg.max_thrust_n*((1-e)*x + e*x*x)*self.motor_scale[i]
            self.thrust[i] = f
            thrust += f
            r = self.motor_r[i]
            T[0] += -r[1]*f
            T[1] += r[0]*f
            # Prop drag torque, plus the reaction to speeding the prop up or
            # slowing it down (thrust ~ speed^2).
            om = cfg.rotor_max_rad_s*math.sqrt(max(0.0, f)/cfg.max_thrust_n)
            T[2] += yf*(cfg.yaw_coef_m*f + cfg.rotor_inertia_kgm2*(om-self.omega[i])/dt)
            self.omega[i] = om
        fb = [0.0, 0.0, -thrust]

        # Air: rotor drag in the rotor plane (grows with thrust) acting at the
        # rotor plane, quadratic body drag at the CG, rotational damping.
        if self.gust_sigma > 0:
            k = dt/self.gust_tau
            s = self.gust_sigma*math.sqrt(2*k)
            for i in range(3):
                self.gust[i] += -k*self.gust[i] + s*self.rng.gauss(0, 1)*(0.4 if i == 2 else 1.0)
        wind = (self.wind[0]+self.gust[0], self.wind[1]+self.gust[1], self.wind[2]+self.gust[2])
        va = _mtv(R, (v[0]-wind[0], v[1]-wind[1], v[2]-wind[2]))
        scale = min(3.0, max(0.3, math.sqrt(max(thrust, 1e-3)/self.hover_n)))
        rd = (-cfg.rotor_drag_kgps*scale*va[0], -cfg.rotor_drag_kgps*scale*va[1], 0.0)
        tr = _cross(self.rotor_r, rd)
        T[0] += tr[0]; T[1] += tr[1]; T[2] += tr[2]
        sp = math.sqrt(va[0]**2+va[1]**2+va[2]**2)
        fb[0] += rd[0] - 0.5*RHO*cfg.body_cda_m2[0]*sp*va[0]
        fb[1] += rd[1] - 0.5*RHO*cfg.body_cda_m2[1]*sp*va[1]
        fb[2] += -0.5*RHO*cfg.body_cda_m2[2]*sp*va[2]
        for i in range(3):
            T[i] -= 0.0015*w[i]
        fw = _mv(R, fb)
        F[0] += fw[0]; F[1] += fw[1]; F[2] += fw[2]
        if self.t < self.torque_until:
            for i in range(3):
                T[i] += self.torque[i]

        # Contacts: four legs on the stand top and the ground, the rest of the
        # airframe on the ground only.
        legs_g = legs_s = 0
        up = (0.0, 0.0, -1.0)
        for i, r in enumerate(self.leg_r):
            rw = _mv(R, r)
            P = (p[0]+rw[0], p[1]+rw[1], p[2]+rw[2])
            wr = _mv(R, _cross(w, r))
            V = (v[0]+wr[0], v[1]+wr[1], v[2]+wr[2])
            if P[2] > 0.0 and self._contact(("g", i), P, V, up, P[2], cfg.mu_ground, R, r, F, T):
                legs_g += 1
            if self.stand_on:
                d = (P[0]-self.stand_o[0], P[1]-self.stand_o[1], P[2]-self.stand_o[2])
                local = _mtv(self.stand_R, d)
                h = -local[2]
                if (abs(local[0]) <= cfg.stand_half_m and abs(local[1]) <= cfg.stand_half_m
                        and -0.05 < h <= 0.0
                        and self._contact(("s", i), P, V, self.stand_up, -h, cfg.mu_stand, R, r, F, T)):
                    legs_s += 1
                else:
                    self.anchor.pop(("s", i), None)
        body_hit = False
        for i, r in enumerate(self.body_r):
            rw = _mv(R, r)
            P = (p[0]+rw[0], p[1]+rw[1], p[2]+rw[2])
            if P[2] > 0.0:
                wr = _mv(R, _cross(w, r))
                V = (v[0]+wr[0], v[1]+wr[1], v[2]+wr[2])
                if self._contact(("b", i), P, V, up, P[2], cfg.mu_ground, R, r, F, T):
                    body_hit = True
            else:
                self.anchor.pop(("b", i), None)

        # Integrate (semi-implicit Euler).
        I = self.I
        Iw = (I[0]*w[0], I[1]*w[1], I[2]*w[2])
        gyro = _cross(w, Iw)
        w = (w[0]+dt*(T[0]-gyro[0])/I[0], w[1]+dt*(T[1]-gyro[1])/I[1], w[2]+dt*(T[2]-gyro[2])/I[2])
        v = (v[0]+dt*F[0]/m, v[1]+dt*F[1]/m, v[2]+dt*(F[2]/m+G))
        p = (p[0]+dt*v[0], p[1]+dt*v[1], p[2]+dt*v[2])
        qw, qx, qy, qz = self.q
        hx, hy, hz = 0.5*dt*w[0], 0.5*dt*w[1], 0.5*dt*w[2]
        q = (qw - qx*hx - qy*hy - qz*hz, qx + qw*hx + qy*hz - qz*hy,
             qy + qw*hy - qx*hz + qz*hx, qz + qw*hz + qx*hy - qy*hx)
        n = math.sqrt(q[0]**2+q[1]**2+q[2]**2+q[3]**2)
        self.q = (q[0]/n, q[1]/n, q[2]/n, q[3]/n)
        self.p, self.v, self.w = p, v, w
        self.spec = _mtv(R, (F[0]/m, F[1]/m, F[2]/m))
        self.t += dt
        self._bookkeeping(legs_g, legs_s, body_hit, R)

    def _bookkeeping(self, legs_g, legs_s, body_hit, R):
        t = self.t
        self.on_stand, self.on_ground = legs_s > 0, legs_g > 0 or body_hit
        self.legs_ground, self.legs_stand = legs_g, legs_s
        if self.stand_on and legs_s == 0:
            d = math.dist(self.p, self.stand_o)
            if self.stand_left_t is None:
                self.stand_left_t = t
            if t-self.stand_left_t > 0.3 and d > 0.5:
                self.stand_on = False
                self.left_stand = True
        elif legs_s:
            self.stand_left_t = None
        if self.on_ground:
            if self.left_stand and t-self.last_ground_t > 0.2:
                vz, vh = self.v[2], math.hypot(self.v[0], self.v[1])
                self.touchdowns.append(dict(t=round(t, 3), vz=round(vz, 2), vh=round(vh, 2)))
                self.max_td_vz, self.max_td_vh = max(self.max_td_vz, vz), max(self.max_td_vh, vh)
                if vz > 2.0 or vh > 4.0:
                    self.hard_landings.append(dict(t=round(t, 3), vz=round(vz, 2), vh=round(vh, 2)))
                if vz > 3.0 and not self.crashed:
                    self.crashed = f"touchdown at {vz:.1f} m/s"
                elif vh > 8.0 and not self.crashed:
                    self.crashed = f"ground strike at {vh:.1f} m/s horizontal"
            self.last_ground_t = t
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2][2]))))
            if tilt > 60.0:
                self.tip_since = t if self.tip_since is None else self.tip_since
                if t-self.tip_since > 0.5 and not self.crashed:
                    self.crashed = f"tipped over ({tilt:.0f} deg)"
            else:
                self.tip_since = None
            if body_hit and not self.crashed:
                self.crashed = "airframe hit the ground"

    def imu(self):
        an, gn = self.cfg.imu_noise
        g = self.rng.gauss
        s, w = self.spec, self.w
        return ((s[0]+g(0, an), s[1]+g(0, an), s[2]+g(0, an)),
                (w[0]+g(0, gn), w[1]+g(0, gn), w[2]+g(0, gn)))

    def event(self, name, **kw):
        if name == "wind":
            self.wind = tuple(kw.get("ned", (0.0, 0.0, 0.0)))
        elif name == "gusts":
            self.gust_sigma = float(kw.get("sigma", 0.0))
            self.gust_tau = float(kw.get("tau", self.gust_tau))
            if self.gust_sigma <= 0:
                self.gust = [0.0, 0.0, 0.0]
        elif name == "torque":
            self.torque = tuple(kw["nm"])
            self.torque_until = self.t + float(kw.get("duration", 0.1))
        elif name == "motor_scale":
            self.motor_scale[int(kw["motor"])-1] = float(kw["scale"])
        elif name == "stand_tilt":
            # Re-seat on the stand at a new angle, turning at deg_s.
            self.tilt_to = (float(kw["deg"]), float(kw.get("deg_s", 15.0)))
        elif name == "reset":
            self.reset()
        else:
            raise ValueError(f"unknown physics event {name}")

    def state(self):
        roll, pitch, yaw = _quat_to_euler(self.q)
        R = _quat_to_dcm(self.q)
        wind = (self.wind[0]+self.gust[0], self.wind[1]+self.gust[1], self.wind[2]+self.gust[2])
        return dict(t=self.t, pos=self.p, vel=self.v,
                    euler=(math.degrees(roll), math.degrees(pitch), math.degrees(yaw)),
                    tilt=math.degrees(math.acos(max(-1.0, min(1.0, R[2][2])))),
                    rates=tuple(math.degrees(x) for x in self.w),
                    height=-self.p[2], on_stand=self.on_stand, on_ground=self.on_ground,
                    left_stand=self.left_stand, legs_ground=self.legs_ground,
                    legs_stand=self.legs_stand, crashed=self.crashed,
                    touchdowns=list(self.touchdowns), hard_landings=list(self.hard_landings), max_td_vz=self.max_td_vz,
                    max_td_vh=self.max_td_vh, thrust=tuple(self.thrust),
                    thrust_frac=sum(self.thrust)/(4*self.cfg.max_thrust_n),
                    airspeed=math.dist(self.v, wind), wind=wind)


def _serve(cfg, cmd_rx, state_tx, ready):
    """Child process: lockstep JSON server for one SITL instance."""
    quad = Quad(cfg)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", cfg.port))
    sock.settimeout(0.2)
    ready.set()
    last_frame = None
    reply = b""
    n = 0
    next_pub = 0.0
    busy = 0.0
    steps = 0
    while True:
        if n % 12 == 0 and cmd_rx.poll():
            msg = cmd_rx.recv()
            if msg[0] == "stop":
                break
            quad.event(msg[1], **msg[2])
        n += 1
        try:
            data, addr = sock.recvfrom(256)
        except socket.timeout:
            continue
        if len(data) < 40:
            continue
        magic, rate, frame = struct.unpack_from("<HHI", data)
        if magic == 18458:
            pwm = struct.unpack_from("<16H", data, 8)
        elif magic == 29569:
            pwm = struct.unpack_from("<32H", data, 8)
        else:
            continue
        t0 = time.perf_counter()
        if last_frame is not None and frame < last_frame:
            quad.reset()
        if frame != last_frame or not reply:
            dt = 1.0/max(50, rate or 1200)
            quad.step(pwm, dt)
            steps += 1
            acc, gyr = quad.imu()
            p, v, q = quad.p, quad.v, quad.q
            reply = ('\n{"timestamp":%.6f,"imu":{"gyro":[%.6f, %.6f, %.6f],'
                     '"accel_body":[%.5f, %.5f, %.5f]},"position":[%.5f, %.5f, %.5f],'
                     '"velocity":[%.5f, %.5f, %.5f],"quaternion":[%.7f, %.7f, %.7f, %.7f]}\n'
                     % (quad.t, gyr[0], gyr[1], gyr[2], acc[0], acc[1], acc[2],
                        p[0], p[1], p[2], v[0], v[1], v[2], q[0], q[1], q[2], q[3])).encode()
        last_frame = frame
        sock.sendto(reply, addr)
        busy += time.perf_counter()-t0
        if quad.t >= next_pub:
            next_pub = quad.t + 0.02
            st = quad.state()
            st["pwm"] = tuple(pwm[:4])
            st["step_us"] = 1e6*busy/max(1, steps)
            state_tx.send(st)
    sock.close()


class PhysicsProcess:
    """The physics in its own process; state() and history() at 50 Hz."""

    def __init__(self, config=None):
        self.config = config or PhysicsConfig()
        self._latest = None
        self._hist = []
        self._lock = threading.Lock()
        self.proc = None

    def start(self):
        ctx = mp.get_context("spawn")
        self._cmd_rx, self._cmd_tx = ctx.Pipe(duplex=False)
        self._st_rx, st_tx = ctx.Pipe(duplex=False)
        ready = ctx.Event()
        self.proc = ctx.Process(target=_serve, args=(self.config, self._cmd_rx, st_tx, ready),
                                daemon=True, name=f"sitl-physics-{self.config.instance}")
        self.proc.start()
        if not ready.wait(10):
            raise RuntimeError("physics process did not start")
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        return self

    def _read(self):
        while True:
            try:
                st = self._st_rx.recv()
            except (EOFError, OSError):
                return
            with self._lock:
                self._latest = st
                self._hist.append(st)

    def state(self):
        with self._lock:
            return self._latest

    def history(self, since=None):
        with self._lock:
            return [s for s in self._hist if since is None or s["t"] >= since]

    def clear_history(self):
        with self._lock:
            self._hist = []

    def event(self, name, **kw):
        self._cmd_tx.send(("event", name, kw))

    def stop(self):
        if self.proc is None:
            return
        try:
            self._cmd_tx.send(("stop",))
        except OSError:
            pass
        self.proc.join(2)
        if self.proc.is_alive():
            self.proc.terminate()
        self.proc = None

    def config_dict(self):
        return asdict(self.config)


if __name__ == "__main__":
    # Offline checks: rests on the stand, hover thrust, step cost.
    q = Quad(PhysicsConfig())
    for _ in range(2400):
        q.step([1000]*4, 1/1200)
    s = q.state()
    print("rest on 45 deg stand 2 s: tilt %.1f deg, drift %.4f m, legs %d, spec %s"
          % (s["tilt"], math.dist(s["pos"], Quad(PhysicsConfig()).p), s["legs_stand"],
             tuple(round(x, 2) for x in q.spec)))
    t0 = time.perf_counter()
    for _ in range(12000):
        q.step([1500]*4, 1/1200)
    print("step cost %.1f us" % ((time.perf_counter()-t0)/12000*1e6))
