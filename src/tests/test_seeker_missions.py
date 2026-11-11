#!/usr/bin/env python3
"""Seeker test missions against a simulated copter (no SITL, seconds to run).

The fake FC obeys GUIDED targets, BRAKE, RTL and LAND, has the GCS and the
battery failsafe, and the pilot's mode switch.  Every mission runs to its
result; ABORT, pilot takeover, the geofence and blocked ARM are checked.
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import seeker_missions as S
from geofence import Geofence

FAILED = []
HOME = (37.9838, 23.7275)


def check(label, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + label + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(label)


class Clock:
    t = 1000.0


class FakeCopter:
    SPEED_GROUND, SPEED_CLIMB, SPEED_DESCENT = 1, 2, 3

    def __init__(self, clock):
        self.k = clock
        self.linked, self.armed, self.mode_name = True, False, "STABILIZE"
        self.gcs_muted = False
        self.link_t = clock.t - 30
        self.fc_text = []
        self.params = {"FS_GCS_ENABLE": 0.0, "FS_GCS_TIMEOUT": 5.0, "BATT_FS_LOW_ACT": 1.0, "BATT_LOW_VOLT": 10.5,
                       "WPNAV_SPEED_UP": 250.0, "WPNAV_ACCEL": 500.0, "WPNAV_ACCEL_Z": 200.0, "DISARM_DELAY": 10.0,
                       "FENCE_ENABLE": 0.0, "SIM_RATE_HZ": 1200.0}
        self.n = self.e = self.alt = 0.0
        self.vn = self.ve = self.vz = 0.0
        self.target = None
        self.vtarget = None
        self.wind = 0.4                           # m/s east the velocity loop never quite cancels
        self.speed, self.climb, self.descent = 10.0, 2.5, 2.5
        self.switch = 5                           # ALT_HOLD, UP: the position before a flight
        self.battery = 100.0
        self.drain = 0.0                          # % per second while armed
        self.mute_t = None
        self.acks = {}
        self.sent = []
        self.counts = {}
        self.rc_ok = True
        self._arm_ack = None
        self.home_pos = HOME
        self.rc = {"throttle": dict(min=1000, max=2000, reversed=False)}
        self.throttle_pwm = 1000                  # the pilot's stick: down on the ground
        self.yaw = 0.0

    # -- what the missions read -------------------------------------------
    def pos(self):
        return S._offset(HOME, self.n, self.e)

    def telemetry(self):
        return dict(rel_alt=self.alt, vn=self.vn, ve=self.ve, vd=-self.vz, gs=math.hypot(self.vn, self.ve),
                    battery_v=10.0 + 2.6*self.battery/100, battery_pct=int(self.battery), current_a=10.0,
                    tilt_deg=min(25.0, 2.0*math.hypot(self.vn, self.ve)), hb_age=0.2, amsl=100.0 + self.alt,
                    rc_age=0.05 if self.rc_ok else None, position=self.pos(), heading=self.yaw)

    def gps_status(self):
        return dict(fix=3, sats=14, ok=True)

    def gps_fix_ok(self):
        return True

    def position_ok(self):
        return True

    def position(self):
        return self.pos()

    def home(self):
        return HOME

    def pilot_mode_switch(self):
        return (self.switch, {0: "ACRO", 2: "STABILIZE", 5: "ALT_HOLD"}.get(self.switch)) if self.rc_ok else None

    def pilot_pwm(self, axis):
        return self.throttle_pwm if axis == "throttle" and self.rc_ok else None

    def link_stats(self):
        return dict(lost=0, received=sum(self.counts.values()), rejected=0, counts=dict(self.counts))

    def command_ack(self, cmd):
        return self.acks.get(cmd)

    # -- commands -----------------------------------------------------------
    def _log(self, *what):
        if self.gcs_muted:
            return False
        self.sent.append((self.k.t,) + what)
        return True

    def _command(self, cmd, *p):
        if not self._log("cmd", cmd):
            return False
        self.acks[cmd] = (self.k.t + 0.05, 0)
        return True

    def request_mode(self, name, allow_disarmed=False):
        if not (self.armed or allow_disarmed) or not self._log("mode", name):
            return False
        self.mode_name = name
        if name == "BRAKE":
            self.target = None
        if name != "GUIDED":
            self.vtarget = None
        return True

    def request_arm(self, mavlink=False):
        if self.mode_name == "GUIDED" and not mavlink:
            self._arm_ack = (False, self.k.t)
            return True
        self._log("arm")
        self.armed = True
        self.alt = 0.0
        self._arm_ack = (True, self.k.t)
        return True

    def arm_ack(self):
        return self._arm_ack

    def request_disarm(self, force=False):
        self._log("disarm")
        if self.alt < 0.3 or force:
            self.armed = False
        return True

    def guided_takeoff(self, alt):
        if self.mode_name == "GUIDED" and self.armed and self._log("takeoff", alt):
            self.target = (self.n, self.e, alt)
            self.acks[S.MAV_CMD_NAV_TAKEOFF] = (self.k.t, 0)
        return True

    def guided_goto(self, lat, lon, alt, yaw_deg=None):
        if yaw_deg is not None and self.mode_name == "GUIDED":
            self.yaw = yaw_deg % 360.0
        if self.mode_name == "GUIDED" and self._log("goto", lat, lon, alt):
            n = math.radians(lat - HOME[0])*S.EARTH_R
            e = math.radians(lon - HOME[1])*S.EARTH_R*math.cos(math.radians(HOME[0]))
            self.target = (n, e, alt)
            self.vtarget = None
        return True

    def guided_velocity(self, vn, ve, vd, yaw_deg=None):
        if yaw_deg is not None and self.mode_name == "GUIDED":
            self.yaw = yaw_deg % 360.0
        if self.mode_name == "GUIDED" and self._log("vel", vn, ve, vd):
            self.vtarget = (vn, ve, -vd, self.k.t)
            self.target = None
        return True

    def change_speed(self, kind, mps):
        if self._log("speed", kind, mps):
            if kind == 1:
                self.speed = min(20.0, mps)
            elif kind == 2:
                self.climb = mps
        return True

    def set_param(self, name, value):
        if self._log("param", name, value):
            self.params[name] = float(value)
        return True

    def request_params(self, names):
        return True

    # -- physics ------------------------------------------------------------
    def step(self, dt):
        for k, hz in (("HEARTBEAT", 1), ("ATTITUDE", 20), ("GLOBAL_POSITION_INT", 5)):
            self.counts[k] = self.counts.get(k, 0) + hz*dt
        if not self.armed:
            self.vn = self.ve = self.vz = 0.0
            return
        self.battery = max(0.0, self.battery - self.drain*dt)
        if self.gcs_muted:
            self.mute_t = self.mute_t or self.k.t
            if (self.params["FS_GCS_ENABLE"] and self.k.t - self.mute_t > self.params["FS_GCS_TIMEOUT"]
                    and self.mode_name not in ("RTL", "LAND")):
                self.mode_name = "RTL" if self.params["FS_GCS_ENABLE"] == 1 else "LAND"
        else:
            self.mute_t = None
        if self.battery < 20 and self.mode_name not in ("RTL", "LAND"):
            self.mode_name = S.BATT_FS_MODES[int(self.params["BATT_FS_LOW_ACT"])]
        if self.mode_name == "GUIDED" and self.vtarget:
            if self.k.t - self.vtarget[3] > 3.0:          # GUID_TIMEOUT: stop and hold
                self.vtarget, self.target = None, (self.n, self.e, self.alt)
            else:
                ap = lambda cur, want, a: cur + max(-a*dt, min(a*dt, want - cur))
                self.vn = ap(self.vn, self.vtarget[0], 5.0)
                self.ve = ap(self.ve, self.vtarget[1] + self.wind, 5.0)
                self.vz = ap(self.vz, min(self.vtarget[2], 8.0), 2.0)
                self.n += self.vn*dt
                self.e += self.ve*dt
                self.alt = max(0.0, self.alt + self.vz*dt)
                return
        tn, te, ta = self.n, self.e, self.alt
        if self.mode_name == "GUIDED" and self.target:
            tn, te, ta = self.target
        elif self.mode_name == "RTL":
            tn, te = 0.0, 0.0
            ta = self.alt if math.hypot(self.n, self.e) > 1 else 0.0
        elif self.mode_name == "LAND":
            ta = 0.0
        elif self.mode_name not in ("GUIDED", "BRAKE", "ALT_HOLD"):
            ta = self.alt - 1.0                        # the pilot, throttle low (ALT_HOLD holds)
        dn, de = tn - self.n, te - self.e
        d = math.hypot(dn, de)
        v = min(self.speed, d/max(dt, 1e-3), 0.8*d + 0.2)
        self.vn, self.ve = (v*dn/d, v*de/d) if d > 0.05 else (0.0, 0.0)
        dz = ta - self.alt
        self.vz = max(-self.descent, min(self.climb, dz/max(dt, 1e-3), 1.5*dz if dz > 0 else 1e9))
        self.n += self.vn*dt
        self.e += self.ve*dt
        self.alt = max(0.0, self.alt + self.vz*dt)
        if self.alt <= 0.0 and self.mode_name in ("RTL", "LAND") and self.k.t - self._arm_ack[1] > 5:
            self.armed = False


class FakeMapper:
    def __init__(self, fc, clock):
        self.backend, self.k = fc, clock
        self._t0 = clock.t - 40
        self.frame_first_t = clock.t - 20
        self.frame_t = clock.t

    @property
    def armed(self):
        return self.backend.armed


class FakeWasp:
    enabled = False


class FakeMission:
    def __init__(self, fence):
        self.fence = fence
        self.wasp = FakeWasp()
        self._fence_state = (None, None)


def square(size):
    dlat = size/2/111320
    dlon = size/2/(111320*math.cos(math.radians(HOME[0])))
    return [[HOME[0]-dlat, HOME[1]-dlon], [HOME[0]+dlat, HOME[1]-dlon],
            [HOME[0]+dlat, HOME[1]+dlon], [HOME[0]-dlat, HOME[1]+dlon]]


def setup(fence_m=2000, **params):
    k = Clock()
    fc = FakeCopter(k)
    mapper = FakeMapper(fc, k)
    f = Geofence()
    f.load(dict(points=square(fence_m)))
    m = FakeMission(f)
    tmp = tempfile.mkdtemp(prefix="seeker-test-")
    sm = S.SeekerMissions(path=os.path.join(tmp, "seeker.json"), log_dir=tmp)
    return k, fc, mapper, m, sm


def run(k, fc, mapper, m, sm, mid, until_s=4000, hook=None, dt=0.05):
    why = sm.command(f"start:{mid}")
    assert not why, why
    t_end = k.t + until_s
    started = False
    while k.t < t_end:
        k.t += dt
        fc.step(dt)
        mapper.frame_t = k.t
        m._fence_state = m.fence.check(fc.pos())
        sm.cycle(k.t, mapper, m, {})
        started = started or sm.active
        if hook:
            hook(k, fc, sm)
        if started and not sm.active:
            break
    return sm.result


def fails(res):
    return [f"{c['name']}: {c['detail']}" for c in res["checks"] if c["ok"] is False]


# ---- every mission, nominal --------------------------------------------------
for mid in ("M1", "M2", "M3", "M4", "M5", "M6"):
    k, fc, mapper, m, sm = setup(fence_m=400 if mid == "M6" else 2000)

    def phone(k, fc, sm):
        # The phone reports the alert once it is on the screen.
        if sm.alert and sm.alert.get("shown_t") is None and k.t - sm.alert["t"] > 0.6:
            sm.command(f"alertshown:{sm.alert['n']}")
    res = run(k, fc, mapper, m, sm, mid, hook=phone)
    check(f"{mid} {S.BY_ID[mid]['name']}: PASS", res and res["verdict"] == "PASS",
          (res or {}).get("verdict", "no result") + " " + "; ".join(fails(res or {"checks": []})[:3]))
    check(f"{mid} landed / disarmed", not fc.armed)
    check(f"{mid} CSV written", res and res["csv"] and os.path.getsize(res["csv"]) > 200)
    if mid == "M6":
        check("M6 never past the fence", m.fence.check(fc.pos())[0] is True)

# ---- M7: RESTART APP in flight; the FC flies home; the new app picks the run up ---
for angle in (45, 90):
    k, fc, mapper, m, sm = setup()
    box = {"sm": sm}
    sm.restart_hook = lambda: box.__setitem__("down_t", k.t)
    sm.command(f"set:M7:angle:{angle}")
    assert not sm.command("start:M7")
    t_end, started = k.t + 2000, False
    while k.t < t_end:
        k.t += 0.05
        fc.step(0.05)
        mapper.frame_t = k.t
        m._fence_state = m.fence.check(fc.pos())
        cur = box["sm"]
        if "down_t" in box and k.t - box["down_t"] > 20 and "up" not in box:
            box["up"] = (fc.mode_name, len(fc.sent))
            fc.gcs_muted = False                  # the new process: a fresh link
            cur = box["sm"] = S.SeekerMissions(path=cur.path, log_dir=cur.log_dir)
        cur.cycle(k.t, mapper, m, {})
        if cur.run and cur.run.get("wait_restart") and "pressed" not in box:
            box["pressed"] = k.t
            check("M7: RESTART APP only at its step", cur.command("restart") == "")
        started = started or cur.active
        if started and not cur.active and "up" in box:
            break
    res = box["sm"].result or {}
    check(f"M7 ({angle}°): restart in flight -> FC RTL -> app back -> PASS", res.get("verdict") == "PASS",
          res.get("verdict", "no result") + " " + "; ".join(fails(res or {"checks": []})[:3]))
    check(f"M7 ({angle}°): the FC was in RTL when the app came back", box.get("up", ("?",))[0] == "RTL", str(box.get("up")))
    check(f"M7 ({angle}°): landed at home, disarmed", not fc.armed and math.hypot(fc.n, fc.e) < 2)
    check(f"M7 ({angle}°): FS_GCS_ENABLE restored", fc.params["FS_GCS_ENABLE"] == 0.0, str(fc.params["FS_GCS_ENABLE"]))
    check(f"M7 ({angle}°): one CSV for the whole run", res.get("csv") and
          sum(1 for row in __import__("csv").DictReader(open(res["csv"])) if row["kind"] == "result") == 1)
k, fc, mapper, m, sm = setup()
check("RESTART APP refused outside M7's step", sm.command("restart") != "")

# ---- ABORT button in the middle of M3 -----------------------------------------
k, fc, mapper, m, sm = setup()


def abort_at_40(k, fc, sm):
    if sm.active and fc.alt > 40 and not sm.run["aborted"]:
        sm.command("abort")


res = run(k, fc, mapper, m, sm, "M3", hook=abort_at_40)
check("ABORT in M3: FAIL verdict, RTL home", res["verdict"] == "FAIL" and not fc.armed
      and math.hypot(fc.n, fc.e) < 2, res["verdict"])
check("ABORT reaction logged", any("ABORT reaction" in c["name"] for c in res["checks"]))

# ---- pilot takeover: nothing sent afterwards ----------------------------------
k, fc, mapper, m, sm = setup()
mark = {}


def takeover(k, fc, sm):
    if sm.active and fc.alt > 20 and "t" not in mark:
        fc.switch = 2
        fc.mode_name = "STABILIZE"
        mark["t"], mark["n"] = k.t, len(fc.sent)
    if "t" in mark and k.t - mark["t"] > 5 and not mark.get("abort"):
        mark["abort"] = True
        sm.command("abort")                       # the operator presses ABORT after the takeover
    if "t" in mark and k.t - mark["t"] > 30 and fc.armed:
        fc.request_disarm(force=True)


res = run(k, fc, mapper, m, sm, "M3", hook=takeover)
check("Pilot takeover: INCOMPLETE", res["verdict"] == "INCOMPLETE", res["verdict"])
after = [s for s in fc.sent[mark["n"]:] if s[1] != "disarm"]
check("Pilot takeover: ABORT afterwards sends nothing either", mark.get("abort") and not after, str(after[:3]))
check("Pilot takeover: no command after it", not after, str(after[:3]))

# ---- blocked ARM -----------------------------------------------------------------
k, fc, mapper, m, sm = setup()
fc.throttle_pwm = 1500
res = run(k, fc, mapper, m, sm, "M3")
check("Throttle stick not down blocks ARM (the FC would refuse it)", "Throttle stick DOWN" in res["why"]
      and not any(s[1] == "arm" for s in fc.sent), res["why"])

k, fc, mapper, m, sm = setup()
fc.switch = 2
sm._cue_started = True
m.camera_mount_deg = 30.0
mapper.law = type("L", (), {"cam_tilt": math.radians(30)})()
res = run(k, fc, mapper, m, sm, "SEEK")
check("SEEK refuses ARM with the mode switch not UP", "Mode switch UP" in res["why"] and not any(s[1] == "arm" for s in fc.sent),
      res["why"])

k, fc, mapper, m, sm = setup()
fc.battery = 30
res = run(k, fc, mapper, m, sm, "M3")
check("Low battery blocks ARM", res["verdict"] == "FAIL" and not any(s[1] == "arm" for s in fc.sent)
      and "Battery" in res["why"], res["why"])

k, fc, mapper, m, sm = setup(fence_m=300)
res = run(k, fc, mapper, m, sm, "M4")
check("M4 ignores the geofence completely (a small one does not block it)", res["verdict"] == "PASS"
      and not any("eofence" in c["name"] or c["name"].endswith("Route") for c in res["checks"]), res["verdict"])

k, fc, mapper, m, sm = setup()
sm.command("set:limits:alt_limit:50")
res = run(k, fc, mapper, m, sm, "M3")
check("Mission above the altitude limit blocks ARM", "Altitude limit" in res["why"], res["why"])

k, fc, mapper, m, sm = setup()
m.fence = Geofence()
res = run(k, fc, mapper, m, sm, "M6")
check("M6 without a geofence: ARM blocked", "Geofence" in res["why"] and not any(s[1] == "arm" for s in fc.sent),
      res["why"])

for mid in ("M1", "M3"):
    k, fc, mapper, m, sm = setup()
    m.fence = Geofence()
    res = run(k, fc, mapper, m, sm, mid)
    check(f"{mid} without a geofence: runs (not needed)", res["verdict"] == "PASS", res["verdict"] + " " + res["why"])

# ---- SEEK: fly the ASTERIX cue, ask, hand over to the AI -------------------------
from asterix_cue import encode_cat062, encode_eketa_cat062, rfd_line


def seek_run(answers=("yes",), lose_after=None, second_at=None, see_at=150.0, rfd=False, abort_after=None,
             until=900, stop_at_ai=True, camera=False, answer_after=0.0):
    """answers: what the operator taps, in order, for each question."""
    k, fc, mapper, m, sm = setup()
    sm._cue_started = True                       # no UDP socket / radio in the test: data goes in directly
    st = dict(asked=[], answers=list(answers))
    t0 = k.t

    def track(n):
        if n == 1:
            return S._offset(HOME, 600.0 + 3.0*(k.t - t0), 100.0)
        return S._offset(HOME, 300.0, -400.0)

    def feed(k, fc, sm):
        if k.t - st.get("tx", -1e9) >= 1.0:
            st["tx"] = k.t
            tr = []
            if lose_after is None or k.t - t0 < lose_after:
                lat, lon = track(1)
                if rfd:
                    sm.cue.ingest_rfd(rfd_line(lat, lon, 180.0, "MSL"), now=k.t)
                else:
                    tr.append(dict(track_no=12, lat=lat, lon=lon, h_msl_m=180.0, vx=0, vy=3))
            if second_at is not None and k.t - t0 >= second_at:
                lat, lon = track(2)
                tr.append(dict(track_no=99, lat=lat, lon=lon, h_msl_m=170.0))
            if tr:
                for _t in tr:
                    _t.setdefault("vx", 0.0); _t.setdefault("vy", 0.0)
                    _t.setdefault("azimuth_deg", 0.0); _t.setdefault("elevation_deg", 0.0)
                    _t.setdefault("slant_range_m", 0.0); _t.setdefault("hrel_m", 0.0)
                sm.cue.ingest(encode_eketa_cat062(tr), now=k.t)
        r = sm.run
        live = r and r.get("cue")
        if live:
            d = S._dist(fc.pos(), (live["lat"], live["lon"]))
            st["closest"] = min(st.get("closest", 1e9), d)
            sm.ctx = dict(det_t=k.t, det_conf=0.8) if d < see_at else {}
            if camera and d < see_at:
                # The camera (30 degrees up) as the AI's law reports the target.
                brg = S._bearing(fc.pos(), (live["lat"], live["lon"]))
                az = (brg - fc.yaw + 180) % 360 - 180
                el = math.degrees(math.atan2(live["alt_rel"] - fc.alt, max(1.0, d))) - 30.0
                sm.ctx["ai_cmd"] = dict(az_deg=az, el_cam_deg=el)
                st["cam_now"] = (az, el, d)
        o = r and r.get("offer")
        if o and o["n"] not in [x[0] for x in st["asked"]]:
            st.setdefault("offer_t", {}).setdefault(o["n"], k.t)
            if k.t - st["offer_t"][o["n"]] < answer_after:
                o = None
        if o and o["n"] not in [x[0] for x in st["asked"]]:
            st["cam_at_answer"] = st.get("cam_now")
            ans = st["answers"].pop(0) if st["answers"] else "yes"
            st["asked"].append((o["n"], o["kind"], o.get("key"), ans))
            sm.command(f"offer:{ans}:{o['n']}")
        if abort_after is not None and sm.active and k.t - t0 > abort_after and not st.get("aborted"):
            st["aborted"] = True
            sm.command("abort")
    feed(k, fc, sm)
    m.min_conf, m.camera_mount_deg = 0.2, 30.0
    m.push_deg = lambda: 30.0
    mapper.law = type("L", (), {"cam_tilt": math.radians(30)})()
    sm.cycle(k.t, mapper, m, {})
    k.t += 0.6
    sm.cycle(k.t, mapper, m, {})
    why = sm.command("start:SEEK")
    assert not why, why
    started = False
    while k.t < t0 + until:
        k.t += 0.05
        fc.step(0.05)
        mapper.frame_t = k.t
        m._fence_state = m.fence.check(fc.pos())
        feed(k, fc, sm)
        sm.cycle(k.t, mapper, m, sm.ctx)
        started = started or sm.active
        if started and not sm.active:
            break
        if stop_at_ai and sm.run and sm.run.get("state") == "AI":
            break
    return fc, sm, st


fc, sm, st = seek_run()
h = sm.take_handover()
check("SEEK: the camera sees it -> ALT_HOLD and the AI, no question", h is not None and fc.mode_name == "ALT_HOLD"
      and fc.armed and st["asked"] == [], f"mode {fc.mode_name}, asked {st['asked']}")
check("SEEK: no result screen over the video", sm.result is None)
check("SEEK: stays on after the handover (AI state), not flying the drone", sm.active and not sm.blocking)
check("SEEK: flew at the cue's height minus 'below'", abs(fc.alt - (80 - 15)) < 3, f"{fc.alt:.1f} m")

fc, sm, st = seek_run(answers=("yes",), lose_after=25, second_at=60)
kinds = [(x[1], x[2], x[3]) for x in st["asked"]]
check("SEEK: cue lost -> waits; a new target asks once; YES flies to it and the AI takes it",
      sm.take_handover() is not None and len(kinds) == 1 and kinds[0][0] in ("follow", "switch")
      and kinds[0][1] == "TRK99" and sm.cue.selected == "TRK99", str(kinds))

fc, sm, st = seek_run(answers=("yes", "yes"), second_at=8, see_at=-1, abort_after=120)
kinds = [(x[1], x[2]) for x in st["asked"]]
check("SEEK: a new target while following asks to switch; YES switches",
      kinds[:1] == [("switch", "TRK99")] and sm.cue.selected == "TRK99", f"{kinds} selected {sm.cue.selected}")
check("SEEK: ABORT flies home (RTL)", not fc.armed and math.hypot(fc.n, fc.e) < 2 and sm.result
      and sm.result["verdict"] == "FAIL", (sm.result or {}).get("why"))

fc, sm, st = seek_run(lose_after=20, see_at=-1, until=200)
check("SEEK: cue lost -> holds and waits (no RTL by itself)", sm.active and fc.armed and fc.mode_name == "GUIDED"
      and abs(fc.vn) + abs(fc.ve) < 0.2, f"active {sm.active} mode {fc.mode_name}")

fc, sm, st = seek_run(rfd=True)
check("SEEK: the cue over the RFD radio ($RFD lines) works the same", sm.take_handover() is not None
      and fc.mode_name == "ALT_HOLD", f"mode {fc.mode_name}, asked {st['asked']}")

fc, sm, st = seek_run(stop_at_ai=False, abort_after=200, see_at=200.0, until=600)
check("SEEK: ABORT while the AI flies -> RTL home", st.get("aborted") and not fc.armed and math.hypot(fc.n, fc.e) < 2,
      f"armed {fc.armed}, {math.hypot(fc.n, fc.e):.0f} m from home, mode {fc.mode_name}")

fc, sm, st = seek_run(camera=True)
az, el, d = st.get("cam_now") or (99, 99, 0)
check("SEEK: the nose is on the target when the AI takes it", abs(az) < 10.0 and sm.take_handover() is not None,
      f"az {az:+.1f} deg at {d:.0f} m")

# ---- settings are validated and saved ---------------------------------------------
k, fc, mapper, m, sm = setup()
check("Setting out of range refused", sm.command("set:M3:alt1:900") != "")
check("Setting saved", sm.command("set:M3:alt1:60") == "" and S.SeekerMissions(path=sm.path).params["M3"]["alt1"] == 60)
check("Unknown mission refused", sm.command("start:M99") != "")

print("\n" + ("ALL OK" if not FAILED else f"{len(FAILED)} FAILED: " + ", ".join(FAILED)))
sys.exit(1 if FAILED else 0)
