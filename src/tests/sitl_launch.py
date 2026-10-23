#!/usr/bin/env python3
"""ArduCopter SITL (JSON physics) with the WASP FC's real parameters.

    sitl = Sitl(0, PhysicsConfig(stand_pitch_deg=45)).start()
    sitl.wait_ready(); sitl.arm()
    app connects to sitl.mavlink_url; sitl.rc.set(6, 1500) flips the pilot's switch

Ports for instance N: app MAVLink tcp 5760+10N (serial0), monitor tcp 5762+10N
(serial1), pilot RC in udp 5501+10N, physics udp 9002+10N.
"""
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import threading
from time import perf_counter, sleep

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sitl_physics import PhysicsConfig, PhysicsProcess  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FC_PARAMS = ROOT/"deploy"/"fc"/"TBS_LUCID_H7_2026-10-22.param"
HOME = "37.9838,23.7275,100,0"

# Hardware the simulator does not have: ports, OSD, LEDs, CAN, sensor ids,
# factory calibrations and the board-level trim (a real board's mounting error
# would tilt a perfectly level simulated IMU), and battery-monitor pins.
DROP_PREFIXES = ("SERIAL", "SR0_", "SR1_", "SR2_", "SR3_", "SR4_", "SR5_", "SR6_", "BRD_",
                 "OSD", "NTF_", "CAN_", "BATT", "GPS", "BARO", "AHRS_TRIM", "STAT_",
                 "FORMAT_VERSION", "SYSID_THISMAV", "RNGFND", "PRX", "FLOW_", "RPM",
                 "EFI_", "ESC_", "VTX_", "MSP_", "FRSKY_", "RSSI_", "TELEM_", "GEN_",
                 "KDE_", "BTN_", "BCN_", "ADSB_", "EAHRS_", "LGR_", "GRIP_", "CHUTE_",
                 "RELAY", "SCR_", "TCAL_", "DEV_", "CUST_", "CAM", "MNT", "SERVO_",
                 "INS_TCAL", "INS_POS", "INS_LOG", "INS_RAW", "LOG_FILE", "LOG_MAX")
DROP_SUFFIXES = ("_DEVID", "_ID", "_CALTEMP", "OFFS_X", "OFFS_Y", "OFFS_Z",
                 "SCAL_X", "SCAL_Y", "SCAL_Z", "_GND_PRESS")
DROP_EXACT = {"INS_GYR_CAL", "COMPASS_DEC", "COMPASS_ORIENT", "COMPASS_EXTERNAL"}
# The only SITL-specific settings. Everything else is the real FC's.
SITL_OVERRIDES = {
    "SIM_RATE_HZ": 1200,        # physics frame rate (> SCHED_LOOP_RATE 400)
    "SIM_GPS1_ENABLE": 1,       # the FC has a GPS on GPS1; the EKF still ignores it
}


def find_binary():
    env = os.environ.get("WASP_SITL_BIN")
    for p in ([env] if env else []) + [str(Path.home()/"ardupilot/build/sitl/bin/arducopter")]:
        if p and Path(p).is_file() and os.access(p, os.X_OK):
            return p
    return None


def _keep(name):
    if name in DROP_EXACT or name.startswith(DROP_PREFIXES):
        return False
    if name.startswith(("INS_", "COMPASS_", "BARO")) and name.endswith(DROP_SUFFIXES):
        return False
    if name.startswith("COMPASS_") and name not in ("COMPASS_ENABLE", "COMPASS_USE",
                                                    "COMPASS_USE2", "COMPASS_USE3"):
        return False
    return True


def fc_params(path=FC_PARAMS):
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "," not in line:
            continue
        name, value = line.split(",", 1)
        if _keep(name):
            out[name] = float(value)
    return out


def write_defaults(path, extra=None):
    params = fc_params()
    params.update(SITL_OVERRIDES)
    params.update(extra or {})
    Path(path).write_text("".join(f"{k} {v:g}\n" for k, v in sorted(params.items())))
    return params


class PilotRadio:
    """The pilot's transmitter: 16 channels to SITL's UDP RC input at 50 Hz.

    Sticks follow the FC's RCMAP (TAER on the WASP FC: throttle is channel 1);
    set_stick takes -1..1 for roll/pitch/yaw and 0..1 for throttle."""

    def __init__(self, port, params=None):
        self.port = port
        p = params or {}
        self.axes = {a: int(p.get(f"RCMAP_{a.upper()}", d)) for a, d in
                     (("roll", 1), ("pitch", 2), ("throttle", 3), ("yaw", 4))}
        self.cal = {ch: (p.get(f"RC{ch}_MIN", 1100), p.get(f"RC{ch}_TRIM", 1500), p.get(f"RC{ch}_MAX", 1900))
                    for ch in range(1, 17)}
        self.ch = {i: 1500 for i in range(1, 17)}
        self.ch.update({self.axes["throttle"]: int(self.cal[self.axes["throttle"]][0]), 5: 1100, 6: 1000})
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def set(self, ch, pwm):
        with self._lock:
            self.ch[int(ch)] = int(pwm)

    def set_stick(self, axis, value):
        ch = self.axes[axis]
        lo, trim, hi = self.cal[ch]
        if axis == "throttle":
            pwm = lo + max(0.0, min(1.0, value))*(hi-lo)
        else:
            v = max(-1.0, min(1.0, value))
            pwm = trim + v*((hi-trim) if v > 0 else (trim-lo))
        self.set(ch, round(pwm))

    def get(self, ch):
        with self._lock:
            return self.ch[int(ch)]

    def _run(self):
        while not self._stop.wait(0.02):
            with self._lock:
                pkt = struct.pack("<16H", *(self.ch[i] for i in range(1, 17)))
            try:
                self._sock.sendto(pkt, ("127.0.0.1", self.port))
            except OSError:
                pass

    def stop(self):
        self._stop.set()
        self._thread.join(1)
        self._sock.close()


class Sitl:
    COPTER_MODES = {0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 9: "LAND"}

    def __init__(self, instance=0, physics_config=None, extra_params=None, speedup=1,
                 workdir=None, binary=None):
        self.instance = instance
        cfg = physics_config or PhysicsConfig()
        cfg.instance = instance
        self.physics = PhysicsProcess(cfg)
        self.extra_params = dict(extra_params or {})
        self.speedup = speedup
        self.binary = binary or find_binary()
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix=f"wasp-sitl{instance}-"))
        self.base = 5760 + 10*instance
        self.mavlink_url = f"tcp:127.0.0.1:{self.base}"
        self.monitor_url = f"tcp:127.0.0.1:{self.base+2}"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.params = write_defaults(self.workdir/"defaults.parm", self.extra_params)
        self.rc = PilotRadio(5501 + 10*instance, self.params)
        self.proc = self.mav = None
        self._hb = None
        self._msgs = {}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.statustext = []

    def start(self):
        if self.binary is None:
            raise RuntimeError("no SITL binary: set WASP_SITL_BIN")
        defaults = self.workdir/"defaults.parm"
        self.physics.start()
        self.rc.start()
        self._log = open(self.workdir/"arducopter.log", "w")
        self.proc = subprocess.Popen(
            [self.binary, "--model", "JSON:127.0.0.1", "--speedup", str(self.speedup),
             "-I", str(self.instance), "--home", HOME, "--defaults", str(defaults), "-w",
             "--serial0", "tcp:0", "--serial1", "tcp:2"],
            cwd=self.workdir, stdout=self._log, stderr=subprocess.STDOUT)
        self._connect_monitor()
        return self

    def _connect_monitor(self, timeout=30):
        from pymavlink import mavutil
        deadline = perf_counter() + timeout
        while True:
            try:
                self.mav = mavutil.mavlink_connection(self.monitor_url, source_system=250,
                                                      autoreconnect=True)
                break
            except OSError:
                if perf_counter() > deadline or self.proc.poll() is not None:
                    raise RuntimeError(f"SITL did not open {self.monitor_url}")
                sleep(0.3)
        self.mav.wait_heartbeat(timeout=timeout)
        for msg_id, hz in ((30, 50), (74, 20), (1, 4), (193, 4), (245, 4), (253, 0)):
            if hz:
                self.set_interval(msg_id, hz)
        threading.Thread(target=self._monitor, daemon=True).start()

    def set_interval(self, msg_id, hz):
        self.mav.mav.command_long_send(1, 1, 511, 0, msg_id, 1e6/hz, 0, 0, 0, 0, 0)

    def _monitor(self):
        while not self._stop.is_set():
            try:
                m = self.mav.recv_match(blocking=True, timeout=0.5)
            except Exception:
                continue
            if m is None:
                continue
            t = m.get_type()
            with self._lock:
                self._msgs[t] = (perf_counter(), m)
                if t == "HEARTBEAT" and m.get_srcSystem() == 1 and m.type != 6:
                    self._hb = (perf_counter(), m)
                elif t == "STATUSTEXT":
                    self.statustext.append((perf_counter(), m.text))

    def msg(self, t):
        with self._lock:
            v = self._msgs.get(t)
        return None if v is None else v[1]

    @property
    def armed(self):
        with self._lock:
            hb = self._hb
        return bool(hb and hb[1].base_mode & 128)

    @property
    def mode(self):
        with self._lock:
            hb = self._hb
        return None if hb is None else self.COPTER_MODES.get(hb[1].custom_mode, str(hb[1].custom_mode))

    def heartbeat_time(self):
        with self._lock:
            return None if self._hb is None else self._hb[0]

    def wait_ready(self, timeout=120):
        """EKF attitude + vertical position good and every prearm check passing."""
        deadline = perf_counter() + timeout
        while perf_counter() < deadline:
            ss, ekf = self.msg("SYS_STATUS"), self.msg("EKF_STATUS_REPORT")
            if (ss and ekf and ss.onboard_control_sensors_health & 0x10000000
                    and ekf.flags & 0x01 and ekf.flags & 0x20):
                return True
            if self.proc.poll() is not None:
                raise RuntimeError("SITL exited during boot")
            sleep(0.2)
        raise RuntimeError("SITL not ready: " + "; ".join(t for _, t in self.statustext[-5:]))

    def arm(self, timeout=5):
        self.rc.set_stick("throttle", 0)
        self.rc.set(5, 1100)
        sleep(0.3)
        self.rc.set(5, 1900)
        deadline = perf_counter() + timeout
        while perf_counter() < deadline:
            if self.armed:
                return True
            sleep(0.05)
        raise RuntimeError("did not arm: " + "; ".join(t for _, t in self.statustext[-5:]))

    def set_mode(self, name, request_heartbeat=True):
        num = {v: k for k, v in self.COPTER_MODES.items()}[name]
        self.mav.mav.command_long_send(1, 1, 176, 0, 1, num, 0, 0, 0, 0, 0)
        if request_heartbeat:
            self.mav.mav.command_long_send(1, 1, 512, 0, 0, 0, 0, 0, 0, 0, 0)

    def stop(self):
        self._stop.set()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.rc.stop()
        self.physics.stop()
        if self.mav is not None:
            try:
                self.mav.close()
            except Exception:
                pass
        if getattr(self, "_log", None):
            self._log.close()


# --- smoke test ---------------------------------------------------------------

def _settle_time(hist, t0, tilt=5.0, vz=0.3, hold=2.0):
    """Seconds after t0 until tilt and vertical speed stay inside the band for `hold` s."""
    start = None
    for s in hist:
        if s["t"] < t0:
            continue
        ok = s["tilt"] <= tilt and abs(s["vel"][2]) <= vz
        if ok and start is None:
            start = s["t"]
        elif not ok:
            start = None
        if start is not None and s["t"]-start >= hold:
            return start - t0
    return None


def _wait_phys(sitl, pred, timeout):
    deadline = perf_counter() + timeout
    while perf_counter() < deadline:
        s = sitl.physics.state()
        if s and pred(s):
            return s
        sleep(0.02)
    return None


def smoke_case(instance, name, cfg, events=(), workroot=None):
    res = dict(case=name)
    sitl = Sitl(instance, cfg, workdir=(Path(workroot)/name) if workroot else None)
    try:
        t_boot = perf_counter()
        sitl.start()
        sitl.wait_ready()
        res["boot_s"] = round(perf_counter()-t_boot, 1)
        for ev, kw in events:
            sitl.physics.event(ev, **kw)
        res["boot_mode"] = sitl.mode
        sitl.arm()
        st0, w0 = sitl.physics.state(), perf_counter()
        res["rest_tilt"] = round(st0["tilt"], 1)
        # (a) WASP-style launch: full stick in ACRO (after ArduCopter's 2 s
        # arming delay) until 0.6 s after liftoff, then STABILIZE with the
        # stick at mid, then ALT_HOLD.
        sleep(2.2)
        t_go = sitl.physics.state()["t"]
        sitl.rc.set_stick("throttle", 1)
        lift = _wait_phys(sitl, lambda s: s["left_stand"] or (not s["on_stand"] and s["height"] > 0.6), 3)
        res["liftoff_s"] = None if lift is None else round(lift["t"]-t_go, 2)
        sleep(0.6)
        sitl.rc.set_stick("throttle", .5)
        sitl.rc.set(6, 1500)
        t_stab = sitl.physics.state()["t"]
        sleep(0.8)
        sitl.rc.set(6, 1900)
        t_alt = sitl.physics.state()["t"]
        sleep(10)
        h = sitl.physics.history(t_go)
        res["liftoff"] = any(s["left_stand"] for s in h) and not any(
            s["on_ground"] for s in h if s["t"] < t_alt)
        res["launch_tilt"] = round(max(s["tilt"] for s in h if s["t"] < t_stab), 1)
        res["v_at_stab"] = round(math.hypot(*[x for x in next(
            s for s in h if s["t"] >= t_stab)["vel"][:2]]), 1)
        res["level_s"] = (lambda t: None if t is None else round(t, 2))(
            next((s["t"]-t_stab for s in h if s["t"] >= t_stab and s["tilt"] <= 5), None))
        alt = [s for s in h if s["t"] >= t_alt]
        h_alt = alt[0]["height"]
        res["h_at_alt"] = round(h_alt, 1)
        res["alt_sink_max"] = round(max(s["vel"][2] for s in alt), 2)
        res["alt_drop"] = round(h_alt - min(s["height"] for s in alt), 2)
        res["alt_settle_s"] = _settle_time(h, t_alt)
        res["alt_tilt_max"] = round(max(s["tilt"] for s in alt if s["t"] > t_alt+1), 1)
        res["alt_vh_end"] = round(math.hypot(*alt[-1]["vel"][:2]), 1)
        tail = [s["height"] for s in alt if s["t"] >= alt[-1]["t"]-4]
        res["hold_band"] = round(max(tail)-min(tail), 2)
        res["alt_trace"] = [(round(x["t"]-t_alt, 1), round(x["height"], 1), round(x["vel"][2], 2),
                              round(math.hypot(*x["vel"][:2]), 1), round(x["tilt"], 1))
                             for x in alt[::50]]
        res["thr_hover"] = round(sum(s["thrust_frac"] for s in alt[-50:])/max(1, len(alt[-50:])), 3)
        # (b) LAND by MAVLink, with an immediate HEARTBEAT request.
        hb_before = sitl.heartbeat_time()
        t_req = perf_counter()
        sitl.set_mode("LAND")
        while perf_counter()-t_req < 3 and sitl.mode != "LAND":
            sleep(0.005)
        res["land_hb_ms"] = round(1000*(perf_counter()-t_req)) if sitl.mode == "LAND" else None
        res["hb_requested"] = bool(hb_before and sitl.heartbeat_time() and
                                   sitl.heartbeat_time()-t_req < 0.5)
        t_land = sitl.physics.state()["t"]
        deadline = perf_counter()+120
        while perf_counter() < deadline and sitl.armed:
            sleep(0.1)
        res["disarmed"] = not sitl.armed
        res["land_s"] = round(sitl.physics.state()["t"]-t_land, 1)
        st = sitl.physics.state()
        ld = [s for s in sitl.physics.history(t_land)]
        res["land_vz"] = round(max(s["vel"][2] for s in ld if s["height"] > 1.5), 2) if any(
            s["height"] > 1.5 for s in ld) else None
        res["td_vz"] = round(st["max_td_vz"], 2)
        res["td_vh"] = round(st["max_td_vh"], 2)
        res["crash"] = st["crashed"] or ""
        res["final_tilt"] = round(st["tilt"], 1)
        res["by_land"] = any("Disarm" in t or "LAND" in t.upper() for _, t in sitl.statustext[-8:])
        res["step_us"] = round(st["step_us"], 1)
        res["rtf"] = round((st["t"]-st0["t"])/(perf_counter()-w0), 3)
        res["hard"] = len(st["hard_landings"])
    except Exception as exc:
        res["error"] = str(exc)
    finally:
        res["statustext"] = [t for _, t in sitl.statustext[-6:]]
        sitl.stop()
    return res


if __name__ == "__main__":
    if find_binary() is None:
        print("no SITL binary (set WASP_SITL_BIN)")
        sys.exit(0)
    work = Path(os.environ.get("WASP_SITL_WORK", tempfile.mkdtemp(prefix="wasp-sitl-smoke-")))
    cases = [
        ("nominal", PhysicsConfig(), ()),
        ("wind6+gusts", PhysicsConfig(wind_ned=(0.0, 6.0, 0.0), gust_sigma=1.5, seed=3), ()),
        # A forward CG beyond ~1.4 cm tips the drone onto its nose on a 45 deg
        # stand before launch, so the forward offset flies from 30 deg.
        ("1.5kg+cg2.5cm-right", PhysicsConfig(mass_kg=1.5, cg_offset_m=(0.0, 0.025, 0.01)), ()),
        ("1.5kg+cg2.5cm-fwd@30", PhysicsConfig(mass_kg=1.5, cg_offset_m=(0.025, 0.0, 0.01),
                                               stand_pitch_deg=30), ()),
    ]
    only = sys.argv[1:]
    results = [None]*len(cases)

    def run(i):
        name, cfg, ev = cases[i]
        results[i] = smoke_case(i, name, cfg, ev, work)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(cases))
               if not only or cases[i][0] in only]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    keys = ["case", "rest_tilt", "boot_mode", "liftoff", "liftoff_s", "launch_tilt", "v_at_stab", "level_s", "h_at_alt",
            "alt_sink_max", "alt_drop", "alt_settle_s", "alt_tilt_max", "alt_vh_end", "hold_band", "thr_hover", "land_hb_ms",
            "hb_requested", "land_vz", "disarmed", "land_s", "td_vz", "td_vh", "final_tilt",
            "hard", "crash", "step_us", "rtf"]
    for r in results:
        if r is None:
            continue
        print(" ".join(f"{k}={r.get(k)}" for k in keys))
        if os.environ.get("WASP_SITL_TRACE"):
            print("   ALT_HOLD (t, h, vz, vh, tilt):", r.get("alt_trace"))
        if r.get("error"):
            print("   ERROR:", r["error"], r.get("statustext"))
    print("work dir:", work)
