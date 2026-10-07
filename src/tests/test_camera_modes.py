#!/usr/bin/env python3
import contextlib
import io
import json
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fc_control import ControlLaw, InterceptLaw
import mission as M

FAILED = []
CX, CY = 2560 / 3280, 1440 / 2464


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def sky_law(edge_guard=0.0):
    law = InterceptLaw(cam_tilt_deg=30, roll_pn_gain=45, max_bank_deg=36, bank_far_deg=30,
                       pn_near_mult=1.4, lock_ramp_s=0.3, max_lean_deg=35, lean_near=0.3,
                       climb_los=True, edge_guard=edge_guard, climb_lead=8, climb_gain=19,
                       cruise_mps=15, climb_trim=6, target_width_m=0.30,
                       angle_max=math.radians(40), max_climb_mps=25.0, max_descent_mps=2.5)
    law.set_stability(slew_per_s=3.0, rate_tau=0.22, deadband_frac=0.05)
    return law


def project(law, fw, fh, az, el, roll, width_m, rng):
    fx = (fw / 2.0) / law._tan_h
    fy = (fh / 2.0) / law._tan_v
    H = fx * math.tan(math.radians(az))
    V = -fy * math.tan(math.radians(el))
    c, s = math.cos(roll), math.sin(roll)
    ex, ey = H * c + V * s, -H * s + V * c
    return ex, ey, fx * width_m / rng, fy * width_m / rng


def fly(law, fw, fh, steps=150):
    out = []
    for i in range(steps):
        k = i / (steps - 1.0)
        rng = 60.0 * (1.0 - k) + 1.0
        az = 9.0 * math.sin(2.1 * k * math.pi)
        el = -8.0 + 14.0 * k
        att = dict(roll=math.radians(12.0 * math.sin(3.0 * k)), pitch=math.radians(-18.0 * k),
                   yaw=math.radians(40.0 * k), yawspeed=0.0, pitchspeed=0.0, rollspeed=0.0)
        ex, ey, bw, bh = project(law, fw, fh, az, el, att["roll"], 0.30, rng)
        out.append(law.compute(ex, ey, fw, fh, dt=1.0 / 30.0, bbox_w=bw, bbox_h=0.6 * bh,
                               att=att, capture_age=0.06))
    return out


KEYS = ("az_deg", "el_deg", "el_cam_deg", "yaw_stick", "roll_stick", "pitch_stick",
        "throttle_stick", "bank_deg", "lean_deg", "climb_mps", "los_az_dps", "los_el_dps",
        "range_m", "fill_frac", "pn_gain_eff", "bank_cap_deg", "terminal")


def worst(a, b):
    d = 0.0
    for x, y in zip(a, b):
        for k in KEYS:
            u, v = x.get(k), y.get(k)
            if u is None or v is None:
                d = max(d, 0.0 if u is v else 1e9)
            else:
                d = max(d, abs(float(u) - float(v)))
    return d


print("=== 1. the intercept law: the same target, the same command, at 30 and 60 fps ===")
a = sky_law()
ref = fly(a, 640, 640)
b = sky_law(); b.set_crop(CX, CY)
check("60 fps view: 50.4 x 29.7 deg", abs(math.degrees(b.h_fov) - 50.42) < 0.02
      and abs(math.degrees(b.v_fov) - 29.70) < 0.02,
      f"{math.degrees(b.h_fov):.2f} x {math.degrees(b.v_fov):.2f}")
d = worst(ref, fly(b, 640 * CX, 640 * CY))
check(f"every output, every frame, equal on the exact crop (worst {d:.2e})", d < 1e-9)
b = sky_law(); b.set_crop(CX, CY)
got = fly(b, 500, 374)
dr = max(abs(x["range_m"] - y["range_m"]) / y["range_m"] for x, y in zip(got, ref))
dang = max(abs(x[k] - y[k]) for x, y in zip(got, ref) for k in ("bank_deg", "lean_deg", "az_deg", "el_deg"))
check(f"...and on the real 500 x 374 frame: angles within {dang:.3f} deg, range within {100 * dr:.2f}%",
      dang < 0.05 and dr < 0.003)
check("the fill schedule saw the same fill (MAX's ease-off, the bank ceiling)",
      max(abs(x["fill_frac"] - y["fill_frac"]) for x, y in zip(got, ref)) < 1e-4)
check("terminal commit on the same frame",
      [x["terminal"] for x in got] == [y["terminal"] for y in ref]
      and any(y["terminal"] for y in ref), str(next((i for i, y in enumerate(ref) if y["terminal"]), None)))

print("=== 2. 30 fps is untouched ===")
c = sky_law(); c.set_crop(CX, CY); c.set_crop(1.0, 1.0)
check("back to 30: the field of view is restored EXACTLY",
      c.h_fov == math.radians(62.2) and c.v_fov == math.radians(48.8)
      and c._tan_h == c.ref_tan_h and c._tan_v == c.ref_tan_v)
check("...and every command is bit-for-bit a fresh law's", worst(ref, fly(c, 640, 640)) == 0.0)

print("=== 3. the frame-edge guard watches the picture the camera really has ===")
for crop, want in (((1.0, 1.0), "free"), ((CX, CY), "eased")):
    g = sky_law(edge_guard=0.55); g.set_crop(*crop)
    fw, fh = 640 * crop[0], 640 * crop[1]
    att = dict(roll=0.0, pitch=0.0, yaw=0.0, yawspeed=0.0, pitchspeed=0.0, rollspeed=0.0)
    ex, ey, bw, bh = project(g, fw, fh, 0.0, 10.0, 0.0, 0.30, 40.0)
    cmd = g.compute(ex, ey, fw, fh, dt=1 / 30.0, bbox_w=bw, bbox_h=bh, att=att, capture_age=0.06)
    gate = cmd["lean_el_gate"]
    check(f"{'60' if crop[0] < 1 else '30'} fps, target at +10 deg: speed {want} (gate {gate:.2f})",
          gate == 1.0 if want == "free" else gate < 0.9)

print("=== 4. the gimbal law (ControlLaw) too ===")
for shape in ("linear", "adaptive"):
    res = []
    for crop in ((1.0, 1.0), (CX, CY)):
        law = ControlLaw(shape=shape, deadband_frac=0.03, kd_yaw=0.1, kd_pitch=0.1)
        law.set_crop(*crop)
        fw, fh = 640 * crop[0], 640 * crop[1]
        r = []
        for i in range(60):
            az, el = 12.0 * math.sin(i / 9.0), -9.0 * math.cos(i / 7.0)
            ex = (fw / 2.0) / law._tan_h * math.tan(math.radians(az))
            ey = -(fh / 2.0) / law._tan_v * math.tan(math.radians(el))
            r.append(law.compute(ex, ey, fw, fh, dt=1 / 30.0))
        res.append(r)
    d = max(abs(x[k] - y[k]) for x, y in zip(*res) for k in ("yaw_stick", "pitch_stick", "az_deg", "el_deg"))
    check(f"{shape}: the same sticks for the same angles at 30 and 60 fps (worst {d:.1e})", d < 1e-9)

print("=== 5. AUTO's aim mark ===")


class Mapper:
    def __init__(self, vfov_deg):
        self.V_FOV = math.radians(vfov_deg)

    def pilot_pwm(self, axis):
        return None


m = M.Mission(Mapper(48.8), log=lambda *_: None)
check("30 fps: the aim mark is -min(line, 20) as before (-20 at line 30)", m.auto_aim_el() == -20.0)
m.AUTO_LINE_DEG = 0.0
m.level["push"] = "SLOW"
check("30 fps, PUSH 20: still exactly -20", m.auto_aim_el() == -20.0)
m6 = M.Mission(Mapper(math.degrees(2 * math.atan(CY * math.tan(math.radians(24.4))))), log=lambda *_: None)
check("60 fps (+/-14.85 deg): the mark comes up to -10.45 deg, inside the picture",
      abs(m6.auto_aim_el() + 10.45) < 0.01, f"{m6.auto_aim_el()}")
m.AUTO_LINE_DEG = 30.0
check("no V_FOV on the mapper (the tests' fakes): as before", M.Mission(object(), log=lambda *_: None)
      .auto_aim_el() == -20.0)

print("=== 6. the saved default ===")
df = os.path.join(tempfile.mkdtemp(), "wasp", "detection.json")
check("CAMERA 60 saved and read back as an int, next to the follower",
      M.save_detection_default("cam", 60, df) == "" and M.save_detection_default("follow", False, df) == ""
      and M.load_detection_defaults(df) == {"cam": 60, "follow": False})
open(df, "w").write('{"cam": "x", "follow": true}')
check("a broken entry is skipped, the rest kept", M.load_detection_defaults(df) == {"follow": True})
check("the SKY GATE is still refused", M.save_detection_default("sky", False, df) != ""
      and "sky" not in json.load(open(df)))

print("=== 7. the tracker: the pipeline, the working frame, the law, the frame counts ===")
with contextlib.redirect_stdout(io.StringIO()):
    import tracker as T


class AP:
    def __init__(self):
        self.law = sky_law()
        self.mission = None


t = T.DroneTracker.__new__(T.DroneTracker)
t.camera_config, t.ref_display, t.ap = T.CameraConfig(sensor_mode=3, width=1280, height=960, fps=30), (640, 640), AP()
t._apply_cam_mode(60)
p = t.camera_pipeline
check("60: sensor mode 4, 1280x720 at 60/1", "sensor-mode=4" in p and "width=1280" in p
      and "height=720" in p and "framerate=60/1" in p, p)
check("60: the working frame is the same crop of 640 x 640 -> 500 x 374", t.display_size == (500, 374)
      and (t.fw, t.fh) == (500, 374), str(t.display_size))
check("60: the law sees 50.4 x 29.7 deg, the picture is 16:9",
      abs(math.degrees(t.ap.law.h_fov) - 50.42) < 0.02 and abs(t.cam_aspect - 16 / 9) < 1e-9)
t._ui_fps = 30.0
f30 = t._frames(20)
t._ui_fps = 45.0
f45 = t._frames(20)
t._ui_fps = 22.0
f22 = t._frames(20)
check(f"60: frame counts keep their time (20 at 30 Hz: {f30}, at 45 Hz: {f45}, at 22 Hz: {f22})",
      f30 == 20 and f45 == 30 and f22 == 20)
t._apply_cam_mode(30)
p = t.camera_pipeline
check("back to 30: sensor mode 3, 1280x960 at 30/1, 640 x 640, the law's full view exactly",
      "sensor-mode=3" in p and "height=960" in p and "framerate=30/1" in p and t.display_size == (640, 640)
      and t.ap.law.h_fov == math.radians(62.2) and t.ap.law.v_fov == math.radians(48.8), p)
t._ui_fps = 45.0
check("30: frame counts are what they always were", t._frames(20) == 20 and t._frames(3) == 3)
t.last_bbox = (0, 0, 30, 30)
small30 = t._target_is_small()
t._apply_cam_mode(60)
check("a 30 px target is 'small' (zoom) in both modes: the same pixels", small30 and t._target_is_small())
t._cam_busy, t._cam_req, t._last_sent, t.cam_mode = False, None, False, 30
check("phone: 60 accepted, 45 / 'x' refused", t.set_camera("60") == "" and t._cam_req == 60
      and t.set_camera("45") != "" and t.set_camera("x") != "")


class Phase:
    phase = "AI"


t._cam_req = None
t.ap.mission = Phase()
check("refused while the AI flies (the picture goes dark a moment)", t.set_camera("60") != "" and t._cam_req is None)

print("=== 8. TEST mode follows the switch ===")
import desk_sim
s = desk_sim.DeskSim(seed=1)
s.touch = lambda: None
check("TEST: cam=60 narrows the pretend camera; the state says so",
      s.command("cam", "60") == "" and abs(s.state()["cam_vfov"] - 29.70) < 0.02
      and s.state()["cam_fps"] == 60 and abs(s.m.auto_aim_el() + 10.45) < 0.01)
check("TEST: back to 30", s.command("cam", "30") == "" and s.state()["cam_vfov"] == 48.8
      and s.m.auto_aim_el() == -20.0)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED:")
    for f in FAILED:
        print("  - " + f)
    sys.exit(1)
print("all passed")
