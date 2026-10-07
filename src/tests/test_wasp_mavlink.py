"""Wasp commands over real MAVLink to FakeArduPilot (no hardware)."""
import contextlib
import io
import math
from pathlib import Path
import socket
import sys
import threading
from time import perf_counter, sleep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fc_control import AcroCommander, InterceptLaw
from fake_fc import FakeArduPilot


def until(predicate, timeout=3):
    deadline = perf_counter() + timeout
    while perf_counter() < deadline:
        if predicate():
            return
        sleep(.02)
    raise AssertionError("condition timed out")


with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
fc = FakeArduPilot(port, armed=False, mode=1)
fc.params.update(ACRO_RP_EXPO=.3, ACRO_Y_EXPO=.2, RC_OVERRIDE_TIME=1,
                 PILOT_SPEED_DN=250, PILOT_ACCEL_Z=500, PSC_JERK_Z=5,
                 ATC_INPUT_TC=.2, ATC_ACCEL_R_MAX=100000, ATC_ACCEL_P_MAX=100000,
                 LOG_BACKEND_TYPE=0, LOG_BITMASK=0, INS_LOG_BAT_MASK=0, FS_GCS_ENABLE=0)
for ch in range(1, 5):
    fc.params[f"RC{ch}_DZ"] = 20
fc.vfr["alt"] = 0
fc.start()
c = AcroCommander(conn=f"udpin:127.0.0.1:{port}", transport="mavlink",
                  axes=("roll", "pitch", "yaw", "throttle"), law=InterceptLaw(),
                  send_hz=50, mission=True, verbose=False, steer_modes={"ACRO", "ALT_HOLD"})
stop = threading.Event()
target = {"on": True}
worker = None
try:
    with contextlib.redirect_stdout(io.StringIO()):
        c.connect()
    m = c.mission
    m.log = lambda line: print(line, flush=True)
    assert not m.set_wasp_enabled(True)
    assert not m.confirm_mount(90)
    assert not m.set_wasp_param("duration", 3)

    def detections():
        while not stop.is_set():
            if target["on"]:
                cmd = c.compute(35, -20, "LOCK", 640, 480, capture_t=perf_counter()-.03)
                c.update(cmd, (35, -20), True, detected=True, conf=.9)
            sleep(.03)

    worker = threading.Thread(target=detections, daemon=True)
    worker.start()
    for tilt in (0, 20, 40, 60, 90):
        throttle = .25 if tilt == 0 else .5 if tilt == 20 else 1
        assert not m.set_wasp_param("throttle", throttle)
        fc.att.update(pitch=-math.radians(tilt), roll=.1, yaw=.7)
        sleep(.5)  # Let synthetic target motion from moving the fixture settle.
        assert not fc.overrides, "nothing moves while disarmed"
        fc.armed = True
        until(lambda: m.phase == "WASP_READY")
        until(lambda: c._cmd.get("wasp"))
        assert not fc.overrides, "arming does not launch"
        assert not m.request_engage()
        until(lambda: fc.overrides.get(3) == 1000+int(throttle*1000))
        assert m.phase == "WASP_FLIGHT", m.phase
        overrides = dict(fc.overrides)
        assert set(overrides) == {1, 2, 3, 4}
        assert overrides[1] == 1500 and overrides[2] == 1500, overrides
        if tilt != 0:  # First pass checks target-loss release during straight launch.
            sleep(.45)
            assert fc.overrides[1] == 1500 and fc.overrides[2] == 1500, fc.overrides
            until(lambda: fc.overrides.get(1, 0) > 1500 and fc.overrides.get(2, 0) > 1500)
            assert perf_counter() - m.wasp.started >= 1, "no AI steering in first second"
        assert m.set_wasp_enabled(False), "cannot change mode while armed"
        if tilt == 0:
            target["on"] = False
            until(lambda: m.phase == "WASP_DONE" and not fc.overrides, timeout=1)
            target["on"] = True
            sleep(.15)
            assert not fc.overrides, "new detection must not relaunch"
        elif tilt == 20:
            m.request_abort()
            until(lambda: not fc.overrides)
        elif tilt == 40:
            fc.mode = 0
            until(lambda: not fc.overrides)
            fc.mode = 1
            sleep(.3)
            assert not fc.overrides, "mode cycling must not relaunch"
        elif tilt == 60:
            fc.attitude = False
            until(lambda: not fc.overrides, timeout=1)
            fc.attitude = True
        else:
            until(lambda: m.phase == "WASP_DONE" and not fc.overrides, timeout=2)
        fc.armed = False
        until(lambda: not c.armed and not m.wasp.used)
        print(f"PASS: platform {tilt} degrees, {throttle:.0%} throttle, launch timing and channel release", flush=True)

    # Exercise initial acquisition through the real sender and RC release path.
    target["on"] = False
    sleep(.35)  # Expire the preceding flight's detection.
    assert not m.set_wasp_param("duration", 5)
    assert not m.set_wasp_param("no_det", 3)
    fc.armed = True
    until(lambda: m.phase == "WASP_READY")
    assert not fc.overrides, "no automatic launch without a target"
    assert not m.request_engage()
    until(lambda: fc.overrides.get(3) == 2000)
    launch_t = m.wasp.started
    until(lambda: perf_counter()-launch_t >= 1.6)
    assert m.wasp.acquired_t is None
    assert all(fc.overrides.get(ch) == 1500 for ch in (1, 2, 4)), fc.overrides
    target["on"] = True
    until(lambda: m.wasp.acquired_t is not None)
    acquired_t = m.wasp.acquired_t
    until(lambda: fc.overrides.get(1, 0) > 1500 and fc.overrides.get(2, 0) > 1500)
    assert perf_counter()-acquired_t < 1.5, "late target must transition without another full straight delay"
    target["on"] = False
    until(lambda: m.phase == "WASP_DONE" and not fc.overrides, timeout=1)
    print("PASS: no-target launch, late acquisition, steering and target-loss release", flush=True)
    fc.armed = False
    until(lambda: not c.armed and not m.wasp.used)

    fc.armed = True
    until(lambda: m.phase == "WASP_READY")
    assert not m.request_engage()
    until(lambda: fc.overrides.get(3) == 2000)
    until(lambda: m.phase == "WASP_DONE" and not fc.overrides, timeout=4)
    assert "no detection time limit" in m.event, m.event
    target["on"] = True
    sleep(.3)
    assert not fc.overrides, "new detection after acquisition timeout must not relaunch"
    assert m.request_engage(), "pilot must disarm before another launch"
    print("PASS: acquisition timeout releases all channels and latches pilot control", flush=True)
    print("All Wasp MAVLink integration checks passed")
finally:
    stop.set()
    if worker is not None:
        worker.join(timeout=2)
    c.close()
    fc.stop.set()
    fc.join(timeout=2)
